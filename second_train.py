import os
import glob
import time
import random
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

import rawpy
from PIL import Image, ImageFilter

# =========================
# USER CONFIG (edit here)
# =========================
RAW_GLOB = "./NEF/*.NEF"  # <-- your dataset
PATCH = 128                   # NxN patch size
STRIDE = None                 # Default = PATCH
BATCH = 16                    # RAW demosaic is heavy; 16 is a good start
EPOCHS = 20
LR = 1e-3
WIDTH = 32                    # U-Net base channels
WORKERS = 4
USE_DITHER = True             # Add stochastic dither before 8-bit quantization (reduces banding)
USE_TONE_JITTER = True        # Mild tone-curve jitter in linear space (robustness)
USE_GRAD_LOSS = False         # Optional gradient loss
OKLAB_W = 0.25                # Oklab term weight
SAVE_NAME = "dequantizer.pt"  # Checkpoint filename
SAVE_EVERY_STEPS = 250         # Save intermediate checkpoint every N training steps (0 disables)
VAL_SPLIT = 0.05              # Fraction of patches for validation
SEED = 42

TARGET_BITS = 16              # <- We train to recover 16-bit/channel sRGB
TARGET_MAX = float((1 << TARGET_BITS) - 1)

# Dequantization head behavior
OFFSET_CAP_LSB = 2.0             # Allow up to ±N/255 residual (robust mode)
STRICT_DEQUANT_BIN = False       # If True, clamp to input's quantization bin (±0.5/255)

# Jitter/dither controls
DITHER_PROB = 0.5                # Probability to apply dither for a sample
TONE_JITTER_MAX_PCT = 0.02       # Max ±% gamma jitter in linear domain

# Extra bit-depth degradation before 8-bit (to learn to fill bins)
EXTRA_Q_PROB = 0.4               # Probability to reduce to [MIN,MAX] bits before 8-bit
EXTRA_Q_BITS_RANGE = (6, 10)     # Inclusive range of bit depths

# Domain augmentations applied to the 8-bit input (simulate real-world pipelines)
USE_DOMAIN_AUG = True
JPEG_PROB = 0.3
JPEG_QUALITY_RANGE = (70, 95)
RESCALE_PROB = 0.3               # Random downscale then upscale
RESCALE_MIN_SCALE = 0.6          # Min scale factor for downscale
SHARPEN_PROB = 0.3               # Mild unsharp mask / sharpen
WB_JITTER_PROB = 0.3             # White-balance gains jitter
WB_JITTER_RANGE = (0.9, 1.1)

# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _worker_init(worker_id: int):
    seed = SEED + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception:
        pass

# ---- Color helpers (sRGB↔linear, Oklab) ----
def srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + a) / (1 + a), 2.4))

# =========================
# RAW demosaic → 16-bit sRGB
# =========================
def demosaic_raw_to_srgb16(path: str) -> np.ndarray:
    """
    Returns HxWx3 uint16, sRGB gamma, no auto brightening.
    """
    with rawpy.imread(path) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(2.222, 4.5),              # sRGB-ish
            output_color=rawpy.ColorSpace.sRGB
        )
    return rgb16  # uint16

def get_raw_size(path: str) -> Tuple[int,int]:
    """
    Get the demosaiced output size without doing full postprocess.
    Uses raw visible area (iheight, iwidth).
    """
    with rawpy.imread(path) as raw:
        return raw.sizes.iheight, raw.sizes.iwidth

# =========================
# 16b → 8b per-channel quantizer
# =========================
def apply_tone_jitter_srgb(img01: np.ndarray, max_gamma_jitter_pct: float = TONE_JITTER_MAX_PCT) -> np.ndarray:
    """
    Mild gamma jitter in *linear* domain for robustness, then back to sRGB.
    img01: float32 [0,1] sRGB
    """
    # sRGB->linear
    a = 0.055
    img_lin = np.where(img01 <= 0.04045, img01/12.92, ((img01+a)/1.055)**2.4).astype(np.float32)
    jitter = float(max(0.0, max_gamma_jitter_pct))
    if jitter > 0:
        g = np.random.uniform(1.0 - jitter, 1.0 + jitter)
        img_lin = np.clip(img_lin, 0, 1) ** g
    else:
        img_lin = np.clip(img_lin, 0, 1)
    # linear->sRGB
    img_srgb = np.where(img_lin <= 0.0031308, 12.92*img_lin, 1.055*np.power(img_lin, 1/2.4) - 0.055)
    return np.clip(img_srgb, 0.0, 1.0).astype(np.float32)

def downquantize_to_8b_from_16b(tgt16_uint16: np.ndarray,
                                dither: bool = True,
                                tone_jitter: bool = True,
                                tone_jitter_max_pct: float = TONE_JITTER_MAX_PCT,
                                extra_bits: Optional[int] = None) -> np.ndarray:
    """
    From 16-bit sRGB uint16 target → synthesize 8-bit-per-channel *input*.
    Returns uint8 HxWx3.
    """
    assert tgt16_uint16.dtype == np.uint16 and tgt16_uint16.ndim == 3 and tgt16_uint16.shape[2] >= 3
    # Normalize target to sRGB [0,1]
    tgt01 = (tgt16_uint16.astype(np.float32) / TARGET_MAX).clip(0, 1)

    if tone_jitter:
        tgt01 = apply_tone_jitter_srgb(tgt01, max_gamma_jitter_pct=tone_jitter_max_pct)

    if extra_bits is not None:
        # Reduce effective precision in sRGB before re-quantizing to 8-bit
        bits = int(extra_bits)
        bits = max(2, min(16, bits))
        levels = float((1 << bits) - 1)
        tgt01 = np.round(np.clip(tgt01, 0.0, 1.0) * levels) / levels

    if dither:
        # Blue-ish noise is ideal, but uniform noise works and is fast:
        # quantization noise amplitude ~ 1 LSB of 8-bit -> +/- 0.5/255
        noise = (np.random.rand(*tgt01.shape).astype(np.float32) - 0.5) / 255.0
        x = np.clip(tgt01 + noise, 0.0, 1.0)
    else:
        x = tgt01

    # Quantize per-channel to 8-bit
    inp8 = np.round(x * 255.0).astype(np.uint8)
    return inp8

# =========================
# Dataset (RAW → patches)
# =========================
class RawPatchDataset(Dataset):
    """
    Each item: (inp, tgt)
      - tgt: 16-bit sRGB normalized to [0,1] float32 (dequantization target)
      - inp: same image quantized to 8-bit per channel, normalized to [0,1] float32
    Patches are cropped from the demosaiced image.
    Small in-memory cache avoids re-demosaicing the same file repeatedly.
    """
    def __init__(self, raw_glob: str, patch: int = 128, stride: Optional[int] = None,
                 use_dither: bool = True, use_tone_jitter: bool = True, cache_images: int = 2,
                 paths_override: Optional[List[str]] = None,
                 dither_prob: Optional[float] = None,
                 tone_jitter_max_pct: float = TONE_JITTER_MAX_PCT,
                 extra_q_prob: float = EXTRA_Q_PROB,
                 extra_q_bits_range: Tuple[int,int] = EXTRA_Q_BITS_RANGE,
                 use_domain_aug: bool = USE_DOMAIN_AUG):
        super().__init__()
        self.patch = int(patch)
        self.stride = int(stride) if stride is not None else int(patch)
        self.use_dither = bool(use_dither)
        self.use_tone_jitter = bool(use_tone_jitter)
        self.paths = sorted(glob.glob(raw_glob)) if paths_override is None else list(paths_override)
        if not self.paths:
            raise FileNotFoundError(f"No RAW files match: {raw_glob}")

        # Per-sample synthesis controls (single gate)
        self.dither_prob = float(0.0 if dither_prob is None else float(np.clip(dither_prob, 0.0, 1.0)))
        self.tone_jitter_max_pct = float(tone_jitter_max_pct)
        self.extra_q_prob = float(extra_q_prob)
        self.extra_q_bits_range = (int(extra_q_bits_range[0]), int(extra_q_bits_range[1]))
        self.use_domain_aug = bool(use_domain_aug)

        # Precompute patch coordinates per file using RAW sizes (no demosaic yet)
        self.index: List[Tuple[int,int,int]] = []  # (path_idx, y, x)
        self.sizes: List[Tuple[int,int]] = []
        for i, p in enumerate(self.paths):
            H, W = get_raw_size(p)
            self.sizes.append((H, W))
            if H < self.patch or W < self.patch:
                continue
            for y in range(0, H - self.patch + 1, self.stride):
                for x in range(0, W - self.patch + 1, self.stride):
                    self.index.append((i, y, x))
        if not self.index:
            raise RuntimeError("No patches created — adjust PATCH/STRIDE or image sizes.")

        # Tiny process-local cache to avoid re-demosaicing the same file for many patches
        self._cache: dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_limit = int(cache_images)

    def _cache_get(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        # Demosaic and insert
        arr16 = demosaic_raw_to_srgb16(path)  # HxWx3 uint16
        if self._cache_limit > 0:
            self._cache[path] = arr16
            self._cache_order.append(path)
            if len(self._cache_order) > self._cache_limit:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return arr16

    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int):
        pidx, y, x = self.index[idx]
        path = self.paths[pidx]
        arr16 = self._cache_get(path)  # uint16 sRGB
        H2, W2 = int(arr16.shape[0]), int(arr16.shape[1])
        # Clamp crop coords to ensure a full patch fits within arr16
        if H2 >= self.patch and W2 >= self.patch:
            y0 = min(int(y), max(0, H2 - self.patch))
            x0 = min(int(x), max(0, W2 - self.patch))
            tgt16_patch = arr16[y0:y0+self.patch, x0:x0+self.patch, :3]
        else:
            # Extremely rare: demosaiced output smaller than PATCH; pad to PATCH
            src = arr16[:, :, :3]
            pad_h = max(0, self.patch - H2)
            pad_w = max(0, self.patch - W2)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            tgt16_padded = np.pad(src, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='edge')
            tgt16_patch = tgt16_padded[:self.patch, :self.patch, :3]

        # Build 8-bit input from this high-bit target
        # Decide stochastic synthesis options
        do_dither = (np.random.rand() < self.dither_prob)
        do_tonejit = self.use_tone_jitter
        extra_bits = None
        if np.random.rand() < self.extra_q_prob:
            lo, hi = self.extra_q_bits_range
            if lo > hi:
                lo, hi = hi, lo
            extra_bits = int(np.random.randint(lo, hi + 1))

        inp8_patch = downquantize_to_8b_from_16b(
            tgt16_patch,
            dither=do_dither,
            tone_jitter=do_tonejit,
            tone_jitter_max_pct=self.tone_jitter_max_pct,
            extra_bits=extra_bits
        )  # uint8

        # Optional domain augmentations on the 8-bit input
        if self.use_domain_aug:
            inp8_patch = self._domain_aug_uint8(inp8_patch)

        # Enforce original patch size after augs (safety against any drift)
        if inp8_patch.shape[0] != self.patch or inp8_patch.shape[1] != self.patch:
            try:
                from PIL import Image as _Img
                inp8_patch = np.array(_Img.fromarray(inp8_patch).resize((self.patch, self.patch), resample=Image.BILINEAR), dtype=np.uint8)
            except Exception:
                # As a last resort, center-crop or pad to target size
                h, w = int(inp8_patch.shape[0]), int(inp8_patch.shape[1])
                out = np.zeros((self.patch, self.patch, 3), dtype=np.uint8)
                y0 = max(0, (self.patch - h) // 2)
                x0 = max(0, (self.patch - w) // 2)
                hh = min(h, self.patch)
                ww = min(w, self.patch)
                out[y0:y0+hh, x0:x0+ww] = inp8_patch[:hh, :ww]
                inp8_patch = out

        # Normalize both to [0,1] float32
        inp = (inp8_patch.astype(np.float32) / 255.0)
        tgt = (tgt16_patch.astype(np.float32) / TARGET_MAX)

        # HWC->CHW torch (clone to ensure torch-owned, resizable storage for safe collation)
        inp = torch.from_numpy(np.transpose(inp, (2,0,1))).float().clone()
        tgt = torch.from_numpy(np.transpose(tgt, (2,0,1))).float().clone()
        return inp, tgt

    def _domain_aug_uint8(self, img: np.ndarray) -> np.ndarray:
        orig_h, orig_w, c = img.shape
        out = img
        # JPEG on 8-bit sRGB
        if np.random.rand() < JPEG_PROB:
            qmin, qmax = JPEG_QUALITY_RANGE
            q = int(np.random.randint(int(qmin), int(qmax)+1))
            try:
                import io
                pil = Image.fromarray(out)
                buf = io.BytesIO()
                # Random subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0
                subsampling = int(np.random.choice([0,1,2], p=[0.2, 0.3, 0.5]))
                pil.save(buf, format='JPEG', quality=q, subsampling=subsampling)
                buf.seek(0)
                out = np.array(Image.open(buf).convert('RGB'), dtype=np.uint8)
            except Exception:
                pass

        # Random downscale-upscale
        if np.random.rand() < RESCALE_PROB:
            try:
                # Use current image size for the downscale, then restore to original patch size
                curr_h, curr_w = int(out.shape[0]), int(out.shape[1])
                if curr_h > 0 and curr_w > 0 and orig_h > 0 and orig_w > 0:
                    scale = float(np.random.uniform(RESCALE_MIN_SCALE, 1.0))
                    new_h = max(1, int(round(curr_h * scale)))
                    new_w = max(1, int(round(curr_w * scale)))
                    pil = Image.fromarray(out)
                    pil_small = pil.resize((new_w, new_h), resample=Image.BILINEAR)
                    pil_back = pil_small.resize((orig_w, orig_h), resample=Image.BILINEAR)
                    out = np.array(pil_back, dtype=np.uint8)
            except Exception:
                # Skip rescale augmentation on any error
                pass

        # Sharpen / Unsharp mask
        if np.random.rand() < SHARPEN_PROB:
            try:
                pil = Image.fromarray(out)
                out = np.array(pil.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=0)), dtype=np.uint8)
            except Exception:
                pass

        # WB jitter as per-channel gain in sRGB
        if np.random.rand() < WB_JITTER_PROB:
            g = np.random.uniform(WB_JITTER_RANGE[0], WB_JITTER_RANGE[1], size=(1,1,3)).astype(np.float32)
            out = np.clip(out.astype(np.float32) * g, 0, 255).astype(np.uint8)

        # Final safety: ensure size is exactly the original patch size
        if out.shape[0] != orig_h or out.shape[1] != orig_w:
            try:
                out = np.array(Image.fromarray(out).resize((orig_w, orig_h), resample=Image.BILINEAR), dtype=np.uint8)
            except Exception:
                # Fallback to crop/pad
                h, w = int(out.shape[0]), int(out.shape[1])
                fixed = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                y0 = max(0, (orig_h - h) // 2)
                x0 = max(0, (orig_w - w) // 2)
                hh = min(h, orig_h)
                ww = min(w, orig_w)
                fixed[y0:y0+hh, x0:x0+ww] = out[:hh, :ww]
                out = fixed

        return out

# =========================
# Model
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )
    def forward(self, x): return self.b(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.d = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.d(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.u = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.u(x)

class DequantUNet(nn.Module):
    """
    Learns to 'dequantize' 8b/channel inputs toward 16b/channel targets.
    Residual prediction with zero-inited head (identity start).
    Output is in [0,1] sRGB, same dynamic range as target.
    """
    def __init__(self, channels=3, width=32):
        super().__init__()
        c = width
        self.enc1 = ConvBlock(channels, c)
        self.down1 = Down(c)
        self.enc2 = ConvBlock(c, c*2)
        self.down2 = Down(c*2)
        self.enc3 = ConvBlock(c*2, c*4)

        self.up2  = Up(c*4, c*2)
        self.dec2 = ConvBlock(c*4, c*2)
        self.up1  = Up(c*2, c)
        self.dec1 = ConvBlock(c*2, c)

        self.head = nn.Conv2d(c, channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # The true underlying 16b value lies within ±1 LSB (8-bit) of the 8-bit level,
        # so allowing offsets up to ~1/255 is theoretically sufficient.
        self.offset_scale = float(OFFSET_CAP_LSB) / 255.0

    def forward(self, x_srgb01):
        e1 = self.enc1(x_srgb01)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        offset = torch.tanh(self.head(d1)) * self.offset_scale
        y = torch.clamp(x_srgb01 + offset, 0.0, 1.0)
        if STRICT_DEQUANT_BIN:
            q = torch.round(x_srgb01 * 255.0) / 255.0
            y = torch.clamp(y, q - (0.5/255.0), q + (0.5/255.0))
            y = torch.clamp(y, 0.0, 1.0)
        return y, offset

# =========================
# Loss & Metrics
# =========================
class ColorAwareLoss(nn.Module):
    def __init__(self, use_grad=True, grad_w=0.05, oklab_w=0.25):
        super().__init__()
        self.use_grad = use_grad
        self.grad_w = float(grad_w)
        self.oklab_w = float(oklab_w)
        if use_grad:
            gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)/4.0
            gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)/4.0
            self.register_buffer('gx', gx)
            self.register_buffer('gy', gy)

        # Oklab matrices as buffers
        M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                           [0.2119034982, 0.6806995451, 0.1073969566],
                           [0.0883024619, 0.2817188376, 0.6299787005]], dtype=torch.float32)
        M2 = torch.tensor([[ 0.2104542553, 0.7936177850, -0.0040720468],
                           [ 1.9779984951,-2.4285922050,  0.4505937099],
                           [ 0.0259040371, 0.7827717662, -0.8086757660]], dtype=torch.float32)
        self.register_buffer('M1', M1)
        self.register_buffer('M2', M2)

        # BT.709 luminance weights (linear light)
        luma = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(1,3,1,1)
        self.register_buffer('luma_w', luma)

    def grad_map(self, x):
        gx = F.conv2d(x, self.gx.expand(x.size(1),1,3,3), padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.gy.expand(x.size(1),1,3,3), padding=1, groups=x.size(1))
        return torch.sqrt(gx*gx + gy*gy + 1e-12)

    def forward(self, pred_srgb01, tgt_srgb01):
        l1 = F.l1_loss(pred_srgb01, tgt_srgb01)
        # Do color math in float32 for stability (disable autocast)
        device_type = pred_srgb01.device.type if hasattr(pred_srgb01, 'device') else 'cuda'
        with torch.autocast(device_type=device_type, enabled=False):
            p = torch.clamp(pred_srgb01, 0, 1).float()
            t = torch.clamp(tgt_srgb01, 0, 1).float()
            p_lin = srgb_to_linear_torch(p)
            t_lin = srgb_to_linear_torch(t)
            # Oklab via preallocated float32 buffers
            p_lms = torch.einsum('ij,bjhw->bihw', self.M1, p_lin).clamp_min(1e-8)
            t_lms = torch.einsum('ij,bjhw->bihw', self.M1, t_lin).clamp_min(1e-8)
            p_lab = torch.einsum('ij,bjhw->bihw', self.M2, p_lms.pow(1/3))
            t_lab = torch.einsum('ij,bjhw->bihw', self.M2, t_lms.pow(1/3))
            l_lab = F.mse_loss(p_lab, t_lab)
            if self.use_grad:
                # Linear-light luminance using BT.709 weights
                p_y = (p_lin * self.luma_w).sum(dim=1, keepdim=True)
                t_y = (t_lin * self.luma_w).sum(dim=1, keepdim=True)
                gp = self.grad_map(p_y)
                gt = self.grad_map(t_y)
                l_grad = F.l1_loss(gp, gt)
            else:
                l_grad = None
        return l1 + self.oklab_w * l_lab + (self.grad_w * l_grad if (self.use_grad and l_grad is not None) else 0.0)

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10((max_val**2) / mse)

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, opt, loss_fn, accelerator, epoch=None, total_epochs=None,
                    run_dir: Optional[str] = None, global_step: int = 0):
    model.train()
    loss_sum = 0.0
    psnr_sum = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}" if accelerator.is_main_process else None, leave=False, dynamic_ncols=True) if accelerator.is_main_process else loader
    for batch_idx, (inp, tgt) in enumerate(pbar):
        inp = inp.to(accelerator.device, non_blocking=True)
        tgt = tgt.to(accelerator.device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with accelerator.autocast():
            pred, _ = model(inp)
            loss = loss_fn(pred, tgt)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Update global step and optionally save checkpoint
        global_step += 1
        if accelerator.is_main_process and run_dir is not None and SAVE_EVERY_STEPS and (global_step % int(SAVE_EVERY_STEPS) == 0):
            to_save = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "config": {
                    "raw_glob": RAW_GLOB,
                    "patch": PATCH, "stride": STRIDE, "batch": BATCH,
                    "epochs": EPOCHS, "lr": LR, "width": WIDTH, "workers": WORKERS,
                    "use_dither": USE_DITHER, "use_tone_jitter": USE_TONE_JITTER,
                    "use_grad_loss": USE_GRAD_LOSS, "oklab_w": OKLAB_W,
                    "val_split": VAL_SPLIT, "seed": SEED,
                    "target_bits": TARGET_BITS, "channels": 3
                },
                "epoch": int(epoch) if epoch is not None else None,
                "global_step": int(global_step)
            }
            to_save["config"].update({
                "offset_cap_lsb": OFFSET_CAP_LSB,
                "strict_dequant_bin": STRICT_DEQUANT_BIN,
                "dither_prob": DITHER_PROB,
                "tone_jitter_max_pct": TONE_JITTER_MAX_PCT,
                "extra_q_prob": EXTRA_Q_PROB,
                "extra_q_bits_range": EXTRA_Q_BITS_RANGE,
                "use_domain_aug": USE_DOMAIN_AUG,
                "jpeg_quality_range": JPEG_QUALITY_RANGE,
                "rescale_min_scale": RESCALE_MIN_SCALE,
                "wb_jitter_range": WB_JITTER_RANGE
            })
            ckpt_path = os.path.join(run_dir, f"step_{global_step:07d}.pt")
            try:
                torch.save(to_save, ckpt_path)
            except Exception:
                pass
        b = inp.size(0)
        loss_sum += loss.detach().item() * b
        psnr_sum += psnr(pred.detach(), tgt).item() * b
        n += b
        if accelerator.is_main_process:
            pbar.set_postfix({'Loss': f'{loss_sum/n:.4f}', 'PSNR': f'{psnr_sum/n:.2f}dB'})
    # Reduce across processes
    loss_mean = float(accelerator.gather_for_metrics(torch.tensor(loss_sum, device=accelerator.device)).sum() /
                      accelerator.gather_for_metrics(torch.tensor(n, device=accelerator.device, dtype=torch.long)).sum().clamp_min(1))
    psnr_mean = float(accelerator.gather_for_metrics(torch.tensor(psnr_sum, device=accelerator.device)).sum() /
                      accelerator.gather_for_metrics(torch.tensor(n, device=accelerator.device, dtype=torch.long)).sum().clamp_min(1))
    return loss_mean, psnr_mean, global_step

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, accelerator, epoch=None, total_epochs=None):
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Val {epoch}/{total_epochs}" if accelerator.is_main_process else None, leave=False, dynamic_ncols=True) if accelerator.is_main_process else loader
    for batch_idx, (inp, tgt) in enumerate(pbar):
        inp = inp.to(accelerator.device, non_blocking=True)
        tgt = tgt.to(accelerator.device, non_blocking=True)
        with accelerator.autocast():
            pred, _ = model(inp)
            loss = loss_fn(pred, tgt)
        b = inp.size(0)
        loss_sum += loss.detach().item() * b
        psnr_sum += psnr(pred, tgt).item() * b
        n += b
        if accelerator.is_main_process:
            pbar.set_postfix({'Val_Loss': f'{loss_sum/n:.4f}', 'Val_PSNR': f'{psnr_sum/n:.2f}dB'})
    loss_mean = float(accelerator.gather_for_metrics(torch.tensor(loss_sum, device=accelerator.device)).sum() /
                      accelerator.gather_for_metrics(torch.tensor(n, device=accelerator.device, dtype=torch.long)).sum().clamp_min(1))
    psnr_mean = float(accelerator.gather_for_metrics(torch.tensor(psnr_sum, device=accelerator.device)).sum() /
                      accelerator.gather_for_metrics(torch.tensor(n, device=accelerator.device, dtype=torch.long)).sum().clamp_min(1))
    return loss_mean, psnr_mean

# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print("Launching with 🤗 Accelerate")
        print(f"Config: PATCH={PATCH}, STRIDE={STRIDE or PATCH}, BATCH={BATCH}, EPOCHS={EPOCHS}, LR={LR}, WIDTH={WIDTH}, TARGET_BITS={TARGET_BITS}")
        print(f"RAW_GLOB: {RAW_GLOB}")

    # Datasets — split BY FILE to avoid leakage
    all_paths = sorted(glob.glob(RAW_GLOB))
    if not all_paths:
        raise FileNotFoundError(f"No RAW files match: {RAW_GLOB}")
    random.shuffle(all_paths)
    n_val_files = max(1, int(round(len(all_paths) * VAL_SPLIT)))
    val_paths = all_paths[:n_val_files]
    train_paths = all_paths[n_val_files:]
    if accelerator.is_main_process:
        print(f"Train files: {len(train_paths)} | Val files: {len(val_paths)}")

    train_ds = RawPatchDataset(
        raw_glob=RAW_GLOB, patch=PATCH, stride=STRIDE,
        use_dither=USE_DITHER, use_tone_jitter=USE_TONE_JITTER, cache_images=2,
        paths_override=train_paths,
        dither_prob=DITHER_PROB,
        tone_jitter_max_pct=TONE_JITTER_MAX_PCT,
        extra_q_prob=EXTRA_Q_PROB,
        extra_q_bits_range=EXTRA_Q_BITS_RANGE,
        use_domain_aug=USE_DOMAIN_AUG
    )

    # Clean validation (no dither, no tone jitter, no extra quantization, no domain augs)
    val_ds_clean = RawPatchDataset(
        raw_glob=RAW_GLOB, patch=PATCH, stride=STRIDE,
        use_dither=False, use_tone_jitter=False, cache_images=2,
        paths_override=val_paths,
        dither_prob=0.0,
        tone_jitter_max_pct=0.0,
        extra_q_prob=0.0,
        extra_q_bits_range=EXTRA_Q_BITS_RANGE,
        use_domain_aug=False
    )

    # Augmented validation (mirrors training augs)
    val_ds_aug = RawPatchDataset(
        raw_glob=RAW_GLOB, patch=PATCH, stride=STRIDE,
        use_dither=USE_DITHER, use_tone_jitter=USE_TONE_JITTER, cache_images=2,
        paths_override=val_paths,
        dither_prob=DITHER_PROB,
        tone_jitter_max_pct=TONE_JITTER_MAX_PCT,
        extra_q_prob=EXTRA_Q_PROB,
        extra_q_bits_range=EXTRA_Q_BITS_RANGE,
        use_domain_aug=USE_DOMAIN_AUG
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True if WORKERS > 0 else False,
                              prefetch_factor=4 if WORKERS > 0 else None,
                              worker_init_fn=_worker_init)
    val_loader_clean = DataLoader(val_ds_clean, batch_size=BATCH, shuffle=False,
                              num_workers=WORKERS, pin_memory=True,
                              persistent_workers=True if WORKERS > 0 else False,
                              prefetch_factor=4 if WORKERS > 0 else None,
                              worker_init_fn=_worker_init)
    val_loader_aug   = DataLoader(val_ds_aug, batch_size=BATCH, shuffle=False,
                              num_workers=WORKERS, pin_memory=True,
                              persistent_workers=True if WORKERS > 0 else False,
                              prefetch_factor=4 if WORKERS > 0 else None,
                              worker_init_fn=_worker_init)

    # Model / Opt / Loss
    model = DequantUNet(channels=3, width=WIDTH).to(accelerator.device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)
    loss_fn = ColorAwareLoss(use_grad=USE_GRAD_LOSS, grad_w=0.05, oklab_w=OKLAB_W)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)

    model, opt, train_loader, val_loader_clean, val_loader_aug, loss_fn = accelerator.prepare(model, opt, train_loader, val_loader_clean, val_loader_aug, loss_fn)

    # Run directory
    run_dir = os.path.join("runs", "dequantizer", time.strftime("%Y%m%d-%H%M%S"))
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    patience = 5
    epochs_no_improve = 0
    epoch_iter = tqdm(range(1, EPOCHS + 1), desc="Training Progress", dynamic_ncols=True) if accelerator.is_main_process else range(1, EPOCHS + 1)
    global_step = 0
    for epoch in epoch_iter:
        tr_loss, tr_psnr, global_step = train_one_epoch(model, train_loader, opt, loss_fn, accelerator, epoch, EPOCHS, run_dir=run_dir, global_step=global_step)
        va_loss_clean, va_psnr_clean = eval_one_epoch(model, val_loader_clean, loss_fn, accelerator, epoch, EPOCHS)
        va_loss_aug, va_psnr_aug = eval_one_epoch(model, val_loader_aug, loss_fn, accelerator, epoch, EPOCHS)
        scheduler.step(va_loss_clean)

        if accelerator.is_main_process:
            epoch_iter.set_postfix({
                'Train_Loss': f'{tr_loss:.4f}', 'Train_PSNR': f'{tr_psnr:.2f}dB',
                'ValClean_L': f'{va_loss_clean:.4f}', 'ValClean_P': f'{va_psnr_clean:.2f}dB',
                'ValAug_L': f'{va_loss_aug:.4f}', 'ValAug_P': f'{va_psnr_aug:.2f}dB',
                'LR': f'{opt.param_groups[0]["lr"]:.2e}',
                'Best': f'{best_val:.4f}' if best_val != float("inf") else 'N/A'
            })
            print(f"\n[Epoch {epoch:03d}/{EPOCHS}] Train: L1={tr_loss:.4f}, PSNR={tr_psnr:.2f}dB | "
                  f"ValClean: L1={va_loss_clean:.4f}, PSNR={va_psnr_clean:.2f}dB | "
                  f"ValAug: L1={va_loss_aug:.4f}, PSNR={va_psnr_aug:.2f}dB | "
                  f"LR={opt.param_groups[0]['lr']:.2e} | Best Val: {best_val:.4f} (Epoch {best_epoch})")

        improved = va_loss_clean < best_val - 0.0
        if accelerator.is_main_process and improved:
            best_val = va_loss_clean
            best_epoch = epoch
            epochs_no_improve = 0
            to_save = {
                "model": accelerator.unwrap_model(model).state_dict(),
                # Keep *lowercase* keys too so inference_utils can read them easily
                "config": {
                    "raw_glob": RAW_GLOB,
                    "patch": PATCH, "stride": STRIDE, "batch": BATCH,
                    "epochs": EPOCHS, "lr": LR, "width": WIDTH, "workers": WORKERS,
                    "use_dither": USE_DITHER, "use_tone_jitter": USE_TONE_JITTER,
                    "use_grad_loss": USE_GRAD_LOSS, "oklab_w": OKLAB_W,
                    "val_split": VAL_SPLIT, "seed": SEED,
                    "target_bits": TARGET_BITS, "channels": 3
                },
                "epoch": epoch, "best_val": best_val
            }
            # Enrich config with new controls
            to_save["config"].update({
                "offset_cap_lsb": OFFSET_CAP_LSB,
                "strict_dequant_bin": STRICT_DEQUANT_BIN,
                "dither_prob": DITHER_PROB,
                "tone_jitter_max_pct": TONE_JITTER_MAX_PCT,
                "extra_q_prob": EXTRA_Q_PROB,
                "extra_q_bits_range": EXTRA_Q_BITS_RANGE,
                "use_domain_aug": USE_DOMAIN_AUG,
                "jpeg_quality_range": JPEG_QUALITY_RANGE,
                "rescale_min_scale": RESCALE_MIN_SCALE,
                "wb_jitter_range": WB_JITTER_RANGE
            })
            torch.save(to_save, os.path.join(run_dir, SAVE_NAME))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if accelerator.is_main_process:
                    print(f"\n🛑 Early stopping at epoch {epoch}. Best val at epoch {best_epoch} (loss={best_val:.4f}).")
                break

    if accelerator.is_main_process:
        print(f"\n✅ Training completed! Best val loss: {best_val:.4f} (epoch {best_epoch}). Checkpoints in: {run_dir}")

if __name__ == "__main__":
    main()
