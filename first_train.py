import os, glob, time, math, random
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

import rawpy

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
VAL_SPLIT = 0.05              # Fraction of patches for validation
SEED = 42

# Step-based checkpointing
SAVE_EVERY_STEPS = 250        # Save a checkpoint every N training steps

TARGET_BITS = 16              # <- We train to recover 16-bit/channel sRGB
TARGET_MAX = float((1 << TARGET_BITS) - 1)

# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---- Color helpers (sRGB↔linear, Oklab) ----
def srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + a) / (1 + a), 2.4))

def rgb_to_oklab_torch(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: [B,3,H,W] in linear RGB [0,1]
    M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]],
                       dtype=rgb.dtype, device=rgb.device)
    lms = torch.einsum('ij, bjhw->bihw', M1, rgb)
    lms = torch.clamp(lms, min=1e-8)
    l_ = torch.pow(lms, 1/3)
    M2 = torch.tensor([[ 0.2104542553, 0.7936177850, -0.0040720468],
                       [ 1.9779984951,-2.4285922050,  0.4505937099],
                       [ 0.0259040371, 0.7827717662, -0.8086757660]],
                       dtype=rgb.dtype, device=rgb.device)
    return torch.einsum('ij, bjhw->bihw', M2, l_)

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
def apply_tone_jitter_srgb(img01: np.ndarray) -> np.ndarray:
    """
    Mild gamma jitter in *linear* domain for robustness, then back to sRGB.
    img01: float32 [0,1] sRGB
    """
    # sRGB->linear
    a = 0.055
    img_lin = np.where(img01 <= 0.04045, img01/12.92, ((img01+a)/1.055)**2.4).astype(np.float32)
    g = np.random.uniform(0.95, 1.05)
    img_lin = np.clip(img_lin, 0, 1) ** g
    # linear->sRGB
    img_srgb = np.where(img_lin <= 0.0031308, 12.92*img_lin, 1.055*np.power(img_lin, 1/2.4) - 0.055)
    return np.clip(img_srgb, 0.0, 1.0).astype(np.float32)

def downquantize_to_8b_from_16b(tgt16_uint16: np.ndarray,
                                dither: bool = True,
                                tone_jitter: bool = True) -> np.ndarray:
    """
    From 16-bit sRGB uint16 target → synthesize 8-bit-per-channel *input*.
    Returns uint8 HxWx3.
    """
    assert tgt16_uint16.dtype == np.uint16 and tgt16_uint16.ndim == 3 and tgt16_uint16.shape[2] >= 3
    # Normalize target to sRGB [0,1]
    tgt01 = (tgt16_uint16.astype(np.float32) / TARGET_MAX).clip(0, 1)

    if tone_jitter:
        tgt01 = apply_tone_jitter_srgb(tgt01)

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
                 use_dither: bool = True, use_tone_jitter: bool = True, cache_images: int = 2):
        super().__init__()
        self.patch = int(patch)
        self.stride = int(stride) if stride is not None else int(patch)
        self.use_dither = bool(use_dither)
        self.use_tone_jitter = bool(use_tone_jitter)
        self.paths = sorted(glob.glob(raw_glob))
        if not self.paths:
            raise FileNotFoundError(f"No RAW files match: {raw_glob}")

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
        H2, W2, _ = arr16.shape
        # Clamp coordinates to ensure we always extract a full patch
        y0 = y if y + self.patch <= H2 else max(0, H2 - self.patch)
        x0 = x if x + self.patch <= W2 else max(0, W2 - self.patch)
        tgt16_patch = arr16[y0:y0+self.patch, x0:x0+self.patch, :3]  # uint16

        # Build 8-bit input from this high-bit target
        inp8_patch = downquantize_to_8b_from_16b(
            tgt16_patch, dither=self.use_dither, tone_jitter=self.use_tone_jitter
        )  # uint8

        # Normalize both to [0,1] float32
        inp = (inp8_patch.astype(np.float32) / 255.0)
        tgt = (tgt16_patch.astype(np.float32) / TARGET_MAX)

        # HWC->CHW torch (clone to ensure torch-owned, resizable storage for safe collation)
        inp = torch.from_numpy(np.transpose(inp, (2,0,1))).float().clone()
        tgt = torch.from_numpy(np.transpose(tgt, (2,0,1))).float().clone()
        return inp, tgt

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
        self.u = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
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
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        # The true underlying 16b value lies within ±1 LSB (8-bit) of the 8-bit level,
        # so allowing offsets up to ~1/255 is theoretically sufficient.
        self.offset_scale = 1.0 / 255.0

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

    def grad_map(self, x):
        gx = F.conv2d(x, self.gx.expand(x.size(1),1,3,3), padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.gy.expand(x.size(1),1,3,3), padding=1, groups=x.size(1))
        return torch.sqrt(gx*gx + gy*gy + 1e-12)

    def forward(self, pred_srgb01, tgt_srgb01):
        l1 = F.l1_loss(pred_srgb01, tgt_srgb01)
        p_lin = srgb_to_linear_torch(torch.clamp(pred_srgb01, 0, 1))
        t_lin = srgb_to_linear_torch(torch.clamp(tgt_srgb01, 0, 1))
        p_lab = rgb_to_oklab_torch(p_lin)
        t_lab = rgb_to_oklab_torch(t_lin)
        l_lab = F.mse_loss(p_lab, t_lab)
        if not self.use_grad:
            return l1 + self.oklab_w * l_lab
        p_y = pred_srgb01.mean(dim=1, keepdim=True)
        t_y = tgt_srgb01.mean(dim=1, keepdim=True)
        gp = self.grad_map(p_y)
        gt = self.grad_map(t_y)
        l_grad = F.l1_loss(gp, gt)
        return l1 + self.oklab_w * l_lab + self.grad_w * l_grad

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10((max_val**2) / mse)

# =========================
# Safe collate
# =========================
def safe_collate(batch):
    # batch: list of tuples (inp, tgt)
    inps, tgts = zip(*batch)
    inps = [x.contiguous() for x in inps]
    tgts = [x.contiguous() for x in tgts]
    return torch.stack(inps, dim=0), torch.stack(tgts, dim=0)

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, opt, loss_fn, accelerator, epoch=None, total_epochs=None,
                    run_dir: Optional[str] = None, save_every_steps: Optional[int] = None,
                    global_step: int = 0, save_optimizer: bool = False):
    model.train()
    loss_sum = 0.0; psnr_sum = 0.0; n = 0
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
        global_step += 1
        b = inp.size(0)
        loss_sum += loss.detach().item() * b
        psnr_sum += psnr(pred.detach(), tgt).item() * b
        n += b
        if accelerator.is_main_process:
            pbar.set_postfix({'Loss': f'{loss_sum/n:.4f}', 'PSNR': f'{psnr_sum/n:.2f}dB'})
        # Step-based checkpoint save (main process only)
        if (save_every_steps is not None) and (run_dir is not None) and accelerator.is_main_process:
            if global_step % int(save_every_steps) == 0:
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
                    "epoch": epoch, "global_step": global_step
                }
                if save_optimizer:
                    try:
                        to_save["optimizer"] = opt.state_dict()
                    except Exception:
                        pass
                step_path = os.path.join(run_dir, f"step_{global_step:09d}.pt")
                torch.save(to_save, step_path)
    # Reduce across processes
    loss_mean = float(accelerator.gather_for_metrics(torch.tensor(loss_sum, device=accelerator.device)).sum() /
                      accelerator.gather_for_metrics(torch.tensor(n, device=accelerator.device, dtype=torch.long)).sum().clamp_min(1))
    psnr_mean = float(accelerator.gather_for_metrics(torch.tensor(psnr_sum, device=accelerator.device)).sum() /
                      accelerator.gather_for_metrics(torch.tensor(n, device=accelerator.device, dtype=torch.long)).sum().clamp_min(1))
    return loss_mean, psnr_mean, global_step

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, accelerator, epoch=None, total_epochs=None):
    model.eval()
    loss_sum = 0.0; psnr_sum = 0.0; n = 0
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

    # Datasets
    full_ds = RawPatchDataset(
        raw_glob=RAW_GLOB, patch=PATCH, stride=STRIDE,
        use_dither=USE_DITHER, use_tone_jitter=USE_TONE_JITTER, cache_images=2
    )
    N = len(full_ds)
    n_val = max(1, int(N * VAL_SPLIT))
    idx = torch.randperm(N)
    val_idx = idx[:n_val].tolist()
    trn_idx = idx[n_val:].tolist()

    train_ds = torch.utils.data.Subset(full_ds, trn_idx)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True if WORKERS > 0 else False,
                              prefetch_factor=4 if WORKERS > 0 else None,
                              collate_fn=safe_collate)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=WORKERS, pin_memory=True,
                              persistent_workers=True if WORKERS > 0 else False,
                              prefetch_factor=4 if WORKERS > 0 else None,
                              collate_fn=safe_collate)

    # Model / Opt / Loss
    model = DequantUNet(channels=3, width=WIDTH).to(accelerator.device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)
    loss_fn = ColorAwareLoss(use_grad=USE_GRAD_LOSS, grad_w=0.05, oklab_w=OKLAB_W)

    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    # Run directory
    run_dir = os.path.join("runs", "dequantizer", time.strftime("%Y%m%d-%H%M%S"))
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)

    best_val = float("inf"); best_epoch = 0; patience = 5; epochs_no_improve = 0
    epoch_iter = tqdm(range(1, EPOCHS + 1), desc="Training Progress", dynamic_ncols=True) if accelerator.is_main_process else range(1, EPOCHS + 1)
    global_step = 0
    for epoch in epoch_iter:
        tr_loss, tr_psnr, global_step = train_one_epoch(
            model, train_loader, opt, loss_fn, accelerator, epoch, EPOCHS,
            run_dir=run_dir, save_every_steps=SAVE_EVERY_STEPS, global_step=global_step, save_optimizer=True
        )
        va_loss, va_psnr = eval_one_epoch(model, val_loader, loss_fn, accelerator, epoch, EPOCHS)

        if accelerator.is_main_process:
            epoch_iter.set_postfix({
                'Train_Loss': f'{tr_loss:.4f}', 'Train_PSNR': f'{tr_psnr:.2f}dB',
                'Val_Loss': f'{va_loss:.4f}', 'Val_PSNR': f'{va_psnr:.2f}dB',
                'Best': f'{best_val:.4f}' if best_val != float("inf") else 'N/A'
            })
            print(f"\n[Epoch {epoch:03d}/{EPOCHS}] Train: L1={tr_loss:.4f}, PSNR={tr_psnr:.2f}dB | "
                  f"Val: L1={va_loss:.4f}, PSNR={va_psnr:.2f}dB | Best Val: {best_val:.4f} (Epoch {best_epoch})")

        improved = va_loss < best_val - 0.0
        if accelerator.is_main_process and improved:
            best_val = va_loss; best_epoch = epoch; epochs_no_improve = 0
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
