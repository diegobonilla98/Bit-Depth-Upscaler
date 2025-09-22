import os
from typing import Tuple, List, Optional, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
from PIL import Image
import tifffile as tiff


# =========================
# Image I/O
# =========================
def _imread_any_8bit(path: str) -> np.ndarray:
    """
    Read an image as HxWxC uint8. Supports PNG/JPEG/TIFF. Ensures HWC layout.
    """
    try:
        img = iio.imread(path)
    except Exception:
        try:
            img = np.array(Image.open(path))
        except Exception as e:
            raise RuntimeError(f"Failed to read 8-bit image {path}: {e}")

    if img.ndim == 2:
        img = img[..., None]
    elif img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _imwrite_tiff_uint32(path: str, arr_uint32: np.ndarray) -> None:
    """
    Write image as 16-bit-per-channel TIFF for broad viewer compatibility.

    Many viewers render 32-bit integer TIFFs as black. We clip to uint16 and tag.
    Accepts HxW, HxWx1 (grayscale) or HxWx3 (RGB).
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    arr = np.asarray(arr_uint32)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Expected integer array for TIFF save, got dtype={arr.dtype}")

    arr16 = np.clip(arr, 0, 65535).astype(np.uint16, copy=False)

    photometric = None
    if arr16.ndim == 3:
        c = arr16.shape[2]
        if c == 1:
            arr16 = arr16[..., 0]
            photometric = 'minisblack'
        elif c == 3:
            photometric = 'rgb'
    elif arr16.ndim == 2:
        photometric = 'minisblack'

    try:
        if photometric is not None:
            tiff.imwrite(path, arr16, photometric=photometric, compression='none')
        else:
            tiff.imwrite(path, arr16, compression='none')
        return
    except Exception:
        pass

    try:
        iio.imwrite(path, arr16)
        return
    except Exception as e:
        raise RuntimeError(f"Failed to write TIFF {path}: {e}")


# =========================
# Model loading (no Triton)
# =========================
def _wrap_predict(model: nn.Module):
    """
    Return pred(x)->y handling models that return (pred, aux) or pred.
    """
    def _pred(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out
    return _pred


def load_model_from_checkpoint(checkpoint_path: str,
                               device: Optional[torch.device],
                               model_class: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load trained model and return (model, config_dict).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('config', {}) or {}

    channels = ckpt.get('channels', cfg.get('channels', 3))
    
    # Handle different parameter names for different model classes
    if hasattr(model_class, '__name__') and 'ThirdDequantUNet' in model_class.__name__:
        # For ThirdDequantUNet, look for base_width in config
        base_width = cfg.get('base_width', cfg.get('WIDTH', 64))
        # Use the same parameters that were used during training
        depths = cfg.get('depths', (3, 6, 9))  # Default from training script
        attn_heads = cfg.get('attn_heads', (0, 4, 8))
        attn_window = cfg.get('attn_window', 8)
        use_se = cfg.get('use_se', True)
        drop_path_rate = cfg.get('drop_path_rate', 0.1)
        offset_cap_lsb = cfg.get('offset_cap_lsb', 2.0)
        strict_dequant_bin = cfg.get('strict_dequant_bin', False)
        
        model = model_class(
            channels=channels, 
            base_width=base_width,
            depths=depths,
            attn_heads=attn_heads,
            attn_window=attn_window,
            use_se=use_se,
            drop_path_rate=drop_path_rate,
            offset_cap_lsb=offset_cap_lsb,
            strict_dequant_bin=strict_dequant_bin
        ).to(device)
    else:
        # For FirstDequantUNet and SecondDequantUNet, use width
        width = cfg.get('width', cfg.get('WIDTH', 32))
        model = model_class(channels=channels, width=width).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    # perf knobs
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    # attach wrapped predict for clarity
    model._pred = _wrap_predict(model)
    return model, cfg


# =========================
# Sliding window (vectorized, micro-batched)
# =========================
@torch.no_grad()
def sliding_window_inference(
    img8: np.ndarray,
    model: torch.nn.Module,
    patch: int,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
    max_batch: int = 16,              # used to control tile micro-batch size
    amp: Optional[bool] = None,
) -> np.ndarray:
    """
    Vectorized sliding-window inference:
      - Build a strided view of all tiles (no per-tile Python loop)
      - Run the model over tiles in micro-batches (max_batch) to maximize throughput
      - Write all predictions into a single column tensor and do ONE global F.fold
      - Exact overlap averaging via a single folded weight map
    Returns: HxWxC float32 in [0,1]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if stride is None:
        stride = patch
    if amp is None:
        amp = (device.type == 'cuda')

    assert img8.ndim == 3, 'img8 must be HxWxC'
    H, W, C = int(img8.shape[0]), int(img8.shape[1]), int(img8.shape[2])

    # HWC uint8 -> NCHW float32 [0,1], channels-last for speed
    x = torch.from_numpy(img8.astype(np.float32) / 255.0)   # HWC
    x = x.permute(2, 0, 1).unsqueeze(0)                     # [1,C,H,W]
    x = x.to(device=device, memory_format=torch.channels_last)

    P = patch
    S = stride
    C_ = x.shape[1]

    # Reflect pad so grid covers the canvas with uniform stride
    Hmin = max(H, P)
    Wmin = max(W, P)
    Hp = ((Hmin - P + S - 1) // S) * S + P
    Wp = ((Wmin - P + S - 1) // S) * S + P
    pad_bottom = max(0, Hp - H)
    pad_right  = max(0, Wp - W)
    if pad_bottom > 0 or pad_right > 0:
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='reflect')

    # Build a strided view of tiles: [1,C,ny,nx,P,P] -> [L,C,P,P]
    ny = (x.shape[2] - P) // S + 1
    nx = (x.shape[3] - P) // S + 1
    L  = ny * nx

    # NOTE: The permute MUST keep the batch axis (0) — fixed here.
    tiles_view = x.unfold(2, P, S).unfold(3, P, S)          # [1,C,ny,nx,P,P]
    tiles_view = tiles_view.permute(0, 2, 3, 1, 4, 5)       # [1,ny,nx,C,P,P]
    tiles_view = tiles_view.reshape(L, C_, P, P)            # [L,C,P,P] (view when possible)

    # Pre-allocate output columns; we'll fill in micro-batches, then fold once
    cols_out = torch.empty((1, C_ * P * P, L), device=device, dtype=torch.float32)

    pred_fn = getattr(model, "_pred", None) or _wrap_predict(model)

    # Micro-batching loop (chunked over tile-batch only; math inside chunk is vectorized)
    B = max(1, int(max_batch))
    if amp and device.type == 'cuda':
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        autocast_ctx = _NoOp()

    with autocast_ctx:
        for start in range(0, L, B):
            end = min(start + B, L)
            # materialize a contiguous channels-last chunk
            tiles_chunk = tiles_view[start:end].contiguous(memory_format=torch.channels_last)  # [b,C,P,P]
            pred_chunk  = pred_fn(tiles_chunk)                                                # [b,C,P,P]

            # accumulate in fp32 for stable averaging
            pred_chunk = pred_chunk.to(dtype=torch.float32)

            # reshape to columns and place into the big out tensor
            pred_cols = pred_chunk.reshape(end - start, C_ * P * P).transpose(0, 1).unsqueeze(0)  # [1, C*P*P, b]
            cols_out[:, :, start:end] = pred_cols

    # One global fold (sum) + weight map
    out_sum = F.fold(cols_out, output_size=(x.shape[2], x.shape[3]), kernel_size=(P, P), stride=(S, S))  # [1,C,Hp,Wp]

    ones_cols = torch.ones((1, 1 * P * P, L), device=device, dtype=torch.float32)
    wmap      = F.fold(ones_cols, output_size=(x.shape[2], x.shape[3]), kernel_size=(P, P), stride=(S, S))  # [1,1,Hp,Wp]

    out = out_sum / (wmap + 1e-8)
    out = out[:, :, :H, :W]
    out = torch.clamp(out, 0.0, 1.0)

    # NCHW -> HWC float32
    out = out.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy().astype(np.float32)
    return out


# =========================
# High-level entry point
# =========================
@torch.no_grad()
def infer_from_file(
    input_path: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    patch: Optional[int] = None,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
    max_batch: int = 16,
    model_class: Optional[nn.Module] = None,
) -> np.ndarray:
    """
    Loads model, runs tiled inference, and (optionally) writes TIFF scaled to target bit-depth.
    Returns uint32 array scaled to [0, 2^target_bits - 1].
    """
    model, cfg = load_model_from_checkpoint(checkpoint_path, device=device, model_class=model_class)
    target_bits = int(cfg.get('target_bits', 16))
    max_code = float((1 << target_bits) - 1)

    img8 = _imread_any_8bit(input_path)

    # Expected channels
    expected_c = None
    head = getattr(model, 'head', None)
    if head is not None:
        expected_c = getattr(head, 'out_channels', None)
    if expected_c is None:
        expected_c = int(cfg.get('channels', img8.shape[2]))

    if img8.shape[2] != expected_c:
        raise ValueError(f"Input image has {img8.shape[2]} channels, but model expects {expected_c}.")

    if patch is None:
        patch = int(cfg.get('patch', cfg.get('PATCH', 128)))
    if stride is None:
        stride = patch

    pred_norm = sliding_window_inference(
        img8=img8,
        model=model,
        patch=patch,
        stride=stride,
        device=device,
        max_batch=max_batch,  # respected here
        amp=(device is None and torch.cuda.is_available()) or (device is not None and device.type == 'cuda'),
    )

    pred_scaled = np.clip(pred_norm * max_code + 0.5, 0, max_code).astype(np.uint32)
    if output_path is not None:
        _imwrite_tiff_uint32(output_path, pred_scaled)
    return pred_scaled
