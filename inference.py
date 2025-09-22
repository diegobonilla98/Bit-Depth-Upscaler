import os
import sys
import json
import math
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional

import models

import torch

import numpy as np
from tqdm import tqdm

from inference_utils import (
    load_model_from_checkpoint,
    sliding_window_inference,
    _imread_any_8bit,
    _imwrite_tiff_uint32,
)

import bit_depth_color_info as bci
import rawpy

# -------------------------
# User Configuration Variables
# -------------------------

model_class = models.ThirdDequantUNet

# Model checkpoint path
CHECKPOINT_PATH = "./third_step_0001750.pt"

# Input configuration - can be directories, individual files, or a text file with basenames
INPUT_DIRS = [
    # r"F:\Flowers102\archive\dataset\valid\1"
]

# Optional: text file with basenames (one per line)
# If provided, will look for these files in INPUT_DIRS
# Example: Set to "val_files.txt" to process only files listed in that file
INPUT_BASENAMES_FILE = "val_files.txt"

# Output configuration
OUTPUT_JSON = "eval_results.json"
SAVE_OUTPUTS = True
OUTPUTS_DIR = "eval_out"

# Model inference parameters
PATCH_SIZE = None  # None to use from checkpoint config
STRIDE = None      # None to use patch size
MAX_BATCH = 16     # Sliding window batch size
DEVICE = None      # None for auto-detect (cuda if available, else cpu)

# -------------------------
# Image IO & classification
# -------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXTS = set(getattr(bci, "RAW_EXTS", [])) or {
    ".nef", ".dng", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".raf", ".srw", ".pef", ".nrw"
}

def is_raw_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTS

def is_img_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS or is_raw_file(path)

def list_images_in_dirs(dirs: List[str], basenames_file: Optional[str] = None) -> List[str]:
    files = []
    
    if basenames_file and os.path.isfile(basenames_file):
        # Load specific basenames from text file
        print(f"Loading basenames from: {basenames_file}")
        with open(basenames_file, 'r', encoding='utf-8') as f:
            basenames = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(basenames)} basenames in file")
        
        # Look for each basename in the provided directories
        for basename in basenames:
            found = False
            
            # First, check if it's already a full path
            if os.path.isfile(basename):
                files.append(basename)
                found = True
            else:
                # Try to find the file in the provided directories
                for d in dirs:
                    if os.path.isdir(d):
                        # Try to find the file in this directory
                        for ext in list(IMG_EXTS) + list(RAW_EXTS):
                            candidate = os.path.join(d, basename)
                            if os.path.isfile(candidate):
                                files.append(candidate)
                                found = True
                                break
                        if found:
                            break
                    elif os.path.isfile(d) and os.path.basename(d) == basename:
                        files.append(d)
                        found = True
                        break
            
            if not found:
                print(f"  Warning: Could not find {basename} in any input directory")
    else:
        # Original behavior: scan directories for all image files
        for d in dirs:
            if os.path.isfile(d) and is_img_file(d):
                files.append(d)
                continue
            for ext in list(IMG_EXTS) + list(RAW_EXTS):
                files.extend(glob.glob(os.path.join(d, f"*{ext}")))
    
    return sorted(list({os.path.abspath(f): None for f in files}.keys()))

# -------------------------
# Color / math helpers
# -------------------------

def srgb_to_linear_np(x: np.ndarray) -> np.ndarray:
    """x in [0,1]"""
    a = 0.055
    out = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return out.astype(np.float32)

def linear_to_srgb_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    out = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1.0/2.4) - a)
    return out.astype(np.float32)

def psnr_np(pred: np.ndarray, tgt: np.ndarray, max_val: float = 1.0) -> float:
    """pred, tgt in [0,1] float32, any shape"""
    diff = (pred.astype(np.float64) - tgt.astype(np.float64))
    mse = float(np.mean(diff * diff))
    mse = max(mse, 1e-12)
    return 10.0 * math.log10((max_val ** 2) / mse)

def l1_np(pred: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float64) - tgt.astype(np.float64))))

def to_chw01(img: np.ndarray) -> np.ndarray:
    """HWC uint8/uint16/float -> CHW float32 in [0,1] (if uint inputs, use correct scale before)"""
    if img.ndim == 2:
        img = img[..., None]
    return np.transpose(img, (2, 0, 1)).astype(np.float32)

def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Take any 8-bit-like array and return HxWx3 uint8 RGB (drop alpha, replicate gray)."""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] >= 3:
            img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def ensure_rgb_uint16(img: np.ndarray) -> np.ndarray:
    """Return HxWx3 uint16 RGB (drop alpha, replicate gray)"""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] >= 3:
            img = img[..., :3]
    if img.dtype != np.uint16:
        # attempt to scale if uint8
        if img.dtype == np.uint8:
            img = (img.astype(np.uint16) * 257)  # 255->65535 scaling
        else:
            # float or other ints: clamp
            img = np.clip(img, 0, 65535).astype(np.uint16)
    return img

def quantize_to_8bit_from01(x01: np.ndarray) -> np.ndarray:
    """x01 in [0,1] -> uint8"""
    return np.clip(np.round(x01 * 255.0), 0, 255).astype(np.uint8)

def rebin_to01_from_uint8(x8: np.ndarray) -> np.ndarray:
    return (x8.astype(np.float32) / 255.0).clip(0.0, 1.0)

def rebin_to01_from_uint16(x16: np.ndarray) -> np.ndarray:
    return (x16.astype(np.float32) / 65535.0).clip(0.0, 1.0)

def demosaic_raw_to_srgb16_uint16(path: str) -> np.ndarray:
    """
    Match training demosaic: 16-bit sRGB gamma-ish, camera WB, no auto brightening.
    Returns HxWx3 uint16.
    """
    if rawpy is None:
        raise RuntimeError("rawpy is not available to decode RAW files.")
    with rawpy.imread(path) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(2.222, 4.5),  # sRGB-ish (like your training)
            output_color=rawpy.ColorSpace.sRGB,
        )
    return rgb16  # uint16

# -------------------------
# Banding / dequant metrics
# -------------------------

def luminance_linear_709(x01_srgb: np.ndarray) -> np.ndarray:
    """x01_srgb: HxWxC in [0,1] -> linear-light 709 luminance (H x W)"""
    x_lin = srgb_to_linear_np(np.clip(x01_srgb[..., :3], 0, 1))
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    y = np.tensordot(x_lin, w, axes=([-1], [0]))
    return y.astype(np.float32)

def zero_run_stats(y: np.ndarray, tau: float = 1.0 / 4096.0) -> Dict[str, float]:
    """
    y: luminance (H x W) float32 in [0,1] (linear)
    tau: threshold below which a first-difference is considered "zero"
    Returns average zero-run length and fraction of small diffs.
    """
    H, W = y.shape
    # diffs
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    zx = (dx <= tau)
    zy = (dy <= tau)

    # fraction of near-zero diffs
    zero_frac = float((zx.sum() + zy.sum()) / (zx.size + zy.size + 1e-12))

    # mean run length (simple scan)
    def mean_run(mask: np.ndarray) -> float:
        # mask: Hx(W-1) or (H-1)xW
        total_runs = 0
        total_len = 0
        # iterate rows for speed/memory
        for row in mask:
            # find run lengths
            if row.size == 0:
                continue
            rlen = 0
            prev = row[0]
            for v in row:
                if v == prev:
                    rlen += 1
                else:
                    total_runs += 1
                    total_len += rlen
                    rlen = 1
                    prev = v
            total_runs += 1
            total_len += rlen
        if total_runs == 0:
            return 0.0
        return total_len / total_runs

    mean_run_len = float((mean_run(zx) + mean_run(zy)) * 0.5)
    return {
        "zero_frac": zero_frac,
        "mean_run_len": mean_run_len,
    }

def banding_stress_metrics(x01_srgb_in: np.ndarray, x01_srgb_out: np.ndarray) -> Dict[str, Any]:
    """
    Apply strong tone edits and compare gradient smoothness (lower zero-runs is better).
    Returns metrics before/after for gamma 0.6 and 1.6, plus improvement percentages.
    """
    def stress_gamma(y01: np.ndarray, g: float) -> np.ndarray:
        y01 = np.clip(y01, 0, 1) ** g
        return y01

    y_in = luminance_linear_709(x01_srgb_in)
    y_out = luminance_linear_709(x01_srgb_out)

    results = {}
    for g in (0.6, 1.6):
        yin = stress_gamma(y_in, g)
        yout = stress_gamma(y_out, g)
        m_in = zero_run_stats(yin)
        m_out = zero_run_stats(yout)
        # improvement: reduction in zero-run length / fraction
        imp_len = (m_in["mean_run_len"] - m_out["mean_run_len"]) / (m_in["mean_run_len"] + 1e-9)
        imp_frac = (m_in["zero_frac"] - m_out["zero_frac"]) / (m_in["zero_frac"] + 1e-9)
        results[f"gamma_{g}"] = {
            "in": m_in,
            "out": m_out,
            "improvement": {
                "mean_run_len_reduction_pct": float(100.0 * imp_len),
                "zero_frac_reduction_pct": float(100.0 * imp_frac),
            }
        }
    return results

def cycle_and_bin_metrics(x8_uint8: np.ndarray, y01: np.ndarray) -> Dict[str, Any]:
    """
    x8_uint8: original 8-bit input (HxWxC)
    y01: model output in [0,1] float32
    """
    x01 = rebin_to01_from_uint8(x8_uint8)
    q8y = quantize_to_8bit_from01(y01)  # re-quantize model output
    q01 = rebin_to01_from_uint8(q8y)

    # channel-wise equality fraction
    match_chan = float(np.mean((q8y == x8_uint8)))
    # pixel-wise all-channels equality fraction
    match_pix = float(np.mean(np.all(q8y == x8_uint8, axis=-1)))

    # idempotence: run model again on quantized output (simulate)
    # NOTE: for speed we don't re-run model here; most users will skip.
    # We'll instead score y01 vs q01.
    idem_psnr = psnr_np(y01, q01)
    idem_l1 = l1_np(y01, q01)

    # bin adherence against the original quantization center
    q01_in = np.round(x01 * 255.0) / 255.0
    delta = np.abs(y01 - q01_in)
    within_0p5 = float(np.mean(delta <= (0.5 / 255.0)))
    within_1p0 = float(np.mean(delta <= (1.0 / 255.0)))
    within_2p0 = float(np.mean(delta <= (2.0 / 255.0)))

    # offset stats in LSBs
    offset_lsb = (y01 - x01) * 255.0
    offset_abs_mean = float(np.mean(np.abs(offset_lsb)))
    offset_abs_p95 = float(np.percentile(np.abs(offset_lsb), 95))
    offset_abs_p99 = float(np.percentile(np.abs(offset_lsb), 99))

    return {
        "requant_match_rate_channel": match_chan,
        "requant_match_rate_pixel": match_pix,
        "idempotence_psnr_vs_requant": idem_psnr,
        "idempotence_l1_vs_requant": idem_l1,
        "bin_adherence_within_0p5lsb": within_0p5,
        "bin_adherence_within_1p0lsb": within_1p0,
        "bin_adherence_within_2p0lsb": within_2p0,
        "offset_abs_mean_lsb": offset_abs_mean,
        "offset_abs_p95_lsb": offset_abs_p95,
        "offset_abs_p99_lsb": offset_abs_p99,
    }

# -------------------------
# Evaluation routines
# -------------------------

def eval_on_8bit(
    path: str,
    model,
    cfg: dict,
    device,
    patch: Optional[int],
    stride: Optional[int],
    max_batch: int,
    save_outputs: bool,
    outputs_dir: Optional[str],
) -> Dict[str, Any]:
    """
    8-bit input: run model; evaluate cycle/bin/banding metrics (no GT).
    """
    # read 8-bit RGB
    x8 = _imread_any_8bit(path)
    x8 = ensure_rgb_uint8(x8)

    # infer
    if patch is None:
        patch = int(cfg.get("patch", cfg.get("PATCH", 128)))
    if stride is None:
        stride = patch

    y01 = sliding_window_inference(
        img8=x8, model=model, patch=patch, stride=stride, device=device,
        max_batch=max_batch, amp=(device.type == "cuda"),
    )
    x01 = rebin_to01_from_uint8(x8)

    # metrics
    cyc = cycle_and_bin_metrics(x8, y01)
    band = banding_stress_metrics(x01, y01)

    # optional save
    out_path = None
    if save_outputs and outputs_dir:
        # save as 16-bit TIFF (uint32 container for safety)
        max_code = float((1 << int(cfg.get("target_bits", 16))) - 1)
        y_uint = np.clip(y01 * max_code + 0.5, 0, max_code).astype(np.uint32)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(outputs_dir, f"{stem}_dequant.tiff")
        _imwrite_tiff_uint32(out_path, y_uint)

    return {
        "type": "8bit",
        "width": int(x8.shape[1]),
        "height": int(x8.shape[0]),
        "channels": int(x8.shape[2]),
        "metrics": {
            **cyc,
            "banding_stress": band,
        },
        "output_path": out_path,
    }

def eval_on_16bit_or_raw(
    path: str,
    model,
    cfg: dict,
    device,
    patch: Optional[int],
    stride: Optional[int],
    max_batch: int,
    save_outputs: bool,
    outputs_dir: Optional[str],
) -> Dict[str, Any]:
    """
    16-bit/RAW input: synthesize 8-bit input, run model, compare to original 16-bit target.
    """
    # Load ground truth 16-bit sRGB (uint16)
    if is_raw_file(path):
        if rawpy is None:
            raise RuntimeError("rawpy not installed; cannot decode RAW.")
        gt16 = demosaic_raw_to_srgb16_uint16(path)
        src_type = "raw16"
    else:
        # assume 16 bpc PNG/TIFF loaded via Pillow
        li = bci.load_image(path)  # uses Pillow
        arr = li.arr
        if arr.dtype != np.uint16:
            # try forcing upcast
            arr = ensure_rgb_uint16(arr)
        else:
            arr = ensure_rgb_uint16(arr)
        gt16 = arr
        src_type = "png16"

    H, W, _ = gt16.shape
    gt01 = rebin_to01_from_uint16(gt16)  # sRGB in [0,1]

    # Synthesize 8-bit input by standard rounding (no dither)
    x8 = quantize_to_8bit_from01(gt01)
    x01 = rebin_to01_from_uint8(x8)

    # Infer
    if patch is None:
        patch = int(cfg.get("patch", cfg.get("PATCH", 128)))
    if stride is None:
        stride = patch

    y01 = sliding_window_inference(
        img8=x8, model=model, patch=patch, stride=stride, device=device,
        max_batch=max_batch, amp=(device.type == "cuda"),
    )

    # Metrics: sRGB
    base_psnr_srgb = psnr_np(x01, gt01)
    pred_psnr_srgb = psnr_np(y01, gt01)
    base_l1_srgb = l1_np(x01, gt01)
    pred_l1_srgb = l1_np(y01, gt01)

    # Metrics: linear
    x_lin = srgb_to_linear_np(x01)
    y_lin = srgb_to_linear_np(y01)
    gt_lin = srgb_to_linear_np(gt01)
    base_psnr_lin = psnr_np(x_lin, gt_lin)
    pred_psnr_lin = psnr_np(y_lin, gt_lin)
    base_l1_lin = l1_np(x_lin, gt_lin)
    pred_l1_lin = l1_np(y_lin, gt_lin)

    # Cycle/bin metrics (relative to synthesized 8-bit input)
    cyc = cycle_and_bin_metrics(x8, y01)

    # Banding stress
    band = banding_stress_metrics(x01, y01)

    # optional save
    out_path = None
    if save_outputs and outputs_dir:
        max_code = float((1 << int(cfg.get("target_bits", 16))) - 1)
        y_uint = np.clip(y01 * max_code + 0.5, 0, max_code).astype(np.uint32)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(outputs_dir, f"{stem}_dequant.tiff")
        _imwrite_tiff_uint32(out_path, y_uint)

    return {
        "type": src_type,
        "width": int(W),
        "height": int(H),
        "channels": 3,
        "metrics": {
            "baseline_psnr_srgb": base_psnr_srgb,
            "pred_psnr_srgb": pred_psnr_srgb,
            "delta_psnr_srgb": pred_psnr_srgb - base_psnr_srgb,
            "baseline_l1_srgb": base_l1_srgb,
            "pred_l1_srgb": pred_l1_srgb,
            "baseline_psnr_linear": base_psnr_lin,
            "pred_psnr_linear": pred_psnr_lin,
            "delta_psnr_linear": pred_psnr_lin - base_psnr_lin,
            "baseline_l1_linear": base_l1_lin,
            "pred_l1_linear": pred_l1_lin,
            "cycle_and_bin": cyc,
            "banding_stress": band,
        },
        "output_path": out_path,
    }

# -------------------------
# Main
# -------------------------

def calculate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics from current results."""
    def agg_mean(key_path: List[str], only_types: Optional[List[str]] = None) -> Optional[float]:
        vals = []
        for rec in results["images"]:
            if "metrics" not in rec:
                continue
            if only_types and rec.get("type") not in only_types:
                continue
            d = rec["metrics"]
            dd = d
            for kp in key_path:
                if kp in dd:
                    dd = dd[kp]
                else:
                    dd = None
                    break
            if isinstance(dd, (float, int)):
                vals.append(float(dd))
        if not vals:
            return None
        return float(np.mean(vals))

    summary: Dict[str, Any] = {
        "count_total": len(results["images"]),
        "count_8bit": sum(1 for r in results["images"] if r.get("type") == "8bit"),
        "count_16bit_or_raw": sum(1 for r in results["images"] if r.get("type") in ("png16", "raw16")),
        # Clean GT comparisons
        "mean_delta_psnr_srgb_over_baseline_on_16bit": agg_mean(["delta_psnr_srgb"], only_types=["png16", "raw16"]),
        "mean_delta_psnr_linear_over_baseline_on_16bit": agg_mean(["delta_psnr_linear"], only_types=["png16", "raw16"]),
        # Cycle/bin health on 8-bit
        "mean_requant_match_rate_pixel_on_8bit": agg_mean(["requant_match_rate_pixel"], only_types=["8bit"]),
        "mean_bin_adherence_0p5lsb_on_8bit": agg_mean(["bin_adherence_within_0p5lsb"], only_types=["8bit"]),
    }
    return summary

def save_results(results: Dict[str, Any], output_json: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def main():
    # Load model
    device = torch.device(DEVICE if DEVICE in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model_from_checkpoint(CHECKPOINT_PATH, device=device, model_class=model_class)
    model.eval()

    if SAVE_OUTPUTS:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Collect files
    files = list_images_in_dirs(INPUT_DIRS, INPUT_BASENAMES_FILE)
    if not files:
        print("No images found.")
        sys.exit(1)

    print(f"Found {len(files)} images.")

    results = {
        "checkpoint": os.path.abspath(CHECKPOINT_PATH),
        "datetime": datetime.utcnow().isoformat() + "Z",
        "model_config": cfg,
        "device": str(device),
        "images": [],
        "summary": {},
    }

    # Save initial empty results
    save_results(results, OUTPUT_JSON)
    print(f"Initialized results file: {os.path.abspath(OUTPUT_JSON)}")

    # Evaluate with progress bar
    for i, path in enumerate(tqdm(files, desc="Evaluating images")):
        try:
            # Use your color-info script to decide type/bit-depth
            li = bci.load_image(path)
            arr = li.arr
            mode = li.mode
            dtype = arr.dtype
            # Decide path type
            if is_raw_file(path):
                kind = "raw16"
            else:
                # Check actual value range, not just dtype (Pillow may load 16-bit as uint8)
                max_val = float(arr.max())
                per_channel_bits = 8 if arr.dtype == np.uint8 else (16 if arr.dtype == np.uint16 else None)
                
                if per_channel_bits == 16 or max_val > 255:
                    kind = "png16"
                elif per_channel_bits == 8 and max_val <= 255:
                    kind = "8bit"
                else:
                    # Fallback: if max>255 treat as 16-bit, else 8-bit
                    kind = "png16" if max_val > 255 else "8bit"

            print(f"{os.path.basename(path)}  →  {kind}")

            if kind == "8bit":
                rec = eval_on_8bit(
                    path, model, cfg, device, PATCH_SIZE, STRIDE,
                    MAX_BATCH, SAVE_OUTPUTS, OUTPUTS_DIR
                )
            else:
                rec = eval_on_16bit_or_raw(
                    path, model, cfg, device, PATCH_SIZE, STRIDE,
                    MAX_BATCH, SAVE_OUTPUTS, OUTPUTS_DIR
                )

            rec.update({
                "path": os.path.abspath(path),
                "mode": mode,
                "dtype": str(dtype),
                "format": li.fmt,
                "note": li.note,
            })
            results["images"].append(rec)
            
            # Update summary and save after each image
            results["summary"] = calculate_summary(results)
            save_results(results, OUTPUT_JSON)
            
            print(f"  ✓ Processed {i+1}/{len(files)} - Updated {os.path.abspath(OUTPUT_JSON)}")
            
        except Exception as e:
            print(f"  ! Skipping {path} due to error: {e}")
            results["images"].append({
                "path": os.path.abspath(path),
                "error": str(e),
            })
            
            # Update summary and save even for failed images
            results["summary"] = calculate_summary(results)
            save_results(results, OUTPUT_JSON)
            
            print(f"  ✓ Error logged for {i+1}/{len(files)} - Updated {os.path.abspath(OUTPUT_JSON)}")

    print(f"\nFinal results saved to {os.path.abspath(OUTPUT_JSON)}")
    if SAVE_OUTPUTS:
        print(f"Saved dequantized outputs to: {os.path.abspath(OUTPUTS_DIR)}")

if __name__ == "__main__":
    main()
