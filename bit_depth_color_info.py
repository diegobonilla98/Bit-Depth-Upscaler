import os
from dataclasses import dataclass
from io import BytesIO
import numpy as np
from PIL import Image, ExifTags, ImageCms
import rawpy

RAW_EXTS = {".nef", ".dng", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".raf", ".srw", ".pef", ".nrw"}

# ----------------------------
# Helpers
# ----------------------------

def pil_mode_channel_names(mode: str):
    mapping = {
        "RGB": ["R", "G", "B"],
        "RGBA": ["R", "G", "B", "A"],
        "L": ["L"],
        "CMYK": ["C", "M", "Y", "K"],
    }
    return mapping.get(mode, [f"C{i}" for i in range(len(mode))])

def dtype_bit_depth(dtype: np.dtype):
    if np.issubdtype(dtype, np.integer):
        return dtype.itemsize * 8
    return None

def estimate_windows_style_bitdepth(mode: str, per_channel_bits: int):
    mu = mode.upper()
    if mu == "RGB": return 3 * per_channel_bits
    if mu == "RGBA": return 3 * per_channel_bits
    if mu == "L": return per_channel_bits
    if mu == "CMYK": return 4 * per_channel_bits
    return None

def exif_color_space_str(img: Image.Image):
    exif = img.getexif()
    if not exif: return None
    tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    cs = tag_map.get("ColorSpace")
    if cs == 1: return "sRGB"
    if cs == 65535: return "Uncalibrated"
    return f"ColorSpace={cs}" if cs is not None else None

def icc_profile_description(img: Image.Image):
    icc_bytes = img.info.get("icc_profile")
    if not icc_bytes: return None
    prof = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
    return ImageCms.getProfileName(prof)

def channel_stats(arr: np.ndarray):
    vmin, vmax = int(np.min(arr)), int(np.max(arr))
    mean, std = float(np.mean(arr)), float(np.std(arr))
    levels = int(np.unique(arr).size)
    hist, _ = np.histogram(arr, bins=min(65536, vmax - vmin + 1), range=(vmin, vmax + 1))
    p = hist.astype(np.float64)
    p /= (p.sum() + 1e-9)
    entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return vmin, vmax, mean, std, levels, entropy

def approx_unique_colors(arr: np.ndarray, sample: int = 2):
    if arr.ndim == 2:
        return int(np.unique(arr[::sample, ::sample]).size)
    sub = arr[::sample, ::sample, :3]
    if arr.dtype != np.uint8:
        sub = (255 * (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)).astype(np.uint8)
    codes = (sub[...,0].astype(np.uint32)<<16) | (sub[...,1].astype(np.uint32)<<8) | sub[...,2].astype(np.uint32)
    return int(np.unique(codes).size)

def is_raw_file(path: str):
    return os.path.splitext(path)[1].lower() in RAW_EXTS

# ----------------------------
# Loader
# ----------------------------

@dataclass
class LoadedImage:
    arr: np.ndarray
    mode: str
    fmt: str
    width: int
    height: int
    pil_image: Image.Image or None
    note: str

def load_image(path: str) -> LoadedImage:
    if is_raw_file(path):
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(
                output_bps=16,
                use_camera_wb=True,
                no_auto_bright=True,
                gamma=(1,1),
                output_color=rawpy.ColorSpace.sRGB
            )
        h, w, _ = rgb16.shape
        return LoadedImage(rgb16, "RGB", "RAW(demosaiced)", w, h, None,
                           "Decoded from RAW with rawpy (16 bits/channel RGB).")
    else:
        img = Image.open(path)
        arr = np.array(img)
        return LoadedImage(arr, img.mode, getattr(img, "format", "Unknown"),
                           img.size[0], img.size[1], img, "Loaded with Pillow.")

# ----------------------------
# Report
# ----------------------------

@dataclass
class ReportConfig:
    sample_stride: int = 1

def print_report(path: str, cfg: ReportConfig):
    li = load_image(path)
    arr, mode, fmt = li.arr, li.mode, li.fmt
    width, height, dtype = li.width, li.height, arr.dtype
    per_channel_bits = dtype_bit_depth(dtype)
    win_bits = estimate_windows_style_bitdepth(mode, per_channel_bits)

    print("="*72)
    print(f"Image: {path}")
    print(f"Format: {fmt}   Size: {width}x{height}   Mode: {mode}")
    print(f"Array shape: {arr.shape}   Dtype: {dtype}")
    print(f"Note: {li.note}")
    print("-"*72)

    if li.pil_image is not None:
        print(f"EXIF ColorSpace: {exif_color_space_str(li.pil_image) or 'None'}")
        print(f"ICC Profile: {icc_profile_description(li.pil_image) or 'None'}")

    print("Bit depth:")
    print(f"  Per-channel: {per_channel_bits} bits")
    print(f"  Windows-style: {win_bits}-bit total" if win_bits else "  N/A")

    print("-"*72)
    ch_names = pil_mode_channel_names(mode)
    if arr.ndim == 3:
        for i, cname in enumerate(ch_names[:arr.shape[2]]):
            vmin,vmax,mean,std,levels,entropy = channel_stats(arr[...,i])
            print(f"  [{cname}] min={vmin} max={vmax} mean={mean:.2f} std={std:.2f} "
                  f"levels={levels} entropy={entropy:.2f}")
    elif arr.ndim == 2:
        vmin,vmax,mean,std,levels,entropy = channel_stats(arr)
        print(f"  [L] min={vmin} max={vmax} mean={mean:.2f} std={std:.2f} "
              f"levels={levels} entropy={entropy:.2f}")

    uniq = approx_unique_colors(arr, sample=cfg.sample_stride)
    print(f"Approx unique colors: {uniq}")
    print("-"*72)

    # ---- Summary ----
    print("Summary:")
    if mode in ("RGB","RGBA"):
        if per_channel_bits == 8:
            print("  → Standard 8 bpc RGB (Windows: 24-bit).")
        elif per_channel_bits == 16:
            print("  → High depth 16 bpc RGB (Windows: 48-bit).")
    elif mode == "L":
        print(f"  → Grayscale {per_channel_bits} bpc.")
    else:
        print("  → Other mode.")
    print("="*72)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # <<< Set your file path here >>>
    image_path = r"F:\RAISE_TIFF\RAISE_6k.csv\NEF\r1ceba29dt.NEF"

    cfg = ReportConfig(sample_stride=2)
    print_report(image_path, cfg)
