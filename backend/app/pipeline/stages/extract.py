# backend/app/pipeline/stages/extract.py
"""
Layout-driven part extraction.

Reads:
  - flattened saree image (S_flat) - any common raster format
  - layout JSON (relative coordinates 0..1): each region defined as [x0, y0, x1, y1]

Produces (per input saree/generation):
  output_dir/parts/{region_name}.png
  output_dir/parts/parts.json  # metadata for each part

Notes:
- Coordinates in layout JSON are normalized relative bbox in format [x0, y0, x1, y1].
- All output images preserve original resolution (no resizing).
- Deterministic: same inputs → same outputs.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage import color, feature

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class RegionMeta:
    name: str
    bbox_rel: Tuple[float, float, float, float]
    bbox_px: Tuple[int, int, int, int]
    width: int
    height: int
    mean_lab: Tuple[float, float, float]
    median_lab: Tuple[float, float, float]
    lbp_hist: List[int]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def resolve_relative_bbox(
    rel_bbox: Tuple[float, float, float, float], image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Convert normalized bbox [x0,y0,x1,y1] to pixel bbox (left,top,right,bottom).
    Ensures coordinates are clamped within image.
    """
    w, h = image_size
    x0, y0, x1, y1 = rel_bbox
    x0 = _clamp(x0)
    y0 = _clamp(y0)
    x1 = _clamp(x1)
    y1 = _clamp(y1)
    left = int(round(x0 * w))
    top = int(round(y0 * h))
    right = int(round(x1 * w))
    bottom = int(round(y1 * h))
    # Ensure minimum size of 1 pixel
    if right <= left:
        right = min(w, left + 1)
    if bottom <= top:
        bottom = min(h, top + 1)
    return left, top, right, bottom


def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def compute_lab_stats(rgb_arr: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute mean and median in CIE-LAB color space.
    Input: HxWx3 uint8 RGB numpy array.
    Output: (mean_L, mean_a, mean_b), (median_L, median_a, median_b)
    """
    # Normalize to 0..1 for skimage
    rgb_norm = rgb_arr.astype("float32") / 255.0
    lab = color.rgb2lab(rgb_norm)  # returns float array
    mean = tuple(float(np.mean(lab[..., i])) for i in range(3))
    median = tuple(float(np.median(lab[..., i])) for i in range(3))
    return mean, median


def compute_lbp_histogram(gray_arr: np.ndarray, P: int = 8, R: int = 1, bins: int = 256) -> List[int]:
    """
    Compute Local Binary Pattern histogram as a simple texture descriptor.
    Returns histogram list (length `bins`).
    """
    # LBP expects a 2D grayscale image (float)
    lbp = feature.local_binary_pattern(gray_arr.astype("float32"), P=P, R=R, method="uniform")
    # Clip negative or large values then histogram
    lbp = lbp.astype("int32")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    return hist.tolist()


def analyze_region(img_region: Image.Image) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], List[int]]:
    """
    Given a PIL Image region (any mode), compute mean/median Lab and LBP histogram.
    """
    rgb = pil_to_rgb_array(img_region)
    mean_lab, median_lab = compute_lab_stats(rgb)
    # convert to grayscale for LBP
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])  # luminance
    lbp_hist = compute_lbp_histogram(gray, P=8, R=1, bins=256)
    return mean_lab, median_lab, lbp_hist


def extract_parts(s_flat_path: Path, layout_json: Path, output_dir: Path) -> Dict[str, RegionMeta]:
    """
    Perform deterministic extraction using the layout JSON and save crops and metadata.
    Returns mapping region_name -> RegionMeta
    """
    s_flat = Image.open(s_flat_path)
    image_size = s_flat.size  # (width, height)
    logger.info("Opened S_flat %s size=%s", s_flat_path, image_size)

    with open(layout_json, "r", encoding="utf-8") as fh:
        layout = json.load(fh)

    regions = layout.get("regions", {})
    if not regions:
        raise ValueError("No regions found in layout JSON: %s" % layout_json)

    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    metas: Dict[str, RegionMeta] = {}
    for name, spec in regions.items():
        bbox_rel = tuple(spec.get("relative_bbox"))
        if len(bbox_rel) != 4:
            raise ValueError(f"region {name} has invalid relative_bbox: {bbox_rel}")

        left, top, right, bottom = resolve_relative_bbox(bbox_rel, image_size)
        crop = s_flat.crop((left, top, right, bottom))
        out_name = f"{name}.png"
        out_path = parts_dir / out_name
        # Save PNG with same mode if possible; force RGBA if input had alpha
        save_kwargs = {}
        if s_flat.mode == "RGBA":
            crop = crop.convert("RGBA")
        else:
            crop = crop.convert("RGBA")
        crop.save(out_path, format="PNG", **save_kwargs)
        logger.info("Saved part %s -> %s (bbox_px=%s)", name, out_path, (left, top, right, bottom))

        mean_lab, median_lab, lbp_hist = analyze_region(crop)
        width = right - left
        height = bottom - top
        meta = RegionMeta(
            name=name,
            bbox_rel=bbox_rel,
            bbox_px=(left, top, right, bottom),
            width=width,
            height=height,
            mean_lab=tuple(round(v, 4) for v in mean_lab),
            median_lab=tuple(round(v, 4) for v in median_lab),
            lbp_hist=[int(x) for x in lbp_hist],
        )
        metas[name] = meta

    # Write parts.json (summary)
    parts_json = parts_dir / "parts.json"
    out = {
        "source_image": str(s_flat_path),
        "layout_used": str(layout_json),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "regions": {
            name: {
                "bbox_rel": metas[name].bbox_rel,
                "bbox_px": metas[name].bbox_px,
                "width": metas[name].width,
                "height": metas[name].height,
                "mean_lab": metas[name].mean_lab,
                "median_lab": metas[name].median_lab,
                # store only downsampled histogram summary (first 64 bins) to avoid huge JSONs
                "lbp_hist_sample": metas[name].lbp_hist[:64],
            }
            for name in metas
        },
    }
    with open(parts_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Wrote parts metadata to %s", parts_json)

    return metas


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract saree parts using authoritative layout JSON.")
    parser.add_argument("s_flat", type=Path, help="Path to flattened saree image (S_flat.png)")
    parser.add_argument("layout_json", type=Path, help="Path to layout JSON (relative normalized coordinates)")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path.cwd() / "output", help="Directory to write parts/ and parts.json"
    )
    args = parser.parse_args()

    if not args.s_flat.exists():
        raise FileNotFoundError(f"S_flat not found: {args.s_flat}")
    if not args.layout_json.exists():
        raise FileNotFoundError(f"Layout JSON not found: {args.layout_json}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extract_parts(args.s_flat, args.layout_json, args.output_dir)


if __name__ == "__main__":
    main()
