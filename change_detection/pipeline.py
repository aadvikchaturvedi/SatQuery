"""
pipeline.py

Core change-detection pipeline, decoupled from any web framework.

Design:

    A frontend uploads T1 band files + T2 band files (in-memory
    bytes, e.g. from FastAPI's UploadFile.read()). This module never
    touches the filesystem or a display — it takes bytes in and
    returns numbers + images (as PIL Images / PNG bytes) out.

    The actual HTTP layer (api.py) is a thin wrapper around
    `run_change_detection()`.
"""

import re

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

from imaging import (
    optical_to_rgb,
    numpy_to_pil,
    confidence_to_pil,
    create_change_overlay,
    pil_to_png_bytes,
)

BAND_ORDER = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "8A", "09", "10", "11", "12",
]


# ============================================================
# READING UPLOADED BANDS (bytes, not disk paths)
# ============================================================

def _band_key_from_filename(fname):
    """Extract the Sentinel-2 band code from an uploaded filename."""

    m = re.search(
        r"(?:^|[_\-])B(0[1-9]|8A|1[0-2])\.tif$",
        fname,
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else None


def load_optical_bands_from_uploads(files, target_shape=None):
    """
    files: dict[filename -> bytes] for one timestep's 13 band files,
           exactly as received from the frontend upload.

    Returns a (13, H, W) float32 numpy array, band-ordered and
    resampled onto a common grid, entirely in memory.
    """

    bands_by_code = {}
    for fname, content in files.items():
        code = _band_key_from_filename(fname)
        if code is not None:
            bands_by_code[code] = content

    missing = [b for b in BAND_ORDER if b not in bands_by_code]
    if missing:
        raise ValueError(
            f"Missing Sentinel-2 bands in upload: {missing}. "
            f"Found: {sorted(bands_by_code.keys())}"
        )

    if target_shape is None:
        with MemoryFile(bands_by_code["02"]) as mem:
            with mem.open() as ref:
                target_shape = (ref.height, ref.width)

    target_h, target_w = target_shape
    arrays = []

    for band in BAND_ORDER:
        with MemoryFile(bands_by_code[band]) as mem:
            with mem.open() as src:
                img = src.read(
                    1,
                    out_shape=(target_h, target_w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
        arrays.append(img)

    optical = np.stack(arrays, axis=0)

    if optical.shape != (13, target_h, target_w):
        raise RuntimeError(f"Unexpected optical shape: {optical.shape}")

    return optical


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_change_detection(t1_files, t2_files, detector, region_id="uploaded_region"):
    """
    t1_files, t2_files: dict[filename -> bytes] — the raw uploads for
                         each timestep, straight from the frontend.
    detector:           an already-loaded model
                         (e.g. change_detection_module.load_change_detector(...))

    Returns a dict ready to be JSON-serialized / sent back to the
    frontend:

        {
            "region": ...,
            "model_mode": ...,
            "change_percentage": float,
            "regions": [ ... ],
            "images": {
                "t2_rgb_png": bytes,
                "confidence_png": bytes,
                "overlay_png": bytes,
            },
        }

    Nothing here reads or writes disk, and nothing here plots to a
    screen — every image comes back as PNG bytes the caller can
    stream straight into an HTTP response.
    """

    optical_t1 = load_optical_bands_from_uploads(t1_files)
    optical_t2 = load_optical_bands_from_uploads(
        t2_files, target_shape=optical_t1.shape[-2:]
    )

    if optical_t1.shape != optical_t2.shape:
        raise RuntimeError(
            f"T1/T2 shape mismatch: {optical_t1.shape} vs {optical_t2.shape}"
        )

    result = detector.predict(
        optical_t1=optical_t1,
        optical_t2=optical_t2,
        sar_t1=None,
        sar_t2=None,
        hw=optical_t1.shape[-2:],
        region_id=region_id,
    )

    # ---- build response images, all as PNG bytes -----------------
    t2_rgb_img = numpy_to_pil(optical_to_rgb(optical_t2))
    confidence_img = confidence_to_pil(result["confidence_map"])
    overlay_img = create_change_overlay(optical_t2, result)

    return {
        "region": result["region"],
        "model_mode": result["model_mode"],
        "change_percentage": float(result["change_percentage"]),
        "regions": result["regions"],
        # Keep the raw arrays around too, in case the caller wants to
        # immediately pass them to something like ask_change_vqa()
        # without re-decoding the PNGs.
        "_optical_t1": optical_t1,
        "_optical_t2": optical_t2,
        "_result": result,
        "images": {
            "t2_rgb_png": pil_to_png_bytes(t2_rgb_img),
            "confidence_png": pil_to_png_bytes(confidence_img),
            "overlay_png": pil_to_png_bytes(overlay_img),
        },
    }
