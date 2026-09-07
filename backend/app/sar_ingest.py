"""
SAR (Sentinel-1 style VV/VH) band ingestion.

Mirrors the optical band-ingestion pattern already established in
`vqa_and_change_using_gemini/pipeline.py::load_optical_bands_from_uploads`
(in-memory bytes in, band-ordered (C, H, W) float32 array out, resampled
onto a caller-supplied grid) so both modalities are ingested the same way.
No SAR loader existed in the repo before this — `UnifiedChangeDetectionNet`
expects `sar_in_ch=2` (VV, VH), which this produces.
"""
import re

import numpy as np
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

from app.exceptions import ValidationFailed

SAR_BAND_ORDER = ["VV", "VH"]


def _sar_band_key_from_filename(fname: str) -> str | None:
    m = re.search(r"(?:^|[_\-.])([Vv][VvHh])(?:[_\-.]|$)", fname)
    return m.group(1).upper() if m else None


def load_sar_bands_from_uploads(
    files: dict[str, bytes], target_shape: tuple[int, int] | None = None
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    files: dict[filename -> bytes], the VV/VH GeoTIFFs for one timestep.
    Returns ((2, H, W) float32 array, (H, W)).
    """
    bands_by_code: dict[str, bytes] = {}
    for fname, content in files.items():
        code = _sar_band_key_from_filename(fname)
        if code is not None:
            bands_by_code[code] = content

    missing = [b for b in SAR_BAND_ORDER if b not in bands_by_code]
    if missing:
        raise ValidationFailed(
            f"Missing SAR bands in upload: {missing}. Found: {sorted(bands_by_code.keys())}",
            details={"missing_bands": missing, "found_bands": sorted(bands_by_code.keys())},
        )

    if target_shape is None:
        with MemoryFile(bands_by_code["VV"]) as mem, mem.open() as ref:
            target_shape = (ref.height, ref.width)

    target_h, target_w = target_shape
    arrays = []
    for band in SAR_BAND_ORDER:
        with MemoryFile(bands_by_code[band]) as mem, mem.open() as src:
            img = src.read(
                1, out_shape=(target_h, target_w), resampling=Resampling.bilinear
            ).astype(np.float32)
        arrays.append(img)

    sar = np.stack(arrays, axis=0)
    if sar.shape != (2, target_h, target_w):
        raise ValidationFailed(f"Unexpected SAR array shape: {sar.shape}")

    return sar, (target_h, target_w)


def sar_to_grayscale_rgb(sar_array: np.ndarray) -> np.ndarray:
    """VV/VH backscatter -> a displayable (H, W, 3) uint8 preview (1-99 pct stretch)."""
    vv = sar_array[0]
    lo, hi = np.percentile(vv, [1, 99])
    norm = np.clip((vv - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)
