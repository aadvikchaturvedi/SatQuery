"""
Upload validation and parsing: turns raw multipart UploadFiles into a typed
InputBundle. This is where "check the number, modality, format, metadata,
and compatibility of the input images" (product spec, agentic controller
requirements) actually happens, before the agent ever sees the data.
"""
from __future__ import annotations

import os

import numpy as np
from fastapi import UploadFile
from PIL import Image

from app import ml_paths  # noqa: F401  (sys.path side effect, must run first)
from app.config import get_settings
from app.exceptions import ValidationFailed
from app.sar_ingest import load_sar_bands_from_uploads, sar_to_grayscale_rgb
from app.types import InputBundle, Modality, SceneImage, SourceKind
from vqa_and_change_using_gemini.imaging import optical_to_rgb
from vqa_and_change_using_gemini.pipeline import load_optical_bands_from_uploads

GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_EXTENSIONS = GEOTIFF_EXTENSIONS | IMAGE_EXTENSIONS


async def _read_group(files: list[UploadFile] | None, group_name: str) -> dict[str, bytes]:
    if not files:
        return {}

    settings = get_settings()
    if len(files) > settings.max_files_per_request:
        raise ValidationFailed(
            f"{group_name}: too many files ({len(files)} > {settings.max_files_per_request})."
        )

    out: dict[str, bytes] = {}
    for f in files:
        name = f.filename or "upload"
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationFailed(
                f"{group_name}: unsupported file extension '{ext}' for '{name}'. "
                f"Supported: {sorted(ALLOWED_EXTENSIONS)}",
                details={"field": group_name, "filename": name},
            )
        content = await f.read()
        if len(content) > settings.max_upload_bytes:
            raise ValidationFailed(
                f"{group_name}: '{name}' exceeds the {settings.max_upload_bytes} byte upload limit.",
                details={"field": group_name, "filename": name},
            )
        if len(content) == 0:
            raise ValidationFailed(f"{group_name}: '{name}' is empty.")
        out[name] = content
    return out


def _build_scene_image(
    modality: Modality,
    files_bytes: dict[str, bytes],
    group_name: str,
    target_shape: tuple[int, int] | None = None,
) -> SceneImage:
    if not files_bytes:
        raise ValidationFailed(f"{group_name}: no files provided.")

    # A single PNG/JPEG upload is treated as a pre-rendered image (the
    # product spec permits this only for benchmark-dataset inputs, since
    # those don't ship as multi-band GeoTIFFs).
    if len(files_bytes) == 1:
        (fname, content), = files_bytes.items()
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            import io

            img = Image.open(io.BytesIO(content)).convert("RGB")
            arr = np.array(img)
            return SceneImage(
                modality=modality,
                source_kind=SourceKind.PNG_JPEG,
                hw=(arr.shape[0], arr.shape[1]),
                array=arr,
                rgb_preview=arr,
            )

    if modality == Modality.OPTICAL:
        try:
            optical = load_optical_bands_from_uploads(files_bytes, target_shape=target_shape)
        except (ValueError, RuntimeError) as exc:
            # load_optical_bands_from_uploads (reused, untouched ML code) raises
            # plain ValueError/RuntimeError, not a SatQueryError subclass — map
            # it here so a bad upload is a 422, not an opaque 500.
            raise ValidationFailed(str(exc)) from exc
        hw = (optical.shape[-2], optical.shape[-1])
        return SceneImage(
            modality=modality,
            source_kind=SourceKind.GEOTIFF_BANDS,
            hw=hw,
            array=optical,
            rgb_preview=optical_to_rgb(optical),
        )

    sar, hw = load_sar_bands_from_uploads(files_bytes, target_shape=target_shape)
    return SceneImage(
        modality=modality,
        source_kind=SourceKind.GEOTIFF_BANDS,
        hw=hw,
        array=sar,
        rgb_preview=sar_to_grayscale_rgb(sar),
    )


async def build_input_bundle(
    query: str,
    optical_t1_files: list[UploadFile] | None,
    optical_t2_files: list[UploadFile] | None,
    sar_t1_files: list[UploadFile] | None,
    sar_t2_files: list[UploadFile] | None,
) -> InputBundle:
    if not query or not query.strip():
        raise ValidationFailed("Query must be a non-empty string.")

    opt_t1_bytes = await _read_group(optical_t1_files, "optical_t1_files")
    opt_t2_bytes = await _read_group(optical_t2_files, "optical_t2_files")
    sar_t1_bytes = await _read_group(sar_t1_files, "sar_t1_files")
    sar_t2_bytes = await _read_group(sar_t2_files, "sar_t2_files")

    if not (opt_t1_bytes or sar_t1_bytes):
        raise ValidationFailed(
            "At least one image is required (optical_t1_files or sar_t1_files)."
        )

    optical_t1 = _build_scene_image(Modality.OPTICAL, opt_t1_bytes, "optical_t1_files") if opt_t1_bytes else None
    optical_t2 = (
        _build_scene_image(
            Modality.OPTICAL, opt_t2_bytes, "optical_t2_files",
            target_shape=optical_t1.hw if optical_t1 else None,
        )
        if opt_t2_bytes
        else None
    )
    sar_t1 = _build_scene_image(Modality.SAR, sar_t1_bytes, "sar_t1_files") if sar_t1_bytes else None
    sar_t2 = (
        _build_scene_image(
            Modality.SAR, sar_t2_bytes, "sar_t2_files",
            target_shape=sar_t1.hw if sar_t1 else None,
        )
        if sar_t2_bytes
        else None
    )

    # Cross-modal compatibility: the change-detection model runs optical and
    # SAR patches through the same (y, x) crop coordinates in fusion mode, so
    # co-registered inputs must share a pixel grid.
    if optical_t1 and sar_t1 and optical_t1.hw != sar_t1.hw:
        raise ValidationFailed(
            "optical_t1 and sar_t1 must be co-registered to the same pixel grid "
            f"(got {optical_t1.hw} vs {sar_t1.hw}). Resample SAR/optical to match before upload.",
            details={"optical_hw": list(optical_t1.hw), "sar_hw": list(sar_t1.hw)},
        )
    if optical_t2 and sar_t2 and optical_t2.hw != sar_t2.hw:
        raise ValidationFailed(
            "optical_t2 and sar_t2 must be co-registered to the same pixel grid "
            f"(got {optical_t2.hw} vs {sar_t2.hw}).",
            details={"optical_hw": list(optical_t2.hw), "sar_hw": list(sar_t2.hw)},
        )

    if optical_t2 and not optical_t1:
        raise ValidationFailed("optical_t2_files provided without optical_t1_files.")
    if sar_t2 and not sar_t1:
        raise ValidationFailed("sar_t2_files provided without sar_t1_files.")

    return InputBundle(
        query=query.strip(),
        optical_t1=optical_t1,
        optical_t2=optical_t2,
        sar_t1=sar_t1,
        sar_t2=sar_t2,
    )
