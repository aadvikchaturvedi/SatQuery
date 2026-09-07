"""Shared internal (non-API) data structures passed between ingest -> agent -> tools."""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"


class SourceKind(str, Enum):
    GEOTIFF_BANDS = "geotiff_bands"
    PNG_JPEG = "png_jpeg"


@dataclass
class SceneImage:
    modality: Modality
    source_kind: SourceKind
    hw: tuple[int, int]
    # Raw model-ready array: (13, H, W) float32 for optical GeoTIFF bands,
    # (2, H, W) float32 for SAR GeoTIFF bands, or (H, W, 3) uint8 for PNG/JPEG.
    array: np.ndarray
    # Always an (H, W, 3) uint8 array, suitable to hand to a VLM or render as PNG.
    rgb_preview: np.ndarray


@dataclass
class InputBundle:
    query: str
    optical_t1: SceneImage | None = None
    optical_t2: SceneImage | None = None
    sar_t1: SceneImage | None = None
    sar_t2: SceneImage | None = None

    @property
    def has_optical_pair(self) -> bool:
        return self.optical_t1 is not None and self.optical_t2 is not None

    @property
    def has_sar_pair(self) -> bool:
        return self.sar_t1 is not None and self.sar_t2 is not None

    @property
    def has_bitemporal(self) -> bool:
        return self.has_optical_pair or self.has_sar_pair

    @property
    def single_image(self) -> SceneImage | None:
        """The one image to use for single-image tasks (VQA/captioning), if unambiguous."""
        candidates = [
            img for img in (self.optical_t1, self.sar_t1) if img is not None
        ]
        if len(candidates) == 1 and self.optical_t2 is None and self.sar_t2 is None:
            return candidates[0]
        return None

    @property
    def is_single_time_cross_modal(self) -> bool:
        return (
            self.optical_t1 is not None
            and self.sar_t1 is not None
            and self.optical_t2 is None
            and self.sar_t2 is None
        )
