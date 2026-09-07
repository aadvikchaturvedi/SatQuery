"""
Bi-temporal change understanding + change-VQA tool.

Wraps the already-trained `unified_changenet_best.pt` checkpoint via the
existing, untouched `change_detection_module.py`. That model natively
supports three modes (see UnifiedChangeDetectionNet.forward): "optical",
"sar", and "fusion" — fusion is exactly the product spec's mandatory
"cross-modal pair analysis" requirement (optical+SAR jointly), and it
already exists in the trained weights, so a bi-temporal optical+SAR
request satisfies both the change-analysis and cross-modal requirements
through one real model call, with no new ML work.
"""
from __future__ import annotations

import os
import threading
import time

from app import ml_paths  # noqa: F401
from app.config import get_settings
from app.exceptions import UpstreamModelError, ValidationFailed
from app.tools.base import Tool, ToolResult
from app.tools.render import render_overlay
from app.types import InputBundle
from vqa_and_change_using_gemini.change_detection_module import ChangeDetector, load_change_detector
from vqa_and_change_using_gemini.imaging import confidence_to_pil, pil_to_png_bytes

_detector: ChangeDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> ChangeDetector:
    """Lazily loads the checkpoint once per process (torch model load is not free)."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                settings = get_settings()
                if not os.path.isfile(settings.change_checkpoint_path):
                    raise UpstreamModelError(
                        f"Change-detection checkpoint not found at "
                        f"{settings.change_checkpoint_path}."
                    )
                _detector = load_change_detector(
                    settings.change_checkpoint_path, device=settings.change_device
                )
    return _detector


def _ask_gemini_about_change(question: str, bundle: InputBundle, result: dict) -> str:
    """
    Prefers the existing, untouched Gemini change-VQA prompt (which expects
    13-band optical arrays) whenever optical imagery is available. Falls
    back to a modality-agnostic prompt, built here, for the SAR-only case
    that `vqa_and_change_using_gemini/vqa.py` does not cover.
    """
    settings = get_settings()
    if bundle.optical_t1 is not None and bundle.optical_t2 is not None:
        os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key or "")
        from vqa_and_change_using_gemini.vqa import ask_change_vqa  # lazy: module-level API key check

        # ask_change_vqa() calls the Gemini SDK directly, bypassing
        # gemini_client.generate_content_with_retry's retry/error-mapping.
        # Retry and re-map here instead, so a transient upstream 5xx (seen
        # directly while testing this integration) surfaces as a proper 502
        # rather than an opaque 500.
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return ask_change_vqa(question, bundle.optical_t1.array, bundle.optical_t2.array, result)
            except Exception as exc:  # noqa: BLE001 - genai raises its own exception types
                last_error = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                is_server_error = status is not None and 500 <= int(status) < 600
                if not is_server_error or attempt == 3:
                    break
                time.sleep(2.0 * attempt)
        raise UpstreamModelError(f"Gemini change-VQA request failed after 3 attempt(s): {last_error}")

    from vqa_and_change_using_gemini.vqa import create_region_summary  # pure text formatting, modality-agnostic

    from app.tools.gemini_client import generate_content_with_retry
    from PIL import Image

    t2_preview = Image.fromarray(bundle.sar_t2.rgb_preview) if bundle.sar_t2 else None
    confidence_img = confidence_to_pil(result["confidence_map"])
    overlay_img = render_overlay(bundle.sar_t2.rgb_preview if bundle.sar_t2 else bundle.sar_t1.rgb_preview, result)

    prompt = (
        "You are a remote-sensing change-analysis assistant. You are given SAR "
        "(VV/VH backscatter) imagery from a change-detection pipeline: a "
        "change-confidence map and a mask/bounding-box overlay.\n\n"
        f"Region: {result['region']}\nModel mode: {result['model_mode']}\n"
        f"Total changed area: {result['change_percentage']:.2f}%\n"
        f"Detected regions:\n{create_region_summary(result)}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Answer concisely, distinguishing model-detected change from any "
        "inferred cause, and say 'uncertain' where the SAR evidence alone is "
        "not conclusive."
    )
    contents = [prompt]
    if t2_preview:
        contents.append(t2_preview)
    contents.extend([confidence_img, overlay_img])
    return generate_content_with_retry(model=settings.gemini_text_model, contents=contents)


class ChangeVqaTool(Tool):
    name = "change_vqa"
    description = (
        "Bi-temporal change detection (unified optical/SAR/fusion ChangeNet) "
        "combined with Gemini-based natural-language change-VQA."
    )
    requires = ["optical_t1+optical_t2 and/or sar_t1+sar_t2"]
    domain_adapted = True  # ChangeNet is fine-tuned on remote-sensing change pairs

    def run(self, bundle: InputBundle) -> ToolResult:
        if not bundle.has_bitemporal:
            raise ValidationFailed("change_vqa requires a bi-temporal pair (T1 + T2).")

        detector = get_detector()
        hw = (bundle.optical_t1 or bundle.sar_t1).hw

        result = detector.predict(
            optical_t1=bundle.optical_t1.array if bundle.optical_t1 else None,
            optical_t2=bundle.optical_t2.array if bundle.optical_t2 else None,
            sar_t1=bundle.sar_t1.array if bundle.sar_t1 else None,
            sar_t2=bundle.sar_t2.array if bundle.sar_t2 else None,
            hw=hw,
            region_id="uploaded_region",
        )

        t2_preview = (bundle.optical_t2 or bundle.sar_t2).rgb_preview
        images = {
            "t2_preview.png": pil_to_png_bytes(_rgb_array_to_pil(t2_preview)),
            "confidence.png": pil_to_png_bytes(confidence_to_pil(result["confidence_map"])),
            "overlay.png": pil_to_png_bytes(render_overlay(t2_preview, result)),
        }

        warnings = []
        answer = None
        settings = get_settings()
        if not settings.gemini_api_key:
            warnings.append(
                "GEMINI_API_KEY not configured: returning quantitative change "
                "results only, without a natural-language answer."
            )
        else:
            try:
                answer = _ask_gemini_about_change(bundle.query, bundle, result)
            except UpstreamModelError as exc:
                # The quantitative detection above already succeeded and is
                # real, useful output — a Gemini failure (quota, outage)
                # shouldn't discard it. Degrade to a warning instead of
                # failing the whole request (observed directly: Gemini's
                # free-tier quota was exhausted mid-session during testing).
                warnings.append(f"Change-VQA answer unavailable: {exc.message}")

        parameters = {
            "model_mode": result["model_mode"],
            "checkpoint": os.path.basename(settings.change_checkpoint_path),
            "device": settings.change_device,
            "threshold": detector.threshold,
            "pixel_size_m": detector.pixel_size_m,
        }
        if answer is not None:
            parameters["vqa_model"] = settings.gemini_text_model

        return ToolResult(
            answer=answer,
            change_percentage=result["change_percentage"],
            regions=result["regions"],
            images=images,
            parameters=parameters,
            warnings=warnings,
        )


def _rgb_array_to_pil(arr):
    from PIL import Image
    import numpy as np

    if arr.dtype != np.uint8:
        arr = arr.astype("uint8")
    return Image.fromarray(arr)
