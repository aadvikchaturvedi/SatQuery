"""
Single-timestep cross-modal (optical + SAR) joint analysis.

`UnifiedChangeDetectionNet` already fuses optical+SAR, but only across two
timesteps (see change_tool.py) — its architecture has no single-timestep
forward path. For a same-time optical/SAR pair (e.g. "use the optical and
SAR images together to identify built-up and water-covered regions"), this
tool reasons over both previews with a SAR-physics-aware prompt via Gemini.
A dedicated trained fusion/segmentation model is future ML work (tracked,
not attempted here per current scope).
"""
from __future__ import annotations

from PIL import Image

from app.config import get_settings
from app.exceptions import ValidationFailed
from app.tools.base import Tool, ToolResult
from app.tools.gemini_client import generate_content_with_retry
from app.types import InputBundle

_PROMPT = """You are a remote-sensing analyst jointly interpreting a co-registered optical and SAR (Synthetic Aperture Radar) image pair of the same location and time.

IMAGE 1 is optical/multispectral (true-color composite): reflects land cover via color and texture, but is affected by clouds and lighting.
IMAGE 2 is SAR VV backscatter (grayscale intensity): bright = strong radar return (e.g. urban/built-up structures, rough surfaces), dark = weak return (e.g. calm open water, smooth surfaces); SAR sees through clouds and at night but has no color/spectral information and can show layover/shadow near tall structures.

Use each modality for what it is good at — do not treat SAR brightness as a color cue, and do not assume optical-only cloud gaps mean "no data" in the fused answer if SAR coverage is available there.

QUESTION: {question}

Give a concise, evidence-grounded answer, and note explicitly if the two modalities agree or disagree in any region.
"""


class CrossModalFusionTool(Tool):
    name = "cross_modal_fusion"
    description = "Joint optical+SAR reasoning for a single-timestep, co-registered image pair."
    requires = ["optical_t1 and sar_t1, no T2 images"]
    domain_adapted = False

    def run(self, bundle: InputBundle) -> ToolResult:
        if not bundle.is_single_time_cross_modal:
            raise ValidationFailed(
                "cross_modal_fusion requires exactly optical_t1 + sar_t1 (no T2 images)."
            )

        settings = get_settings()
        prompt = _PROMPT.format(question=bundle.query)
        answer = generate_content_with_retry(
            model=settings.gemini_text_model,
            contents=[
                prompt,
                Image.fromarray(bundle.optical_t1.rgb_preview),
                Image.fromarray(bundle.sar_t1.rgb_preview),
            ],
        )
        return ToolResult(
            answer=answer,
            parameters={"backend": "gemini", "model": settings.gemini_text_model},
            warnings=[
                "Cross-modal fusion answered by prompting a general-purpose VLM "
                "with both previews, not a trained optical-SAR fusion/segmentation model."
            ],
        )
