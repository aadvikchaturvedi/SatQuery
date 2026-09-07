"""
Single-image captioning / scene description — the product spec's second
mandatory single-image task (chosen over text-guided region grounding:
grounding needs pixel-accurate bounding boxes, which a general-purpose VLM
cannot reliably produce for remote-sensing imagery without adaptation;
captioning degrades gracefully to a general but honest description).
"""
from __future__ import annotations

from PIL import Image

from app.config import get_settings
from app.exceptions import ValidationFailed
from app.tools.base import Tool, ToolResult
from app.tools.gemini_client import generate_content_with_retry
from app.types import InputBundle

_PROMPT = """You are a remote-sensing scene-description assistant. Describe this satellite/aerial image for someone who cannot see it.

Cover, where visible:
- Dominant land-cover types (e.g. urban/built-up, agricultural, forest, water, bare soil).
- Notable man-made structures or infrastructure.
- Approximate spatial layout (e.g. "water body in the lower-left").

Do not invent place names, dates, or facts not visible in the image. If the user gave additional guidance, incorporate it: {guidance}
"""


class CaptioningTool(Tool):
    name = "single_image_captioning"
    description = "Generates a scene-description caption for one optical or SAR image."
    requires = ["exactly one single-timestep image (optical_t1 or sar_t1)"]
    domain_adapted = False

    def run(self, bundle: InputBundle) -> ToolResult:
        image = bundle.single_image
        if image is None:
            raise ValidationFailed("single_image_captioning requires exactly one single-timestep image.")

        settings = get_settings()
        prompt = _PROMPT.format(guidance=bundle.query or "none")
        answer = generate_content_with_retry(
            model=settings.gemini_text_model,
            contents=[prompt, Image.fromarray(image.rgb_preview)],
        )
        return ToolResult(
            answer=answer,
            parameters={"backend": "gemini", "model": settings.gemini_text_model, "modality": image.modality.value},
            warnings=[
                "Captioning answered by a general-purpose VLM (Gemini), not a "
                "remote-sensing-fine-tuned captioning model."
            ],
        )
