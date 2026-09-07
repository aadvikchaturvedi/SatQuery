"""
Single-image VQA — the mandatory baseline task from the product spec.

VQA_BACKEND=gemini (current default) calls Gemini directly with a
remote-sensing-framed prompt: a real, working integration, but not the
domain-fine-tuned model the spec ultimately requires.

VQA_BACKEND=qwen is a reserved switch for `VQA/inference.py`
(Qwen2.5-VL-3B + a LoRA adapter fine-tuned on BigEarthNet, satisfying the
spec's mandatory remote-sensing-adaptation requirement) once that adapter's
weights exist — see `final_qwen_lora/`, not produced yet. Wiring it in only
touches this file; the agent/controller/API layers are unaffected either way.
"""
from __future__ import annotations

from PIL import Image

from app.config import get_settings
from app.exceptions import NotFoundError, UpstreamModelError, ValidationFailed
from app.tools.base import Tool, ToolResult
from app.tools.gemini_client import generate_content_with_retry
from app.types import InputBundle

_PROMPT_TEMPLATE = """You are a remote-sensing visual question answering assistant analyzing a satellite/aerial image.

Rules:
1. Answer only what is visually supported by the image.
2. If you cannot determine the answer confidently, say "uncertain" rather than guessing.
3. Do not invent geographic, temporal, or administrative information not visible in the image.
4. Keep the answer concise and direct.

QUESTION: {question}
"""


def _run_gemini_vqa(image_array, question: str) -> str:
    settings = get_settings()
    image = Image.fromarray(image_array)
    prompt = _PROMPT_TEMPLATE.format(question=question)
    return generate_content_with_retry(model=settings.gemini_text_model, contents=[prompt, image])


class SingleImageVqaTool(Tool):
    name = "single_image_vqa"
    description = "Answers a natural-language question about one optical or SAR image."
    requires = ["exactly one single-timestep image (optical_t1 or sar_t1)"]
    domain_adapted = False  # true once VQA_BACKEND=qwen is wired to a trained adapter

    def run(self, bundle: InputBundle) -> ToolResult:
        image = bundle.single_image
        if image is None:
            raise ValidationFailed("single_image_vqa requires exactly one single-timestep image.")

        settings = get_settings()
        if settings.vqa_backend == "qwen":
            raise NotFoundError(
                "VQA_BACKEND=qwen is configured, but the fine-tuned Qwen-LoRA "
                "adapter (final_qwen_lora/) has not been produced/wired in yet. "
                "Set VQA_BACKEND=gemini, or complete the ML integration in "
                "app/tools/vqa_tool.py."
            )
        if settings.vqa_backend != "gemini":
            raise UpstreamModelError(f"Unknown VQA_BACKEND: {settings.vqa_backend!r}")

        answer = _run_gemini_vqa(image.rgb_preview, bundle.query)
        return ToolResult(
            answer=answer,
            parameters={"backend": "gemini", "model": settings.gemini_text_model, "modality": image.modality.value},
            warnings=[
                "VQA answered by a general-purpose VLM (Gemini), not the "
                "remote-sensing-fine-tuned model required by the product spec; "
                "see VQA_BACKEND=qwen for the intended swap-in path."
            ],
        )
