"""
Single-image VQA — the mandatory baseline task from the product spec.

VQA_BACKEND=gemini (current default) calls Gemini directly with a
remote-sensing-framed prompt: a real, working integration, but not the
domain-fine-tuned model the spec ultimately requires.

VQA_BACKEND=qwen calls `VQA/inference.py` (Qwen2.5-VL-3B + a LoRA adapter
fine-tuned on BigEarthNet, satisfying the spec's mandatory
remote-sensing-adaptation requirement) via `qwen_client.py`, loading
`final_qwen_lora/` from the repo root.
"""
from __future__ import annotations

from PIL import Image

from app.config import get_settings
from app.exceptions import UpstreamModelError, ValidationFailed
from app.tools.base import Tool, ToolResult
from app.tools.gemini_client import generate_content_with_retry
from app.tools.qwen_client import run_qwen_vqa
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


def _run_qwen_vqa(image_array, question: str) -> dict:
    return run_qwen_vqa(Image.fromarray(image_array), question)


class SingleImageVqaTool(Tool):
    name = "single_image_vqa"
    description = "Answers a natural-language question about one optical or SAR image."
    requires = ["exactly one single-timestep image (optical_t1 or sar_t1)"]

    @property
    def domain_adapted(self) -> bool:
        return get_settings().vqa_backend == "qwen"

    def run(self, bundle: InputBundle) -> ToolResult:
        image = bundle.single_image
        if image is None:
            raise ValidationFailed("single_image_vqa requires exactly one single-timestep image.")

        settings = get_settings()
        if settings.vqa_backend == "qwen":
            result = _run_qwen_vqa(image.rgb_preview, bundle.query)
            return ToolResult(
                answer=result["answer"],
                parameters={
                    "backend": "qwen",
                    "model": result["model"],
                    "device": result["device"],
                    "modality": image.modality.value,
                    "inference_time_seconds": result["inference_time_seconds"],
                },
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
