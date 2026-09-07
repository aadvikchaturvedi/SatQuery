"""Bridge to VQA/inference.py (Qwen2.5-VL-3B + SatQuery LoRA adapter), mirroring
gemini_client.py's role for the Gemini backend.

The import of `inference` (and the model load it triggers) is deferred into
a function so a server running VQA_BACKEND=gemini never pays the cost of
loading the ~3B-parameter base model + adapter.
"""
from __future__ import annotations

import sys
from functools import lru_cache

from PIL.Image import Image

from app.config import REPO_ROOT
from app.exceptions import UpstreamModelError

_VQA_DIR = REPO_ROOT / "VQA"


@lru_cache
def _inference_module():
    path = str(_VQA_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import inference as qwen_inference  # noqa: PLC0415 - deferred: loads a 3B model on first call

    return qwen_inference


def run_qwen_vqa(image: Image, question: str, max_new_tokens: int = 60) -> dict:
    try:
        module = _inference_module()
        return module.predict_image(image, question, max_new_tokens=max_new_tokens)
    except (OSError, FileNotFoundError) as exc:
        raise UpstreamModelError(f"Qwen LoRA adapter could not be loaded: {exc}") from exc
