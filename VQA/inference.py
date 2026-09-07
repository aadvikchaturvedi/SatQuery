import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# The adapter is checked out at the repo root (VQA/../final_qwen_lora), not
# inside this directory, so it can be gitignored (*.safetensors) as a local
# artifact shared by both the notebooks here and the backend.
MODEL_DIR = str(Path(__file__).resolve().parent.parent / "final_qwen_lora")

_state: dict = {}


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model_and_processor():
    """Lazily loads the base model + LoRA adapter on first use, so importing
    this module (e.g. to reuse `predict_image`) doesn't force a model load
    for callers who never invoke it."""
    if "model" in _state:
        return _state["model"], _state["processor"], _state["device"]

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Qwen LoRA adapter not found at {MODEL_DIR}. Expected the "
            "final_qwen_lora/ directory at the repo root."
        )

    device = _select_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    ).to(device)

    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()

    _state["model"] = model
    _state["processor"] = processor
    _state["device"] = device
    return model, processor, device


def predict_image(image: Image.Image, question: str, max_new_tokens: int = 60) -> dict:
    """Core inference path, operating on an already-loaded PIL image so
    callers (e.g. the backend, which has image bytes/arrays in memory) don't
    need to round-trip through a temp file. `predict()` below wraps this for
    file-path/CLI/notebook use."""
    if not isinstance(question, str):
        raise TypeError("question must be a string.")

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    model, processor, device = _get_model_and_processor()
    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Answer the following satellite image "
                        "question concisely.\n\n"
                        f"{question}"
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    start_time = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    inference_time = time.time() - start_time

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]

    answer = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return {
        "answer": answer,
        "question": question,
        "model": "Qwen2.5-VL-3B-Instruct + SatQuery LoRA",
        "device": device,
        "inference_time_seconds": round(inference_time, 3),
    }


def predict(image_path, question, max_new_tokens=60):

    if not isinstance(image_path, str):
        raise TypeError("image_path must be a string.")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    extension = os.path.splitext(
        image_path
    )[1].lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            "Only JPG, JPEG and PNG images are supported."
        )

    image = Image.open(image_path)

    result = predict_image(image, question, max_new_tokens=max_new_tokens)
    result["image_mode"] = "RGB"
    result["image_format"] = extension.replace(".", "").upper()
    return result
