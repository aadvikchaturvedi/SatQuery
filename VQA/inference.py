import os
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "final_qwen_lora"
)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")


processor = AutoProcessor.from_pretrained(
    MODEL_DIR
)


base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)


model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR
)

model.eval()


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

    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": (
                        "Answer the following satellite image "
                        "question concisely.\n\n"
                        f"{question}"
                    )
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }

    start_time = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    inference_time = time.time() - start_time

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, input_length:]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    return {
        "answer": answer,
        "question": question,
        "model": "Qwen2.5-VL-3B-Instruct + SatQuery LoRA",
        "image_mode": "RGB",
        "image_format": extension.replace(".", "").upper(),
        "inference_time_seconds": round(
            inference_time,
            3
        )
    }
