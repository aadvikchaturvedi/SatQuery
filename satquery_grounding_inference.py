"""
SatQuery AI - RGB Satellite Grounding Inference
================================================

Deployment target:
    Backend/server machine with a CUDA GPU.

Model:
    Qwen/Qwen2.5-VL-3B-Instruct + trained PEFT LoRA adapter.

This file DOES NOT require:
    - DIOR dataset
    - training annotations
    - Google Drive
    - Colab

The backend only needs:
    1. This file
    2. The saved LoRA adapter directory (latest_adapter)
    3. Internet access on first run OR a local Hugging Face copy of the base model
    4. A CUDA-capable GPU with enough VRAM for practical inference

The public API is the GroundingPredictor.predict_* methods.

Example:

    predictor = GroundingPredictor(
        adapter_dir="/models/satquery/latest_adapter"
    )

    result = predictor.predict_file(
        image_path="/data/satellite.jpg",
        query="Find all vehicles."
    )

The returned dict is JSON-serializable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont

from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# This must point to the LoRA adapter produced by training.
# Example:
#   /models/satquery/latest_adapter
DEFAULT_ADAPTER_DIR = os.environ.get(
    "SATQUERY_ADAPTER_DIR",
    "./latest_adapter",
)

# Optional local Hugging Face cache.
DEFAULT_HF_CACHE = os.environ.get(
    "SATQUERY_HF_CACHE",
    "./hf_cache",
)

# Same maximum image dimension used by the latest training run.
MAX_IMAGE_SIZE = 280

# Generation is deterministic for grounding.
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a satellite image analysis assistant. "
    "When asked to locate objects, respond ONLY with bounding box "
    "coordinates in the format: "
    "<boxes>[[x1,y1,x2,y2],...]</boxes> "
    "where coordinates are absolute pixel values "
    "(left,top,right,bottom). "
    "Do not include any other text."
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_for_model(
    image: Image.Image,
    max_size: int = MAX_IMAGE_SIZE,
) -> Tuple[Image.Image, float, float]:
    """
    Resize image exactly in the same general manner as training:
    longest side <= max_size.

    Returns:
        resized_image,
        scale_x = resized_width / original_width,
        scale_y = resized_height / original_height
    """
    image = image.convert("RGB")

    orig_w, orig_h = image.size

    scale = max_size / max(orig_w, orig_h)

    if scale >= 1.0:
        return image.copy(), 1.0, 1.0

    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    return (
        resized,
        new_w / orig_w,
        new_h / orig_h,
    )


# ============================================================
# BOX VALIDATION / SCALING
# ============================================================

def validate_and_clip_box(
    box: Sequence[Any],
    width: int,
    height: int,
) -> Optional[List[int]]:
    """Convert a candidate box to a valid integer xyxy box."""
    if len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = [
            float(v) for v in box
        ]
    except (TypeError, ValueError):
        return None

    if not all(
        __import__("math").isfinite(v)
        for v in (x1, y1, x2, y2)
    ):
        return None

    # Clamp to image bounds.
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))

    # Keep valid boxes at least 1 pixel wide/high.
    if x2 <= x1:
        if x1 < width:
            x2 = min(float(width), x1 + 1.0)
        else:
            x1 = max(0.0, x2 - 1.0)

    if y2 <= y1:
        if y1 < height:
            y2 = min(float(height), y1 + 1.0)
        else:
            y1 = max(0.0, y2 - 1.0)

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
    ]


def scale_boxes_to_original(
    boxes_resized: Sequence[Sequence[Any]],
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
) -> List[List[int]]:
    """Map model-coordinate boxes back to original image pixels."""
    result: List[List[int]] = []

    for box in boxes_resized:
        if len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [
                float(v) for v in box
            ]
        except (TypeError, ValueError):
            continue

        original_box = [
            x1 / scale_x,
            y1 / scale_y,
            x2 / scale_x,
            y2 / scale_y,
        ]

        valid = validate_and_clip_box(
            original_box,
            original_width,
            original_height,
        )

        if valid is not None:
            result.append(valid)

    return result


def normalize_boxes(
    boxes: Sequence[Sequence[int]],
    width: int,
    height: int,
) -> List[List[float]]:
    """Return xyxy boxes normalized to [0,1]."""
    if width <= 0 or height <= 0:
        return []

    return [
        [
            round(box[0] / width, 6),
            round(box[1] / height, 6),
            round(box[2] / width, 6),
            round(box[3] / height, 6),
        ]
        for box in boxes
    ]


# ============================================================
# ROBUST OUTPUT PARSER
# ============================================================

def _extract_balanced_json_array(text: str) -> Optional[str]:
    """
    Find the first balanced JSON-like array beginning with '['.
    This is deliberately conservative.
    """
    start = text.find("[")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        start = text.find("[", start + 1)

    return None


def parse_bounding_boxes(
    text: str,
    width: int,
    height: int,
) -> Tuple[List[List[int]], str]:
    """
    Parse model output supporting:
      1. <boxes>[[...], ...]</boxes>
      2. bare [[...], ...]
      3. [{"bbox_2d":[...], "label":"..."}]
      4. markdown fenced JSON
      5. regex fallback

    Returns:
        boxes, parser_mode
    """
    if not text:
        return [], "empty"

    candidates: List[Tuple[Any, str]] = []

    # --------------------------------------------------------
    # 1. Explicit <boxes>...</boxes>
    # --------------------------------------------------------

    m = re.search(
        r"<boxes>\s*(.*?)\s*</boxes>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        inner = m.group(1).strip()

        candidates.append(
            (inner, "boxes_tag")
        )

    # --------------------------------------------------------
    # 2. Extract arrays from full text
    # --------------------------------------------------------

    balanced = _extract_balanced_json_array(text)

    if balanced:
        candidates.append(
            (balanced, "json_array")
        )

    # --------------------------------------------------------
    # 3. Markdown fences
    # --------------------------------------------------------

    fenced = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for item in fenced:
        candidates.append(
            (item.strip(), "markdown_json")
        )

    # --------------------------------------------------------
    # Try structured JSON
    # --------------------------------------------------------

    for candidate, mode in candidates:

        obj: Any

        if isinstance(candidate, str):
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
        else:
            obj = candidate

        # Case A: [[x1,y1,x2,y2], ...]
        if isinstance(obj, list):

            boxes: List[List[int]] = []

            for item in obj:

                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 4
                ):
                    valid = validate_and_clip_box(
                        item,
                        width,
                        height,
                    )

                    if valid is not None:
                        boxes.append(valid)

                elif isinstance(item, dict):

                    bbox = item.get(
                        "bbox_2d",
                        item.get("bbox"),
                    )

                    if (
                        isinstance(
                            bbox,
                            (list, tuple)
                        )
                        and len(bbox) == 4
                    ):
                        valid = validate_and_clip_box(
                            bbox,
                            width,
                            height,
                        )

                        if valid is not None:
                            boxes.append(valid)

            if boxes:
                return boxes, mode

        # Case B: [{"bbox_2d": [...]}]
        if isinstance(obj, dict):

            for key in (
                "boxes",
                "bboxes",
                "bounding_boxes",
                "predictions",
            ):

                value = obj.get(key)

                if not isinstance(
                    value,
                    list
                ):
                    continue

                boxes = []

                for item in value:

                    if isinstance(
                        item,
                        dict
                    ):
                        bbox = item.get(
                            "bbox_2d",
                            item.get("bbox"),
                        )
                    else:
                        bbox = item

                    if (
                        isinstance(
                            bbox,
                            (list, tuple)
                        )
                        and len(bbox) == 4
                    ):
                        valid = (
                            validate_and_clip_box(
                                bbox,
                                width,
                                height,
                            )
                        )

                        if valid is not None:
                            boxes.append(valid)

                if boxes:
                    return boxes, mode

    # --------------------------------------------------------
    # Regex fallback
    # --------------------------------------------------------

    regex_matches = re.findall(
        r"\[\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*\]",
        text,
    )

    regex_boxes: List[List[int]] = []

    for match in regex_matches:

        valid = validate_and_clip_box(
            match,
            width,
            height,
        )

        if valid is not None:
            regex_boxes.append(valid)

    if regex_boxes:
        return regex_boxes, "regex"

    return [], "parse_failed"


# ============================================================
# VISUALIZATION
# ============================================================

def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[int]],
) -> Image.Image:
    """Return a copy of image with predicted boxes."""
    output = image.convert("RGB").copy()

    draw = ImageDraw.Draw(output)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes, start=1):

        x1, y1, x2, y2 = box

        # Draw multiple outlines to keep them visible.
        for offset in range(3):
            draw.rectangle(
                [
                    x1 - offset,
                    y1 - offset,
                    x2 + offset,
                    y2 + offset,
                ],
                outline="red",
                width=1,
            )

        label = str(i)

        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )

        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_x = max(
            0,
            min(
                image.width - tw - 4,
                x1,
            ),
        )

        label_y = max(
            0,
            y1 - th - 4,
        )

        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + tw + 4,
                label_y + th + 4,
            ],
            fill="red",
        )

        draw.text(
            (
                label_x + 2,
                label_y + 2,
            ),
            label,
            fill="white",
            font=font,
        )

    return output


def image_to_base64_jpeg(
    image: Image.Image,
    quality: int = 90,
) -> str:
    """Encode PIL image as a data URI suitable for a JSON response."""
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# MODEL
# ============================================================

class GroundingPredictor:
    """
    Loads the fine-tuned grounding model once and reuses it
    for incoming backend requests.

    A lock is used because multiple simultaneous generations
    on one GPU can cause unnecessary VRAM spikes.
    """

    def __init__(
        self,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        model_id: str = DEFAULT_MODEL_ID,
        hf_cache_dir: Optional[str] = DEFAULT_HF_CACHE,
        max_image_size: int = MAX_IMAGE_SIZE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        load_in_4bit: bool = True,
    ):
        self.adapter_dir = os.path.abspath(
            adapter_dir
        )

        self.model_id = model_id

        self.hf_cache_dir = (
            os.path.abspath(hf_cache_dir)
            if hf_cache_dir
            else None
        )

        self.max_image_size = max_image_size
        self.max_new_tokens = max_new_tokens

        self.device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        if self.device.type != "cuda":
            raise RuntimeError(
                "SatQuery grounding inference is configured "
                "for CUDA. No CUDA GPU was detected. "
                "Use a CUDA-capable backend machine."
            )

        self.lock = Lock()

        print("=" * 70)
        print("SATQUERY RGB GROUNDING MODEL")
        print("=" * 70)

        print("Base model :", self.model_id)
        print("Adapter    :", self.adapter_dir)
        print("Device     :", self.device)
        print("GPU        :", torch.cuda.get_device_name(0))
        print(
            "Adapter exists:",
            os.path.isdir(self.adapter_dir),
        )

        if not os.path.isdir(
            self.adapter_dir
        ):
            raise FileNotFoundError(
                "LoRA adapter directory not found:\n"
                + self.adapter_dir
            )

        adapter_config = os.path.join(
            self.adapter_dir,
            "adapter_config.json",
        )

        if not os.path.isfile(
            adapter_config
        ):
            raise FileNotFoundError(
                "adapter_config.json not found in:\n"
                + self.adapter_dir
            )

        if load_in_4bit:

            if self.device.type != "cuda":
                raise RuntimeError(
                    "4-bit bitsandbytes inference requires "
                    "CUDA in this deployment."
                )

            if torch.cuda.is_bf16_supported():

                compute_dtype = (
                    torch.bfloat16
                )

            else:

                compute_dtype = (
                    torch.float16
                )

            bnb_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=(
                        compute_dtype
                    ),
                    bnb_4bit_use_double_quant=True,
                )
            )

            print(
                "Quantization : 4-bit NF4"
            )

            print(
                "Compute dtype:",
                compute_dtype,
            )

        else:

            bnb_config = None
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            print(
                "Quantization : disabled"
            )

        print("\nLoading base model...")

        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": compute_dtype,
        }

        if bnb_config is not None:
            load_kwargs[
                "quantization_config"
            ] = bnb_config

        if self.hf_cache_dir:
            os.makedirs(
                self.hf_cache_dir,
                exist_ok=True
            )

            load_kwargs[
                "cache_dir"
            ] = self.hf_cache_dir

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        )

        print("Loading LoRA adapter...")

        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_dir,
            is_trainable=False,
        )

        self.model.eval()

        # Processor was saved together with the adapter by the
        # training notebook. If unavailable, fall back to base model.
        try:
            self.processor = (
                AutoProcessor.from_pretrained(
                    self.adapter_dir
                )
            )
            print(
                "Processor source: adapter directory"
            )

        except Exception:

            if self.hf_cache_dir:

                self.processor = (
                    AutoProcessor.from_pretrained(
                        self.model_id,
                        cache_dir=self.hf_cache_dir,
                    )
                )

            else:

                self.processor = (
                    AutoProcessor.from_pretrained(
                        self.model_id
                    )
                )

            print(
                "Processor source: base model"
            )

        self.processor.tokenizer.padding_side = (
            "left"
        )

        print("\n✅ Model loaded.")
        print("✅ LoRA adapter loaded.")
        print("✅ Model set to eval mode.")

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"GPU memory allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU memory reserved : "
                f"{reserved:.2f} GB"
            )

    # --------------------------------------------------------
    # Internal generation
    # --------------------------------------------------------

    def _generate(
        self,
        image: Image.Image,
        query: str,
    ) -> Dict[str, Any]:

        original_width, original_height = (
            image.size
        )

        model_image, scale_x, scale_y = (
            resize_for_model(
                image,
                max_size=self.max_image_size,
            )
        )

        model_width, model_height = (
            model_image.size
        )

        messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": [

                    {
                        "type": "image",
                        "image": model_image,
                    },

                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        ]

        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(messages)
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Move only tensor values.
        inputs = {
            key: (
                value.to(
                    self.device
                )
                if isinstance(
                    value,
                    torch.Tensor
                )
                else value
            )
            for key, value in inputs.items()
        }

        input_token_length = (
            inputs["input_ids"].shape[1]
        )

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            with torch.autocast(
                device_type="cuda",
                dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            ):

                generated_ids = (
                    self.model.generate(
                        **inputs,
                        max_new_tokens=(
                            self.max_new_tokens
                        ),
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )
                )

        # Only decode newly generated tokens.
        generated_new_tokens = (
            generated_ids[
                :,
                input_token_length:
            ]
        )

        raw_text = (
            self.processor.batch_decode(
                generated_new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ).strip()

        # ----------------------------------------------------
        # Parse model-coordinate boxes
        # ----------------------------------------------------

        boxes_resized, parser_mode = (
            parse_bounding_boxes(
                raw_text,
                model_width,
                model_height,
            )
        )

        # ----------------------------------------------------
        # Convert to original image coordinates
        # ----------------------------------------------------

        boxes_original = (
            scale_boxes_to_original(
                boxes_resized,
                scale_x,
                scale_y,
                original_width,
                original_height,
            )
        )

        normalized = normalize_boxes(
            boxes_original,
            original_width,
            original_height,
        )

        return {
            "query": query,
            "boxes": boxes_original,
            "num_boxes": len(boxes_original),
            "image_width": original_width,
            "image_height": original_height,
            "normalized_boxes": normalized,
            "raw_model_output": raw_text,
            "parser_mode": parser_mode,
            "model_image_width": model_width,
            "model_image_height": model_height,
        }

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def predict_pil(
        self,
        image: Image.Image,
        query: str,
        include_annotated_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Main API for a backend.

        Args:
            image:
                PIL RGB image.

            query:
                Natural language grounding request.
                Examples:
                    "Find the airport."
                    "Find all vehicles."
                    "Locate all ships."

            include_annotated_image:
                Add a base64 JPEG with predicted boxes.

        Returns:
            JSON-serializable dictionary.
        """
        if not isinstance(
            query,
            str
        ) or not query.strip():

            raise ValueError(
                "query must be a non-empty string."
            )

        if not isinstance(
            image,
            Image.Image
        ):

            raise TypeError(
                "image must be a PIL.Image.Image."
            )

        image = image.convert(
            "RGB"
        )

        start = time.perf_counter()

        with self.lock:

            result = self._generate(
                image,
                query.strip(),
            )

        latency_ms = (
            (time.perf_counter() - start)
            * 1000.0
        )

        result[
            "latency_ms"
        ] = round(
            latency_ms,
            2,
        )

        if include_annotated_image:

            annotated = draw_boxes(
                image,
                result["boxes"],
            )

            result[
                "annotated_image_base64"
            ] = image_to_base64_jpeg(
                annotated
            )

        return result

    def predict_file(
        self,
        image_path: str,
        query: str,
        include_annotated_image: bool = False,
    ) -> Dict[str, Any]:
        """Run inference from an image file path."""
        if not os.path.isfile(
            image_path
        ):
            raise FileNotFoundError(
                image_path
            )

        with Image.open(
            image_path
        ) as image:

            return self.predict_pil(
                image.copy(),
                query,
                include_annotated_image,
            )

    def predict_bytes(
        self,
        image_bytes: bytes,
        query: str,
        include_annotated_image: bool = False,
    ) -> Dict[str, Any]:
        """Run inference directly on uploaded HTTP/file bytes."""
        if not isinstance(
            image_bytes,
            (bytes, bytearray)
        ):
            raise TypeError(
                "image_bytes must be bytes."
            )

        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            return self.predict_pil(
                image.copy(),
                query,
                include_annotated_image,
            )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "SatQuery AI RGB satellite "
            "image grounding inference"
        )
    )

    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER_DIR,
        help=(
            "Path to trained LoRA adapter "
            "(latest_adapter directory)."
        ),
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to RGB satellite image.",
    )

    parser.add_argument(
        "--query",
        required=True,
        help=(
            "Natural language grounding query."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional path for an annotated image."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    args = parser.parse_args()

    predictor = GroundingPredictor(
        adapter_dir=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )

    result = predictor.predict_file(
        args.image,
        args.query,
        include_annotated_image=False,
    )

    # Print pure JSON for easy backend integration.
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # Optional local visualization.
    if args.output:

        with Image.open(
            args.image
        ) as image:

            annotated = draw_boxes(
                image.convert("RGB"),
                result["boxes"],
            )

            annotated.save(
                args.output,
                quality=95,
            )

        print(
            f"\nAnnotated image saved: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
