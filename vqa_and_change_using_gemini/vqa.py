"""
Gemini Change-VQA

Pipeline:

T1 optical image
       +
T2 optical image
       +
Change detection output
       |
       v
    Gemini VQA
       |
       v
"What changed?"
"Where did it change?"
"What type of change is visible?"
"""

import os
import json

from google import genai
from google.genai import types
from getpass import getpass

from imaging import (
    optical_to_rgb,
    numpy_to_pil,
    confidence_to_pil,
    create_change_overlay,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Put your Gemini API key in the GEMINI_API_KEY environment
# variable before running the script.
#
# Recommended:
#   set GEMINI_API_KEY=your_key
#
# Or:
#   os.environ["GEMINI_API_KEY"] = "your_key"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Run: setx GEMINI_API_KEY \"YOUR_API_KEY\" and restart VS Code."
    )

# Current Gemini model suitable for multimodal VQA.
MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# REGION INFORMATION
# ============================================================

def create_region_summary(
    result,
):
    """
    Convert detected regions into compact text
    that Gemini can use alongside the images.
    """

    regions = result[
        "regions"
    ]

    if not regions:
        return "No changed regions were detected."

    sorted_regions = sorted(
        regions,
        key=lambda x: -x["area_m2"],
    )

    lines = []

    for r in sorted_regions[:30]:

        x, y, w, h = r[
            "bbox_px"
        ]

        lines.append(
            f"""
Region {r['region_id']}:
- Bounding box: x={x}, y={y}, width={w}, height={h}
- Area: {r['area_m2']:.0f} m²
- Mean confidence: {r['mean_confidence']:.3f}
- Centroid: ({r['centroid_px'][0]:.0f}, {r['centroid_px'][1]:.0f})
"""
        )

    return "\n".join(lines)


# ============================================================
# GEMINI CHANGE VQA
# ============================================================

def ask_change_vqa(
    question,
    optical_t1,
    optical_t2,
    result,
):
    """
    Ask Gemini a question about the detected changes.

    Gemini receives:

    1. T1 RGB
    2. T2 RGB
    3. Confidence map
    4. Change mask + bounding boxes
    5. Numeric detection information
    """

    # --------------------------------------------------------
    # Convert images
    # --------------------------------------------------------

    t1_rgb = numpy_to_pil(
        optical_to_rgb(
            optical_t1
        )
    )

    t2_rgb = numpy_to_pil(
        optical_to_rgb(
            optical_t2
        )
    )

    confidence_img = confidence_to_pil(
        result["confidence_map"]
    )

    change_overlay = create_change_overlay(
        optical_t2,
        result,
    )

    # --------------------------------------------------------
    # Region information
    # --------------------------------------------------------

    region_summary = create_region_summary(
        result
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a remote-sensing change-analysis assistant.

You are given four images from a satellite change-detection
pipeline.

IMAGE 1:
T1 = earlier satellite image.

IMAGE 2:
T2 = later satellite image.

IMAGE 3:
Change-confidence map.
Bright/yellow areas indicate higher model confidence of change.
Dark/purple areas indicate lower confidence.

IMAGE 4:
T2 image with the predicted change mask and bounding boxes.
Each bounding box has its mean model confidence written above it.

The change detector found:

Region: {result["region"]}
Model mode: {result["model_mode"]}
Total changed area: {result["change_percentage"]:.2f}%
Number of detected regions: {len(result["regions"])}

Detected regions:
{region_summary}

USER QUESTION:
{question}

Instructions:
1. Compare T1 and T2 carefully.
2. Identify only changes that are visually supported by both images.
3. Use the change mask and confidence map as supporting evidence.
4. Do not assume a change is caused by season, construction, agriculture,
   weather, or any other factor unless it is clearly visible.
5. If you can identify the changed object, state it.
6. If you cannot identify the object confidently, say "uncertain".
7. Do not invent geographic information.
8. Distinguish between:
   - visible change
   - model-detected change
   - inferred cause
9. Give a concise answer first.
10. Then list the major detected changes with their approximate regions.
"""

    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,

            t1_rgb,

            t2_rgb,

            confidence_img,

            change_overlay,
        ],
    )

    return response.text


# ============================================================
# MAIN VQA LOOP
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "GEMINI CHANGE VQA"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These variables come from your change detection script.
    #
    # If this is a separate file, import them from wherever
    # your inference result is stored.
    # --------------------------------------------------------

    print(
        """
This file expects:

    optical_t1
    optical_t2
    result

from your ChangeNet inference.

The easiest way is to call ask_change_vqa()
at the END of your existing change detection script.
"""
    )