"""
Generic change-mask overlay renderer.

`vqa_and_change_using_gemini/imaging.py::create_change_overlay` hard-codes
an optical (13-band) -> RGB conversion internally, so it only works when an
optical T2 image exists. This is the same drawing logic (mask + boxes +
confidence labels) built on top of an already-computed (H, W, 3) uint8
preview instead, so it also covers the SAR-only change-detection path,
without editing the existing optical-specific helper.
"""
from __future__ import annotations

import io

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image


def render_overlay(rgb_preview, result: dict) -> Image.Image:
    mask = result["change_mask"]
    boxes = result["regions"]

    fig = Figure(figsize=(10, 7), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_preview)
    ax.imshow(mask, cmap="Reds", alpha=0.40, vmin=0, vmax=1)

    for box in boxes:
        x, y, w, h = box["bbox_px"]
        conf = box["mean_confidence"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))
        ax.text(
            x, max(0, y - 4), f"{conf:.2f}",
            color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.7, pad=1),
        )

    ax.set_title(f"Predicted Changes ({result['model_mode']} mode)")
    ax.axis("off")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")
