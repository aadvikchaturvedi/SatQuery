"""
imaging.py

Pure image-generation helpers.

Rules these functions follow (this is what makes them safe to call
from a web request handler, a notebook, or a CLI script without any
of them stepping on each other):

    1. No pyplot. Everything uses an isolated Figure + FigureCanvasAgg,
       so nothing here ever touches global plotting state.
    2. No file I/O. Every function takes arrays in and returns either
       a numpy array, a PIL.Image, or raw PNG bytes. Saving to disk
       or sending over HTTP is the caller's job, not this module's.
    3. No side effects / no globals. Same input always gives the same
       output, so these are safe to call concurrently (e.g. multiple
       API requests at once).
"""

import io

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from PIL import Image


# ============================================================
# OPTICAL -> RGB
# ============================================================

def optical_to_rgb(optical_array, clip_max=3000.0):
    """
    Convert a 13-band Sentinel-2 array (13, H, W) to an (H, W, 3)
    uint8 RGB array.

    B04 -> Red, B03 -> Green, B02 -> Blue
    """

    r = np.clip(optical_array[3] / clip_max, 0, 1)
    g = np.clip(optical_array[2] / clip_max, 0, 1)
    b = np.clip(optical_array[1] / clip_max, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)

    return (rgb * 255).astype(np.uint8)


def numpy_to_pil(array):
    """Convert a numpy RGB array to a PIL Image."""

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array)


def pil_to_png_bytes(image):
    """Serialize a PIL Image to raw PNG bytes (for HTTP responses)."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# CONFIDENCE MAP
# ============================================================

def confidence_to_pil(confidence_map):
    """
    Render a change-confidence heatmap as a PIL Image.
    """

    fig = Figure(figsize=(8, 6), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    im = ax.imshow(confidence_map, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Change confidence")

    ax.set_title("Change Detection Confidence")
    ax.axis("off")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


# ============================================================
# CHANGE OVERLAY (mask + bounding boxes)
# ============================================================

def create_change_overlay(optical_t2, result):
    """
    Render the T2 image with the predicted change mask and
    bounding boxes drawn on top, as a PIL Image.

    `result` is expected to have:
        result["change_mask"]      -> (H, W) array
        result["confidence_map"]   -> (H, W) array (unused here, kept
                                        for signature parity)
        result["regions"]          -> list of dicts with
                                        "bbox_px" and "mean_confidence"
    """

    rgb = optical_to_rgb(optical_t2)
    mask = result["change_mask"]
    boxes = result["regions"]

    fig = Figure(figsize=(10, 7), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.imshow(rgb)
    ax.imshow(mask, cmap="Reds", alpha=0.40, vmin=0, vmax=1)

    for box in boxes:
        x, y, w, h = box["bbox_px"]
        conf = box["mean_confidence"]

        rectangle = Rectangle(
            (x, y), w, h,
            fill=False, edgecolor="lime", linewidth=2,
        )
        ax.add_patch(rectangle)

        ax.text(
            x, max(0, y - 4), f"{conf:.2f}",
            color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.7, pad=1),
        )

    ax.set_title("Predicted Changes and Confidence")
    ax.axis("off")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    return Image.open(buffer).convert("RGB")
