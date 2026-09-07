"""
Deterministic, rule-based task classification.

The product spec requires the controller to "interpret the query and
classify the requested task" and produce "an auditable execution summary" —
a rule-based classifier is chosen over an LLM-based router specifically so
every routing decision is inspectable and reproducible (no hidden model
call decides which specialist runs), which is what "auditable" means here.

Task selection is driven primarily by *which images were actually
supplied* (per spec: "check the number, modality, ... of the input
images" before selecting), with query keywords only breaking the one
remaining ambiguity (VQA vs. captioning for a single image).
"""
from __future__ import annotations

import re

from app.exceptions import ValidationFailed
from app.schemas import TaskType
from app.types import InputBundle

_CAPTION_PATTERNS = re.compile(
    r"\b(describe|caption|summar(y|ize)|scene description|what('?s| is) (visible|in this image))\b",
    re.IGNORECASE,
)


def classify(bundle: InputBundle) -> TaskType:
    if bundle.has_bitemporal:
        # Mandatory change-analysis task. If both modalities' T1/T2 pairs are
        # present, ChangeVqaTool runs the model in fusion mode, which also
        # satisfies the cross-modal requirement in the same call.
        return TaskType.CHANGE_VQA

    if bundle.is_single_time_cross_modal:
        return TaskType.CROSS_MODAL_FUSION

    if bundle.single_image is not None:
        if _CAPTION_PATTERNS.search(bundle.query) and "?" not in bundle.query:
            return TaskType.CAPTIONING
        return TaskType.VQA

    raise ValidationFailed(
        "Could not classify a task from the supplied inputs: expected a "
        "single image, a same-time optical+SAR pair, or a bi-temporal pair."
    )
