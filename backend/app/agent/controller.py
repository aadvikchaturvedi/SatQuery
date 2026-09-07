"""
Agentic controller: the single place that turns (query, validated inputs)
into (selected task, selected tool(s), execution trace, combined output).

Per the product spec, only the observable execution trace — selected task,
models/tools, permitted parameters, and outputs — needs to be produced;
internal planning text is neither required nor evaluated, so this stays a
straight classify -> select -> execute -> summarize pipeline rather than an
LLM "reasoning" loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.agent.classifier import classify
from app.agent.registry import get_tool
from app.schemas import TaskType
from app.tools.base import ToolResult
from app.types import InputBundle


@dataclass
class AgentRunResult:
    task: TaskType
    tool_name: str
    result: ToolResult
    models_used: list[str]
    duration_ms: int
    input_summary: dict


def _input_summary(bundle: InputBundle) -> dict:
    def describe(img):
        if img is None:
            return None
        return {"modality": img.modality.value, "source": img.source_kind.value, "hw": list(img.hw)}

    return {
        "optical_t1": describe(bundle.optical_t1),
        "optical_t2": describe(bundle.optical_t2),
        "sar_t1": describe(bundle.sar_t1),
        "sar_t2": describe(bundle.sar_t2),
    }


def _models_used(result: ToolResult) -> list[str]:
    models = []
    if "checkpoint" in result.parameters:
        models.append(str(result.parameters["checkpoint"]))
    for key in ("model", "vqa_model"):
        if key in result.parameters:
            models.append(str(result.parameters[key]))
    return models or ["none"]


def run(bundle: InputBundle) -> AgentRunResult:
    start = time.perf_counter()

    task = classify(bundle)
    tool = get_tool(task)
    result = tool.run(bundle)

    duration_ms = int((time.perf_counter() - start) * 1000)

    return AgentRunResult(
        task=task,
        tool_name=tool.name,
        result=result,
        models_used=_models_used(result),
        duration_ms=duration_ms,
        input_summary=_input_summary(bundle),
    )
