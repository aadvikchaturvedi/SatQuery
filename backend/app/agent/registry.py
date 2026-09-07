"""Tool registry: the predefined set of specialist tools the controller may
select from (product spec: "select one or more models or tools from a
predefined registry"). Adding a new specialist means adding one entry here."""
from __future__ import annotations

from app.schemas import TaskType, ToolDescriptor
from app.tools.base import Tool
from app.tools.caption_tool import CaptioningTool
from app.tools.change_tool import ChangeVqaTool
from app.tools.fusion_tool import CrossModalFusionTool
from app.tools.vqa_tool import SingleImageVqaTool

_REGISTRY: dict[TaskType, Tool] = {
    TaskType.VQA: SingleImageVqaTool(),
    TaskType.CAPTIONING: CaptioningTool(),
    TaskType.CHANGE_VQA: ChangeVqaTool(),
    TaskType.CROSS_MODAL_FUSION: CrossModalFusionTool(),
}


def get_tool(task: TaskType) -> Tool:
    return _REGISTRY[task]


def list_tools() -> list[ToolDescriptor]:
    descriptors = []
    for task, tool in _REGISTRY.items():
        status = "ready"
        if task == TaskType.CHANGE_VQA:
            from app.config import get_settings

            if not get_settings().gemini_api_key:
                status = "degraded (GEMINI_API_KEY unset: quantitative results only)"
        elif not tool.domain_adapted:
            status = "ready (generic VLM, not remote-sensing fine-tuned)"
        descriptors.append(
            ToolDescriptor(
                name=tool.name,
                task=task,
                description=tool.description,
                requires=tool.requires,
                domain_adapted=tool.domain_adapted,
                status=status,
            )
        )
    return descriptors
