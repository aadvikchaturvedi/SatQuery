"""Pydantic response DTOs. Nothing here ever mirrors a raw internal object —
every field is explicitly whitelisted for what the client is allowed to see."""
from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    VQA = "single_image_vqa"
    CAPTIONING = "single_image_captioning"
    CHANGE_VQA = "change_vqa"
    CROSS_MODAL_FUSION = "cross_modal_fusion"


class ChangedRegion(BaseModel):
    region_id: int
    bbox_px: list[int]
    area_m2: float
    mean_confidence: float
    centroid_px: list[float]


class ExecutionSummary(BaseModel):
    task: TaskType
    models_used: list[str]
    parameters: dict
    input_summary: dict
    duration_ms: int
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    execution_id: str
    task: TaskType
    answer: str | None = None
    confidence: float | None = None
    change_percentage: float | None = None
    regions: list[ChangedRegion] | None = None
    image_urls: dict[str, str] = Field(default_factory=dict)
    execution_summary: ExecutionSummary
    report_url: str


class ToolDescriptor(BaseModel):
    name: str
    task: TaskType
    description: str
    requires: list[str]
    domain_adapted: bool
    status: str


class CapabilitiesResponse(BaseModel):
    tools: list[ToolDescriptor]


class ExecutionListItem(BaseModel):
    id: str
    created_at: str
    task: TaskType
    status: str
    query: str


class ExecutionListResponse(BaseModel):
    items: list[ExecutionListItem]
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: str | None = None
