"""Tool contract every specialist implements. The agent controller only ever
talks to tools through this interface, so a tool's internals (Gemini call,
local torch model, future Qwen adapter) are swappable without touching the
controller or the API layer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.types import InputBundle


@dataclass
class ToolResult:
    answer: str | None = None
    confidence: float | None = None
    change_percentage: float | None = None
    regions: list[dict] | None = None
    # name -> raw PNG bytes, persisted by the caller and exposed as report URLs.
    images: dict[str, bytes] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Tool(ABC):
    name: str
    description: str
    requires: list[str]
    domain_adapted: bool

    @abstractmethod
    def run(self, bundle: InputBundle) -> ToolResult:
        ...
