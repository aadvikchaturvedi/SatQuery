"""
Analysis orchestration service: glues ingestion -> agent -> persistence
together and shapes the result into the public API response.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.agent.controller import AgentRunResult, run as run_agent
from app.config import get_settings
from app.db import insert_execution
from app.schemas import AnalyzeResponse, ChangedRegion, ExecutionSummary
from app.types import InputBundle


def _persist_images(execution_id: str, images: dict[str, bytes]) -> Path:
    settings = get_settings()
    out_dir = settings.reports_dir / execution_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in images.items():
        (out_dir / name).write_bytes(content)
    return out_dir


def run_analysis(bundle: InputBundle, api_key_hash: str) -> AnalyzeResponse:
    execution_id = str(uuid.uuid4())
    agent_result: AgentRunResult = run_agent(bundle)
    tool_result = agent_result.result

    image_dir = None
    image_urls: dict[str, str] = {}
    if tool_result.images:
        image_dir = _persist_images(execution_id, tool_result.images)
        image_urls = {
            name: f"/api/v1/reports/{execution_id}/images/{name}" for name in tool_result.images
        }

    insert_execution(
        {
            "id": execution_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "api_key_hash": api_key_hash,
            "query": bundle.query,
            "task": agent_result.task.value,
            "models_used": agent_result.models_used,
            "parameters": tool_result.parameters,
            "status": "success",
            "answer": tool_result.answer,
            "change_percentage": tool_result.change_percentage,
            "regions": tool_result.regions,
            "confidence": tool_result.confidence,
            "warnings": tool_result.warnings,
            "error_message": None,
            "image_dir": str(image_dir) if image_dir else None,
            "duration_ms": agent_result.duration_ms,
        }
    )

    return AnalyzeResponse(
        execution_id=execution_id,
        task=agent_result.task,
        answer=tool_result.answer,
        confidence=tool_result.confidence,
        change_percentage=tool_result.change_percentage,
        regions=[ChangedRegion(**r) for r in tool_result.regions] if tool_result.regions else None,
        image_urls=image_urls,
        execution_summary=ExecutionSummary(
            task=agent_result.task,
            models_used=agent_result.models_used,
            parameters=tool_result.parameters,
            input_summary=agent_result.input_summary,
            duration_ms=agent_result.duration_ms,
            warnings=tool_result.warnings,
        ),
        report_url=f"/api/v1/reports/{execution_id}",
    )
