from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import get_execution, list_executions
from app.exceptions import NotFoundError, ValidationFailed
from app.schemas import ExecutionListItem, ExecutionListResponse
from app.security import hash_api_key, require_api_key

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("", response_model=ExecutionListResponse)
async def list_reports(
    api_key: str = Depends(require_api_key),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Lists past executions belonging to the caller's API key (audit trail)."""
    rows = list_executions(hash_api_key(api_key), limit=limit, offset=offset)
    return ExecutionListResponse(
        items=[
            ExecutionListItem(
                id=r["id"], created_at=r["created_at"], task=r["task"], status=r["status"], query=r["query"]
            )
            for r in rows
        ],
        limit=limit,
        offset=offset,
    )


@router.get("/{execution_id}")
async def get_report(execution_id: str, api_key: str = Depends(require_api_key)):
    """Full downloadable execution report: task, models used, parameters, outputs."""
    record = get_execution(execution_id, hash_api_key(api_key))
    if record is None:
        raise NotFoundError(f"No report found for execution_id={execution_id!r}.")

    record.pop("api_key_hash", None)
    image_dir = record.pop("image_dir", None)  # absolute server path: internal detail, not for clients
    record["image_urls"] = (
        {p.name: f"/api/v1/reports/{execution_id}/images/{p.name}" for p in Path(image_dir).glob("*.png")}
        if image_dir
        else {}
    )
    return record


@router.get("/{execution_id}/images/{image_name}")
async def get_report_image(execution_id: str, image_name: str, api_key: str = Depends(require_api_key)):
    """Serves one of the PNGs produced for an execution (e.g. overlay.png)."""
    record = get_execution(execution_id, hash_api_key(api_key))
    if record is None or not record.get("image_dir"):
        raise NotFoundError(f"No images found for execution_id={execution_id!r}.")

    # image_name comes from the client; resolve it strictly inside the
    # execution's own image directory to rule out path traversal.
    if "/" in image_name or "\\" in image_name or ".." in image_name:
        raise ValidationFailed("Invalid image name.")

    reports_root = get_settings().reports_dir.resolve()
    image_path = (Path(record["image_dir"]) / image_name).resolve()
    if reports_root not in image_path.parents or not image_path.is_file():
        raise NotFoundError(f"Image {image_name!r} not found for this execution.")

    return FileResponse(image_path, media_type="image/png")
