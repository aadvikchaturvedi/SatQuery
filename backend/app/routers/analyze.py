from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ingest import build_input_bundle
from app.schemas import AnalyzeResponse
from app.security import hash_api_key, require_api_key
from app.services import run_analysis

router = APIRouter(prefix="/api/v1", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    query: str = Form(..., description="Natural-language question about the uploaded imagery."),
    optical_t1_files: list[UploadFile] = File(default=[]),
    optical_t2_files: list[UploadFile] = File(default=[]),
    sar_t1_files: list[UploadFile] = File(default=[]),
    sar_t2_files: list[UploadFile] = File(default=[]),
    api_key: str = Depends(require_api_key),
):
    """
    Single entry point for the agentic pipeline: validates + parses the
    uploaded imagery, classifies the task, executes the selected specialist
    tool, and returns an evidence-grounded response with a persisted,
    downloadable execution report.

    Field naming (`t1_files` / `t2_files`) mirrors the existing convention
    in `vqa_and_change_using_gemini/pipeline.py::run_change_detection`.
    """
    bundle = await build_input_bundle(
        query, optical_t1_files, optical_t2_files, sar_t1_files, sar_t2_files
    )
    # Model inference (torch, Gemini) is blocking; keep it off the event loop.
    return await run_in_threadpool(run_analysis, bundle, hash_api_key(api_key))
