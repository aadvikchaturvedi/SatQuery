"""
SatQuery AI backend — agentic remote-sensing vision-language assistant.

FastAPI was chosen because the only pre-existing hint of an intended HTTP
layer in this repo (`vqa_and_change_using_gemini/pipeline.py` docstring:
"The actual HTTP layer (api.py) is a thin wrapper around
run_change_detection()", and its use of `UploadFile.read()` semantics)
already assumes it; there is no other backend framework anywhere in the
repository to be consistent with instead.
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import init_db
from app.exceptions import SatQueryError
from app.routers import analyze, capabilities, health, reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("satquery")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("SatQuery backend started (environment=%s)", get_settings().environment)
    yield


app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic vision-language assistant for remote-sensing image analysis.",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(health.router)
app.include_router(capabilities.router)
app.include_router(analyze.router)
app.include_router(reports.router)


@app.exception_handler(SatQueryError)
async def handle_satquery_error(request: Request, exc: SatQueryError):
    request_id = str(uuid.uuid4())
    logger.warning(
        "request_id=%s status=%s error=%s message=%s",
        request_id, exc.status_code, type(exc).__name__, exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    # Full detail goes to the server log only; the client gets a safe, generic message.
    logger.exception("request_id=%s unhandled_error", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred.",
            "details": {},
            "request_id": request_id,
        },
    )
