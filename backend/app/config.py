"""
Centralized runtime configuration, read from environment variables / .env.

Nothing outside this module should call os.getenv() directly for backend
settings — keeping every knob in one place is what makes the security
review in the engineering report possible.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- identity / environment -------------------------------------------------
    environment: str = Field(default="development")

    # --- auth ---------------------------------------------------------------
    # Shared-secret API key. There is no user/account system in the product
    # spec (single evaluator-facing tool), so per-request bearer auth against
    # one operator-issued key is the smallest reasonable auth model rather
    # than building out unused user/session infrastructure.
    backend_api_key: str | None = Field(default=None)
    allow_no_auth_in_dev: bool = Field(default=True)

    # --- upstream model providers --------------------------------------------
    gemini_api_key: str | None = Field(default=None)
    gemini_text_model: str = Field(default="gemini-3.8-flash")

    # VQA_BACKEND selects the implementation behind the single-image VQA /
    # captioning tools. "gemini" is the only backend wired up today; "qwen"
    # is a reserved value for the fine-tuned Qwen2.5-VL-LoRA adapter once its
    # weights exist (see VQA/inference.py) — swapping it in later should not
    # require touching the agent/router layer, only tools/vqa_tool.py.
    vqa_backend: str = Field(default="gemini")

    # --- change-detection checkpoint -----------------------------------------
    change_checkpoint_path: str = Field(
        default=str(REPO_ROOT / "vqa_and_change_using_gemini" / "unified_changenet_best.pt")
    )
    change_device: str = Field(default="cpu")

    # --- uploads / limits -----------------------------------------------------
    max_upload_bytes: int = Field(default=25 * 1024 * 1024)  # 25 MB per file
    max_files_per_request: int = Field(default=64)

    # --- rate limiting --------------------------------------------------------
    rate_limit_requests: int = Field(default=20)
    rate_limit_window_seconds: int = Field(default=60)
    # If set, rate limiting is enforced in Redis (correct across multiple
    # worker processes/nodes). If unset, falls back to an in-process limiter
    # — correct for the single-worker local/dev deployment this repo
    # currently targets, but each process would have its own independent
    # budget under multiple workers.
    redis_url: str | None = Field(default=None)

    # --- CORS -------------------------------------------------------------
    cors_allow_origins: str = Field(default="http://localhost:3000,http://localhost:5173")

    # --- storage ------------------------------------------------------------
    data_dir: str = Field(default=str(BACKEND_DIR / "data"))
    db_path: str = Field(default=str(BACKEND_DIR / "data" / "satquery.db"))

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def reports_dir(self) -> Path:
        p = Path(self.data_dir) / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
