"""
Execution-record persistence.

Schema is one table because there is exactly one entity the product spec
requires us to persist: a completed analysis run (the "auditable execution
summary" + "downloadable report"). Stdlib sqlite3 is used deliberately
instead of pulling in an ORM — the repo has no existing database layer to
match, the schema is a single flat table, and a single-file embedded DB
needs no separate service to run locally, which matches "can actually be
run locally by following the repo's existing setup conventions" (there is
no docker-compose / Postgres anywhere in this repo to extend).
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    api_key_hash    TEXT NOT NULL,
    query           TEXT NOT NULL,
    task            TEXT NOT NULL,
    models_used     TEXT NOT NULL,      -- JSON array
    parameters      TEXT NOT NULL,      -- JSON object
    status          TEXT NOT NULL,      -- "success" | "error"
    answer          TEXT,
    change_percentage REAL,
    regions         TEXT,               -- JSON array, nullable
    confidence      REAL,
    warnings        TEXT NOT NULL,      -- JSON array
    error_message   TEXT,
    image_dir       TEXT,
    duration_ms     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_executions_created_at ON executions(created_at);
CREATE INDEX IF NOT EXISTS idx_executions_api_key_hash ON executions(api_key_hash);
"""

_local = threading.local()


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = _connect()
    return _local.conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(_SCHEMA)
    conn.commit()


@contextmanager
def transaction():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def insert_execution(record: dict) -> None:
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO executions (
                id, created_at, api_key_hash, query, task, models_used, parameters,
                status, answer, change_percentage, regions, confidence,
                warnings, error_message, image_dir, duration_ms
            ) VALUES (
                :id, :created_at, :api_key_hash, :query, :task, :models_used, :parameters,
                :status, :answer, :change_percentage, :regions, :confidence,
                :warnings, :error_message, :image_dir, :duration_ms
            )
            """,
            {
                **record,
                "models_used": json.dumps(record["models_used"]),
                "parameters": json.dumps(record["parameters"]),
                "regions": json.dumps(record["regions"]) if record.get("regions") is not None else None,
                "warnings": json.dumps(record.get("warnings", [])),
            },
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["models_used"] = json.loads(d["models_used"])
    d["parameters"] = json.loads(d["parameters"])
    d["regions"] = json.loads(d["regions"]) if d["regions"] else None
    d["warnings"] = json.loads(d["warnings"])
    return d


def get_execution(execution_id: str, api_key_hash: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM executions WHERE id = ? AND api_key_hash = ?",
        (execution_id, api_key_hash),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_executions(api_key_hash: str, limit: int = 20, offset: int = 0) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM executions WHERE api_key_hash = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (api_key_hash, limit, offset),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]
