import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Must be set before app.config.get_settings() is first called anywhere
# (it's lru_cached), so tests run offline/deterministically by default:
# no GEMINI_API_KEY -> Gemini-backed tools degrade gracefully instead of
# making real network calls.
_TMP_DATA_DIR = tempfile.mkdtemp(prefix="satquery-test-")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("ALLOW_NO_AUTH_IN_DEV", "true")
os.environ.setdefault("BACKEND_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("CHANGE_DEVICE", "cpu")
os.environ.setdefault("DATA_DIR", _TMP_DATA_DIR)
os.environ.setdefault("DB_PATH", str(Path(_TMP_DATA_DIR) / "test.db"))

REPO_ROOT = BACKEND_DIR.parent
SAMPLE_T1_DIR = REPO_ROOT / "vqa_and_change_using_gemini" / "imgs_1"
SAMPLE_T2_DIR = REPO_ROOT / "vqa_and_change_using_gemini" / "imgs_2"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.db import init_db
    from app.main import app

    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_t1_files() -> dict[str, bytes]:
    assert SAMPLE_T1_DIR.is_dir(), f"missing fixture dir: {SAMPLE_T1_DIR}"
    return {p.name: p.read_bytes() for p in SAMPLE_T1_DIR.glob("*.tif")}


@pytest.fixture(scope="session")
def sample_t2_files() -> dict[str, bytes]:
    assert SAMPLE_T2_DIR.is_dir(), f"missing fixture dir: {SAMPLE_T2_DIR}"
    return {p.name: p.read_bytes() for p in SAMPLE_T2_DIR.glob("*.tif")}


def make_geotiff_bytes(width: int, height: int, value: float = 500.0) -> bytes:
    """A minimal single-band GeoTIFF, for ingestion validation tests that
    don't need real satellite content, only a well-formed raster."""
    import io

    data = np.full((height, width), value, dtype=np.float32)
    transform = from_origin(0, 0, 10, 10)
    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as mem:
        with mem.open(
            driver="GTiff", height=height, width=width, count=1,
            dtype="float32", crs="EPSG:32631", transform=transform,
        ) as dataset:
            dataset.write(data, 1)
        buf.write(mem.read())
    return buf.getvalue()
