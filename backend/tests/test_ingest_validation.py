"""
Ingestion validation, exercised through the real HTTP layer (multipart
upload -> FastAPI UploadFile) rather than hand-constructing UploadFile
objects, since its exact constructor is version-sensitive and the point of
these tests is what a real client experiences.
"""
from tests.conftest import make_geotiff_bytes


def test_blank_query_is_rejected(client):
    # A whitespace-only string (rather than "") avoids an httpx multipart
    # quirk where an empty-string form field is dropped entirely, which
    # would otherwise trip FastAPI's own "field required" 422 instead of
    # exercising our blank-query check.
    resp = client.post(
        "/api/v1/analyze",
        data={"query": "   "},
        files=[("optical_t1_files", ("B02.tif", make_geotiff_bytes(8, 8), "image/tiff"))],
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "ValidationFailed"


def test_no_images_is_rejected(client):
    resp = client.post("/api/v1/analyze", data={"query": "What is visible here?"})
    assert resp.status_code == 422


def test_unsupported_extension_is_rejected(client):
    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What is visible here?"},
        files=[("optical_t1_files", ("scene.exe", b"not an image", "application/octet-stream"))],
    )
    assert resp.status_code == 422
    assert "unsupported file extension" in resp.json()["message"].lower()


def test_incomplete_optical_band_set_is_rejected(client):
    """Only B02 uploaded: 12 of the 13 required Sentinel-2 bands are missing."""
    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What is visible here?"},
        files=[("optical_t1_files", ("B02.tif", make_geotiff_bytes(8, 8), "image/tiff"))],
    )
    assert resp.status_code == 422
    assert "missing sentinel-2 bands" in resp.json()["message"].lower()


def test_t2_without_t1_is_rejected(client):
    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What changed?"},
        files=[("optical_t2_files", ("B02.tif", make_geotiff_bytes(8, 8), "image/tiff"))],
    )
    assert resp.status_code == 422


def test_oversized_file_is_rejected(client, monkeypatch):
    from app import config

    tiny_limit_settings = config.Settings(max_upload_bytes=10)
    monkeypatch.setattr("app.ingest.get_settings", lambda: tiny_limit_settings)
    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What is visible here?"},
        files=[("optical_t1_files", ("B02.tif", make_geotiff_bytes(8, 8), "image/tiff"))],
    )
    assert resp.status_code == 422
    assert "upload limit" in resp.json()["message"].lower()
