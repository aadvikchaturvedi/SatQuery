"""
End-to-end integration test against the real trained checkpoint and the
real sample Sentinel-2 tiles checked into the repo (no ML code modified).
GEMINI_API_KEY is unset in the test environment (see conftest.py) so this
exercises the pure quantitative change-detection path deterministically,
without depending on network availability or Gemini's uptime — both of
which were observed to be flaky against the live API in manual testing.
"""


def test_change_detection_end_to_end_without_gemini(client, sample_t1_files, sample_t2_files):
    files = [("optical_t1_files", (name, content, "image/tiff")) for name, content in sample_t1_files.items()]
    files += [("optical_t2_files", (name, content, "image/tiff")) for name, content in sample_t2_files.items()]

    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What changed between these two dates?"},
        files=files,
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["task"] == "change_vqa"
    assert body["change_percentage"] is not None
    assert 0.0 <= body["change_percentage"] <= 100.0
    assert len(body["regions"]) > 0
    assert body["answer"] is None  # no GEMINI_API_KEY configured
    assert any("GEMINI_API_KEY" in w for w in body["execution_summary"]["warnings"])
    assert body["execution_summary"]["parameters"]["model_mode"] == "optical"
    assert "unified_changenet_best.pt" in body["execution_summary"]["models_used"]
    assert set(body["image_urls"]) == {"t2_preview.png", "confidence.png", "overlay.png"}

    # Report + image retrieval should work for the execution just created.
    report = client.get(body["report_url"])
    assert report.status_code == 200
    assert report.json()["id"] == body["execution_id"]

    img = client.get(body["image_urls"]["overlay.png"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"


def test_change_detection_degrades_gracefully_when_gemini_fails(
    client, monkeypatch, sample_t1_files, sample_t2_files
):
    """
    A Gemini failure (quota exhaustion, outage — both observed directly
    against the live API during manual testing) must not discard the
    already-computed, valid quantitative detection results.
    """
    from app.config import Settings
    from app.exceptions import UpstreamModelError

    def _boom(*a, **k):
        raise UpstreamModelError("simulated Gemini outage")

    settings_with_key = Settings(gemini_api_key="fake-key-for-this-test", change_device="cpu")
    monkeypatch.setattr("app.tools.change_tool.get_settings", lambda: settings_with_key)
    monkeypatch.setattr("app.tools.change_tool._ask_gemini_about_change", _boom)

    files = [("optical_t1_files", (name, content, "image/tiff")) for name, content in sample_t1_files.items()]
    files += [("optical_t2_files", (name, content, "image/tiff")) for name, content in sample_t2_files.items()]

    resp = client.post(
        "/api/v1/analyze",
        data={"query": "What changed between these two dates?"},
        files=files,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["change_percentage"] is not None
    assert len(body["regions"]) > 0
    assert body["answer"] is None
    assert any("simulated Gemini outage" in w for w in body["execution_summary"]["warnings"])
