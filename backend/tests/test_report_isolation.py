"""Authorization: one caller must not be able to read another caller's
execution reports, even though there is no user-account system."""
from app.config import Settings
from app.exceptions import UnauthorizedError


def _any_nonempty_key(x_api_key):
    if not x_api_key:
        raise UnauthorizedError("Missing API key.")
    return x_api_key


def test_reports_are_isolated_per_api_key(client, monkeypatch, sample_t1_files, sample_t2_files):
    settings = Settings(backend_api_key="key-for-alice-and-bob", environment="production")
    # Any non-empty key is accepted (single shared-secret auth model), but
    # records are scoped by the hash of whichever key was actually sent.
    monkeypatch.setattr("app.security.get_settings", lambda: settings)
    monkeypatch.setattr("app.security._check_api_key", _any_nonempty_key)

    # change_vqa runs fully offline (no Gemini call needed to get a 200,
    # see test_analyze_change_integration.py), so this test isolates the
    # authorization behavior from upstream Gemini availability.
    files = [("optical_t1_files", (name, content, "image/tiff")) for name, content in sample_t1_files.items()]
    files += [("optical_t2_files", (name, content, "image/tiff")) for name, content in sample_t2_files.items()]

    resp_alice = client.post(
        "/api/v1/analyze",
        data={"query": "What changed between these two dates?"},
        files=files,
        headers={"X-API-Key": "alice-key"},
    )
    assert resp_alice.status_code == 200, resp_alice.text
    execution_id = resp_alice.json()["execution_id"]

    # Alice can read her own report.
    own = client.get(f"/api/v1/reports/{execution_id}", headers={"X-API-Key": "alice-key"})
    assert own.status_code == 200

    # Bob, a different caller, cannot.
    other = client.get(f"/api/v1/reports/{execution_id}", headers={"X-API-Key": "bob-key"})
    assert other.status_code == 404

    # Nor can an unauthenticated caller.
    anon = client.get(f"/api/v1/reports/{execution_id}")
    assert anon.status_code == 401
