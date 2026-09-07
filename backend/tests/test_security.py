import pytest

from app.config import Settings
from app.exceptions import UnauthorizedError
from app.security import _check_api_key, hash_api_key


def test_dev_mode_allows_missing_key(monkeypatch):
    settings = Settings(backend_api_key=None, allow_no_auth_in_dev=True, environment="development")
    monkeypatch.setattr("app.security.get_settings", lambda: settings)
    assert _check_api_key(None) == "dev-no-auth"


def test_production_without_configured_key_rejects_everyone(monkeypatch):
    settings = Settings(backend_api_key=None, allow_no_auth_in_dev=True, environment="production")
    monkeypatch.setattr("app.security.get_settings", lambda: settings)
    with pytest.raises(UnauthorizedError):
        _check_api_key(None)
    with pytest.raises(UnauthorizedError):
        _check_api_key("anything")


def test_valid_key_accepted_invalid_key_rejected(monkeypatch):
    settings = Settings(backend_api_key="correct-horse-battery-staple", environment="production")
    monkeypatch.setattr("app.security.get_settings", lambda: settings)

    assert _check_api_key("correct-horse-battery-staple") == "correct-horse-battery-staple"
    with pytest.raises(UnauthorizedError):
        _check_api_key("wrong-key")
    with pytest.raises(UnauthorizedError):
        _check_api_key(None)


def test_hash_api_key_is_deterministic_and_one_way():
    h1 = hash_api_key("some-secret")
    h2 = hash_api_key("some-secret")
    assert h1 == h2
    assert h1 != "some-secret"
    assert hash_api_key("different-secret") != h1
