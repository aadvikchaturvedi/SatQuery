"""Shared Gemini client + retry/error-mapping used by every Gemini-backed tool."""
from __future__ import annotations

import time
from functools import lru_cache

from google import genai

from app.config import get_settings
from app.exceptions import UpstreamModelError

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0


@lru_cache
def get_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise UpstreamModelError(
            "GEMINI_API_KEY is not configured on the server; this tool cannot run."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def generate_content_with_retry(*, model: str, contents: list):
    """
    Gemini's multimodal endpoint 503s under transient upstream load (observed
    directly while testing this integration). Retries only on 5xx; a 4xx
    (bad request, invalid key, etc.) is a real error and is not retried.
    """
    client = get_client()
    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            if not getattr(response, "text", None):
                raise UpstreamModelError("Gemini returned an empty response.")
            return response.text
        except Exception as exc:  # noqa: BLE001 - genai raises its own exception types
            last_error = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            is_server_error = status is not None and 500 <= int(status) < 600
            if not is_server_error or attempt == _MAX_ATTEMPTS:
                break
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    raise UpstreamModelError(
        f"Gemini request failed after {_MAX_ATTEMPTS} attempt(s): {last_error}"
    )
