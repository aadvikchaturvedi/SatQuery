"""
API-key authentication and per-key rate limiting.

Scope decision: the product spec describes a single evaluator-facing tool
with no user accounts, roles, or per-user data isolation — so authorization
here is single-tier ("valid key -> allowed") rather than a full auth system.
If multi-tenant use ever becomes a requirement, replace `_check_api_key`
with a real identity lookup; the request-handling code above it does not
need to change.

Rate limiting itself lives in rate_limiter.py (in-process vs. Redis-backed
implementations behind one interface); this module just calls it.
"""
import hashlib
import hmac
import secrets

from fastapi import Header, Request

from app.config import get_settings
from app.exceptions import UnauthorizedError
from app.rate_limiter import get_rate_limiter


def _check_api_key(x_api_key: str | None) -> str:
    settings = get_settings()

    if not settings.backend_api_key:
        if settings.allow_no_auth_in_dev and settings.environment != "production":
            return "dev-no-auth"
        raise UnauthorizedError(
            "Server is not configured with BACKEND_API_KEY; refusing to serve "
            "requests in this environment."
        )

    if not x_api_key or not hmac.compare_digest(x_api_key, settings.backend_api_key):
        raise UnauthorizedError("Missing or invalid API key.")

    return x_api_key


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: authenticates the caller and applies rate limiting."""
    key = _check_api_key(x_api_key)
    await get_rate_limiter().check(key)
    request.state.api_key = key
    return key


def generate_api_key() -> str:
    """Used by the `scripts/create_api_key.py` helper — not called at request time."""
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """
    One-way digest used as the DB scoping column so execution records can be
    filtered per caller without persisting the raw secret at rest.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
