"""Typed domain exceptions. Each maps to exactly one HTTP status code in main.py."""


class SatQueryError(Exception):
    """Base class for all handled backend errors."""
    status_code = 500

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationFailed(SatQueryError):
    """Malformed request: missing fields, unsupported format, incompatible inputs."""
    status_code = 422


class UnauthorizedError(SatQueryError):
    status_code = 401


class RateLimitedError(SatQueryError):
    status_code = 429


class NotFoundError(SatQueryError):
    status_code = 404


class UpstreamModelError(SatQueryError):
    """A specialist tool (Gemini, ChangeNet) failed or returned something unusable."""
    status_code = 502
