"""
Per-API-key rate limiting, with two interchangeable backends behind one
interface so the request-handling code in security.py never needs to know
which is active.

- InProcessSlidingWindowLimiter: correct for exactly one worker process.
  This was the only implementation until now — fine for local/dev, wrong
  the moment a second uvicorn worker or a second node is added, since each
  process would enforce its own independent budget.
- RedisSlidingWindowLimiter: correct across any number of worker
  processes/nodes, backed by Redis sorted sets (the standard sliding-window
  rate-limiting pattern: https://redis.io/glossary/rate-limiting/). Used
  automatically once REDIS_URL is configured.
"""
from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque

from app.config import get_settings
from app.exceptions import RateLimitedError


class RateLimiter(ABC):
    @abstractmethod
    async def check(self, key: str) -> None:
        """Raises RateLimitedError if `key` has exceeded its budget."""


class InProcessSlidingWindowLimiter(RateLimiter):
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    async def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                raise RateLimitedError(
                    f"Rate limit exceeded: {self.max_requests} requests per "
                    f"{self.window_seconds}s."
                )
            bucket.append(now)


class RedisSlidingWindowLimiter(RateLimiter):
    """
    Sliding-window-log algorithm on a per-key sorted set: each request adds
    a unique member scored by its timestamp; entries older than the window
    are trimmed first; the remaining cardinality is the request's decision.
    The whole sequence runs as one Redis MULTI/EXEC transaction (the
    asyncio client's `pipeline()` defaults to `transaction=True`), so
    concurrent callers across any number of processes never interleave.

    If Redis itself is unreachable, requests are allowed through rather
    than rejected — an infra blip in the rate limiter should not take the
    whole API down; this is a deliberate availability-over-strictness
    choice, logged so it's visible, not silent.
    """

    def __init__(self, redis_client, max_requests: int, window_seconds: int):
        self._redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def check(self, key: str) -> None:
        import logging

        redis_key = f"satquery:ratelimit:{key}"
        now = time.time()
        cutoff = now - self.window_seconds

        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            pipe.zadd(redis_key, {str(uuid.uuid4()): now})
            pipe.zcard(redis_key)
            pipe.expire(redis_key, self.window_seconds)
            _, _, count, _ = await pipe.execute()
        except Exception:
            logging.getLogger("satquery").warning(
                "Redis rate limiter unreachable; allowing request through (fail-open).",
                exc_info=True,
            )
            return

        if count > self.max_requests:
            raise RateLimitedError(
                f"Rate limit exceeded: {self.max_requests} requests per "
                f"{self.window_seconds}s."
            )


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                settings = get_settings()
                if settings.redis_url:
                    import redis.asyncio as redis_asyncio

                    client = redis_asyncio.from_url(settings.redis_url, decode_responses=False)
                    _limiter = RedisSlidingWindowLimiter(
                        client, settings.rate_limit_requests, settings.rate_limit_window_seconds
                    )
                else:
                    _limiter = InProcessSlidingWindowLimiter(
                        settings.rate_limit_requests, settings.rate_limit_window_seconds
                    )
    return _limiter


def reset_rate_limiter_for_tests() -> None:
    """Test-only: forces the next get_rate_limiter() call to re-read settings."""
    global _limiter
    _limiter = None
