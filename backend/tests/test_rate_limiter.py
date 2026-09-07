import pytest

from app.config import Settings
from app.exceptions import RateLimitedError
from app.rate_limiter import InProcessSlidingWindowLimiter, get_rate_limiter, reset_rate_limiter_for_tests


async def test_in_process_limiter_blocks_after_limit():
    limiter = InProcessSlidingWindowLimiter(max_requests=2, window_seconds=60)
    await limiter.check("key-a")
    await limiter.check("key-a")
    with pytest.raises(RateLimitedError):
        await limiter.check("key-a")
    # A different key has its own bucket.
    await limiter.check("key-b")


def test_factory_picks_in_process_limiter_without_redis_url(monkeypatch):
    reset_rate_limiter_for_tests()
    settings = Settings(redis_url=None, rate_limit_requests=5, rate_limit_window_seconds=30)
    monkeypatch.setattr("app.rate_limiter.get_settings", lambda: settings)
    try:
        limiter = get_rate_limiter()
        assert isinstance(limiter, InProcessSlidingWindowLimiter)
    finally:
        reset_rate_limiter_for_tests()


def test_factory_picks_redis_limiter_when_configured(monkeypatch):
    reset_rate_limiter_for_tests()
    settings = Settings(redis_url="redis://localhost:6379/0", rate_limit_requests=5, rate_limit_window_seconds=30)
    monkeypatch.setattr("app.rate_limiter.get_settings", lambda: settings)
    try:
        from app.rate_limiter import RedisSlidingWindowLimiter

        limiter = get_rate_limiter()
        assert isinstance(limiter, RedisSlidingWindowLimiter)
    finally:
        reset_rate_limiter_for_tests()


@pytest.fixture
async def redis_client():
    redis = pytest.importorskip("redis")
    client = redis.asyncio.from_url("redis://localhost:6379/0", decode_responses=False)
    try:
        await client.ping()
    except Exception:
        pytest.skip("no local Redis server reachable at redis://localhost:6379/0")
    yield client
    await client.flushdb()
    await client.aclose()


async def test_redis_limiter_blocks_after_limit_and_isolates_keys(redis_client):
    from app.rate_limiter import RedisSlidingWindowLimiter

    limiter = RedisSlidingWindowLimiter(redis_client, max_requests=2, window_seconds=60)
    await limiter.check("alice")
    await limiter.check("alice")
    with pytest.raises(RateLimitedError):
        await limiter.check("alice")
    # A different key is unaffected — this is exactly what an in-process
    # limiter cannot guarantee across multiple worker processes.
    await limiter.check("bob")


async def test_redis_limiter_expires_old_entries(redis_client):
    from app.rate_limiter import RedisSlidingWindowLimiter

    limiter = RedisSlidingWindowLimiter(redis_client, max_requests=1, window_seconds=1)
    await limiter.check("carol")
    with pytest.raises(RateLimitedError):
        await limiter.check("carol")

    import asyncio

    await asyncio.sleep(1.2)
    await limiter.check("carol")  # window has rolled forward; should succeed again


async def test_redis_limiter_fails_open_when_redis_unreachable():
    from app.rate_limiter import RedisSlidingWindowLimiter

    class _BrokenPipeline:
        def zremrangebyscore(self, *a, **k):
            return self

        def zadd(self, *a, **k):
            return self

        def zcard(self, *a, **k):
            return self

        def expire(self, *a, **k):
            return self

        async def execute(self):
            raise ConnectionError("simulated Redis outage")

    class _BrokenClient:
        def pipeline(self, transaction=True):
            return _BrokenPipeline()

    limiter = RedisSlidingWindowLimiter(_BrokenClient(), max_requests=1, window_seconds=60)
    # Must not raise: an unreachable rate-limit backend fails open, not closed.
    await limiter.check("anyone")
    await limiter.check("anyone")
