from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class TokenBucketLimiter:
    """Simple token bucket limiter for request pacing."""

    def __init__(self, capacity: int, window_seconds: float) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(capacity) / float(window_seconds)
        self.updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            wait_for = 0.0
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.updated_at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait_for = (1.0 - self.tokens) / self.refill_rate
            if wait_for > 0:
                time.sleep(wait_for)


class JsonHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        limiter: TokenBucketLimiter,
        timeout_s: float,
        max_retries: int,
        backoff_base_s: float,
        backoff_jitter_s: float,
        metrics: dict[str, int],
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = limiter
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_jitter_s = backoff_jitter_s
        self.metrics = metrics
        self.session = requests.Session()

    def get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        cache_dir: Path | None = None,
        cache_ttl_s: int | None = None,
    ) -> Any:
        if cache_dir is not None:
            cache_path = self._cache_path(cache_dir, endpoint, params or {})
            if cache_ttl_s and cache_path.exists():
                age_s = time.time() - cache_path.stat().st_mtime
                if age_s <= cache_ttl_s:
                    with cache_path.open("r", encoding="utf-8") as handle:
                        return json.load(handle)

        payload = self._request_with_retry(endpoint, params=params)

        if cache_dir is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

        return payload

    def _request_with_retry(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            self.metrics["requests"] = self.metrics.get("requests", 0) + 1
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_s)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise requests.HTTPError(
                        f"Retryable status {response.status_code} for {url}",
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as exc:
                last_error = exc
                if isinstance(exc, requests.HTTPError):
                    response = exc.response
                    if response is not None and response.status_code not in RETRYABLE_STATUS_CODES:
                        raise
                if attempt >= self.max_retries:
                    break
                self.metrics["retries"] = self.metrics.get("retries", 0) + 1
                backoff_s = (self.backoff_base_s * (2 ** attempt)) + random.uniform(
                    0.0, self.backoff_jitter_s
                )
                LOGGER.warning("Retrying %s (attempt %s/%s) in %.2fs", url, attempt + 1, self.max_retries, backoff_s)
                time.sleep(backoff_s)

        raise RuntimeError(f"Failed GET {url} after {self.max_retries + 1} attempts") from last_error

    @staticmethod
    def _cache_path(cache_dir: Path, endpoint: str, params: dict[str, Any]) -> Path:
        encoded = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return cache_dir / f"{digest}.json"


class PolymarketClients:
    def __init__(
        self,
        *,
        timeout_s: float,
        max_retries: int,
        backoff_base_s: float,
        backoff_jitter_s: float,
        gamma_rate_limit: int,
        clob_rate_limit: int,
        data_rate_limit: int,
        rate_window_s: int,
        metrics: dict[str, int],
    ) -> None:
        self.gamma = JsonHttpClient(
            base_url="https://gamma-api.polymarket.com",
            limiter=TokenBucketLimiter(gamma_rate_limit, rate_window_s),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_jitter_s=backoff_jitter_s,
            metrics=metrics,
        )
        self.clob = JsonHttpClient(
            base_url="https://clob.polymarket.com",
            limiter=TokenBucketLimiter(clob_rate_limit, rate_window_s),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_jitter_s=backoff_jitter_s,
            metrics=metrics,
        )
        self.data = JsonHttpClient(
            base_url="https://data-api.polymarket.com",
            limiter=TokenBucketLimiter(data_rate_limit, rate_window_s),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_jitter_s=backoff_jitter_s,
            metrics=metrics,
        )

    def fetch_events_page(
        self,
        *,
        limit: int,
        offset: int,
        active: bool = True,
        closed: bool = False,
        order: str = "volume_24hr",
        cache_dir: Path | None = None,
        cache_ttl_s: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self.gamma.get_json(
            "/events",
            params={
                "active": str(active).lower(),
                "closed": str(closed).lower(),
                "limit": limit,
                "offset": offset,
                "order": order,
            },
            cache_dir=cache_dir,
            cache_ttl_s=cache_ttl_s,
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("events", "data", "items", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError("Unexpected Gamma /events response shape")

    def fetch_prices_history(
        self,
        *,
        asset_id: str,
        interval: str,
        start_ts: int,
        end_ts: int,
    ) -> list[dict[str, Any]]:
        payload = self.clob.get_json(
            "/prices-history",
            params={
                "market": str(asset_id),
                "interval": interval,
                "startTs": int(start_ts),
                "endTs": int(end_ts),
            },
        )
        if isinstance(payload, dict):
            history = payload.get("history", [])
            return history if isinstance(history, list) else []
        if isinstance(payload, list):
            return payload
        return []

    def fetch_trades_page(
        self,
        *,
        condition_id: str,
        limit: int = 10_000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        payload = self.data.get_json(
            "/trades",
            params={"market": condition_id, "limit": int(limit), "offset": int(offset)},
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("trades", "data", "items", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []
