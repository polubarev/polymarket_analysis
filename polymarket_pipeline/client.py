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
from requests.adapters import HTTPAdapter

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
        connection_pool_maxsize: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.limiter = limiter
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_jitter_s = backoff_jitter_s
        self.metrics = metrics
        self._metrics_lock = threading.Lock()
        self.session = requests.Session()
        pool_size = max(1, int(connection_pool_maxsize))
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0, pool_block=True)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def increment_metric(self, key: str, delta: int = 1) -> None:
        with self._metrics_lock:
            self.metrics[key] = self.metrics.get(key, 0) + int(delta)

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
            self.increment_metric("requests", 1)
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
                self.increment_metric("retries", 1)
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
        connection_pool_maxsize: int = 64,
    ) -> None:
        self.gamma = JsonHttpClient(
            base_url="https://gamma-api.polymarket.com",
            limiter=TokenBucketLimiter(gamma_rate_limit, rate_window_s),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_jitter_s=backoff_jitter_s,
            metrics=metrics,
            connection_pool_maxsize=connection_pool_maxsize,
        )
        self.clob = JsonHttpClient(
            base_url="https://clob.polymarket.com",
            limiter=TokenBucketLimiter(clob_rate_limit, rate_window_s),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_jitter_s=backoff_jitter_s,
            metrics=metrics,
            connection_pool_maxsize=connection_pool_maxsize,
        )
        self.data = JsonHttpClient(
            base_url="https://data-api.polymarket.com",
            limiter=TokenBucketLimiter(data_rate_limit, rate_window_s),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            backoff_jitter_s=backoff_jitter_s,
            metrics=metrics,
            connection_pool_maxsize=connection_pool_maxsize,
        )

    @staticmethod
    def _interval_to_fidelity(interval: str) -> int | None:
        # CLOB prices-history fidelity is minute-based.
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "6h": 360,
            "1d": 1_440,
            "1w": 10_080,
        }
        return mapping.get(str(interval).strip().lower())

    def fetch_events_page(
        self,
        *,
        limit: int,
        offset: int,
        active: bool = True,
        closed: bool = False,
        order: str | None = "volume24hr",
        cache_dir: Path | None = None,
        cache_ttl_s: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": int(limit),
            "offset": int(offset),
        }
        if order:
            params["order"] = order

        try:
            payload = self.gamma.get_json(
                "/events",
                params=params,
                cache_dir=cache_dir,
                cache_ttl_s=cache_ttl_s,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 422 and order:
                LOGGER.warning("Gamma rejected /events order=%s; retrying without order", order)
                params.pop("order", None)
                payload = self.gamma.get_json(
                    "/events",
                    params=params,
                    cache_dir=cache_dir,
                    cache_ttl_s=cache_ttl_s,
                )
            else:
                raise
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
        params: dict[str, Any] = {"market": str(asset_id)}
        if start_ts and end_ts:
            params["startTs"] = int(start_ts)
            params["endTs"] = int(end_ts)
            fidelity = self._interval_to_fidelity(interval)
            if fidelity is not None:
                params["fidelity"] = fidelity
            else:
                # If interval is unknown, rely on API-side interval parsing.
                params["interval"] = interval
        else:
            params["interval"] = interval
        try:
            payload = self.clob.get_json("/prices-history", params=params)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 400:
                fallback_interval = str(interval).strip().lower()
                if fallback_interval not in {"1m", "1h", "6h", "1d", "1w", "max"}:
                    fallback_interval = "max"
                fallback_params = {"market": str(asset_id), "interval": fallback_interval}
                try:
                    payload = self.clob.get_json("/prices-history", params=fallback_params)
                except requests.HTTPError as fallback_exc:
                    fallback_status = fallback_exc.response.status_code if fallback_exc.response is not None else None
                    if fallback_status in {400, 404}:
                        self.clob.increment_metric("price_history_http_error", 1)
                        return []
                    raise
            elif status == 404:
                self.clob.increment_metric("price_history_http_error", 1)
                return []
            else:
                raise
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
