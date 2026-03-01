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
                if response.status_code == 429:
                    self.increment_metric("http_429", 1)
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

    @staticmethod
    def _normalize_book_levels(raw_levels: Any) -> list[tuple[float, float]]:
        levels: list[tuple[float, float]] = []
        if not isinstance(raw_levels, list):
            return levels
        for level in raw_levels:
            price: Any = None
            size: Any = None
            if isinstance(level, dict):
                price = level.get("price", level.get("p"))
                size = level.get("size", level.get("s", level.get("quantity", level.get("q"))))
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price, size = level[0], level[1]
            if price is None or size is None:
                continue
            try:
                p = float(price)
                s = float(size)
            except (TypeError, ValueError):
                continue
            if not (p > 0 and s >= 0):
                continue
            levels.append((p, s))
        return levels

    def fetch_order_book(self, *, asset_id: str) -> dict[str, Any]:
        # Endpoint name and parameter name vary across CLOB versions.
        attempts: list[tuple[str, dict[str, Any]]] = [
            ("/book", {"token_id": str(asset_id)}),
            ("/book", {"tokenId": str(asset_id)}),
            ("/book", {"asset_id": str(asset_id)}),
            ("/book", {"market": str(asset_id)}),
            ("/orderbook", {"token_id": str(asset_id)}),
            ("/orderbook", {"tokenId": str(asset_id)}),
            ("/orderbook", {"asset_id": str(asset_id)}),
            ("/orderbook", {"market": str(asset_id)}),
        ]
        for endpoint, params in attempts:
            try:
                payload = self.clob.get_json(endpoint, params=params)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in {400, 404, 422}:
                    continue
                raise

            if not isinstance(payload, dict):
                continue
            bids = self._normalize_book_levels(payload.get("bids", payload.get("buyOrders", payload.get("buy"))))
            asks = self._normalize_book_levels(payload.get("asks", payload.get("sellOrders", payload.get("sell"))))
            if not bids and not asks:
                # Some payloads wrap levels in nested object.
                book = payload.get("book")
                if isinstance(book, dict):
                    bids = self._normalize_book_levels(book.get("bids", book.get("buyOrders", book.get("buy"))))
                    asks = self._normalize_book_levels(book.get("asks", book.get("sellOrders", book.get("sell"))))
            return {"asset_id": str(asset_id), "bids": bids, "asks": asks, "raw": payload}
        return {"asset_id": str(asset_id), "bids": [], "asks": [], "raw": {}}

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

    def fetch_trades_all(
        self,
        *,
        condition_id: str,
        limit: int = 10_000,
        max_pages: int = 100,
        min_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        offset = 0

        for _ in range(max_pages):
            page = self.fetch_trades_page(condition_id=condition_id, limit=limit, offset=offset)
            if not page:
                break
            all_rows.extend(page)
            if len(page) < limit:
                break
            offset += limit

            # If the API is sorted descending by time, we can stop once the
            # page is fully older than min_ts.
            if min_ts is not None:
                page_ts: list[int] = []
                for row in page:
                    if not isinstance(row, dict):
                        continue
                    value = row.get("timestamp", row.get("ts", row.get("time")))
                    try:
                        ts = int(float(value))
                    except (TypeError, ValueError):
                        continue
                    if ts > 10_000_000_000:
                        ts //= 1000
                    page_ts.append(ts)
                if page_ts and max(page_ts) < int(min_ts):
                    break

        return all_rows
