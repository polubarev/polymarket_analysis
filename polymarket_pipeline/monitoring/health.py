from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _status_rank(status: str) -> int:
    mapping = {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}
    return mapping.get(status, 0)


def _status_from_checks(checks: list[dict[str, Any]]) -> str:
    status = "HEALTHY"
    for check in checks:
        if bool(check.get("passed", True)):
            continue
        severity = str(check.get("severity", "WARNING")).upper()
        if _status_rank(severity) > _status_rank(status):
            status = severity if severity in {"WARNING", "CRITICAL"} else "WARNING"
    return status


def _config_hash(config_payload: dict[str, Any]) -> str:
    encoded = json.dumps(config_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_health_check(
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
    runtime_s: float,
    coverage: dict[str, Any],
    metrics: dict[str, int],
    events_count: int,
    markets_count: int,
    signals_generated: int,
    include_resolved: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    coverage_pct = float(coverage.get("pct_targeted_with_history", 0.0) or 0.0)
    median_points = float(coverage.get("median_points_per_token", 0.0) or 0.0)
    retries = int(metrics.get("retries", 0))
    rate_limited = int(metrics.get("http_429", 0))
    cadence_checked_assets = int(metrics.get("cadence_checked_assets", 0))
    cadence_mismatch_assets = int(metrics.get("cadence_mismatch_assets", 0))
    cadence_mismatch_ratio = (
        float(cadence_mismatch_assets / cadence_checked_assets) if cadence_checked_assets > 0 else 0.0
    )
    pipeline_profile = str(config_payload.get("pipeline_profile", "default"))
    signals_expected = bool(
        config_payload.get("run_signals")
        or config_payload.get("run_backtest")
        or config_payload.get("generate_candidates")
    )

    checks.extend(
        [
            {
                "name": "events_discovered",
                "value": int(events_count),
                "threshold": 100,
                "severity": "CRITICAL",
                "passed": int(events_count) >= 100,
            },
            {
                "name": "price_history_coverage_pct",
                "value": coverage_pct,
                "threshold": 0.5,
                "severity": "WARNING",
                "passed": coverage_pct >= 0.5,
            },
            {
                "name": "median_points_per_series",
                "value": median_points,
                "threshold": 10,
                "severity": "WARNING",
                "passed": median_points >= 10,
            },
            {
                "name": "http_429_count",
                "value": rate_limited,
                "threshold": 50,
                "severity": "WARNING",
                "passed": rate_limited <= 50,
            },
            {
                "name": "http_retry_count",
                "value": retries,
                "threshold": 250,
                "severity": "WARNING",
                "passed": retries <= 250,
            },
            {
                "name": "history_cadence_mismatch_ratio",
                "value": cadence_mismatch_ratio,
                "threshold": "<= 0.2",
                "severity": "CRITICAL",
                "passed": cadence_checked_assets == 0 or cadence_mismatch_ratio <= 0.2,
            },
        ]
    )

    run_history_path = output_dir / "pipeline_runs.parquet"
    runtime_passed = True
    new_markets_passed = True
    if run_history_path.exists():
        try:
            history = pd.read_parquet(run_history_path)
        except Exception:
            history = pd.DataFrame()
        if not history.empty and "duration_s" in history.columns:
            avg_runtime = pd.to_numeric(history["duration_s"], errors="coerce").dropna().tail(20).mean()
            if pd.notna(avg_runtime) and float(avg_runtime) > 0:
                runtime_passed = float(runtime_s) <= float(avg_runtime) * 2.0
        if not history.empty and "markets_count" in history.columns:
            new_markets_passed = int(markets_count) > 0

    checks.append(
        {
            "name": "runtime_vs_rolling_avg",
            "value": float(runtime_s),
            "threshold": "2x rolling average",
            "severity": "WARNING",
            "passed": bool(runtime_passed),
        }
    )
    checks.append(
        {
            "name": "new_markets_discovered",
            "value": int(markets_count),
            "threshold": "> 0",
            "severity": "WARNING",
            "passed": bool(new_markets_passed),
        }
    )

    now_ts = int(datetime.now(timezone.utc).timestamp())
    stale_price_passed = True
    price_path = output_dir / "price_history.parquet"
    if price_path.exists():
        try:
            price_df = pd.read_parquet(price_path, columns=["ingested_at"])
            latest_ingest = pd.to_numeric(price_df["ingested_at"], errors="coerce").max()
            if pd.notna(latest_ingest):
                stale_price_passed = int(latest_ingest) >= (now_ts - 24 * 3600)
        except Exception:
            stale_price_passed = True
    checks.append(
        {
            "name": "price_history_staleness_24h",
            "value": stale_price_passed,
            "threshold": "updated within 24h",
            "severity": "CRITICAL",
            "passed": bool(stale_price_passed),
        }
    )

    resolution_stale_passed = True
    if include_resolved:
        resolutions_path = output_dir / "resolutions.parquet"
        if resolutions_path.exists():
            try:
                resolved_df = pd.read_parquet(resolutions_path, columns=["ingested_at"])
                latest_resolution_ingest = pd.to_numeric(resolved_df["ingested_at"], errors="coerce").max()
                if pd.notna(latest_resolution_ingest):
                    resolution_stale_passed = int(latest_resolution_ingest) >= (now_ts - 7 * 24 * 3600)
            except Exception:
                resolution_stale_passed = True
    checks.append(
        {
            "name": "resolution_staleness_7d",
            "value": resolution_stale_passed,
            "threshold": "updated within 7d",
            "severity": "WARNING",
            "passed": bool(resolution_stale_passed),
        }
    )

    status = _status_from_checks(checks)
    for check in checks:
        if bool(check.get("passed", True)):
            continue
        message = f"{check['name']} failed (value={check['value']}, threshold={check['threshold']})"
        if str(check.get("severity", "")).upper() == "CRITICAL":
            errors.append(message)
        else:
            warnings.append(message)

    payload = {
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pipeline_profile": pipeline_profile,
        "signals_expected": signals_expected,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }
    run_row = {
        "run_id": str(hashlib.sha256(payload["run_ts"].encode("utf-8")).hexdigest()[:24]),
        "run_ts": now_ts,
        "duration_s": float(runtime_s),
        "events_count": int(events_count),
        "markets_count": int(markets_count),
        "price_coverage_pct": coverage_pct,
        "signals_generated": int(signals_generated),
        "signals_expected": bool(signals_expected),
        "pipeline_profile": pipeline_profile,
        "status": status,
        "config_hash": _config_hash(config_payload),
    }
    return payload, run_row
