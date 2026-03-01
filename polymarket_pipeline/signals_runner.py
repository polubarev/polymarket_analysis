from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
import pandas as pd

from .backtesting.sizing import choose_position_size
from .signals import Signal, build_registry


def _build_asset_series(price_history_df: pd.DataFrame) -> dict[str, pd.Series]:
    if price_history_df.empty:
        return {}
    working = price_history_df[["asset_id", "ts", "price"]].copy()
    working["asset_id"] = working["asset_id"].astype(str)
    working["ts"] = pd.to_numeric(working["ts"], errors="coerce")
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.dropna(subset=["asset_id", "ts", "price"])
    out: dict[str, pd.Series] = {}
    for asset_id, group in working.groupby("asset_id", dropna=False):
        series = pd.Series(group["price"].to_numpy(dtype=float), index=group["ts"].astype("int64").to_numpy()).sort_index()
        out[str(asset_id)] = series
    return out


def _build_asset_volume_series(volume_bars_df: pd.DataFrame) -> dict[str, pd.Series]:
    if volume_bars_df is None or volume_bars_df.empty:
        return {}
    working = volume_bars_df[["asset_id", "ts", "volume"]].copy()
    working["asset_id"] = working["asset_id"].astype(str)
    working["ts"] = pd.to_numeric(working["ts"], errors="coerce")
    working["volume"] = pd.to_numeric(working["volume"], errors="coerce")
    working = working.dropna(subset=["asset_id", "ts", "volume"])
    out: dict[str, pd.Series] = {}
    for asset_id, group in working.groupby("asset_id", dropna=False):
        series = pd.Series(group["volume"].to_numpy(dtype=float), index=group["ts"].astype("int64").to_numpy()).sort_index()
        out[str(asset_id)] = series
    return out


def compute_base_rate_by_tag(
    *,
    resolutions_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> dict[str, float]:
    if resolutions_df.empty or markets_df.empty:
        return {}
    valid = resolutions_df[resolutions_df["resolution_outcome"].isin(["Yes", "No"])].copy()
    if valid.empty:
        return {}
    valid["market_id"] = valid["market_id"].astype(str)
    valid["resolved_yes"] = (valid["resolution_outcome"] == "Yes").astype(float)

    event_tags = pd.DataFrame(columns=["event_id", "primary_tag"])
    if not events_df.empty and "event_id" in events_df.columns:
        event_tags = events_df[["event_id", "tags"]].copy()
        event_tags["primary_tag"] = (
            event_tags["tags"].astype(str).str.extract(r'^\["?([^",\]]+)')[0].fillna("unknown")
        )

    market_tags = markets_df[["market_id", "event_id"]].copy()
    market_tags["market_id"] = market_tags["market_id"].astype(str)
    if not event_tags.empty:
        market_tags = market_tags.merge(event_tags[["event_id", "primary_tag"]], on="event_id", how="left")
    if "primary_tag" not in market_tags.columns:
        market_tags["primary_tag"] = "unknown"
    market_tags["primary_tag"] = market_tags["primary_tag"].fillna("unknown").astype(str).str.lower()

    tagged = valid.merge(market_tags[["market_id", "primary_tag"]], on="market_id", how="left")
    tagged["primary_tag"] = tagged["primary_tag"].fillna("unknown")
    summary = tagged.groupby("primary_tag", dropna=False)["resolved_yes"].mean()
    return {str(k): float(v) for k, v in summary.items()}


def run_signal_generation(
    *,
    config: Any,
    features_df: pd.DataFrame,
    target_tokens_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    events_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    volume_bars_df: pd.DataFrame | None,
    resolutions_df: pd.DataFrame | None,
    as_of_ts: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Signal]]:
    columns = [
        "signal_name",
        "asset_id",
        "market_id",
        "ts",
        "direction",
        "confidence",
        "entry_price",
        "target_price",
        "edge",
        "metadata_json",
    ]
    if features_df.empty:
        return pd.DataFrame(columns=columns), {}

    base_rates = compute_base_rate_by_tag(
        resolutions_df=resolutions_df if resolutions_df is not None else pd.DataFrame(),
        markets_df=markets_df,
        events_df=events_df,
    )
    registry = build_registry(config, base_rate_by_tag=base_rates)
    registry_by_name: dict[str, Signal] = {}
    for _, signal in registry.items():
        registry_by_name[signal.name] = signal
    registry_by_name.update(registry)
    requested = [value.strip().lower() for value in getattr(config, "active_signals", ["all"]) if value]
    if not requested or "all" in requested:
        active_names = list(registry.keys())
    else:
        active_names = [name for name in requested if name in registry]
    if not active_names:
        return pd.DataFrame(columns=columns), registry_by_name

    token_market = target_tokens_df[["asset_id", "market_id"]].drop_duplicates().copy()
    token_market["asset_id"] = token_market["asset_id"].astype(str)
    token_market["market_id"] = token_market["market_id"].astype(str)

    market_fields = ["market_id", "question", "liquidity", "resolved", "resolution_outcome", "resolution_ts"]
    if "event_id" in markets_df.columns:
        market_fields.append("event_id")
    market_meta = markets_df[[col for col in market_fields if col in markets_df.columns]].copy()
    market_meta["market_id"] = market_meta["market_id"].astype(str)

    event_meta = pd.DataFrame(columns=["event_id", "primary_tag"])
    if not events_df.empty and "event_id" in events_df.columns:
        event_meta = events_df[["event_id", "tags"]].copy()
        event_meta["primary_tag"] = (
            event_meta["tags"].astype(str).str.extract(r'^\["?([^",\]]+)')[0].fillna("unknown")
        )
    if not event_meta.empty and "event_id" in market_meta.columns:
        market_meta = market_meta.merge(event_meta[["event_id", "primary_tag"]], on="event_id", how="left")
    if "primary_tag" not in market_meta.columns:
        market_meta["primary_tag"] = "unknown"
    market_meta["primary_tag"] = market_meta["primary_tag"].fillna("unknown")

    features = features_df.copy()
    features["asset_id"] = features["asset_id"].astype(str)
    merged = token_market.merge(features, on="asset_id", how="inner").merge(market_meta, on="market_id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=columns), registry_by_name

    series_map = _build_asset_series(price_history_df)
    volume_map = _build_asset_volume_series(volume_bars_df if volume_bars_df is not None else pd.DataFrame())
    now_ts = int(as_of_ts) if as_of_ts is not None else int(datetime.now(timezone.utc).timestamp())

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        asset_id = str(row["asset_id"])
        market_id = str(row["market_id"])
        price_series = series_map.get(asset_id)
        if price_series is None or price_series.empty:
            continue
        volume_series = volume_map.get(asset_id)
        market_row = row.to_dict()
        feature_row = row.to_dict()

        for signal_name in active_names:
            signal = registry[signal_name]
            output = signal.compute(
                asset_id=asset_id,
                market_row=market_row,
                features=feature_row,
                price_history=price_series,
                volume_history=volume_series,
                as_of_ts=now_ts,
            )
            if output is None:
                continue
            rows.append(output.as_record(signal_name=signal.name, asset_id=asset_id, market_id=market_id, ts=now_ts))

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out, registry
    out["asset_id"] = out["asset_id"].astype(str)
    out["market_id"] = out["market_id"].astype(str)
    out = out.sort_values(["ts", "signal_name", "asset_id"]).reset_index(drop=True)
    return out, registry_by_name


def generate_trade_candidates(
    *,
    config: Any,
    signals_df: pd.DataFrame,
    signal_registry: dict[str, Signal],
    markets_df: pd.DataFrame,
    features_df: pd.DataFrame,
    orderbook_df: pd.DataFrame | None,
    quality_df: pd.DataFrame | None,
    clusters_df: pd.DataFrame | None,
    bankroll: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if signals_df.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bankroll": float(bankroll),
            "open_positions": 0,
            "candidates": [],
        }
        return payload, pd.DataFrame()

    signals = signals_df.copy()
    signals["confidence"] = pd.to_numeric(signals["confidence"], errors="coerce")
    signals["edge"] = pd.to_numeric(signals["edge"], errors="coerce")
    signals["entry_price"] = pd.to_numeric(signals["entry_price"], errors="coerce")
    signals = signals.dropna(subset=["confidence", "edge", "entry_price"])
    signals = signals[signals["confidence"] >= float(config.min_confidence)]
    signals = signals[signals["edge"] >= float(config.min_edge)]
    if signals.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bankroll": float(bankroll),
            "open_positions": 0,
            "candidates": [],
        }
        return payload, pd.DataFrame()

    market_meta = markets_df[["market_id", "question", "liquidity"]].copy() if not markets_df.empty else pd.DataFrame()
    if not market_meta.empty:
        market_meta["market_id"] = market_meta["market_id"].astype(str)
        signals = signals.merge(market_meta, on="market_id", how="left")

    feature_meta = features_df[["asset_id", "days_to_resolution", "num_points"]].copy() if not features_df.empty else pd.DataFrame()
    if not feature_meta.empty:
        feature_meta["asset_id"] = feature_meta["asset_id"].astype(str)
        signals = signals.merge(feature_meta, on="asset_id", how="left")

    if quality_df is not None and not quality_df.empty:
        q = quality_df[["asset_id", "quality_pass"]].copy()
        q["asset_id"] = q["asset_id"].astype(str)
        signals = signals.merge(q, on="asset_id", how="left")
    if "quality_pass" in signals.columns:
        signals = signals[signals["quality_pass"].fillna(False)]

    if orderbook_df is not None and not orderbook_df.empty and "asset_id" in orderbook_df.columns:
        ob = orderbook_df.copy()
        ob["asset_id"] = ob["asset_id"].astype(str)
        fields = [col for col in ["asset_id", "spread", "spread_pct"] if col in ob.columns]
        ob = ob[fields]
        ob = ob.groupby("asset_id", as_index=False).last()
        signals = signals.merge(ob, on="asset_id", how="left")
        if "spread_pct" in signals.columns:
            spread_pct = pd.to_numeric(signals["spread_pct"], errors="coerce")
            signals = signals[(spread_pct.isna()) | (spread_pct <= 0.10)]

    if clusters_df is not None and not clusters_df.empty and "asset_id" in clusters_df.columns:
        cluster_map = clusters_df[["asset_id", "cluster_id"]].drop_duplicates("asset_id", keep="last").copy()
        cluster_map["asset_id"] = cluster_map["asset_id"].astype(str)
        signals = signals.merge(cluster_map, on="asset_id", how="left")

    if "num_points" in signals.columns:
        signals = signals[pd.to_numeric(signals["num_points"], errors="coerce").fillna(0) >= 7]

    if signals.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bankroll": float(bankroll),
            "open_positions": 0,
            "candidates": [],
        }
        return payload, pd.DataFrame()

    candidate_rows: list[dict[str, Any]] = []
    for _, row in signals.iterrows():
        signal_name = str(row.get("signal_name", ""))
        signal_impl = signal_registry.get(signal_name.replace("_signal", ""), signal_registry.get(signal_name))
        metadata = {}
        try:
            metadata = json.loads(str(row.get("metadata_json", "{}")))
        except json.JSONDecodeError:
            metadata = {}
        direction = str(row.get("direction", "buy")).lower()
        confidence = float(row["confidence"])
        entry_price = float(row["entry_price"])
        sizing = choose_position_size(
            sizing_mode="kelly",
            bankroll=float(bankroll),
            confidence=confidence,
            entry_price=entry_price if direction == "buy" else (1.0 - entry_price),
            flat_position_size=float(getattr(config, "flat_position_size", 100.0)),
            fractional_kelly=float(config.kelly_fraction),
            max_position_pct=float(config.max_position_pct),
            min_position_size=float(config.min_position_size),
        )
        if sizing.position_size <= 0:
            continue
        expected_value = float(row["edge"]) * float(confidence)
        output_stub = type("Obj", (), {"direction": direction, "edge": float(row["edge"]), "metadata": metadata})()
        reasoning = (
            signal_impl.explain(output_stub) if signal_impl is not None and hasattr(signal_impl, "explain") else ""
        )
        candidate_rows.append(
            {
                "market_id": str(row.get("market_id", "")),
                "asset_id": str(row.get("asset_id", "")),
                "question": row.get("question"),
                "current_price": entry_price,
                "signal_name": signal_name,
                "direction": direction,
                "confidence": confidence,
                "edge": float(row["edge"]),
                "expected_value": expected_value,
                "suggested_size": float(sizing.position_size),
                "kelly_fraction": float(sizing.kelly_fraction_raw),
                "spread": float(row["spread"]) if "spread" in row and pd.notna(row["spread"]) else np.nan,
                "spread_pct": float(row["spread_pct"]) if "spread_pct" in row and pd.notna(row["spread_pct"]) else np.nan,
                "liquidity": float(row["liquidity"]) if "liquidity" in row and pd.notna(row["liquidity"]) else np.nan,
                "days_to_resolution": float(row["days_to_resolution"])
                if "days_to_resolution" in row and pd.notna(row["days_to_resolution"])
                else np.nan,
                "cluster_id": int(row["cluster_id"]) if "cluster_id" in row and pd.notna(row["cluster_id"]) else None,
                "reasoning": reasoning,
                "ts": int(row.get("ts", int(datetime.now(timezone.utc).timestamp()))),
            }
        )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bankroll": float(bankroll),
            "open_positions": 0,
            "candidates": [],
        }
        return payload, candidates

    candidates = candidates.sort_values(["expected_value", "edge", "confidence"], ascending=False).head(int(config.max_candidates))
    candidates = candidates.reset_index(drop=True)
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates["run_date"] = datetime.now(timezone.utc).date().isoformat()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bankroll": float(bankroll),
        "open_positions": 0,
        "candidates": candidates[
            [
                "rank",
                "market_id",
                "question",
                "current_price",
                "signal_name",
                "direction",
                "confidence",
                "edge",
                "expected_value",
                "suggested_size",
                "kelly_fraction",
                "spread",
                "liquidity",
                "days_to_resolution",
                "cluster_id",
                "reasoning",
            ]
        ].to_dict(orient="records"),
    }
    return payload, candidates
