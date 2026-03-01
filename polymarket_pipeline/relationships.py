from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _latest_prices(price_history_df: pd.DataFrame) -> pd.DataFrame:
    if price_history_df.empty:
        return pd.DataFrame(columns=["asset_id", "price", "ts"])
    working = price_history_df[["asset_id", "price", "ts"]].copy()
    working["asset_id"] = working["asset_id"].astype(str)
    working["ts"] = pd.to_numeric(working["ts"], errors="coerce")
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.dropna(subset=["asset_id", "ts", "price"])
    if working.empty:
        return pd.DataFrame(columns=["asset_id", "price", "ts"])
    latest_idx = working.groupby("asset_id")["ts"].idxmax()
    return working.loc[latest_idx, ["asset_id", "price", "ts"]].reset_index(drop=True)


def _anchor_tokens(tokens_df: pd.DataFrame) -> pd.DataFrame:
    if tokens_df.empty:
        return pd.DataFrame(columns=["market_id", "asset_id"])
    working = tokens_df[["market_id", "asset_id", "outcome"]].copy()
    working["asset_id"] = working["asset_id"].astype(str)
    working["outcome_norm"] = working["outcome"].fillna("").astype(str).str.strip().str.lower()
    working["is_anchor"] = working["outcome_norm"].isin({"yes", "true"})
    working = working.sort_values(["market_id", "is_anchor"], ascending=[True, False])
    return working.groupby("market_id", as_index=False).first()[["market_id", "asset_id"]]


def _daily_series(price_history_df: pd.DataFrame, asset_id: str) -> pd.Series:
    frame = price_history_df.loc[price_history_df["asset_id"].astype(str) == str(asset_id), ["ts", "price"]].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["ts", "price"])
    if frame.empty:
        return pd.Series(dtype=float)
    frame["dt"] = pd.to_datetime(frame["ts"].astype("int64"), unit="s", utc=True).dt.floor("1d")
    daily = frame.groupby("dt", dropna=False)["price"].last().sort_index()
    return daily


def detect_market_relationships(
    *,
    events_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    correlation_threshold: float = 0.5,
    min_overlap_days: int = 30,
) -> pd.DataFrame:
    columns = [
        "asset_id_a",
        "asset_id_b",
        "relationship_type",
        "correlation",
        "overround",
        "overlap_days",
    ]
    if tokens_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    latest_prices = _latest_prices(price_history_df)
    latest_map = dict(zip(latest_prices["asset_id"].astype(str), latest_prices["price"].astype(float)))

    # Intra-event/market overround on latest available prices.
    token_prices = tokens_df[["market_id", "asset_id"]].copy()
    token_prices["asset_id"] = token_prices["asset_id"].astype(str)
    token_prices["price"] = token_prices["asset_id"].map(latest_map)
    for market_id, group in token_prices.groupby("market_id", dropna=False):
        valid = group.dropna(subset=["price"])
        if len(valid) < 2:
            continue
        assets = valid["asset_id"].astype(str).tolist()
        overround = float(valid["price"].sum() - 1.0)
        for a, b in combinations(sorted(set(assets)), 2):
            rows.append(
                {
                    "asset_id_a": a,
                    "asset_id_b": b,
                    "relationship_type": "intra_event",
                    "correlation": np.nan,
                    "overround": overround,
                    "overlap_days": np.nan,
                }
            )

    # Cross-event correlation on anchor token daily closes.
    anchors = _anchor_tokens(tokens_df)
    if anchors.empty:
        out = pd.DataFrame(rows, columns=columns)
        if out.empty:
            return out
        return out.drop_duplicates(subset=["asset_id_a", "asset_id_b", "relationship_type"], keep="last")

    market_meta = markets_df[["market_id", "event_id"]].copy() if not markets_df.empty else pd.DataFrame(columns=["market_id", "event_id"])
    if not events_df.empty and "tags" in events_df.columns:
        event_tags = events_df[["event_id", "tags"]].copy()
        event_tags["primary_tag"] = (
            event_tags["tags"]
            .astype(str)
            .str.extract(r'^\["?([^",\]]+)')[0]
            .fillna("unknown")
        )
        market_meta = market_meta.merge(event_tags[["event_id", "primary_tag"]], on="event_id", how="left")
    if "primary_tag" not in market_meta.columns:
        market_meta["primary_tag"] = "unknown"
    market_meta["primary_tag"] = market_meta["primary_tag"].fillna("unknown")
    anchors = anchors.merge(market_meta[["market_id", "primary_tag"]], on="market_id", how="left")
    anchors["primary_tag"] = anchors["primary_tag"].fillna("unknown")

    series_map: dict[str, pd.Series] = {}
    for asset_id in anchors["asset_id"].astype(str).drop_duplicates().tolist():
        series = _daily_series(price_history_df, asset_id)
        if len(series) >= int(min_overlap_days):
            series_map[asset_id] = series

    # Pre-filter pairs by primary_tag to control O(n^2) blowup.
    for _, tag_group in anchors.groupby("primary_tag", dropna=False):
        asset_ids = [asset for asset in tag_group["asset_id"].astype(str).tolist() if asset in series_map]
        if len(asset_ids) < 2:
            continue
        for asset_a, asset_b in combinations(sorted(set(asset_ids)), 2):
            joined = pd.concat([series_map[asset_a], series_map[asset_b]], axis=1, join="inner")
            joined.columns = ["a", "b"]
            joined = joined.dropna()
            overlap_days = int(len(joined))
            if overlap_days < int(min_overlap_days):
                continue
            corr = float(joined["a"].corr(joined["b"]))
            if not np.isfinite(corr):
                continue
            if abs(corr) < float(correlation_threshold):
                continue
            rows.append(
                {
                    "asset_id_a": asset_a,
                    "asset_id_b": asset_b,
                    "relationship_type": "cross_event",
                    "correlation": corr,
                    "overround": np.nan,
                    "overlap_days": overlap_days,
                }
            )

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["asset_id_a", "asset_id_b", "relationship_type"], keep="last")
