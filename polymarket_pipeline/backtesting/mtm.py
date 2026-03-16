"""Mark-to-market backtest engine.

Evaluates signals by tracking actual price movement after signal fires,
rather than waiting for market resolution. This works with active markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MtmConfig:
    horizons_hours: list[int]
    spread_assumption: float = 0.03
    initial_capital: float = 10_000.0
    flat_position_size: float = 100.0


def run_mtm_backtest(
    *,
    signals_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    orderbook_df: pd.DataFrame | None = None,
    config: MtmConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate each signal by looking at price at future horizons.

    Returns a DataFrame with one row per (signal, horizon) and a summary dict.
    """
    if signals_df.empty or price_history_df.empty:
        return pd.DataFrame(), {"error": "no data"}

    # Prepare signals
    sigs = signals_df.copy()
    sigs["asset_id"] = sigs["asset_id"].astype(str)
    sigs["entry_ts"] = pd.to_numeric(sigs["ts"], errors="coerce")
    sigs["entry_price"] = pd.to_numeric(sigs["entry_price"], errors="coerce")
    sigs = sigs.dropna(subset=["entry_ts", "entry_price"])

    # Prepare price history — build lookup structure
    ph = price_history_df[["asset_id", "ts", "price"]].copy()
    ph["asset_id"] = ph["asset_id"].astype(str)
    ph["ts"] = pd.to_numeric(ph["ts"], errors="coerce")
    ph["price"] = pd.to_numeric(ph["price"], errors="coerce")
    ph = ph.dropna().sort_values(["asset_id", "ts"])

    # Spread lookup
    spread_by_asset: dict[str, float] = {}
    if orderbook_df is not None and not orderbook_df.empty and "spread" in orderbook_df.columns:
        ob = orderbook_df.copy()
        ob["asset_id"] = ob["asset_id"].astype(str)
        ob["spread"] = pd.to_numeric(ob["spread"], errors="coerce")
        ob = ob.dropna(subset=["spread"])
        if not ob.empty:
            spread_by_asset = ob.groupby("asset_id")["spread"].median().to_dict()

    # Group price history by asset for fast lookup
    asset_prices: dict[str, pd.DataFrame] = {
        aid: group.reset_index(drop=True)
        for aid, group in ph.groupby("asset_id")
    }

    rows: list[dict[str, Any]] = []
    horizon_seconds = [h * 3600 for h in config.horizons_hours]

    for _, sig in sigs.iterrows():
        asset_id = str(sig["asset_id"])
        if asset_id not in asset_prices:
            continue

        ap = asset_prices[asset_id]
        entry_ts = int(sig["entry_ts"])
        entry_price = float(sig["entry_price"])
        direction = str(sig.get("direction", "buy")).lower()
        signal_name = str(sig["signal_name"])
        market_id = str(sig.get("market_id", ""))
        confidence = float(sig.get("confidence", 0.0))
        edge = float(sig.get("edge", 0.0))

        spread = spread_by_asset.get(asset_id, config.spread_assumption)
        half_spread = max(0.0, spread / 2.0)

        for horizon_h, horizon_s in zip(config.horizons_hours, horizon_seconds):
            target_ts = entry_ts + horizon_s
            # Find closest price at or after target_ts
            future = ap[ap["ts"] >= target_ts]
            if future.empty:
                continue
            exit_row = future.iloc[0]
            exit_price = float(exit_row["price"])
            exit_ts = int(exit_row["ts"])

            # P&L calculation (same logic as resolution-based)
            if direction == "buy":
                entry_cost = entry_price + half_spread
                unit_pnl = (exit_price - half_spread) - entry_cost
            else:
                entry_cost = (1.0 - entry_price) + half_spread
                unit_pnl = ((1.0 - exit_price) - half_spread) - entry_cost

            position_size = config.flat_position_size
            pnl = unit_pnl * position_size
            return_pct = unit_pnl / max(1e-9, entry_cost)

            # Price move (raw, without spread)
            if direction == "buy":
                price_move = exit_price - entry_price
            else:
                price_move = entry_price - exit_price

            rows.append({
                "signal_name": signal_name,
                "asset_id": asset_id,
                "market_id": market_id,
                "direction": direction,
                "confidence": confidence,
                "edge": edge,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "exit_ts": exit_ts,
                "exit_price": exit_price,
                "horizon_hours": horizon_h,
                "price_move": price_move,
                "pnl": pnl,
                "return_pct": return_pct,
                "position_size": position_size,
            })

    results = pd.DataFrame(rows)
    if results.empty:
        return results, {"error": "no signals matched price data"}

    summary = _summarize_mtm(results, config)
    return results, summary


def _summarize_mtm(results: pd.DataFrame, config: MtmConfig) -> dict[str, Any]:
    """Summarize MTM backtest results by signal and horizon."""
    summary: dict[str, Any] = {"by_signal_horizon": {}, "by_horizon": {}, "by_signal": {}}

    for horizon_h in config.horizons_hours:
        h_df = results[results["horizon_hours"] == horizon_h]
        if h_df.empty:
            continue
        summary["by_horizon"][f"{horizon_h}h"] = _stats(h_df)

        for sig_name, sig_df in h_df.groupby("signal_name"):
            key = f"{sig_name}__{horizon_h}h"
            summary["by_signal_horizon"][key] = _stats(sig_df)

    for sig_name, sig_df in results.groupby("signal_name"):
        # Use shortest horizon for per-signal summary
        shortest = min(config.horizons_hours)
        sig_h = sig_df[sig_df["horizon_hours"] == shortest]
        if not sig_h.empty:
            summary["by_signal"][str(sig_name)] = _stats(sig_h)

    # Overall
    shortest = min(config.horizons_hours)
    overall = results[results["horizon_hours"] == shortest]
    if not overall.empty:
        summary["aggregate"] = _stats(overall)

    return summary


def _stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary stats for a group of MTM results."""
    n = len(df)
    pnl = df["pnl"].astype(float)
    price_move = df["price_move"].astype(float)
    correct = (price_move > 0).sum()

    return {
        "trades": n,
        "hit_rate": round(float(correct / n), 4) if n > 0 else 0.0,
        "avg_price_move": round(float(price_move.mean()), 6),
        "median_price_move": round(float(price_move.median()), 6),
        "total_pnl": round(float(pnl.sum()), 2),
        "avg_pnl": round(float(pnl.mean()), 4),
        "win_rate": round(float((pnl > 0).mean()), 4) if n > 0 else 0.0,
        "avg_return_pct": round(float(df["return_pct"].mean()), 4),
        "sharpe": round(_sharpe(pnl), 4),
        "best_trade": round(float(pnl.max()), 2),
        "worst_trade": round(float(pnl.min()), 2),
    }


def _sharpe(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    std = float(pnl.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(pnl.mean() / std * np.sqrt(365.0))
