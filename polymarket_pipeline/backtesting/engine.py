from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from typing import Any

import numpy as np
import pandas as pd

from .metrics import summarize_backtest
from .sizing import choose_position_size


@dataclass(slots=True)
class BacktestConfig:
    start_date: str | None
    end_date: str | None
    initial_capital: float
    spread_assumption: float
    max_positions: int
    stop_loss: float
    timeout_days: int
    sizing_mode: str
    kelly_fraction: float
    max_position_pct: float
    min_position_size: float
    flat_position_size: float


def _to_ts(date_text: str | None) -> int | None:
    if not date_text:
        return None
    parsed = pd.to_datetime(date_text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.timestamp())


def _resolution_yes_value(outcome: Any) -> float | None:
    if outcome is None:
        return None
    text = str(outcome).strip().lower()
    if text == "yes":
        return 1.0
    if text == "no":
        return 0.0
    return None


def run_backtest(
    *,
    signals_df: pd.DataFrame,
    resolutions_df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "backtest_id",
        "signal_name",
        "asset_id",
        "market_id",
        "direction",
        "entry_ts",
        "entry_price",
        "entry_cost",
        "exit_ts",
        "exit_price",
        "exit_reason",
        "pnl",
        "return_pct",
        "hold_days",
        "kelly_fraction",
        "position_size",
        "bankroll_at_entry",
    ]
    if signals_df.empty or resolutions_df.empty:
        return pd.DataFrame(columns=columns), summarize_backtest(pd.DataFrame(columns=columns))

    start_ts = _to_ts(config.start_date)
    end_ts = _to_ts(config.end_date)

    signals = signals_df.copy()
    signals["asset_id"] = signals["asset_id"].astype(str)
    signals["market_id"] = signals["market_id"].astype(str)
    signals["signal_name"] = signals["signal_name"].astype(str)
    signals["entry_ts"] = pd.to_numeric(signals["ts"], errors="coerce")
    signals["entry_price"] = pd.to_numeric(signals["entry_price"], errors="coerce")
    signals["confidence"] = pd.to_numeric(signals["confidence"], errors="coerce").fillna(0.0)
    signals = signals.dropna(subset=["entry_ts", "entry_price"])
    if start_ts is not None:
        signals = signals[signals["entry_ts"] >= int(start_ts)]
    if end_ts is not None:
        signals = signals[signals["entry_ts"] <= int(end_ts)]
    if signals.empty:
        return pd.DataFrame(columns=columns), summarize_backtest(pd.DataFrame(columns=columns))

    resolutions = resolutions_df.copy()
    resolutions["market_id"] = resolutions["market_id"].astype(str)
    resolutions["resolution_ts"] = pd.to_numeric(resolutions["resolution_ts"], errors="coerce")
    resolution_keep = resolutions[["market_id", "resolution_outcome", "resolution_ts"]].drop_duplicates("market_id", keep="last")
    signals = signals.merge(resolution_keep, on="market_id", how="inner")
    if signals.empty:
        return pd.DataFrame(columns=columns), summarize_backtest(pd.DataFrame(columns=columns))

    valid_outcome = signals["resolution_outcome"].isin(["Yes", "No"])
    signals = signals.loc[valid_outcome].copy()
    if signals.empty:
        return pd.DataFrame(columns=columns), summarize_backtest(pd.DataFrame(columns=columns))

    # Keep one position per market/signal for baseline backtest.
    signals = signals.sort_values("entry_ts").groupby(["market_id", "signal_name"], as_index=False).first()
    signals = signals.sort_values("entry_ts").reset_index(drop=True)

    spread_by_asset: dict[str, float] = {}
    if orderbook_df is not None and not orderbook_df.empty and "spread" in orderbook_df.columns:
        ob = orderbook_df.copy()
        ob["asset_id"] = ob["asset_id"].astype(str)
        ob["spread"] = pd.to_numeric(ob["spread"], errors="coerce")
        ob = ob.dropna(subset=["spread"])
        if not ob.empty:
            spread_by_asset = ob.groupby("asset_id")["spread"].median().to_dict()

    bankroll = float(config.initial_capital)
    active_market_ids: set[str] = set()
    active_positions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    backtest_id = str(uuid.uuid4())

    for _, signal in signals.iterrows():
        entry_ts = int(float(signal["entry_ts"]))
        market_id = str(signal["market_id"])
        asset_id = str(signal["asset_id"])
        signal_name = str(signal["signal_name"])

        # Release positions that should have exited by this entry time.
        still_active: list[dict[str, Any]] = []
        for open_pos in active_positions:
            if int(open_pos["exit_ts"]) <= entry_ts:
                bankroll += float(open_pos["pnl"])
                active_market_ids.discard(str(open_pos["market_id"]))
            else:
                still_active.append(open_pos)
        active_positions = still_active

        if len(active_positions) >= int(config.max_positions):
            continue
        if market_id in active_market_ids:
            continue

        direction = str(signal.get("direction", "buy")).lower()
        if direction not in {"buy", "sell"}:
            continue
        entry_price = float(signal["entry_price"])
        confidence = float(signal.get("confidence", 0.0))
        spread = float(spread_by_asset.get(asset_id, float(config.spread_assumption)))
        half_spread = max(0.0, spread / 2.0)

        sizing = choose_position_size(
            sizing_mode=config.sizing_mode,
            bankroll=bankroll,
            confidence=confidence,
            entry_price=entry_price if direction == "buy" else (1.0 - entry_price),
            flat_position_size=float(config.flat_position_size),
            fractional_kelly=float(config.kelly_fraction),
            max_position_pct=float(config.max_position_pct),
            min_position_size=float(config.min_position_size),
        )
        if sizing.position_size <= 0:
            continue

        resolution_ts = signal.get("resolution_ts")
        if pd.isna(resolution_ts):
            resolution_ts = entry_ts + int(config.timeout_days) * 24 * 3600
            exit_reason = "timeout"
        else:
            resolution_ts = int(float(resolution_ts))
            timeout_ts = entry_ts + int(config.timeout_days) * 24 * 3600
            if resolution_ts > timeout_ts:
                resolution_ts = timeout_ts
                exit_reason = "timeout"
            else:
                exit_reason = "resolution"

        outcome_yes = _resolution_yes_value(signal.get("resolution_outcome"))
        if outcome_yes is None:
            continue
        exit_price = float(outcome_yes)

        if direction == "buy":
            entry_cost = entry_price + half_spread
            unit_pnl = (exit_price - half_spread) - entry_cost
        else:
            entry_cost = (1.0 - entry_price) + half_spread
            unit_pnl = ((1.0 - exit_price) - half_spread) - entry_cost

        pnl = float(unit_pnl * sizing.position_size)
        return_pct = float(unit_pnl / max(1e-9, entry_cost))
        hold_days = float((resolution_ts - entry_ts) / 86_400.0)

        row = {
            "backtest_id": backtest_id,
            "signal_name": signal_name,
            "asset_id": asset_id,
            "market_id": market_id,
            "direction": "long" if direction == "buy" else "short",
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "entry_cost": float(entry_cost),
            "exit_ts": int(resolution_ts),
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "pnl": pnl,
            "return_pct": return_pct,
            "hold_days": hold_days,
            "kelly_fraction": float(sizing.kelly_fraction_raw),
            "position_size": float(sizing.position_size),
            "bankroll_at_entry": float(bankroll),
        }
        rows.append(row)
        active_positions.append(row)
        active_market_ids.add(market_id)

    results = pd.DataFrame(rows, columns=columns)
    summary = summarize_backtest(results)
    summary["backtest_id"] = backtest_id
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["config"] = {
        "start_date": config.start_date,
        "end_date": config.end_date,
        "initial_capital": config.initial_capital,
        "spread_assumption": config.spread_assumption,
        "max_positions": config.max_positions,
        "stop_loss": config.stop_loss,
        "timeout_days": config.timeout_days,
        "sizing_mode": config.sizing_mode,
        "kelly_fraction": config.kelly_fraction,
        "max_position_pct": config.max_position_pct,
        "min_position_size": config.min_position_size,
        "flat_position_size": config.flat_position_size,
    }
    return results, summary
