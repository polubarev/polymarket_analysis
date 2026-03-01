from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    return float(drawdown.min())


def _sharpe_from_daily(daily_pnl: pd.Series) -> float:
    if daily_pnl.empty:
        return 0.0
    std = float(daily_pnl.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(daily_pnl.mean() / std * np.sqrt(365.0))


def summarize_backtest(results_df: pd.DataFrame) -> dict[str, Any]:
    if results_df.empty:
        return {"by_signal": {}, "aggregate": {"total_trades": 0, "total_pnl": 0.0}}

    working = results_df.copy()
    working["pnl"] = pd.to_numeric(working["pnl"], errors="coerce").fillna(0.0)
    working["return_pct"] = pd.to_numeric(working["return_pct"], errors="coerce")
    working["hold_days"] = pd.to_numeric(working["hold_days"], errors="coerce")
    working["exit_ts"] = pd.to_numeric(working["exit_ts"], errors="coerce")
    working["exit_date"] = pd.to_datetime(working["exit_ts"], unit="s", utc=True, errors="coerce").dt.floor("1d")

    def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
        pnl = frame["pnl"].astype(float)
        total_trades = int(len(frame))
        win_rate = float((pnl > 0).mean()) if total_trades else 0.0
        avg_pnl = float(pnl.mean()) if total_trades else 0.0
        median_pnl = float(pnl.median()) if total_trades else 0.0
        total_pnl = float(pnl.sum())
        profits = float(pnl[pnl > 0].sum())
        losses = float(pnl[pnl < 0].sum())
        profit_factor = float(profits / abs(losses)) if losses < 0 else float("inf") if profits > 0 else 0.0
        avg_hold_days = float(frame["hold_days"].dropna().mean()) if total_trades else 0.0
        daily = frame.groupby("exit_date", dropna=False)["pnl"].sum().sort_index()
        equity = daily.cumsum()
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "total_pnl": total_pnl,
            "max_drawdown": _max_drawdown(equity),
            "sharpe_ratio": _sharpe_from_daily(daily),
            "avg_hold_days": avg_hold_days,
            "profit_factor": profit_factor,
        }

    by_signal = {
        str(signal): summarize_frame(group)
        for signal, group in working.groupby("signal_name", dropna=False)
    }
    aggregate = summarize_frame(working)
    return {"by_signal": by_signal, "aggregate": aggregate}
