from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Position:
    asset_id: str
    market_id: str
    signal_name: str
    direction: str
    entry_ts: int
    entry_price: float
    entry_cost: float
    size: float
    exit_ts: int | None
    exit_price: float | None
    exit_reason: str
    pnl: float | None
    return_pct: float | None
    kelly_fraction: float | None = None
    bankroll_at_entry: float | None = None
