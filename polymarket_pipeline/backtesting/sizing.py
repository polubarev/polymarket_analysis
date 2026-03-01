from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SizingDecision:
    kelly_fraction_raw: float
    position_size: float


def kelly_fraction_for_binary(*, probability: float, entry_price: float) -> float:
    p = min(1.0, max(0.0, float(probability)))
    c = min(0.999999, max(0.000001, float(entry_price)))
    b = (1.0 - c) / c
    if b <= 0:
        return 0.0
    kelly = (p * b - (1.0 - p)) / b
    return min(1.0, max(0.0, float(kelly)))


def choose_position_size(
    *,
    sizing_mode: str,
    bankroll: float,
    confidence: float,
    entry_price: float,
    flat_position_size: float,
    fractional_kelly: float,
    max_position_pct: float,
    min_position_size: float,
) -> SizingDecision:
    cash = max(0.0, float(bankroll))
    if cash <= 0:
        return SizingDecision(kelly_fraction_raw=0.0, position_size=0.0)

    if str(sizing_mode).lower() == "kelly":
        raw = kelly_fraction_for_binary(probability=float(confidence), entry_price=float(entry_price))
        if raw <= 0:
            return SizingDecision(kelly_fraction_raw=raw, position_size=0.0)
        position_size = cash * raw * float(fractional_kelly)
    else:
        raw = 0.0
        position_size = float(flat_position_size)

    max_size = cash * float(max_position_pct)
    if max_size > 0:
        position_size = min(position_size, max_size)
    if position_size < float(min_position_size):
        return SizingDecision(kelly_fraction_raw=raw, position_size=0.0)
    return SizingDecision(kelly_fraction_raw=raw, position_size=float(position_size))
