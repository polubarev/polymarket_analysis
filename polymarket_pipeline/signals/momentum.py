from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import Signal, SignalOutput, _latest_price


@dataclass(slots=True)
class MomentumBreakoutSignal(Signal):
    lookback_days: int = 30
    name: str = "momentum_breakout"

    def compute(
        self,
        *,
        asset_id: str,
        market_row: dict[str, Any],
        features: dict[str, Any],
        price_history: pd.Series,
        volume_history: pd.Series | None,
        as_of_ts: int,
        record_debug: Callable[[str], None] | None = None,
    ) -> SignalOutput | None:
        current_price = _latest_price(price_history)
        if current_price is None:
            if record_debug is not None:
                record_debug("missing_price")
            return None
        if len(price_history) < int(max(24, self.lookback_days * 24)):
            if record_debug is not None:
                record_debug("insufficient_points")
            return None

        lookback_points = int(max(24, self.lookback_days * 24))
        window = pd.to_numeric(price_history.tail(lookback_points), errors="coerce").dropna()
        if len(window) < 24:
            if record_debug is not None:
                record_debug("insufficient_clean_points")
            return None
        current = float(window.iloc[-1])
        prior = window.iloc[:-1]
        if prior.empty:
            if record_debug is not None:
                record_debug("insufficient_clean_points")
            return None
        breakout_up = current > float(prior.max())
        breakout_down = current < float(prior.min())
        if not (breakout_up or breakout_down):
            if record_debug is not None:
                record_debug("no_breakout")
            return None

        if volume_history is not None and not volume_history.empty:
            vol = pd.to_numeric(volume_history, errors="coerce").dropna()
            if len(vol) >= 5:
                if float(vol.iloc[-1]) <= float(vol.mean()):
                    if record_debug is not None:
                        record_debug("volume_not_confirming")
                    return None

        direction = "buy" if breakout_up else "sell"
        confidence = 0.65
        target_price = min(1.0, current + 0.10) if breakout_up else max(0.0, current - 0.10)
        edge = abs(target_price - current)
        return SignalOutput(
            direction=direction,
            confidence=float(confidence),
            entry_price=current,
            target_price=float(target_price),
            edge=float(edge),
            metadata={
                "lookback_days": int(self.lookback_days),
                "breakout_up": bool(breakout_up),
                "breakout_down": bool(breakout_down),
            },
        )

    def explain(self, output: SignalOutput) -> str:
        return (
            f"Price broke {'up' if output.direction == 'buy' else 'down'} beyond {self.lookback_days}d range "
            f"with confirming momentum."
        )
