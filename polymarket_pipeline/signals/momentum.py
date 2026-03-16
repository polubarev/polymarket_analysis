from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import Signal, SignalOutput, _latest_price, _actual_spread, _calibrated_confidence, _safe_float


@dataclass(slots=True)
class VolumeConfirmedMomentumSignal(Signal):
    """Volume-confirmed momentum: trade when price moves on above-average volume.

    Only fires when:
    - Recent price move (return_1d) > 2%
    - Volume is above rolling average (surge confirmation)
    - buy_sell_ratio confirms the direction
    - Spread is tight enough
    """

    volume_surge_threshold: float = 2.0
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
            if record_debug:
                record_debug("missing_price")
            return None

        # Need sufficient price data
        if len(price_history) < 48:
            if record_debug:
                record_debug("insufficient_points")
            return None

        # Check for significant recent price move
        return_1d = _safe_float(features.get("return_1d"))
        if return_1d is None or abs(return_1d) < 0.02:
            if record_debug:
                record_debug("price_move_too_small")
            return None

        # Volume must confirm — latest volume above average
        if volume_history is not None and not volume_history.empty:
            vol = pd.to_numeric(volume_history, errors="coerce").dropna()
            if len(vol) >= 5:
                latest_vol = float(vol.iloc[-1])
                avg_vol = float(vol.mean())
                if avg_vol > 0 and latest_vol < avg_vol * self.volume_surge_threshold:
                    if record_debug:
                        record_debug("volume_below_surge_threshold")
                    return None
            else:
                if record_debug:
                    record_debug("insufficient_volume_data")
                return None
        else:
            if record_debug:
                record_debug("no_volume_data")
            return None

        # buy_sell_ratio must confirm direction
        bsr = _safe_float(features.get("buy_sell_ratio"))
        if bsr is None:
            if record_debug:
                record_debug("missing_buy_sell_ratio")
            return None

        if return_1d > 0 and bsr < 1.3:
            if record_debug:
                record_debug("bsr_contradicts_buy")
            return None
        if return_1d < 0 and bsr > 0.77:
            if record_debug:
                record_debug("bsr_contradicts_sell")
            return None

        # Spread filter
        spread = _actual_spread(features)
        if spread > 0.04:
            if record_debug:
                record_debug("spread_too_wide")
            return None

        direction = "buy" if return_1d > 0 else "sell"
        # Target: continuation of the move
        target_price = min(0.99, current_price + 0.05) if direction == "buy" else max(0.01, current_price - 0.05)

        # Signal strength based on move magnitude and volume surge
        move_strength = min(1.0, abs(return_1d) / 0.05)
        vol_series = pd.to_numeric(volume_history, errors="coerce").dropna()
        vol_ratio = float(vol_series.iloc[-1]) / max(1e-9, float(vol_series.mean()))
        volume_strength = min(1.0, vol_ratio / 5.0)
        signal_strength = move_strength * volume_strength

        confidence = _calibrated_confidence(signal_strength, features)
        edge = abs(return_1d) - spread
        if edge <= 0:
            if record_debug:
                record_debug("no_edge_after_spread")
            return None

        return SignalOutput(
            direction=direction,
            confidence=confidence,
            entry_price=float(current_price),
            target_price=float(target_price),
            edge=float(edge),
            metadata={
                "return_1d": round(return_1d, 4),
                "buy_sell_ratio": round(bsr, 4),
                "volume_ratio": round(vol_ratio, 2),
                "spread": spread,
            },
        )

    def explain(self, output: SignalOutput) -> str:
        ret = output.metadata.get("return_1d", 0)
        vr = output.metadata.get("volume_ratio", 0)
        return f"Price moved {ret:+.1%} on {vr:.1f}x avg volume; momentum {output.direction}."
