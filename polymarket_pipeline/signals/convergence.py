from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import Signal, SignalOutput, _latest_price, _actual_spread, _calibrated_confidence


@dataclass(slots=True)
class ResolutionConvergenceSignal(Signal):
    """Late-stage convergence: bet on near-certain outcomes close to resolution.

    Only fires when:
    - Market is within 2 days of resolution
    - Price already > 0.85 (buy → 1.0) or < 0.15 (sell → 0.0)
    - Slope confirms the direction
    - Spread is tight enough to trade
    """

    days_threshold: int = 2
    name: str = "resolution_convergence"

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

        # Must be near an extreme — high conviction already priced in
        if not (current_price > 0.85 or current_price < 0.15):
            if record_debug:
                record_debug("price_not_extreme")
            return None

        # Must be close to resolution
        days_to_resolution = features.get("days_to_resolution")
        try:
            dtr = float(days_to_resolution)
        except (TypeError, ValueError):
            if record_debug:
                record_debug("missing_days_to_resolution")
            return None
        if not np.isfinite(dtr) or dtr < 0 or dtr > float(self.days_threshold):
            if record_debug:
                record_debug("out_of_window")
            return None

        # Slope must confirm direction
        slope = features.get("slope")
        try:
            slope_value = float(slope)
        except (TypeError, ValueError):
            slope_value = 0.0

        if current_price > 0.85 and slope_value <= 0:
            if record_debug:
                record_debug("slope_contradicts_buy")
            return None
        if current_price < 0.15 and slope_value >= 0:
            if record_debug:
                record_debug("slope_contradicts_sell")
            return None

        # Spread must be tight
        spread = _actual_spread(features)
        if spread > 0.03:
            if record_debug:
                record_debug("spread_too_wide")
            return None

        direction = "buy" if current_price > 0.85 else "sell"
        target_price = 1.0 if direction == "buy" else 0.0

        # Signal strength: how extreme the price is × how close to resolution
        price_extremity = abs(current_price - 0.5) / 0.5  # 0.7 at 0.85, 1.0 at edges
        time_urgency = 1.0 - (dtr / max(1e-9, float(self.days_threshold)))
        signal_strength = price_extremity * time_urgency

        confidence = _calibrated_confidence(signal_strength, features)
        edge = abs(target_price - current_price) - spread
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
                "days_to_resolution": dtr,
                "slope": slope_value,
                "spread": spread,
                "price_extremity": round(price_extremity, 4),
            },
        )

    def explain(self, output: SignalOutput) -> str:
        dtr = output.metadata.get("days_to_resolution")
        return f"Market {dtr:.1f}d from resolution, price near {'1.0' if output.direction == 'buy' else '0.0'}."
