from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import Signal, SignalOutput, _latest_price


@dataclass(slots=True)
class ResolutionConvergenceSignal(Signal):
    days_threshold: int = 7
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
            if record_debug is not None:
                record_debug("missing_price")
            return None
        if not (0.25 <= float(current_price) <= 0.75):
            if record_debug is not None:
                record_debug("price_not_midrange")
            return None

        days_to_resolution = features.get("days_to_resolution")
        try:
            dtr = float(days_to_resolution)
        except (TypeError, ValueError):
            if record_debug is not None:
                record_debug("missing_days_to_resolution")
            return None
        if not np.isfinite(dtr) or dtr < 0 or dtr > float(self.days_threshold):
            if record_debug is not None:
                record_debug("out_of_window")
            return None

        slope = features.get("slope")
        try:
            slope_value = float(slope)
        except (TypeError, ValueError):
            slope_value = 0.0
        direction = "buy" if slope_value >= 0 else "sell"
        confidence = min(1.0, max(0.0, 1.0 - (dtr / max(1e-9, float(self.days_threshold)))))
        target_price = 1.0 if direction == "buy" else 0.0
        edge = abs(target_price - float(current_price))
        return SignalOutput(
            direction=direction,
            confidence=confidence,
            entry_price=float(current_price),
            target_price=float(target_price),
            edge=float(edge),
            metadata={
                "days_to_resolution": dtr,
                "slope": slope_value,
                "threshold_days": int(self.days_threshold),
            },
        )

    def explain(self, output: SignalOutput) -> str:
        dtr = output.metadata.get("days_to_resolution")
        return f"Market is {dtr:.1f} days from resolution; convergence signal favors {output.direction}."
