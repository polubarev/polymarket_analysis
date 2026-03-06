from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from typing import Any

import pandas as pd

from . import Signal, SignalOutput, _latest_price


@dataclass(slots=True)
class CalibrationMispricingSignal(Signal):
    threshold: float = 0.15
    base_rate_by_tag: dict[str, float] = field(default_factory=dict)
    name: str = "calibration_mispricing"

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
        tag = str(market_row.get("primary_tag", "unknown")).strip().lower() or "unknown"
        base_rate = self.base_rate_by_tag.get(tag)
        if base_rate is None:
            if record_debug is not None:
                record_debug("missing_base_rate")
            return None
        diff = float(base_rate - current_price)
        if abs(diff) < float(self.threshold):
            if record_debug is not None:
                record_debug("threshold_fail")
            return None

        direction = "buy" if diff > 0 else "sell"
        confidence = min(1.0, abs(diff))
        edge = abs(diff) - float(self.threshold)
        return SignalOutput(
            direction=direction,
            confidence=confidence,
            entry_price=float(current_price),
            target_price=float(base_rate),
            edge=float(max(0.0, edge)),
            metadata={
                "base_rate": float(base_rate),
                "tag": tag,
                "threshold": float(self.threshold),
                "price_diff": float(diff),
            },
        )

    def explain(self, output: SignalOutput) -> str:
        base_rate = output.metadata.get("base_rate")
        tag = output.metadata.get("tag")
        return (
            f"Base rate for {tag} is {base_rate:.3f}; "
            f"price dislocation suggests {output.direction} (edge {output.edge:.3f})."
        )
