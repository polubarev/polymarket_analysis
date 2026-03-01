from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import Signal, SignalOutput, _latest_price


@dataclass(slots=True)
class MeanReversionSpikeSignal(Signal):
    zscore_threshold: float = 2.5
    name: str = "mean_reversion_spike"

    def compute(
        self,
        *,
        asset_id: str,
        market_row: dict[str, Any],
        features: dict[str, Any],
        price_history: pd.Series,
        volume_history: pd.Series | None,
        as_of_ts: int,
    ) -> SignalOutput | None:
        current_price = _latest_price(price_history)
        if current_price is None or len(price_history) < 24:
            return None

        zscore = features.get("zscore_7d")
        try:
            z = float(zscore)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(z) or abs(z) < float(self.zscore_threshold):
            return None

        # Optional thin-volume filter: spike should happen on below-median volume.
        if volume_history is not None and not volume_history.empty:
            volume_series = pd.to_numeric(volume_history, errors="coerce").dropna()
            if not volume_series.empty:
                latest_volume = float(volume_series.iloc[-1])
                median_volume = float(volume_series.median())
                if latest_volume >= median_volume:
                    return None

        direction = "sell" if z > 0 else "buy"
        confidence = min(1.0, abs(z) / max(1e-9, float(self.zscore_threshold) * 2.0))
        edge = max(0.0, abs(z) / 10.0)
        target_price = float(np.nanmean(pd.to_numeric(price_history.tail(24), errors="coerce").dropna()))
        return SignalOutput(
            direction=direction,
            confidence=confidence,
            entry_price=float(current_price),
            target_price=target_price,
            edge=float(edge),
            metadata={
                "zscore_7d": float(z),
                "threshold": float(self.zscore_threshold),
            },
        )

    def explain(self, output: SignalOutput) -> str:
        zscore = output.metadata.get("zscore_7d")
        return f"7d z-score={zscore:.2f} exceeded threshold; mean-reversion suggests {output.direction}."
