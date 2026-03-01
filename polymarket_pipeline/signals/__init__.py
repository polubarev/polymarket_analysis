from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalOutput:
    direction: str
    confidence: float
    entry_price: float
    target_price: float
    edge: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(
        self,
        *,
        signal_name: str,
        asset_id: str,
        market_id: str,
        ts: int,
    ) -> dict[str, Any]:
        return {
            "signal_name": signal_name,
            "asset_id": str(asset_id),
            "market_id": str(market_id),
            "ts": int(ts),
            "direction": self.direction,
            "confidence": float(self.confidence),
            "entry_price": float(self.entry_price),
            "target_price": float(self.target_price),
            "edge": float(self.edge),
            "metadata_json": json.dumps(self.metadata, ensure_ascii=True),
        }


class Signal(ABC):
    name: str = "signal"

    @abstractmethod
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
        raise NotImplementedError

    def explain(self, output: SignalOutput) -> str:
        return f"{self.name} generated {output.direction} with edge {output.edge:.3f}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _latest_price(price_history: pd.Series) -> float | None:
    if price_history is None or price_history.empty:
        return None
    value = _safe_float(price_history.iloc[-1])
    return value


def build_registry(config: Any, base_rate_by_tag: dict[str, float] | None = None) -> dict[str, Signal]:
    from .calibration import CalibrationMispricingSignal
    from .convergence import ResolutionConvergenceSignal
    from .mean_reversion import MeanReversionSpikeSignal
    from .momentum import MomentumBreakoutSignal

    base_rates = base_rate_by_tag or {}
    registry: dict[str, Signal] = {
        "calibration": CalibrationMispricingSignal(
            threshold=float(getattr(config, "calibration_threshold", 0.15)),
            base_rate_by_tag=base_rates,
        ),
        "mean_reversion": MeanReversionSpikeSignal(
            zscore_threshold=float(getattr(config, "spike_zscore_threshold", 2.5))
        ),
        "convergence": ResolutionConvergenceSignal(
            days_threshold=int(getattr(config, "convergence_days_threshold", 7))
        ),
        "momentum": MomentumBreakoutSignal(
            lookback_days=int(getattr(config, "breakout_lookback_days", 30))
        ),
    }
    return registry
