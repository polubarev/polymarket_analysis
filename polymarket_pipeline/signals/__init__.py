from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
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
        record_debug: Callable[[str], None] | None = None,
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


def _actual_spread(features: dict[str, Any]) -> float:
    """Return the actual spread for this asset, falling back to 3% default."""
    spread = features.get("avg_spread")
    try:
        val = float(spread)
        if np.isfinite(val) and val > 0:
            return val
    except (TypeError, ValueError):
        pass
    return 0.03


def _calibrated_confidence(
    signal_strength: float,
    features: dict[str, Any],
) -> float:
    """Compute confidence as product of signal strength, liquidity, and data quality.

    Prevents confidence from ever reaching 1.0 and penalizes illiquid or data-poor markets.
    """
    strength = min(0.8, max(0.0, float(signal_strength)))
    volume = features.get("avg_daily_volume")
    try:
        liq = min(1.0, float(volume) / 500.0) if volume is not None else 0.5
    except (TypeError, ValueError):
        liq = 0.5
    points = features.get("num_points")
    try:
        dq = min(1.0, float(points) / 100.0) if points is not None else 0.5
    except (TypeError, ValueError):
        dq = 0.5
    return round(max(0.0, strength * liq * dq), 4)


def build_registry(config: Any, base_rate_by_tag: dict[str, float] | None = None) -> dict[str, Signal]:
    from .convergence import ResolutionConvergenceSignal
    from .momentum import VolumeConfirmedMomentumSignal
    from .orderbook_imbalance import OrderbookImbalanceSignal

    registry: dict[str, Signal] = {
        "convergence": ResolutionConvergenceSignal(
            days_threshold=int(getattr(config, "convergence_days_threshold", 2))
        ),
        "momentum": VolumeConfirmedMomentumSignal(
            volume_surge_threshold=float(getattr(config, "volume_surge_threshold", 2.0))
        ),
        "orderbook_imbalance": OrderbookImbalanceSignal(
            imbalance_ratio_threshold=float(getattr(config, "imbalance_ratio_threshold", 3.0))
        ),
    }
    return registry
