from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import Signal, SignalOutput, _latest_price, _actual_spread, _calibrated_confidence, _safe_float


@dataclass(slots=True)
class OrderbookImbalanceSignal(Signal):
    """Orderbook imbalance: trade when bid/ask depth is heavily skewed.

    When bid depth significantly exceeds ask depth, demand is building → buy.
    When ask depth significantly exceeds bid depth, supply pressure → sell.

    Only fires when:
    - Imbalance ratio > threshold (default 3x)
    - Price in tradeable range (0.10–0.90)
    - Spread is tight
    - Sufficient daily volume (liquidity)
    """

    imbalance_ratio_threshold: float = 3.0
    name: str = "orderbook_imbalance"

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

        # Price must be in tradeable range — at extremes, imbalance is expected
        if not (0.10 <= current_price <= 0.90):
            if record_debug:
                record_debug("price_at_extreme")
            return None

        # Get bid/ask depth
        bid_depth = _safe_float(features.get("avg_bid_depth"))
        ask_depth = _safe_float(features.get("avg_ask_depth"))
        if bid_depth is None or ask_depth is None:
            if record_debug:
                record_debug("missing_depth_data")
            return None
        if bid_depth <= 0 and ask_depth <= 0:
            if record_debug:
                record_debug("zero_depth")
            return None

        # Compute imbalance ratio
        if ask_depth > 0:
            imbalance_ratio = bid_depth / ask_depth
        elif bid_depth > 0:
            imbalance_ratio = float("inf")
        else:
            if record_debug:
                record_debug("zero_depth")
            return None

        # Must exceed threshold
        is_buy = imbalance_ratio > self.imbalance_ratio_threshold
        is_sell = imbalance_ratio < (1.0 / self.imbalance_ratio_threshold)
        if not (is_buy or is_sell):
            if record_debug:
                record_debug("imbalance_below_threshold")
            return None

        # Spread filter
        spread = _actual_spread(features)
        if spread > 0.03:
            if record_debug:
                record_debug("spread_too_wide")
            return None

        # Volume filter — need sufficient liquidity
        avg_vol = _safe_float(features.get("avg_daily_volume"))
        if avg_vol is None or avg_vol < 100:
            if record_debug:
                record_debug("insufficient_volume")
            return None

        direction = "buy" if is_buy else "sell"
        target_price = min(0.95, current_price + 0.03) if direction == "buy" else max(0.05, current_price - 0.03)

        # Signal strength from imbalance magnitude
        effective_ratio = imbalance_ratio if is_buy else (1.0 / imbalance_ratio)
        signal_strength = min(1.0, effective_ratio / 10.0)

        confidence = _calibrated_confidence(signal_strength, features)
        edge = min(0.05, (abs(effective_ratio - 1.0) / max(1.0, effective_ratio)) * 0.1)
        edge = edge - spread
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
                "bid_depth": round(bid_depth, 2),
                "ask_depth": round(ask_depth, 2),
                "imbalance_ratio": round(imbalance_ratio, 2) if np.isfinite(imbalance_ratio) else 999.0,
                "spread": spread,
                "avg_daily_volume": round(avg_vol, 0),
            },
        )

    def explain(self, output: SignalOutput) -> str:
        ratio = output.metadata.get("imbalance_ratio", 0)
        return f"Orderbook imbalance {ratio:.1f}x favors {output.direction}."
