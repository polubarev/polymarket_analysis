from __future__ import annotations

import unittest

import pandas as pd

from polymarket_pipeline.config import PipelineConfig
from polymarket_pipeline.signals_runner import run_signal_generation


class SignalDebugTests(unittest.TestCase):
    def test_signal_debug_tracks_missing_feature_reason(self) -> None:
        config = PipelineConfig(
            run_signals=True,
            active_signals=["mean_reversion"],
            signal_debug=True,
            signal_debug_limit=2,
        )
        features_df = pd.DataFrame(
            [
                {
                    "asset_id": "asset-1",
                    "zscore_7d": None,
                    "days_to_resolution": 3.0,
                    "slope": 0.1,
                }
            ]
        )
        target_tokens_df = pd.DataFrame([{"asset_id": "asset-1", "market_id": "market-1"}])
        markets_df = pd.DataFrame(
            [{"market_id": "market-1", "event_id": 1, "question": "Question?", "resolved": False}]
        )
        events_df = pd.DataFrame([{"event_id": 1, "tags": '["politics"]'}])
        price_history_df = pd.DataFrame(
            [
                {"asset_id": "asset-1", "ts": 1_000 + (i * 3600), "price": 0.5 + (i * 0.001)}
                for i in range(24)
            ]
        )

        signals_df, _, debug_payload = run_signal_generation(
            config=config,
            features_df=features_df,
            target_tokens_df=target_tokens_df,
            markets_df=markets_df,
            events_df=events_df,
            price_history_df=price_history_df,
            volume_bars_df=None,
            resolutions_df=None,
            as_of_ts=2_000_000,
        )

        self.assertTrue(signals_df.empty)
        self.assertIsNotNone(debug_payload)
        mean_reversion = debug_payload["signals"]["mean_reversion"]
        self.assertEqual(mean_reversion["considered"], 1)
        self.assertEqual(mean_reversion["missing_zscore_7d"], 1)
        self.assertEqual(mean_reversion["generated"], 0)
        self.assertEqual(mean_reversion["samples"]["missing_zscore_7d"][0]["asset_id"], "asset-1")


if __name__ == "__main__":
    unittest.main()
