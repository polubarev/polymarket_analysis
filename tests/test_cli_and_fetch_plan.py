from __future__ import annotations

import unittest

import pandas as pd

from polymarket_pipeline.cli import _build_run_parser, _config_from_run_args
from polymarket_pipeline.config import PipelineConfig
from polymarket_pipeline.pipeline import PipelineRunner


class CliConfigTests(unittest.TestCase):
    def test_new_flags_round_trip(self) -> None:
        parser = _build_run_parser()
        args = parser.parse_args(
            [
                "--pipeline-profile",
                "research-daily",
                "--rate-window-s",
                "15",
                "--fetch-priority-mode",
                "category_round_robin",
                "--no-skip-inactive-priced-assets",
                "--signal-debug",
                "--signal-debug-limit",
                "7",
            ]
        )

        config = _config_from_run_args(args)

        self.assertEqual(config.pipeline_profile, "research-daily")
        self.assertEqual(config.rate_window_s, 15)
        self.assertEqual(config.fetch_priority_mode, "category_round_robin")
        self.assertFalse(config.skip_inactive_priced_assets)
        self.assertTrue(config.signal_debug)
        self.assertEqual(config.signal_debug_limit, 7)


class FetchPlanTests(unittest.TestCase):
    def test_history_first_prioritizes_existing_then_low_coverage_active_and_skips_inactive_existing(self) -> None:
        config = PipelineConfig(
            fetch_priority_mode="history_first",
            incremental_prices=True,
            incremental_mode="tail",
            incremental_overlap_points=2,
            skip_inactive_priced_assets=True,
            interval="1h",
        )
        runner = PipelineRunner(config)
        now_ts = 1_000_000
        global_start_ts = now_ts - 30 * 24 * 3600

        target_tokens_df = pd.DataFrame(
            [
                {"asset_id": "a_existing_sports", "market_id": "m1"},
                {"asset_id": "a_existing_legacy", "market_id": "m2"},
                {"asset_id": "a_new_politics", "market_id": "m3"},
                {"asset_id": "a_new_sports", "market_id": "m4"},
                {"asset_id": "a_new_legacy", "market_id": "m5"},
            ]
        )
        markets_df = pd.DataFrame(
            [
                {"market_id": "m1", "event_id": 1, "active": True, "closed": False, "resolved": False},
                {"market_id": "m2", "event_id": 2, "active": False, "closed": True, "resolved": True},
                {"market_id": "m3", "event_id": 3, "active": True, "closed": False, "resolved": False},
                {"market_id": "m4", "event_id": 4, "active": True, "closed": False, "resolved": False},
                {"market_id": "m5", "event_id": 5, "active": False, "closed": True, "resolved": True},
            ]
        )
        events_df = pd.DataFrame(
            [
                {"event_id": 1, "end_ts": now_ts + 10_000, "tags": '["sports"]'},
                {"event_id": 2, "end_ts": now_ts - 10_000, "tags": '["elections"]'},
                {"event_id": 3, "end_ts": now_ts + 10_000, "tags": '["politics"]'},
                {"event_id": 4, "end_ts": now_ts + 10_000, "tags": '["sports"]'},
                {"event_id": 5, "end_ts": now_ts - 10_000, "tags": '["elections"]'},
            ]
        )

        plan = runner._build_price_fetch_plan(
            target_tokens_df=target_tokens_df,
            markets_df=markets_df,
            events_df=events_df,
            existing_latest_ts={"a_existing_sports": 980_000, "a_existing_legacy": 970_000},
            incremental_mode="tail",
            now_ts=now_ts,
            global_start_ts=global_start_ts,
        )

        self.assertEqual([task.asset_id for task in plan], [
            "a_existing_sports",
            "a_new_politics",
            "a_new_sports",
            "a_new_legacy",
        ])
        self.assertEqual(plan[0].bucket, "existing")
        self.assertEqual(plan[0].start_ts, 980_000 - (2 * 3600))
        self.assertEqual(plan[-1].bucket, "new_inactive")
        self.assertEqual(runner.metrics["skipped_inactive_priced_assets"], 1)

    def test_category_round_robin_alternates_tags_for_uncovered_assets(self) -> None:
        config = PipelineConfig(fetch_priority_mode="category_round_robin")
        runner = PipelineRunner(config)
        rows = pd.DataFrame(
            [
                {"asset_id": "a1", "primary_tag": "alpha", "tag_coverage": 0.0},
                {"asset_id": "a2", "primary_tag": "alpha", "tag_coverage": 0.0},
                {"asset_id": "b1", "primary_tag": "beta", "tag_coverage": 0.0},
                {"asset_id": "b2", "primary_tag": "beta", "tag_coverage": 0.0},
            ]
        )

        ordered = runner._order_price_fetch_rows(rows)

        self.assertEqual(ordered["asset_id"].tolist(), ["a1", "b1", "a2", "b2"])


if __name__ == "__main__":
    unittest.main()
