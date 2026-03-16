from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from polymarket_pipeline.config import PipelineConfig
from polymarket_pipeline.pipeline import PipelineInputError, PipelineRunner


class AnalysisOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name)

    def _write_base_inputs(self) -> dict[str, pd.DataFrame]:
        now_ts = int(time.time())
        events_df = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "slug": "will-it-rain",
                    "title": "Will it rain?",
                    "start_ts": now_ts - 14_400,
                    "end_ts": now_ts + 7 * 24 * 3600,
                    "tags": '["weather"]',
                }
            ]
        )
        markets_df = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "event_id": 1,
                    "condition_id": "c1",
                    "question": "Will it rain?",
                    "active": True,
                    "closed": False,
                    "resolved": False,
                    "resolution_outcome": None,
                    "resolution_ts": None,
                    "market_type": "binary",
                    "liquidity": 1000.0,
                }
            ]
        )
        tokens_df = pd.DataFrame(
            [
                {"market_id": "m1", "outcome": "Yes", "asset_id": "yes-1"},
                {"market_id": "m1", "outcome": "No", "asset_id": "no-1"},
            ]
        )
        price_history_df = pd.DataFrame(
            [
                {"asset_id": "yes-1", "ts": now_ts - 7200, "price": 0.42, "ingested_at": now_ts - 120},
                {"asset_id": "yes-1", "ts": now_ts - 3600, "price": 0.47, "ingested_at": now_ts - 120},
            ]
        )
        volume_bars_df = pd.DataFrame(
            [
                {"asset_id": "yes-1", "market_id": "m1", "ts": now_ts - 3600, "volume": 150.0},
            ]
        )
        resolutions_df = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "condition_id": "c1",
                    "resolution_outcome": "Yes",
                    "resolution_ts": now_ts - 1800,
                    "ingested_at": now_ts - 60,
                }
            ]
        )
        orderbook_df = pd.DataFrame(
            [
                {"asset_id": "yes-1", "spread": 0.02, "spread_pct": 0.03, "ts": now_ts - 60},
            ]
        )

        frames = {
            "events": events_df,
            "markets": markets_df,
            "tokens": tokens_df,
            "price_history": price_history_df,
            "volume_bars": volume_bars_df,
            "resolutions": resolutions_df,
            "orderbook_snapshots": orderbook_df,
        }
        for name, frame in frames.items():
            frame.to_parquet(self.output_dir / f"{name}.parquet", index=False)
        return frames

    def test_run_analysis_only_uses_stored_inputs_and_preserves_base_tables(self) -> None:
        frames = self._write_base_inputs()
        config = PipelineConfig(
            output_dir=self.output_dir,
            yes_only_binary=True,
            run_signals=True,
            run_backtest=True,
            generate_candidates=True,
            signal_debug=True,
        )
        runner = PipelineRunner(config)
        base_stats = {
            name: (path.stat().st_mtime_ns, path.stat().st_size)
            for name in ("events", "markets", "tokens", "price_history", "resolutions", "volume_bars")
            for path in [self.output_dir / f"{name}.parquet"]
        }
        observed: dict[str, object] = {}

        def fake_analyze(
            self,
            *,
            events_df: pd.DataFrame,
            markets_df: pd.DataFrame,
            all_tokens_df: pd.DataFrame,
            target_tokens_df: pd.DataFrame,
            price_history_df: pd.DataFrame,
            orderbook_df: pd.DataFrame,
            volume_bars_df: pd.DataFrame,
        ) -> tuple[Path, Path, Path, Path, pd.DataFrame, dict[str, object]]:
            observed["events_len"] = len(events_df)
            observed["all_tokens"] = sorted(all_tokens_df["asset_id"].astype(str).tolist())
            observed["target_tokens"] = sorted(target_tokens_df["asset_id"].astype(str).tolist())
            observed["orderbook_empty"] = bool(orderbook_df.empty)
            observed["volume_rows"] = int(len(volume_bars_df))

            features_df = pd.DataFrame([{"asset_id": "yes-1", "days_to_resolution": 7.0, "num_points": 2}])
            features_path = self.config.output_dir / "features.parquet"
            quality_path = self.config.output_dir / "market_quality.parquet"
            clusters_path = self.config.output_dir / "clusters.parquet"
            report_path = self.config.analysis_dir / "report.json"
            features_df.to_parquet(features_path, index=False)
            pd.DataFrame([{"asset_id": "yes-1", "quality_pass": True}]).to_parquet(quality_path, index=False)
            pd.DataFrame([{"asset_id": "yes-1", "market_id": "m1", "cluster_id": 0}]).to_parquet(
                clusters_path,
                index=False,
            )
            with (self.config.analysis_dir / "feature_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump({"feature_columns": ["days_to_resolution", "num_points"]}, handle, indent=2)
            report_payload = {"coverage": {"pct_targeted_with_history": 1.0, "median_points_per_token": 2.0}}
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report_payload, handle, indent=2)
            return report_path, clusters_path, quality_path, features_path, features_df, report_payload

        def fake_run_signal_generation(**kwargs: object) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
            observed["signal_target_tokens"] = sorted(
                kwargs["target_tokens_df"]["asset_id"].astype(str).tolist()  # type: ignore[index]
            )
            observed["signal_resolutions_rows"] = int(len(kwargs["resolutions_df"]))  # type: ignore[arg-type]
            signals_df = pd.DataFrame(
                [
                    {
                        "signal_name": "test_signal",
                        "asset_id": "yes-1",
                        "market_id": "m1",
                        "ts": int(time.time()),
                        "direction": "buy",
                        "confidence": 0.8,
                        "entry_price": 0.47,
                        "target_price": 0.60,
                        "edge": 0.13,
                        "metadata_json": "{}",
                    }
                ]
            )
            return signals_df, {"test_signal": object()}, {"signals": {"test_signal": {"generated": 1}}}

        def fake_run_backtest(**kwargs: object) -> tuple[pd.DataFrame, dict[str, object]]:
            observed["backtest_orderbook_empty"] = bool(kwargs["orderbook_df"].empty)  # type: ignore[attr-defined]
            return (
                pd.DataFrame(
                    [
                        {
                            "backtest_id": "b1",
                            "signal_name": "test_signal",
                            "asset_id": "yes-1",
                            "market_id": "m1",
                            "direction": "long",
                            "entry_ts": int(time.time()) - 3600,
                            "entry_price": 0.47,
                            "entry_cost": 0.48,
                            "exit_ts": int(time.time()),
                            "exit_price": 1.0,
                            "exit_reason": "resolution",
                            "pnl": 52.0,
                            "return_pct": 1.08,
                            "hold_days": 0.04,
                            "kelly_fraction": 0.1,
                            "position_size": 100.0,
                            "bankroll_at_entry": 10_000.0,
                        }
                    ]
                ),
                {"trades": 1},
            )

        def fake_generate_trade_candidates(**kwargs: object) -> tuple[dict[str, object], pd.DataFrame]:
            observed["candidate_orderbook_empty"] = bool(kwargs["orderbook_df"].empty)  # type: ignore[attr-defined]
            return (
                {"generated_at_utc": "2026-03-10T00:00:00+00:00", "candidates": [{"asset_id": "yes-1"}]},
                pd.DataFrame(
                    [
                        {
                            "run_date": "2026-03-10",
                            "rank": 1,
                            "market_id": "m1",
                            "signal_name": "test_signal",
                            "asset_id": "yes-1",
                        }
                    ]
                ),
            )

        with (
            patch.object(PipelineRunner, "_analyze", autospec=True, side_effect=fake_analyze),
            patch("polymarket_pipeline.pipeline.run_signal_generation", side_effect=fake_run_signal_generation),
            patch("polymarket_pipeline.pipeline.run_backtest", side_effect=fake_run_backtest),
            patch("polymarket_pipeline.pipeline.generate_trade_candidates", side_effect=fake_generate_trade_candidates),
            patch.object(PipelineRunner, "_build_clients", side_effect=AssertionError("clients should not be built")),
            patch.object(PipelineRunner, "_discover_active_events", side_effect=AssertionError("discovery should not run")),
            patch.object(PipelineRunner, "_discover_resolved_events", side_effect=AssertionError("resolved discovery should not run")),
            patch.object(PipelineRunner, "_ingest_price_history", side_effect=AssertionError("price ingest should not run")),
            patch.object(PipelineRunner, "_ingest_resolutions", side_effect=AssertionError("resolution ingest should not run")),
            patch.object(PipelineRunner, "_snapshot_orderbook", side_effect=AssertionError("orderbook ingest should not run")),
            patch.object(PipelineRunner, "_ingest_volume_bars", side_effect=AssertionError("volume ingest should not run")),
        ):
            outputs = runner.run_analysis_only()

        self.assertEqual(observed["events_len"], len(frames["events"]))
        self.assertEqual(observed["all_tokens"], ["no-1", "yes-1"])
        self.assertEqual(observed["target_tokens"], ["yes-1"])
        self.assertEqual(observed["signal_target_tokens"], ["yes-1"])
        self.assertEqual(observed["signal_resolutions_rows"], 1)
        self.assertTrue(observed["orderbook_empty"])
        self.assertTrue(observed["backtest_orderbook_empty"])
        self.assertTrue(observed["candidate_orderbook_empty"])
        self.assertEqual(observed["volume_rows"], 1)

        for name, before in base_stats.items():
            path = self.output_dir / f"{name}.parquet"
            self.assertEqual((path.stat().st_mtime_ns, path.stat().st_size), before)

        expected_output_keys = {
            "clusters",
            "market_quality",
            "features",
            "report",
            "feature_metadata",
            "signals",
            "signal_debug",
            "backtest_results",
            "backtest_summary",
            "trade_candidates_json",
            "trade_candidates",
            "health_check",
            "pipeline_runs",
        }
        self.assertTrue(expected_output_keys.issubset(outputs))
        for key in expected_output_keys:
            self.assertTrue(outputs[key].exists())

        health_payload = json.loads((self.output_dir / "analysis" / "health_check.json").read_text(encoding="utf-8"))
        self.assertEqual(health_payload["input_sources"]["mode"], "analysis_only")
        self.assertTrue(health_payload["input_sources"]["optional_inputs"]["orderbook_snapshots_ignored"])
        self.assertTrue(health_payload["input_sources"]["optional_inputs"]["resolutions_available"])
        self.assertTrue(health_payload["input_sources"]["optional_inputs"]["volume_bars_available"])

    def test_run_analysis_only_surfaces_missing_optional_inputs(self) -> None:
        self._write_base_inputs()
        (self.output_dir / "resolutions.parquet").unlink()
        (self.output_dir / "volume_bars.parquet").unlink()
        config = PipelineConfig(
            output_dir=self.output_dir,
            run_signals=True,
            run_backtest=True,
            generate_candidates=True,
            signal_debug=True,
        )
        runner = PipelineRunner(config)
        build_health_calls: list[dict[str, object]] = []

        def fake_analyze(
            self,
            *,
            events_df: pd.DataFrame,
            markets_df: pd.DataFrame,
            all_tokens_df: pd.DataFrame,
            target_tokens_df: pd.DataFrame,
            price_history_df: pd.DataFrame,
            orderbook_df: pd.DataFrame,
            volume_bars_df: pd.DataFrame,
        ) -> tuple[Path, Path, Path, Path, pd.DataFrame, dict[str, object]]:
            features_df = pd.DataFrame([{"asset_id": "yes-1"}])
            features_path = self.config.output_dir / "features.parquet"
            quality_path = self.config.output_dir / "market_quality.parquet"
            clusters_path = self.config.output_dir / "clusters.parquet"
            report_path = self.config.analysis_dir / "report.json"
            features_df.to_parquet(features_path, index=False)
            pd.DataFrame([{"asset_id": "yes-1", "quality_pass": True}]).to_parquet(quality_path, index=False)
            pd.DataFrame([{"asset_id": "yes-1", "market_id": "m1", "cluster_id": 0}]).to_parquet(
                clusters_path,
                index=False,
            )
            with (self.config.analysis_dir / "feature_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump({"feature_columns": ["asset_id"]}, handle, indent=2)
            report_payload = {"coverage": {"pct_targeted_with_history": 1.0, "median_points_per_token": 2.0}}
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report_payload, handle, indent=2)
            return report_path, clusters_path, quality_path, features_path, features_df, report_payload

        def fake_build_health_check(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
            build_health_calls.append(kwargs)
            return (
                {"warnings": [], "errors": [], "checks": [], "status": "HEALTHY"},
                {
                    "run_id": "run-1",
                    "run_ts": int(time.time()),
                    "duration_s": 1.0,
                    "events_count": 1,
                    "markets_count": 1,
                    "price_coverage_pct": 1.0,
                    "signals_generated": 0,
                    "signals_expected": True,
                    "pipeline_profile": "research-weekly",
                    "status": "HEALTHY",
                    "config_hash": "hash",
                },
            )

        with (
            patch.object(PipelineRunner, "_analyze", autospec=True, side_effect=fake_analyze),
            patch(
                "polymarket_pipeline.pipeline.run_signal_generation",
                return_value=(pd.DataFrame(), {}, {"signals": {}}),
            ),
            patch("polymarket_pipeline.pipeline.run_backtest", return_value=(pd.DataFrame(), {"trades": 0})),
            patch(
                "polymarket_pipeline.pipeline.generate_trade_candidates",
                return_value=({"generated_at_utc": "2026-03-10T00:00:00+00:00", "candidates": []}, pd.DataFrame()),
            ),
            patch("polymarket_pipeline.pipeline.build_health_check", side_effect=fake_build_health_check),
        ):
            runner.run_analysis_only()

        self.assertEqual(len(build_health_calls), 1)
        self.assertFalse(build_health_calls[0]["include_resolved"])

        health_payload = json.loads((self.output_dir / "analysis" / "health_check.json").read_text(encoding="utf-8"))
        self.assertIn("resolutions.parquet missing or empty; continuing without resolution data.", health_payload["warnings"])
        self.assertIn("volume_bars.parquet missing or empty; continuing without volume data.", health_payload["warnings"])

        signal_debug_payload = json.loads((self.output_dir / "analysis" / "signal_debug.json").read_text(encoding="utf-8"))
        self.assertIn("resolutions.parquet missing or empty; continuing without resolution data.", signal_debug_payload["warnings"])
        self.assertIn("volume_bars.parquet missing or empty; continuing without volume data.", signal_debug_payload["warnings"])

    def test_run_analysis_only_requires_existing_base_inputs(self) -> None:
        runner = PipelineRunner(PipelineConfig(output_dir=self.output_dir))

        with self.assertRaisesRegex(PipelineInputError, "Analyze mode requires existing base parquet inputs"):
            runner.run_analysis_only()


if __name__ == "__main__":
    unittest.main()
