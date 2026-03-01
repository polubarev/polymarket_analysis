from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import PipelineRunner


def _add_common_pipeline_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="data", help="Output directory for raw + parquet + reports")
    parser.add_argument("--max-events", type=int, default=2000, help="Maximum number of active events to ingest")
    parser.add_argument("--page-limit", type=int, default=100, help="Gamma events page size")
    parser.add_argument("--window-days", type=int, default=90, help="Lookback window for prices-history")
    parser.add_argument("--interval", default="1h", help="CLOB prices-history interval (e.g. 1h)")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=5, help="HTTP retry count")
    parser.add_argument("--yes-only-binary", action="store_true", help="Only fetch YES token for binary markets")
    parser.add_argument("--all-outcomes", action="store_true", help="Fetch all outcomes, including binary NO tokens")
    parser.add_argument("--fetch-trades-sample", type=int, default=0, help="Optional number of condition IDs for trades")
    parser.add_argument("--cluster-k", type=int, default=8, help="Requested cluster count for KMeans")
    parser.add_argument("--gap-fill-limit", type=int, default=6, help="Forward-fill cap (in points) for sparse series")
    parser.add_argument("--quality-min-points", type=int, default=24, help="Minimum points for quality pass")
    parser.add_argument(
        "--quality-max-missing-ratio",
        type=float,
        default=0.8,
        help="Maximum missing ratio allowed for quality pass",
    )
    parser.add_argument(
        "--quality-min-price-range",
        type=float,
        default=0.02,
        help="Minimum max-min price range for quality pass",
    )
    parser.add_argument(
        "--quality-min-liquidity",
        type=float,
        default=0.0,
        help="Minimum market liquidity for quality pass (0 disables filter)",
    )
    parser.add_argument("--tag-rank-top-n", type=int, default=10, help="Top-N size for tag rankings in report")
    parser.add_argument(
        "--price-fetch-workers",
        type=int,
        default=16,
        help="Number of concurrent workers for prices-history fetching",
    )
    parser.add_argument(
        "--no-incremental-prices",
        action="store_true",
        help="Always refetch prices even when existing interval data is already stored",
    )
    parser.add_argument(
        "--incremental-mode",
        choices=["tail", "skip"],
        default="tail",
        help="Incremental behavior: tail refresh existing assets or skip them",
    )
    parser.add_argument(
        "--incremental-overlap-points",
        type=int,
        default=2,
        help="When using tail mode, overlap this many points to avoid boundary gaps",
    )
    parser.add_argument(
        "--skip-raw-price-files",
        action="store_true",
        help="Do not write per-asset raw/parquet price files (faster, less disk)",
    )
    parser.add_argument(
        "--http-pool-maxsize",
        type=int,
        default=64,
        help="HTTP connection pool max size (increase if workers are high)",
    )
    parser.add_argument("--include-resolved", action="store_true", help="Also ingest closed/resolved events")
    parser.add_argument(
        "--resolved-lookback-days",
        type=int,
        default=365,
        help="Lookback days for resolved event ingestion",
    )
    parser.add_argument("--snapshot-orderbook", action="store_true", help="Fetch orderbook snapshots")
    parser.add_argument("--orderbook-workers", type=int, default=8, help="Orderbook snapshot worker count")
    parser.add_argument("--ingest-volume", action="store_true", help="Fetch trades and build volume bars")
    parser.add_argument("--volume-fetch-workers", type=int, default=8, help="Volume ingestion worker count")
    parser.add_argument("--run-signals", action="store_true", help="Run signal generation")
    parser.add_argument(
        "--signals",
        default="all",
        help="Comma-separated signal names, e.g. calibration,mean_reversion",
    )
    parser.add_argument("--calibration-threshold", type=float, default=0.15)
    parser.add_argument("--spike-zscore-threshold", type=float, default=2.5)
    parser.add_argument("--convergence-days-threshold", type=int, default=7)
    parser.add_argument("--breakout-lookback-days", type=int, default=30)
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--backtest-start-date", default=None, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--backtest-end-date", default=None, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--backtest-initial-capital", type=float, default=10_000.0)
    parser.add_argument("--backtest-spread-assumption", type=float, default=0.03)
    parser.add_argument("--backtest-max-positions", type=int, default=50)
    parser.add_argument("--backtest-stop-loss", type=float, default=0.50)
    parser.add_argument("--backtest-timeout-days", type=int, default=90)
    parser.add_argument("--sizing-mode", choices=["flat", "kelly"], default="flat")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Fractional Kelly multiplier")
    parser.add_argument("--max-position-pct", type=float, default=0.05)
    parser.add_argument("--min-position-size", type=float, default=1.0)
    parser.add_argument("--flat-position-size", type=float, default=100.0)
    parser.add_argument("--generate-candidates", action="store_true", help="Generate ranked trade candidates")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--detect-relationships", action="store_true", help="Detect cross-market relationships")
    parser.add_argument("--correlation-threshold", type=float, default=0.5)
    parser.add_argument("--min-overlap-days", type=int, default=30)
    parser.add_argument("--ui-mode", choices=["discovery", "full"], default="discovery")
    parser.add_argument("--log-level", default="INFO", help="Python log level")


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polymarket strategy research pipeline")
    _add_common_pipeline_flags(parser)
    return parser


def _build_resolve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolved-market ingestion")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--lookback-days", type=int, default=365, help="How far back to ingest closed events")
    parser.add_argument("--page-limit", type=int, default=100, help="Gamma page size")
    parser.add_argument("--max-events", type=int, default=20_000, help="Maximum resolved events to ingest")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=5, help="HTTP retry count")
    parser.add_argument("--http-pool-maxsize", type=int, default=64, help="HTTP connection pool max size")
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    return parser


def _parse_active_signals(raw: str) -> list[str]:
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    return values or ["all"]


def _config_from_run_args(args: argparse.Namespace) -> PipelineConfig:
    yes_only = True
    if args.all_outcomes:
        yes_only = False
    elif args.yes_only_binary:
        yes_only = True

    return PipelineConfig(
        output_dir=Path(args.output_dir),
        max_events=args.max_events,
        page_limit=args.page_limit,
        window_days=args.window_days,
        interval=args.interval,
        request_timeout_s=args.timeout,
        max_retries=args.max_retries,
        yes_only_binary=yes_only,
        fetch_trades_sample=args.fetch_trades_sample,
        cluster_k=args.cluster_k,
        gap_fill_limit=args.gap_fill_limit,
        quality_min_points=args.quality_min_points,
        quality_max_missing_ratio=args.quality_max_missing_ratio,
        quality_min_price_range=args.quality_min_price_range,
        quality_min_liquidity=args.quality_min_liquidity,
        tag_rank_top_n=args.tag_rank_top_n,
        price_fetch_workers=args.price_fetch_workers,
        incremental_prices=not args.no_incremental_prices,
        incremental_mode=args.incremental_mode,
        incremental_overlap_points=args.incremental_overlap_points,
        write_raw_price_files=not args.skip_raw_price_files,
        http_pool_maxsize=args.http_pool_maxsize,
        include_resolved=args.include_resolved,
        resolved_lookback_days=args.resolved_lookback_days,
        snapshot_orderbook=args.snapshot_orderbook,
        orderbook_workers=args.orderbook_workers,
        ingest_volume=args.ingest_volume,
        volume_fetch_workers=args.volume_fetch_workers,
        run_signals=args.run_signals,
        active_signals=_parse_active_signals(args.signals),
        calibration_threshold=args.calibration_threshold,
        spike_zscore_threshold=args.spike_zscore_threshold,
        convergence_days_threshold=args.convergence_days_threshold,
        breakout_lookback_days=args.breakout_lookback_days,
        run_backtest=args.backtest,
        backtest_start_date=args.backtest_start_date,
        backtest_end_date=args.backtest_end_date,
        backtest_initial_capital=args.backtest_initial_capital,
        backtest_spread_assumption=args.backtest_spread_assumption,
        backtest_max_positions=args.backtest_max_positions,
        backtest_stop_loss=args.backtest_stop_loss,
        backtest_timeout_days=args.backtest_timeout_days,
        sizing_mode=args.sizing_mode,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position_pct,
        min_position_size=args.min_position_size,
        flat_position_size=args.flat_position_size,
        generate_candidates=args.generate_candidates,
        min_confidence=args.min_confidence,
        min_edge=args.min_edge,
        max_candidates=args.max_candidates,
        ui_mode=args.ui_mode,
        detect_relationships=args.detect_relationships,
        correlation_threshold=args.correlation_threshold,
        min_overlap_days=args.min_overlap_days,
    )


def _config_from_resolve_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        output_dir=Path(args.output_dir),
        max_events=int(args.max_events),
        page_limit=int(args.page_limit),
        request_timeout_s=float(args.timeout),
        max_retries=int(args.max_retries),
        include_resolved=True,
        resolved_lookback_days=int(args.lookback_days),
        http_pool_maxsize=int(args.http_pool_maxsize),
    )


def main() -> None:
    argv = sys.argv[1:]
    command = "run"
    if argv and not argv[0].startswith("-") and argv[0] in {"run", "resolve"}:
        command = argv[0]
        argv = argv[1:]

    parser = _build_resolve_parser() if command == "resolve" else _build_run_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = _config_from_resolve_args(args) if command == "resolve" else _config_from_run_args(args)
    runner = PipelineRunner(config)
    outputs = runner.run_resolutions_only() if command == "resolve" else runner.run()

    for key, value in outputs.items():
        logging.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
