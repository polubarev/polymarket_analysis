from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .pipeline import PipelineRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polymarket baseline data pipeline")
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
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    yes_only = True
    if args.all_outcomes:
        yes_only = False
    elif args.yes_only_binary:
        yes_only = True

    config = PipelineConfig(
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
    )
    runner = PipelineRunner(config)
    outputs = runner.run()

    for key, value in outputs.items():
        logging.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
