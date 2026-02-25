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
    )
    runner = PipelineRunner(config)
    outputs = runner.run()

    for key, value in outputs.items():
        logging.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
