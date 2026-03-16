"""Run signals + backtest + candidates on existing prod data (no API calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polymarket_pipeline.config import PipelineConfig
from polymarket_pipeline.signals_runner import run_signal_generation
from polymarket_pipeline.backtesting.engine import BacktestConfig, run_backtest
from polymarket_pipeline.signals_runner import generate_trade_candidates
from polymarket_pipeline.storage import upsert_parquet, read_parquet_if_exists


def main() -> None:
    prod_dir = Path("C:/Users/Igor/PycharmProjects/polymarket/data/prod")

    # Load existing data
    print("Loading existing prod data...")
    features_df = pd.read_parquet(prod_dir / "features.parquet")
    tokens_df = pd.read_parquet(prod_dir / "tokens.parquet")
    markets_df = pd.read_parquet(prod_dir / "markets.parquet")
    events_df = pd.read_parquet(prod_dir / "events.parquet")
    price_history_df = pd.read_parquet(prod_dir / "price_history.parquet")
    volume_bars_df = read_parquet_if_exists(prod_dir / "volume_bars.parquet")
    resolutions_df = read_parquet_if_exists(prod_dir / "resolutions.parquet")
    orderbook_df = read_parquet_if_exists(prod_dir / "orderbook_snapshots.parquet")
    quality_df = read_parquet_if_exists(prod_dir / "market_quality.parquet")
    clusters_df = read_parquet_if_exists(prod_dir / "clusters.parquet")

    print(f"  features: {len(features_df):,}")
    print(f"  tokens: {len(tokens_df):,}")
    print(f"  markets: {len(markets_df):,}")
    print(f"  price_history: {len(price_history_df):,}")
    print(f"  volume_bars: {len(volume_bars_df) if volume_bars_df is not None else 0:,}")
    print(f"  resolutions: {len(resolutions_df) if resolutions_df is not None else 0:,}")

    config = PipelineConfig(
        output_dir=prod_dir,
        run_signals=True,
        run_backtest=True,
        generate_candidates=True,
        signal_debug=True,
        signal_debug_limit=20,
    )

    # 1. Generate signals
    print("\n--- Running signal generation ---")
    signals_new, signal_registry, signal_debug_payload = run_signal_generation(
        config=config,
        features_df=features_df,
        target_tokens_df=tokens_df,
        markets_df=markets_df,
        events_df=events_df,
        price_history_df=price_history_df,
        volume_bars_df=volume_bars_df if volume_bars_df is not None else pd.DataFrame(),
        resolutions_df=resolutions_df if resolutions_df is not None else pd.DataFrame(),
    )
    print(f"  Signals generated: {len(signals_new)}")
    if len(signals_new) > 0:
        signals_path = prod_dir / "signals.parquet"
        signals_df = upsert_parquet(
            signals_path,
            signals_new,
            dedupe_keys=["signal_name", "asset_id", "market_id", "ts"],
            sort_keys=["ts", "signal_name", "asset_id"],
        )
        print(f"  Total signals (after upsert): {len(signals_df)}")
        print(f"  Signal types: {signals_df['signal_name'].value_counts().to_dict()}")

        if signal_debug_payload:
            debug_path = prod_dir / "analysis" / "signal_debug.json"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("w", encoding="utf-8") as f:
                json.dump(signal_debug_payload, f, indent=2, ensure_ascii=True)
            print(f"  Signal debug written to {debug_path}")
    else:
        signals_df = read_parquet_if_exists(prod_dir / "signals.parquet")
        if signals_df is None:
            signals_df = pd.DataFrame()
        print("  No new signals generated, using existing signals file")

    # 2. Run backtest
    print("\n--- Running backtest ---")
    if len(signals_df) == 0:
        print("  SKIP: No signals to backtest")
    else:
        backtest_results_df, backtest_summary = run_backtest(
            signals_df=signals_df,
            resolutions_df=resolutions_df if resolutions_df is not None else pd.DataFrame(),
            orderbook_df=orderbook_df if orderbook_df is not None else pd.DataFrame(),
            config=BacktestConfig(
                start_date=None,
                end_date=None,
                initial_capital=10_000.0,
                spread_assumption=0.03,
                max_positions=50,
                stop_loss=0.50,
                timeout_days=90,
                sizing_mode="flat",
                kelly_fraction=0.25,
                max_position_pct=0.05,
                min_position_size=1.0,
                flat_position_size=100.0,
            ),
        )
        backtest_path = prod_dir / "backtest_results.parquet"
        backtest_results_df.to_parquet(backtest_path, index=False)
        summary_path = prod_dir / "analysis" / "backtest_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(backtest_summary, f, indent=2, ensure_ascii=True)
        print(f"  Backtest results: {len(backtest_results_df)} rows")
        print(f"  Summary: {json.dumps(backtest_summary, indent=2)}")

    # 3. Generate trade candidates
    print("\n--- Generating trade candidates ---")
    if len(signals_df) == 0:
        print("  SKIP: No signals for candidates")
    else:
        candidates_payload, candidates_df = generate_trade_candidates(
            config=config,
            signals_df=signals_df,
            signal_registry=signal_registry,
            markets_df=markets_df,
            features_df=features_df,
            orderbook_df=orderbook_df if orderbook_df is not None else pd.DataFrame(),
            quality_df=quality_df,
            clusters_df=clusters_df,
            bankroll=10_000.0,
        )
        candidates_path = prod_dir / "trade_candidates.parquet"
        candidates_df = upsert_parquet(
            candidates_path,
            candidates_df,
            dedupe_keys=["asset_id", "signal_name"],
            sort_keys=["rank"],
        )
        candidates_json = prod_dir / "analysis" / "trade_candidates.json"
        with candidates_json.open("w", encoding="utf-8") as f:
            json.dump(candidates_payload, f, indent=2, ensure_ascii=True)
        print(f"  Trade candidates: {len(candidates_df)}")
        if len(candidates_df) > 0:
            cols = ["rank", "signal_name", "direction", "confidence", "edge"]
            available = [c for c in cols if c in candidates_df.columns]
            print(candidates_df[available].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
