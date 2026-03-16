"""Run mark-to-market backtest on existing prod data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polymarket_pipeline.backtesting.mtm import MtmConfig, run_mtm_backtest
from polymarket_pipeline.storage import read_parquet_if_exists


def main() -> None:
    prod = Path("C:/Users/Igor/PycharmProjects/polymarket/data/prod")

    print("Loading data...")
    signals_df = pd.read_parquet(prod / "signals.parquet")
    price_history_df = pd.read_parquet(prod / "price_history.parquet")
    orderbook_df = read_parquet_if_exists(prod / "orderbook_snapshots.parquet")

    # Filter out old broken signals (mean_reversion_spike, calibration_mispricing)
    old_signals = {"mean_reversion_spike", "calibration_mispricing"}
    signals_df = signals_df[~signals_df["signal_name"].isin(old_signals)]
    print(f"  Signals: {len(signals_df)} (excluded old: {old_signals})")
    print(f"  Price history: {len(price_history_df):,}")
    print(f"  Signal types: {signals_df['signal_name'].value_counts().to_dict()}")

    config = MtmConfig(
        horizons_hours=[1, 6, 24, 72, 168],  # 1h, 6h, 1d, 3d, 7d
        spread_assumption=0.03,
        initial_capital=10_000.0,
        flat_position_size=100.0,
    )

    print(f"\nRunning MTM backtest (horizons: {config.horizons_hours}h)...")
    results_df, summary = run_mtm_backtest(
        signals_df=signals_df,
        price_history_df=price_history_df,
        orderbook_df=orderbook_df,
        config=config,
    )

    print(f"\nResults: {len(results_df)} rows")

    if "error" in summary:
        print(f"  Error: {summary['error']}")
        return

    # Print by horizon
    print("\n=== RESULTS BY HORIZON ===")
    print(f"{'Horizon':>10} {'Trades':>7} {'Hit%':>7} {'Win%':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8} {'AvgMove':>10}")
    print("-" * 75)
    for h_key, stats in sorted(summary.get("by_horizon", {}).items()):
        print(f"{h_key:>10} {stats['trades']:>7} {stats['hit_rate']:>7.1%} {stats['win_rate']:>7.1%} {stats['avg_pnl']:>9.2f} {stats['total_pnl']:>10.0f} {stats['sharpe']:>8.2f} {stats['avg_price_move']:>10.4f}")

    # Print by signal at shortest horizon
    print("\n=== RESULTS BY SIGNAL (shortest horizon) ===")
    print(f"{'Signal':>25} {'Trades':>7} {'Hit%':>7} {'Win%':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8}")
    print("-" * 82)
    for sig_name, stats in summary.get("by_signal", {}).items():
        print(f"{sig_name:>25} {stats['trades']:>7} {stats['hit_rate']:>7.1%} {stats['win_rate']:>7.1%} {stats['avg_pnl']:>9.2f} {stats['total_pnl']:>10.0f} {stats['sharpe']:>8.2f}")

    # Print by signal x horizon
    print("\n=== DETAILED: SIGNAL x HORIZON ===")
    print(f"{'Key':>40} {'Trades':>7} {'Hit%':>7} {'Win%':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8}")
    print("-" * 95)
    for key, stats in sorted(summary.get("by_signal_horizon", {}).items()):
        print(f"{key:>40} {stats['trades']:>7} {stats['hit_rate']:>7.1%} {stats['win_rate']:>7.1%} {stats['avg_pnl']:>9.2f} {stats['total_pnl']:>10.0f} {stats['sharpe']:>8.2f}")

    # Save
    out_dir = prod / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(prod / "mtm_backtest_results.parquet", index=False)
    with (out_dir / "mtm_backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"\nSaved to {prod / 'mtm_backtest_results.parquet'}")
    print(f"Summary: {out_dir / 'mtm_backtest_summary.json'}")


if __name__ == "__main__":
    main()
