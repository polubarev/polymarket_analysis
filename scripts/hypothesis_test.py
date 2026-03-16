"""Empirical hypothesis testing on price history data.

Tests whether our signal hypotheses actually predict price moves,
independent of the signal code itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    prod = Path("C:/Users/Igor/PycharmProjects/polymarket/data/prod")
    feat = pd.read_parquet(prod / "features.parquet")
    ph = pd.read_parquet(prod / "price_history.parquet")

    # Only tight-spread assets (tradeable)
    tight_ids = set(feat[feat["avg_spread"] <= 0.03]["asset_id"])
    ph = ph[ph["asset_id"].isin(tight_ids)].copy()
    ph.sort_values(["asset_id", "ts"], inplace=True)
    print(f"Analyzing {ph['asset_id'].nunique()} tight-spread assets, {len(ph):,} price points")

    # Compute forward returns at various horizons (in seconds)
    horizons = {
        "1h": 3600,
        "6h": 21600,
        "24h": 86400,
        "72h": 259200,
        "168h": 604800,
    }

    # Build a lookup: for each (asset_id, ts) -> price
    # Then for each observation, find the future price
    print("\nBuilding forward returns...")
    results = []
    for asset_id, group in ph.groupby("asset_id"):
        if len(group) < 24:  # Skip assets with too few data points
            continue
        ts_arr = group["ts"].values
        price_arr = group["price"].values

        for i in range(len(group)):
            current_ts = ts_arr[i]
            current_price = price_arr[i]

            # Compute 24h backward return
            target_ts_back = current_ts - 86400
            idx_back = np.searchsorted(ts_arr, target_ts_back)
            if idx_back < len(ts_arr) and abs(ts_arr[min(idx_back, len(ts_arr) - 1)] - target_ts_back) < 7200:
                price_24h_ago = price_arr[min(idx_back, len(ts_arr) - 1)]
                return_1d = current_price - price_24h_ago
            else:
                return_1d = np.nan

            # Compute forward returns
            fwd = {}
            for h_name, h_secs in horizons.items():
                target_ts = current_ts + h_secs
                idx = np.searchsorted(ts_arr, target_ts)
                if idx < len(ts_arr) and abs(ts_arr[idx] - target_ts) < 7200:  # within 2h
                    fwd[f"fwd_{h_name}"] = price_arr[idx] - current_price
                else:
                    fwd[f"fwd_{h_name}"] = np.nan

            results.append({
                "asset_id": asset_id,
                "ts": current_ts,
                "price": current_price,
                "return_1d": return_1d,
                **fwd,
            })

    df = pd.DataFrame(results)
    print(f"Total observations: {len(df):,}")

    # === HYPOTHESIS 1: Convergence to 1.0 ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Price > 0.85 → converges toward 1.0")
    print("=" * 70)
    high = df[df["price"] > 0.85]
    print(f"Observations with price > 0.85: {len(high):,}")
    for h in horizons:
        col = f"fwd_{h}"
        valid = high[col].dropna()
        if len(valid) == 0:
            continue
        positive = (valid > 0).sum()
        mean_move = valid.mean()
        median_move = valid.median()
        print(f"  {h:>5}: n={len(valid):>6}, up={positive/len(valid):.1%}, "
              f"mean_move={mean_move:+.4f}, median={median_move:+.4f}")

    # Sub-test: price > 0.90
    very_high = df[df["price"] > 0.90]
    print(f"\nPrice > 0.90: {len(very_high):,} observations")
    for h in horizons:
        col = f"fwd_{h}"
        valid = very_high[col].dropna()
        if len(valid) == 0:
            continue
        positive = (valid > 0).sum()
        mean_move = valid.mean()
        print(f"  {h:>5}: n={len(valid):>6}, up={positive/len(valid):.1%}, "
              f"mean_move={mean_move:+.4f}")

    # === HYPOTHESIS 2: Convergence to 0.0 ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Price < 0.15 → converges toward 0.0")
    print("=" * 70)
    low = df[df["price"] < 0.15]
    print(f"Observations with price < 0.15: {len(low):,}")
    for h in horizons:
        col = f"fwd_{h}"
        valid = low[col].dropna()
        if len(valid) == 0:
            continue
        negative = (valid < 0).sum()
        mean_move = valid.mean()
        median_move = valid.median()
        print(f"  {h:>5}: n={len(valid):>6}, down={negative/len(valid):.1%}, "
              f"mean_move={mean_move:+.4f}, median={median_move:+.4f}")

    # Sub-test: price < 0.05
    very_low = df[df["price"] < 0.05]
    print(f"\nPrice < 0.05: {len(very_low):,} observations")
    for h in horizons:
        col = f"fwd_{h}"
        valid = very_low[col].dropna()
        if len(valid) == 0:
            continue
        negative = (valid < 0).sum()
        mean_move = valid.mean()
        print(f"  {h:>5}: n={len(valid):>6}, down={negative/len(valid):.1%}, "
              f"mean_move={mean_move:+.4f}")

    # === HYPOTHESIS 3: Momentum continuation ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: |return_1d| > 2% → price continues in same direction")
    print("=" * 70)
    mom = df[df["return_1d"].abs() > 0.02].copy()
    mom["direction"] = np.where(mom["return_1d"] > 0, "up", "down")
    print(f"Observations with |return_1d| > 2%: {len(mom):,}")
    print(f"  Up moves: {(mom['direction'] == 'up').sum():,}, Down moves: {(mom['direction'] == 'down').sum():,}")

    for h in horizons:
        col = f"fwd_{h}"
        valid = mom[[col, "direction", "return_1d"]].dropna()
        if len(valid) == 0:
            continue
        # Continuation = forward move same sign as return_1d
        continuation = ((valid[col] > 0) & (valid["direction"] == "up")) | \
                       ((valid[col] < 0) & (valid["direction"] == "down"))
        mean_fwd = valid[col].mean()
        # By direction
        up_valid = valid[valid["direction"] == "up"]
        down_valid = valid[valid["direction"] == "down"]
        up_cont = (up_valid[col] > 0).mean() if len(up_valid) > 0 else 0
        down_cont = (down_valid[col] < 0).mean() if len(down_valid) > 0 else 0
        print(f"  {h:>5}: n={len(valid):>6}, continuation={continuation.mean():.1%}, "
              f"up_cont={up_cont:.1%}, down_cont={down_cont:.1%}, "
              f"mean_fwd={mean_fwd:+.4f}")

    # === HYPOTHESIS 3b: Stronger momentum threshold ===
    print("\n--- Momentum with |return_1d| > 5% ---")
    mom5 = df[df["return_1d"].abs() > 0.05].copy()
    mom5["direction"] = np.where(mom5["return_1d"] > 0, "up", "down")
    print(f"Observations with |return_1d| > 5%: {len(mom5):,}")

    for h in horizons:
        col = f"fwd_{h}"
        valid = mom5[[col, "direction"]].dropna()
        if len(valid) == 0:
            continue
        continuation = ((valid[col] > 0) & (valid["direction"] == "up")) | \
                       ((valid[col] < 0) & (valid["direction"] == "down"))
        print(f"  {h:>5}: n={len(valid):>6}, continuation={continuation.mean():.1%}")

    # === HYPOTHESIS 4: Mean reversion (for comparison) ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4 (sanity check): |return_1d| > 5% → MEAN REVERSION")
    print("=" * 70)
    for h in horizons:
        col = f"fwd_{h}"
        valid = mom5[[col, "direction"]].dropna()
        if len(valid) == 0:
            continue
        reversion = ((valid[col] < 0) & (valid["direction"] == "up")) | \
                    ((valid[col] > 0) & (valid["direction"] == "down"))
        print(f"  {h:>5}: n={len(valid):>6}, reversion={reversion.mean():.1%}")

    # === NET EDGE AFTER SPREAD ===
    print("\n" + "=" * 70)
    print("NET EDGE: Mean forward move MINUS 3% spread (round-trip)")
    print("=" * 70)
    spread = 0.03

    # Convergence buy (price > 0.85)
    print("\nConvergence buy (price > 0.85):")
    for h in horizons:
        col = f"fwd_{h}"
        valid = high[col].dropna()
        if len(valid) == 0:
            continue
        net = valid.mean() - spread
        profitable = (valid > spread).mean()
        print(f"  {h:>5}: mean_fwd={valid.mean():+.4f}, net_after_spread={net:+.4f}, "
              f"profitable_trades={profitable:.1%}")

    # Convergence sell (price < 0.15)
    print("\nConvergence sell (price < 0.15):")
    for h in horizons:
        col = f"fwd_{h}"
        valid = low[col].dropna()
        if len(valid) == 0:
            continue
        # For sell: profit when price goes DOWN, so -fwd_move - spread
        net = -valid.mean() - spread
        profitable = (valid < -spread).mean()
        print(f"  {h:>5}: mean_fwd={valid.mean():+.4f}, net_after_spread={net:+.4f}, "
              f"profitable_trades={profitable:.1%}")

    # Momentum: average abs forward move vs spread
    print("\nMomentum (|return_1d| > 2%):")
    for h in horizons:
        col = f"fwd_{h}"
        valid = mom[[col, "direction", "return_1d"]].dropna()
        if len(valid) == 0:
            continue
        # Directional profit: fwd * sign(return_1d) - spread
        directional_pnl = valid[col] * np.where(valid["direction"] == "up", 1, -1)
        net = directional_pnl.mean() - spread
        profitable = (directional_pnl > spread).mean()
        print(f"  {h:>5}: mean_directional={directional_pnl.mean():+.4f}, "
              f"net_after_spread={net:+.4f}, profitable={profitable:.1%}")


if __name__ == "__main__":
    main()
