# Pipeline Status — PAUSED

**Last updated: 2026-03-15**
**Status: PAUSED / ARCHIVED** — no active development

## Why Paused

After 3 rounds of signal development and rigorous empirical testing on 991K price observations across 1,067 tradeable assets, we conclusively proved that **statistical price-pattern signals cannot beat the bid-ask spread on Polymarket**.

| Hypothesis | Result | Mean Forward Move | Net After Spread |
|---|---|---|---|
| Price > 0.85 → converges to 1.0 | No edge | +0.03% at 24h | **-3.0%** |
| Price < 0.15 → converges to 0.0 | No edge | +0.04% wrong dir | **-3.0%** |
| Momentum (big move continues) | Negative edge | -1.0% at 24h | **-4.0%** |
| Mean reversion | Barely better than random | +1.0% at 24h | **-2.0%** |

Even on the 88 most liquid assets (spread ≤1%, volume ≥$500/day): best case is +0.5% at 72h, still -0.5% net after spread.

**Root cause**: Polymarket is efficient. Prices are event-driven (news), not pattern-driven. Statistical edges are ~0% before costs. The 1-3% spread kills everything.

## What Works & What Doesn't

### ✅ Working well
- Data pipeline (ingestion, normalization, storage)
- Feature engineering (30+ features per asset)
- Orderbook + volume bar collection
- MTM backtesting engine
- Cloud Run deployment (3 scheduled jobs)
- GCS sync for data persistence

### ❌ Not viable
- Mean reversion signals (deleted)
- Convergence signals (no edge after spread)
- Momentum signals (prices actually revert)
- Orderbook imbalance signals (data too sparse)
- Any market-order strategy on Polymarket

## To Resume: What Would Need to Change

Only resume if you have one of:
1. **Limit order execution** — CLOB API for placing limit orders at mid-price (0% spread). The convergence signal has +0.5% raw edge that becomes profitable at zero spread.
2. **Informational edge** — news/event data source that lets you predict resolution outcomes before the market prices them.
3. **A specific thesis** — don't spend time unless you have a concrete idea to test.

## Before Resuming: Quick Start

```bash
# Read this file first
# Activate venv
source .venv/Scripts/activate

# Sync latest data from GCS
source .env && python -m polymarket_pipeline.gcs_sync --local-dir data/prod --gcs-uri "$GCS_OUTPUT_URI" --mode download

# Run DQ check
PYTHONIOENCODING=utf-8 python scripts/dq_check.py --dir data/prod

# Run hypothesis test (validates whether anything changed)
PYTHONIOENCODING=utf-8 python scripts/hypothesis_test.py
```

## Cloud Run: Consider Stopping

The pipeline is still running on Cloud Run (daily + weekly schedules), costing money. If pausing indefinitely:
```powershell
# From PowerShell (not WSL):
gcloud scheduler jobs pause polymarket-hourly --location us-central1
gcloud scheduler jobs pause polymarket-nightly --location us-central1
gcloud scheduler jobs pause polymarket-daily-research --location us-central1
```

## Data Snapshot (as of 2026-03-14)

| File | Rows |
|------|------|
| events | 944 |
| markets | 5,679 |
| tokens | 11,358 |
| price_history | 3,923,900 |
| features | 6,701 |
| orderbook_snapshots | 5,672 |
| volume_bars | 63,748 |
| resolutions | 7 |

## Project Timeline
- **Feb 2026**: Pipeline built, deployed to Cloud Run
- **Mar 8**: Full feature deploy (all outcomes, orderbook, volume, backtest)
- **Mar 14**: Signal analysis begins, MTM backtest built, all 3 signals proven unprofitable
- **Mar 15**: Empirical hypothesis test on raw data confirms no statistical edge exists. **Project paused.**

## Worktree Note
Signal rework code lives in worktree `bold-shirley` (not merged to main). The code is correct but the strategy doesn't work, so it's not worth merging.
