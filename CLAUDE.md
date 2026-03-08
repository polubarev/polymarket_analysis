# Polymarket Pipeline

## Project Overview
Automated data pipeline for Polymarket prediction markets. Ingests events, markets, tokens, and price history from Polymarket APIs, computes features, clusters assets, generates trading signals, runs backtests, and produces trade candidates. Deployed on Google Cloud Run with hourly/nightly/research schedules.

## Quick Start
```bash
# Activate venv
source .venv/Scripts/activate  # or .venv/bin/activate on Linux

# Run pipeline locally (dev)
polymarket-pipeline --output-dir data/dev --pipeline-profile local

# Run DQ check
python scripts/dq_check.py --dir data/prod

# Run tests
python -m unittest discover tests/ -v

# Deploy to Cloud Run (needs gcloud CLI)
bash scripts/deploy_cloud_run_jobs.sh
```

## Architecture
```
Polymarket APIs (Gamma, CLOB, Data)
        ↓
  polymarket_pipeline/
    client.py          — API clients with rate limiting
    normalize.py       — Extract/transform tables
    pipeline.py        — Main orchestrator (PipelineRunner)
    config.py          — PipelineConfig dataclass
    cli.py             — CLI entry point
    analysis.py        — Feature engineering, clustering, reports
    signals_runner.py  — Signal generation
    backtesting/       — Backtest engine
    monitoring/        — Health checks, run tracking
    storage.py         — Parquet I/O helpers
    gcs_sync.py        — GCS upload/download
  scripts/
    cloud_run_entrypoint.sh  — Container entry point
    deploy_cloud_run_jobs.sh — Cloud Run + Scheduler deploy
    dq_check.py              — Data quality audit
```

## Data Pipeline Flow
```
Events → Markets → Tokens → Price History → Features → Clusters
                                          → Orderbook Snapshots
                                          → Volume Bars
                     Resolutions ──┐
                     Signals ──────┤→ Backtest → Trade Candidates
```

## Key Files
- `data/prod/` — Production data (GCS-synced)
- `data/dev/` — Local development data
- `.env` — GCP credentials and deploy config (not committed)

## Cloud Run Jobs (3 schedules)
| Job | Schedule | Profile | What it does |
|-----|----------|---------|--------------|
| polymarket-hourly | `:10 * * * *` | ingest-hourly | Ingest + orderbook + volume |
| polymarket-nightly | `25 3 * * *` | ingest-nightly | Full refetch + signals + backtest + candidates + resolutions |
| polymarket-daily-research | `55 4 * * *` | research-daily | Same as nightly + signal debug |

## Current Status & Next Steps
See `STATUS.md` for current pipeline state, recent changes, and planned next steps.
Read this file at the start of every new conversation.

## Environment Notes
- Python 3.14, venv at `.venv/`
- In WSL: use `/c/Users/Igor/PycharmProjects/polymarket/.venv/Scripts/python.exe`
- `gcloud` is only available from PowerShell/CMD, not WSL bash
- Worktrees are used for parallel work: `.claude/worktrees/<name>/`
