# GCP Scheduled Pipeline (Cloud Run Jobs)

This setup runs the pipeline on a schedule and persists outputs to GCS.

## Architecture

- Containerized pipeline in Cloud Run Jobs.
- `polymarket-hourly`: daily ingest run focused on events, prices, and volume.
- `polymarket-nightly-reconcile`: weekly repair run with `--no-incremental-prices` plus resolved-market refresh.
- `polymarket-daily-research`: weekly analysis-only run that reads stored parquet data and writes research artifacts without calling external APIs.
- Cloud Scheduler triggers the jobs through the Cloud Run Jobs API.
- Outputs are uploaded to `gs://...` at the end of each run.

## Prerequisites

- `gcloud` configured with your project.
- Artifact Registry, Cloud Run, Cloud Build, Cloud Scheduler APIs enabled.
- A GCS bucket for pipeline artifacts.
- IAM:
  - deployer needs permissions for Cloud Build, Cloud Run Jobs, Artifact Registry, Scheduler
  - scheduler service account needs permission to run Cloud Run Jobs

## Deploy

Create `.env` from the template:

```bash
cp .env.example .env
# edit .env with your values
```

Deploy image + jobs + scheduler:

```bash
bash scripts/deploy_cloud_run_jobs.sh
```

`scripts/deploy_cloud_run_jobs.sh` auto-loads `.env`. To use a different env file:

```bash
ENV_FILE=.env.prod bash scripts/deploy_cloud_run_jobs.sh
```

## Trigger Manually

```bash
gcloud run jobs execute polymarket-hourly --region us-central1
gcloud run jobs execute polymarket-nightly-reconcile --region us-central1
gcloud run jobs execute polymarket-daily-research --region us-central1
```

## Schedules

- Daily ingest: `10 3 * * *` (default in deploy script, UTC)
- Weekly reconcile: `25 3 * * 6` (default in deploy script, UTC)
- Weekly research: `30 5 * * 6` (default in deploy script, UTC)

Research is scheduled on the same UTC day as reconcile so it runs after fresh repair/resolution data lands instead of working from day-old inputs.

Adjust with env vars before deploy:

```bash
export HOURLY_CRON="10 3 * * *"
export NIGHTLY_CRON="25 3 * * 6"
export RESEARCH_CRON="30 5 * * 6"
export TIME_ZONE="America/New_York"
```

## Job Profiles

- Daily ingest:
  - `PIPELINE_COMMAND=run`
  - `PIPELINE_PROFILE=ingest-daily`
  - `MAX_EVENTS=1000`
  - `INGEST_VOLUME=true`
  - `SNAPSHOT_ORDERBOOK=false`
  - `NO_INCREMENTAL_PRICES=false`
  - `INCLUDE_RESOLVED=false`
- Weekly reconcile:
  - `PIPELINE_COMMAND=run`
  - `PIPELINE_PROFILE=reconcile-weekly`
  - `MAX_EVENTS=1000`
  - `INGEST_VOLUME=true`
  - `SNAPSHOT_ORDERBOOK=false`
  - `NO_INCREMENTAL_PRICES=true`
  - `INCLUDE_RESOLVED=true`
- Weekly research:
  - `PIPELINE_COMMAND=analyze`
  - `PIPELINE_PROFILE=research-weekly`
  - Reads `events.parquet`, `markets.parquet`, `tokens.parquet`, `price_history.parquet`, and optional `volume_bars.parquet` / `resolutions.parquet`
  - Runs features, market quality, clustering, signals, backtests, candidates, signal debug, health, and run tracking
  - Ignores any stored `orderbook_snapshots.parquet` so stale orderbook data cannot affect research outputs

## Cost Guidance

Exact pricing changes by region and time, so use the GCP calculator for final numbers. For this workload, typical monthly cost is usually in low tens of USD if you keep:

- daily ingest at ~1000 events, 30d window
- weekly reconcile at ~1000 events, 30d window
- weekly research on stored parquet data with signals/backtests enabled
- raw per-asset price files disabled

Main cost drivers:

- Cloud Run vCPU/memory runtime
- Cloud Build image builds
- GCS storage and uploads (especially `price_history.parquet`)

## Notes

- Incremental behavior is controlled by `--incremental-mode`:
  - `tail` (recommended for scheduled ingestion): refreshes each asset near its latest point
  - `skip` (fastest for ad-hoc reruns): skips assets that already have interval data
- Scheduled jobs default to `PRICE_FETCH_WORKERS=2` and `RATE_WINDOW_S=10` to keep CLOB 429s down.
- `FETCH_PRIORITY_MODE=history_first` prioritizes assets that already have history before spending budget on uncovered assets.
- The weekly research job is the canonical source for `signals.parquet`, `backtest_results.parquet`, `trade_candidates.parquet`, and `analysis/signal_debug.json`.
- Health thresholds stay unchanged in budget mode: prices within 24 hours and resolutions within 7 days.
- `scripts/cloud_run_entrypoint.sh` now accepts `PIPELINE_COMMAND=run|resolve|analyze`; download and upload still wrap every command.
- Cloud Run filesystem is ephemeral; GCS upload is what makes runs persistent.
- `scripts/cloud_run_entrypoint.sh` also supports `ENV_FILE=...` for local/container runs.

## Rollback

To revert budget mode:

1. Restore the old cron defaults and job env payloads in `scripts/deploy_cloud_run_jobs.sh`.
2. Redeploy with `bash scripts/deploy_cloud_run_jobs.sh`.
3. Keep the same GCS path and parquet schemas.

Rollback does not require schema migrations or data backfills because budget mode does not change parquet layouts or storage locations.
