# GCP Scheduled Pipeline (Cloud Run Jobs)

This setup runs the pipeline on a schedule and persists outputs to GCS.

## Architecture

- Containerized pipeline in Cloud Run Jobs.
- Hourly job: incremental `tail` refresh for fresh history points.
- Nightly job: reconciliation run with `--no-incremental-prices` for data repair.
- Daily research job: enriches with orderbook + volume, then runs signals, backtests, and trade-candidate generation.
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

- Hourly incremental: `10 * * * *` (default in deploy script)
- Nightly reconcile: `25 3 * * *` (default in deploy script, UTC)
- Daily research: `55 4 * * *` (default in deploy script, UTC)

Adjust with env vars before deploy:

```bash
export HOURLY_CRON="15 * * * *"
export NIGHTLY_CRON="40 2 * * *"
export RESEARCH_CRON="55 4 * * *"
export TIME_ZONE="America/New_York"
```

## Cost Guidance

Exact pricing changes by region and time, so use the GCP calculator for final numbers. For this workload, typical monthly cost is usually in low tens of USD if you keep:

- hourly profile at ~300 events, 30d window
- nightly reconcile at ~1000 events, 30d window
- daily research at ~1000 events with signals/backtests enabled
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
- The daily research job is the canonical source for `signals.parquet`, `backtest_results.parquet`, `trade_candidates.parquet`, and `analysis/signal_debug.json`.
- Cloud Run filesystem is ephemeral; GCS upload is what makes runs persistent.
- `scripts/cloud_run_entrypoint.sh` also supports `ENV_FILE=...` for local/container runs.
