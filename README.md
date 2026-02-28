# Polymarket Baseline Pipeline

Implements the baseline in `docs/project.md`:

- discovers active events/markets from Gamma
- extracts token asset IDs and condition IDs
- fetches CLOB price history per asset ID
- writes raw and normalized parquet/jsonl outputs
- computes coverage + bet-type summaries + simple shape clustering

## Install

```bash
python -m pip install -e .
```

## Run

```bash
polymarket-pipeline \
  --output-dir data \
  --max-events 2000 \
  --window-days 90 \
  --interval 1h \
  --yes-only-binary
```

Optional trades validation sample:

```bash
polymarket-pipeline --fetch-trades-sample 100
```

## Data Discovery UI

After the parquet outputs exist, launch the interactive discovery app:

```bash
streamlit run polymarket_pipeline/discovery_ui.py -- --data-dir data
```

What it includes:

- filterable bet list (search, tags, market type, status, liquidity, quality gates)
- per-bet outcome price chart with token-level quality checks
- market map scatter (liquidity vs return) and biggest movers table
- tag coverage view and cluster distribution summary

## Recommended Parameters

Use the same `--output-dir` across runs to benefit from incremental updates.

Recommended daily incremental profile:

```bash
polymarket-pipeline \
  --output-dir data \
  --max-events 300 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --price-fetch-workers 24 \
  --http-pool-maxsize 64 \
  --quality-min-points 24 \
  --quality-max-missing-ratio 0.9 \
  --quality-min-price-range 0.005 \
  --incremental-mode tail \
  --incremental-overlap-points 2 \
  --skip-raw-price-files
```

Fastest ad-hoc rerun (skips already-ingested assets):

```bash
polymarket-pipeline \
  --output-dir data \
  --max-events 1000 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --price-fetch-workers 24 \
  --http-pool-maxsize 64 \
  --incremental-mode skip \
  --skip-raw-price-files
```

Recommended larger refresh (more coverage, longer runtime):

```bash
polymarket-pipeline \
  --output-dir data \
  --max-events 2000 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --price-fetch-workers 24 \
  --http-pool-maxsize 64 \
  --incremental-mode tail \
  --incremental-overlap-points 2 \
  --skip-raw-price-files
```

When you need richer multi-outcome analysis, replace `--yes-only-binary` with `--all-outcomes`.

## Outputs

- `data/raw/events_YYYYMMDD.jsonl`
- `data/raw/prices/<asset_id>.parquet`
- `data/raw/trades/<condition_id>.parquet` (optional)
- `data/events.parquet`
- `data/markets.parquet`
- `data/tokens.parquet`
- `data/price_history.parquet`
- `data/market_quality.parquet`
- `data/clusters.parquet`
- `data/analysis/report.json`
- `data/analysis/coverage_by_bet_type.png`
- `data/analysis/cluster_<id>.png`

## GCP Scheduling

Deployable Cloud Run Jobs setup is included:

- container entrypoint: `scripts/cloud_run_entrypoint.sh`
- deploy script: `scripts/deploy_cloud_run_jobs.sh`
- runbook: `docs/gcp_cloud_run.md`

This gives you:

- hourly incremental ingestion (`tail` mode)
- nightly reconciliation (`--no-incremental-prices`)
- persistence to GCS via `GCS_OUTPUT_URI`

Quick setup with `.env`:

```bash
cp .env.example .env
# edit .env
bash scripts/deploy_cloud_run_jobs.sh
```

## Notes

- Rate limits are enforced with token buckets per endpoint group.
- Retries use exponential backoff with jitter.
- Gamma event calls are disk-cached under `data/.cache/gamma_events`.
- Price history uses `asset_id`; trades use `condition_id`.
- Quality gating can be tuned via CLI flags (`--quality-min-points`, `--quality-max-missing-ratio`, etc.).
- Price ingestion can run in parallel (`--price-fetch-workers`) and use either `tail` refresh (default) or `skip` mode.
- For high worker counts, set `--http-pool-maxsize` >= workers to avoid connection-pool warnings.
- `--skip-raw-price-files` is usually best for speed; remove it if you need per-asset raw parquet snapshots.
