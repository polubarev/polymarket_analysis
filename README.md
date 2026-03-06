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

## Data Directories

```
data/
  dev/    ← local pipeline runs (default --output-dir)
  prod/   ← snapshot pulled from GCS (production)
```

Pull latest production data from GCS:

```bash
python -m polymarket_pipeline.gcs_sync \
  --mode download \
  --local-dir data/prod \
  --gcs-uri gs://polymarket-analysis-488820-polymarket-data/prod_v2
```

## Run

```bash
polymarket-pipeline \
  --output-dir data/dev \
  --max-events 2000 \
  --window-days 90 \
  --interval 1h \
  --yes-only-binary
```

## Phase 2 Strategy Mode

Run the full research pipeline (resolved ingestion, optional orderbook/volume, signals, backtest, candidates):

```bash
polymarket-pipeline \
  --output-dir data/dev \
  --max-events 1000 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --include-resolved \
  --resolved-lookback-days 365 \
  --snapshot-orderbook \
  --ingest-volume \
  --run-signals \
  --backtest \
  --sizing-mode kelly \
  --generate-candidates
```

Resolved-only backfill command:

```bash
polymarket-pipeline resolve --output-dir data/dev --lookback-days 365
```

Optional trades validation sample:

```bash
polymarket-pipeline --fetch-trades-sample 100
```

## Data Discovery UI

After the parquet outputs exist, launch the interactive discovery app:

```bash
streamlit run polymarket_pipeline/discovery_ui.py -- --data-dir data/dev
```

Enable full research tabs (backtests, candidates, signal analysis):

```bash
streamlit run polymarket_pipeline/discovery_ui.py -- --data-dir data/dev --ui-mode full
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
  --output-dir data/dev \
  --max-events 300 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --price-fetch-workers 2 \
  --rate-window-s 10 \
  --fetch-priority-mode history_first \
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
  --output-dir data/dev \
  --max-events 1000 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --price-fetch-workers 2 \
  --rate-window-s 10 \
  --http-pool-maxsize 64 \
  --incremental-mode skip \
  --skip-raw-price-files
```

Recommended larger refresh (more coverage, longer runtime):

```bash
polymarket-pipeline \
  --output-dir data/dev \
  --max-events 2000 \
  --window-days 30 \
  --interval 1h \
  --yes-only-binary \
  --price-fetch-workers 2 \
  --rate-window-s 10 \
  --http-pool-maxsize 64 \
  --incremental-mode tail \
  --incremental-overlap-points 2 \
  --skip-raw-price-files
```

When you need richer multi-outcome analysis, replace `--yes-only-binary` with `--all-outcomes`.

## Outputs

- `data/dev/raw/events_YYYYMMDD.jsonl`
- `data/dev/raw/prices/<asset_id>.parquet`
- `data/dev/raw/trades/<condition_id>.parquet` (optional)
- `data/dev/events.parquet`
- `data/dev/markets.parquet`
- `data/dev/tokens.parquet`
- `data/dev/price_history.parquet`
- `data/dev/resolutions.parquet` (when `--include-resolved`)
- `data/dev/orderbook_snapshots.parquet` (when `--snapshot-orderbook`)
- `data/dev/volume_bars.parquet` (when `--ingest-volume`)
- `data/dev/market_quality.parquet`
- `data/dev/features.parquet`
- `data/dev/clusters.parquet`
- `data/dev/signals.parquet` (when `--run-signals`)
- `data/dev/backtest_results.parquet` + `data/dev/analysis/backtest_summary.json` (when `--backtest`)
- `data/dev/trade_candidates.parquet` + `data/dev/analysis/trade_candidates.json` (when `--generate-candidates`)
- `data/dev/analysis/signal_debug.json` (when `--signal-debug`)
- `data/dev/market_relationships.parquet` (when `--detect-relationships`)
- `data/dev/analysis/report.json`
- `data/dev/analysis/feature_metadata.json`
- `data/dev/analysis/health_check.json`
- `data/dev/pipeline_runs.parquet`
- `data/dev/analysis/coverage_by_bet_type.png`
- `data/dev/analysis/cluster_<id>.png`

## GCP Scheduling

Deployable Cloud Run Jobs setup is included:

- container entrypoint: `scripts/cloud_run_entrypoint.sh`
- deploy script: `scripts/deploy_cloud_run_jobs.sh`
- runbook: `docs/gcp_cloud_run.md`

This gives you:

- hourly incremental ingestion (`tail` mode)
- nightly reconciliation (`--no-incremental-prices`)
- daily research generation (`--snapshot-orderbook --ingest-volume --run-signals --backtest --generate-candidates`)
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
- Gamma event calls are disk-cached under `data/dev/.cache/gamma_events`.
- Price history uses `asset_id`; trades use `condition_id`.
- Quality gating can be tuned via CLI flags (`--quality-min-points`, `--quality-max-missing-ratio`, etc.).
- Price ingestion can run in parallel (`--price-fetch-workers`) and use either `tail` refresh (default) or `skip` mode.
- Scheduled ingestion should stay conservative: `--price-fetch-workers 2 --rate-window-s 10`.
- Fetch ordering can be tuned with `--fetch-priority-mode history_first|category_round_robin`.
- Use `--signal-debug` to write `analysis/signal_debug.json` when diagnosing empty signal runs.
- For high worker counts, set `--http-pool-maxsize` >= workers to avoid connection-pool warnings.
- `--skip-raw-price-files` is usually best for speed; remove it if you need per-asset raw parquet snapshots.
