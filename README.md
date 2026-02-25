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

## Outputs

- `data/raw/events_YYYYMMDD.jsonl`
- `data/raw/prices/<asset_id>.parquet`
- `data/raw/trades/<condition_id>.parquet` (optional)
- `data/events.parquet`
- `data/markets.parquet`
- `data/tokens.parquet`
- `data/price_history.parquet`
- `data/clusters.parquet`
- `data/analysis/report.json`
- `data/analysis/coverage_by_bet_type.png`
- `data/analysis/cluster_<id>.png`

## Notes

- Rate limits are enforced with token buckets per endpoint group.
- Retries use exponential backoff with jitter.
- Gamma event calls are disk-cached under `data/.cache/gamma_events`.
- Price history uses `asset_id`; trades use `condition_id`.
