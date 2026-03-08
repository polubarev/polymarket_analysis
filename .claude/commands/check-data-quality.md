# Data Quality Check for Polymarket Pipeline

Run the persistent audit script against the pipeline's parquet output.

## Instructions

### Step 0: Find Python

Always use the venv Python (avoid Windows Store alias):

```
PYTHON="<project_root>/.venv/Scripts/python.exe"
# fallback: <project_root>/.venv/bin/python
```

Verify with `$PYTHON --version`.

### Step 1: Run the audit script

```bash
$PYTHON scripts/dq_check.py
```

Optional flags:
- `--dir data/prod`  — explicit output directory (default: auto-detects `data/prod` else `data/`)

The script covers all 8 audit steps and automatically:
- Tracks row-count deltas vs the previous run (stored in `scripts/.dq_baseline.json`)
- Distinguishes expected nulls (no `--snapshot-orderbook` / `--ingest-volume` / `--include-resolved`) from unexpected ones
- Prints a structured PASS / CRITICAL / WARNING / VERDICT summary

### Step 2: Interpret & report results

After running, produce a structured markdown summary with:

1. **What changed since last run** — row-count deltas table
2. **PASS** — things that look good
3. **CRITICAL** — blockers (unexpected nulls, invalid prices, missing files, time-travel gaps)
4. **WARNING** — issues to watch (rate limits, stuck coverage, zero signals, null momentum cols)
5. **VERDICT** — "✅ Ready for analysis" or "❌ NOT READY" with next steps

### Common Issues & Root Causes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Price history spans < 1 day | CLOB API `fidelity` param returns fixed count not time range | Omit fidelity or paginate differently |
| All `num_points` ≤ 5 in features | Only ~1h of data resampled to 1h buckets | Fix price ingestion first |
| 8 microstructure/volume cols 100% null | Normal — need `--snapshot-orderbook --ingest-volume` | Run with those flags if needed |
| `return_30d` 100% null | Need 30+ days of history | Accumulate more data |
| `resolution_outcome` 100% null | Normal — need `--include-resolved` | Run with that flag if needed |
| Low quality_pass rate (< 20%) | `check_price_range` fails with short price window | Fix price history span first |
| HTTP 429 count rising | Rate limiting — too many concurrent fetchers | Reduce `--price-fetch-workers` or increase `--rate-window-s` |
| `price_coverage_pct` stuck | Same tokens failing every run | Check logs for repeated 429s or timeouts on specific assets |
| `signals_generated` = 0 every run | Signal engine running but producing nothing | Check signal thresholds and feature availability |
