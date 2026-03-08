# Pipeline Status & Next Steps

**Last updated: 2026-03-08**

## Recent Session Summary (2026-03-08)

### What was done
1. **Ran full data quality audit** (`python scripts/dq_check.py --dir data/prod`)
   - 18 passes, 0 criticals (after fixes), 23 warnings
   - Verdict: READY FOR ANALYSIS

2. **Fixed false-alarm criticals in DQ check**
   - `pipeline_runs.parquet` had 57% nulls in `signals_expected` and `pipeline_profile` — these were historical rows from before those columns existed
   - Backfilled 73 old rows with defaults (`False` / `"ingest-hourly"`)
   - Added these columns to `EXPECTED_NULL_COLS` in `dq_check.py`
   - Fixed FutureWarning on `fillna` downcasting

3. **Enabled full pipeline features across all Cloud Run jobs**
   - `yes_only_binary` → `False` (fetch both Yes AND No tokens; expect coverage ~59% → ~95%)
   - Hourly job: added `--snapshot-orderbook` and `--ingest-volume`
   - Nightly job: added signals, backtest, candidates, resolutions, orderbook, volume
   - Research job: added `INCLUDE_RESOLVED=true`
   - Added `INCLUDE_RESOLVED` env var support to `cloud_run_entrypoint.sh`

4. **Deployed to Cloud Run** — all 3 jobs + schedulers updated, image rebuilt

### Commit
`26a8dec` — "Enable full pipeline: all outcomes, orderbook, volume, backtest" (merged to main)

---

## Current Data Snapshot (data/prod, 2026-03-08)

| File | Rows | Notes |
|------|------|-------|
| events | 943 | |
| markets | 5,672 | |
| tokens | 11,344 | |
| price_history | 1,915,514 | 385 MB, 35.1 days |
| features | 3,351 | |
| clusters | 3,124 | 8 clusters |
| market_quality | 5,672 | 55.1% quality pass |
| pipeline_runs | 129 | all HEALTHY |
| signals | 474 | |
| backtest_results | 0 | was disabled, now enabled |
| trade_candidates | 20 | |
| orderbook_snapshots | 5,672 | |
| volume_bars | 39,436 | |

Coverage funnel: Events 943 → Markets 5,672 → Tokens 11,344 → w/price 3,364 (29.7% of all tokens) → Features 3,351 → Clustered 3,124

Pipeline runs stable: 10 consecutive HEALTHY runs, ~860s each, hourly cadence.

---

## Next Steps (Prioritized)

### Immediate — verify deploy worked (next session)
- [ ] Wait for 1-2 hourly runs after deploy, then run DQ check again
- [ ] Verify `price_coverage_pct` jumped from 59% toward 95%
- [ ] Verify orderbook/volume feature nulls are filling in
- [ ] After nightly run: check `backtest_results.parquet` has rows
- [ ] After nightly run: check `resolutions.parquet` appeared

### Short-term — data quality improvements
- [ ] Fix `markets.liquidity` dtype (currently `str`, should be `float`) in `normalize.py`
- [ ] Fix `pct_lifetime_elapsed` computation bug (values >1 and <0 for resolved markets)
- [ ] Add logging to `client.py` for HTTP 400/404 failures (currently silently returns `[]`)
- [ ] Investigate which tokens still fail API calls after all-outcomes enabled

### Medium-term — analysis & strategy
- [ ] Analyze backtest results once populated — are signals profitable?
- [ ] Review signal quality: 474 signals generated, what's the hit rate?
- [ ] Evaluate trade candidates quality (currently 20)
- [ ] Consider tuning cluster_k (currently 8, silhouette ~0.2)
- [ ] Consider enabling `--detect-relationships` for cross-market correlation

### Long-term — infrastructure
- [ ] Monitor Cloud Run costs after enabling orderbook + volume (more API calls)
- [ ] Consider adding alerting on pipeline failures (currently just health check JSON)
- [ ] Add CI/CD pipeline for automated testing before deploy
- [ ] Consider Kelly sizing mode instead of flat for position sizing

---

## Known Issues
1. **Orderbook/volume nulls (20-35%)** — now being filled by hourly runs, but historical data won't have it
2. **`spread_trend` always null** — needs multiple orderbook snapshots over time (will accumulate)
3. **`volume_price_corr` ~50% null** — needs both volume AND sufficient price history
4. **`return_30d` ~61% null** — many assets younger than 30 days
5. **`gcloud` not in WSL PATH** — deploy must be run from PowerShell/CMD
