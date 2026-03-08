#!/usr/bin/env python
"""
Polymarket Pipeline — Data Quality Audit
=========================================
Run: python scripts/dq_check.py [--dir data/prod]

Checks all core parquet files for schema integrity, nulls, duplicates,
referential integrity, price history validity, feature completeness,
clustering quality, and health check / pipeline run status.

Row-count deltas are tracked automatically in scripts/.dq_baseline.json
so every run shows what changed since the last audit.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = Path(__file__).resolve().parent / ".dq_baseline.json"

CORE_FILES = [
    "events.parquet", "markets.parquet", "tokens.parquet",
    "price_history.parquet", "features.parquet", "clusters.parquet",
    "market_quality.parquet", "pipeline_runs.parquet",
]
OPTIONAL_FILES = [
    "signals.parquet", "backtest_results.parquet", "trade_candidates.parquet",
    "resolutions.parquet", "orderbook_snapshots.parquet", "volume_bars.parquet",
    "market_relationships.parquet",
]

# Columns that are 100% null by design when optional pipeline flags are not used.
# These are NOT flagged as critical — they are informational only.
EXPECTED_NULL_COLS = {
    # needs --snapshot-orderbook
    "avg_spread", "avg_spread_pct", "spread_trend", "avg_depth",
    # needs --ingest-volume
    "avg_daily_volume", "volume_trend", "buy_sell_ratio", "volume_price_corr",
    # needs --include-resolved
    "resolution_outcome",
    # needs longer history (30 days)
    "return_30d",
    # partially expected with short history
    "return_1d", "return_7d", "zscore_7d", "rsi_14",
    # pipeline_runs: columns added after initial runs — historical rows have nulls
    "signals_expected", "pipeline_profile",
}

PK_MAP = {
    "events":        ["event_id"],
    "markets":       ["market_id"],
    "tokens":        ["asset_id"],
    "price_history": ["asset_id", "ts"],
    "features":      ["asset_id"],
    "clusters":      ["asset_id"],
}

FK_CHECKS = [
    ("markets",       "event_id",  "events",        "event_id"),
    ("tokens",        "market_id", "markets",        "market_id"),
    ("price_history", "asset_id",  "tokens",         "asset_id"),
    ("features",      "asset_id",  "tokens",         "asset_id"),
    ("features",      "asset_id",  "price_history",  "asset_id"),
    ("clusters",      "asset_id",  "features",       "asset_id"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def load_baseline():
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline(counts: dict):
    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": counts,
    }
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def ts_to_utc(series: pd.Series) -> pd.Series:
    if series.dtype in [np.int64, np.float64]:
        return pd.to_datetime(series, unit="s", utc=True)
    return pd.to_datetime(series, utc=True)


# ── Main audit ────────────────────────────────────────────────────────────────

def run_audit(prod_dir: Path):
    dfs: dict[str, pd.DataFrame] = {}
    issues_critical: list[str] = []
    issues_warning: list[str] = []
    passes: list[str] = []
    baseline = load_baseline()
    prev_counts: dict = baseline.get("row_counts", {})

    # ── STEP 1: File Inventory & Schema ───────────────────────────────────────
    sep("STEP 1: FILE INVENTORY & SCHEMA CHECK")

    for fname in CORE_FILES:
        fpath = prod_dir / fname
        if not fpath.exists():
            issues_critical.append(f"Missing core file: {fname}")
            print(f"  CRITICAL: {fname} NOT FOUND")
            continue
        df = pd.read_parquet(fpath)
        key = fname.replace(".parquet", "")
        dfs[key] = df
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"\n  {fname}: {len(df):,} rows × {len(df.columns)} cols  ({mem_mb:.1f} MB)  [mod {mtime}]")
        for col in df.columns:
            null_n = df[col].isna().sum()
            null_pct = null_n / len(df) * 100 if len(df) else 0
            flag = ""
            if null_pct > 50:
                if col in EXPECTED_NULL_COLS:
                    flag = "  ← expected (no flag/volume/resolved data)"
                else:
                    flag = "  [CRITICAL: >50% null — UNEXPECTED]"
                    issues_critical.append(f"{fname}:{col} has {null_pct:.0f}% nulls (unexpected)")
            elif null_pct > 10:
                flag = "  [WARN: >10% null]"
                issues_warning.append(f"{fname}:{col} has {null_pct:.0f}% nulls")
            print(f"    {col}: {str(df[col].dtype):<12}  {null_n:>7,} nulls  ({null_pct:5.1f}%){flag}")

    print("\n  Optional files:")
    for fname in OPTIONAL_FILES:
        fpath = prod_dir / fname
        if fpath.exists():
            n = len(pd.read_parquet(fpath))
            print(f"    FOUND   {fname}: {n:,} rows")
        else:
            print(f"    absent  {fname}")

    # ── STEP 2: Duplicates ────────────────────────────────────────────────────
    sep("STEP 2: DUPLICATES CHECK")
    for name, pks in PK_MAP.items():
        if name not in dfs:
            continue
        df = dfs[name]
        missing = [pk for pk in pks if pk not in df.columns]
        if missing:
            issues_warning.append(f"{name}: PK columns missing: {missing}")
            print(f"  WARN  {name}: PK columns missing: {missing}")
            continue
        dupes = len(df) - len(df.drop_duplicates(subset=pks))
        if dupes:
            issues_warning.append(f"{name}: {dupes:,} duplicates on {pks}")
            print(f"  WARN  {name}: {dupes:,} duplicates on {pks}")
        else:
            passes.append(f"{name}: No duplicates on {pks}")
            print(f"  PASS  {name}: no duplicates on {pks}")

    # ── STEP 3: Referential Integrity ─────────────────────────────────────────
    sep("STEP 3: REFERENTIAL INTEGRITY")
    for child, ccol, parent, pcol in FK_CHECKS:
        if child not in dfs or parent not in dfs:
            print(f"  SKIP  {child}.{ccol} → {parent}.{pcol}  (df missing)")
            continue
        if ccol not in dfs[child].columns or pcol not in dfs[parent].columns:
            issues_warning.append(f"FK column missing: {child}.{ccol}")
            print(f"  WARN  {child}.{ccol}: column missing")
            continue
        orphans = set(dfs[child][ccol].dropna()) - set(dfs[parent][pcol].dropna())
        if orphans:
            pct = len(orphans) / dfs[child][ccol].nunique() * 100
            issues_warning.append(f"{child}.{ccol}→{parent}.{pcol}: {len(orphans):,} orphans ({pct:.1f}%)")
            print(f"  WARN  {child}.{ccol} → {parent}.{pcol}: {len(orphans):,} orphans ({pct:.1f}%)")
        else:
            passes.append(f"{child}.{ccol} → {parent}.{pcol}: no orphans")
            print(f"  PASS  {child}.{ccol} → {parent}.{pcol}: no orphans")

    # Coverage funnel
    n_ev = len(dfs.get("events", pd.DataFrame()))
    n_mk = len(dfs.get("markets", pd.DataFrame()))
    n_tk = len(dfs.get("tokens", pd.DataFrame()))
    n_ph = dfs["price_history"]["asset_id"].nunique() if "price_history" in dfs else 0
    n_ft = len(dfs.get("features", pd.DataFrame()))
    n_cl = len(dfs.get("clusters", pd.DataFrame()))
    cov  = f"({n_ph / n_tk * 100:.1f}% of tokens)" if n_tk else ""
    print(f"\n  Coverage Funnel:")
    print(f"    Events              {n_ev:>7,}")
    print(f"    Markets             {n_mk:>7,}")
    print(f"    Tokens              {n_tk:>7,}")
    print(f"    Tokens w/ price     {n_ph:>7,}  {cov}")
    print(f"    Features computed   {n_ft:>7,}")
    print(f"    Clustered           {n_cl:>7,}")

    # ── STEP 4: Price History ─────────────────────────────────────────────────
    sep("STEP 4: PRICE HISTORY DEEP DIVE")
    if "price_history" in dfs:
        ph = dfs["price_history"]
        ts_col = ts_to_utc(ph["ts"])
        span_h = (ts_col.max() - ts_col.min()).total_seconds() / 3600
        span_d = span_h / 24
        print(f"  Time span:  {ts_col.min()}  →  {ts_col.max()}")
        print(f"  Total span: {span_h:.1f} h  ({span_d:.1f} days)")
        if span_d < 1:
            issues_critical.append(f"price_history spans only {span_h:.1f}h — pipeline failed to fetch full history")
            print("  CRITICAL: span < 1 day!")
        elif span_d < 7:
            issues_warning.append(f"price_history spans only {span_d:.1f} days (expected ~90)")
            print(f"  WARN: span < 7 days")
        else:
            passes.append(f"price_history spans {span_d:.1f} days")
            print(f"  PASS: good time span ({span_d:.1f} days)")

        pts = ph.groupby("asset_id").size()
        print(f"\n  Points per asset:")
        print(f"    min={pts.min()}  max={pts.max()}  mean={pts.mean():.1f}  median={pts.median():.1f}")
        print(f"    p5={pts.quantile(.05):.0f}  p25={pts.quantile(.25):.0f}  p75={pts.quantile(.75):.0f}  p95={pts.quantile(.95):.0f}")
        if pts.nunique() == 1:
            issues_warning.append(f"All assets have exactly {pts.iloc[0]} points — suspicious uniformity")
            print(f"  WARN: all assets have identical point count ({pts.iloc[0]}) — suspicious!")
        if pts.max() <= 10:
            issues_critical.append(f"Max points/asset={pts.max()} — insufficient for analysis")
            print(f"  CRITICAL: very few points (max={pts.max()})")
        else:
            passes.append(f"Sufficient points/asset (max={pts.max()}, median={pts.median():.0f})")

        # Cadence
        print(f"\n  Cadence (50-asset sample):")
        gaps = []
        for aid in ph["asset_id"].unique()[:50]:
            sub = ph[ph["asset_id"] == aid].sort_values("ts")
            if len(sub) > 1:
                if ph["ts"].dtype in [np.int64, np.float64]:
                    gaps.extend(sub["ts"].diff().dropna().tolist())
                else:
                    gaps.extend(sub["ts"].diff().dropna().dt.total_seconds().tolist())
        if gaps:
            g = np.array(gaps)
            med = float(np.median(g))
            print(f"    median gap: {med:.0f}s ({med / 3600:.2f}h)  |  min={g.min():.0f}s  max={g.max():.0f}s  p95={np.percentile(g, 95):.0f}s")
            neg = int((g <= 0).sum())
            if neg:
                issues_critical.append(f"{neg:,} zero/negative time gaps in price_history")
                print(f"  CRITICAL: {neg:,} non-positive time gaps!")
            else:
                passes.append("No negative time gaps in price_history")
                print("  PASS: no negative time gaps")

        # Price sanity
        lo = int((ph["price"] < 0).sum())
        hi = int((ph["price"] > 1).sum())
        p0 = int((ph["price"] == 0).sum())
        p1 = int((ph["price"] == 1).sum())
        print(f"\n  Price sanity:")
        print(f"    out-of-range: {lo} below 0,  {hi} above 1")
        print(f"    price==0: {p0:,}   price==1: {p1:,}")
        print(f"    min={ph['price'].min():.4f}  max={ph['price'].max():.4f}  mean={ph['price'].mean():.4f}  median={ph['price'].median():.4f}")
        if lo or hi:
            issues_critical.append(f"Prices outside [0,1]: {lo} below, {hi} above")
            print("  CRITICAL: invalid prices!")
        else:
            passes.append("All prices in valid [0,1] range")
            print("  PASS: all prices valid")

    # ── STEP 5: Features ──────────────────────────────────────────────────────
    sep("STEP 5: FEATURES INTEGRITY")
    if "features" in dfs:
        feat = dfs["features"]
        non_id = [c for c in feat.columns if c != "asset_id"]
        all_null  = [c for c in non_id if feat[c].isna().all()]
        has_data  = [c for c in non_id if not feat[c].isna().all()]

        print(f"  ALL-null cols  ({len(all_null)}): {all_null}")
        print(f"  Has-data cols  ({len(has_data)}): {has_data}")

        unexpected = [c for c in all_null if c not in EXPECTED_NULL_COLS]
        if unexpected:
            issues_warning.append(f"Unexpected 100%-null feature cols: {unexpected}")
            print(f"  WARN: unexpected all-null cols: {unexpected}")
        else:
            passes.append("All 100%-null feature columns are expected")
            print("  PASS: all-null cols are expected (no orderbook/volume data)")

        for col in has_data:
            s = feat[col].dropna()
            nu = s.nunique()
            if nu <= 1:
                issues_warning.append(f"features.{col} is CONSTANT (={s.iloc[0] if len(s) else 'N/A'})")
                print(f"  WARN: {col} CONSTANT  (value={s.iloc[0] if len(s) else 'N/A'})")
            elif nu <= 3:
                issues_warning.append(f"features.{col} LOW VARIANCE ({nu} unique values)")
                print(f"  WARN: {col} LOW VARIANCE  ({nu} unique values)")
            else:
                print(f"  PASS: {col}  |  {nu} unique  |  [{s.min():.4f}, {s.max():.4f}]")

        if "num_points" in feat.columns:
            np_s = feat["num_points"].dropna()
            print(f"\n  num_points:     min={np_s.min()}  max={np_s.max()}  mean={np_s.mean():.1f}  median={np_s.median():.1f}")
            if np_s.nunique() == 1 and np_s.iloc[0] <= 5:
                issues_critical.append(f"num_points is constant={np_s.iloc[0]} — features from minimal data")
        if "missing_ratio" in feat.columns:
            mr = feat["missing_ratio"].dropna()
            print(f"  missing_ratio:  min={mr.min():.4f}  max={mr.max():.4f}  mean={mr.mean():.4f}")
            if mr.nunique() == 1:
                issues_warning.append(f"missing_ratio is constant={mr.iloc[0]:.4f}")

    # ── STEP 6: Clusters & Market Quality ────────────────────────────────────
    sep("STEP 6: CLUSTERS & MARKET QUALITY")
    if "clusters" in dfs and "cluster_id" in dfs["clusters"].columns:
        cl = dfs["clusters"]
        dist = cl["cluster_id"].value_counts().sort_index()
        print(f"  Clusters: {dist.shape[0]}  |  Assets: {len(cl):,}")
        print(f"  Sizes: min={dist.min()}  max={dist.max()}  mean={dist.mean():.1f}")
        tiny = int((dist < 3).sum())
        if tiny:
            issues_warning.append(f"{tiny} clusters with < 3 members")
            print(f"  WARN: {tiny} clusters have < 3 members")
        else:
            passes.append("All clusters have ≥ 3 members")
            print("  PASS: all clusters ≥ 3 members")
        print(f"  Distribution: {dist.to_dict()}")

    if "market_quality" in dfs:
        mq = dfs["market_quality"]
        print(f"\n  Market quality  ({len(mq):,} rows):")
        if "quality_pass" in mq.columns:
            pr = mq["quality_pass"].mean()
            label = "✓" if pr >= 0.4 else "WARN"
            print(f"    quality_pass:         {pr:.1%}  ({int(mq['quality_pass'].sum()):,}/{len(mq):,})  [{label}]")
            if pr < 0.2:
                issues_warning.append(f"Low quality_pass rate: {pr:.1%}")
        for chk in ["check_min_points", "check_missing_ratio", "check_price_range", "check_liquidity"]:
            if chk in mq.columns:
                r = mq[chk].mean()
                print(f"    {chk:<28}  {r:.1%} pass")

    # ── STEP 7: Delta vs last audit ───────────────────────────────────────────
    sep("STEP 7: DELTA vs LAST AUDIT")
    current_counts = {fname.replace(".parquet", ""): len(dfs[fname.replace(".parquet", "")])
                      for fname in CORE_FILES if fname.replace(".parquet", "") in dfs}

    if not prev_counts:
        print("  No baseline found — this run establishes the baseline.")
    else:
        prev_ts = baseline.get("updated_at", "unknown")
        print(f"  Baseline from: {prev_ts}")
        print(f"  {'File':<30} {'Previous':>12} {'Now':>12} {'Delta':>10}  Note")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}  {'-'*15}")
        for key, now in current_counts.items():
            prev = prev_counts.get(key, 0)
            delta = now - prev
            note = ""
            if delta < -10:
                note = "⚠ SHRUNK"
                issues_warning.append(f"{key}.parquet shrank by {abs(delta):,} rows vs baseline")
            elif delta > 0:
                note = "grew"
            print(f"  {key+'.parquet':<30} {prev:>12,} {now:>12,} {delta:>+10,}  {note}")

    # Save new baseline
    save_baseline(current_counts)
    print(f"\n  Baseline updated → {BASELINE_FILE.name}")

    # ── STEP 8: Health Check & Pipeline Runs ──────────────────────────────────
    sep("STEP 8: HEALTH CHECK & PIPELINE RUNS")
    for jname in ["health_check.json", "report.json"]:
        jpath = prod_dir / "analysis" / jname
        if jpath.exists():
            with open(jpath) as f:
                data = json.load(f)
            print(f"\n  {jname}:")
            print(json.dumps(data, indent=2)[:4000])
        else:
            print(f"  {jname}: NOT FOUND at {jpath}")

    if "pipeline_runs" in dfs:
        pr = dfs["pipeline_runs"].copy()
        pr["run_dt"] = pd.to_datetime(pr["run_ts"], unit="s").dt.strftime("%m-%d %H:%M")
        show_cols = [c for c in ["run_dt", "pipeline_profile", "duration_s", "events_count", "markets_count",
                                  "price_coverage_pct", "signals_expected", "signals_generated", "status"] if c in pr.columns]
        pd.set_option("display.width", 150)
        pd.set_option("display.max_columns", None)
        print(f"\n  pipeline_runs: {len(pr)} total entries (last 10):")
        print(pr[show_cols].tail(10).to_string(index=False))

        # Trend check: coverage stuck?
        if "price_coverage_pct" in pr.columns and len(pr) >= 5:
            recent = pr["price_coverage_pct"].tail(5)
            if recent.std() < 0.001:
                issues_warning.append(f"price_coverage_pct stuck at {recent.mean():.4f} for last {len(recent)} runs")
                print(f"\n  WARN: price_coverage_pct has not changed in last {len(recent)} runs ({recent.mean():.4f})")

        if "signals_generated" in pr.columns:
            signal_runs = pr.copy()
            if "signals_expected" in signal_runs.columns:
                expected_mask = signal_runs["signals_expected"].fillna(False).infer_objects(copy=False).astype(bool)
                signal_runs = signal_runs.loc[expected_mask].copy()
            if not signal_runs.empty and (signal_runs["signals_generated"].tail(10) == 0).all():
                issues_warning.append("signals_generated=0 for last 10 signal-enabled runs — signal engine producing nothing")
                print("  WARN: signals_generated=0 across last 10 signal-enabled runs")

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    sep("FINAL REPORT")
    print(f"\n  Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Data dir: {prod_dir}\n")

    print(f"  PASS  ({len(passes)}):")
    for p in passes:
        print(f"    +  {p}")

    print(f"\n  CRITICAL  ({len(issues_critical)}):")
    for c in issues_critical:
        print(f"    !!  {c}")

    print(f"\n  WARNING  ({len(issues_warning)}):")
    for w in issues_warning:
        print(f"    !   {w}")

    print("\n" + "=" * 70)
    if not issues_critical:
        print("  VERDICT:  ✅  READY FOR ANALYSIS")
        if issues_warning:
            print(f"  ({len(issues_warning)} warnings — not blocking)")
    else:
        print("  VERDICT:  ❌  NOT READY — resolve critical issues first")
        for c in issues_critical:
            print(f"    →  {c}")
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket pipeline data quality audit")
    parser.add_argument(
        "--dir",
        default=None,
        help="Path to parquet output directory. Defaults to data/prod if it exists, else data/.",
    )
    args = parser.parse_args()

    if args.dir:
        prod_dir = Path(args.dir).resolve()
    else:
        candidate = PROJECT_ROOT / "data" / "prod"
        prod_dir = candidate if candidate.exists() else PROJECT_ROOT / "data"

    if not prod_dir.exists():
        print(f"ERROR: directory not found: {prod_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Polymarket Data Quality Audit  —  {prod_dir}")
    print(f"{'=' * 70}")
    run_audit(prod_dir)


if __name__ == "__main__":
    main()
