from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .normalize import primary_tag_from_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURE_COLUMNS = [
    "p_start",
    "p_end",
    "max_price",
    "min_price",
    "time_of_max",
    "return_total",
    "volatility",
    "max_drawdown",
    "slope",
    "missing_ratio",
    "num_points",
]


def _to_pandas_freq(interval: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    return mapping.get(interval.lower(), interval)


def _max_drawdown(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdown = running_max - series
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _slope(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def compute_asset_features(
    price_history_df: pd.DataFrame,
    *,
    interval: str,
    window_days: int,
    gap_fill_limit: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if price_history_df.empty:
        return pd.DataFrame(columns=["asset_id", *FEATURE_COLUMNS]), {}

    rows: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {}
    freq = _to_pandas_freq(interval)

    working = price_history_df.copy()
    working["asset_id"] = working["asset_id"].astype(str)
    working["ts"] = pd.to_numeric(working["ts"], errors="coerce")
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.dropna(subset=["asset_id", "ts", "price"])

    for asset_id, group in working.groupby("asset_id"):
        timestamps = pd.to_datetime(group["ts"].astype(np.int64), unit="s", utc=True)
        series = pd.Series(group["price"].to_numpy(dtype=float), index=timestamps).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if series.empty:
            continue

        end = series.index.max()
        start = end - pd.Timedelta(days=window_days)
        base = series[series.index >= start]
        if len(base) < 3:
            continue
        observed_start = base.index.min()

        # Normalize irregular timestamps to a fixed interval grid before filling.
        base = base.resample(freq).last().dropna()
        if base.empty:
            continue
        grid = pd.date_range(start=observed_start.floor(freq), end=end.ceil(freq), freq=freq, tz="UTC")
        aligned = base.reindex(grid)
        missing_ratio = float(aligned.isna().mean()) if len(aligned) else 1.0
        # Thin markets are common; only forward-fill short gaps and keep missingness as a feature.
        filled = aligned.ffill(limit=gap_fill_limit)
        clean = filled.dropna()
        if len(clean) < 3:
            continue

        diff = clean.diff().dropna()
        max_idx = int(np.argmax(clean.to_numpy(dtype=float)))
        time_of_max = max_idx / max(1, len(clean) - 1)

        rows.append(
            {
                "asset_id": asset_id,
                "p_start": float(clean.iloc[0]),
                "p_end": float(clean.iloc[-1]),
                "max_price": float(clean.max()),
                "min_price": float(clean.min()),
                "time_of_max": float(time_of_max),
                "return_total": float(clean.iloc[-1] - clean.iloc[0]),
                "volatility": float(diff.std(ddof=0)) if len(diff) else 0.0,
                "max_drawdown": _max_drawdown(clean),
                "slope": _slope(clean),
                "missing_ratio": missing_ratio,
                "num_points": int(len(clean)),
            }
        )
        curves[asset_id] = filled.to_numpy(dtype=float)

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        return pd.DataFrame(columns=["asset_id", *FEATURE_COLUMNS]), curves

    return features_df, curves


def cluster_features(
    feature_df: pd.DataFrame,
    *,
    cluster_k: int,
    random_seed: int,
) -> tuple[pd.DataFrame, float | None]:
    if feature_df.empty:
        return feature_df.assign(cluster_id=pd.Series(dtype=int)), None

    n_samples = len(feature_df)
    if n_samples < 3:
        clustered = feature_df.copy()
        clustered["cluster_id"] = np.arange(n_samples, dtype=int)
        return clustered, None

    k = min(max(2, int(cluster_k)), n_samples - 1)
    x = feature_df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = model.fit_predict(x_scaled)

    silhouette: float | None = None
    if 1 < k < n_samples:
        silhouette = float(silhouette_score(x_scaled, labels))

    clustered = feature_df.copy()
    clustered["cluster_id"] = labels.astype(int)
    return clustered, silhouette


def _stack_curves(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        return np.empty((0, 0))
    max_len = max(len(curve) for curve in curves)
    out = np.full((len(curves), max_len), np.nan, dtype=float)
    for idx, curve in enumerate(curves):
        out[idx, : len(curve)] = curve
    return out


def describe_curve(median_curve: np.ndarray) -> str:
    if median_curve.size == 0:
        return "no data"
    valid = median_curve[np.isfinite(median_curve)]
    if valid.size == 0:
        return "no data"

    spread = float(np.nanmax(valid) - np.nanmin(valid))
    slope = float(np.polyfit(np.arange(len(valid), dtype=float), valid, 1)[0]) if len(valid) > 1 else 0.0
    peak_idx = int(np.nanargmax(valid)) / max(1, len(valid) - 1)
    end_drop = float(np.nanmax(valid) - valid[-1])
    total_ret = float(valid[-1] - valid[0])

    if peak_idx < 0.35 and end_drop > 0.05:
        return "early pump then fade"
    if total_ret > 0.05 and slope > 0:
        return "monotonic drift up"
    if total_ret < -0.05 and slope < 0:
        return "monotonic drift down"
    if spread < 0.05:
        return "mostly flat"
    return "range-bound and choppy"


def save_cluster_plots(
    clustered_df: pd.DataFrame,
    curves: dict[str, np.ndarray],
    *,
    analysis_dir: Path,
) -> dict[int, str]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    descriptions: dict[int, str] = {}

    if clustered_df.empty:
        return descriptions

    for cluster_id in sorted(clustered_df["cluster_id"].dropna().astype(int).unique()):
        assets = clustered_df.loc[clustered_df["cluster_id"] == cluster_id, "asset_id"].astype(str).tolist()
        matrices = [curves[asset] for asset in assets if asset in curves]
        stacked = _stack_curves(matrices)
        if stacked.size == 0:
            descriptions[cluster_id] = "no data"
            continue

        valid_columns = np.isfinite(stacked).any(axis=0)
        if not bool(valid_columns.any()):
            descriptions[cluster_id] = "no data"
            continue
        stacked = stacked[:, valid_columns]

        median = np.nanmedian(stacked, axis=0)
        q1 = np.nanpercentile(stacked, 25, axis=0)
        q3 = np.nanpercentile(stacked, 75, axis=0)
        description = describe_curve(median)
        descriptions[cluster_id] = description

        x = np.arange(len(median), dtype=int)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, median, color="#0B4F6C", linewidth=2, label="Median")
        ax.fill_between(x, q1, q3, color="#88B04B", alpha=0.25, label="IQR")
        ax.set_title(f"Cluster {cluster_id}: {description}")
        ax.set_xlabel("Relative time index")
        ax.set_ylabel("Price")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / f"cluster_{cluster_id}.png", dpi=150)
        plt.close(fig)

    return descriptions


def save_bet_type_plot(summary_df: pd.DataFrame, *, analysis_dir: Path) -> None:
    if summary_df.empty:
        return

    top = summary_df.sort_values("tokens", ascending=False).head(15).copy()
    top["coverage_pct"] = top["coverage_pct"] * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top["primary_tag"], top["coverage_pct"], color="#2A9D8F")
    ax.set_ylabel("Token history coverage (%)")
    ax.set_xlabel("Bet type")
    ax.set_title("Coverage by Bet Type (top tags)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(analysis_dir / "coverage_by_bet_type.png", dpi=150)
    plt.close(fig)


def build_bet_type_summary(
    *,
    events_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if tokens_df.empty:
        return pd.DataFrame(
            columns=[
                "primary_tag",
                "markets",
                "tokens",
                "tokens_with_history",
                "coverage_pct",
                "avg_volatility",
                "avg_return_total",
                "avg_slope",
                "median_num_points",
                "median_missing_ratio",
                "tokens_quality_pass",
                "quality_pass_rate",
            ]
        )

    event_tags = events_df[["event_id", "tags"]].copy()
    event_tags["primary_tag"] = event_tags["tags"].apply(primary_tag_from_json)
    market_tags = markets_df[["market_id", "event_id"]].merge(
        event_tags[["event_id", "primary_tag"]],
        on="event_id",
        how="left",
    )
    market_tags["primary_tag"] = market_tags["primary_tag"].fillna("unknown")

    token_tags = tokens_df[["market_id", "asset_id"]].copy()
    token_tags["asset_id"] = token_tags["asset_id"].astype(str)
    token_tags = token_tags.merge(market_tags[["market_id", "primary_tag"]], on="market_id", how="left")
    token_tags["primary_tag"] = token_tags["primary_tag"].fillna("unknown")

    has_history = set(price_history_df["asset_id"].astype(str).unique()) if not price_history_df.empty else set()
    token_tags["has_history"] = token_tags["asset_id"].isin(has_history)
    token_tags = token_tags.merge(
        feature_df[["asset_id", "volatility", "return_total", "slope", "num_points", "missing_ratio"]].copy(),
        on="asset_id",
        how="left",
    )
    if quality_df is not None and not quality_df.empty:
        quality = quality_df[["asset_id", "quality_pass"]].copy()
        quality["asset_id"] = quality["asset_id"].astype(str)
        token_tags = token_tags.merge(quality, on="asset_id", how="left")
    else:
        token_tags["quality_pass"] = False
    token_tags["quality_pass"] = token_tags["quality_pass"].fillna(False).astype(bool)

    grouped = token_tags.groupby("primary_tag", dropna=False).agg(
        markets=("market_id", "nunique"),
        tokens=("asset_id", "nunique"),
        tokens_with_history=("has_history", "sum"),
        coverage_pct=("has_history", "mean"),
        avg_volatility=("volatility", "mean"),
        avg_return_total=("return_total", "mean"),
        avg_slope=("slope", "mean"),
        median_num_points=("num_points", "median"),
        median_missing_ratio=("missing_ratio", "median"),
        tokens_quality_pass=("quality_pass", "sum"),
        quality_pass_rate=("quality_pass", "mean"),
    )
    grouped = grouped.reset_index().sort_values(["tokens", "markets"], ascending=False)
    grouped["coverage_pct"] = grouped["coverage_pct"].fillna(0.0)
    grouped["avg_volatility"] = grouped["avg_volatility"].fillna(0.0)
    grouped["avg_return_total"] = grouped["avg_return_total"].fillna(0.0)
    grouped["avg_slope"] = grouped["avg_slope"].fillna(0.0)
    grouped["median_num_points"] = grouped["median_num_points"].fillna(0.0)
    grouped["median_missing_ratio"] = grouped["median_missing_ratio"].fillna(1.0)
    grouped["quality_pass_rate"] = grouped["quality_pass_rate"].fillna(0.0)
    return grouped


def build_market_quality_table(
    *,
    target_tokens_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    min_points: int,
    max_missing_ratio: float,
    min_price_range: float,
    min_liquidity: float,
) -> pd.DataFrame:
    columns = [
        "market_id",
        "asset_id",
        "liquidity",
        "history_points_raw",
        "has_history",
        "num_points",
        "missing_ratio",
        "price_range",
        "volatility",
        "return_total",
        "slope",
        "check_min_points",
        "check_missing_ratio",
        "check_price_range",
        "check_liquidity",
        "quality_pass",
    ]
    if target_tokens_df.empty:
        return pd.DataFrame(columns=columns)

    quality = target_tokens_df[["market_id", "asset_id"]].drop_duplicates().copy()
    quality["asset_id"] = quality["asset_id"].astype(str)

    history_counts = (
        price_history_df.groupby("asset_id", dropna=False)["ts"].count().rename("history_points_raw")
        if not price_history_df.empty
        else pd.Series(dtype=int, name="history_points_raw")
    )
    if not history_counts.empty:
        history_counts.index = history_counts.index.astype(str)
        quality = quality.merge(history_counts.to_frame(), left_on="asset_id", right_index=True, how="left")
    else:
        quality["history_points_raw"] = 0

    feature_work = feature_df[
        ["asset_id", "num_points", "missing_ratio", "max_price", "min_price", "volatility", "return_total", "slope"]
    ].copy()
    if not feature_work.empty:
        feature_work["asset_id"] = feature_work["asset_id"].astype(str)
    quality = quality.merge(feature_work, on="asset_id", how="left")

    quality = quality.merge(markets_df[["market_id", "liquidity"]], on="market_id", how="left")
    quality["liquidity"] = pd.to_numeric(quality["liquidity"], errors="coerce")
    quality["history_points_raw"] = pd.to_numeric(quality["history_points_raw"], errors="coerce").fillna(0).astype(int)
    quality["has_history"] = quality["history_points_raw"] > 0
    quality["price_range"] = (quality["max_price"] - quality["min_price"]).fillna(0.0)

    quality["check_min_points"] = quality["history_points_raw"].fillna(0) >= float(min_points)
    quality["check_missing_ratio"] = quality["missing_ratio"].fillna(1.0) <= float(max_missing_ratio)
    quality["check_price_range"] = quality["price_range"].fillna(0.0) >= float(min_price_range)

    if float(min_liquidity) > 0.0:
        quality["check_liquidity"] = quality["liquidity"].fillna(-np.inf) >= float(min_liquidity)
    else:
        quality["check_liquidity"] = True

    quality["quality_pass"] = (
        quality["has_history"]
        & quality["check_min_points"]
        & quality["check_missing_ratio"]
        & quality["check_price_range"]
        & quality["check_liquidity"]
    )

    for col in ("missing_ratio", "volatility", "return_total", "slope"):
        quality[col] = pd.to_numeric(quality[col], errors="coerce")
    quality["num_points"] = pd.to_numeric(quality["num_points"], errors="coerce")
    return quality[columns]


def build_tag_rankings(bet_type_summary_df: pd.DataFrame, *, top_n: int = 10) -> dict[str, list[dict[str, Any]]]:
    if bet_type_summary_df.empty:
        return {
            "highest_volatility_tags": [],
            "strongest_upward_trend_tags": [],
            "strongest_downward_trend_tags": [],
            "lowest_coverage_tags": [],
            "highest_quality_pass_rate_tags": [],
        }

    n = max(1, int(top_n))
    summary = bet_type_summary_df.copy()
    summary = summary[summary["tokens"] >= 5]
    if summary.empty:
        summary = bet_type_summary_df.copy()

    base_cols = ["primary_tag", "tokens", "coverage_pct", "quality_pass_rate", "avg_volatility", "avg_slope"]

    return {
        "highest_volatility_tags": summary.sort_values("avg_volatility", ascending=False)
        .head(n)[base_cols]
        .to_dict(orient="records"),
        "strongest_upward_trend_tags": summary.sort_values("avg_slope", ascending=False)
        .head(n)[base_cols]
        .to_dict(orient="records"),
        "strongest_downward_trend_tags": summary.sort_values("avg_slope", ascending=True)
        .head(n)[base_cols]
        .to_dict(orient="records"),
        "lowest_coverage_tags": summary.sort_values("coverage_pct", ascending=True)
        .head(n)[base_cols]
        .to_dict(orient="records"),
        "highest_quality_pass_rate_tags": summary.sort_values(
            ["quality_pass_rate", "tokens"], ascending=[False, False]
        )
        .head(n)[base_cols]
        .to_dict(orient="records"),
    }


def build_report_payload(
    *,
    events_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    all_tokens_df: pd.DataFrame,
    target_tokens_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
    silhouette: float | None,
    bet_type_summary_df: pd.DataFrame,
    cluster_descriptions: dict[int, str],
    cluster_input_assets: int,
    tag_rank_top_n: int = 10,
) -> dict[str, Any]:
    targeted_assets = target_tokens_df["asset_id"].astype(str).nunique() if not target_tokens_df.empty else 0
    targeted_set = set(target_tokens_df["asset_id"].astype(str).unique()) if not target_tokens_df.empty else set()
    extracted_assets = all_tokens_df["asset_id"].astype(str).nunique() if not all_tokens_df.empty else 0

    history_counts = (
        price_history_df.groupby("asset_id", dropna=False)["ts"].count()
        if not price_history_df.empty
        else pd.Series(dtype=int)
    )
    history_assets = set(history_counts.index.astype(str).tolist()) if not history_counts.empty else set()
    tokens_with_history = len(targeted_set.intersection(history_assets))
    median_points = 0.0
    if not history_counts.empty:
        target_counts = history_counts[history_counts.index.astype(str).isin(targeted_set)]
        if not target_counts.empty:
            median_points = float(target_counts.median())

    quality_total = len(quality_df) if quality_df is not None else 0
    quality_pass_count = 0
    quality_failures: dict[str, int] = {
        "failed_min_points": 0,
        "failed_missing_ratio": 0,
        "failed_price_range": 0,
        "failed_liquidity": 0,
    }
    if quality_df is not None and not quality_df.empty and "quality_pass" in quality_df.columns:
        quality_pass_count = int(quality_df["quality_pass"].fillna(False).astype(bool).sum())
        for metric, col in (
            ("failed_min_points", "check_min_points"),
            ("failed_missing_ratio", "check_missing_ratio"),
            ("failed_price_range", "check_price_range"),
            ("failed_liquidity", "check_liquidity"),
        ):
            if col in quality_df.columns:
                quality_failures[metric] = int((~quality_df[col].fillna(False).astype(bool)).sum())

    cluster_sizes: list[dict[str, Any]] = []
    tag_distribution: list[dict[str, Any]] = []
    if not clustered_df.empty:
        size_df = clustered_df.groupby("cluster_id", dropna=False).size().reset_index(name="size")
        cluster_sizes = size_df.sort_values("cluster_id").to_dict(orient="records")

        if "primary_tag" in clustered_df.columns:
            tag_df = (
                clustered_df.groupby(["cluster_id", "primary_tag"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(["cluster_id", "count"], ascending=[True, False])
            )
            tag_distribution = tag_df.to_dict(orient="records")

    by_tag_records = bet_type_summary_df.to_dict(orient="records") if not bet_type_summary_df.empty else []
    tag_rankings = build_tag_rankings(bet_type_summary_df, top_n=tag_rank_top_n)

    return {
        "coverage": {
            "events_discovered": int(len(events_df)),
            "markets_discovered": int(len(markets_df)),
            "tokens_extracted": int(extracted_assets),
            "tokens_targeted_for_history": int(targeted_assets),
            "tokens_with_non_empty_history": int(tokens_with_history),
            "pct_targeted_with_history": float(tokens_with_history / targeted_assets) if targeted_assets else 0.0,
            "median_points_per_token": median_points,
        },
        "quality": {
            "quality_total_assets": int(quality_total),
            "quality_pass_assets": int(quality_pass_count),
            "quality_pass_rate": float(quality_pass_count / quality_total) if quality_total else 0.0,
            "failure_counts": quality_failures,
        },
        "by_bet_type": by_tag_records,
        "tag_rankings": tag_rankings,
        "clusters": {
            "num_clusters": int(clustered_df["cluster_id"].nunique()) if not clustered_df.empty else 0,
            "cluster_input_assets": int(cluster_input_assets),
            "silhouette_score": silhouette,
            "cluster_sizes": cluster_sizes,
            "cluster_descriptions": [
                {"cluster_id": int(cluster_id), "description": description}
                for cluster_id, description in sorted(cluster_descriptions.items())
            ],
            "tag_distribution": tag_distribution,
        },
    }


def write_report_json(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
