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
    "days_to_resolution",
    "pct_lifetime_elapsed",
    "days_since_creation",
    "avg_spread",
    "avg_spread_pct",
    "spread_trend",
    "avg_depth",
    "avg_bid_depth",
    "avg_ask_depth",
    "avg_daily_volume",
    "volume_trend",
    "buy_sell_ratio",
    "volume_price_corr",
    "return_1d",
    "return_7d",
    "return_30d",
    "zscore_7d",
    "rsi_14",
]

FEATURE_METADATA: list[dict[str, Any]] = [
    {"name": "p_start", "type": "price", "window": "lookback", "description": "Starting price of cleaned window"},
    {"name": "p_end", "type": "price", "window": "lookback", "description": "Ending price of cleaned window"},
    {"name": "max_price", "type": "price", "window": "lookback", "description": "Maximum price in window"},
    {"name": "min_price", "type": "price", "window": "lookback", "description": "Minimum price in window"},
    {"name": "time_of_max", "type": "price", "window": "lookback", "description": "Relative index of max price"},
    {"name": "return_total", "type": "price", "window": "lookback", "description": "Net price change p_end-p_start"},
    {"name": "volatility", "type": "price", "window": "lookback", "description": "Std of first differences"},
    {"name": "max_drawdown", "type": "price", "window": "lookback", "description": "Peak-to-trough drawdown"},
    {"name": "slope", "type": "price", "window": "lookback", "description": "Linear trend slope"},
    {"name": "missing_ratio", "type": "price", "window": "lookback", "description": "Ratio of missing points on grid"},
    {"name": "num_points", "type": "price", "window": "lookback", "description": "Non-missing points after fill"},
    {"name": "days_to_resolution", "type": "time", "window": "point-in-time", "description": "Days until event end"},
    {
        "name": "pct_lifetime_elapsed",
        "type": "time",
        "window": "point-in-time",
        "description": "Elapsed fraction between market start and end",
    },
    {"name": "days_since_creation", "type": "time", "window": "point-in-time", "description": "Days since market start"},
    {"name": "avg_spread", "type": "microstructure", "window": "recent snapshots", "description": "Mean top-of-book spread"},
    {"name": "avg_spread_pct", "type": "microstructure", "window": "recent snapshots", "description": "Mean spread / mid"},
    {"name": "spread_trend", "type": "microstructure", "window": "recent snapshots", "description": "Slope of spread"},
    {"name": "avg_depth", "type": "microstructure", "window": "recent snapshots", "description": "Average bid/ask depth at 5%"},
    {"name": "avg_bid_depth", "type": "microstructure", "window": "recent snapshots", "description": "Average bid depth at 5%"},
    {"name": "avg_ask_depth", "type": "microstructure", "window": "recent snapshots", "description": "Average ask depth at 5%"},
    {"name": "avg_daily_volume", "type": "volume", "window": "lookback", "description": "Mean daily traded volume"},
    {"name": "volume_trend", "type": "volume", "window": "lookback", "description": "Trend in daily volume"},
    {"name": "buy_sell_ratio", "type": "volume", "window": "lookback", "description": "Total buy volume / sell volume"},
    {
        "name": "volume_price_corr",
        "type": "volume",
        "window": "lookback",
        "description": "Correlation of daily volume with absolute daily price move",
    },
    {"name": "return_1d", "type": "momentum", "window": "1d", "description": "Return over last day"},
    {"name": "return_7d", "type": "momentum", "window": "7d", "description": "Return over last seven days"},
    {"name": "return_30d", "type": "momentum", "window": "30d", "description": "Return over last thirty days"},
    {"name": "zscore_7d", "type": "momentum", "window": "7d", "description": "Z-score of latest price vs 7d mean/std"},
    {"name": "rsi_14", "type": "momentum", "window": "14 periods", "description": "Wilder RSI on cleaned series"},
]


def _to_pandas_freq(interval: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    return mapping.get(interval.lower(), interval)


def _interval_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14_400,
        "6h": 21_600,
        "1d": 86_400,
        "1w": 604_800,
    }
    return int(mapping.get(str(interval).lower(), 3600))


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


def _return_over(clean: pd.Series, periods: int) -> float:
    if periods <= 0 or len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-1 - periods])


def _rsi_wilder(clean: pd.Series, periods: int = 14) -> float:
    values = pd.to_numeric(clean, errors="coerce").dropna()
    if len(values) <= periods:
        return np.nan
    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
    avg_loss = losses.ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if rsi.empty:
        return np.nan
    return float(rsi.iloc[-1])


def write_feature_metadata_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"features": FEATURE_METADATA}, handle, indent=2, ensure_ascii=True)


def compute_asset_features(
    price_history_df: pd.DataFrame,
    *,
    interval: str,
    window_days: int,
    gap_fill_limit: int,
    market_context_df: pd.DataFrame | None = None,
    orderbook_df: pd.DataFrame | None = None,
    volume_bars_df: pd.DataFrame | None = None,
    as_of_ts: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if price_history_df.empty:
        return pd.DataFrame(columns=["asset_id", *FEATURE_COLUMNS]), {}

    rows: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {}
    freq = _to_pandas_freq(interval)
    interval_s = _interval_to_seconds(interval)
    points_per_day = max(1, int(round(86_400 / interval_s)))
    now_ts = int(as_of_ts) if as_of_ts is not None else int(pd.Timestamp.utcnow().timestamp())

    working = price_history_df.copy()
    working["asset_id"] = working["asset_id"].astype(str)
    working["ts"] = pd.to_numeric(working["ts"], errors="coerce")
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.dropna(subset=["asset_id", "ts", "price"])

    market_context_map: dict[str, dict[str, Any]] = {}
    if market_context_df is not None and not market_context_df.empty and "asset_id" in market_context_df.columns:
        context = market_context_df.copy()
        context["asset_id"] = context["asset_id"].astype(str)
        for key in ("start_ts", "end_ts"):
            if key in context.columns:
                context[key] = pd.to_numeric(context[key], errors="coerce")
        market_context_map = context.drop_duplicates("asset_id", keep="last").set_index("asset_id").to_dict(orient="index")

    orderbook_map: dict[str, pd.DataFrame] = {}
    if orderbook_df is not None and not orderbook_df.empty and "asset_id" in orderbook_df.columns:
        ob = orderbook_df.copy()
        ob["asset_id"] = ob["asset_id"].astype(str)
        for key in ("spread", "spread_pct", "bid_depth_5pct", "ask_depth_5pct", "snapshot_ts"):
            if key in ob.columns:
                ob[key] = pd.to_numeric(ob[key], errors="coerce")
        for asset_id, group in ob.groupby("asset_id", dropna=False):
            orderbook_map[str(asset_id)] = group.sort_values("snapshot_ts")

    volume_map: dict[str, pd.DataFrame] = {}
    if volume_bars_df is not None and not volume_bars_df.empty and "asset_id" in volume_bars_df.columns:
        vb = volume_bars_df.copy()
        vb["asset_id"] = vb["asset_id"].astype(str)
        for key in ("ts", "volume", "buy_volume", "sell_volume", "trade_count"):
            if key in vb.columns:
                vb[key] = pd.to_numeric(vb[key], errors="coerce")
        vb = vb.dropna(subset=["asset_id", "ts"])
        for asset_id, group in vb.groupby("asset_id", dropna=False):
            volume_map[str(asset_id)] = group.sort_values("ts")

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

        context = market_context_map.get(str(asset_id), {})
        start_ts = context.get("start_ts")
        end_ts = context.get("end_ts")
        try:
            start_ts_value = float(start_ts) if start_ts is not None and np.isfinite(start_ts) else np.nan
        except Exception:
            start_ts_value = np.nan
        try:
            end_ts_value = float(end_ts) if end_ts is not None and np.isfinite(end_ts) else np.nan
        except Exception:
            end_ts_value = np.nan
        days_to_resolution = (end_ts_value - now_ts) / 86_400.0 if np.isfinite(end_ts_value) else np.nan
        days_since_creation = (now_ts - start_ts_value) / 86_400.0 if np.isfinite(start_ts_value) else np.nan
        pct_lifetime_elapsed = np.nan
        if np.isfinite(start_ts_value) and np.isfinite(end_ts_value) and end_ts_value > start_ts_value:
            pct_lifetime_elapsed = (now_ts - start_ts_value) / (end_ts_value - start_ts_value)

        # Microstructure features
        ob_group = orderbook_map.get(str(asset_id))
        avg_spread = np.nan
        avg_spread_pct = np.nan
        spread_trend = np.nan
        avg_depth = np.nan
        avg_bid_depth = np.nan
        avg_ask_depth = np.nan
        if ob_group is not None and not ob_group.empty:
            spreads = pd.to_numeric(ob_group.get("spread"), errors="coerce").dropna()
            spread_pct = pd.to_numeric(ob_group.get("spread_pct"), errors="coerce").dropna()
            if not spreads.empty:
                avg_spread = float(spreads.mean())
                if len(spreads) >= 2:
                    spread_trend = _slope(spreads.reset_index(drop=True))
            if not spread_pct.empty:
                avg_spread_pct = float(spread_pct.mean())
            if "bid_depth_5pct" in ob_group.columns and "ask_depth_5pct" in ob_group.columns:
                bid_depth_s = pd.to_numeric(ob_group["bid_depth_5pct"], errors="coerce").dropna()
                ask_depth_s = pd.to_numeric(ob_group["ask_depth_5pct"], errors="coerce").dropna()
                if not bid_depth_s.empty:
                    avg_bid_depth = float(bid_depth_s.mean())
                if not ask_depth_s.empty:
                    avg_ask_depth = float(ask_depth_s.mean())
                depth_series = ((bid_depth_s.reindex(ob_group.index).fillna(0) + ask_depth_s.reindex(ob_group.index).fillna(0)) / 2.0).dropna()
                if not depth_series.empty:
                    avg_depth = float(depth_series.mean())

        # Volume features
        avg_daily_volume = np.nan
        volume_trend = np.nan
        buy_sell_ratio = np.nan
        volume_price_corr = np.nan
        volume_group = volume_map.get(str(asset_id))
        if volume_group is not None and not volume_group.empty:
            day_frame = volume_group.copy()
            day_frame["dt"] = pd.to_datetime(day_frame["ts"].astype("int64"), unit="s", utc=True).dt.floor("1d")
            daily = day_frame.groupby("dt", dropna=False).agg(
                volume=("volume", "sum"),
                buy_volume=("buy_volume", "sum"),
                sell_volume=("sell_volume", "sum"),
            )
            if not daily.empty:
                avg_daily_volume = float(pd.to_numeric(daily["volume"], errors="coerce").dropna().mean())
                vol_series = pd.to_numeric(daily["volume"], errors="coerce").dropna()
                if len(vol_series) >= 2:
                    volume_trend = _slope(vol_series.reset_index(drop=True))
                buy_total = float(pd.to_numeric(daily["buy_volume"], errors="coerce").fillna(0.0).sum())
                sell_total = float(pd.to_numeric(daily["sell_volume"], errors="coerce").fillna(0.0).sum())
                if sell_total > 0:
                    buy_sell_ratio = buy_total / sell_total

                clean_daily = clean.resample("1d").last().dropna()
                move_daily = clean_daily.diff().abs().dropna()
                if not move_daily.empty:
                    aligned = pd.concat([vol_series.rename("volume"), move_daily.rename("abs_move")], axis=1).dropna()
                    if len(aligned) >= 3:
                        volume_price_corr = float(aligned["volume"].corr(aligned["abs_move"]))

        return_1d = _return_over(clean, points_per_day * 1)
        return_7d = _return_over(clean, points_per_day * 7)
        return_30d = _return_over(clean, points_per_day * 30)
        lookback_7d = points_per_day * 7
        zscore_7d = np.nan
        if len(clean) > lookback_7d:
            sample = clean.iloc[-lookback_7d:]
            mean_ = float(sample.mean())
            std_ = float(sample.std(ddof=0))
            if std_ > 0:
                zscore_7d = float((clean.iloc[-1] - mean_) / std_)
        rsi_14 = _rsi_wilder(clean, periods=14)

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
                "days_to_resolution": days_to_resolution,
                "pct_lifetime_elapsed": pct_lifetime_elapsed,
                "days_since_creation": days_since_creation,
                "avg_spread": avg_spread,
                "avg_spread_pct": avg_spread_pct,
                "spread_trend": spread_trend,
                "avg_depth": avg_depth,
                "avg_bid_depth": avg_bid_depth,
                "avg_ask_depth": avg_ask_depth,
                "avg_daily_volume": avg_daily_volume,
                "volume_trend": volume_trend,
                "buy_sell_ratio": buy_sell_ratio,
                "volume_price_corr": volume_price_corr,
                "return_1d": return_1d,
                "return_7d": return_7d,
                "return_30d": return_30d,
                "zscore_7d": zscore_7d,
                "rsi_14": rsi_14,
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
