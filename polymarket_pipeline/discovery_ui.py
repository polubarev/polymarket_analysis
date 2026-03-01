from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from polymarket_pipeline.normalize import primary_tag_from_json

try:
    import altair as alt
except Exception:  # pragma: no cover - optional UI dependency path
    alt = None


REQUIRED_FILES = (
    "events.parquet",
    "markets.parquet",
    "tokens.parquet",
    "price_history.parquet",
)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-points-per-line", type=int, default=1200)
    parser.add_argument("--ui-mode", choices=["discovery", "full"], default="discovery")
    args, _ = parser.parse_known_args()
    return args


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
              radial-gradient(circle at 12% 10%, rgba(48, 110, 147, 0.30) 0%, rgba(48, 110, 147, 0) 34%),
              radial-gradient(circle at 90% 0%, rgba(50, 140, 120, 0.26) 0%, rgba(50, 140, 120, 0) 38%),
              linear-gradient(180deg, #0b1118 0%, #0f1723 100%);
            color: #e5edf8;
        }
        html, body, [class*="css"] {
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1b2c 0%, #0e1824 100%);
            border-right: 1px solid rgba(126, 162, 202, 0.25);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #173958 0%, #226f83 58%, #2f9689 100%);
            color: #f6fbff;
            border: 1px solid rgba(149, 213, 239, 0.35);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
        }
        .hero h1 {
            margin: 0;
            font-size: 1.45rem;
            line-height: 1.2;
        }
        .hero p {
            margin: 0.35rem 0 0;
            font-size: 0.98rem;
            opacity: 0.92;
        }
        .section-note {
            border-left: 4px solid #49b2d6;
            background: rgba(34, 56, 84, 0.45);
            border-radius: 0 8px 8px 0;
            padding-left: 0.7rem;
            color: #d4e9ff;
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
        }
        [data-testid="stMetric"] {
            background: rgba(23, 35, 52, 0.72);
            border: 1px solid rgba(109, 141, 178, 0.33);
            border-radius: 12px;
            padding: 0.45rem 0.6rem;
        }
        button[data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(53, 93, 136, 0.35);
            color: #e5f1ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_altair_chart(chart: Any) -> Any:
    if alt is None:
        return chart
    return (
        chart.configure(background="#101927")
        .configure_view(fill="#101927", strokeOpacity=0)
        .configure_axis(
            labelColor="#d7e6f7",
            titleColor="#d7e6f7",
            gridColor="#30445d",
            domainColor="#4b6587",
            tickColor="#4b6587",
        )
        .configure_legend(
            labelColor="#d7e6f7",
            titleColor="#d7e6f7",
        )
        .configure_title(
            color="#e9f2ff",
            anchor="start",
        )
    )


def _unix_to_datetime(series: pd.Series) -> pd.Series:
    ts = pd.to_numeric(series, errors="coerce")
    ts = ts.where(ts <= 10_000_000_000, ts // 1000)
    return pd.to_datetime(ts, unit="s", utc=True, errors="coerce").dt.tz_convert(None)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _validate_data_dir(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing required files in {data_dir}: {missing_text}")


@st.cache_data(show_spinner=False)
def load_data_bundle(data_dir_text: str) -> dict[str, Any]:
    data_dir = Path(data_dir_text)
    _validate_data_dir(data_dir)

    events = _safe_read_parquet(data_dir / "events.parquet")
    markets = _safe_read_parquet(data_dir / "markets.parquet")
    tokens = _safe_read_parquet(data_dir / "tokens.parquet")
    price_history = _safe_read_parquet(data_dir / "price_history.parquet")
    market_quality = _safe_read_parquet(data_dir / "market_quality.parquet")
    clusters = _safe_read_parquet(data_dir / "clusters.parquet")
    signals = _safe_read_parquet(data_dir / "signals.parquet")
    backtest_results = _safe_read_parquet(data_dir / "backtest_results.parquet")
    trade_candidates = _safe_read_parquet(data_dir / "trade_candidates.parquet")

    report_path = data_dir / "analysis" / "report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}

    backtest_summary_path = data_dir / "analysis" / "backtest_summary.json"
    backtest_summary: dict[str, Any] = {}
    if backtest_summary_path.exists():
        try:
            backtest_summary = json.loads(backtest_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backtest_summary = {}

    candidates_json_path = data_dir / "analysis" / "trade_candidates.json"
    trade_candidates_json: dict[str, Any] = {}
    if candidates_json_path.exists():
        try:
            trade_candidates_json = json.loads(candidates_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            trade_candidates_json = {}

    events = events.copy()
    markets = markets.copy()
    tokens = tokens.copy()
    price_history = price_history.copy()
    market_quality = market_quality.copy()
    clusters = clusters.copy()
    signals = signals.copy()
    backtest_results = backtest_results.copy()
    trade_candidates = trade_candidates.copy()

    if not events.empty:
        events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype("Int64")
        events["primary_tag"] = events["tags"].apply(primary_tag_from_json)
        events["start_dt"] = _unix_to_datetime(events["start_ts"])
        events["end_dt"] = _unix_to_datetime(events["end_ts"])

    if not markets.empty:
        markets["market_id"] = markets["market_id"].astype(str)
        markets["event_id"] = pd.to_numeric(markets["event_id"], errors="coerce").astype("Int64")
        markets["liquidity"] = pd.to_numeric(markets["liquidity"], errors="coerce")
        markets["active"] = markets["active"].fillna(False).astype(bool)
        markets["closed"] = markets["closed"].fillna(False).astype(bool)

    if not tokens.empty:
        tokens["market_id"] = tokens["market_id"].astype(str)
        tokens["asset_id"] = tokens["asset_id"].astype(str)
        tokens["outcome"] = tokens["outcome"].fillna("Unknown")

    if not price_history.empty:
        price_history["asset_id"] = price_history["asset_id"].astype(str)
        price_history["ts"] = pd.to_numeric(price_history["ts"], errors="coerce")
        price_history["price"] = pd.to_numeric(price_history["price"], errors="coerce")
        price_history = price_history.dropna(subset=["asset_id", "ts", "price"])
        price_history = price_history.sort_values(["asset_id", "ts"])
        price_history["timestamp"] = _unix_to_datetime(price_history["ts"])

    if not market_quality.empty:
        market_quality["market_id"] = market_quality["market_id"].astype(str)
        market_quality["asset_id"] = market_quality["asset_id"].astype(str)
        for col in (
            "liquidity",
            "history_points_raw",
            "num_points",
            "missing_ratio",
            "price_range",
            "volatility",
            "return_total",
            "slope",
        ):
            if col in market_quality.columns:
                market_quality[col] = pd.to_numeric(market_quality[col], errors="coerce")
        if "quality_pass" in market_quality.columns:
            market_quality["quality_pass"] = market_quality["quality_pass"].fillna(False).astype(bool)

    if not clusters.empty:
        clusters["market_id"] = clusters["market_id"].astype(str)
        clusters["asset_id"] = clusters["asset_id"].astype(str)
        clusters["cluster_id"] = pd.to_numeric(clusters["cluster_id"], errors="coerce").astype("Int64")

    if not signals.empty:
        signals["asset_id"] = signals["asset_id"].astype(str)
        signals["market_id"] = signals["market_id"].astype(str)
        if "ts" in signals.columns:
            signals["ts"] = pd.to_numeric(signals["ts"], errors="coerce")
            signals["timestamp"] = _unix_to_datetime(signals["ts"])
        for col in ("confidence", "edge", "entry_price"):
            if col in signals.columns:
                signals[col] = pd.to_numeric(signals[col], errors="coerce")

    if not backtest_results.empty:
        for col in ("entry_ts", "exit_ts"):
            if col in backtest_results.columns:
                backtest_results[col] = pd.to_numeric(backtest_results[col], errors="coerce")
        if "exit_ts" in backtest_results.columns:
            backtest_results["exit_timestamp"] = _unix_to_datetime(backtest_results["exit_ts"])
        for col in ("pnl", "return_pct", "hold_days", "position_size", "bankroll_at_entry"):
            if col in backtest_results.columns:
                backtest_results[col] = pd.to_numeric(backtest_results[col], errors="coerce")

    if not trade_candidates.empty:
        for col in ("expected_value", "edge", "confidence", "suggested_size", "current_price"):
            if col in trade_candidates.columns:
                trade_candidates[col] = pd.to_numeric(trade_candidates[col], errors="coerce")

    market_table = build_market_table(
        events=events,
        markets=markets,
        tokens=tokens,
        price_history=price_history,
        market_quality=market_quality,
    )
    tag_table = build_tag_table(market_table)

    return {
        "data_dir": data_dir,
        "events": events,
        "markets": markets,
        "tokens": tokens,
        "price_history": price_history,
        "market_quality": market_quality,
        "clusters": clusters,
        "signals": signals,
        "backtest_results": backtest_results,
        "trade_candidates": trade_candidates,
        "market_table": market_table,
        "tag_table": tag_table,
        "report": report,
        "backtest_summary": backtest_summary,
        "trade_candidates_json": trade_candidates_json,
    }


def build_market_table(
    *,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    tokens: pd.DataFrame,
    price_history: pd.DataFrame,
    market_quality: pd.DataFrame,
) -> pd.DataFrame:
    market_table = markets.merge(
        events[
            [
                "event_id",
                "title",
                "slug",
                "primary_tag",
                "start_dt",
                "end_dt",
            ]
        ].rename(columns={"title": "event_title", "slug": "event_slug"}),
        on="event_id",
        how="left",
    )

    token_count = tokens.groupby("market_id", dropna=False)["asset_id"].nunique().rename("token_count")
    market_table = market_table.merge(token_count, left_on="market_id", right_index=True, how="left")

    anchor = tokens.copy()
    anchor["outcome_norm"] = anchor["outcome"].astype(str).str.strip().str.lower()
    anchor["is_anchor"] = anchor["outcome_norm"].isin({"yes", "true"})
    anchor = anchor.sort_values(["market_id", "is_anchor"], ascending=[True, False])
    anchor = anchor.groupby("market_id", as_index=False).first()[["market_id", "asset_id", "outcome"]]
    anchor = anchor.rename(columns={"asset_id": "anchor_asset_id", "outcome": "anchor_outcome"})
    market_table = market_table.merge(anchor, on="market_id", how="left")

    price_summary = summarize_prices(price_history)
    market_table = market_table.merge(
        price_summary,
        left_on="anchor_asset_id",
        right_on="asset_id",
        how="left",
    )

    if not market_quality.empty:
        quality = market_quality.groupby("market_id", dropna=False).agg(
            quality_pass_rate=("quality_pass", "mean"),
            quality_pass_any=("quality_pass", "max"),
            avg_missing_ratio=("missing_ratio", "mean"),
            avg_volatility=("volatility", "mean"),
            avg_price_range=("price_range", "mean"),
            history_points=("history_points_raw", "max"),
        )
        quality = quality.reset_index()
        market_table = market_table.merge(quality, on="market_id", how="left")

    market_table["token_count"] = market_table["token_count"].fillna(0).astype(int)
    market_table["has_history"] = market_table["points"].fillna(0) > 0
    if "quality_pass_any" not in market_table.columns:
        market_table["quality_pass_any"] = False
    else:
        market_table["quality_pass_any"] = market_table["quality_pass_any"].fillna(False).astype(bool)
    if "quality_pass_rate" not in market_table.columns:
        market_table["quality_pass_rate"] = 0.0
    else:
        market_table["quality_pass_rate"] = pd.to_numeric(market_table["quality_pass_rate"], errors="coerce").fillna(0.0)
    market_table["liquidity"] = pd.to_numeric(market_table["liquidity"], errors="coerce")

    if "event_title" in market_table.columns:
        market_table["event_title"] = market_table["event_title"].fillna("Unknown event")
    market_table["primary_tag"] = market_table["primary_tag"].fillna("unknown")
    market_table["question"] = market_table["question"].fillna("Untitled market")
    return market_table


def summarize_prices(price_history: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "asset_id",
        "points",
        "first_price",
        "last_price",
        "window_return",
        "window_return_pct",
        "max_price",
        "min_price",
        "first_timestamp",
        "last_timestamp",
    ]
    if price_history.empty:
        return pd.DataFrame(columns=cols)

    base = price_history[["asset_id", "price", "timestamp"]].copy()
    grouped = base.groupby("asset_id", dropna=False)
    summary = grouped.agg(
        points=("price", "size"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        first_timestamp=("timestamp", "first"),
        last_timestamp=("timestamp", "last"),
    )
    summary = summary.reset_index()
    summary["window_return"] = summary["last_price"] - summary["first_price"]
    summary["window_return_pct"] = summary["window_return"] * 100.0
    return summary


def build_tag_table(market_table: pd.DataFrame) -> pd.DataFrame:
    if market_table.empty:
        return pd.DataFrame(
            columns=[
                "primary_tag",
                "markets",
                "coverage_pct",
                "quality_pass_pct",
                "avg_liquidity",
                "median_return_pct",
            ]
        )

    grouped = market_table.groupby("primary_tag", dropna=False).agg(
        markets=("market_id", "nunique"),
        coverage_pct=("has_history", "mean"),
        quality_pass_pct=("quality_pass_any", "mean"),
        avg_liquidity=("liquidity", "mean"),
        median_return_pct=("window_return_pct", "median"),
    )
    grouped = grouped.reset_index().sort_values("markets", ascending=False)
    return grouped


def apply_market_filters(
    market_table: pd.DataFrame,
    *,
    search_text: str,
    selected_tags: list[str],
    selected_types: list[str],
    status_mode: str,
    min_liquidity: float,
    require_history: bool,
    require_quality: bool,
) -> pd.DataFrame:
    filtered = market_table.copy()

    if search_text:
        query = search_text.lower().strip()
        mask = (
            filtered["question"].astype(str).str.lower().str.contains(query, na=False)
            | filtered["event_title"].astype(str).str.lower().str.contains(query, na=False)
            | filtered["market_id"].astype(str).str.lower().str.contains(query, na=False)
        )
        filtered = filtered.loc[mask]

    if selected_tags:
        filtered = filtered.loc[filtered["primary_tag"].isin(selected_tags)]

    if selected_types:
        filtered = filtered.loc[filtered["market_type"].isin(selected_types)]

    if status_mode == "active_only":
        filtered = filtered.loc[filtered["active"]]
    elif status_mode == "closed_only":
        filtered = filtered.loc[filtered["closed"]]

    if min_liquidity > 0:
        filtered = filtered.loc[filtered["liquidity"].fillna(0) >= min_liquidity]

    if require_history:
        filtered = filtered.loc[filtered["has_history"]]

    if require_quality:
        filtered = filtered.loc[filtered["quality_pass_any"]]

    return filtered


def downsample_history(
    df: pd.DataFrame,
    *,
    max_points_per_line: int,
    group_col: str = "outcome_label",
) -> pd.DataFrame:
    if df.empty:
        return df

    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(group_col, dropna=False):
        if len(group) <= max_points_per_line:
            pieces.append(group)
            continue
        step = max(1, len(group) // max_points_per_line)
        sampled = group.iloc[::step].copy()
        if sampled.iloc[-1]["timestamp"] != group.iloc[-1]["timestamp"]:
            sampled = pd.concat([sampled, group.tail(1)], ignore_index=True)
        pieces.append(sampled)
    return pd.concat(pieces, ignore_index=True)


def market_price_frame(
    market_id: str,
    *,
    tokens: pd.DataFrame,
    price_history: pd.DataFrame,
) -> pd.DataFrame:
    market_tokens = tokens.loc[tokens["market_id"] == str(market_id), ["asset_id", "outcome"]].copy()
    if market_tokens.empty or price_history.empty:
        return pd.DataFrame(columns=["timestamp", "price", "asset_id", "outcome_label"])

    market_tokens["asset_id"] = market_tokens["asset_id"].astype(str)
    market_tokens["outcome_label"] = market_tokens["outcome"].fillna("Unknown")

    frame = price_history.loc[price_history["asset_id"].isin(market_tokens["asset_id"]), ["timestamp", "price", "asset_id"]]
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "price", "asset_id", "outcome_label"])

    frame = frame.merge(market_tokens[["asset_id", "outcome_label"]], on="asset_id", how="left")
    frame["outcome_label"] = frame["outcome_label"].fillna(frame["asset_id"])
    frame = frame.sort_values(["outcome_label", "timestamp"])
    return frame


def event_price_frames(
    event_id: int,
    *,
    market_table: pd.DataFrame,
    price_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_series = pd.DataFrame(columns=["timestamp", "price", "asset_id", "market_id", "series_label"])
    empty_agg = pd.DataFrame(columns=["timestamp", "median_price", "p25", "p75", "mean_price", "market_count"])

    event_markets = market_table.loc[market_table["event_id"] == int(event_id)].copy()
    if event_markets.empty:
        return event_markets, empty_series, empty_agg

    event_markets = event_markets.dropna(subset=["anchor_asset_id"]).copy()
    if event_markets.empty or price_history.empty:
        return event_markets, empty_series, empty_agg

    event_markets["anchor_asset_id"] = event_markets["anchor_asset_id"].astype(str)

    mapping = event_markets[["anchor_asset_id", "market_id", "question"]].copy()
    mapping = mapping.rename(columns={"anchor_asset_id": "asset_id"})
    mapping["series_label"] = mapping["question"].astype(str).str.slice(0, 65) + " | " + mapping["market_id"].astype(str)

    event_series = price_history.loc[
        price_history["asset_id"].isin(mapping["asset_id"]),
        ["timestamp", "price", "asset_id"],
    ].copy()
    if event_series.empty:
        return event_markets, empty_series, empty_agg

    event_series = event_series.merge(mapping, on="asset_id", how="left")
    event_series = event_series.sort_values(["series_label", "timestamp"])

    event_agg = (
        event_series.groupby("timestamp", dropna=False)["price"]
        .agg(
            median_price="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            mean_price="mean",
            market_count="size",
        )
        .reset_index()
        .sort_values("timestamp")
    )

    return event_markets, event_series, event_agg


def render_kpis(filtered: pd.DataFrame, total_markets: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    filtered_count = len(filtered)

    history_rate = float((filtered["has_history"].mean() * 100.0) if filtered_count else 0.0)
    quality_rate = float((filtered["quality_pass_any"].mean() * 100.0) if filtered_count else 0.0)
    avg_liquidity = float(filtered["liquidity"].mean()) if filtered_count else 0.0

    c1.metric("Visible Bets", f"{filtered_count:,}", delta=f"of {total_markets:,}")
    c2.metric("With History", f"{history_rate:.1f}%")
    c3.metric("Quality Pass", f"{quality_rate:.1f}%")
    c4.metric("Avg Liquidity", f"{avg_liquidity:,.0f}")


def render_market_list(filtered: pd.DataFrame) -> None:
    list_columns = [
        "market_id",
        "question",
        "event_title",
        "primary_tag",
        "market_type",
        "liquidity",
        "last_price",
        "window_return_pct",
        "points",
        "quality_pass_any",
    ]
    view = filtered[list_columns].copy()
    view = view.rename(
        columns={
            "market_id": "Market ID",
            "question": "Question",
            "event_title": "Event",
            "primary_tag": "Tag",
            "market_type": "Type",
            "liquidity": "Liquidity",
            "last_price": "Last Price",
            "window_return_pct": "Window Return %",
            "points": "Points",
            "quality_pass_any": "Quality Pass",
        }
    )
    st.dataframe(view, width="stretch", hide_index=True, height=420)


def render_bet_detail(
    selected_market_id: str,
    *,
    market_table: pd.DataFrame,
    tokens: pd.DataFrame,
    price_history: pd.DataFrame,
    market_quality: pd.DataFrame,
    clusters: pd.DataFrame,
    max_points_per_line: int,
) -> None:
    row = market_table.loc[market_table["market_id"] == str(selected_market_id)]
    if row.empty:
        st.warning("Select a market from the list to inspect details.")
        return

    selected = row.iloc[0]
    st.markdown("### Bet Detail")

    info_cols = st.columns(3)
    info_cols[0].markdown(f"**Question**: {selected['question']}")
    info_cols[0].markdown(f"**Event**: {selected['event_title']}")
    info_cols[1].markdown(f"**Tag**: {selected['primary_tag']}")
    info_cols[1].markdown(f"**Type**: {selected['market_type']}")
    info_cols[2].markdown(f"**Liquidity**: {selected['liquidity'] if pd.notna(selected['liquidity']) else 'n/a'}")
    info_cols[2].markdown(f"**Market ID**: `{selected['market_id']}`")

    history = market_price_frame(str(selected_market_id), tokens=tokens, price_history=price_history)
    history = downsample_history(history, max_points_per_line=max_points_per_line)

    if history.empty:
        st.info("No price history available for this market's tokens.")
    else:
        if alt is not None:
            chart = (
                alt.Chart(history)
                .mark_line(interpolate="monotone")
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("price:Q", title="Price", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("outcome_label:N", title="Outcome"),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("outcome_label:N", title="Outcome"),
                        alt.Tooltip("price:Q", format=".4f", title="Price"),
                        alt.Tooltip("asset_id:N", title="Asset ID"),
                    ],
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(_style_altair_chart(chart), width="stretch")
        else:
            pivot = history.pivot_table(index="timestamp", columns="outcome_label", values="price", aggfunc="last")
            st.line_chart(pivot, height=350)

        latest = history.sort_values("timestamp").groupby("outcome_label", as_index=False).tail(1)
        latest = latest[["outcome_label", "price", "timestamp", "asset_id"]].rename(
            columns={
                "outcome_label": "Outcome",
                "price": "Latest Price",
                "timestamp": "Timestamp",
                "asset_id": "Asset ID",
            }
        )
        st.dataframe(latest.sort_values("Outcome"), width="stretch", hide_index=True)

    quality_columns = [
        "asset_id",
        "history_points_raw",
        "missing_ratio",
        "price_range",
        "volatility",
        "return_total",
        "quality_pass",
    ]
    available_quality_cols = [col for col in quality_columns if col in market_quality.columns]
    quality_rows = pd.DataFrame()
    if "market_id" in market_quality.columns and available_quality_cols:
        quality_rows = market_quality.loc[
            market_quality["market_id"] == str(selected_market_id),
            available_quality_cols,
        ]
    if not quality_rows.empty:
        st.markdown("#### Token Quality")
        st.dataframe(quality_rows, width="stretch", hide_index=True)

    cluster_rows = pd.DataFrame()
    cluster_columns = [col for col in ("asset_id", "cluster_id", "primary_tag") if col in clusters.columns]
    if not clusters.empty and "market_id" in clusters.columns and cluster_columns:
        cluster_rows = clusters.loc[clusters["market_id"] == str(selected_market_id), cluster_columns]
        if not cluster_rows.empty:
            st.markdown("#### Cluster Membership")
            st.dataframe(cluster_rows, width="stretch", hide_index=True)


def render_event_view(
    *,
    filtered: pd.DataFrame,
    market_table: pd.DataFrame,
    price_history: pd.DataFrame,
    max_points_per_line: int,
) -> None:
    st.markdown("### Event Explorer")
    st.markdown("<p class='section-note'>Pick one event to see a composite price trajectory.</p>", unsafe_allow_html=True)

    source = filtered if not filtered.empty else market_table
    event_options = source[["event_id", "event_title", "primary_tag"]].dropna(subset=["event_id"]).copy()
    if event_options.empty:
        st.info("No events available for the current filters.")
        return

    event_options["event_id"] = pd.to_numeric(event_options["event_id"], errors="coerce")
    event_options = event_options.dropna(subset=["event_id"]).drop_duplicates(subset=["event_id"]).copy()
    if event_options.empty:
        st.info("No events available for the current filters.")
        return

    event_options["event_id"] = event_options["event_id"].astype(int)
    event_options = event_options.sort_values(["event_title", "event_id"])
    event_options["label"] = (
        event_options["event_title"].astype(str)
        + " | "
        + event_options["primary_tag"].astype(str)
        + " | "
        + event_options["event_id"].astype(str)
    )

    selected_label = st.selectbox("Select event", options=event_options["label"].tolist())
    selected_event_id = int(event_options.loc[event_options["label"] == selected_label, "event_id"].iloc[0])

    event_markets, event_series, event_agg = event_price_frames(
        selected_event_id,
        market_table=market_table,
        price_history=price_history,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Markets", f"{event_markets['market_id'].nunique():,}" if not event_markets.empty else "0")
    c2.metric("With History", f"{int(event_markets['has_history'].sum()):,}" if "has_history" in event_markets.columns else "0")
    c3.metric(
        "Median Liquidity",
        f"{float(event_markets['liquidity'].median()):,.0f}" if "liquidity" in event_markets.columns and event_markets["liquidity"].notna().any() else "n/a",
    )
    c4.metric(
        "Median Last Price",
        f"{float(event_markets['last_price'].median()):.3f}" if "last_price" in event_markets.columns and event_markets["last_price"].notna().any() else "n/a",
    )

    if event_agg.empty:
        st.info("No anchor-token price history found for this event.")
    else:
        if alt is not None:
            band = (
                alt.Chart(event_agg)
                .mark_area(opacity=0.22, color="#2e9d90")
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("p25:Q", title="Composite Price", scale=alt.Scale(domain=[0, 1])),
                    y2="p75:Q",
                )
            )
            median = (
                alt.Chart(event_agg)
                .mark_line(color="#0f3d5e", strokeWidth=2.6)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("median_price:Q", title="Composite Price", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("median_price:Q", title="Median", format=".4f"),
                        alt.Tooltip("mean_price:Q", title="Mean", format=".4f"),
                        alt.Tooltip("market_count:Q", title="Markets"),
                    ],
                )
            )
            st.altair_chart(_style_altair_chart((band + median).properties(height=340).interactive()), width="stretch")
        else:
            st.line_chart(event_agg.set_index("timestamp")[["median_price"]], height=340)

    show_market_lines = st.checkbox("Show top market lines for this event", value=False)
    if show_market_lines and not event_series.empty:
        top_n = st.slider("Top markets by liquidity", min_value=5, max_value=30, value=12, step=1)
        top_assets = (
            event_markets.sort_values("liquidity", ascending=False)["anchor_asset_id"].dropna().astype(str).head(top_n).tolist()
        )
        line_df = event_series.loc[event_series["asset_id"].isin(top_assets)].copy()
        line_df = downsample_history(line_df, max_points_per_line=max_points_per_line, group_col="series_label")
        if not line_df.empty and alt is not None:
            line_chart = (
                alt.Chart(line_df)
                .mark_line(opacity=0.75)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("price:Q", title="Price", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("series_label:N", title="Market"),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("series_label:N", title="Market"),
                        alt.Tooltip("price:Q", title="Price", format=".4f"),
                    ],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(_style_altair_chart(line_chart), width="stretch")
        elif not line_df.empty:
            pivot = line_df.pivot_table(index="timestamp", columns="series_label", values="price", aggfunc="last")
            st.line_chart(pivot, height=360)

    table_cols = [
        "market_id",
        "question",
        "liquidity",
        "last_price",
        "window_return_pct",
        "points",
        "quality_pass_any",
    ]
    available = [col for col in table_cols if col in event_markets.columns]
    if available:
        st.markdown("#### Event Markets")
        st.dataframe(
            event_markets[available].sort_values("liquidity", ascending=False).head(250),
            width="stretch",
            hide_index=True,
        )


def render_market_map(filtered: pd.DataFrame) -> None:
    st.markdown("### Market Map")
    st.markdown("<p class='section-note'>Liquidity vs. recent return for currently filtered bets.</p>", unsafe_allow_html=True)

    if filtered.empty:
        st.info("No rows to plot for current filters.")
        return

    map_df = filtered[["market_id", "question", "primary_tag", "liquidity", "window_return_pct", "points"]].copy()
    map_df = map_df.dropna(subset=["liquidity", "window_return_pct"])

    if map_df.empty:
        st.info("Current filters have no rows with both liquidity and return values.")
        return

    top_tags = map_df["primary_tag"].value_counts().head(10).index.tolist()
    map_df["tag_bucket"] = map_df["primary_tag"].where(map_df["primary_tag"].isin(top_tags), "Other")

    if alt is not None:
        chart = (
            alt.Chart(map_df)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("liquidity:Q", title="Liquidity"),
                y=alt.Y("window_return_pct:Q", title="Window Return (%)"),
                size=alt.Size("points:Q", title="History Points"),
                color=alt.Color("tag_bucket:N", title="Tag"),
                tooltip=[
                    alt.Tooltip("market_id:N", title="Market ID"),
                    alt.Tooltip("question:N", title="Question"),
                    alt.Tooltip("primary_tag:N", title="Tag"),
                    alt.Tooltip("liquidity:Q", title="Liquidity", format=",.0f"),
                    alt.Tooltip("window_return_pct:Q", title="Return %", format=".2f"),
                    alt.Tooltip("points:Q", title="Points"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(_style_altair_chart(chart), width="stretch")
    else:
        st.scatter_chart(map_df, x="liquidity", y="window_return_pct", size="points", color="tag_bucket")

    movers = map_df.copy()
    movers["abs_return"] = movers["window_return_pct"].abs()
    movers = movers.sort_values("abs_return", ascending=False).head(20)
    st.markdown("#### Biggest Movers (Absolute Return)")
    st.dataframe(
        movers[["market_id", "question", "primary_tag", "liquidity", "window_return_pct", "points"]],
        width="stretch",
        hide_index=True,
    )


def render_tag_cluster_views(tag_table: pd.DataFrame, clusters: pd.DataFrame, report: dict[str, Any]) -> None:
    st.markdown("### Tag and Cluster Lens")

    if not tag_table.empty:
        show = tag_table.head(30).copy()
        show["coverage_pct"] = show["coverage_pct"] * 100.0
        show["quality_pass_pct"] = show["quality_pass_pct"] * 100.0

        if alt is not None:
            chart = (
                alt.Chart(show)
                .mark_bar()
                .encode(
                    x=alt.X("markets:Q", title="Markets"),
                    y=alt.Y("primary_tag:N", sort="-x", title="Tag"),
                    color=alt.Color("coverage_pct:Q", title="Coverage %", scale=alt.Scale(scheme="tealblues")),
                    tooltip=[
                        alt.Tooltip("primary_tag:N", title="Tag"),
                        alt.Tooltip("markets:Q", title="Markets"),
                        alt.Tooltip("coverage_pct:Q", title="Coverage %", format=".1f"),
                        alt.Tooltip("quality_pass_pct:Q", title="Quality Pass %", format=".1f"),
                        alt.Tooltip("avg_liquidity:Q", title="Avg Liquidity", format=",.0f"),
                    ],
                )
                .properties(height=520)
            )
            st.altair_chart(_style_altair_chart(chart), width="stretch")

        st.markdown("#### Tag Table")
        st.dataframe(show, width="stretch", hide_index=True)

    if not clusters.empty:
        st.markdown("#### Cluster Sizes")
        cluster_sizes = clusters.groupby("cluster_id", dropna=False).size().reset_index(name="assets")
        cluster_sizes = cluster_sizes.sort_values("assets", ascending=False)
        st.dataframe(cluster_sizes, width="stretch", hide_index=True)

    if report:
        st.markdown("#### Run Snapshot")
        overview = {
            "generated_at_utc": report.get("generated_at_utc"),
            "coverage": report.get("coverage"),
            "quality": report.get("quality"),
            "clusters": report.get("clusters", {}).get("num_clusters") if isinstance(report.get("clusters"), dict) else None,
        }
        st.json(overview, expanded=False)


def render_backtest_results(backtest_results: pd.DataFrame, backtest_summary: dict[str, Any]) -> None:
    st.markdown("<p class='section-note'>Historical strategy P&L and trade outcomes.</p>", unsafe_allow_html=True)
    if backtest_results is None or backtest_results.empty:
        st.info("No backtest data available.")
        return

    c1, c2, c3, c4 = st.columns(4)
    pnl_total = float(pd.to_numeric(backtest_results.get("pnl"), errors="coerce").fillna(0.0).sum())
    trades = int(len(backtest_results))
    win_rate = float((pd.to_numeric(backtest_results.get("pnl"), errors="coerce").fillna(0.0) > 0).mean()) if trades else 0.0
    avg_hold = float(pd.to_numeric(backtest_results.get("hold_days"), errors="coerce").dropna().mean()) if trades else 0.0
    c1.metric("Trades", f"{trades:,}")
    c2.metric("Total PnL", f"{pnl_total:,.2f}")
    c3.metric("Win Rate", f"{win_rate * 100:.1f}%")
    c4.metric("Avg Hold Days", f"{avg_hold:.1f}")

    curve = backtest_results[["exit_timestamp", "pnl"]].copy()
    curve = curve.dropna(subset=["exit_timestamp"])
    curve = curve.sort_values("exit_timestamp")
    curve["cum_pnl"] = pd.to_numeric(curve["pnl"], errors="coerce").fillna(0.0).cumsum()
    if not curve.empty:
        st.line_chart(curve.set_index("exit_timestamp")[["cum_pnl"]], height=320)

    if backtest_summary:
        by_signal = backtest_summary.get("by_signal", {})
        if isinstance(by_signal, dict) and by_signal:
            rows = []
            for signal_name, stats in by_signal.items():
                if not isinstance(stats, dict):
                    continue
                rows.append({"signal_name": signal_name, **stats})
            if rows:
                st.markdown("#### Summary by Signal")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Trade Log")
    show_cols = [
        "signal_name",
        "asset_id",
        "market_id",
        "direction",
        "entry_price",
        "exit_price",
        "pnl",
        "return_pct",
        "hold_days",
        "exit_reason",
    ]
    available = [col for col in show_cols if col in backtest_results.columns]
    st.dataframe(backtest_results[available].sort_values("pnl", ascending=False), width="stretch", hide_index=True)


def render_trade_candidates_panel(trade_candidates: pd.DataFrame, trade_candidates_json: dict[str, Any]) -> None:
    st.markdown("<p class='section-note'>Current ranked trade ideas from active signals.</p>", unsafe_allow_html=True)
    if trade_candidates is None or trade_candidates.empty:
        rows = trade_candidates_json.get("candidates", []) if isinstance(trade_candidates_json, dict) else []
        if not rows:
            st.info("No trade candidates available.")
            return
        trade_candidates = pd.DataFrame(rows)

    sort_col = "expected_value" if "expected_value" in trade_candidates.columns else "edge"
    if sort_col in trade_candidates.columns:
        trade_candidates = trade_candidates.sort_values(sort_col, ascending=False)
    st.dataframe(trade_candidates, width="stretch", hide_index=True)


def render_signal_analysis(signals: pd.DataFrame) -> None:
    st.markdown("<p class='section-note'>Signal frequency, confidence, and edge diagnostics.</p>", unsafe_allow_html=True)
    if signals is None or signals.empty:
        st.info("No signal data available.")
        return

    if "signal_name" in signals.columns:
        counts = signals.groupby("signal_name", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        st.markdown("#### Signal Counts")
        st.dataframe(counts, width="stretch", hide_index=True)

    if "timestamp" in signals.columns and "confidence" in signals.columns:
        frame = signals.dropna(subset=["timestamp", "confidence"]).copy()
        if not frame.empty:
            frame["day"] = frame["timestamp"].dt.floor("1d")
            roll = frame.groupby(["day", "signal_name"], dropna=False)["confidence"].mean().reset_index()
            pivot = roll.pivot_table(index="day", columns="signal_name", values="confidence", aggfunc="mean")
            if not pivot.empty:
                st.markdown("#### Mean Confidence Over Time")
                st.line_chart(pivot, height=300)

    if "confidence" in signals.columns and "edge" in signals.columns:
        scatter = signals[["confidence", "edge", "signal_name"]].dropna().copy()
        if not scatter.empty:
            st.markdown("#### Confidence vs Edge")
            st.scatter_chart(scatter, x="confidence", y="edge", color="signal_name")


def run_app() -> None:
    args = _parse_cli_args()

    st.set_page_config(
        page_title="Polymarket Data Discovery",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Polymarket Data Discovery</h1>
            <p>Filter bets, inspect price trajectories by outcome, and surface market/tag outliers from the latest parquet snapshot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Data")
    data_dir_text = st.sidebar.text_input("Data directory", value=args.data_dir)
    if st.sidebar.button("Reload data", width="stretch"):
        load_data_bundle.clear()

    st.sidebar.markdown("### Filters")

    try:
        bundle = load_data_bundle(data_dir_text)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `polymarket-pipeline --output-dir data` first, or point to an existing output dir.")
        st.stop()

    market_table = bundle["market_table"]
    tags = sorted(market_table["primary_tag"].dropna().astype(str).unique().tolist())
    market_types = sorted(market_table["market_type"].dropna().astype(str).unique().tolist())

    search_text = st.sidebar.text_input("Search question / event / market ID", value="")
    selected_tags = st.sidebar.multiselect("Tags", options=tags)
    selected_types = st.sidebar.multiselect("Market types", options=market_types, default=market_types)

    status_mode = st.sidebar.radio(
        "Status",
        options=["all", "active_only", "closed_only"],
        format_func=lambda x: {
            "all": "All",
            "active_only": "Active only",
            "closed_only": "Closed only",
        }[x],
        horizontal=False,
    )

    max_liq = float(market_table["liquidity"].fillna(0).max()) if not market_table.empty else 0.0
    min_liquidity = st.sidebar.number_input(
        "Min liquidity",
        min_value=0.0,
        max_value=max(0.0, max_liq),
        value=0.0,
        step=max(1.0, max_liq / 100.0) if max_liq > 0 else 1.0,
    )

    require_history = st.sidebar.checkbox("Require history", value=False)
    require_quality = st.sidebar.checkbox("Require quality pass", value=False)

    sort_column = st.sidebar.selectbox(
        "Sort by",
        options=[
            "liquidity",
            "window_return_pct",
            "last_price",
            "points",
            "quality_pass_rate",
        ],
        index=0,
    )
    sort_desc = st.sidebar.checkbox("Descending", value=True)
    row_limit = st.sidebar.slider("Max rows", min_value=25, max_value=1000, value=250, step=25)

    filtered = apply_market_filters(
        market_table,
        search_text=search_text,
        selected_tags=selected_tags,
        selected_types=selected_types,
        status_mode=status_mode,
        min_liquidity=float(min_liquidity),
        require_history=require_history,
        require_quality=require_quality,
    )

    if sort_column in filtered.columns:
        filtered = filtered.sort_values(sort_column, ascending=not sort_desc)
    filtered = filtered.head(row_limit)

    render_kpis(filtered, total_markets=len(market_table))

    tab_names = ["Bet Explorer", "Event Explorer", "Market Map", "Tags and Clusters"]
    if args.ui_mode == "full":
        tab_names.extend(["Backtest Results", "Trade Candidates", "Signal Analysis"])
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("<p class='section-note'>Explore the current filtered list, then inspect one market in detail.</p>", unsafe_allow_html=True)
        render_market_list(filtered)

        if filtered.empty:
            st.info("No bets match current filters.")
        else:
            selection_rows = filtered[["market_id", "question", "primary_tag", "event_title"]].copy()
            selection_rows["label"] = (
                selection_rows["question"].astype(str).str.slice(0, 90)
                + " | "
                + selection_rows["primary_tag"].astype(str)
                + " | "
                + selection_rows["market_id"].astype(str)
            )
            labels = selection_rows["label"].tolist()
            label_to_id = dict(zip(selection_rows["label"], selection_rows["market_id"], strict=True))
            selected_label = st.selectbox("Inspect market", options=labels)
            selected_market_id = str(label_to_id[selected_label])

            render_bet_detail(
                selected_market_id,
                market_table=market_table,
                tokens=bundle["tokens"],
                price_history=bundle["price_history"],
                market_quality=bundle["market_quality"],
                clusters=bundle["clusters"],
                max_points_per_line=max(200, int(args.max_points_per_line)),
            )

    with tabs[1]:
        render_event_view(
            filtered=filtered,
            market_table=market_table,
            price_history=bundle["price_history"],
            max_points_per_line=max(200, int(args.max_points_per_line)),
        )

    with tabs[2]:
        render_market_map(filtered)

    with tabs[3]:
        render_tag_cluster_views(bundle["tag_table"], bundle["clusters"], bundle["report"])

    if args.ui_mode == "full":
        with tabs[4]:
            render_backtest_results(bundle.get("backtest_results", pd.DataFrame()), bundle.get("backtest_summary", {}))
        with tabs[5]:
            render_trade_candidates_panel(
                bundle.get("trade_candidates", pd.DataFrame()),
                bundle.get("trade_candidates_json", {}),
            )
        with tabs[6]:
            render_signal_analysis(bundle.get("signals", pd.DataFrame()))


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
