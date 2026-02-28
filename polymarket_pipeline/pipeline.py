from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import (
    FEATURE_COLUMNS,
    build_bet_type_summary,
    build_market_quality_table,
    build_report_payload,
    cluster_features,
    compute_asset_features,
    save_bet_type_plot,
    save_cluster_plots,
    write_report_json,
)
from .client import PolymarketClients
from .config import PipelineConfig
from .normalize import extract_tables, primary_tag_from_json, select_price_tokens
from .storage import upsert_parquet, write_jsonl

LOGGER = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.metrics: dict[str, int] = {
            "requests": 0,
            "retries": 0,
            "empty_histories": 0,
            "price_history_errors": 0,
        }

    def run(self) -> dict[str, Path]:
        self.config.ensure_dirs()

        clients = PolymarketClients(
            timeout_s=self.config.request_timeout_s,
            max_retries=self.config.max_retries,
            backoff_base_s=self.config.backoff_base_s,
            backoff_jitter_s=self.config.backoff_jitter_s,
            gamma_rate_limit=self.config.gamma_rate_limit,
            clob_rate_limit=self.config.clob_rate_limit,
            data_rate_limit=self.config.data_rate_limit,
            rate_window_s=self.config.rate_window_s,
            metrics=self.metrics,
            connection_pool_maxsize=self.config.http_pool_maxsize,
        )

        events = self._discover_events(clients)
        raw_events_path = self.config.raw_dir / f"events_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        write_jsonl(raw_events_path, events, mode="w")

        events_new, markets_new, tokens_new = extract_tables(events)
        events_path = self.config.output_dir / "events.parquet"
        markets_path = self.config.output_dir / "markets.parquet"
        tokens_path = self.config.output_dir / "tokens.parquet"

        events_df = upsert_parquet(events_path, events_new, dedupe_keys=["event_id"], sort_keys=["event_id"])
        markets_df = upsert_parquet(markets_path, markets_new, dedupe_keys=["market_id"], sort_keys=["market_id"])
        tokens_df = upsert_parquet(
            tokens_path,
            tokens_new,
            dedupe_keys=["market_id", "asset_id"],
            sort_keys=["market_id", "asset_id"],
        )

        target_tokens_df = select_price_tokens(
            tokens_df,
            markets_df,
            yes_only_binary=self.config.yes_only_binary,
        )
        price_history_df = self._ingest_price_history(clients, target_tokens_df)

        if self.config.fetch_trades_sample > 0:
            self._fetch_trades_sample(clients, markets_df)

        report_path, clusters_path, quality_path = self._analyze(
            events_df=events_df,
            markets_df=markets_df,
            all_tokens_df=tokens_df,
            target_tokens_df=target_tokens_df,
            price_history_df=price_history_df,
        )

        LOGGER.info(
            "Pipeline done. requests=%s retries=%s empty_histories=%s price_history_errors=%s skipped_price_assets=%s",
            self.metrics.get("requests", 0),
            self.metrics.get("retries", 0),
            self.metrics.get("empty_histories", 0),
            self.metrics.get("price_history_errors", 0),
            self.metrics.get("skipped_price_assets", 0),
        )

        return {
            "events": events_path,
            "markets": markets_path,
            "tokens": tokens_path,
            "price_history": self.config.output_dir / "price_history.parquet",
            "clusters": clusters_path,
            "market_quality": quality_path,
            "report": report_path,
            "raw_events": raw_events_path,
        }

    def _discover_events(self, clients: PolymarketClients) -> list[dict[str, Any]]:
        LOGGER.info("Discovering events from Gamma...")
        events: list[dict[str, Any]] = []
        offset = 0

        while len(events) < self.config.max_events:
            limit = min(self.config.page_limit, self.config.max_events - len(events))
            page = clients.fetch_events_page(
                limit=limit,
                offset=offset,
                active=True,
                closed=False,
                order="volume24hr",
                cache_dir=self.config.cache_dir / "gamma_events",
                cache_ttl_s=self.config.gamma_cache_ttl_s,
            )
            if not page:
                break
            events.extend(page)
            LOGGER.info("Fetched %s events so far", len(events))
            if len(page) < limit:
                break
            offset += limit

        return events[: self.config.max_events]

    def _ingest_price_history(self, clients: PolymarketClients, target_tokens_df: pd.DataFrame) -> pd.DataFrame:
        price_path = self.config.output_dir / "price_history.parquet"
        output_columns = ["asset_id", "ts", "price", "interval", "ingested_at"]
        if target_tokens_df.empty:
            return upsert_parquet(
                price_path,
                pd.DataFrame(columns=output_columns),
                dedupe_keys=["asset_id", "ts", "interval"],
                sort_keys=["asset_id", "ts"],
            )

        asset_ids = target_tokens_df["asset_id"].dropna().astype(str).drop_duplicates().tolist()
        now_ts = int(time.time())
        global_start_ts = now_ts - self.config.window_days * 24 * 60 * 60
        ingested_at = int(time.time())

        existing_latest_ts: dict[str, int] = {}
        if self.config.incremental_prices and price_path.exists():
            try:
                existing_df = pd.read_parquet(price_path, columns=["asset_id", "interval", "ts"])
            except Exception:
                existing_df = pd.read_parquet(price_path)
            if not existing_df.empty and "asset_id" in existing_df.columns:
                existing_df["asset_id"] = existing_df["asset_id"].astype(str)
                if "interval" in existing_df.columns:
                    interval_mask = existing_df["interval"].fillna("").astype(str) == str(self.config.interval)
                    existing_df = existing_df.loc[interval_mask].copy()
                if not existing_df.empty and "ts" in existing_df.columns:
                    existing_df["ts"] = pd.to_numeric(existing_df["ts"], errors="coerce")
                    existing_df = existing_df.dropna(subset=["ts"])
                    if not existing_df.empty:
                        existing_df["ts"] = existing_df["ts"].astype("int64")
                        existing_latest_ts = (
                            existing_df.groupby("asset_id", as_index=True)["ts"].max().astype(int).to_dict()
                        )

        incremental_mode = str(self.config.incremental_mode).strip().lower()
        if incremental_mode not in {"tail", "skip"}:
            incremental_mode = "tail"

        fetch_plan: list[tuple[str, int]] = []
        if not self.config.incremental_prices:
            fetch_plan = [(asset_id, global_start_ts) for asset_id in asset_ids]
            self.metrics["skipped_price_assets"] = 0
            LOGGER.info("Incremental disabled; fetching prices-history for %s assets", len(fetch_plan))
        elif incremental_mode == "skip":
            fetch_plan = [
                (asset_id, global_start_ts)
                for asset_id in asset_ids
                if asset_id not in existing_latest_ts
            ]
            skipped_assets = len(asset_ids) - len(fetch_plan)
            self.metrics["skipped_price_assets"] = skipped_assets
            if skipped_assets > 0:
                LOGGER.info(
                    "Skipping %s assets with existing %s interval history; fetching %s assets",
                    skipped_assets,
                    self.config.interval,
                    len(fetch_plan),
                )
            else:
                LOGGER.info("Fetching prices-history for %s assets", len(fetch_plan))
        else:
            interval_s = self._interval_to_seconds(self.config.interval) or 3600
            overlap_points = max(0, int(self.config.incremental_overlap_points))
            overlap_s = overlap_points * interval_s
            existing_count = 0
            for asset_id in asset_ids:
                latest_ts = existing_latest_ts.get(asset_id)
                if latest_ts is None:
                    fetch_plan.append((asset_id, global_start_ts))
                else:
                    existing_count += 1
                    fetch_plan.append((asset_id, max(global_start_ts, int(latest_ts) - overlap_s)))
            self.metrics["skipped_price_assets"] = 0
            LOGGER.info(
                "Tail refresh for %s assets (%s existing, %s new), overlap_points=%s",
                len(fetch_plan),
                existing_count,
                len(fetch_plan) - existing_count,
                overlap_points,
            )

        if not fetch_plan:
            return upsert_parquet(
                price_path,
                pd.DataFrame(columns=output_columns),
                dedupe_keys=["asset_id", "ts", "interval"],
                sort_keys=["asset_id", "ts"],
            )

        all_rows: list[dict[str, Any]] = []

        def fetch_one(asset_id: str, start_ts: int) -> tuple[str, list[dict[str, Any]], str | None]:
            try:
                history = clients.fetch_prices_history(
                    asset_id=asset_id,
                    interval=self.config.interval,
                    start_ts=start_ts,
                    end_ts=now_ts,
                )
                rows = self._normalize_history_rows(
                    asset_id,
                    history,
                    interval=self.config.interval,
                    ingested_at=ingested_at,
                )
                return asset_id, rows, None
            except Exception as exc:
                return asset_id, [], str(exc)

        worker_count = max(1, int(self.config.price_fetch_workers))
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(fetch_one, asset_id, fetch_start_ts): asset_id
                for asset_id, fetch_start_ts in fetch_plan
            }
            for future in as_completed(future_map):
                asset_id = future_map[future]
                try:
                    _, rows, error = future.result()
                except Exception as exc:
                    rows = []
                    error = str(exc)

                if error is not None:
                    self.metrics["price_history_errors"] = self.metrics.get("price_history_errors", 0) + 1
                    error_count = self.metrics["price_history_errors"]
                    if error_count <= 20 or error_count % 500 == 0:
                        LOGGER.warning("prices-history failed for asset %s: %s", asset_id, error)

                if not rows:
                    self.metrics["empty_histories"] = self.metrics.get("empty_histories", 0) + 1
                else:
                    all_rows.extend(rows)
                    if self.config.write_raw_price_files:
                        raw_df = pd.DataFrame(rows, columns=output_columns)
                        raw_path = self.config.raw_prices_dir / f"{asset_id}.parquet"
                        raw_df.to_parquet(raw_path, index=False)

                completed += 1
                if completed % 100 == 0 or completed == len(fetch_plan):
                    LOGGER.info("Price histories fetched: %s / %s", completed, len(fetch_plan))

        new_df = pd.DataFrame(all_rows, columns=output_columns)
        self._log_history_cadence_sanity(new_df)

        return upsert_parquet(
            price_path,
            new_df,
            dedupe_keys=["asset_id", "ts", "interval"],
            sort_keys=["asset_id", "ts"],
        )

    @staticmethod
    def _interval_to_seconds(interval: str) -> int | None:
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
        return mapping.get(str(interval).strip().lower())

    def _log_history_cadence_sanity(self, history_df: pd.DataFrame) -> None:
        if history_df.empty:
            return

        expected_interval_s = self._interval_to_seconds(self.config.interval)
        if expected_interval_s is None:
            return

        cadence = (
            history_df.groupby("asset_id", dropna=False)["ts"]
            .agg(min_ts="min", max_ts="max", points="count")
            .reset_index(drop=True)
        )
        cadence = cadence[cadence["points"] >= 12].copy()
        if cadence.empty:
            return

        denom = (cadence["points"] - 1).clip(lower=1)
        cadence["step_s"] = (cadence["max_ts"] - cadence["min_ts"]) / denom
        cadence = cadence[cadence["step_s"] > 0]
        if cadence.empty:
            return

        median_step_s = float(cadence["step_s"].median())
        if median_step_s < (expected_interval_s * 0.5):
            LOGGER.warning(
                "Observed history cadence looks faster than requested interval: requested=%s (~%ss) median_observed_step=%.1fs",
                self.config.interval,
                expected_interval_s,
                median_step_s,
            )

    @staticmethod
    def _normalize_history_rows(
        asset_id: str,
        history: list[dict[str, Any]],
        *,
        interval: str,
        ingested_at: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for point in history:
            if not isinstance(point, dict):
                continue
            ts_value = point.get("t", point.get("ts", point.get("timestamp")))
            p_value = point.get("p", point.get("price"))
            if ts_value is None or p_value is None:
                continue
            try:
                ts = int(float(ts_value))
                if ts > 10_000_000_000:
                    ts //= 1000
                price = float(p_value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "asset_id": str(asset_id),
                    "ts": ts,
                    "price": price,
                    "interval": interval,
                    "ingested_at": ingested_at,
                }
            )
        rows.sort(key=lambda row: row["ts"])
        deduped: dict[int, dict[str, Any]] = {row["ts"]: row for row in rows}
        return list(deduped.values())

    def _fetch_trades_sample(self, clients: PolymarketClients, markets_df: pd.DataFrame) -> None:
        condition_ids = (
            markets_df["condition_id"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(self.config.fetch_trades_sample)
            .tolist()
        )
        if not condition_ids:
            return

        LOGGER.info("Fetching trades sample for %s condition IDs", len(condition_ids))
        for condition_id in condition_ids:
            # Data API /trades expects condition_id in the `market` query parameter.
            trades = clients.fetch_trades_page(condition_id=condition_id, limit=10_000, offset=0)
            trades_df = pd.DataFrame(trades)
            raw_path = self.config.raw_trades_dir / f"{condition_id}.parquet"
            trades_df.to_parquet(raw_path, index=False)

    def _analyze(
        self,
        *,
        events_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        all_tokens_df: pd.DataFrame,
        target_tokens_df: pd.DataFrame,
        price_history_df: pd.DataFrame,
    ) -> tuple[Path, Path, Path]:
        LOGGER.info("Running feature extraction and clustering...")
        features_df, curves = compute_asset_features(
            price_history_df,
            interval=self.config.interval,
            window_days=self.config.window_days,
            gap_fill_limit=self.config.gap_fill_limit,
        )

        feature_market_df = target_tokens_df[["asset_id", "market_id"]].drop_duplicates().merge(
            features_df,
            on="asset_id",
            how="inner",
        )
        feature_market_df["asset_id"] = feature_market_df["asset_id"].astype(str)

        event_tags = events_df[["event_id", "tags"]].copy()
        event_tags["primary_tag"] = event_tags["tags"].apply(primary_tag_from_json)
        market_tags = markets_df[["market_id", "event_id"]].merge(
            event_tags[["event_id", "primary_tag"]],
            on="event_id",
            how="left",
        )
        market_tags["primary_tag"] = market_tags["primary_tag"].fillna("unknown")
        feature_market_df = feature_market_df.merge(
            market_tags[["market_id", "primary_tag"]],
            on="market_id",
            how="left",
        )
        feature_market_df["primary_tag"] = feature_market_df["primary_tag"].fillna("unknown")

        quality_df = build_market_quality_table(
            target_tokens_df=target_tokens_df,
            markets_df=markets_df,
            price_history_df=price_history_df,
            feature_df=features_df,
            min_points=self.config.quality_min_points,
            max_missing_ratio=self.config.quality_max_missing_ratio,
            min_price_range=self.config.quality_min_price_range,
            min_liquidity=self.config.quality_min_liquidity,
        )
        quality_path = self.config.output_dir / "market_quality.parquet"
        quality_df.to_parquet(quality_path, index=False)

        cluster_input_df = feature_market_df.merge(
            quality_df[["asset_id", "quality_pass"]].drop_duplicates(),
            on="asset_id",
            how="left",
        )
        cluster_input_df["quality_pass"] = cluster_input_df["quality_pass"].fillna(False).astype(bool)
        quality_cluster_input_df = cluster_input_df[cluster_input_df["quality_pass"]].copy()
        if len(quality_cluster_input_df) >= 3:
            cluster_base_df = quality_cluster_input_df[["asset_id", "market_id", "primary_tag", *FEATURE_COLUMNS]]
            LOGGER.info("Clustering with quality filter: %s assets", len(cluster_base_df))
        else:
            cluster_base_df = cluster_input_df[["asset_id", "market_id", "primary_tag", *FEATURE_COLUMNS]]
            LOGGER.info(
                "Quality filter left too few assets (%s). Falling back to all feature assets (%s).",
                len(quality_cluster_input_df),
                len(cluster_base_df),
            )

        clustered_df, silhouette = cluster_features(
            cluster_base_df,
            cluster_k=self.config.cluster_k,
            random_seed=self.config.random_seed,
        )

        clusters_path = self.config.output_dir / "clusters.parquet"
        clustered_df.to_parquet(clusters_path, index=False)

        cluster_descriptions = save_cluster_plots(
            clustered_df,
            curves,
            analysis_dir=self.config.analysis_dir,
        )
        bet_type_summary_df = build_bet_type_summary(
            events_df=events_df,
            markets_df=markets_df,
            tokens_df=target_tokens_df,
            price_history_df=price_history_df,
            feature_df=features_df,
            quality_df=quality_df,
        )
        save_bet_type_plot(bet_type_summary_df, analysis_dir=self.config.analysis_dir)

        report = build_report_payload(
            events_df=events_df,
            markets_df=markets_df,
            all_tokens_df=all_tokens_df,
            target_tokens_df=target_tokens_df,
            price_history_df=price_history_df,
            feature_df=features_df,
            quality_df=quality_df,
            clustered_df=clustered_df,
            silhouette=silhouette,
            bet_type_summary_df=bet_type_summary_df,
            cluster_descriptions=cluster_descriptions,
            cluster_input_assets=len(cluster_base_df),
            tag_rank_top_n=self.config.tag_rank_top_n,
        )
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["config"] = self._serializable_config()
        report_path = self.config.analysis_dir / "report.json"
        write_report_json(report, report_path)
        return report_path, clusters_path, quality_path

    def _serializable_config(self) -> dict[str, Any]:
        payload = asdict(self.config)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
