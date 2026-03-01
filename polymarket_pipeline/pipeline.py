from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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
    write_feature_metadata_json,
    write_report_json,
)
from .backtesting.engine import BacktestConfig, run_backtest
from .client import PolymarketClients
from .config import PipelineConfig
from .monitoring import append_pipeline_run, build_health_check
from .normalize import extract_resolutions, extract_tables, primary_tag_from_json, select_price_tokens
from .relationships import detect_market_relationships
from .signals_runner import generate_trade_candidates, run_signal_generation
from .storage import read_parquet_if_exists, upsert_parquet, write_jsonl

LOGGER = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.metrics: dict[str, int] = {
            "requests": 0,
            "retries": 0,
            "empty_histories": 0,
            "price_history_errors": 0,
            "http_429": 0,
        }

    def run(self) -> dict[str, Path]:
        started_at = time.time()
        self.config.ensure_dirs()
        clients = self._build_clients()
        outputs: dict[str, Path] = {}

        events = self._discover_active_events(clients)
        raw_events_path = self.config.raw_dir / f"events_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        write_jsonl(raw_events_path, events, mode="w")
        outputs["raw_events"] = raw_events_path

        resolved_events: list[dict[str, Any]] = []
        if self.config.include_resolved:
            resolved_events = self._discover_resolved_events(clients, lookback_days=self.config.resolved_lookback_days)
            raw_resolved_path = self.config.raw_dir / f"resolved_events_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
            write_jsonl(raw_resolved_path, resolved_events, mode="w")
            outputs["raw_resolved_events"] = raw_resolved_path

        events_new, markets_new, tokens_new = extract_tables(events)
        if resolved_events:
            res_events_new, res_markets_new, res_tokens_new = extract_tables(resolved_events)
            events_new = pd.concat([events_new, res_events_new], ignore_index=True)
            markets_new = pd.concat([markets_new, res_markets_new], ignore_index=True)
            tokens_new = pd.concat([tokens_new, res_tokens_new], ignore_index=True)

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
        outputs["events"] = events_path
        outputs["markets"] = markets_path
        outputs["tokens"] = tokens_path

        target_tokens_df = select_price_tokens(tokens_df, markets_df, yes_only_binary=self.config.yes_only_binary)
        price_history_df = self._ingest_price_history(clients, target_tokens_df)
        outputs["price_history"] = self.config.output_dir / "price_history.parquet"

        resolutions_df = pd.DataFrame()
        if self.config.include_resolved or (self.config.output_dir / "resolutions.parquet").exists():
            resolutions_df = self._ingest_resolutions(
                resolved_events=resolved_events,
                markets_df=markets_df,
                tokens_df=tokens_df,
                price_history_df=price_history_df,
                markets_path=markets_path,
            )
            if not resolutions_df.empty:
                outputs["resolutions"] = self.config.output_dir / "resolutions.parquet"

        orderbook_df = pd.DataFrame()
        if self.config.snapshot_orderbook:
            orderbook_df = self._snapshot_orderbook(clients, target_tokens_df)
            outputs["orderbook_snapshots"] = self.config.output_dir / "orderbook_snapshots.parquet"
        else:
            orderbook_df = read_parquet_if_exists(self.config.output_dir / "orderbook_snapshots.parquet")

        volume_bars_df = pd.DataFrame()
        if self.config.ingest_volume:
            volume_bars_df = self._ingest_volume_bars(clients, markets_df, tokens_df)
            outputs["volume_bars"] = self.config.output_dir / "volume_bars.parquet"
        else:
            volume_bars_df = read_parquet_if_exists(self.config.output_dir / "volume_bars.parquet")

        if self.config.fetch_trades_sample > 0:
            self._fetch_trades_sample(clients, markets_df)

        (
            report_path,
            clusters_path,
            quality_path,
            features_path,
            features_df,
            report_payload,
        ) = self._analyze(
            events_df=events_df,
            markets_df=markets_df,
            all_tokens_df=tokens_df,
            target_tokens_df=target_tokens_df,
            price_history_df=price_history_df,
            orderbook_df=orderbook_df,
            volume_bars_df=volume_bars_df,
        )
        outputs["clusters"] = clusters_path
        outputs["market_quality"] = quality_path
        outputs["features"] = features_path
        outputs["report"] = report_path
        outputs["feature_metadata"] = self.config.analysis_dir / "feature_metadata.json"

        if self.config.detect_relationships:
            rel_df = detect_market_relationships(
                events_df=events_df,
                markets_df=markets_df,
                tokens_df=tokens_df,
                price_history_df=price_history_df,
                correlation_threshold=self.config.correlation_threshold,
                min_overlap_days=self.config.min_overlap_days,
            )
            relationships_path = self.config.output_dir / "market_relationships.parquet"
            rel_df.to_parquet(relationships_path, index=False)
            outputs["market_relationships"] = relationships_path

        signals_df = pd.DataFrame()
        signal_registry: dict[str, Any] = {}
        if self.config.run_signals or self.config.run_backtest or self.config.generate_candidates:
            signals_new, signal_registry = run_signal_generation(
                config=self.config,
                features_df=features_df,
                target_tokens_df=target_tokens_df,
                markets_df=markets_df,
                events_df=events_df,
                price_history_df=price_history_df,
                volume_bars_df=volume_bars_df,
                resolutions_df=resolutions_df,
            )
            signals_path = self.config.output_dir / "signals.parquet"
            signals_df = upsert_parquet(
                signals_path,
                signals_new,
                dedupe_keys=["signal_name", "asset_id", "market_id", "ts"],
                sort_keys=["ts", "signal_name", "asset_id"],
            )
            outputs["signals"] = signals_path

        if self.config.run_backtest:
            backtest_results_df, backtest_summary = run_backtest(
                signals_df=signals_df,
                resolutions_df=resolutions_df,
                orderbook_df=orderbook_df,
                config=BacktestConfig(
                    start_date=self.config.backtest_start_date,
                    end_date=self.config.backtest_end_date,
                    initial_capital=self.config.backtest_initial_capital,
                    spread_assumption=self.config.backtest_spread_assumption,
                    max_positions=self.config.backtest_max_positions,
                    stop_loss=self.config.backtest_stop_loss,
                    timeout_days=self.config.backtest_timeout_days,
                    sizing_mode=self.config.sizing_mode,
                    kelly_fraction=self.config.kelly_fraction,
                    max_position_pct=self.config.max_position_pct,
                    min_position_size=self.config.min_position_size,
                    flat_position_size=self.config.flat_position_size,
                ),
            )
            backtest_path = self.config.output_dir / "backtest_results.parquet"
            backtest_results_df.to_parquet(backtest_path, index=False)
            summary_path = self.config.analysis_dir / "backtest_summary.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(backtest_summary, handle, indent=2, ensure_ascii=True)
            outputs["backtest_results"] = backtest_path
            outputs["backtest_summary"] = summary_path

        if self.config.generate_candidates:
            quality_df = read_parquet_if_exists(quality_path)
            clusters_df = read_parquet_if_exists(clusters_path)
            candidates_payload, candidates_df = generate_trade_candidates(
                config=self.config,
                signals_df=signals_df,
                signal_registry=signal_registry,
                markets_df=markets_df,
                features_df=features_df,
                orderbook_df=orderbook_df,
                quality_df=quality_df,
                clusters_df=clusters_df,
                bankroll=self.config.backtest_initial_capital,
            )
            candidates_json_path = self.config.analysis_dir / "trade_candidates.json"
            with candidates_json_path.open("w", encoding="utf-8") as handle:
                json.dump(candidates_payload, handle, indent=2, ensure_ascii=True)
            candidates_path = self.config.output_dir / "trade_candidates.parquet"
            candidates_df = upsert_parquet(
                candidates_path,
                candidates_df,
                dedupe_keys=["run_date", "market_id", "signal_name", "asset_id"],
                sort_keys=["run_date", "rank"],
            )
            outputs["trade_candidates_json"] = candidates_json_path
            outputs["trade_candidates"] = candidates_path

        runtime_s = float(time.time() - started_at)
        coverage = report_payload.get("coverage", {}) if isinstance(report_payload, dict) else {}
        health_payload, run_row = build_health_check(
            output_dir=self.config.output_dir,
            config_payload=self._serializable_config(),
            runtime_s=runtime_s,
            coverage=coverage,
            metrics=self.metrics,
            events_count=int(len(events_df)),
            markets_count=int(len(markets_df)),
            signals_generated=int(len(signals_df)),
            include_resolved=bool(self.config.include_resolved),
        )
        health_path = self.config.analysis_dir / "health_check.json"
        with health_path.open("w", encoding="utf-8") as handle:
            json.dump(health_payload, handle, indent=2, ensure_ascii=True)
        append_pipeline_run(self.config.output_dir / "pipeline_runs.parquet", run_row)
        outputs["health_check"] = health_path
        outputs["pipeline_runs"] = self.config.output_dir / "pipeline_runs.parquet"

        LOGGER.info(
            "Pipeline done. requests=%s retries=%s empty_histories=%s price_history_errors=%s skipped_price_assets=%s",
            self.metrics.get("requests", 0),
            self.metrics.get("retries", 0),
            self.metrics.get("empty_histories", 0),
            self.metrics.get("price_history_errors", 0),
            self.metrics.get("skipped_price_assets", 0),
        )
        return outputs

    def run_resolutions_only(self) -> dict[str, Path]:
        self.config.ensure_dirs()
        clients = self._build_clients()
        outputs: dict[str, Path] = {}

        resolved_events = self._discover_resolved_events(clients, lookback_days=self.config.resolved_lookback_days)
        raw_path = self.config.raw_dir / f"resolved_events_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
        write_jsonl(raw_path, resolved_events, mode="w")
        outputs["raw_resolved_events"] = raw_path

        events_new, markets_new, tokens_new = extract_tables(resolved_events)
        events_path = self.config.output_dir / "events.parquet"
        markets_path = self.config.output_dir / "markets.parquet"
        tokens_path = self.config.output_dir / "tokens.parquet"
        events_df = upsert_parquet(events_path, events_new, dedupe_keys=["event_id"], sort_keys=["event_id"])
        markets_df = upsert_parquet(markets_path, markets_new, dedupe_keys=["market_id"], sort_keys=["market_id"])
        tokens_df = upsert_parquet(tokens_path, tokens_new, dedupe_keys=["market_id", "asset_id"], sort_keys=["market_id", "asset_id"])

        price_history_df = read_parquet_if_exists(self.config.output_dir / "price_history.parquet")
        resolutions_df = self._ingest_resolutions(
            resolved_events=resolved_events,
            markets_df=markets_df,
            tokens_df=tokens_df,
            price_history_df=price_history_df,
            markets_path=markets_path,
        )
        outputs["events"] = events_path
        outputs["markets"] = markets_path
        outputs["tokens"] = tokens_path
        if not resolutions_df.empty:
            outputs["resolutions"] = self.config.output_dir / "resolutions.parquet"
        return outputs

    def _build_clients(self) -> PolymarketClients:
        return PolymarketClients(
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

    def _discover_active_events(self, clients: PolymarketClients) -> list[dict[str, Any]]:
        return self._discover_events_generic(
            clients=clients,
            active=True,
            closed=False,
            order="volume24hr",
            max_events=max(0, int(self.config.max_events)),
            cache_subdir="gamma_events",
            cutoff_ts=None,
        )

    def _discover_resolved_events(self, clients: PolymarketClients, *, lookback_days: int) -> list[dict[str, Any]]:
        cutoff_ts = int(time.time()) - int(max(1, lookback_days)) * 24 * 3600
        resolved_target = max(int(self.config.max_events), 1000)
        events = self._discover_events_generic(
            clients=clients,
            active=False,
            closed=True,
            order="endDate",
            max_events=resolved_target,
            cache_subdir="gamma_events_resolved",
            cutoff_ts=cutoff_ts,
        )
        return events

    def _discover_events_generic(
        self,
        *,
        clients: PolymarketClients,
        active: bool,
        closed: bool,
        order: str | None,
        max_events: int,
        cache_subdir: str,
        cutoff_ts: int | None,
    ) -> list[dict[str, Any]]:
        if max_events <= 0:
            return []

        mode = "resolved" if closed else "active"
        LOGGER.info("Discovering %s events from Gamma...", mode)
        events: list[dict[str, Any]] = []
        offset = 0
        max_scan = max_events * 20

        while len(events) < max_events and offset <= max_scan:
            limit = min(self.config.page_limit, max_events - len(events))
            page = clients.fetch_events_page(
                limit=limit,
                offset=offset,
                active=active,
                closed=closed,
                order=order,
                cache_dir=self.config.cache_dir / cache_subdir,
                cache_ttl_s=self.config.gamma_cache_ttl_s,
            )
            if not page:
                break

            if cutoff_ts is not None:
                filtered: list[dict[str, Any]] = []
                for event in page:
                    event_ts = self._event_timestamp(event)
                    if event_ts is None or event_ts >= int(cutoff_ts):
                        filtered.append(event)
                page = filtered
                if not page and len(filtered) == 0:
                    offset += limit
                    continue

            events.extend(page)
            LOGGER.info("Fetched %s %s events so far", len(events), mode)
            if len(page) < limit:
                break
            offset += limit

        return events[:max_events]

    @staticmethod
    def _event_timestamp(event: dict[str, Any]) -> int | None:
        for key in ("resolutionDate", "resolvedAt", "endDate", "end_ts", "endTs"):
            if key not in event:
                continue
            value = event.get(key)
            try:
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    ts = int(value)
                else:
                    ts = int(pd.to_datetime(value, utc=True, errors="coerce").timestamp())
                if ts > 10_000_000_000:
                    ts //= 1000
                return ts
            except Exception:
                continue
        return None

    def _ingest_resolutions(
        self,
        *,
        resolved_events: list[dict[str, Any]],
        markets_df: pd.DataFrame,
        tokens_df: pd.DataFrame,
        price_history_df: pd.DataFrame,
        markets_path: Path,
    ) -> pd.DataFrame:
        resolutions_path = self.config.output_dir / "resolutions.parquet"
        base = extract_resolutions(resolved_events) if resolved_events else pd.DataFrame(
            columns=["market_id", "condition_id", "resolution_outcome", "resolution_ts"]
        )
        if base.empty and resolutions_path.exists():
            return pd.read_parquet(resolutions_path)

        ingested_at = int(time.time())
        base["market_id"] = base["market_id"].astype(str)

        token_map = tokens_df[["market_id", "asset_id", "outcome"]].copy() if not tokens_df.empty else pd.DataFrame(
            columns=["market_id", "asset_id", "outcome"]
        )
        if not token_map.empty:
            token_map["market_id"] = token_map["market_id"].astype(str)
            token_map["asset_id"] = token_map["asset_id"].astype(str)
            token_map["outcome_norm"] = token_map["outcome"].fillna("").astype(str).str.strip().str.lower()
            token_map["anchor"] = token_map["outcome_norm"].isin({"yes", "true"})
            token_map = token_map.sort_values(["market_id", "anchor"], ascending=[True, False])
            token_map = token_map.groupby("market_id", as_index=False).first()[["market_id", "asset_id"]]

        base = base.merge(token_map, on="market_id", how="left")
        base["final_price"] = pd.NA

        if not price_history_df.empty:
            ph = price_history_df[["asset_id", "ts", "price"]].copy()
            ph["asset_id"] = ph["asset_id"].astype(str)
            ph["ts"] = pd.to_numeric(ph["ts"], errors="coerce")
            ph["price"] = pd.to_numeric(ph["price"], errors="coerce")
            ph = ph.dropna(subset=["asset_id", "ts", "price"]).sort_values(["asset_id", "ts"])
            per_asset = {asset_id: frame[["ts", "price"]] for asset_id, frame in ph.groupby("asset_id", dropna=False)}

            final_prices: list[float | pd._libs.missing.NAType] = []
            for _, row in base.iterrows():
                asset_id = row.get("asset_id")
                resolution_ts = row.get("resolution_ts")
                frame = per_asset.get(str(asset_id)) if asset_id is not None else None
                if frame is None or frame.empty:
                    final_prices.append(pd.NA)
                    continue
                if pd.notna(resolution_ts):
                    candidates = frame[frame["ts"] <= float(resolution_ts)]
                    if candidates.empty:
                        final_prices.append(float(frame["price"].iloc[-1]))
                    else:
                        final_prices.append(float(candidates["price"].iloc[-1]))
                else:
                    final_prices.append(float(frame["price"].iloc[-1]))
            base["final_price"] = final_prices

        base["ingested_at"] = ingested_at
        resolutions_df = upsert_parquet(
            resolutions_path,
            base[["market_id", "condition_id", "resolution_outcome", "resolution_ts", "final_price", "ingested_at"]],
            dedupe_keys=["market_id"],
            sort_keys=["market_id"],
        )

        if not markets_df.empty:
            market_update = resolutions_df[["market_id", "resolution_outcome", "resolution_ts"]].copy()
            market_update["resolved"] = market_update["resolution_outcome"].isin(["Yes", "No", "Invalid", "Unresolved"])
            merged = markets_df.drop(columns=[col for col in ("resolved", "resolution_outcome", "resolution_ts") if col in markets_df.columns]).merge(
                market_update,
                on="market_id",
                how="left",
            )
            if "resolved" not in merged.columns:
                merged["resolved"] = False
            merged["resolved"] = merged["resolved"].fillna(False).astype(bool)
            merged.to_parquet(markets_path, index=False)
        return resolutions_df

    def _snapshot_orderbook(self, clients: PolymarketClients, target_tokens_df: pd.DataFrame) -> pd.DataFrame:
        output_columns = [
            "asset_id",
            "snapshot_ts",
            "best_bid",
            "best_ask",
            "bid_size",
            "ask_size",
            "mid_price",
            "spread",
            "spread_pct",
            "bid_depth_5pct",
            "ask_depth_5pct",
        ]
        path = self.config.output_dir / "orderbook_snapshots.parquet"
        if target_tokens_df.empty:
            return upsert_parquet(path, pd.DataFrame(columns=output_columns), dedupe_keys=["asset_id", "snapshot_ts"])

        asset_ids = target_tokens_df["asset_id"].dropna().astype(str).drop_duplicates().tolist()
        snapshot_ts = int(time.time() // 3600 * 3600)
        rows: list[dict[str, Any]] = []

        def fetch_one(asset_id: str) -> dict[str, Any]:
            book = clients.fetch_order_book(asset_id=asset_id)
            metrics = self._compute_orderbook_metrics(book.get("bids", []), book.get("asks", []))
            metrics["asset_id"] = str(asset_id)
            metrics["snapshot_ts"] = snapshot_ts
            return metrics

        workers = max(1, int(self.config.orderbook_workers))
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(fetch_one, asset_id): asset_id for asset_id in asset_ids}
            for future in as_completed(future_map):
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "asset_id": future_map[future],
                        "snapshot_ts": snapshot_ts,
                        "best_bid": pd.NA,
                        "best_ask": pd.NA,
                        "bid_size": pd.NA,
                        "ask_size": pd.NA,
                        "mid_price": pd.NA,
                        "spread": pd.NA,
                        "spread_pct": pd.NA,
                        "bid_depth_5pct": pd.NA,
                        "ask_depth_5pct": pd.NA,
                    }
                    LOGGER.warning("orderbook snapshot failed for asset %s: %s", future_map[future], exc)
                rows.append(row)
                completed += 1
                if completed % 200 == 0 or completed == len(asset_ids):
                    LOGGER.info("Orderbook snapshots fetched: %s / %s", completed, len(asset_ids))

        return upsert_parquet(
            path,
            pd.DataFrame(rows, columns=output_columns),
            dedupe_keys=["asset_id", "snapshot_ts"],
            sort_keys=["snapshot_ts", "asset_id"],
        )

    @staticmethod
    def _compute_orderbook_metrics(bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> dict[str, Any]:
        best_bid = pd.NA
        best_ask = pd.NA
        bid_size = pd.NA
        ask_size = pd.NA
        mid_price = pd.NA
        spread = pd.NA
        spread_pct = pd.NA
        bid_depth = pd.NA
        ask_depth = pd.NA

        bids_sorted = sorted([(float(p), float(s)) for p, s in bids if p > 0 and s >= 0], key=lambda x: x[0], reverse=True)
        asks_sorted = sorted([(float(p), float(s)) for p, s in asks if p > 0 and s >= 0], key=lambda x: x[0])
        if bids_sorted:
            best_bid, bid_size = bids_sorted[0]
        if asks_sorted:
            best_ask, ask_size = asks_sorted[0]
        if bids_sorted and asks_sorted and best_ask >= best_bid:
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            spread_pct = spread / mid_price if mid_price > 0 else pd.NA
            lower = mid_price * 0.95
            upper = mid_price * 1.05
            bid_depth = float(sum(size for price, size in bids_sorted if price >= lower))
            ask_depth = float(sum(size for price, size in asks_sorted if price <= upper))

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "mid_price": mid_price,
            "spread": spread,
            "spread_pct": spread_pct,
            "bid_depth_5pct": bid_depth,
            "ask_depth_5pct": ask_depth,
        }

    def _ingest_volume_bars(self, clients: PolymarketClients, markets_df: pd.DataFrame, tokens_df: pd.DataFrame) -> pd.DataFrame:
        output_columns = ["asset_id", "ts", "interval", "volume", "trade_count", "buy_volume", "sell_volume", "vwap"]
        path = self.config.output_dir / "volume_bars.parquet"
        if markets_df.empty or tokens_df.empty:
            return upsert_parquet(path, pd.DataFrame(columns=output_columns), dedupe_keys=["asset_id", "ts", "interval"])

        market_tokens = tokens_df[["market_id", "asset_id", "outcome"]].copy()
        market_tokens["market_id"] = market_tokens["market_id"].astype(str)
        market_tokens["asset_id"] = market_tokens["asset_id"].astype(str)
        market_tokens["outcome_norm"] = market_tokens["outcome"].fillna("").astype(str).str.strip().str.lower()
        market_tokens["anchor"] = market_tokens["outcome_norm"].isin({"yes", "true"})
        market_tokens = market_tokens.sort_values(["market_id", "anchor"], ascending=[True, False])
        anchor_tokens = market_tokens.groupby("market_id", as_index=False).first()[["market_id", "asset_id"]]

        condition_map = markets_df[["market_id", "condition_id"]].copy()
        condition_map["market_id"] = condition_map["market_id"].astype(str)
        condition_map["condition_id"] = condition_map["condition_id"].astype(str)
        condition_map = condition_map.dropna(subset=["condition_id"]).drop_duplicates("market_id")
        condition_map = condition_map.merge(anchor_tokens, on="market_id", how="left").dropna(subset=["asset_id"])
        if condition_map.empty:
            return upsert_parquet(path, pd.DataFrame(columns=output_columns), dedupe_keys=["asset_id", "ts", "interval"])

        existing_latest: dict[str, int] = {}
        if self.config.incremental_prices and path.exists():
            existing = pd.read_parquet(path, columns=["asset_id", "interval", "ts"])
            existing["asset_id"] = existing["asset_id"].astype(str)
            existing["interval"] = existing["interval"].fillna("").astype(str)
            existing = existing[existing["interval"] == str(self.config.interval)]
            if not existing.empty:
                existing["ts"] = pd.to_numeric(existing["ts"], errors="coerce")
                existing = existing.dropna(subset=["ts"])
                existing_latest = existing.groupby("asset_id")["ts"].max().astype(int).to_dict()

        interval_s = self._interval_to_seconds(self.config.interval) or 3600
        now_ts = int(time.time())
        global_start_ts = now_ts - int(self.config.window_days) * 24 * 3600

        condition_tasks: list[tuple[str, str, int]] = []
        for _, row in condition_map.iterrows():
            condition_id = str(row["condition_id"])
            anchor_asset = str(row["asset_id"])
            if not self.config.incremental_prices:
                start_ts = global_start_ts
            elif self.config.incremental_mode == "skip" and anchor_asset in existing_latest:
                continue
            elif anchor_asset in existing_latest:
                start_ts = max(global_start_ts, int(existing_latest[anchor_asset]) - int(self.config.incremental_overlap_points) * interval_s)
            else:
                start_ts = global_start_ts
            condition_tasks.append((condition_id, anchor_asset, int(start_ts)))

        rows: list[dict[str, Any]] = []

        def fetch_condition(condition_id: str, anchor_asset: str, min_ts: int) -> list[dict[str, Any]]:
            trades = clients.fetch_trades_all(condition_id=condition_id, limit=10_000, max_pages=50, min_ts=min_ts)
            return self._normalize_trades_to_bars(
                trades=trades,
                asset_id=anchor_asset,
                interval_s=interval_s,
                interval_label=self.config.interval,
                min_ts=min_ts,
            )

        workers = max(1, int(self.config.volume_fetch_workers))
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(fetch_condition, condition_id, anchor_asset, start_ts): (condition_id, anchor_asset)
                for condition_id, anchor_asset, start_ts in condition_tasks
            }
            for future in as_completed(future_map):
                condition_id, _ = future_map[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    LOGGER.warning("Volume ingest failed for condition %s: %s", condition_id, exc)
                completed += 1
                if completed % 100 == 0 or completed == len(condition_tasks):
                    LOGGER.info("Volume histories fetched: %s / %s", completed, len(condition_tasks))

        new_df = pd.DataFrame(rows, columns=output_columns)
        return upsert_parquet(
            path,
            new_df,
            dedupe_keys=["asset_id", "ts", "interval"],
            sort_keys=["asset_id", "ts"],
        )

    @staticmethod
    def _normalize_trade_ts(row: dict[str, Any]) -> int | None:
        for key in ("timestamp", "ts", "time", "createdAt"):
            if key not in row:
                continue
            value = row.get(key)
            try:
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    ts = int(float(value))
                else:
                    ts = int(pd.to_datetime(value, utc=True, errors="coerce").timestamp())
                if ts > 10_000_000_000:
                    ts //= 1000
                return ts
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_trades_to_bars(
        *,
        trades: list[dict[str, Any]],
        asset_id: str,
        interval_s: int,
        interval_label: str,
        min_ts: int,
    ) -> list[dict[str, Any]]:
        agg: dict[int, dict[str, Any]] = {}
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            ts = PipelineRunner._normalize_trade_ts(trade)
            if ts is None or ts < int(min_ts):
                continue
            price = trade.get("price", trade.get("p"))
            size = trade.get("size", trade.get("amount", trade.get("volume", trade.get("shares", 0.0))))
            side = str(trade.get("side", "")).strip().lower()
            try:
                p = float(price)
                v = float(size)
            except (TypeError, ValueError):
                continue
            if not (v > 0 and p > 0):
                continue
            bucket_ts = int(ts - (ts % int(interval_s)))
            slot = agg.setdefault(
                bucket_ts,
                {
                    "asset_id": str(asset_id),
                    "ts": bucket_ts,
                    "interval": interval_label,
                    "volume": 0.0,
                    "trade_count": 0,
                    "buy_volume": 0.0,
                    "sell_volume": 0.0,
                    "_vwap_num": 0.0,
                },
            )
            slot["volume"] += v
            slot["trade_count"] += 1
            slot["_vwap_num"] += p * v
            if side == "buy":
                slot["buy_volume"] += v
            elif side == "sell":
                slot["sell_volume"] += v

        rows: list[dict[str, Any]] = []
        for bucket_ts, slot in sorted(agg.items()):
            volume = float(slot["volume"])
            vwap = float(slot["_vwap_num"] / volume) if volume > 0 else pd.NA
            rows.append(
                {
                    "asset_id": slot["asset_id"],
                    "ts": int(bucket_ts),
                    "interval": slot["interval"],
                    "volume": volume,
                    "trade_count": int(slot["trade_count"]),
                    "buy_volume": float(slot["buy_volume"]),
                    "sell_volume": float(slot["sell_volume"]),
                    "vwap": vwap,
                }
            )
        return rows

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
                        existing_latest_ts = existing_df.groupby("asset_id", as_index=True)["ts"].max().astype(int).to_dict()

        incremental_mode = str(self.config.incremental_mode).strip().lower()
        if incremental_mode not in {"tail", "skip"}:
            incremental_mode = "tail"

        fetch_plan: list[tuple[str, int]] = []
        if not self.config.incremental_prices:
            fetch_plan = [(asset_id, global_start_ts) for asset_id in asset_ids]
            self.metrics["skipped_price_assets"] = 0
            LOGGER.info("Incremental disabled; fetching prices-history for %s assets", len(fetch_plan))
        elif incremental_mode == "skip":
            fetch_plan = [(asset_id, global_start_ts) for asset_id in asset_ids if asset_id not in existing_latest_ts]
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
        orderbook_df: pd.DataFrame,
        volume_bars_df: pd.DataFrame,
    ) -> tuple[Path, Path, Path, Path, pd.DataFrame, dict[str, Any]]:
        LOGGER.info("Running feature extraction and clustering...")

        market_context_df = target_tokens_df[["asset_id", "market_id"]].drop_duplicates().merge(
            markets_df[["market_id", "event_id"]],
            on="market_id",
            how="left",
        )
        if not events_df.empty:
            event_time = events_df[["event_id", "start_ts", "end_ts"]].copy()
            market_context_df = market_context_df.merge(event_time, on="event_id", how="left")

        features_df, curves = compute_asset_features(
            price_history_df,
            interval=self.config.interval,
            window_days=self.config.window_days,
            gap_fill_limit=self.config.gap_fill_limit,
            market_context_df=market_context_df,
            orderbook_df=orderbook_df,
            volume_bars_df=volume_bars_df,
        )
        features_path = self.config.output_dir / "features.parquet"
        features_df.to_parquet(features_path, index=False)
        write_feature_metadata_json(self.config.analysis_dir / "feature_metadata.json")

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
        return report_path, clusters_path, quality_path, features_path, features_df, report

    def _serializable_config(self) -> dict[str, Any]:
        payload = asdict(self.config)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
