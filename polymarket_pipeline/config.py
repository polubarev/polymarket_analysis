from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    output_dir: Path = Path("data/dev")
    pipeline_profile: str = "default"
    max_events: int = 2000
    page_limit: int = 100
    window_days: int = 90
    interval: str = "1h"
    request_timeout_s: float = 20.0
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_jitter_s: float = 0.5
    gamma_cache_ttl_s: int = 24 * 60 * 60
    gamma_rate_limit: int = 500
    clob_rate_limit: int = 1000
    data_rate_limit: int = 200
    rate_window_s: int = 10
    yes_only_binary: bool = False
    fetch_trades_sample: int = 0
    cluster_k: int = 8
    gap_fill_limit: int = 6
    random_seed: int = 42
    quality_min_points: int = 24
    quality_max_missing_ratio: float = 0.8
    quality_min_price_range: float = 0.02
    quality_min_liquidity: float = 0.0
    tag_rank_top_n: int = 10
    price_fetch_workers: int = 5
    fetch_priority_mode: str = "history_first"
    incremental_prices: bool = True
    incremental_mode: str = "tail"
    incremental_overlap_points: int = 2
    skip_inactive_priced_assets: bool = True
    write_raw_price_files: bool = True
    http_pool_maxsize: int = 64

    # Task 1: Resolutions
    include_resolved: bool = False
    resolved_lookback_days: int = 365

    # Task 2: Order Book
    snapshot_orderbook: bool = False
    orderbook_workers: int = 8

    # Task 3: Volume
    ingest_volume: bool = False
    volume_fetch_workers: int = 8

    # Task 5: Signals
    run_signals: bool = False
    active_signals: list[str] = field(default_factory=lambda: ["all"])
    signal_debug: bool = False
    signal_debug_limit: int = 20
    calibration_threshold: float = 0.15
    spike_zscore_threshold: float = 2.5
    convergence_days_threshold: int = 2
    breakout_lookback_days: int = 30
    imbalance_ratio_threshold: float = 3.0
    volume_surge_threshold: float = 2.0

    # Task 6: Backtesting
    run_backtest: bool = False
    backtest_start_date: str | None = None
    backtest_end_date: str | None = None
    backtest_initial_capital: float = 10_000.0
    backtest_spread_assumption: float = 0.03
    backtest_max_positions: int = 50
    backtest_stop_loss: float = 0.50
    backtest_timeout_days: int = 90

    # Task 7: Kelly sizing
    sizing_mode: str = "flat"
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    min_position_size: float = 1.0
    flat_position_size: float = 100.0

    # Task 8: Candidates
    generate_candidates: bool = False
    min_confidence: float = 0.55
    min_edge: float = 0.05
    max_candidates: int = 20

    # Task 9: Dashboard
    ui_mode: str = "discovery"

    # Task 10: Relationships
    detect_relationships: bool = False
    correlation_threshold: float = 0.5
    min_overlap_days: int = 30

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def raw_prices_dir(self) -> Path:
        return self.raw_dir / "prices"

    @property
    def raw_trades_dir(self) -> Path:
        return self.raw_dir / "trades"

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / ".cache"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.raw_prices_dir.mkdir(parents=True, exist_ok=True)
        self.raw_trades_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
