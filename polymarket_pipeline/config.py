from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    output_dir: Path = Path("data")
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
    yes_only_binary: bool = True
    fetch_trades_sample: int = 0
    cluster_k: int = 8
    gap_fill_limit: int = 6
    random_seed: int = 42
    quality_min_points: int = 24
    quality_max_missing_ratio: float = 0.8
    quality_min_price_range: float = 0.02
    quality_min_liquidity: float = 0.0
    tag_rank_top_n: int = 10
    price_fetch_workers: int = 16
    incremental_prices: bool = True
    incremental_mode: str = "tail"
    incremental_overlap_points: int = 2
    write_raw_price_files: bool = True
    http_pool_maxsize: int = 64

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
