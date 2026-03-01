from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymarket_pipeline.storage import upsert_parquet


def append_pipeline_run(path: Path, run_row: dict) -> pd.DataFrame:
    frame = pd.DataFrame([run_row])
    return upsert_parquet(
        path,
        frame,
        dedupe_keys=["run_id"],
        sort_keys=["run_ts"],
    )
