from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def write_jsonl(path: Path, records: Iterable[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def upsert_parquet(
    path: Path,
    new_df: pd.DataFrame,
    *,
    dedupe_keys: list[str] | None = None,
    sort_keys: list[str] | None = None,
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_df.empty and path.exists():
        return pd.read_parquet(path)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if dedupe_keys:
        available = [key for key in dedupe_keys if key in combined.columns]
        if available:
            combined = combined.drop_duplicates(subset=available, keep="last")

    if sort_keys:
        available = [key for key in sort_keys if key in combined.columns]
        if available:
            combined = combined.sort_values(available).reset_index(drop=True)

    combined.to_parquet(path, index=False)
    return combined
