from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _to_unix_ts(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric > 10_000_000_000:
            return numeric // 1000
        return numeric
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.timestamp())


def _normalize_tags(raw_tags: Any) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [raw_tags]
    if not isinstance(raw_tags, list):
        return []

    tags: list[str] = []
    for item in raw_tags:
        if isinstance(item, str):
            tag = item.strip()
            if tag:
                tags.append(tag)
            continue
        if isinstance(item, dict):
            for key in ("label", "name", "slug", "id"):
                value = item.get(key)
                if value:
                    tags.append(str(value))
                    break
    return tags


def _first(value_map: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value_map and value_map[key] not in (None, ""):
            return value_map[key]
    return None


def _parse_list_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return []

    stripped = value.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        parts = [part.strip().strip('"').strip("'") for part in stripped.split(",")]
        return [part for part in parts if part]

    if isinstance(parsed, list):
        return parsed
    if parsed in (None, ""):
        return []
    return [parsed]


def _normalize_resolution_outcome(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    if text in {"yes", "true", "1", "winner_yes", "resolved_yes"}:
        return "Yes"
    if text in {"no", "false", "0", "winner_no", "resolved_no"}:
        return "No"
    if text in {"invalid", "void", "voided", "cancelled", "canceled"}:
        return "Invalid"
    if text in {"unresolved", "pending"}:
        return "Unresolved"
    return None


def _extract_market_resolution(market: dict[str, Any]) -> tuple[bool, str | None, int | None]:
    resolved_flag = _coerce_bool(_first(market, ("resolved", "isResolved", "is_resolved")))
    outcome_raw = _first(
        market,
        (
            "winningOutcome",
            "winning_outcome",
            "outcome",
            "resolutionOutcome",
            "resolution_outcome",
            "result",
        ),
    )
    resolution_outcome = _normalize_resolution_outcome(outcome_raw)
    resolution_ts = _to_unix_ts(
        _first(
            market,
            (
                "resolutionDate",
                "resolvedTime",
                "resolvedAt",
                "resolution_ts",
                "resolutionTs",
                "endDate",
            ),
        )
    )

    if resolved_flag is None:
        resolved_flag = resolution_outcome in {"Yes", "No", "Invalid"}
    if resolved_flag and resolution_outcome is None:
        resolution_outcome = "Unresolved"
    if not resolved_flag:
        resolution_outcome = resolution_outcome if resolution_outcome in {"Unresolved"} else None

    return bool(resolved_flag), resolution_outcome, resolution_ts


def _extract_market_tokens(market: dict[str, Any]) -> list[dict[str, str | None]]:
    extracted: list[dict[str, str | None]] = []

    tokens = _first(market, ("tokens",))
    if isinstance(tokens, list):
        for token in tokens:
            if not isinstance(token, dict):
                continue
            asset_id = _first(token, ("token_id", "tokenId", "asset_id", "assetId", "id"))
            if asset_id in (None, ""):
                continue
            outcome = _first(token, ("outcome", "name", "title"))
            extracted.append(
                {
                    "asset_id": str(asset_id),
                    "outcome": str(outcome) if outcome not in (None, "") else None,
                }
            )
    if extracted:
        return extracted

    outcomes = _parse_list_field(_first(market, ("outcomes", "outcomeNames", "answers")))
    clob_token_ids = _parse_list_field(_first(market, ("clobTokenIds", "clob_token_ids")))
    for idx, asset_id in enumerate(clob_token_ids):
        if asset_id in (None, ""):
            continue
        outcome = outcomes[idx] if idx < len(outcomes) else None
        extracted.append(
            {
                "asset_id": str(asset_id),
                "outcome": str(outcome) if outcome not in (None, "") else None,
            }
        )
    return extracted


def extract_tables(events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_cols = ["event_id", "slug", "title", "start_ts", "end_ts", "tags"]
    market_cols = [
        "market_id",
        "event_id",
        "condition_id",
        "question",
        "active",
        "closed",
        "resolved",
        "resolution_outcome",
        "resolution_ts",
        "market_type",
        "liquidity",
    ]
    token_cols = ["market_id", "outcome", "asset_id"]

    event_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for event in events:
        event_id = _first(event, ("id", "eventId", "event_id"))
        try:
            event_id = int(event_id) if event_id is not None else None
        except (TypeError, ValueError):
            event_id = None

        tags = _normalize_tags(_first(event, ("tags", "categories")))
        event_row = {
            "event_id": event_id,
            "slug": _first(event, ("slug",)),
            "title": _first(event, ("title", "name", "question")),
            "start_ts": _to_unix_ts(_first(event, ("startDate", "start_ts", "startTs", "startTime"))),
            "end_ts": _to_unix_ts(_first(event, ("endDate", "end_ts", "endTs", "endTime"))),
            "tags": json.dumps(tags, ensure_ascii=True),
        }
        event_rows.append(event_row)

        markets = _first(event, ("markets",)) or []
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            market_id = _first(market, ("id", "marketId", "market_id"))
            condition_id = _first(market, ("conditionId", "condition_id"))
            question = _first(market, ("question", "title", "name"))
            active = _coerce_bool(_first(market, ("active",)))
            closed = _coerce_bool(_first(market, ("closed", "isClosed")))
            resolved, resolution_outcome, resolution_ts = _extract_market_resolution(market)
            token_pairs = _extract_market_tokens(market)
            outcome_values = _parse_list_field(_first(market, ("outcomes", "outcomeNames", "answers")))
            market_type = _first(market, ("marketType", "type"))
            if market_type is None:
                token_count = len(token_pairs) if token_pairs else len(outcome_values)
                if token_count == 2:
                    market_type = "binary"
                elif token_count > 2:
                    market_type = "multi"
                else:
                    market_type = "unknown"
            market_rows.append(
                {
                    "market_id": str(market_id) if market_id is not None else None,
                    "event_id": event_id,
                    "condition_id": str(condition_id) if condition_id is not None else None,
                    "question": question,
                    "active": active,
                    "closed": closed,
                    "resolved": resolved,
                    "resolution_outcome": resolution_outcome,
                    "resolution_ts": resolution_ts,
                    "market_type": str(market_type).lower(),
                    "liquidity": _first(market, ("liquidity", "liquidityNum")),
                }
            )

            for token in token_pairs:
                asset_id = token["asset_id"]
                if asset_id in (None, ""):
                    continue
                token_rows.append(
                    {
                        "market_id": str(market_id) if market_id is not None else None,
                        "outcome": token["outcome"],
                        "asset_id": str(asset_id),
                    }
                )

    events_df = pd.DataFrame(event_rows, columns=event_cols)
    markets_df = pd.DataFrame(market_rows, columns=market_cols)
    tokens_df = pd.DataFrame(token_rows, columns=token_cols)

    if not events_df.empty:
        events_df = events_df.drop_duplicates(subset=["event_id"], keep="last")
    if not markets_df.empty:
        markets_df = markets_df.drop_duplicates(subset=["market_id"], keep="last")
    if not tokens_df.empty:
        tokens_df = tokens_df.drop_duplicates(subset=["market_id", "asset_id"], keep="last")

    return events_df, markets_df, tokens_df


def extract_resolutions(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["market_id", "condition_id", "resolution_outcome", "resolution_ts"]

    for event in events:
        markets = _first(event, ("markets",)) or []
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            market_id = _first(market, ("id", "marketId", "market_id"))
            condition_id = _first(market, ("conditionId", "condition_id"))
            resolved, resolution_outcome, resolution_ts = _extract_market_resolution(market)
            if not resolved and resolution_outcome is None and resolution_ts is None:
                continue
            rows.append(
                {
                    "market_id": str(market_id) if market_id is not None else None,
                    "condition_id": str(condition_id) if condition_id is not None else None,
                    "resolution_outcome": resolution_outcome,
                    "resolution_ts": resolution_ts,
                }
            )

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["market_id"], keep="last")


def primary_tag_from_json(tags_json: Any) -> str:
    if tags_json is None:
        return "unknown"
    if isinstance(tags_json, list) and tags_json:
        return str(tags_json[0])
    if isinstance(tags_json, dict):
        for key in ("label", "name", "slug", "id"):
            if tags_json.get(key):
                return str(tags_json[key])
        return "unknown"
    if not isinstance(tags_json, str):
        tags_json = str(tags_json)
    if not tags_json:
        return "unknown"
    try:
        values = json.loads(tags_json)
    except (json.JSONDecodeError, TypeError):
        stripped = tags_json.strip()
        if stripped and stripped != "nan":
            return stripped
        return "unknown"
    if isinstance(values, list) and values:
        first = values[0]
        if first:
            return str(first)
    return "unknown"


def select_price_tokens(
    tokens_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    *,
    yes_only_binary: bool,
) -> pd.DataFrame:
    if tokens_df.empty:
        return tokens_df.copy()

    working = tokens_df.merge(
        markets_df[["market_id", "market_type"]],
        on="market_id",
        how="left",
    )
    working["market_type"] = working["market_type"].fillna("unknown")
    working["outcome_norm"] = working["outcome"].fillna("").str.strip().str.lower()

    if not yes_only_binary:
        return working.drop(columns=["outcome_norm"])

    selected_rows: list[dict[str, Any]] = []
    for _, group in working.groupby("market_id", dropna=False):
        market_type = str(group["market_type"].iloc[0]).lower()
        if market_type == "binary":
            yes_rows = group[group["outcome_norm"].isin({"yes", "true"})]
            if yes_rows.empty:
                selected_rows.extend(group.iloc[[0]].to_dict(orient="records"))
            else:
                selected_rows.extend(yes_rows.iloc[[0]].to_dict(orient="records"))
        else:
            selected_rows.extend(group.to_dict(orient="records"))

    if not selected_rows:
        return pd.DataFrame(columns=working.columns).drop(columns=["outcome_norm"])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected = selected.drop(columns=["outcome_norm"])
    selected = selected.drop_duplicates(subset=["asset_id"], keep="first")
    return selected
