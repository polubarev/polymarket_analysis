#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  echo "Loaded environment from ${ENV_FILE}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/polymarket-data}"
PIPELINE_PROFILE="${PIPELINE_PROFILE:-default}"
MAX_EVENTS="${MAX_EVENTS:-300}"
WINDOW_DAYS="${WINDOW_DAYS:-30}"
INTERVAL="${INTERVAL:-1h}"
PRICE_FETCH_WORKERS="${PRICE_FETCH_WORKERS:-2}"
RATE_WINDOW_S="${RATE_WINDOW_S:-10}"
FETCH_PRIORITY_MODE="${FETCH_PRIORITY_MODE:-history_first}"
HTTP_POOL_MAXSIZE="${HTTP_POOL_MAXSIZE:-64}"
QUALITY_MIN_POINTS="${QUALITY_MIN_POINTS:-24}"
QUALITY_MAX_MISSING_RATIO="${QUALITY_MAX_MISSING_RATIO:-0.9}"
QUALITY_MIN_PRICE_RANGE="${QUALITY_MIN_PRICE_RANGE:-0.005}"
INCREMENTAL_MODE="${INCREMENTAL_MODE:-tail}"
INCREMENTAL_OVERLAP_POINTS="${INCREMENTAL_OVERLAP_POINTS:-2}"
YES_ONLY_BINARY="${YES_ONLY_BINARY:-false}"
SKIP_INACTIVE_PRICED_ASSETS="${SKIP_INACTIVE_PRICED_ASSETS:-true}"
SKIP_RAW_PRICE_FILES="${SKIP_RAW_PRICE_FILES:-true}"
NO_INCREMENTAL_PRICES="${NO_INCREMENTAL_PRICES:-false}"
SNAPSHOT_ORDERBOOK="${SNAPSHOT_ORDERBOOK:-false}"
INGEST_VOLUME="${INGEST_VOLUME:-false}"
RUN_SIGNALS="${RUN_SIGNALS:-false}"
RUN_BACKTEST="${RUN_BACKTEST:-false}"
GENERATE_CANDIDATES="${GENERATE_CANDIDATES:-false}"
INCLUDE_RESOLVED="${INCLUDE_RESOLVED:-false}"
SIGNAL_DEBUG="${SIGNAL_DEBUG:-false}"
SIGNAL_DEBUG_LIMIT="${SIGNAL_DEBUG_LIMIT:-20}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

args=(
  --output-dir "${OUTPUT_DIR}"
  --pipeline-profile "${PIPELINE_PROFILE}"
  --max-events "${MAX_EVENTS}"
  --window-days "${WINDOW_DAYS}"
  --interval "${INTERVAL}"
  --price-fetch-workers "${PRICE_FETCH_WORKERS}"
  --rate-window-s "${RATE_WINDOW_S}"
  --fetch-priority-mode "${FETCH_PRIORITY_MODE}"
  --http-pool-maxsize "${HTTP_POOL_MAXSIZE}"
  --quality-min-points "${QUALITY_MIN_POINTS}"
  --quality-max-missing-ratio "${QUALITY_MAX_MISSING_RATIO}"
  --quality-min-price-range "${QUALITY_MIN_PRICE_RANGE}"
  --incremental-mode "${INCREMENTAL_MODE}"
  --incremental-overlap-points "${INCREMENTAL_OVERLAP_POINTS}"
  --signal-debug-limit "${SIGNAL_DEBUG_LIMIT}"
  --log-level "${LOG_LEVEL}"
)

if [[ "${YES_ONLY_BINARY}" == "true" ]]; then
  args+=(--yes-only-binary)
else
  args+=(--all-outcomes)
fi

if [[ "${SKIP_RAW_PRICE_FILES}" == "true" ]]; then
  args+=(--skip-raw-price-files)
fi

if [[ "${SKIP_INACTIVE_PRICED_ASSETS}" == "true" ]]; then
  args+=(--skip-inactive-priced-assets)
else
  args+=(--no-skip-inactive-priced-assets)
fi

if [[ "${NO_INCREMENTAL_PRICES}" == "true" ]]; then
  args+=(--no-incremental-prices)
fi

if [[ "${SNAPSHOT_ORDERBOOK}" == "true" ]]; then
  args+=(--snapshot-orderbook)
fi

if [[ "${INGEST_VOLUME}" == "true" ]]; then
  args+=(--ingest-volume)
fi

if [[ "${RUN_SIGNALS}" == "true" ]]; then
  args+=(--run-signals)
fi

if [[ "${RUN_BACKTEST}" == "true" ]]; then
  args+=(--backtest)
fi

if [[ "${GENERATE_CANDIDATES}" == "true" ]]; then
  args+=(--generate-candidates)
fi

if [[ "${SIGNAL_DEBUG}" == "true" ]]; then
  args+=(--signal-debug)
fi

if [[ "${INCLUDE_RESOLVED}" == "true" ]]; then
  args+=(--include-resolved)
fi

if [[ -n "${EXTRA_PIPELINE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_PIPELINE_ARGS})
  args+=("${extra_args[@]}")
fi

# Download existing data from GCS so incremental mode has something to build on.
# Cloud Run containers start with an empty filesystem, so without this step every
# run would discard accumulated history and start from scratch.
if [[ -n "${GCS_OUTPUT_URI:-}" ]]; then
  echo "Downloading existing data from ${GCS_OUTPUT_URI} to ${OUTPUT_DIR}"
  python -m polymarket_pipeline.gcs_sync \
    --mode download \
    --local-dir "${OUTPUT_DIR}" \
    --gcs-uri "${GCS_OUTPUT_URI}" || {
      echo "WARNING: GCS download failed (first run or bucket empty). Continuing with empty state."
    }
fi

echo "Running pipeline with output dir: ${OUTPUT_DIR}"
polymarket-pipeline "${args[@]}"

if [[ -n "${GCS_OUTPUT_URI:-}" ]]; then
  echo "Uploading output to ${GCS_OUTPUT_URI}"
  python -m polymarket_pipeline.gcs_sync \
    --mode upload \
    --local-dir "${OUTPUT_DIR}" \
    --gcs-uri "${GCS_OUTPUT_URI}"
fi
