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
MAX_EVENTS="${MAX_EVENTS:-300}"
WINDOW_DAYS="${WINDOW_DAYS:-30}"
INTERVAL="${INTERVAL:-1h}"
PRICE_FETCH_WORKERS="${PRICE_FETCH_WORKERS:-24}"
HTTP_POOL_MAXSIZE="${HTTP_POOL_MAXSIZE:-64}"
QUALITY_MIN_POINTS="${QUALITY_MIN_POINTS:-24}"
QUALITY_MAX_MISSING_RATIO="${QUALITY_MAX_MISSING_RATIO:-0.9}"
QUALITY_MIN_PRICE_RANGE="${QUALITY_MIN_PRICE_RANGE:-0.005}"
INCREMENTAL_MODE="${INCREMENTAL_MODE:-tail}"
INCREMENTAL_OVERLAP_POINTS="${INCREMENTAL_OVERLAP_POINTS:-2}"
YES_ONLY_BINARY="${YES_ONLY_BINARY:-true}"
SKIP_RAW_PRICE_FILES="${SKIP_RAW_PRICE_FILES:-true}"
NO_INCREMENTAL_PRICES="${NO_INCREMENTAL_PRICES:-false}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

args=(
  --output-dir "${OUTPUT_DIR}"
  --max-events "${MAX_EVENTS}"
  --window-days "${WINDOW_DAYS}"
  --interval "${INTERVAL}"
  --price-fetch-workers "${PRICE_FETCH_WORKERS}"
  --http-pool-maxsize "${HTTP_POOL_MAXSIZE}"
  --quality-min-points "${QUALITY_MIN_POINTS}"
  --quality-max-missing-ratio "${QUALITY_MAX_MISSING_RATIO}"
  --quality-min-price-range "${QUALITY_MIN_PRICE_RANGE}"
  --incremental-mode "${INCREMENTAL_MODE}"
  --incremental-overlap-points "${INCREMENTAL_OVERLAP_POINTS}"
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

if [[ "${NO_INCREMENTAL_PRICES}" == "true" ]]; then
  args+=(--no-incremental-prices)
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
