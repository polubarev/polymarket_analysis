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

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=us-central1}"
: "${GCS_OUTPUT_URI:?Set GCS_OUTPUT_URI to gs://bucket/prefix}"

AR_REPO="${AR_REPO:-polymarket}"
IMAGE_NAME="${IMAGE_NAME:-polymarket-pipeline}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

HOURLY_JOB_NAME="${HOURLY_JOB_NAME:-polymarket-hourly}"
NIGHTLY_JOB_NAME="${NIGHTLY_JOB_NAME:-polymarket-nightly-reconcile}"
RESEARCH_JOB_NAME="${RESEARCH_JOB_NAME:-polymarket-daily-research}"
HOURLY_SCHEDULER_NAME="${HOURLY_SCHEDULER_NAME:-polymarket-hourly-trigger}"
NIGHTLY_SCHEDULER_NAME="${NIGHTLY_SCHEDULER_NAME:-polymarket-nightly-trigger}"
RESEARCH_SCHEDULER_NAME="${RESEARCH_SCHEDULER_NAME:-polymarket-daily-research-trigger}"
HOURLY_CRON="${HOURLY_CRON:-10 * * * *}"
NIGHTLY_CRON="${NIGHTLY_CRON:-25 3 * * *}"
RESEARCH_CRON="${RESEARCH_CRON:-55 4 * * *}"
TIME_ZONE="${TIME_ZONE:-UTC}"
SCHEDULER_SERVICE_ACCOUNT="${SCHEDULER_SERVICE_ACCOUNT:-}"

function ensure_services() {
  gcloud services enable \
    run.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --project "${PROJECT_ID}"
}

function ensure_artifact_repo() {
  if ! gcloud artifacts repositories describe "${AR_REPO}" \
    --location "${REGION}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${AR_REPO}" \
      --repository-format docker \
      --location "${REGION}" \
      --project "${PROJECT_ID}"
  fi
}

function build_image() {
  gcloud builds submit --tag "${IMAGE_URI}" --project "${PROJECT_ID}" .
}

function deploy_hourly_job() {
  gcloud run jobs deploy "${HOURLY_JOB_NAME}" \
    --image "${IMAGE_URI}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --cpu "2" \
    --memory "8Gi" \
    --task-timeout "3600s" \
    --max-retries "1" \
    --set-env-vars "OUTPUT_DIR=/tmp/polymarket-data,PIPELINE_PROFILE=ingest-hourly,GCS_OUTPUT_URI=${GCS_OUTPUT_URI},MAX_EVENTS=300,WINDOW_DAYS=30,INTERVAL=1h,YES_ONLY_BINARY=true,PRICE_FETCH_WORKERS=2,RATE_WINDOW_S=10,FETCH_PRIORITY_MODE=history_first,HTTP_POOL_MAXSIZE=64,QUALITY_MIN_POINTS=24,QUALITY_MAX_MISSING_RATIO=0.9,QUALITY_MIN_PRICE_RANGE=0.005,INCREMENTAL_MODE=tail,INCREMENTAL_OVERLAP_POINTS=2,SKIP_INACTIVE_PRICED_ASSETS=true,SKIP_RAW_PRICE_FILES=true,NO_INCREMENTAL_PRICES=false,SNAPSHOT_ORDERBOOK=false,INGEST_VOLUME=false,RUN_SIGNALS=false,RUN_BACKTEST=false,GENERATE_CANDIDATES=false,SIGNAL_DEBUG=false,LOG_LEVEL=INFO"
}

function deploy_nightly_job() {
  gcloud run jobs deploy "${NIGHTLY_JOB_NAME}" \
    --image "${IMAGE_URI}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --cpu "4" \
    --memory "16Gi" \
    --task-timeout "7200s" \
    --max-retries "1" \
    --set-env-vars "OUTPUT_DIR=/tmp/polymarket-data,PIPELINE_PROFILE=ingest-nightly,GCS_OUTPUT_URI=${GCS_OUTPUT_URI},MAX_EVENTS=1000,WINDOW_DAYS=30,INTERVAL=1h,YES_ONLY_BINARY=true,PRICE_FETCH_WORKERS=2,RATE_WINDOW_S=10,FETCH_PRIORITY_MODE=history_first,HTTP_POOL_MAXSIZE=64,QUALITY_MIN_POINTS=24,QUALITY_MAX_MISSING_RATIO=0.9,QUALITY_MIN_PRICE_RANGE=0.005,INCREMENTAL_MODE=tail,INCREMENTAL_OVERLAP_POINTS=2,SKIP_INACTIVE_PRICED_ASSETS=true,SKIP_RAW_PRICE_FILES=true,NO_INCREMENTAL_PRICES=true,SNAPSHOT_ORDERBOOK=false,INGEST_VOLUME=false,RUN_SIGNALS=false,RUN_BACKTEST=false,GENERATE_CANDIDATES=false,SIGNAL_DEBUG=false,LOG_LEVEL=INFO"
}

function deploy_research_job() {
  gcloud run jobs deploy "${RESEARCH_JOB_NAME}" \
    --image "${IMAGE_URI}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --cpu "4" \
    --memory "16Gi" \
    --task-timeout "7200s" \
    --max-retries "1" \
    --set-env-vars "OUTPUT_DIR=/tmp/polymarket-data,PIPELINE_PROFILE=research-daily,GCS_OUTPUT_URI=${GCS_OUTPUT_URI},MAX_EVENTS=1000,WINDOW_DAYS=30,INTERVAL=1h,YES_ONLY_BINARY=true,PRICE_FETCH_WORKERS=2,RATE_WINDOW_S=10,FETCH_PRIORITY_MODE=history_first,HTTP_POOL_MAXSIZE=64,QUALITY_MIN_POINTS=24,QUALITY_MAX_MISSING_RATIO=0.9,QUALITY_MIN_PRICE_RANGE=0.005,INCREMENTAL_MODE=tail,INCREMENTAL_OVERLAP_POINTS=2,SKIP_INACTIVE_PRICED_ASSETS=true,SKIP_RAW_PRICE_FILES=true,NO_INCREMENTAL_PRICES=false,SNAPSHOT_ORDERBOOK=true,INGEST_VOLUME=true,RUN_SIGNALS=true,RUN_BACKTEST=true,GENERATE_CANDIDATES=true,SIGNAL_DEBUG=true,SIGNAL_DEBUG_LIMIT=20,LOG_LEVEL=INFO"
}

function upsert_scheduler_job() {
  local scheduler_name="$1"
  local schedule="$2"
  local target_job="$3"
  local uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${target_job}:run"

  if gcloud scheduler jobs describe "${scheduler_name}" \
    --location "${REGION}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud scheduler jobs update http "${scheduler_name}" \
      --location "${REGION}" \
      --project "${PROJECT_ID}" \
      --schedule "${schedule}" \
      --time-zone "${TIME_ZONE}" \
      --uri "${uri}" \
      --http-method POST \
      --oauth-service-account-email "${SCHEDULER_SERVICE_ACCOUNT}" \
      --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform"
  else
    gcloud scheduler jobs create http "${scheduler_name}" \
      --location "${REGION}" \
      --project "${PROJECT_ID}" \
      --schedule "${schedule}" \
      --time-zone "${TIME_ZONE}" \
      --uri "${uri}" \
      --http-method POST \
      --oauth-service-account-email "${SCHEDULER_SERVICE_ACCOUNT}" \
      --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform"
  fi
}

ensure_services
ensure_artifact_repo
build_image
deploy_hourly_job
deploy_nightly_job
deploy_research_job

if [[ -n "${SCHEDULER_SERVICE_ACCOUNT}" ]]; then
  upsert_scheduler_job "${HOURLY_SCHEDULER_NAME}" "${HOURLY_CRON}" "${HOURLY_JOB_NAME}"
  upsert_scheduler_job "${NIGHTLY_SCHEDULER_NAME}" "${NIGHTLY_CRON}" "${NIGHTLY_JOB_NAME}"
  upsert_scheduler_job "${RESEARCH_SCHEDULER_NAME}" "${RESEARCH_CRON}" "${RESEARCH_JOB_NAME}"
else
  echo "Skipping Cloud Scheduler setup because SCHEDULER_SERVICE_ACCOUNT is not set."
fi

echo "Deployed image: ${IMAGE_URI}"
echo "Cloud Run jobs: ${HOURLY_JOB_NAME}, ${NIGHTLY_JOB_NAME}, ${RESEARCH_JOB_NAME}"
