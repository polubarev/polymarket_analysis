from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    value = str(uri).strip()
    if not value.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    path = value[5:]
    bucket, _, prefix = path.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket name in URI: {uri}")
    return bucket, prefix.strip("/")


def upload_directory(local_dir: Path, gcs_uri: str) -> int:
    try:
        from google.cloud import storage
    except ModuleNotFoundError as exc:
        raise RuntimeError("google-cloud-storage is required for GCS sync; run pip install -e .") from exc

    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{relative}" if prefix else relative
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(path))
        uploaded += 1
        if uploaded % 100 == 0:
            LOGGER.info("Uploaded %s files to %s", uploaded, gcs_uri)

    return uploaded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload pipeline output directory to GCS")
    parser.add_argument("--local-dir", required=True, help="Local directory to upload")
    parser.add_argument("--gcs-uri", required=True, help="Target gs://bucket/prefix URI")
    parser.add_argument("--log-level", default="INFO", help="Python log level")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    local_dir = Path(args.local_dir)
    uploaded = upload_directory(local_dir, args.gcs_uri)
    LOGGER.info("Uploaded %s files from %s to %s", uploaded, local_dir, args.gcs_uri)


if __name__ == "__main__":
    main()
