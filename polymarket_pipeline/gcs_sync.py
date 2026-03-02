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


def download_directory(local_dir: Path, gcs_uri: str) -> int:
    """Download all files from a GCS prefix into a local directory.

    Preserves relative paths so that ``gs://bucket/prefix/a/b.parquet``
    becomes ``<local_dir>/a/b.parquet``.  Existing local files are
    overwritten (GCS is the source of truth between Cloud Run jobs).
    """
    try:
        from google.cloud import storage
    except ModuleNotFoundError as exc:
        raise RuntimeError("google-cloud-storage is required for GCS sync; run pip install -e .") from exc

    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for blob in client.list_blobs(bucket, prefix=f"{prefix}/" if prefix else ""):
        # Skip directory markers.
        if blob.name.endswith("/"):
            continue
        relative = blob.name
        if prefix:
            relative = blob.name[len(prefix) :].lstrip("/")
        if not relative:
            continue
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        downloaded += 1
        if downloaded % 100 == 0:
            LOGGER.info("Downloaded %s files from %s", downloaded, gcs_uri)

    LOGGER.info("Downloaded %s files from %s to %s", downloaded, gcs_uri, local_dir)
    return downloaded


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
    parser = argparse.ArgumentParser(description="Sync pipeline output directory with GCS")
    parser.add_argument("--local-dir", required=True, help="Local directory")
    parser.add_argument("--gcs-uri", required=True, help="Target gs://bucket/prefix URI")
    parser.add_argument(
        "--mode",
        choices=["upload", "download"],
        default="upload",
        help="Direction of sync (default: upload)",
    )
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
    if args.mode == "download":
        count = download_directory(local_dir, args.gcs_uri)
        LOGGER.info("Downloaded %s files from %s to %s", count, args.gcs_uri, local_dir)
    else:
        count = upload_directory(local_dir, args.gcs_uri)
        LOGGER.info("Uploaded %s files from %s to %s", count, local_dir, args.gcs_uri)


if __name__ == "__main__":
    main()
