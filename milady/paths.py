from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "cache"
COLLECTION_ROOT = CACHE_ROOT / "collections"
COLLECTION_MANIFEST_PATH = COLLECTION_ROOT / "manifest.json"
INGEST_ROOT = CACHE_ROOT / "ingest"
EXPORT_ROOT = CACHE_ROOT / "exports" / "raw"
AVATAR_ROOT = CACHE_ROOT / "avatars" / "files"
DATASET_ROOT = CACHE_ROOT / "dataset"
SPLIT_ROOT = DATASET_ROOT / "splits"
SPLIT_MANIFEST_PATH = DATASET_ROOT / "split_manifest.json"
INFERENCE_VARIANT_CACHE_VERSION = "cover-center-top-v1"
INFERENCE_VARIANT_ROOT = DATASET_ROOT / "inference_variants" / INFERENCE_VARIANT_CACHE_VERSION
OFFLINE_CACHE_PATH = DATASET_ROOT / "offline_cache.sqlite"
MODEL_RUN_ROOT = CACHE_ROOT / "models" / "mobilenet_v3_small"
MODEL_COMPARE_ROOT = MODEL_RUN_ROOT / "compare"
CATALOG_PATH = DATASET_ROOT / "avatar_catalog.sqlite"
PUBLIC_MODEL_PATH = PROJECT_ROOT / "public" / "models" / "milady-mobilenetv3-small.onnx"
PUBLIC_METADATA_PATH = PROJECT_ROOT / "public" / "generated" / "milady-mobilenetv3-small.meta.json"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_layout() -> None:
    COLLECTION_ROOT.mkdir(parents=True, exist_ok=True)
    INGEST_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    AVATAR_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    INFERENCE_VARIANT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARE_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def discover_export_paths(inputs: list[str]) -> list[Path]:
    if inputs:
        paths = [Path(value) for value in inputs]
    else:
        paths = sorted(INGEST_ROOT.glob("*.json"))
    return [path for path in paths if path.exists()]
