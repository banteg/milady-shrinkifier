from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .catalog_db import connect_offline_cache_db
from .dataset_runtime import DatasetEntry
from .image_files import convert_image_to_rgb, get_file_fingerprint, inference_variant_cache_path, write_npz_atomic
from .modeling import MODEL_IMAGE_SIZE, POSITIVE_LABEL, score_logits_to_probabilities
from .paths import MODEL_RUN_ROOT, PUBLIC_METADATA_PATH
from .preprocess import crop_variant_for_source, prepare_inference_variant_array, tensor_from_variant_array
from .wire import PublicModelMetadata, load_json


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_latest_run_id(*, exclude: set[str] | None = None) -> str | None:
    if not MODEL_RUN_ROOT.exists():
        return None
    excluded = exclude or set()
    candidates = [
        path.name
        for path in sorted(MODEL_RUN_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        if path.is_dir()
        and path.name not in excluded
        and (path / "summary.json").exists()
        and (path / "best.pt").exists()
    ]
    return candidates[0] if candidates else None


def load_promoted_run_id() -> str:
    return load_json(PUBLIC_METADATA_PATH, PublicModelMetadata).run_id


def evaluate_entries(
    model: nn.Module,
    entries: list[DatasetEntry],
    device: torch.device,
    batch_size: int = 64,
    cache_connection: sqlite3.Connection | None = None,
) -> tuple[list[float], list[int]]:
    probabilities = probabilities_from_model(
        model,
        [entry.path for entry in entries],
        [entry.source for entry in entries],
        device,
        batch_size=batch_size,
        connection=cache_connection or connect_offline_cache_db(),
    ).tolist()
    labels = [1 if entry.label == POSITIVE_LABEL else 0 for entry in entries]
    return probabilities, labels


def load_image_for_inference_with_cache_for_source(
    path: Path,
    source: str,
    connection: sqlite3.Connection,
) -> torch.Tensor:
    fingerprint = get_file_fingerprint(connection, path, MODEL_IMAGE_SIZE)
    if not fingerprint.readable:
        raise ValueError(f"Unreadable image: {path}")
    center, top = load_or_create_inference_variant_arrays(path, fingerprint.raw_sha)
    variant = top if crop_variant_for_source(source) == "top" else center
    return tensor_from_variant_array(variant)


def load_or_create_inference_variant_arrays(path: Path, raw_sha: str) -> tuple[np.ndarray, np.ndarray]:
    cache_path = inference_variant_cache_path(raw_sha)
    if cache_path.exists():
        try:
            with np.load(cache_path) as payload:
                return payload["center"], payload["top"]
        except Exception:  # noqa: BLE001
            cache_path.unlink(missing_ok=True)

    with Image.open(path) as image:
        prepared = convert_image_to_rgb(image)
        center = prepare_inference_variant_array(prepared, "center")
        top = prepare_inference_variant_array(prepared, "top")
    write_npz_atomic(cache_path, center=center, top=top)
    return center, top


def probabilities_from_model(
    model: nn.Module,
    paths: list[Path],
    sources: list[str],
    device: torch.device,
    connection: sqlite3.Connection,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_sources = sources[start : start + batch_size]
            tensors = torch.stack(
                [
                    load_image_for_inference_with_cache_for_source(path, source, connection)
                    for path, source in zip(batch_paths, batch_sources, strict=True)
                ],
                dim=0,
            ).to(device)
            probabilities = score_logits_to_probabilities(model(tensors))
            batches.append(probabilities.detach().cpu().numpy())
    return np.concatenate(batches) if batches else np.array([], dtype=np.float32)
