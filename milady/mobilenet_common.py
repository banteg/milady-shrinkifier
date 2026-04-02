from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import msgspec
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .pipeline_common import (
    connect_offline_cache_db,
    convert_image_to_rgb,
    get_file_fingerprint,
    inference_variant_cache_path,
    write_npz_atomic,
)
from .wire import DatasetEntryPayload, MetricSummary, dump_jsonl, load_jsonl

MODEL_IMAGE_SIZE = 128
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
POSITIVE_LABEL = "milady"
NEGATIVE_LABEL = "not_milady"
CLASS_NAMES = [NEGATIVE_LABEL, POSITIVE_LABEL]
POSITIVE_INDEX = 1
SPLIT_SEED = 1337
TOP_CROP_SOURCES = frozenset({"milady-maker", "pixelady"})


class DatasetEntry(msgspec.Struct, kw_only=True):
    sample_id: str
    path: Path
    label: str
    source: str
    split: str
    label_source: str
    label_tier: str
    sample_weight: float


class AvatarDataset(Dataset[tuple[torch.Tensor, int, float]]):
    def __init__(self, entries: list[DatasetEntry], training: bool) -> None:
        self.entries = entries
        self.training = training
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        entry = self.entries[index]
        with Image.open(entry.path) as image:
            prepared = convert_image_to_rgb(image)
            if self.training:
                tensor = tensor_from_variant_array(
                    prepare_inference_variant_array(prepared, crop_variant_for_source(entry.source))
                )
            else:
                prepared = prepare_base_image(prepared)
                tensor = self.to_tensor(prepared)
        label_index = POSITIVE_INDEX if entry.label == POSITIVE_LABEL else 0
        return tensor, label_index, float(entry.sample_weight)


def create_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return self.softmax(logits)


def prepare_base_image(image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        image,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def load_dataset_entries(path: Path) -> list[DatasetEntry]:
    return [
        DatasetEntry(
            sample_id=payload.id,
            path=Path(payload.path),
            label=payload.label,
            source=payload.source,
            split=payload.split,
            label_source=payload.label_source,
            label_tier=payload.label_tier,
            sample_weight=payload.sample_weight,
        )
        for payload in load_jsonl(path, DatasetEntryPayload)
    ]


def compute_metrics(probabilities: list[float], labels: list[int], threshold: float) -> MetricSummary:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for probability, label in zip(probabilities, labels, strict=True):
        predicted = 1 if probability >= threshold else 0
        if predicted == 1 and label == 1:
            true_positive += 1
        elif predicted == 1 and label == 0:
            false_positive += 1
        elif predicted == 0 and label == 0:
            true_negative += 1
        else:
            false_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    accuracy = (true_positive + true_negative) / max(1, len(labels))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return MetricSummary(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=float(true_positive),
        false_positive=float(false_positive),
        true_negative=float(true_negative),
        false_negative=float(false_negative),
    )


def choose_threshold(probabilities: list[float], labels: list[int], precision_floor: float) -> tuple[float, MetricSummary]:
    if not probabilities:
        return 0.995, compute_metrics(probabilities, labels, 0.995)

    candidates = sorted({0.0, 1.0, *probabilities})
    scored_candidates = [
        (float(threshold), compute_metrics(probabilities, labels, float(threshold)))
        for threshold in candidates
    ]
    passing_candidates = [
        (threshold, metrics)
        for threshold, metrics in scored_candidates
        if metrics.precision >= precision_floor
    ]

    if passing_candidates:
        best_threshold, best_metrics = max(
            passing_candidates,
            key=lambda item: (
                item[1].recall,
                item[1].f1,
                item[0],
            ),
        )
        return best_threshold, best_metrics

    best_threshold, best_metrics = max(
        scored_candidates,
        key=lambda item: (
            item[1].precision,
            item[1].recall,
            item[1].f1,
            item[0],
        ),
    )
    return best_threshold, best_metrics


def score_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, POSITIVE_INDEX]


def dataset_entries_to_jsonl(entries: list[DatasetEntry], path: Path) -> None:
    dump_jsonl(
        path,
        [
            DatasetEntryPayload(
                id=entry.sample_id,
                path=str(entry.path),
                label=entry.label,
                source=entry.source,
                split=entry.split,
                label_source=entry.label_source,
                label_tier=entry.label_tier,
                sample_weight=entry.sample_weight,
            )
            for entry in entries
        ],
    )


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


def prepare_inference_variant_array(image: Image.Image, variant: Literal["center", "top"]) -> np.ndarray:
    centering = (0.5, 0.0) if variant == "top" else (0.5, 0.5)
    prepared = ImageOps.fit(
        convert_image_to_rgb(image),
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        method=Image.Resampling.BICUBIC,
        centering=centering,
    )
    return np.asarray(prepared, dtype=np.uint8)


def crop_variant_for_source(source: str) -> Literal["center", "top"]:
    return "top" if source in TOP_CROP_SOURCES else "center"


def tensor_from_variant_array(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD)(tensor)


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
