from __future__ import annotations

from pathlib import Path

import msgspec
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .image_files import convert_image_to_rgb
from .modeling import MODEL_MEAN, MODEL_STD, POSITIVE_INDEX, POSITIVE_LABEL
from .preprocess import crop_variant_for_source, prepare_base_image, prepare_inference_variant_array, tensor_from_variant_array
from .wire import DatasetEntryPayload, dump_jsonl, load_jsonl

HEADLINE_EVAL_POLICY = "manual_export_gold_plus_collection_positive_holdout"
MANUAL_LABEL_SOURCE = "manual"
MODEL_LABEL_SOURCE = "model"
COLLECTION_LABEL_SOURCE = "collection_corpus"
SPLIT_SEED = 1337


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
