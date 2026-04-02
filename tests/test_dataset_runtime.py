from pathlib import Path

from PIL import Image

from milady.dataset_runtime import AvatarDataset, DatasetEntry
from milady.preprocess import crop_variant_for_source


def test_crop_variant_for_source_matches_collection_policy() -> None:
    assert crop_variant_for_source("export") == "center"
    assert crop_variant_for_source("remilio") == "center"
    assert crop_variant_for_source("milady-maker") == "top"
    assert crop_variant_for_source("pixelady") == "top"


def test_training_dataset_uses_single_crop_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (128, 192), color=(255, 0, 0)).save(image_path)
    dataset = AvatarDataset(
        [
            DatasetEntry(
                sample_id="collection:milady-maker:1",
                path=image_path,
                label="milady",
                source="milady-maker",
                split="train",
                label_source="collection_corpus",
                label_tier="trusted",
                sample_weight=0.5,
            )
        ],
        training=True,
    )

    tensor, label, weight = dataset[0]

    assert tuple(tensor.shape) == (3, 128, 128)
    assert label == 1
    assert weight == 0.5
