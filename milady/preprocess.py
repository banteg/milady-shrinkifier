from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .image_files import convert_image_to_rgb
from .modeling import MODEL_IMAGE_SIZE, MODEL_MEAN, MODEL_STD

TOP_CROP_SOURCES = frozenset({"milady-maker", "pixelady"})


def prepare_base_image(image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        image,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


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
