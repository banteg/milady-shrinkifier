from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


MODEL_IMAGE_SIZE = 128
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
POSITIVE_LABEL = "milady"
NEGATIVE_LABEL = "not_milady"
CLASS_NAMES = [NEGATIVE_LABEL, POSITIVE_LABEL]
POSITIVE_INDEX = 1


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


def score_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, POSITIVE_INDEX]
