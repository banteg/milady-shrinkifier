from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import onnx
import torch

from .modeling import CLASS_NAMES, ExportWrapper, MODEL_IMAGE_SIZE, MODEL_MEAN, MODEL_STD, POSITIVE_INDEX, create_model
from .paths import MODEL_RUN_ROOT, PUBLIC_METADATA_PATH, PUBLIC_MODEL_PATH
from .wire import PublicModelMetadata, RunSummary, dump_json, encode_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained MobileNetV3-Small classifier to ONNX for the extension runtime.")
    parser.add_argument("--run-id", required=True, help="Training run id under cache/models/mobilenet_v3_small/")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to cache/models/mobilenet_v3_small/<run-id>/best.pt")
    parser.add_argument("--threshold", type=float, default=None, help="Override the exported decision threshold.")
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = MODEL_RUN_ROOT / args.run_id
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not summary_path.exists():
        raise SystemExit(f"Training summary not found: {summary_path}")

    summary = load_json(summary_path, RunSummary)
    threshold = float(args.threshold if args.threshold is not None else summary.threshold)

    model = create_model(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, dtype=torch.float32)
    PUBLIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        PUBLIC_MODEL_PATH,
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=args.opset,
        dynamo=False,
        external_data=False,
        dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )
    stale_external_data = PUBLIC_MODEL_PATH.with_suffix(PUBLIC_MODEL_PATH.suffix + ".data")
    if stale_external_data.exists():
        stale_external_data.unlink()
    onnx.checker.check_model(str(PUBLIC_MODEL_PATH))

    metadata = PublicModelMetadata(
        architecture="mobilenet_v3_small",
        generated_at=datetime.now(UTC).isoformat(),
        input_size=MODEL_IMAGE_SIZE,
        channels=3,
        class_names=CLASS_NAMES,
        mean=MODEL_MEAN,
        std=MODEL_STD,
        threshold=threshold,
        positive_index=POSITIVE_INDEX,
        run_id=args.run_id,
    )
    dump_json(PUBLIC_METADATA_PATH, metadata)
    print(
        encode_json(
            {
                "model_path": str(PUBLIC_MODEL_PATH),
                "metadata_path": str(PUBLIC_METADATA_PATH),
                "threshold": threshold,
            },
            pretty=True,
        ).decode("utf-8")
    )


if __name__ == "__main__":
    main()
