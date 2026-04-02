from __future__ import annotations

import argparse
import re
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

from .mobilenet_common import create_model, prepare_base_image, score_logits_to_probabilities
from .pipeline_common import MODEL_RUN_ROOT, convert_image_to_rgb
from .wire import RunSummary, encode_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a single profile image URL or local image file with the current Milady classifier.")
    parser.add_argument("input", help="Profile image URL to fetch and score, or a local image file path.")
    parser.add_argument("--run-id", help="Training run id under cache/models/mobilenet_v3_small/. Defaults to the newest run with a summary.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the decision threshold.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or resolve_latest_run_id()
    if not run_id:
        raise SystemExit("No classifier runs found. Train a classifier first.")

    run_dir = MODEL_RUN_ROOT / run_id
    checkpoint_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not summary_path.exists():
        raise SystemExit(f"Training summary not found: {summary_path}")

    summary = load_json(summary_path, RunSummary)
    threshold = float(args.threshold if args.threshold is not None else summary.threshold)

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    output: dict[str, object] = {
        "input": args.input,
        "run_id": run_id,
        "threshold": threshold,
    }
    local_path = Path(args.input).expanduser()
    if local_path.exists() and local_path.is_file():
        probability = infer_probability(model, local_path.read_bytes())
        output["local_path"] = str(local_path.resolve())
    else:
        normalized_url = normalize_profile_image_url(args.input)
        response = httpx.get(normalized_url, timeout=args.timeout, follow_redirects=True)
        response.raise_for_status()
        probability = infer_probability(model, response.content)
        output["url"] = args.input
        output["normalized_url"] = normalized_url

    print(
        encode_json(
            {
                **output,
                "probability": probability,
                "matched": probability >= threshold,
            },
            pretty=True,
        ).decode("utf-8")
    )


def resolve_latest_run_id() -> str | None:
    if not MODEL_RUN_ROOT.exists():
        return None
    candidates = [
        path.name
        for path in sorted(MODEL_RUN_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        if path.is_dir() and (path / "summary.json").exists() and (path / "best.pt").exists()
    ]
    return candidates[0] if candidates else None


def normalize_profile_image_url(url: str) -> str:
    return re.sub(
        r"_(normal|bigger|mini|reasonably_small|(?:\d+x\d+)|(?:x\d+))(\.[a-z0-9]+)$",
        r"_400x400\2",
        url,
        flags=re.IGNORECASE,
    )


def infer_probability(model: torch.nn.Module, image_bytes: bytes) -> float:
    with Image.open(BytesIO(image_bytes)) as image:
        prepared = prepare_base_image(convert_image_to_rgb(image))
        tensor = torch.from_numpy(np.asarray(prepared, dtype=np.uint8).copy()).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        tensor[0].sub_(0.485).div_(0.229)
        tensor[1].sub_(0.456).div_(0.224)
        tensor[2].sub_(0.406).div_(0.225)
    with torch.no_grad():
        probability = score_logits_to_probabilities(model(tensor.unsqueeze(0)))[0]
    return float(probability.item())


if __name__ == "__main__":
    main()
