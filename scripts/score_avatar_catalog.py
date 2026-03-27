from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mobilenet_common import create_model, load_image_for_inference, score_logits_to_probabilities
from pipeline_common import MODEL_RUN_ROOT, connect_db, now_iso, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score downloaded catalog avatars with a trained MobileNetV3-Small checkpoint.")
    parser.add_argument("--run-id", required=True, help="Training run id under cache/models/mobilenet_v3_small/")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to cache/models/mobilenet_v3_small/<run-id>/best.pt")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold when writing predicted labels.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--limit", type=int, default=None)
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

    summary = json.loads(summary_path.read_text())
    threshold = float(args.threshold if args.threshold is not None else summary["threshold"])

    device = choose_device(args.cpu)
    model = create_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    connection = connect_db()
    rows = connection.execute(
        """
        SELECT sha256, local_path, split
        FROM images
        WHERE local_path IS NOT NULL
        ORDER BY updated_at DESC
        """
    ).fetchall()
    if args.limit is not None:
        rows = rows[: args.limit]

    created_at = now_iso()
    scored = 0
    for row in rows:
        path = resolve_repo_path(str(row["local_path"]))
        if not path.exists():
            continue
        probability = infer_probability(model, path, device)
        connection.execute(
            """
            INSERT INTO model_scores (
              run_id,
              image_sha256,
              score,
              predicted_label,
              split,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, image_sha256) DO UPDATE SET
              score = excluded.score,
              predicted_label = excluded.predicted_label,
              split = excluded.split,
              created_at = excluded.created_at
            """,
            (
                args.run_id,
                str(row["sha256"]),
                probability,
                "milady" if probability >= threshold else "not_milady",
                row["split"],
                created_at,
            ),
        )
        scored += 1

    connection.commit()
    print(f"Scored {scored} image(s) for run {args.run_id}.")


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_probability(model: torch.nn.Module, path: Path, device: torch.device) -> float:
    tensor = load_image_for_inference(path).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probability = score_logits_to_probabilities(logits)[0]
    return float(probability.detach().cpu().item())


if __name__ == "__main__":
    main()
