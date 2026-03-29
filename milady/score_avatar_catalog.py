from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mobilenet_common import probabilities_from_model, create_model
from .pipeline_common import (
    MODEL_RUN_ROOT,
    PUBLIC_METADATA_PATH,
    connect_db,
    connect_offline_cache_db,
    now_iso,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score downloaded catalog avatars with a trained MobileNetV3-Small checkpoint.")
    parser.add_argument(
        "--run-id",
        help="Training run id under cache/models/mobilenet_v3_small/. Defaults to the currently promoted run.",
    )
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to cache/models/mobilenet_v3_small/<run-id>/best.pt")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold when writing predicted labels.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of catalog images to score per chunk.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or load_default_run_id()
    run_dir = MODEL_RUN_ROOT / run_id
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
    cache_connection = connect_offline_cache_db()
    try:
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
        existing = 0
        for offset in range(0, len(rows), max(1, args.batch_size)):
            raw_batch = rows[offset : offset + max(1, args.batch_size)]
            batch_paths: list[Path] = []
            batch_rows = []
            for row in raw_batch:
                path = resolve_repo_path(str(row["local_path"]))
                if not path.exists():
                    continue
                batch_paths.append(path)
                batch_rows.append(row)

            if not batch_rows:
                continue

            probabilities = probabilities_from_model(
                model,
                batch_paths,
                device,
                batch_size=max(1, args.batch_size),
                connection=cache_connection,
            )
            payload = [
                (
                    run_id,
                    str(row["sha256"]),
                    probability,
                    "milady" if probability >= threshold else "not_milady",
                    row["split"],
                    created_at,
                )
                for row, probability in zip(batch_rows, probabilities.tolist(), strict=True)
            ]
            connection.executemany(
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
                payload,
            )
            connection.commit()
            scored += len(payload)
            existing = offset + len(raw_batch)
            print(
                f"[score] processed={min(existing, len(rows))}/{len(rows)} scored={scored}",
                flush=True,
            )

        print(f"Scored {scored} image(s) for run {run_id}.")
    finally:
        cache_connection.close()
        connection.close()


def load_default_run_id() -> str:
    if not PUBLIC_METADATA_PATH.exists():
        raise SystemExit(
            "No default promoted model metadata found. Pass --run-id explicitly or export a promoted model first."
        )
    payload = json.loads(PUBLIC_METADATA_PATH.read_text())
    run_id = payload.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise SystemExit(
            f"Promoted model metadata at {PUBLIC_METADATA_PATH} does not contain a valid runId."
        )
    return run_id


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if __name__ == "__main__":
    main()
