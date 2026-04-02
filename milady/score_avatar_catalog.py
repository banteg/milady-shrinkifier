from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mobilenet_common import MODEL_LABEL_SOURCE, choose_device, create_model, load_promoted_run_id, probabilities_from_model
from .pipeline_common import (
    MODEL_RUN_ROOT,
    connect_db,
    connect_offline_cache_db,
    now_iso,
    resolve_repo_path,
)
from .wire import RunSummary, encode_json, load_json


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
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only write model_scores. Skip refreshing automatic model labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_id = args.run_id or load_default_run_id()
    run_dir = MODEL_RUN_ROOT / run_id
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not summary_path.exists():
        raise SystemExit(f"Training summary not found: {summary_path}")

    summary = load_json(summary_path, RunSummary)
    threshold = float(args.threshold if args.threshold is not None else summary.threshold)

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
                ["export"] * len(batch_paths),
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

        model_label_summary = None
        if not args.score_only:
            model_label_summary = refresh_model_labels(
                connection,
                run_id,
            )

        output = {
            "run_id": run_id,
            "threshold": threshold,
            "scored_images": scored,
            "score_only": args.score_only,
            "limit": args.limit,
        }
        if model_label_summary is not None:
            output["model_labels"] = model_label_summary

        print(encode_json(output, pretty=True).decode("utf-8"))
    finally:
        cache_connection.close()
        connection.close()


def refresh_model_labels(
    connection,
    run_id: str,
)-> dict[str, int]:
    candidates = connection.execute(
        """
        SELECT images.sha256, score_records.score, score_records.predicted_label
        FROM images
        INNER JOIN model_scores AS score_records
          ON score_records.image_sha256 = images.sha256
        WHERE score_records.run_id = ?
          AND (
            images.label IS NULL
            OR images.label_source = ?
          )
        ORDER BY score_records.image_sha256 ASC
        """,
        (run_id, MODEL_LABEL_SOURCE),
    ).fetchall()

    updates = [
        build_model_label_payload(run_id, str(row["sha256"]), str(row["predicted_label"]), float(row["score"]))
        for row in candidates
    ]

    connection.execute(
        """
        UPDATE images
        SET label = NULL,
            label_source = NULL,
            labeled_at = NULL,
            review_notes = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE label_source = ?
        """,
        (MODEL_LABEL_SOURCE,),
    )
    for update in updates:
        connection.execute(
            """
            UPDATE images
            SET label = ?,
                label_source = ?,
                labeled_at = CURRENT_TIMESTAMP,
                review_notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE sha256 = ?
              AND (label IS NULL OR label_source = ?)
            """,
            (
                update["label"],
                MODEL_LABEL_SOURCE,
                update["review_note"],
                update["sha256"],
                MODEL_LABEL_SOURCE,
            ),
        )
    connection.commit()
    return {
        "negative_count": sum(1 for update in updates if update["label"] == "not_milady"),
        "positive_count": sum(1 for update in updates if update["label"] == "milady"),
        "updated_count": len(updates),
    }


def load_default_run_id() -> str:
    try:
        return load_promoted_run_id()
    except FileNotFoundError as error:
        raise SystemExit(
            "No default promoted model metadata found. Pass --run-id explicitly or export a promoted model first."
        ) from error


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if args.limit is not None and not args.score_only:
        raise SystemExit("--limit requires --score-only so partial scoring does not refresh model labels.")


def build_model_label_payload(run_id: str, sha256: str, label: str, score: float) -> dict[str, str | float]:
    return {
        "sha256": sha256,
        "label": label,
        "score": score,
        "review_note": json.dumps(
            {
                "type": "model_label",
                "run_id": run_id,
                "score": score,
                "predicted_label": label,
            },
            sort_keys=True,
        ),
    }
if __name__ == "__main__":
    main()
