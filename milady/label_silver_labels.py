from __future__ import annotations

import argparse
import json

from .pipeline_common import connect_db

SILVER_LABEL_SOURCE = "silver"
DEFAULT_NEGATIVE_MAX_PROBABILITY = 0.005
DEFAULT_POSITIVE_MIN_PROBABILITY = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote extreme-confidence scored unlabeled avatars into weak silver labels."
    )
    parser.add_argument("--run-id", required=True, help="Model run id to read scores from.")
    parser.add_argument(
        "--max-negative-probability",
        type=float,
        default=DEFAULT_NEGATIVE_MAX_PROBABILITY,
        help="Auto-label unlabeled images with score <= this threshold as not_milady.",
    )
    parser.add_argument(
        "--min-positive-probability",
        type=float,
        default=DEFAULT_POSITIVE_MIN_PROBABILITY,
        help="Optional auto-label threshold for milady. Disabled by default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of unlabeled images considered, ordered by strongest confidence first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which rows would be updated without changing the catalog.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    connection = connect_db()
    try:
        score_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM model_scores WHERE run_id = ?",
                (args.run_id,),
            ).fetchone()[0]
        )
        if score_count == 0:
            raise SystemExit(f"No catalog scores found for run {args.run_id}. Run `uv run milady score --run-id {args.run_id}` first.")

        negative_candidates = connection.execute(
            """
            SELECT images.sha256, score_records.score
            FROM images
            INNER JOIN model_scores AS score_records
              ON score_records.image_sha256 = images.sha256
            WHERE score_records.run_id = ?
              AND images.label IS NULL
              AND score_records.score <= ?
            ORDER BY score_records.score ASC, images.sha256 ASC
            LIMIT COALESCE(?, -1)
            """,
            (args.run_id, args.max_negative_probability, args.limit),
        ).fetchall()

        positive_candidates = []
        if args.min_positive_probability is not None:
            positive_candidates = connection.execute(
                """
                SELECT images.sha256, score_records.score
                FROM images
                INNER JOIN model_scores AS score_records
                  ON score_records.image_sha256 = images.sha256
                WHERE score_records.run_id = ?
                  AND images.label IS NULL
                  AND score_records.score >= ?
                ORDER BY score_records.score DESC, images.sha256 ASC
                LIMIT COALESCE(?, -1)
                """,
                (args.run_id, args.min_positive_probability, args.limit),
            ).fetchall()

        updates = [
            build_update_payload(args.run_id, str(row["sha256"]), "not_milady", float(row["score"]))
            for row in negative_candidates
        ] + [
            build_update_payload(args.run_id, str(row["sha256"]), "milady", float(row["score"]))
            for row in positive_candidates
        ]

        if not args.dry_run:
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
                      AND label IS NULL
                    """,
                    (
                        update["label"],
                        SILVER_LABEL_SOURCE,
                        update["review_note"],
                        update["sha256"],
                    ),
                )
            connection.commit()

        print(
            json.dumps(
                {
                    "runId": args.run_id,
                    "maxNegativeProbability": args.max_negative_probability,
                    "minPositiveProbability": args.min_positive_probability,
                    "negativeSilverLabels": len(negative_candidates),
                    "positiveSilverLabels": len(positive_candidates),
                    "updatedLabels": len(updates),
                    "dryRun": args.dry_run,
                    "limit": args.limit,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.max_negative_probability <= 1.0:
        raise SystemExit("--max-negative-probability must be between 0 and 1.")
    if args.min_positive_probability is not None and not 0.0 <= args.min_positive_probability <= 1.0:
        raise SystemExit("--min-positive-probability must be between 0 and 1.")
    if (
        args.min_positive_probability is not None
        and args.min_positive_probability <= args.max_negative_probability
    ):
        raise SystemExit("--min-positive-probability must be greater than --max-negative-probability.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")


def build_update_payload(run_id: str, sha256: str, label: str, score: float) -> dict[str, str | float]:
    return {
        "sha256": sha256,
        "label": label,
        "score": score,
        "review_note": json.dumps(
            {
                "type": "silver_label",
                "runId": run_id,
                "score": score,
                "predictedLabel": label,
            },
            sort_keys=True,
        ),
    }


if __name__ == "__main__":
    main()
