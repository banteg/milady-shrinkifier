from __future__ import annotations

import sqlite3

import msgspec

from .catalog_db import bool_from_db, max_timestamp, merge_string_lists, normalize_label
from .paths import MODEL_RUN_ROOT
from .wire import ReviewItem, RunSummary, decode_string_list, load_json

REVIEW_QUEUES = (
    "needs_review",
    "model_disagreements",
    "exempted",
    "high_impact",
    "notifications",
    "boundary_review",
    "model_backlog",
    "hard_negatives",
)
LABELS = ("milady", "not_milady", "unclear")
LABELED_GRID_FILTERS = ("all", "milady", "not_milady", "unclear")


def load_review_items(connection: sqlite3.Connection, run_id: str | None = None) -> list[ReviewItem]:
    if run_id is None:
        image_rows = connection.execute(
            """
            SELECT images.*,
                   latest_scores.score AS latest_model_score,
                   latest_scores.predicted_label AS latest_model_predicted_label,
                   latest_scores.run_id AS latest_model_run_id
            FROM images
            LEFT JOIN (
              SELECT score_records.image_sha256,
                     score_records.score,
                     score_records.predicted_label,
                     score_records.run_id
              FROM model_scores AS score_records
              INNER JOIN (
                SELECT image_sha256, MAX(created_at) AS latest_created_at
                FROM model_scores
                GROUP BY image_sha256
              ) AS latest
                ON latest.image_sha256 = score_records.image_sha256
               AND latest.latest_created_at = score_records.created_at
            ) AS latest_scores
              ON latest_scores.image_sha256 = images.sha256
            WHERE images.local_path IS NOT NULL
            ORDER BY images.updated_at DESC
            """
        ).fetchall()
    else:
        image_rows = connection.execute(
            """
            SELECT images.*,
                   score_records.score AS latest_model_score,
                   score_records.predicted_label AS latest_model_predicted_label,
                   score_records.run_id AS latest_model_run_id
            FROM images
            LEFT JOIN model_scores AS score_records
              ON score_records.image_sha256 = images.sha256
             AND score_records.run_id = ?
            WHERE images.local_path IS NOT NULL
            ORDER BY images.updated_at DESC
            """,
            (run_id,),
        ).fetchall()

    avatar_rows = connection.execute(
        """
        SELECT *
        FROM avatar_urls
        WHERE image_sha256 IS NOT NULL
        """
    ).fetchall()

    avatar_by_sha: dict[str, list[sqlite3.Row]] = {}
    for row in avatar_rows:
        avatar_by_sha.setdefault(str(row["image_sha256"]), []).append(row)

    thresholds_by_run = load_model_thresholds(
        {
            str(row["latest_model_run_id"])
            for row in image_rows
            if row["latest_model_run_id"] is not None
        }
    )

    review_items: list[ReviewItem] = []
    for image_row in image_rows:
        sha256 = str(image_row["sha256"])
        related = avatar_by_sha.get(sha256, [])
        handles: list[str] = []
        display_names: list[str] = []
        source_surfaces: list[str] = []
        seen_count = 0
        whitelisted = False
        example_profile_url: str | None = None
        example_notification_url: str | None = None
        example_tweet_url: str | None = None
        last_seen_at: str | None = None

        for row in related:
            handles = merge_string_lists(handles, decode_string_list(row["handles_json"]))
            display_names = merge_string_lists(display_names, decode_string_list(row["display_names_json"]))
            source_surfaces = merge_string_lists(source_surfaces, decode_string_list(row["source_surfaces_json"]))
            seen_count += int(row["seen_count"])
            whitelisted = whitelisted or bool_from_db(row["whitelisted"])
            example_profile_url = example_profile_url or row["example_profile_url"]
            example_notification_url = example_notification_url or row["example_notification_url"]
            example_tweet_url = example_tweet_url or row["example_tweet_url"]
            if row["last_seen_at"]:
                last_seen_at = max_timestamp(last_seen_at, str(row["last_seen_at"]))

        human_label = normalize_label(image_row["label"])
        latest_model_predicted_label = (
            str(image_row["latest_model_predicted_label"])
            if image_row["latest_model_predicted_label"] is not None
            else None
        )
        latest_model_run_id = (
            str(image_row["latest_model_run_id"])
            if image_row["latest_model_run_id"] is not None
            else None
        )
        latest_model_threshold = thresholds_by_run.get(latest_model_run_id) if latest_model_run_id else None
        latest_model_score = float(image_row["latest_model_score"]) if image_row["latest_model_score"] is not None else None
        latest_model_distance_to_threshold = (
            abs(latest_model_score - latest_model_threshold)
            if latest_model_score is not None and latest_model_threshold is not None
            else None
        )
        disagreement_flags: list[str] = []
        if human_label and human_label != "unclear":
            if latest_model_predicted_label and human_label != latest_model_predicted_label:
                disagreement_flags.append("human_vs_model")

        review_items.append(
            ReviewItem(
                sha256=sha256,
                label=human_label,
                label_source=str(image_row["label_source"]) if image_row["label_source"] is not None else None,
                local_path=str(image_row["local_path"]),
                byte_size=int(image_row["byte_size"]) if image_row["byte_size"] is not None else None,
                width=int(image_row["width"]) if image_row["width"] is not None else None,
                height=int(image_row["height"]) if image_row["height"] is not None else None,
                handles=handles,
                display_names=display_names,
                source_surfaces=source_surfaces,
                seen_count=seen_count,
                whitelisted=whitelisted,
                max_model_score=latest_model_score,
                latest_model_predicted_label=latest_model_predicted_label,
                latest_model_run_id=latest_model_run_id,
                latest_model_threshold=latest_model_threshold,
                latest_model_distance_to_threshold=latest_model_distance_to_threshold,
                disagreement_flags=disagreement_flags,
                labeled_at=str(image_row["labeled_at"]) if image_row["labeled_at"] is not None else None,
                example_profile_url=example_profile_url,
                example_notification_url=example_notification_url,
                example_tweet_url=example_tweet_url,
                last_seen_at=last_seen_at,
                image_url_count=len(related),
            )
        )

    return review_items


def queue_items(items: list[ReviewItem], queue_name: str) -> list[ReviewItem]:
    if queue_name not in REVIEW_QUEUES:
        raise ValueError(f"Unsupported review queue: {queue_name}")

    def needs_review(item: ReviewItem) -> bool:
        return item.label_source != "manual"

    if queue_name == "needs_review":
        filtered = [item for item in items if needs_review(item)]
        return sorted(
            filtered,
            key=lambda item: (
                item.seen_count,
                item.last_seen_at or "",
                item.max_model_score if item.max_model_score is not None else -1.0,
            ),
            reverse=True,
        )

    if queue_name == "model_disagreements":
        return sorted(
            (item for item in items if "human_vs_model" in item.disagreement_flags),
            key=lambda item: (
                item.max_model_score if item.max_model_score is not None else -1.0,
                item.seen_count,
                item.labeled_at or "",
            ),
            reverse=True,
        )

    if queue_name == "exempted":
        return sorted((item for item in items if item.whitelisted), key=lambda item: item.seen_count, reverse=True)

    if queue_name == "high_impact":
        return sorted(items, key=lambda item: item.seen_count, reverse=True)

    if queue_name == "notifications":
        return sorted(
            (item for item in items if "notification-group" in item.source_surfaces),
            key=lambda item: item.seen_count,
            reverse=True,
        )

    if queue_name == "boundary_review":
        return sorted(
            (
                item
                for item in items
                if needs_review(item)
                and item.max_model_score is not None
                and item.latest_model_threshold is not None
                and item.latest_model_distance_to_threshold is not None
            ),
            key=lambda item: (
                item.latest_model_distance_to_threshold if item.latest_model_distance_to_threshold is not None else float("inf"),
                -(item.seen_count),
            ),
        )

    if queue_name == "model_backlog":
        return sorted(
            (
                item
                for item in items
                if needs_review(item) and item.max_model_score is not None
            ),
            key=lambda item: item.max_model_score if item.max_model_score is not None else -1.0,
            reverse=True,
        )

    return sorted(
        (
            item
            for item in items
            if item.label == "not_milady" and item.max_model_score is not None
        ),
        key=lambda item: item.max_model_score if item.max_model_score is not None else -1.0,
        reverse=True,
    )


def load_model_thresholds(run_ids: set[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for run_id in sorted(run_ids):
        summary_path = MODEL_RUN_ROOT / run_id / "summary.json"
        if not summary_path.exists():
            continue
        try:
            payload = load_json(summary_path, RunSummary)
            thresholds[run_id] = float(payload.threshold)
        except (OSError, msgspec.DecodeError, TypeError, ValueError):
            continue
    return thresholds


def load_review_run_ids(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT run_id, MAX(created_at) AS latest_created_at
        FROM model_scores
        GROUP BY run_id
        ORDER BY latest_created_at DESC, run_id DESC
        """
    ).fetchall()
    return [str(row["run_id"]) for row in rows]


def labeled_grid_items(items: list[ReviewItem], filter_name: str) -> list[ReviewItem]:
    if filter_name not in LABELED_GRID_FILTERS:
        raise ValueError(f"Unsupported labeled grid filter: {filter_name}")

    filtered = [item for item in items if item.label is not None]
    if filter_name != "all":
        filtered = [item for item in filtered if item.label == filter_name]

    return sorted(
        filtered,
        key=lambda item: (
            len(item.disagreement_flags),
            item.max_model_score if item.max_model_score is not None else -1.0,
            item.seen_count,
            item.labeled_at or "",
        ),
        reverse=True,
    )
