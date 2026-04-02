from __future__ import annotations

import argparse
import threading
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

import msgspec
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .mobilenet_common import MANUAL_LABEL_SOURCE
from .pipeline_common import (
    CATALOG_PATH,
    LABELS,
    LABELED_GRID_FILTERS,
    REVIEW_QUEUES,
    ReviewItem,
    connect_db,
    labeled_grid_items,
    load_review_items,
    load_review_run_ids,
    queue_items,
    resolve_repo_path,
)
from .wire import (
    ReviewBatchLabelRequest,
    ReviewBatchLabelResponse,
    ReviewBatchPayload,
    ReviewChangedResponse,
    ReviewGridPayload,
    ReviewHistoryEntry,
    ReviewHistoryPayload,
    ReviewItemResponse,
    ReviewLabelRequest,
    ReviewQueuePayload,
    ReviewSummaryPayload,
    ReviewUndoBatchResponse,
    ReviewUndoSingleResponse,
    decode_json,
    encode_json,
)


REVIEW_STATIC_ROOT = Path(__file__).resolve().with_name("review_static")
REVIEW_INDEX_PATH = REVIEW_STATIC_ROOT / "review.html"
REVIEW_ASSET_ROOT = REVIEW_STATIC_ROOT / "assets"


class ReviewSnapshot(msgspec.Struct, kw_only=True):
    catalog_path: str
    selected_run_id: str | None
    available_run_ids: list[str]
    items: list[ReviewItem]
    items_by_sha: dict[str, ReviewItem]
    queue_lists: dict[str, list[ReviewItem]]
    labeled_lists: dict[str, list[ReviewItem]]
    image_paths: dict[str, Path]
    label_counts: dict[str, int]
    needs_review: int
    recent_events: list[dict[str, Any]]
    can_undo: bool


class ReviewState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[str | None, ReviewSnapshot] = {}

    def get(self, run_id: str | None = None) -> ReviewSnapshot:
        snapshot = self._snapshots.get(run_id)
        if snapshot is not None:
            return snapshot
        with self._lock:
            snapshot = self._snapshots.get(run_id)
            if snapshot is None:
                snapshot = self._build_snapshot(run_id)
                self._snapshots[run_id] = snapshot
            return snapshot

    def refresh(self) -> ReviewSnapshot:
        with self._lock:
            self._snapshots.clear()
            snapshot = self._build_snapshot(None)
            self._snapshots[None] = snapshot
            return snapshot

    def _build_snapshot(self, requested_run_id: str | None) -> ReviewSnapshot:
        with closing(connect_db()) as connection:
            available_run_ids = load_review_run_ids(connection)
            if requested_run_id is not None and requested_run_id not in available_run_ids:
                raise KeyError(requested_run_id)
            selected_run_id = requested_run_id or (available_run_ids[0] if available_run_ids else None)
            items = load_review_items(connection, selected_run_id)
            items_by_sha = {item.sha256: item for item in items}
            queue_lists = {queue_name: queue_items(items, queue_name) for queue_name in REVIEW_QUEUES}
            labeled_lists = {filter_name: labeled_grid_items(items, filter_name) for filter_name in LABELED_GRID_FILTERS}
            image_paths = {item.sha256: resolve_repo_path(item.local_path) for item in items}
            label_counts: dict[str, int] = {label: 0 for label in LABELS}
            needs_review = 0
            for item in items:
                if item.label_source != MANUAL_LABEL_SOURCE:
                    needs_review += 1
                elif item.label in label_counts:
                    label_counts[item.label] += 1
            recent_events = [dict(row) for row in recent_label_events(connection, 200)]

        return ReviewSnapshot(
            catalog_path=str(CATALOG_PATH),
            selected_run_id=selected_run_id,
            available_run_ids=available_run_ids,
            items=items,
            items_by_sha=items_by_sha,
            queue_lists=queue_lists,
            labeled_lists=labeled_lists,
            image_paths=image_paths,
            label_counts=label_counts,
            needs_review=needs_review,
            recent_events=recent_events,
            can_undo=bool(recent_events),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local avatar review app for labeling Milady classifier data.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser.parse_args()


app = FastAPI(title="Milady Shrinkifier Review")
app.mount("/assets", StaticFiles(directory=str(REVIEW_ASSET_ROOT), check_dir=False), name="review-assets")


def require_snapshot(run_id: str | None = None) -> ReviewSnapshot:
    try:
        return STATE.get(run_id)
    except KeyError as error:
        raise HTTPException(status_code=400, detail=f"Unknown review run id: {error.args[0]}") from error


def require_queue_name(queue_name: str) -> str:
    if queue_name not in REVIEW_QUEUES:
        raise HTTPException(status_code=400, detail=f"Unsupported review queue: {queue_name}")
    return queue_name


def require_labeled_filter(filter_name: str) -> str:
    if filter_name not in LABELED_GRID_FILTERS:
        raise HTTPException(status_code=400, detail=f"Unsupported labeled grid filter: {filter_name}")
    return filter_name


def index_payload(snapshot: ReviewSnapshot, queue_name: str, index: int) -> ReviewQueuePayload:
    items = snapshot.queue_lists[queue_name]
    if not items:
        return ReviewQueuePayload(queue=queue_name, index=0, total=0, item=None)
    bounded_index = min(index, len(items) - 1)
    return ReviewQueuePayload(
        queue=queue_name,
        index=bounded_index,
        total=len(items),
        item=items[bounded_index],
    )


def json_response(payload: Any, status_code: int = 200) -> Response:
    return Response(content=encode_json(payload), media_type="application/json", status_code=status_code)


@app.get("/", response_model=None)
def root():
    if REVIEW_INDEX_PATH.exists():
        return FileResponse(REVIEW_INDEX_PATH, media_type="text/html")
    return HTMLResponse(
        "<h1>Review frontend not built</h1><p>Run <code>pnpm run build:review</code> and reload.</p>",
        status_code=503,
    )


@app.get("/api/summary")
def summary(run_id: str | None = Query(None)) -> Response:
    snapshot = require_snapshot(run_id)
    counts = {queue_name: len(snapshot.queue_lists[queue_name]) for queue_name in REVIEW_QUEUES}
    return json_response(
        ReviewSummaryPayload(
            catalog_path=snapshot.catalog_path,
            selected_run_id=snapshot.selected_run_id,
            available_run_ids=snapshot.available_run_ids,
            total_images=len(snapshot.items),
            queue_counts=counts,
            label_counts=snapshot.label_counts,
            needs_review=snapshot.needs_review,
            can_undo=snapshot.can_undo,
        )
    )


@app.get("/api/queue")
def get_queue(
    queue: str = Query("needs_review"),
    index: int = Query(0, ge=0),
    run_id: str | None = Query(None),
) -> Response:
    snapshot = require_snapshot(run_id)
    queue_name = require_queue_name(queue)
    return json_response(index_payload(snapshot, queue_name, index))


@app.get("/api/batch")
def get_batch(
    queue: str = Query("needs_review"),
    limit: int = Query(9, ge=1, le=25),
    offset: int = Query(0, ge=0),
    run_id: str | None = Query(None),
) -> Response:
    snapshot = require_snapshot(run_id)
    queue_name = require_queue_name(queue)
    items = snapshot.queue_lists[queue_name]
    bounded_offset = min(offset, max(0, len(items) - 1)) if items else 0
    sliced_items = items[bounded_offset : bounded_offset + limit]
    return json_response(
        ReviewBatchPayload(
            queue=queue_name,
            total=len(items),
            offset=bounded_offset,
            items=sliced_items,
        )
    )


@app.get("/api/item/{sha256}")
def get_item(sha256: str, run_id: str | None = Query(None)) -> Response:
    snapshot = require_snapshot(run_id)
    item = snapshot.items_by_sha.get(sha256)
    if item is None:
        raise HTTPException(status_code=404, detail="Review item not found")
    return json_response(ReviewItemResponse(item=item))


@app.get("/api/history")
def get_history(limit: int = Query(24, ge=1, le=100), run_id: str | None = Query(None)) -> Response:
    snapshot = require_snapshot(run_id)
    history: list[ReviewHistoryEntry] = []
    for event in snapshot.recent_events[:limit]:
        item = snapshot.items_by_sha.get(str(event["image_sha256"]))
        history.append(
            ReviewHistoryEntry(
                event_id=int(event["id"]),
                sha256=str(event["image_sha256"]),
                created_at=str(event["created_at"]),
                new_label=str(event["new_label"]),
                previous_label=event["previous_label"],
                item=item,
            )
        )
    return json_response(ReviewHistoryPayload(history=history))


@app.get("/api/labeled-grid")
def get_labeled_grid(
    filter_name: str = Query("all"),
    limit: int | None = Query(None, ge=1),
    run_id: str | None = Query(None),
) -> Response:
    snapshot = require_snapshot(run_id)
    selected_filter = require_labeled_filter(filter_name)
    items = snapshot.labeled_lists[selected_filter]
    sliced_items = items[:limit] if limit is not None else items
    return json_response(
        ReviewGridPayload(
            total=len(items),
            items=sliced_items,
        )
    )


@app.get("/api/queue-grid")
def get_queue_grid(
    queue: str = Query("needs_review"),
    limit: int | None = Query(None, ge=1),
    run_id: str | None = Query(None),
) -> Response:
    snapshot = require_snapshot(run_id)
    queue_name = require_queue_name(queue)
    items = snapshot.queue_lists[queue_name]
    sliced_items = items[:limit] if limit is not None else items
    return json_response(
        ReviewGridPayload(
            total=len(items),
            items=sliced_items,
        )
    )


@app.post("/api/label")
async def label_avatar(request: Request) -> Response:
    payload = decode_request_body(await request.body(), ReviewLabelRequest)
    if payload.label not in LABELS:
        raise HTTPException(status_code=400, detail=f"Unsupported label: {payload.label}")

    with closing(connect_db()) as connection:
        existing = connection.execute(
            """
            SELECT sha256, label, label_source, labeled_at, review_notes
            FROM images
            WHERE sha256 = ?
            """,
            (payload.sha256,),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail=f"Unknown avatar sha256: {payload.sha256}")

        if (
            existing["label"] == payload.label
            and existing["review_notes"] == payload.note
            and existing["label_source"] == MANUAL_LABEL_SOURCE
        ):
            return json_response(ReviewChangedResponse(ok=True, changed=False))

        connection.execute(
            """
            INSERT INTO label_events (
              image_sha256,
              previous_label,
              previous_label_source,
              previous_labeled_at,
              previous_review_notes,
              new_label,
              new_review_notes,
              batch_id,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                payload.sha256,
                existing["label"],
                existing["label_source"],
                existing["labeled_at"],
                existing["review_notes"],
                payload.label,
                payload.note,
                None,
            ),
        )
        connection.execute(
            """
            UPDATE images
            SET label = ?,
                label_source = ?,
                labeled_at = CURRENT_TIMESTAMP,
                review_notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE sha256 = ?
            """,
            (payload.label, MANUAL_LABEL_SOURCE, payload.note, payload.sha256),
        )
        connection.commit()

    STATE.refresh()
    return json_response(ReviewChangedResponse(ok=True, changed=True))


@app.post("/api/batch-label")
async def batch_label(request: Request) -> Response:
    payload = decode_request_body(await request.body(), ReviewBatchLabelRequest)
    if not payload.items:
        raise HTTPException(status_code=400, detail="Batch is empty")

    batch_id = str(uuid.uuid4())
    changed_count = 0
    with closing(connect_db()) as connection:
        for item in payload.items:
            if item.label not in LABELS:
                raise HTTPException(status_code=400, detail=f"Unsupported label: {item.label}")
            existing = connection.execute(
                """
                SELECT sha256, label, label_source, labeled_at, review_notes
                FROM images
                WHERE sha256 = ?
                """,
                (item.sha256,),
            ).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail=f"Unknown avatar sha256: {item.sha256}")

            if existing["label"] == item.label and existing["label_source"] == MANUAL_LABEL_SOURCE:
                continue

            connection.execute(
                """
                INSERT INTO label_events (
                  image_sha256,
                  previous_label,
                  previous_label_source,
                  previous_labeled_at,
                  previous_review_notes,
                  new_label,
                  new_review_notes,
                  batch_id,
                  created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    item.sha256,
                    existing["label"],
                    existing["label_source"],
                    existing["labeled_at"],
                    existing["review_notes"],
                    item.label,
                    None,
                    batch_id,
                ),
            )
            connection.execute(
                """
            UPDATE images
            SET label = ?,
                label_source = ?,
                labeled_at = CURRENT_TIMESTAMP,
                review_notes = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE sha256 = ?
            """,
                (item.label, MANUAL_LABEL_SOURCE, item.sha256),
            )
            changed_count += 1

        connection.commit()

    if changed_count > 0:
        STATE.refresh()
    return json_response(ReviewBatchLabelResponse(ok=True, batch_id=batch_id, count=changed_count))


@app.post("/api/undo")
def undo_last_label() -> Response:
    with closing(connect_db()) as connection:
        event = latest_label_event(connection)
        if event is None:
            raise HTTPException(status_code=409, detail="No label action to undo")

        batch_id = event["batch_id"]
        if batch_id:
            events = connection.execute(
                """
                SELECT *
                FROM label_events
                WHERE batch_id = ?
                ORDER BY id DESC
                """,
                (batch_id,),
            ).fetchall()
            undone_sha256 = []
            for batch_event in events:
                connection.execute(
                    """
                    UPDATE images
                    SET label = ?,
                        label_source = ?,
                        labeled_at = ?,
                        review_notes = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE sha256 = ?
                    """,
                    (
                        batch_event["previous_label"],
                        batch_event["previous_label_source"],
                        batch_event["previous_labeled_at"],
                        batch_event["previous_review_notes"],
                        batch_event["image_sha256"],
                    ),
                )
                undone_sha256.append(str(batch_event["image_sha256"]))
            connection.execute("DELETE FROM label_events WHERE batch_id = ?", (batch_id,))
            connection.commit()
            STATE.refresh()
            return json_response(
                ReviewUndoBatchResponse(
                    ok=True,
                    batch_id=str(batch_id),
                    undone_sha256_list=undone_sha256,
                    undone_sha256=undone_sha256[0] if undone_sha256 else None,
                )
            )

        connection.execute(
            """
            UPDATE images
            SET label = ?,
                label_source = ?,
                labeled_at = ?,
                review_notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE sha256 = ?
            """,
            (
                event["previous_label"],
                event["previous_label_source"],
                event["previous_labeled_at"],
                event["previous_review_notes"],
                event["image_sha256"],
            ),
        )
        connection.execute("DELETE FROM label_events WHERE id = ?", (event["id"],))
        connection.commit()
        undone_sha = str(event["image_sha256"])

    snapshot = STATE.refresh()
    item = snapshot.items_by_sha.get(undone_sha)
    return json_response(
        ReviewUndoSingleResponse(
            ok=True,
            undone_sha256=undone_sha,
            item=item,
        )
    )


@app.get("/api/image/{sha256}")
def get_image(sha256: str) -> FileResponse:
    snapshot = require_snapshot()
    path = snapshot.image_paths.get(sha256)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(path)


def recent_label_events(connection, limit: int) -> list[Any]:
    return connection.execute(
        """
        SELECT id, image_sha256, previous_label, new_label, created_at, batch_id
        FROM label_events
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def latest_label_event(connection):
    rows = recent_label_events(connection, 1)
    return rows[0] if rows else None


def decode_request_body(payload: bytes, type_: type[Any]) -> Any:
    try:
        return decode_json(payload, type_)
    except msgspec.DecodeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


STATE = ReviewState()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
