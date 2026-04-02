from __future__ import annotations

import sqlite3
from pathlib import Path

from .paths import CATALOG_PATH, OFFLINE_CACHE_PATH, ensure_layout, now_iso, resolve_repo_path

LABELS = ("milady", "not_milady", "unclear")


def merge_string_lists(left: list[str], right: list[str]) -> list[str]:
    return sorted({entry for entry in [*left, *right] if entry})


def bool_from_db(value: object | None) -> bool:
    return bool(value) if value is not None else False


def coalesce_latest(existing: str | None, incoming: str | None) -> str | None:
    return incoming or existing


def min_timestamp(left: str | None, right: str | None) -> str:
    candidates = [value for value in (left, right) if value]
    return min(candidates) if candidates else now_iso()


def max_timestamp(left: str | None, right: str | None) -> str:
    candidates = [value for value in (left, right) if value]
    return max(candidates) if candidates else now_iso()


def normalize_label(value: str | None) -> str | None:
    if value in LABELS:
        return value
    return None


def connect_db(path: Path = CATALOG_PATH) -> sqlite3.Connection:
    ensure_layout()
    resolved_path = resolve_repo_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(resolved_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    init_db(connection)
    return connection


def connect_offline_cache_db(path: Path = OFFLINE_CACHE_PATH) -> sqlite3.Connection:
    ensure_layout()
    resolved_path = resolve_repo_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(resolved_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    init_offline_cache_db(connection)
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS exports (
          export_path TEXT PRIMARY KEY,
          export_name TEXT NOT NULL,
          exported_at TEXT,
          ingested_at TEXT NOT NULL,
          version INTEGER,
          avatar_count INTEGER NOT NULL,
          total_sightings INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS avatar_urls (
          normalized_url TEXT PRIMARY KEY,
          original_url TEXT NOT NULL,
          handles_json TEXT NOT NULL,
          display_names_json TEXT NOT NULL,
          source_surfaces_json TEXT NOT NULL,
          seen_count INTEGER NOT NULL,
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          example_profile_url TEXT,
          example_notification_url TEXT,
          example_tweet_url TEXT,
          whitelisted INTEGER NOT NULL DEFAULT 0,
          image_sha256 TEXT,
          download_status TEXT NOT NULL DEFAULT 'pending',
          last_download_error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(image_sha256) REFERENCES images(sha256)
        );

        CREATE TABLE IF NOT EXISTS images (
          sha256 TEXT PRIMARY KEY,
          local_path TEXT NOT NULL,
          mime_type TEXT,
          width INTEGER,
          height INTEGER,
          byte_size INTEGER,
          split TEXT,
          label TEXT,
          label_source TEXT,
          labeled_at TEXT,
          review_notes TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_scores (
          run_id TEXT NOT NULL,
          image_sha256 TEXT NOT NULL,
          score REAL NOT NULL,
          predicted_label TEXT NOT NULL,
          split TEXT,
          created_at TEXT NOT NULL,
          PRIMARY KEY(run_id, image_sha256),
          FOREIGN KEY(image_sha256) REFERENCES images(sha256)
        );

        CREATE TABLE IF NOT EXISTS label_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          image_sha256 TEXT NOT NULL,
          previous_label TEXT,
          previous_label_source TEXT,
          previous_labeled_at TEXT,
          previous_review_notes TEXT,
          new_label TEXT NOT NULL,
          new_review_notes TEXT,
          created_at TEXT NOT NULL,
          FOREIGN KEY(image_sha256) REFERENCES images(sha256)
        );

        CREATE INDEX IF NOT EXISTS idx_avatar_urls_image_sha256 ON avatar_urls(image_sha256);
        CREATE INDEX IF NOT EXISTS idx_avatar_urls_download_status ON avatar_urls(download_status);
        CREATE INDEX IF NOT EXISTS idx_images_label ON images(label);
        CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);
        CREATE INDEX IF NOT EXISTS idx_model_scores_image_sha256 ON model_scores(image_sha256);
        CREATE INDEX IF NOT EXISTS idx_label_events_created_at ON label_events(created_at DESC, id DESC);
        """
    )
    ensure_column(connection, "label_events", "batch_id", "TEXT")
    connection.commit()


def init_offline_cache_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS file_fingerprints (
          path TEXT PRIMARY KEY,
          file_size INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          image_size INTEGER NOT NULL,
          raw_sha TEXT NOT NULL,
          pixel_digest TEXT NOT NULL,
          width INTEGER,
          height INTEGER,
          readable INTEGER NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_file_fingerprints_raw_sha ON file_fingerprints(raw_sha);
        CREATE INDEX IF NOT EXISTS idx_file_fingerprints_pixel_digest ON file_fingerprints(pixel_digest);
        """
    )
    connection.commit()


def ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_definition: str) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in columns:
        return
    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
