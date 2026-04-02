from __future__ import annotations

import hashlib
import mimetypes
import os
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import msgspec
import numpy as np
from PIL import Image

from .paths import INFERENCE_VARIANT_ROOT, now_iso, resolve_repo_path


class FileFingerprint(msgspec.Struct, kw_only=True):
    path: str
    file_size: int
    mtime_ns: int
    image_size: int
    raw_sha: str
    pixel_digest: str
    width: int | None
    height: int | None
    readable: bool
    updated_at: str


def guess_extension(content_type: str | None, url: str) -> str:
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed
    suffix = Path(url).suffix
    return suffix if suffix else ".img"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def convert_image_to_rgb(image: Image.Image) -> Image.Image:
    transparency = image.info.get("transparency")
    if image.mode == "P" and isinstance(transparency, bytes):
        return image.convert("RGBA").convert("RGB")
    return image.convert("RGB")


def inspect_image_bytes(payload: bytes) -> tuple[int, int, str | None]:
    with Image.open(BytesIO(payload)) as image:
        width, height = image.size
        mime_type = Image.MIME.get(image.format) if image.format is not None else None
    return width, height, mime_type


def get_file_fingerprint(connection: sqlite3.Connection, path: Path, image_size: int) -> FileFingerprint:
    resolved_path = resolve_repo_path(path)
    stat_result = resolved_path.stat()
    cache_row = connection.execute(
        """
        SELECT *
        FROM file_fingerprints
        WHERE path = ?
          AND file_size = ?
          AND mtime_ns = ?
          AND image_size = ?
        """,
        (str(resolved_path), stat_result.st_size, stat_result.st_mtime_ns, image_size),
    ).fetchone()
    if cache_row is not None:
        return row_to_file_fingerprint(cache_row)

    updated_at = now_iso()
    readable = True
    raw_sha = ""
    pixel_digest = ""
    width: int | None = None
    height: int | None = None
    try:
        payload = resolved_path.read_bytes()
        raw_sha = sha256_bytes(payload)
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            prepared_source = convert_image_to_rgb(image)
            prepared = prepared_source.resize((image_size, image_size), Image.Resampling.BICUBIC)
            pixel_digest = sha256_bytes(prepared.tobytes())
    except Exception:  # noqa: BLE001
        readable = False

    connection.execute(
        """
        INSERT INTO file_fingerprints (
          path,
          file_size,
          mtime_ns,
          image_size,
          raw_sha,
          pixel_digest,
          width,
          height,
          readable,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          file_size = excluded.file_size,
          mtime_ns = excluded.mtime_ns,
          image_size = excluded.image_size,
          raw_sha = excluded.raw_sha,
          pixel_digest = excluded.pixel_digest,
          width = excluded.width,
          height = excluded.height,
          readable = excluded.readable,
          updated_at = excluded.updated_at
        """,
        (
            str(resolved_path),
            stat_result.st_size,
            stat_result.st_mtime_ns,
            image_size,
            raw_sha,
            pixel_digest,
            width,
            height,
            1 if readable else 0,
            updated_at,
        ),
    )
    return FileFingerprint(
        path=str(resolved_path),
        file_size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        image_size=image_size,
        raw_sha=raw_sha,
        pixel_digest=pixel_digest,
        width=width,
        height=height,
        readable=readable,
        updated_at=updated_at,
    )


def row_to_file_fingerprint(row: sqlite3.Row) -> FileFingerprint:
    return FileFingerprint(
        path=str(row["path"]),
        file_size=int(row["file_size"]),
        mtime_ns=int(row["mtime_ns"]),
        image_size=int(row["image_size"]),
        raw_sha=str(row["raw_sha"]),
        pixel_digest=str(row["pixel_digest"]),
        width=int(row["width"]) if row["width"] is not None else None,
        height=int(row["height"]) if row["height"] is not None else None,
        readable=bool(row["readable"]),
        updated_at=str(row["updated_at"]),
    )


def inference_variant_cache_path(raw_sha: str) -> Path:
    directory = INFERENCE_VARIANT_ROOT / raw_sha[:2]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{raw_sha}.npz"


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(cast(Any, handle), **cast(Any, arrays))
    temporary_path.replace(path)
