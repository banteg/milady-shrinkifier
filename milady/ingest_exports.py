from __future__ import annotations

import argparse
import shutil

from .pipeline_common import (
    INGEST_ROOT,
    EXPORT_ROOT,
    connect_db,
    coalesce_latest,
    discover_export_paths,
    encode_json_list,
    max_timestamp,
    merge_string_lists,
    min_timestamp,
    now_iso,
    parse_json_list,
)
from .wire import IngestExportPayload, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Milady Shrinkifier avatar exports into the local SQLite catalog.")
    parser.add_argument("inputs", nargs="*", help="Export JSON files to ingest. Defaults to cache/ingest/*.json")
    parser.add_argument("--copy-into-cache", action="store_true", help="Copy each ingested export into cache/exports/raw/ before recording it.")
    parser.add_argument("--force", action="store_true", help="Re-ingest exports even if they were previously seen by path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = connect_db()
    export_paths = discover_export_paths(args.inputs)
    if not export_paths:
        raise SystemExit("No export files found to ingest.")

    ingested = 0
    skipped = 0
    merged_avatars = 0

    for export_path in export_paths:
        source_path = export_path.resolve()
        should_archive = args.copy_into_cache or source_path.parent == INGEST_ROOT.resolve()
        stored_path = EXPORT_ROOT / export_path.name if should_archive else source_path
        if should_archive:
            stored_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path != stored_path.resolve():
                shutil.copy2(source_path, stored_path)
        export_record_path = str(stored_path.resolve())
        source_record_path = str(source_path)

        existing_exports = connection.execute(
            "SELECT export_path FROM exports WHERE export_path IN (?, ?)",
            (export_record_path, source_record_path),
        ).fetchall()
        existing_paths = {str(row["export_path"]) for row in existing_exports}
        existing_export = export_record_path in existing_paths
        existing_source_export = source_record_path in existing_paths

        if existing_export and not args.force:
            skipped += 1
            continue
        if not args.force and should_archive and existing_source_export and source_record_path != export_record_path:
            connection.execute(
                """
                UPDATE exports
                SET export_path = ?, export_name = ?
                WHERE export_path = ?
                """,
                (export_record_path, stored_path.name, source_record_path),
            )
            connection.commit()
            skipped += 1
            continue

        payload = load_json(source_path, IngestExportPayload)
        avatars = payload.avatars

        if args.force and existing_paths:
            connection.executemany("DELETE FROM exports WHERE export_path = ?", ((path,) for path in existing_paths))

        for avatar in avatars:
            normalized_url = avatar.normalized_url
            existing = connection.execute(
                "SELECT * FROM avatar_urls WHERE normalized_url = ?",
                (normalized_url,),
            ).fetchone()
            now = now_iso()

            incoming_handles = avatar.handles
            incoming_display_names = avatar.display_names
            incoming_source_surfaces = avatar.source_surfaces

            if existing:
                merged_handles = merge_string_lists(parse_json_list(existing["handles_json"]), incoming_handles)
                merged_display_names = merge_string_lists(parse_json_list(existing["display_names_json"]), incoming_display_names)
                merged_sources = merge_string_lists(parse_json_list(existing["source_surfaces_json"]), incoming_source_surfaces)

                connection.execute(
                    """
                    UPDATE avatar_urls
                    SET original_url = ?,
                        handles_json = ?,
                        display_names_json = ?,
                        source_surfaces_json = ?,
                        seen_count = ?,
                        first_seen_at = ?,
                        last_seen_at = ?,
                        example_profile_url = ?,
                        example_notification_url = ?,
                        example_tweet_url = ?,
                        whitelisted = ?,
                        updated_at = ?
                    WHERE normalized_url = ?
                    """,
                    (
                        avatar.original_url,
                        encode_json_list(merged_handles),
                        encode_json_list(merged_display_names),
                        encode_json_list(merged_sources),
                        int(existing["seen_count"]) + avatar.seen_count,
                        min_timestamp(existing["first_seen_at"], avatar.first_seen_at),
                        max_timestamp(existing["last_seen_at"], avatar.last_seen_at),
                        coalesce_latest(existing["example_profile_url"], avatar.example_profile_url),
                        coalesce_latest(existing["example_notification_url"], avatar.example_notification_url),
                        coalesce_latest(existing["example_tweet_url"], avatar.example_tweet_url),
                        1 if (bool(existing["whitelisted"]) or avatar.whitelisted) else 0,
                        now,
                        normalized_url,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO avatar_urls (
                      normalized_url,
                      original_url,
                      handles_json,
                      display_names_json,
                      source_surfaces_json,
                      seen_count,
                      first_seen_at,
                      last_seen_at,
                      example_profile_url,
                      example_notification_url,
                      example_tweet_url,
                      whitelisted,
                      created_at,
                      updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_url,
                        avatar.original_url,
                        encode_json_list(incoming_handles),
                        encode_json_list(incoming_display_names),
                        encode_json_list(incoming_source_surfaces),
                        avatar.seen_count,
                        avatar.first_seen_at,
                        avatar.last_seen_at,
                        avatar.example_profile_url,
                        avatar.example_notification_url,
                        avatar.example_tweet_url,
                        1 if avatar.whitelisted else 0,
                        now,
                        now,
                    ),
                )
            merged_avatars += 1

        connection.execute(
            """
            INSERT INTO exports (
              export_path,
              export_name,
              exported_at,
              ingested_at,
              version,
              avatar_count,
              total_sightings
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                export_record_path,
                stored_path.name,
                payload.exported_at,
                now_iso(),
                payload.version,
                payload.avatar_count,
                payload.total_sightings,
            ),
        )
        connection.commit()
        ingested += 1

    print(
        f"Ingested {ingested} export(s), skipped {skipped}, and merged {merged_avatars} avatar records into {connection.execute('SELECT COUNT(*) FROM avatar_urls').fetchone()[0]} catalog rows."
    )


if __name__ == "__main__":
    main()
