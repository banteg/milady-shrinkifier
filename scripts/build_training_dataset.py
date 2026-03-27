from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from mobilenet_common import DatasetEntry, dataset_entries_to_jsonl, deterministic_split_ids
from pipeline_common import OFFICIAL_IMAGE_ROOT, SPLIT_ROOT, connect_db, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize train/val/test JSONL files for MobileNetV3-Small training.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = connect_db()

    official_paths = [
        path
        for path in sorted(OFFICIAL_IMAGE_ROOT.glob("*.png"))
        if path.is_file() and path.stat().st_size > 0 and path.stem.isdigit() and is_readable_image(path)
    ]
    official_assignments = deterministic_split_ids([path.stem for path in official_paths], (args.train_ratio, args.val_ratio, args.test_ratio))

    official_entries = [
        DatasetEntry(
            sample_id=f"official:{path.stem}",
            path=path,
            label="milady",
            source="official",
            split=official_assignments[path.stem],
        )
        for path in official_paths
    ]

    exported_rows = connection.execute(
        """
        SELECT sha256, local_path, label
        FROM images
        WHERE label IN ('milady', 'not_milady')
          AND local_path IS NOT NULL
        ORDER BY sha256 ASC
        """
    ).fetchall()
    exported_assignments = deterministic_split_ids(
        [str(row["sha256"]) for row in exported_rows],
        (args.train_ratio, args.val_ratio, args.test_ratio),
    )

    exported_entries = [
        DatasetEntry(
            sample_id=f"export:{row['sha256']}",
            path=resolve_repo_path(str(row["local_path"])),
            label=str(row["label"]),
            source="export",
            split=exported_assignments[str(row["sha256"])],
        )
        for row in exported_rows
        if resolve_repo_path(str(row["local_path"])).exists()
    ]

    for row in exported_rows:
        split = exported_assignments[str(row["sha256"])]
        connection.execute(
            "UPDATE images SET split = ?, updated_at = CURRENT_TIMESTAMP WHERE sha256 = ?",
            (split, str(row["sha256"])),
        )
    connection.commit()

    by_split = {
        "train": [entry for entry in [*official_entries, *exported_entries] if entry.split == "train"],
        "val": [entry for entry in [*official_entries, *exported_entries] if entry.split == "val"],
        "test": [entry for entry in [*official_entries, *exported_entries] if entry.split == "test"],
    }

    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    for split_name, entries in by_split.items():
        dataset_entries_to_jsonl(entries, SPLIT_ROOT / f"{split_name}.jsonl")

    summary = {
        split_name: {
            "total": len(entries),
            "milady": sum(1 for entry in entries if entry.label == "milady"),
            "not_milady": sum(1 for entry in entries if entry.label == "not_milady"),
        }
        for split_name, entries in by_split.items()
    }
    (SPLIT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
