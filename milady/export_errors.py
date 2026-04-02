from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .paths import MODEL_COMPARE_ROOT
from .wire import CompareErrorItem, CompareSummary, dump_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compare false positives/negatives as browsable image folders.")
    parser.add_argument(
        "--compare-dir",
        type=Path,
        help="Compare output directory. Defaults to the newest directory under cache/models/.../compare.",
    )
    parser.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        help="Optional run id filter. Pass multiple times to export a subset of runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory. Defaults to <compare-dir>/exported_errors.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize images. Defaults to hardlink with copy fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_dir = resolve_compare_dir(args.compare_dir)
    summary_path = compare_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Compare summary not found: {summary_path}")

    summary = load_json(summary_path, CompareSummary)
    runs = summary.runs

    selected_run_ids = [run_id for run_id in runs.keys() if not args.run_ids or run_id in args.run_ids]
    if not selected_run_ids:
        raise SystemExit("No matching runs found in compare summary.")

    output_dir = args.output_dir or compare_dir / "exported_errors"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for run_id in selected_run_ids:
        run_summary = runs[run_id]
        false_positive_path = Path(run_summary.false_positives_path)
        false_negative_path = Path(run_summary.false_negatives_path)
        exported += export_error_set(run_id, "false_positives", false_positive_path, output_dir, args.mode)
        exported += export_error_set(run_id, "false_negatives", false_negative_path, output_dir, args.mode)

    print(f"Exported {exported} image(s) to {output_dir}")


def resolve_compare_dir(compare_dir: Path | None) -> Path:
    if compare_dir is not None:
        return compare_dir.resolve()
    candidates = [
        path
        for path in MODEL_COMPARE_ROOT.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    ]
    if not candidates:
        raise SystemExit("No compare output directories found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export_error_set(
    run_id: str,
    category: str,
    manifest_path: Path,
    output_dir: Path,
    mode: str,
) -> int:
    items = load_json(manifest_path, list[CompareErrorItem])

    category_dir = output_dir / run_id / category
    category_dir.mkdir(parents=True, exist_ok=True)
    dump_json(category_dir / "manifest.json", items)

    exported = 0
    for index, item in enumerate(items, start=1):
        source_path = Path(item.path)
        if not source_path.exists():
            raise SystemExit(f"Missing compare source image: {source_path}")
        probability = item.probability
        sample_id = item.id.replace(":", "__")
        target_name = f"{index:03d}__p{probability:.3f}__{sample_id}{source_path.suffix.lower()}"
        target_path = category_dir / target_name
        materialize_file(source_path, target_path, mode)
        exported += 1
    return exported


def materialize_file(source_path: Path, target_path: Path, mode: str) -> None:
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    if mode == "copy":
        shutil.copy2(source_path, target_path)
        return
    if mode == "symlink":
        target_path.symlink_to(source_path)
        return
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


if __name__ == "__main__":
    main()
