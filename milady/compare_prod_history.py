from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .compare_runs import run_compare
from .pipeline_common import MODEL_COMPARE_ROOT, MODEL_RUN_ROOT
from .wire import CompareSummary, dump_json, encode_json

PROD_RELEASES: list[tuple[str, str]] = [
    ("v0.2.2", "20260327T142224Z"),
    ("v0.3.0", "20260327T212453Z"),
    ("v0.4.0", "20260328T144735Z"),
    ("v0.5.0", "20260328T223931Z"),
    ("v0.6.0", "20260329T124912Z"),
    ("v0.7.0", "20260329T181946Z"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all promoted production models on a shared evaluation set.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument(
        "--eval-set",
        choices=("blind", "all-manual", "all-exported"),
        default="all-manual",
        help="Evaluation population. Defaults to the current manually reviewed exported corpus used in the README accuracy table.",
    )
    parser.add_argument("--prod-only", action="store_true", help="Skip the latest non-promoted work-in-progress run.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory. Defaults under cache/models/.../compare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.eval_set)
    releases = list(PROD_RELEASES)
    if not args.prod_only:
        latest_wip = find_latest_wip_run()
        if latest_wip is not None:
            releases.append(("wip", latest_wip))
    results, summary_output = run_compare(
        run_ids=[run_id for _, run_id in releases],
        eval_set=args.eval_set,
        batch_size=args.batch_size,
        force_cpu=args.cpu,
        output_dir=output_dir,
    )
    results.releases = {version: run_id for version, run_id in releases}
    dump_json(summary_output, results)
    print(encode_json({"releases": results.releases}, pretty=True).decode("utf-8"))


def default_output_dir(eval_set: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return MODEL_COMPARE_ROOT / f"prod-history-{eval_set}-{stamp}"


def find_latest_wip_run() -> str | None:
    prod_run_ids = {run_id for _, run_id in PROD_RELEASES}
    candidates: list[tuple[float, str]] = []
    for run_dir in MODEL_RUN_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_id in prod_run_ids:
            continue
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "best.pt"
        if not summary_path.exists() or not checkpoint_path.exists():
            continue
        candidates.append((summary_path.stat().st_mtime, run_id))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


if __name__ == "__main__":
    main()
