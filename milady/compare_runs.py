from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from .mobilenet_common import DatasetEntry, choose_threshold, compute_metrics, create_model, load_dataset_entries, probabilities_from_model
from .pipeline_common import (
    MODEL_COMPARE_ROOT,
    MODEL_RUN_ROOT,
    PUBLIC_METADATA_PATH,
    SPLIT_MANIFEST_PATH,
    SPLIT_ROOT,
    connect_offline_cache_db,
    ensure_layout,
)
from .wire import (
    CompareErrorItem,
    CompareRunSummary,
    CompareSummary,
    CompareSummaryEvaluationPolicy,
    DiagnosticBucket,
    PublicModelMetadata,
    RunSummary,
    SplitManifest,
    dump_json,
    load_json,
)

HEADLINE_EVAL_POLICY = "manual_export_gold_plus_collection_positive_holdout"
ALL_MANUAL_EVAL_POLICY = "all_manual_export_labels"
ALL_EXPORTED_EVAL_POLICY = "all_exported_labels"
PROD_RELEASES: list[tuple[str, str]] = [
    ("v0.2.2", "20260327T142224Z"),
    ("v0.3.0", "20260327T212453Z"),
    ("v0.4.0", "20260328T144735Z"),
    ("v0.5.0", "20260328T223931Z"),
    ("v0.6.0", "20260329T124912Z"),
    ("v0.7.0", "20260329T181946Z"),
    ("v0.8.0", "20260329T220050Z"),
    ("v0.9.0", "20260330Tlr1e4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained classifier checkpoints on a shared evaluation set.")
    parser.add_argument("--run-id", dest="run_ids", action="append", help="Run ID to evaluate. Pass multiple times.")
    parser.add_argument(
        "--group",
        choices=("prod-history", "latest-vs-prod"),
        help="Named run group to evaluate instead of passing explicit --run-id values.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory. Defaults under cache/models/.../compare.")
    parser.add_argument(
        "--eval-set",
        choices=("blind", "all-manual", "all-exported"),
        default="blind",
        help="Evaluation population: blind val/test splits, all deduped manually labeled exported avatars, or all deduped exported avatars including model labels.",
    )
    parser.add_argument("--prod-only", action="store_true", help="For prod-history, skip the latest non-promoted work-in-progress run.")

    args = parser.parse_args()
    if args.group is not None and args.run_ids:
        raise SystemExit("Pass either explicit --run-id values or --group, not both.")
    if args.group is None and not args.run_ids:
        raise SystemExit("Pass at least one --run-id, or select a preset with --group.")
    if args.group == "latest-vs-prod" and args.prod_only:
        raise SystemExit("--prod-only only applies to --group prod-history.")
    return args


def main() -> None:
    args = parse_args()
    run_ids, releases = resolve_run_selection(args)
    results, summary_output = run_compare(
        run_ids=run_ids,
        eval_set=args.eval_set,
        batch_size=args.batch_size,
        force_cpu=args.cpu,
        output_dir=args.output_dir or default_group_output_dir(args.group, args.eval_set),
    )
    if releases is not None:
        results.releases = releases
        dump_json(summary_output, results)
        print_releases(results.releases)


def run_compare(
    *,
    run_ids: list[str],
    eval_set: str = "blind",
    batch_size: int = 64,
    force_cpu: bool = False,
    output_dir: Path | None = None,
) -> tuple[CompareSummary, Path]:
    run_ids = dedupe(run_ids)
    if not run_ids:
        raise SystemExit("Pass at least one --run-id value.")

    ensure_layout()
    resolved_output_dir = output_dir or default_output_dir(run_ids)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(force_cpu)
    cache_connection = connect_offline_cache_db()
    try:
        val_entries, test_entries, evaluation_policy = load_evaluation_entries(eval_set)
        if not val_entries or not test_entries:
            raise SystemExit("Missing evaluation entries. Run `uv run milady build-dataset` first.")
        print(
            f"[compare] device={device.type} runs={len(run_ids)} val={len(val_entries)} test={len(test_entries)}",
            flush=True,
        )
        print(f"[compare] output_dir={resolved_output_dir}", flush=True)

        result_runs: dict[str, CompareRunSummary] = {}

        for run_id in run_ids:
            summary_path = MODEL_RUN_ROOT / run_id / "summary.json"
            checkpoint_path = MODEL_RUN_ROOT / run_id / "best.pt"
            if not summary_path.exists() or not checkpoint_path.exists():
                raise SystemExit(f"Missing summary or checkpoint for run {run_id}")

            summary = load_json(summary_path, RunSummary)
            precision_floor = float(summary.precision_floor)

            model = create_model(pretrained=False).to(device)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)

            val_probabilities, val_labels = evaluate(model, val_entries, device, batch_size, cache_connection)
            threshold, val_metrics = choose_threshold(val_probabilities, val_labels, precision_floor)
            print(
                f"[compare:{run_id}] validation done threshold={threshold:.4f} "
                f"precision={val_metrics.precision:.4f} recall={val_metrics.recall:.4f}",
                flush=True,
            )
            test_probabilities, test_labels = evaluate(model, test_entries, device, batch_size, cache_connection)
            test_metrics = compute_metrics(test_probabilities, test_labels, threshold)

            false_positives = collect_errors(test_entries, test_probabilities, test_labels, threshold, want_predicted=1, want_label=0)
            false_negatives = collect_errors(test_entries, test_probabilities, test_labels, threshold, want_predicted=0, want_label=1)

            false_positives_path = resolved_output_dir / f"{run_id}.false_positives.json"
            false_negatives_path = resolved_output_dir / f"{run_id}.false_negatives.json"
            dump_json(false_positives_path, false_positives)
            dump_json(false_negatives_path, false_negatives)
            print(
                f"[compare:{run_id}] test done precision={test_metrics.precision:.4f} "
                f"recall={test_metrics.recall:.4f} fp={len(false_positives)} fn={len(false_negatives)}",
                flush=True,
            )

            result_runs[run_id] = CompareRunSummary(
                threshold=threshold,
                precision_floor=precision_floor,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                val_diagnostics_by_source=diagnostic_metrics_by(val_entries, val_probabilities, threshold),
                test_diagnostics_by_source=diagnostic_metrics_by(test_entries, test_probabilities, threshold),
                false_positive_count=len(false_positives),
                false_negative_count=len(false_negatives),
                false_positives_path=str(false_positives_path),
                false_negatives_path=str(false_negatives_path),
            )

        results = CompareSummary(
            generated_at=datetime.now(UTC).isoformat(),
            device=device.type,
            val_size=len(val_entries),
            test_size=len(test_entries),
            evaluation_policy=CompareSummaryEvaluationPolicy(headline=evaluation_policy),
            run_ids=run_ids,
            runs=result_runs,
        )
        summary_output = resolved_output_dir / "summary.json"
        dump_json(summary_output, results)
        print(format_compare_report(results))
        print(f"[saved] {summary_output}")
        return results, summary_output
    finally:
        cache_connection.close()


def resolve_run_selection(args: argparse.Namespace) -> tuple[list[str], dict[str, str] | None]:
    if args.group == "prod-history":
        releases = list(PROD_RELEASES)
        if not args.prod_only:
            latest_wip = find_latest_wip_run()
            if latest_wip is not None:
                releases.append(("wip", latest_wip))
        return [run_id for _, run_id in releases], {version: run_id for version, run_id in releases}
    if args.group == "latest-vs-prod":
        latest_wip = find_latest_wip_run()
        if latest_wip is None:
            raise SystemExit("No non-promoted work-in-progress run found.")
        current_prod_run_id = load_current_promoted_run_id()
        releases = [("prod", current_prod_run_id), ("wip", latest_wip)]
        return [current_prod_run_id, latest_wip], {version: run_id for version, run_id in releases}
    return args.run_ids, None


def dedupe(run_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        ordered.append(run_id)
    return ordered


def default_output_dir(run_ids: list[str]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = "__".join(run_ids)
    return MODEL_COMPARE_ROOT / f"{stamp}__{slug}"


def default_group_output_dir(group: str | None, eval_set: str) -> Path | None:
    if group is None:
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    if group == "prod-history":
        return MODEL_COMPARE_ROOT / f"prod-history-{eval_set}-{stamp}"
    if group == "latest-vs-prod":
        return MODEL_COMPARE_ROOT / f"latest-vs-prod-{eval_set}-{stamp}"
    return None


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


def load_current_promoted_run_id() -> str:
    metadata = load_json(PUBLIC_METADATA_PATH, PublicModelMetadata)
    return metadata.run_id


def print_releases(releases: dict[str, str]) -> None:
    print("")
    print("Run labels")
    for label, run_id in releases.items():
        print(f"  {label:<8} {run_id}")


def load_evaluation_entries(eval_set: str) -> tuple[list, list, str]:
    if eval_set == "blind":
        val_entries = load_dataset_entries(SPLIT_ROOT / "val.jsonl")
        test_entries = load_dataset_entries(SPLIT_ROOT / "test.jsonl")
        return val_entries, test_entries, HEADLINE_EVAL_POLICY
    if eval_set == "all-manual":
        entries = load_all_manual_export_entries()
        return entries, entries, ALL_MANUAL_EVAL_POLICY
    if eval_set == "all-exported":
        entries = load_all_exported_entries()
        return entries, entries, ALL_EXPORTED_EVAL_POLICY
    raise SystemExit(f"Unknown eval set: {eval_set}")


def load_all_manual_export_entries() -> list:
    return load_all_exported_entries({"manual"})


def load_all_exported_entries(allowed_label_sources: set[str] | None = None) -> list:
    if not SPLIT_MANIFEST_PATH.exists():
        raise SystemExit("Missing split manifest. Run `uv run milady build-dataset` first.")
    manifest = load_json(SPLIT_MANIFEST_PATH, SplitManifest)
    groups = manifest.groups
    entries = []
    for group in groups:
        canonical = group.canonical
        if canonical.source != "export":
            continue
        label_source = canonical.label_source
        if allowed_label_sources is not None and label_source not in allowed_label_sources:
            continue
        label = group.label
        if label not in ("milady", "not_milady"):
            continue
        entries.append(
            DatasetEntry(
                sample_id=canonical.id,
                path=Path(canonical.path),
                label=label,
                source="export",
                split="all-exported",
                label_source=label_source,
                label_tier=canonical.label_tier,
                sample_weight=canonical.sample_weight,
            )
        )
    return entries


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(
    model: torch.nn.Module,
    entries: list,
    device: torch.device,
    batch_size: int = 64,
    cache_connection=None,
) -> tuple[list[float], list[int]]:
    probabilities = probabilities_from_model(
        model,
        [entry.path for entry in entries],
        [entry.source for entry in entries],
        device,
        batch_size=batch_size,
        connection=cache_connection,
    ).tolist()
    labels = [1 if entry.label == "milady" else 0 for entry in entries]
    return probabilities, labels

def collect_errors(
    entries,
    probabilities: list[float],
    labels: list[int],
    threshold: float,
    *,
    want_predicted: int,
    want_label: int,
) -> list[dict[str, object]]:
    items: list[CompareErrorItem] = []
    for entry, probability, label in zip(entries, probabilities, labels, strict=True):
        predicted = 1 if probability >= threshold else 0
        if predicted != want_predicted or label != want_label:
            continue
        items.append(
            CompareErrorItem(
                id=entry.sample_id,
                path=str(entry.path),
                label=entry.label,
                source=entry.source,
                label_source=entry.label_source,
                label_tier=entry.label_tier,
                split=entry.split,
                probability=probability,
                threshold=threshold,
                predicted_label="milady" if predicted == 1 else "not_milady",
            )
        )
    return items


def diagnostic_metrics_by(entries, probabilities: list[float], threshold: float) -> dict[str, dict[str, DiagnosticBucket]]:
    diagnostics: dict[str, dict[str, DiagnosticBucket]] = {}
    group_fields = {
        "source": "source",
        "label_source": "label_source",
        "label_tier": "label_tier",
    }
    for group_name, field_name in group_fields.items():
        values = sorted({str(getattr(entry, field_name)) for entry in entries})
        grouped_metrics: dict[str, DiagnosticBucket] = {}
        for value in values:
            indices = [
                index
                for index, entry in enumerate(entries)
                if str(getattr(entry, field_name)) == value
            ]
            if not indices:
                continue
            grouped_metrics[value] = DiagnosticBucket(
                count=len(indices),
                metrics=compute_metrics(
                    [probabilities[index] for index in indices],
                    [1 if entries[index].label == "milady" else 0 for index in indices],
                    threshold,
                ),
            )
        diagnostics[group_name] = grouped_metrics
    return diagnostics


def format_compare_report(results: CompareSummary) -> str:
    baseline_run_id = results.run_ids[0]
    baseline = results.runs[baseline_run_id]
    lines = [
        "",
        f"Compare summary ({results.evaluation_policy.headline})",
        f"Baseline: {baseline_run_id}",
        "",
        format_compare_table(results, baseline_run_id, baseline),
    ]
    return "\n".join(lines)


def format_compare_table(
    results: CompareSummary,
    baseline_run_id: str,
    baseline: CompareRunSummary,
) -> str:
    headers = [
        "run",
        "thr",
        "val_p",
        "val_r",
        "test_p",
        "Δp",
        "test_r",
        "Δr",
        "fp",
        "Δfp",
        "fn",
        "Δfn",
        "verdict",
    ]
    rows = []
    for run_id in results.run_ids:
        run = results.runs[run_id]
        rows.append(
            [
                run_id,
                f"{run.threshold:.4f}",
                f"{run.val_metrics.precision:.4f}",
                f"{run.val_metrics.recall:.4f}",
                f"{run.test_metrics.precision:.4f}",
                metric_delta(run.test_metrics.precision, baseline.test_metrics.precision, higher_is_better=True),
                f"{run.test_metrics.recall:.4f}",
                metric_delta(run.test_metrics.recall, baseline.test_metrics.recall, higher_is_better=True),
                str(run.false_positive_count),
                count_delta(run.false_positive_count, baseline.false_positive_count, lower_is_better=True),
                str(run.false_negative_count),
                count_delta(run.false_negative_count, baseline.false_negative_count, lower_is_better=True),
                compare_verdict(run_id, run, baseline_run_id, baseline),
            ]
        )
    widths = [
        max(len(headers[index]), max(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    line_parts = []
    line_parts.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    line_parts.append("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        line_parts.append("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(line_parts)


def metric_delta(current: float, baseline: float, *, higher_is_better: bool) -> str:
    if abs(current - baseline) < 1e-9:
        return "="
    delta = current - baseline
    if higher_is_better:
        direction = "↑" if delta > 0 else "↓"
    else:
        direction = "↓" if delta < 0 else "↑"
    return f"{direction}{abs(delta):.4f}"


def count_delta(current: int, baseline: int, *, lower_is_better: bool) -> str:
    if current == baseline:
        return "="
    delta = current - baseline
    if lower_is_better:
        direction = "↓" if delta < 0 else "↑"
    else:
        direction = "↑" if delta > 0 else "↓"
    return f"{direction}{abs(delta)}"


def compare_verdict(
    run_id: str,
    run: CompareRunSummary,
    baseline_run_id: str,
    baseline: CompareRunSummary,
) -> str:
    if run_id == baseline_run_id:
        return "baseline"

    comparisons: list[tuple[bool, bool]] = [
        (
            run.test_metrics.precision >= baseline.test_metrics.precision,
            run.test_metrics.precision > baseline.test_metrics.precision,
        ),
        (
            run.test_metrics.recall >= baseline.test_metrics.recall,
            run.test_metrics.recall > baseline.test_metrics.recall,
        ),
        (
            run.false_positive_count <= baseline.false_positive_count,
            run.false_positive_count < baseline.false_positive_count,
        ),
        (
            run.false_negative_count <= baseline.false_negative_count,
            run.false_negative_count < baseline.false_negative_count,
        ),
    ]
    all_non_worse = all(non_worse for non_worse, _ in comparisons)
    any_better = any(strict for _, strict in comparisons)

    worse_comparisons: list[tuple[bool, bool]] = [
        (
            run.test_metrics.precision <= baseline.test_metrics.precision,
            run.test_metrics.precision < baseline.test_metrics.precision,
        ),
        (
            run.test_metrics.recall <= baseline.test_metrics.recall,
            run.test_metrics.recall < baseline.test_metrics.recall,
        ),
        (
            run.false_positive_count >= baseline.false_positive_count,
            run.false_positive_count > baseline.false_positive_count,
        ),
        (
            run.false_negative_count >= baseline.false_negative_count,
            run.false_negative_count > baseline.false_negative_count,
        ),
    ]
    all_non_better = all(non_better for non_better, _ in worse_comparisons)
    any_worse = any(strict for _, strict in worse_comparisons)

    if all_non_worse and any_better:
        return "better"
    if all_non_better and any_worse:
        return "worse"
    return "mixed"


if __name__ == "__main__":
    main()
