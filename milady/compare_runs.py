from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from .mobilenet_common import DatasetEntry, choose_threshold, compute_metrics, create_model, load_dataset_entries, probabilities_from_model
from .pipeline_common import MODEL_COMPARE_ROOT, MODEL_RUN_ROOT, SPLIT_MANIFEST_PATH, SPLIT_ROOT, connect_offline_cache_db, ensure_layout
from .wire import (
    CompareErrorItem,
    CompareRunSummary,
    CompareSummary,
    CompareSummaryEvaluationPolicy,
    DiagnosticBucket,
    RunSummary,
    SplitManifest,
    dump_json,
    encode_json,
    load_json,
)

HEADLINE_EVAL_POLICY = "manual_export_gold_plus_collection_positive_holdout"
ALL_MANUAL_EVAL_POLICY = "all_manual_export_labels"
ALL_EXPORTED_EVAL_POLICY = "all_exported_labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained classifier checkpoints on the current dataset splits.")
    parser.add_argument("--run-id", dest="run_ids", action="append", required=True, help="Run ID to compare. Pass multiple times.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory. Defaults under cache/models/.../compare.")
    parser.add_argument(
        "--eval-set",
        choices=("blind", "all-manual", "all-exported"),
        default="blind",
        help="Evaluation population: blind val/test splits, all deduped manually labeled exported avatars, or all deduped exported avatars including model labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_compare(
        run_ids=args.run_ids,
        eval_set=args.eval_set,
        batch_size=args.batch_size,
        force_cpu=args.cpu,
        output_dir=args.output_dir,
    )


def run_compare(
    *,
    run_ids: list[str],
    eval_set: str = "blind",
    batch_size: int = 64,
    force_cpu: bool = False,
    output_dir: Path | None = None,
) -> tuple[CompareSummary, Path]:
    run_ids = dedupe(run_ids)
    if len(run_ids) < 2:
        raise SystemExit("Pass at least two --run-id values.")

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
            print(f"[compare:{run_id}] loading checkpoint", flush=True)

            summary = load_json(summary_path, RunSummary)
            precision_floor = float(summary.precision_floor)

            model = create_model(pretrained=False).to(device)
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)

            print(f"[compare:{run_id}] evaluating validation split", flush=True)
            val_probabilities, val_labels = evaluate(model, val_entries, device, batch_size, cache_connection)
            threshold, val_metrics = choose_threshold(val_probabilities, val_labels, precision_floor)
            print(
                f"[compare:{run_id}] validation done threshold={threshold:.4f} "
                f"precision={val_metrics.precision:.4f} recall={val_metrics.recall:.4f}",
                flush=True,
            )
            print(f"[compare:{run_id}] evaluating test split", flush=True)
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
        print(encode_json(results, pretty=True).decode("utf-8"))
        print(f"[saved] {summary_output}")
        return results, summary_output
    finally:
        cache_connection.close()


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
    if torch.cuda.is_available():
        return torch.device("cuda")
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


if __name__ == "__main__":
    main()
