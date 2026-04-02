from __future__ import annotations

from .dataset_runtime import DatasetEntry
from .modeling import POSITIVE_LABEL
from .wire import DiagnosticBucket, MetricSummary


def compute_metrics(probabilities: list[float], labels: list[int], threshold: float) -> MetricSummary:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for probability, label in zip(probabilities, labels, strict=True):
        predicted = 1 if probability >= threshold else 0
        if predicted == 1 and label == 1:
            true_positive += 1
        elif predicted == 1 and label == 0:
            false_positive += 1
        elif predicted == 0 and label == 0:
            true_negative += 1
        else:
            false_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    accuracy = (true_positive + true_negative) / max(1, len(labels))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return MetricSummary(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=float(true_positive),
        false_positive=float(false_positive),
        true_negative=float(true_negative),
        false_negative=float(false_negative),
    )


def choose_threshold(probabilities: list[float], labels: list[int], precision_floor: float) -> tuple[float, MetricSummary]:
    if not probabilities:
        return 0.995, compute_metrics(probabilities, labels, 0.995)

    candidates = sorted({0.0, 1.0, *probabilities})
    scored_candidates = [
        (float(threshold), compute_metrics(probabilities, labels, float(threshold)))
        for threshold in candidates
    ]
    passing_candidates = [
        (threshold, metrics)
        for threshold, metrics in scored_candidates
        if metrics.precision >= precision_floor
    ]

    if passing_candidates:
        best_threshold, best_metrics = max(
            passing_candidates,
            key=lambda item: (
                item[1].recall,
                item[1].f1,
                item[0],
            ),
        )
        return best_threshold, best_metrics

    best_threshold, best_metrics = max(
        scored_candidates,
        key=lambda item: (
            item[1].precision,
            item[1].recall,
            item[1].f1,
            item[0],
        ),
    )
    return best_threshold, best_metrics


def diagnostic_metrics_by(
    entries: list[DatasetEntry],
    probabilities: list[float],
    threshold: float,
) -> dict[str, dict[str, DiagnosticBucket]]:
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
                    [1 if entries[index].label == POSITIVE_LABEL else 0 for index in indices],
                    threshold,
                ),
            )
        diagnostics[group_name] = grouped_metrics
    return diagnostics
