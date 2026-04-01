from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import msgspec


T = TypeVar("T")


def decode_json(data: bytes | str, type_: type[T] | Any) -> T:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return msgspec.json.decode(payload, type=type_)


def load_json(path: Path, type_: type[T] | Any) -> T:
    return decode_json(path.read_bytes(), type_)


def encode_json(data: Any, *, pretty: bool = False) -> bytes:
    encoded = msgspec.json.encode(data)
    if pretty:
        encoded = msgspec.json.format(encoded, indent=2)
    return encoded


def dump_json(path: Path, data: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encode_json(data, pretty=pretty)
    path.write_bytes(payload + (b"\n" if pretty and not payload.endswith(b"\n") else b""))


def load_jsonl(path: Path, type_: type[T] | Any) -> list[T]:
    if not path.exists():
        return []
    entries: list[T] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        entries.append(msgspec.json.decode(line, type=type_))
    return entries


def dump_jsonl(path: Path, items: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [msgspec.json.encode(item) for item in items]
    payload = b"\n".join(lines) + (b"\n" if lines else b"")
    path.write_bytes(payload)


def decode_string_list(value: str | None) -> list[str]:
    if not value:
        return []
    return decode_json(value, list[str])


def encode_string_list(values: list[str]) -> str:
    unique_values = sorted(set(values))
    return msgspec.json.encode(unique_values).decode("utf-8")


class IngestExportAvatar(msgspec.Struct, omit_defaults=True, kw_only=True):
    normalized_url: str
    original_url: str
    handles: list[str] = msgspec.field(default_factory=list)
    display_names: list[str] = msgspec.field(default_factory=list)
    source_surfaces: list[str] = msgspec.field(default_factory=list)
    seen_count: int = 0
    first_seen_at: str
    last_seen_at: str
    example_profile_url: str | None = None
    example_notification_url: str | None = None
    example_tweet_url: str | None = None
    whitelisted: bool = False


class IngestExportPayload(msgspec.Struct, omit_defaults=True, kw_only=True):
    version: int
    exported_at: str
    avatar_count: int
    total_sightings: int
    avatars: list[IngestExportAvatar] = msgspec.field(default_factory=list)


class CollectionSample(msgspec.Struct, kw_only=True):
    token_id: int
    local_path: str
    image_url: str | None = None
    metadata_url: str | None = None


class CollectionFailure(msgspec.Struct, kw_only=True):
    token_id: int
    error: str


class CollectionManifestCollection(msgspec.Struct, omit_defaults=True, kw_only=True):
    slug: str
    name: str
    contract: str | None = None
    total_supply: int
    target_count: int
    sample_count: int
    downloaded_count: int
    failed_count: int
    samples: list[CollectionSample] = msgspec.field(default_factory=list)
    failures: list[CollectionFailure] = msgspec.field(default_factory=list)


class CollectionManifest(msgspec.Struct, kw_only=True):
    version: int = 1
    generated_at: str | None = None
    collections: list[CollectionManifestCollection]


class DatasetEntryPayload(msgspec.Struct, kw_only=True):
    id: str
    path: str
    label: str
    source: str
    split: str
    label_source: str
    label_tier: str
    sample_weight: float


class SplitSummaryPayload(msgspec.Struct, kw_only=True):
    total: int
    milady: int
    not_milady: int
    gold: int
    trusted: int


class SplitManifestEvaluationPolicy(msgspec.Struct, kw_only=True):
    blind_eval_includes_collection_holdout_positives: bool
    gold_label_source: str
    trusted_label_sources: list[str]
    trusted_collection_weight: float
    model_label_weight: float
    collection_blind_holdout_val_count: int
    collection_blind_holdout_test_count: int


class SplitManifestRatios(msgspec.Struct, kw_only=True):
    train: float
    val: float
    test: float


class SplitManifestCanonical(msgspec.Struct, kw_only=True):
    id: str
    path: str
    source: str
    label_source: str
    label_tier: str
    sample_weight: float
    blind_eval_eligible: bool
    raw_sha: str
    pixel_digest: str


class SplitManifestMember(msgspec.Struct, kw_only=True):
    id: str
    path: str
    source: str
    label_source: str
    label_tier: str
    sample_weight: float
    blind_eval_eligible: bool
    raw_sha: str
    pixel_digest: str
    exported_sha: str | None = None


class SplitManifestGroup(msgspec.Struct, kw_only=True):
    group_id: str
    label: str
    split: str
    canonical: SplitManifestCanonical
    members: list[SplitManifestMember]


class SplitManifest(msgspec.Struct, kw_only=True):
    version: int = 1
    generated_at: str
    mode: str
    evaluation_policy: SplitManifestEvaluationPolicy
    ratios: SplitManifestRatios
    groups: list[SplitManifestGroup]


class MetricSummary(msgspec.Struct, kw_only=True):
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: float
    false_positive: float
    true_negative: float
    false_negative: float


class DiagnosticBucket(msgspec.Struct, kw_only=True):
    count: int
    metrics: MetricSummary


class RunEvaluationPolicy(msgspec.Struct, omit_defaults=True, kw_only=True):
    headline: str = "legacy"
    train_includes_trusted_synthetic: bool | None = None
    train_includes_weak_labels: bool | None = None


class RunDatasetSplitSummary(msgspec.Struct, kw_only=True):
    total: int
    class_counts: dict[str, int]
    source_counts: dict[str, int]
    label_source_counts: dict[str, int]
    label_tier_counts: dict[str, int]


class RunHistoryEntry(msgspec.Struct, kw_only=True):
    epoch: int
    phase: str = "finetune"
    learning_rate: float = 0.0
    train_loss: float
    val_precision: float
    val_recall: float
    val_f1: float
    threshold: float


class RunSummary(msgspec.Struct, omit_defaults=True, kw_only=True):
    run_id: str
    architecture: str
    class_names: list[str]
    positive_index: int
    image_size: int
    mean: list[float]
    std: list[float]
    precision_floor: float
    seed: int = 1337
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool = False
    head_warmup_epochs: int = 0
    scheduler: str = ""
    head_learning_rate: float = 0.0
    learning_rate: float = 0.0
    label_smoothing: float = 0.0
    augment: bool = False
    evaluation_policy: RunEvaluationPolicy = msgspec.field(default_factory=RunEvaluationPolicy)
    dataset_splits: dict[str, RunDatasetSplitSummary] = msgspec.field(default_factory=dict)
    best_epoch: int
    threshold: float
    history: list[RunHistoryEntry]
    val_metrics: MetricSummary
    test_metrics: MetricSummary
    val_diagnostics_by_source: dict[str, dict[str, DiagnosticBucket]] = msgspec.field(default_factory=dict)
    test_diagnostics_by_source: dict[str, dict[str, DiagnosticBucket]] = msgspec.field(default_factory=dict)
    checkpoint_path: str = ""


class PublicModelMetadata(msgspec.Struct, omit_defaults=True, kw_only=True):
    architecture: str
    generated_at: str
    input_size: int
    channels: int
    class_names: list[str]
    mean: list[float]
    std: list[float]
    positive_index: int
    threshold: float
    run_id: str


class CompareErrorItem(msgspec.Struct, kw_only=True):
    id: str
    path: str
    label: str
    source: str
    label_source: str
    label_tier: str
    split: str
    probability: float
    threshold: float
    predicted_label: str


class CompareRunSummary(msgspec.Struct, kw_only=True):
    threshold: float
    precision_floor: float
    val_metrics: MetricSummary
    test_metrics: MetricSummary
    val_diagnostics_by_source: dict[str, dict[str, DiagnosticBucket]] = msgspec.field(default_factory=dict)
    test_diagnostics_by_source: dict[str, dict[str, DiagnosticBucket]] = msgspec.field(default_factory=dict)
    false_positive_count: int
    false_negative_count: int
    false_positives_path: str
    false_negatives_path: str


class CompareSummaryEvaluationPolicy(msgspec.Struct, kw_only=True):
    headline: str = "legacy"


class CompareSummary(msgspec.Struct, omit_defaults=True, kw_only=True):
    generated_at: str
    device: str
    val_size: int = 0
    test_size: int = 0
    evaluation_policy: CompareSummaryEvaluationPolicy = msgspec.field(default_factory=CompareSummaryEvaluationPolicy)
    run_ids: list[str]
    runs: dict[str, CompareRunSummary]
    releases: dict[str, str] | None = None


class ReviewItem(msgspec.Struct, omit_defaults=True, kw_only=True):
    sha256: str
    label: str | None = None
    label_source: str | None = None
    local_path: str
    byte_size: int | None = None
    width: int | None = None
    height: int | None = None
    handles: list[str]
    display_names: list[str]
    source_surfaces: list[str]
    seen_count: int
    whitelisted: bool
    max_model_score: float | None = None
    latest_model_predicted_label: str | None = None
    latest_model_run_id: str | None = None
    latest_model_threshold: float | None = None
    latest_model_distance_to_threshold: float | None = None
    disagreement_flags: list[str]
    labeled_at: str | None = None
    example_profile_url: str | None = None
    example_notification_url: str | None = None
    example_tweet_url: str | None = None
    last_seen_at: str | None = None
    image_url_count: int


class ReviewSummaryPayload(msgspec.Struct, kw_only=True):
    catalog_path: str
    selected_run_id: str | None
    available_run_ids: list[str]
    total_images: int
    queue_counts: dict[str, int]
    label_counts: dict[str, int]
    needs_review: int
    can_undo: bool


class ReviewQueuePayload(msgspec.Struct, kw_only=True):
    queue: str
    index: int
    total: int
    item: ReviewItem | None


class ReviewBatchPayload(msgspec.Struct, kw_only=True):
    queue: str
    total: int
    offset: int
    items: list[ReviewItem]


class ReviewItemResponse(msgspec.Struct, kw_only=True):
    item: ReviewItem


class ReviewHistoryEntry(msgspec.Struct, omit_defaults=True, kw_only=True):
    event_id: int
    sha256: str
    created_at: str
    new_label: str
    previous_label: str | None = None
    item: ReviewItem | None = None


class ReviewHistoryPayload(msgspec.Struct, kw_only=True):
    history: list[ReviewHistoryEntry]


class ReviewGridPayload(msgspec.Struct, kw_only=True):
    total: int
    items: list[ReviewItem]


class ReviewLabelRequest(msgspec.Struct, omit_defaults=True, kw_only=True):
    sha256: str
    label: str
    note: str | None = None


class ReviewBatchLabelItem(msgspec.Struct, kw_only=True):
    sha256: str
    label: str


class ReviewBatchLabelRequest(msgspec.Struct, kw_only=True):
    items: list[ReviewBatchLabelItem]


class ReviewChangedResponse(msgspec.Struct, kw_only=True):
    ok: bool
    changed: bool


class ReviewBatchLabelResponse(msgspec.Struct, kw_only=True):
    ok: bool
    batch_id: str
    count: int


class ReviewUndoBatchResponse(msgspec.Struct, omit_defaults=True, kw_only=True):
    ok: bool
    batch_id: str
    undone_sha256_list: list[str]
    undone_sha256: str | None = None


class ReviewUndoSingleResponse(msgspec.Struct, omit_defaults=True, kw_only=True):
    ok: bool
    undone_sha256: str
    item: ReviewItem | None = None
