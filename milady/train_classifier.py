from __future__ import annotations

import argparse
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .mobilenet_common import (
    AvatarDataset,
    CLASS_NAMES,
    DatasetEntry,
    HEADLINE_EVAL_POLICY,
    MODEL_IMAGE_SIZE,
    MODEL_MEAN,
    MODEL_STD,
    POSITIVE_INDEX,
    choose_device,
    choose_threshold,
    compute_metrics,
    create_model,
    diagnostic_metrics_by,
    evaluate_entries,
    load_dataset_entries,
)
from .pipeline_common import COLLECTION_MANIFEST_PATH, MODEL_RUN_ROOT, SPLIT_MANIFEST_PATH, SPLIT_ROOT, connect_db, connect_offline_cache_db
from .wire import CollectionManifest, MetricSummary, RunDatasetSplitSummary, RunHistoryEntry, RunSummary, SplitManifest, dump_json, encode_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV3-Small binary Milady classifier.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--scheduler", choices=("cosine", "off"), default="cosine")
    parser.add_argument("--head-learning-rate", type=float, help="Optional LR for classifier-head warmup. Defaults to learning rate.")
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=25, help="Print a batch progress update every N training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--precision-floor", type=float, default=0.995)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even when MPS/CUDA is available.")
    parser.add_argument("--refit", action="store_true", help="Fit on train+val and use test as the selection/eval split for a final refit experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_dataset_is_fresh()
    train_entries = load_dataset_entries(SPLIT_ROOT / "train.jsonl")
    val_entries = load_dataset_entries(SPLIT_ROOT / "val.jsonl")
    test_entries = load_dataset_entries(SPLIT_ROOT / "test.jsonl")
    if args.refit:
        train_entries = [*train_entries, *val_entries]
        val_entries = test_entries
    if not train_entries or not val_entries:
        raise SystemExit("Missing train/val split files. Run `uv run milady build-dataset` first.")

    seed_everything(args.seed)
    device = choose_device(args.cpu)
    head_warmup_epochs = max(0, min(args.head_warmup_epochs, args.epochs))
    finetune_epochs = max(0, args.epochs - head_warmup_epochs)
    head_learning_rate = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    train_loader = DataLoader(
        AvatarDataset(train_entries, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=build_loader_generator(args.seed),
        num_workers=max(0, args.num_workers),
        pin_memory=False,
        persistent_workers=max(0, args.num_workers) > 0,
        worker_init_fn=worker_init_fn if max(0, args.num_workers) > 0 else None,
        prefetch_factor=max(1, args.prefetch_factor) if args.num_workers > 0 else None,
    )
    model = create_model(pretrained=True).to(device)
    set_trainable_parameters(model, head_only=head_warmup_epochs > 0)
    optimizer = create_optimizer(model, args.weight_decay, head_learning_rate if head_warmup_epochs > 0 else args.learning_rate)
    scheduler = create_scheduler(args.scheduler, optimizer, args.learning_rate, len(train_loader), finetune_epochs) if head_warmup_epochs == 0 else None
    criterion = build_loss(train_entries, args.label_smoothing).to(device)
    run_dir = MODEL_RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_connection = connect_offline_cache_db()
    try:
        print_run_header(
            args,
            device,
            train_entries,
            val_entries,
            test_entries,
            run_dir,
            head_warmup_epochs,
            head_learning_rate,
            finetune_epochs,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_threshold = 0.995
        best_selection_key = (-1.0, -1.0)
        best_epoch = -1
        best_val_metrics: MetricSummary | None = None
        history: list[RunHistoryEntry] = []
        stale_epochs = 0
        training_started_at = perf_counter()
        completed_epoch_durations: list[float] = []
        global_step = 0
        phase = "warmup" if head_warmup_epochs > 0 else "finetune"

        for epoch in range(1, args.epochs + 1):
            if epoch == head_warmup_epochs + 1 and head_warmup_epochs > 0:
                phase = "finetune"
                set_trainable_parameters(model, head_only=False)
                optimizer = create_optimizer(model, args.weight_decay, args.learning_rate)
                scheduler = create_scheduler(args.scheduler, optimizer, args.learning_rate, len(train_loader), finetune_epochs)
                stale_epochs = 0
                print(
                    f"[phase] switching to full fine-tune lr={args.learning_rate:g} scheduler={args.scheduler}",
                    flush=True,
                )
            print(f"[epoch {epoch}/{args.epochs}] start", flush=True)
            epoch_started_at = perf_counter()
            train_loss, global_step = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                args.epochs,
                args.log_every,
                global_step,
                scheduler,
                phase,
            )
            val_probabilities, val_labels = evaluate_entries(model, val_entries, device, args.batch_size, cache_connection)
            threshold, threshold_metrics = choose_threshold(val_probabilities, val_labels, args.precision_floor)
            epoch_duration_seconds = perf_counter() - epoch_started_at
            completed_epoch_durations.append(epoch_duration_seconds)
            history.append(
                RunHistoryEntry(
                    epoch=epoch,
                    phase=phase,
                    learning_rate=current_learning_rate(optimizer),
                    train_loss=train_loss,
                    val_precision=threshold_metrics.precision,
                    val_recall=threshold_metrics.recall,
                    val_f1=threshold_metrics.f1,
                    threshold=threshold,
                )
            )
            selection_key = (
                threshold_metrics.recall,
                threshold_metrics.f1,
            )
            improved = selection_key > best_selection_key
            stale_after_epoch = 0 if improved else (stale_epochs + 1 if phase == "finetune" else stale_epochs)
            overall_eta_seconds = estimate_overall_eta(args.epochs, epoch, completed_epoch_durations)
            print_epoch_summary(
                epoch,
                args.epochs,
                train_loss,
                phase,
                current_learning_rate(optimizer),
                threshold,
                threshold_metrics,
                improved,
                stale_after_epoch,
                args.patience,
                epoch_duration_seconds,
                perf_counter() - training_started_at,
                overall_eta_seconds,
            )

            if improved:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_threshold = threshold
                best_selection_key = selection_key
                best_epoch = epoch
                best_val_metrics = threshold_metrics
                if phase == "finetune":
                    stale_epochs = 0
                print(
                    f"[epoch {epoch}/{args.epochs}] new best checkpoint "
                    f"(recall={threshold_metrics.recall:.4f}, threshold={best_threshold:.4f})",
                    flush=True,
                )
            else:
                if phase == "finetune":
                    stale_epochs += 1
                if phase == "finetune" and stale_epochs >= args.patience:
                    print(
                        f"[epoch {epoch}/{args.epochs}] early stopping after {stale_epochs} stale epoch(s)",
                        flush=True,
                    )
                    break

        if best_state is None or best_val_metrics is None:
            raise SystemExit("Training did not produce a checkpoint.")

        checkpoint_path = run_dir / "best.pt"
        torch.save(best_state, checkpoint_path)
        print(f"[checkpoint] saved best weights to {checkpoint_path}", flush=True)

        model.load_state_dict(best_state)
        val_probabilities, val_labels = evaluate_entries(model, val_entries, device, args.batch_size, cache_connection)
        best_threshold, best_val_metrics = choose_threshold(val_probabilities, val_labels, args.precision_floor)
        print("[test] evaluating best checkpoint on test split", flush=True)
        test_probabilities, test_labels = evaluate_entries(model, test_entries, device, args.batch_size, cache_connection)
        test_metrics = compute_metrics(test_probabilities, test_labels, best_threshold)

        summary = RunSummary(
            run_id=args.run_id,
            architecture="mobilenet_v3_small",
            class_names=CLASS_NAMES,
            positive_index=POSITIVE_INDEX,
            image_size=MODEL_IMAGE_SIZE,
            mean=MODEL_MEAN,
            std=MODEL_STD,
            precision_floor=args.precision_floor,
            seed=args.seed,
            num_workers=max(0, args.num_workers),
            prefetch_factor=max(1, args.prefetch_factor) if args.num_workers > 0 else None,
            pin_memory=False,
            head_warmup_epochs=head_warmup_epochs,
            scheduler=args.scheduler,
            head_learning_rate=head_learning_rate,
            learning_rate=args.learning_rate,
            label_smoothing=args.label_smoothing,
            evaluation_policy_headline="refit_train_plus_val_with_test_selection" if args.refit else HEADLINE_EVAL_POLICY,
            dataset_splits={
                "train": split_summary(train_entries),
                "val": split_summary(val_entries),
                "test": split_summary(test_entries),
            },
            best_epoch=best_epoch,
            threshold=best_threshold,
            history=history,
            val_metrics=best_val_metrics,
            test_metrics=test_metrics,
            val_diagnostics_by_source=diagnostic_metrics_by(entries=val_entries, probabilities=val_probabilities, threshold=best_threshold),
            test_diagnostics_by_source=diagnostic_metrics_by(entries=test_entries, probabilities=test_probabilities, threshold=best_threshold),
            checkpoint_path=str(checkpoint_path),
        )
        dump_json(run_dir / "summary.json", summary)
        print(
            "[done] "
            f"best_epoch={best_epoch} "
            f"threshold={best_threshold:.4f} "
            f"blind_val_precision={best_val_metrics.precision:.4f} "
            f"blind_val_recall={best_val_metrics.recall:.4f} "
            f"blind_test_precision={test_metrics.precision:.4f} "
            f"blind_test_recall={test_metrics.recall:.4f}",
            flush=True,
        )
        print(encode_json(summary, pretty=True).decode("utf-8"))
    finally:
        cache_connection.close()


def assert_dataset_is_fresh() -> None:
    if not SPLIT_MANIFEST_PATH.exists():
        raise SystemExit("Missing split manifest. Run `uv run milady build-dataset` first.")

    split_manifest = load_json(SPLIT_MANIFEST_PATH, SplitManifest)
    split_generated_at = parse_timestamp(split_manifest.generated_at)
    stale_reasons: list[str] = []

    connection = connect_db()
    try:
        label_row = connection.execute(
            """
            SELECT MAX(updated_at) AS latest_updated_at
            FROM images
            WHERE label IN ('milady', 'not_milady')
              AND label_source IN ('manual', 'model')
            """
        ).fetchone()
    finally:
        connection.close()

    latest_label_timestamp = None if label_row is None else label_row["latest_updated_at"]
    if latest_label_timestamp is not None:
        latest_label_update = parse_timestamp(str(latest_label_timestamp))
        if latest_label_update > split_generated_at:
            stale_reasons.append(
                "exported labels changed after the last dataset build "
                f"({latest_label_update.isoformat()} > {split_generated_at.isoformat()})"
            )

    if COLLECTION_MANIFEST_PATH.exists():
        collection_manifest = load_json(COLLECTION_MANIFEST_PATH, CollectionManifest)
        if collection_manifest.generated_at is not None:
            collection_generated_at = parse_timestamp(collection_manifest.generated_at)
            if collection_generated_at > split_generated_at:
                stale_reasons.append(
                    "collection manifest changed after the last dataset build "
                    f"({collection_generated_at.isoformat()} > {split_generated_at.isoformat()})"
                )

    if stale_reasons:
        details = "\n".join(f"- {reason}" for reason in stale_reasons)
        raise SystemExit(
            "Dataset splits are stale. Run `uv run milady build-dataset` before training.\n"
            f"{details}"
        )


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count // 2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_loss(train_entries: list[DatasetEntry], label_smoothing: float) -> nn.Module:
    positive_weight_total = sum(entry.sample_weight for entry in train_entries if entry.label == "milady")
    negative_weight_total = sum(entry.sample_weight for entry in train_entries if entry.label != "milady")
    positive_weight = negative_weight_total / max(1e-8, positive_weight_total)
    return nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, positive_weight], dtype=torch.float32),
        reduction="none",
        label_smoothing=label_smoothing,
    )


def create_optimizer(model: nn.Module, weight_decay: float, learning_rate: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    total_steps = max(1, steps_per_epoch * epochs)
    if scheduler_name == "off" or epochs <= 0:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


def set_trainable_parameters(model: nn.Module, *, head_only: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier") if head_only else True


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_backbone_batchnorm_mode(model: nn.Module, *, frozen: bool) -> None:
    for name, module in model.named_modules():
        if name.startswith("classifier"):
            continue
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval() if frozen else module.train()


def training_loss_values_from_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    logits = model(inputs)
    return criterion(logits, labels)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_every: int,
    global_step_base: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    phase: str,
) -> tuple[float, int]:
    model.train()
    set_backbone_batchnorm_mode(model, frozen=phase == "warmup")
    total_loss = 0.0
    total_items = 0
    total_batches = max(1, len(loader))
    epoch_started_at = perf_counter()
    for batch_index, (inputs, labels, sample_weights) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss_values = training_loss_values_from_batch(model, inputs, labels, criterion)
        loss = (loss_values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_items += inputs.size(0)
        if should_log_batch(batch_index, total_batches, log_every):
            average_loss = total_loss / max(1, total_items)
            elapsed_seconds = perf_counter() - epoch_started_at
            average_batch_seconds = elapsed_seconds / max(1, batch_index)
            epoch_eta_seconds = average_batch_seconds * max(0, total_batches - batch_index)
            print(
                f"[epoch {epoch}/{total_epochs}] batch {batch_index}/{total_batches} "
                f"loss={loss.item():.4f} avg_loss={average_loss:.4f} "
                f"elapsed={format_duration(elapsed_seconds)} eta={format_duration(epoch_eta_seconds)}",
                flush=True,
            )
    return total_loss / max(1, total_items), global_step_base + total_batches


def print_run_header(
    args: argparse.Namespace,
    device: torch.device,
    train_entries: list[DatasetEntry],
    val_entries: list[DatasetEntry],
    test_entries: list[DatasetEntry],
    run_dir: Path,
    head_warmup_epochs: int,
    head_learning_rate: float,
    finetune_epochs: int,
) -> None:
    positives = sum(1 for entry in train_entries if entry.label == "milady")
    negatives = len(train_entries) - positives
    print(
        f"[setup] run_id={args.run_id} device={device.type} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.learning_rate:g} "
        f"weight_decay={args.weight_decay:g} patience={args.patience} precision_floor={args.precision_floor:.4f} "
        f"seed={args.seed} refit={'on' if args.refit else 'off'} "
        f"warmup_epochs={head_warmup_epochs} head_lr={head_learning_rate:g} "
        f"scheduler={args.scheduler} label_smoothing={args.label_smoothing:g}",
        flush=True,
    )
    print(
        f"[setup] splits train={len(train_entries)} val={len(val_entries)} test={len(test_entries)} "
        f"train_milady={positives} train_not_milady={negatives} "
        f"num_workers={max(0, args.num_workers)} prefetch_factor={(max(1, args.prefetch_factor) if args.num_workers > 0 else 'n/a')} "
        f"pin_memory=off finetune_epochs={finetune_epochs}",
        flush=True,
    )
    print(f"[setup] artifacts={run_dir}", flush=True)


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    phase: str,
    learning_rate: float,
    threshold: float,
    threshold_metrics: MetricSummary,
    improved: bool,
    stale_epochs: int,
    patience: int,
    epoch_duration_seconds: float,
    total_elapsed_seconds: float,
    overall_eta_seconds: float,
) -> None:
    status = "best" if improved else f"stale={stale_epochs}/{patience}"
    print(
        f"[epoch {epoch}/{total_epochs}] "
        f"phase={phase} "
        f"lr={learning_rate:.6g} "
        f"train_loss={train_loss:.4f} "
        f"val_precision={threshold_metrics.precision:.4f} "
        f"val_recall={threshold_metrics.recall:.4f} "
        f"val_f1={threshold_metrics.f1:.4f} "
        f"threshold={threshold:.4f} "
        f"epoch_time={format_duration(epoch_duration_seconds)} "
        f"total_elapsed={format_duration(total_elapsed_seconds)} "
        f"overall_eta={format_duration(overall_eta_seconds)} "
        f"{status}",
        flush=True,
    )


def should_log_batch(batch_index: int, total_batches: int, log_every: int) -> bool:
    if batch_index == 1 or batch_index == total_batches:
        return True
    if log_every <= 0:
        return False
    return batch_index % log_every == 0


def estimate_overall_eta(total_epochs: int, completed_epochs: int, epoch_durations: list[float]) -> float:
    if completed_epochs >= total_epochs or not epoch_durations:
        return 0.0
    average_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
    return average_epoch_seconds * max(0, total_epochs - completed_epochs)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def split_summary(entries: list[DatasetEntry]) -> RunDatasetSplitSummary:
    return RunDatasetSplitSummary(
        total=len(entries),
        class_counts={
            "milady": sum(1 for entry in entries if entry.label == "milady"),
            "not_milady": sum(1 for entry in entries if entry.label != "milady"),
        },
        source_counts=count_by(entries, "source"),
        label_source_counts=count_by(entries, "label_source"),
        label_tier_counts=count_by(entries, "label_tier"),
    )


def count_by(entries: list[DatasetEntry], attribute: str) -> dict[str, int]:
    values = sorted({str(getattr(entry, attribute)) for entry in entries})
    return {
        value: sum(1 for entry in entries if str(getattr(entry, attribute)) == value)
        for value in values
    }

if __name__ == "__main__":
    main()
