from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mobilenet_common import (
    AvatarDataset,
    CLASS_NAMES,
    MODEL_IMAGE_SIZE,
    MODEL_MEAN,
    MODEL_STD,
    POSITIVE_INDEX,
    choose_threshold,
    compute_metrics,
    create_model,
    load_dataset_entries,
    score_logits_to_probabilities,
)
from pipeline_common import MODEL_RUN_ROOT, SPLIT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV3-Small binary Milady classifier.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=25, help="Print a batch progress update every N training steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--precision-floor", type=float, default=0.995)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even when MPS/CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_entries = load_dataset_entries(SPLIT_ROOT / "train.jsonl")
    val_entries = load_dataset_entries(SPLIT_ROOT / "val.jsonl")
    test_entries = load_dataset_entries(SPLIT_ROOT / "test.jsonl")
    if not train_entries or not val_entries:
        raise SystemExit("Missing train/val split files. Run build_training_dataset.py first.")

    device = choose_device(args.cpu)
    model = create_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = build_loss(train_entries).to(device)

    train_loader = DataLoader(AvatarDataset(train_entries, training=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AvatarDataset(val_entries, training=False), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(AvatarDataset(test_entries, training=False), batch_size=args.batch_size, shuffle=False)

    run_dir = MODEL_RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print_run_header(args, device, train_entries, val_entries, test_entries, run_dir)

    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.995
    best_recall = -1.0
    best_epoch = -1
    best_val_metrics: dict[str, float] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch}/{args.epochs}] start", flush=True)
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, args.log_every)
        val_probabilities, val_labels = evaluate(model, val_loader, device)
        threshold, threshold_metrics = choose_threshold(val_probabilities, val_labels, args.precision_floor)
        history.append(
            {
                "epoch": epoch,
                "trainLoss": train_loss,
                "valPrecision": threshold_metrics["precision"],
                "valRecall": threshold_metrics["recall"],
                "valF1": threshold_metrics["f1"],
                "threshold": threshold,
            }
        )
        improved = threshold_metrics["recall"] > best_recall
        stale_after_epoch = 0 if improved else stale_epochs + 1
        print_epoch_summary(epoch, args.epochs, train_loss, threshold, threshold_metrics, improved, stale_after_epoch, args.patience)

        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold = threshold
            best_recall = threshold_metrics["recall"]
            best_epoch = epoch
            best_val_metrics = threshold_metrics
            stale_epochs = 0
            print(
                f"[epoch {epoch}/{args.epochs}] new best checkpoint "
                f"(recall={best_recall:.4f}, threshold={best_threshold:.4f})",
                flush=True,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
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
    print("[test] evaluating best checkpoint on test split", flush=True)
    test_probabilities, test_labels = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_probabilities, test_labels, best_threshold)

    summary = {
        "runId": args.run_id,
        "architecture": "mobilenet_v3_small",
        "classNames": CLASS_NAMES,
        "positiveIndex": POSITIVE_INDEX,
        "imageSize": MODEL_IMAGE_SIZE,
        "mean": MODEL_MEAN,
        "std": MODEL_STD,
        "precisionFloor": args.precision_floor,
        "bestEpoch": best_epoch,
        "threshold": best_threshold,
        "history": history,
        "valMetrics": best_val_metrics,
        "testMetrics": test_metrics,
        "checkpointPath": str(checkpoint_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "[done] "
        f"best_epoch={best_epoch} "
        f"threshold={best_threshold:.4f} "
        f"val_precision={best_val_metrics['precision']:.4f} "
        f"val_recall={best_val_metrics['recall']:.4f} "
        f"test_precision={test_metrics['precision']:.4f} "
        f"test_recall={test_metrics['recall']:.4f}",
        flush=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loss(train_entries: list) -> nn.Module:
    positives = sum(1 for entry in train_entries if entry.label == "milady")
    negatives = max(1, len(train_entries) - positives)
    positive_weight = negatives / max(1, positives)
    return nn.CrossEntropyLoss(weight=torch.tensor([1.0, positive_weight], dtype=torch.float32))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    total_batches = max(1, len(loader))
    for batch_index, (inputs, labels) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        total_items += inputs.size(0)
        if should_log_batch(batch_index, total_batches, log_every):
            average_loss = total_loss / max(1, total_items)
            print(
                f"[epoch {epoch}/{total_epochs}] batch {batch_index}/{total_batches} "
                f"loss={loss.item():.4f} avg_loss={average_loss:.4f}",
                flush=True,
            )
    return total_loss / max(1, total_items)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], list[int]]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            batch_probabilities = score_logits_to_probabilities(logits).detach().cpu().tolist()
            probabilities.extend(float(value) for value in batch_probabilities)
            labels.extend(int(value) for value in batch_labels.tolist())
    return probabilities, labels


def print_run_header(
    args: argparse.Namespace,
    device: torch.device,
    train_entries: list,
    val_entries: list,
    test_entries: list,
    run_dir: Path,
) -> None:
    positives = sum(1 for entry in train_entries if entry.label == "milady")
    negatives = len(train_entries) - positives
    print(
        f"[setup] run_id={args.run_id} device={device.type} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.learning_rate:g} "
        f"weight_decay={args.weight_decay:g} patience={args.patience} precision_floor={args.precision_floor:.4f}",
        flush=True,
    )
    print(
        f"[setup] splits train={len(train_entries)} val={len(val_entries)} test={len(test_entries)} "
        f"train_milady={positives} train_not_milady={negatives}",
        flush=True,
    )
    print(f"[setup] artifacts={run_dir}", flush=True)


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    threshold: float,
    threshold_metrics: dict[str, float],
    improved: bool,
    stale_epochs: int,
    patience: int,
) -> None:
    status = "best" if improved else f"stale={stale_epochs}/{patience}"
    print(
        f"[epoch {epoch}/{total_epochs}] "
        f"train_loss={train_loss:.4f} "
        f"val_precision={threshold_metrics['precision']:.4f} "
        f"val_recall={threshold_metrics['recall']:.4f} "
        f"val_f1={threshold_metrics['f1']:.4f} "
        f"threshold={threshold:.4f} "
        f"{status}",
        flush=True,
    )


def should_log_batch(batch_index: int, total_batches: int, log_every: int) -> bool:
    if batch_index == 1 or batch_index == total_batches:
        return True
    if log_every <= 0:
        return False
    return batch_index % log_every == 0


if __name__ == "__main__":
    main()
