"""Train the U-Net on contiguous training slices and select by validation Dice."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import KneeMRISliceDataset, discover_pairs, split_contiguous
    from src.model import UNet
else:
    from .data import KneeMRISliceDataset, discover_pairs, split_contiguous
    from .model import UNet


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root
        / "dataset"
        / "diego-romano-KneeBones3Dify-Annotated-Dataset-01f567d",
        help="Dataset folder containing Data and GT-Masks subfolders",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice.mean()


def batch_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5
    dimensions = tuple(range(1, predictions.ndim))
    intersection = (predictions & targets).sum(dim=dimensions).float()
    denominator = predictions.sum(dim=dimensions) + targets.sum(dim=dimensions)
    scores = (2 * intersection + 1e-6) / (denominator.float() + 1e-6)
    return float(scores.mean().item())


def save_learning_curves(
    history: list[dict[str, int | float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="Train")
    axes[0].plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="Validation",
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE + soft Dice loss")
    axes[1].plot(epochs, [float(row["train_dice"]) for row in history], label="Train")
    axes[1].plot(
        epochs,
        [float(row["validation_dice"]) for row in history],
        label="Validation",
    )
    axes[1].set(title="Binary Dice", xlabel="Epoch", ylabel="Mean Dice")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_epoch(
    model: UNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_dice = 0.0
    sample_count = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, masks) + soft_dice_loss(logits, masks)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_count = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_count
        total_dice += batch_dice(logits.detach(), masks) * batch_count
        sample_count += batch_count

    if sample_count == 0:
        raise ValueError("A data loader produced no samples")
    return total_loss / sample_count, total_dice / sample_count


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch-size, and threads must be positive")
    if args.image_size < 32 or args.image_size % 16 != 0:
        raise ValueError("image-size must be at least 32 and divisible by 16")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = discover_pairs(args.dataset)
    splits = split_contiguous(pairs)
    train_dataset = KneeMRISliceDataset(splits["train"], image_size=args.image_size)
    validation_dataset = KneeMRISliceDataset(
        splits["validation"], image_size=args.image_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    results_dir = project_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_unet.pt"
    history_path = results_dir / "training_history.csv"
    graph_path = results_dir / "graphs" / "training_curves.png"
    summary_path = results_dir / "training_summary.json"

    model = UNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_validation_dice = -1.0
    history: list[dict[str, float | int]] = []

    print(
        f"Device: {device}; slices: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}; test={len(splits['test'])} (not used)"
    )
    print(
        f"Training: epochs={args.epochs}, image_size={args.image_size}, "
        f"batch_size={args.batch_size}, base_channels={args.base_channels}"
    )

    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_dice = run_epoch(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, Dice {train_dice:.4f} | "
            f"validation loss {validation_loss:.4f}, Dice {validation_dice:.4f}"
        )

        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": "UNet",
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": args.base_channels,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "validation_dice": validation_dice,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    save_learning_curves(history, graph_path)
    elapsed_seconds = time.time() - started_at
    summary = {
        "device": str(device),
        "dataset": str(args.dataset.resolve()),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "test_split_used_for_training_or_selection": False,
        "epochs_completed": args.epochs,
        "best_epoch": max(history, key=lambda row: float(row["validation_dice"]))[
            "epoch"
        ],
        "best_validation_dice": best_validation_dice,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "learning_curves": str(graph_path),
        "single_volume_exploratory_split": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"Summary: {summary_path}")
    print(f"Elapsed: {elapsed_seconds / 60:.1f} minutes")


if __name__ == "__main__":
    main()
