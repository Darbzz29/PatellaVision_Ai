"""Measure model sensitivity to controlled test-image quality changes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import KneeMRISliceDataset, discover_pairs, split_contiguous
    from src.model import UNet
else:
    from .data import KneeMRISliceDataset, discover_pairs, split_contiguous
    from .model import UNet


CONDITIONS: tuple[tuple[str, str, float], ...] = (
    ("clean", "Unmodified", 0.0),
    ("noise_005", "Gaussian noise (sigma 0.05)", 0.05),
    ("noise_010", "Gaussian noise (sigma 0.10)", 0.10),
    ("contrast_075", "Reduced contrast (factor 0.75)", 0.75),
    ("contrast_050", "Reduced contrast (factor 0.50)", 0.50),
    ("blur_1", "Gaussian blur (sigma 1 px)", 1.0),
    ("blur_2", "Gaussian blur (sigma 2 px)", 2.0),
)


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "models" / "best_unet.pt",
        help="Trained U-Net checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _score(
    intersection: int,
    prediction_count: int,
    target_count: int,
) -> tuple[float, float]:
    union = prediction_count + target_count - intersection
    dice = (
        1.0
        if prediction_count + target_count == 0
        else 2.0 * intersection / (prediction_count + target_count)
    )
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


def _gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=images.dtype,
        device=images.device,
    )
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d[None, None, :, :]
    padded = nn.functional.pad(
        images,
        (radius, radius, radius, radius),
        mode="reflect",
    )
    return nn.functional.conv2d(padded, kernel).clamp(0, 1)


def _perturb(
    images: torch.Tensor,
    condition: str,
    severity: float,
    slice_indices: list[int],
    seed: int,
) -> torch.Tensor:
    if condition == "clean":
        return images
    if condition.startswith("noise_"):
        noise = torch.empty_like(images)
        for offset, slice_index in enumerate(slice_indices):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + slice_index)
            noise[offset] = torch.randn(
                images[offset].shape,
                generator=generator,
                dtype=images.dtype,
                device="cpu",
            )
        return (images + severity * noise).clamp(0, 1)
    if condition.startswith("contrast_"):
        return ((images - 0.5) * severity + 0.5).clamp(0, 1)
    if condition.startswith("blur_"):
        return _gaussian_blur(images, severity)
    raise ValueError(f"Unknown perturbation condition: {condition}")


def _overlay(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    base = np.repeat(image[:, :, None], 3, axis=2)
    result = base.copy()
    result[target] = 0.55 * result[target] + 0.45 * np.array([0.1, 0.9, 0.2])
    result[prediction] = 0.55 * result[prediction] + 0.45 * np.array([1.0, 0.15, 0.1])
    overlap = target & prediction
    result[overlap] = 0.55 * base[overlap] + 0.45 * np.array([1.0, 0.8, 0.05])
    return np.clip(result, 0, 1)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    required_keys = {
        "model_state_dict",
        "base_channels",
        "image_size",
        "in_channels",
        "out_channels",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"Checkpoint is missing metadata: {sorted(missing_keys)}")

    test_pairs = split_contiguous(discover_pairs(args.dataset))["test"]
    dataset = KneeMRISliceDataset(
        test_pairs,
        image_size=int(checkpoint["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = UNet(
        in_channels=int(checkpoint["in_channels"]),
        out_channels=int(checkpoint["out_channels"]),
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    output_dir = results_dir / "reliability"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_slice_rows: list[dict[str, int | str | float]] = []
    summary_rows: list[dict[str, int | str | float]] = []
    worst_clean_slice = -1
    worst_clean_dice = float("inf")
    worst_slice_images: dict[str, np.ndarray] = {}
    worst_slice_target: np.ndarray | None = None

    with torch.inference_mode():
        for condition, label, severity in CONDITIONS:
            total_intersection = 0
            total_prediction = 0
            total_target = 0
            slice_dice_values: list[float] = []
            slice_iou_values: list[float] = []
            pair_offset = 0

            for images, masks in loader:
                batch_pairs = test_pairs[pair_offset : pair_offset + len(images)]
                slice_indices = [pair.slice_index for pair in batch_pairs]
                changed_images = _perturb(
                    images,
                    condition,
                    severity,
                    slice_indices,
                    args.seed,
                )
                logits = model(changed_images.to(device))
                predictions = torch.sigmoid(logits).cpu().numpy()[:, 0] >= args.threshold
                image_arrays = changed_images.numpy()[:, 0]
                target_arrays = masks.numpy()[:, 0] >= 0.5

                for image, target, prediction, pair in zip(
                    image_arrays,
                    target_arrays,
                    predictions,
                    batch_pairs,
                ):
                    intersection = int(np.logical_and(target, prediction).sum())
                    prediction_count = int(prediction.sum())
                    target_count = int(target.sum())
                    dice, iou = _score(
                        intersection,
                        prediction_count,
                        target_count,
                    )
                    total_intersection += intersection
                    total_prediction += prediction_count
                    total_target += target_count
                    slice_dice_values.append(dice)
                    slice_iou_values.append(iou)
                    per_slice_rows.append(
                        {
                            "condition": condition,
                            "condition_label": label,
                            "severity": severity,
                            "slice_index": pair.slice_index,
                            "dice": dice,
                            "iou": iou,
                            "target_pixels": target_count,
                            "predicted_pixels": prediction_count,
                        }
                    )
                    if condition == "clean" and dice < worst_clean_dice:
                        worst_clean_dice = dice
                        worst_clean_slice = pair.slice_index
                        worst_slice_target = target.copy()
                        worst_slice_images = {}
                    if pair.slice_index == worst_clean_slice:
                        worst_slice_images[condition] = image.copy()
                        worst_slice_images[f"{condition}_prediction"] = prediction.copy()
                pair_offset += len(images)

            if pair_offset != len(test_pairs):
                raise RuntimeError(
                    f"Condition {condition} evaluated {pair_offset} slices, "
                    f"expected {len(test_pairs)}"
                )
            pooled_dice, pooled_iou = _score(
                total_intersection,
                total_prediction,
                total_target,
            )
            summary_rows.append(
                {
                    "condition": condition,
                    "condition_label": label,
                    "severity": severity,
                    "mean_per_slice_dice": float(np.mean(slice_dice_values)),
                    "mean_per_slice_iou": float(np.mean(slice_iou_values)),
                    "pooled_pixel_dice": pooled_dice,
                    "pooled_pixel_iou": pooled_iou,
                }
            )

    if worst_slice_target is None or len(worst_slice_images) != len(CONDITIONS) * 2:
        raise RuntimeError("Could not assemble the worst-clean-slice comparison")

    csv_path = output_dir / "reliability_per_slice.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_slice_rows[0]))
        writer.writeheader()
        writer.writerows(per_slice_rows)

    clean_dice = float(summary_rows[0]["mean_per_slice_dice"])
    clean_iou = float(summary_rows[0]["mean_per_slice_iou"])
    for row in summary_rows:
        row["dice_change_vs_clean"] = float(row["mean_per_slice_dice"]) - clean_dice
        row["iou_change_vs_clean"] = float(row["mean_per_slice_iou"]) - clean_iou

    summary_csv_path = output_dir / "reliability_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [str(row["condition"]) for row in summary_rows]
    x_positions = np.arange(len(labels))
    for axis, metric, title in (
        (axes[0], "mean_per_slice_dice", "Mean per-slice Dice"),
        (axes[1], "mean_per_slice_iou", "Mean per-slice IoU"),
    ):
        values = [float(row[metric]) for row in summary_rows]
        axis.bar(x_positions, values, color="#4472C4")
        axis.axhline(values[0], color="#C00000", linestyle="--", label="Clean baseline")
        axis.set_xticks(x_positions, labels, rotation=35, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    chart_path = output_dir / "reliability_comparison.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(len(CONDITIONS), 1, figsize=(9, 2.8 * len(CONDITIONS)))
    for axis, (condition, label, _) in zip(axes, CONDITIONS):
        image = worst_slice_images[condition]
        prediction = worst_slice_images[f"{condition}_prediction"]
        axis.imshow(_overlay(image, worst_slice_target, prediction))
        score_row = next(
            row
            for row in per_slice_rows
            if row["condition"] == condition
            and row["slice_index"] == worst_clean_slice
        )
        axis.set_title(
            f"{label} | slice {worst_clean_slice} | "
            f"Dice {float(score_row['dice']):.3f}, IoU {float(score_row['iou']):.3f}"
        )
        axis.axis("off")
    figure.tight_layout()
    montage_path = output_dir / "reliability_worst_slice.png"
    figure.savefig(montage_path, dpi=140)
    plt.close(figure)

    summary = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "test_slice_count": len(test_pairs),
        "test_slice_indices": [pair.slice_index for pair in test_pairs],
        "threshold": args.threshold,
        "image_size": int(checkpoint["image_size"]),
        "seed": args.seed,
        "perturbations": {
            "noise": "Additive zero-mean Gaussian noise after normalization; clip to [0, 1]",
            "contrast": "Multiply intensity deviations from 0.5, then clip to [0, 1]",
            "blur": "Gaussian blur applied to normalized resized images",
        },
        "conditions": summary_rows,
        "worst_clean_slice": worst_clean_slice,
        "test_data_used_for_model_training_or_selection": False,
        "single_volume_exploratory_experiment": True,
        "per_slice_metrics_csv": str(csv_path),
        "summary_csv": str(summary_csv_path),
        "comparison_plot": str(chart_path),
        "worst_slice_montage": str(montage_path),
    }
    summary_path = output_dir / "reliability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Evaluated {len(test_pairs)} held-out slices across {len(CONDITIONS)} conditions")
    print("Condition                 mean Dice   mean IoU   change in Dice")
    for row in summary_rows:
        print(
            f"{str(row['condition']):24s} "
            f"{float(row['mean_per_slice_dice']):.4f}      "
            f"{float(row['mean_per_slice_iou']):.4f}      "
            f"{float(row['dice_change_vs_clean']):+.4f}"
        )
    print(f"Worst clean slice: {worst_clean_slice} (Dice {worst_clean_dice:.4f})")
    print(f"Summary: {summary_path}")
    print(f"Comparison plot: {chart_path}")
    print(f"Slice comparisons: {montage_path}")


if __name__ == "__main__":
    main()
