"""Evaluate the saved U-Net once on the held-out contiguous test block."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "models" / "best_unet.pt",
        help="Trained U-Net checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _score(intersection: int, prediction_count: int, target_count: int) -> tuple[float, float]:
    union = prediction_count + target_count - intersection
    dice = (
        1.0
        if prediction_count + target_count == 0
        else 2.0 * intersection / (prediction_count + target_count)
    )
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    tint = np.asarray(color, dtype=np.float32)
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * tint
    return np.clip(rgb, 0, 1)


def _save_overlay(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    slice_index: int,
    dice: float,
    iou: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"MRI slice {slice_index}")
    axes[1].imshow(_overlay(image, target, (0.1, 0.9, 0.2)))
    axes[1].set_title("Ground truth")
    axes[2].imshow(_overlay(image, prediction, (1.0, 0.15, 0.1)))
    axes[2].set_title(f"Prediction\nDice {dice:.3f} | IoU {iou:.3f}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_montage(
    rows: list[tuple[int, float, Path]],
    output_path: Path,
) -> None:
    selected = sorted(rows, key=lambda row: row[1])[: min(8, len(rows))]
    figure, axes = plt.subplots(len(selected), 1, figsize=(9, 3.2 * len(selected)))
    axes_list = np.atleast_1d(axes)
    for axis, (slice_index, dice, image_path) in zip(axes_list, selected):
        image = plt.imread(image_path)
        axis.imshow(image)
        axis.set_title(f"Slice {slice_index} | Dice {dice:.3f}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


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

    pairs = discover_pairs(args.dataset)
    test_pairs = split_contiguous(pairs)["test"]
    test_dataset = KneeMRISliceDataset(
        test_pairs,
        image_size=int(checkpoint["image_size"]),
    )
    loader = DataLoader(
        test_dataset,
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
    prediction_dir = results_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str | float]] = []
    montage_rows: list[tuple[int, float, Path]] = []
    total_intersection = 0
    total_prediction = 0
    total_target = 0
    example_offset = 0

    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            predictions = torch.sigmoid(logits).cpu().numpy()[:, 0] >= args.threshold
            image_arrays = images.numpy()[:, 0]
            target_arrays = masks.numpy()[:, 0] >= 0.5

            for offset, (image, target, prediction) in enumerate(
                zip(image_arrays, target_arrays, predictions)
            ):
                pair = test_pairs[example_offset + offset]
                intersection = int(np.logical_and(target, prediction).sum())
                prediction_count = int(prediction.sum())
                target_count = int(target.sum())
                dice, iou = _score(intersection, prediction_count, target_count)
                total_intersection += intersection
                total_prediction += prediction_count
                total_target += target_count
                rows.append(
                    {
                        "slice_index": pair.slice_index,
                        "dicom_file": pair.image_path.name,
                        "mask_file": pair.mask_path.name,
                        "dice": dice,
                        "iou": iou,
                        "target_pixels": target_count,
                        "predicted_pixels": prediction_count,
                    }
                )
                overlay_path = prediction_dir / f"test_slice_{pair.slice_index:03d}.png"
                _save_overlay(image, target, prediction, pair.slice_index, dice, iou, overlay_path)
                montage_rows.append((pair.slice_index, dice, overlay_path))
            example_offset += len(images)

    if len(rows) != len(test_pairs):
        raise RuntimeError(
            f"Evaluated {len(rows)} slices, but the test split contains {len(test_pairs)}"
        )

    mean_slice_dice = float(np.mean([float(row["dice"]) for row in rows]))
    mean_slice_iou = float(np.mean([float(row["iou"]) for row in rows]))
    pooled_dice, pooled_iou = _score(
        total_intersection,
        total_prediction,
        total_target,
    )
    csv_path = results_dir / "test_metrics_per_slice.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    montage_path = results_dir / "test_predictions_montage.png"
    _save_montage(montage_rows, montage_path)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "threshold": args.threshold,
        "image_size": int(checkpoint["image_size"]),
        "test_slice_count": len(rows),
        "test_slice_indices": [int(row["slice_index"]) for row in rows],
        "mean_per_slice_dice": mean_slice_dice,
        "mean_per_slice_iou": mean_slice_iou,
        "pooled_pixel_dice": pooled_dice,
        "pooled_pixel_iou": pooled_iou,
        "test_split_used_for_training_or_model_selection": False,
        "single_volume_exploratory_evaluation": True,
        "per_slice_metrics_csv": str(csv_path),
        "prediction_overlays_directory": str(prediction_dir),
        "prediction_montage": str(montage_path),
    }
    summary_path = results_dir / "test_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Evaluated {len(rows)} held-out slices on {device}")
    print(f"Mean per-slice Dice: {mean_slice_dice:.4f}")
    print(f"Mean per-slice IoU:  {mean_slice_iou:.4f}")
    print(f"Pooled pixel Dice:   {pooled_dice:.4f}")
    print(f"Pooled pixel IoU:    {pooled_iou:.4f}")
    print(f"Per-slice metrics: {csv_path}")
    print(f"Prediction montage: {montage_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
