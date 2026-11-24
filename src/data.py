"""Dataset pairing, preprocessing, and leakage-aware slice splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SlicePair:
    """A DICOM slice and its corresponding ground-truth mask."""

    image_path: Path
    mask_path: Path
    slice_index: int


def discover_pairs(dataset_root: str | Path) -> list[SlicePair]:
    """Pair DICOM suffix _0.._N with Mask_001..Mask_N by numeric index."""
    root = Path(dataset_root)
    image_dir = root / "Data"
    mask_dir = root / "GT-Masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    dicom_by_index: dict[int, Path] = {}
    for path in image_dir.glob("*.dcm"):
        match = re.search(r"_(\d+)$", path.stem)
        if match is None:
            raise ValueError(f"Cannot read slice index from DICOM filename: {path.name}")
        index = int(match.group(1))
        if index in dicom_by_index:
            raise ValueError(f"Duplicate DICOM slice index {index}")
        dicom_by_index[index] = path

    mask_by_index: dict[int, Path] = {}
    for path in mask_dir.glob("Mask_*.png"):
        match = re.fullmatch(r"Mask_(\d+)", path.stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot read slice index from mask filename: {path.name}")
        index = int(match.group(1))
        if index in mask_by_index:
            raise ValueError(f"Duplicate mask index {index}")
        mask_by_index[index] = path

    if not dicom_by_index:
        raise ValueError(f"No .dcm files found in {image_dir}")
    if not mask_by_index:
        raise ValueError(f"No Mask_*.png files found in {mask_dir}")

    expected_dicom_indices = set(range(len(dicom_by_index)))
    expected_mask_indices = set(range(1, len(mask_by_index) + 1))
    if set(dicom_by_index) != expected_dicom_indices:
        raise ValueError("DICOM slice indices must be consecutive and start at 0")
    if set(mask_by_index) != expected_mask_indices:
        raise ValueError("Mask indices must be consecutive and start at 1")
    if len(dicom_by_index) != len(mask_by_index):
        raise ValueError(
            f"Found {len(dicom_by_index)} DICOM files and {len(mask_by_index)} masks"
        )

    return [
        SlicePair(
            image_path=dicom_by_index[index],
            mask_path=mask_by_index[index + 1],
            slice_index=index,
        )
        for index in range(len(dicom_by_index))
    ]


def split_contiguous(
    pairs: list[SlicePair],
    *,
    gap_slices: int = 2,
) -> dict[str, list[SlicePair]]:
    """Split one ordered volume into contiguous train/validation/test blocks."""
    if gap_slices < 0:
        raise ValueError("gap_slices cannot be negative")
    usable_count = len(pairs) - 2 * gap_slices
    if usable_count < 20:
        raise ValueError("At least 20 usable slices are required for this split")

    train_end = int(usable_count * 0.70)
    validation_end = int(usable_count * 0.85)
    validation_start = train_end + gap_slices
    test_start = validation_end + 2 * gap_slices
    result = {
        "train": pairs[:train_end],
        "validation": pairs[validation_start : validation_end + gap_slices],
        "test": pairs[test_start:],
    }
    if any(not subset for subset in result.values()):
        raise ValueError("The selected split produced an empty subset")
    return result


def normalize_mri_array(
    pixel_array: np.ndarray,
    photometric_interpretation: str = "MONOCHROME2",
) -> np.ndarray:
    """Apply the training pipeline's orientation and intensity normalization."""
    image = np.asarray(pixel_array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D MRI slice, got shape {image.shape}")
    if photometric_interpretation == "MONOCHROME1":
        image = image.max() - image

    low, high = np.percentile(image, (1, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("MRI slice has no usable intensity range")
    image = np.clip(image, low, high)
    return ((image - low) / (high - low)).astype(np.float32)


def _read_mri(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path))
    return normalize_mri_array(
        dataset.pixel_array,
        dataset.get("PhotometricInterpretation", "MONOCHROME2"),
    )


class KneeMRISliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load paired 2D MRI slices and binary masks as model-ready tensors."""

    def __init__(
        self,
        pairs: list[SlicePair],
        image_size: int = 256,
    ) -> None:
        if image_size < 1:
            raise ValueError("image_size must be a positive integer")
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        image = torch.from_numpy(_read_mri(pair.image_path))[None, None, :, :]
        with Image.open(pair.mask_path) as mask_file:
            mask_array = np.asarray(mask_file.convert("L"), dtype=np.uint8)
        if mask_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D ground-truth mask in {pair.mask_path}, "
                f"got shape {mask_array.shape}"
            )
        mask = torch.from_numpy((mask_array > 0).astype(np.float32))[None, None, :, :]
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                f"Image and mask dimensions differ for slice {pair.slice_index}: "
                f"{tuple(image.shape[-2:])} vs {tuple(mask.shape[-2:])}"
            )

        image = functional.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mask = functional.interpolate(
            mask,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        return image.squeeze(0), mask.squeeze(0)
