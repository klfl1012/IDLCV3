"""Segmentation dataset utilities for DRIVE and PH2 datasets."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F


# Default dataset roots provided by the user.
DEFAULT_DATA_ROOTS = {
	"drive": Path("/dtu/datasets1/02516/DRIVE"),
	"ph2": Path("/dtu/datasets1/02516/PH2_Dataset_images"),
}


@dataclass(frozen=True)
class SegmentationSample:
	image_path: Path
	mask_path: Path


def _discover_ph2_samples(root: Path) -> List[SegmentationSample]:
	"""Collect (image, mask) pairs from the PH2 dataset layout."""

	if not root.is_dir():
		raise FileNotFoundError(f"PH2 root directory not found: {root}")

	samples: List[SegmentationSample] = []
	for patient_dir in sorted(root.iterdir()):
		if not patient_dir.is_dir() or not patient_dir.name.upper().startswith("IMD"):
			continue

		image_dir = next(
			(p for p in patient_dir.iterdir() if p.is_dir() and "dermoscopic" in p.name.lower()),
			None,
		)
		mask_dir = next(
			(p for p in patient_dir.iterdir() if p.is_dir() and "lesion" in p.name.lower()),
			None,
		)

		if image_dir is None or mask_dir is None:
			raise FileNotFoundError(
				f"Expected dermoscopic_image and lesion folders inside {patient_dir}."
			)

		def _first_image(folder: Path) -> Optional[Path]:
			for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
				matches = sorted(folder.glob(ext))
				if matches:
					return matches[0]
			return None

		image_path = _first_image(image_dir)
		mask_path = _first_image(mask_dir)

		if image_path is None or mask_path is None:
			raise FileNotFoundError(
				f"Unable to find image/mask pair in {patient_dir}."
			)

		samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))

	if not samples:
		raise ValueError(f"No PH2 samples found under {root}")

	return samples


def _discover_drive_samples(root: Path, split: str) -> List[SegmentationSample]:
	"""Collect (image, mask) pairs from the DRIVE dataset layout."""

	split = split.lower()
	if split not in {"training", "train", "testing", "test"}:
		raise ValueError("DRIVE split must be 'train'|'training' or 'test'|'testing'.")

	split_dir = "training" if split.startswith("train") else "test"

	images_dir = root / split_dir / "images"
	masks_dir = root / split_dir / "1st_manual"

	if not images_dir.is_dir():
		raise FileNotFoundError(f"DRIVE images directory not found: {images_dir}")
	if not masks_dir.is_dir():
		raise FileNotFoundError(f"DRIVE manual masks directory not found: {masks_dir}")

	samples: List[SegmentationSample] = []
	for image_path in sorted(images_dir.glob("*.tif")):
		stem = image_path.name.replace("_training", "").replace("_test", "")
		mask_candidates = list(masks_dir.glob(f"{stem.split('.')[0]}*_manual1.*"))
		if not mask_candidates:
			# Fallback to exact name with manual1.gif pattern (e.g. 21_manual1.gif)
			stem_no_suffix = image_path.stem.split("_")[0]
			mask_candidates = list(masks_dir.glob(f"{stem_no_suffix}_manual1.*"))

		if not mask_candidates:
			raise FileNotFoundError(
				f"Mask for DRIVE image {image_path.name} not found under {masks_dir}."
			)

		samples.append(SegmentationSample(image_path=image_path, mask_path=mask_candidates[0]))

	if not samples:
		raise ValueError(f"No DRIVE samples found in split '{split_dir}'.")

	return samples


class SegmentationDataset(Dataset):
	"""Generic semantic segmentation dataset supporting DRIVE and PH2."""

	def __init__(
		self,
		dataset: str,
		*,
		split: str = "train",
		root: Optional[Path] = None,
		image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
		mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
		mask_threshold: float = 0.5,
		resize_to: Optional[Tuple[int, int]] = None,
		augment: bool = False,
		brightness_delta: float = 0.15,
	) -> None:
		dataset_key = dataset.lower()
		if dataset_key not in {"drive", "ph2"}:
			raise ValueError("dataset must be either 'drive' or 'ph2'.")

		root_path = Path(root) if root is not None else DEFAULT_DATA_ROOTS[dataset_key]
		if dataset_key == "ph2":
			samples = _discover_ph2_samples(root_path)
		else:
			samples = _discover_drive_samples(root_path, split)

		self.dataset = dataset_key
		self.samples = samples
		self.mask_threshold = mask_threshold
		self.augment = bool(augment)
		self.brightness_delta = float(brightness_delta)

		default_image_transforms: List[Callable] = []
		default_mask_transforms: List[Callable] = []
		if resize_to is not None:
			default_image_transforms.append(
				transforms.Resize(resize_to, interpolation=InterpolationMode.BILINEAR, antialias=True)
			)
			default_mask_transforms.append(
				transforms.Resize(resize_to, interpolation=InterpolationMode.NEAREST)
			)
		default_image_transforms.append(transforms.ToTensor())
		default_mask_transforms.append(transforms.ToTensor())

		self.image_transform = image_transform or transforms.Compose(default_image_transforms)
		self.mask_transform = mask_transform or transforms.Compose(default_mask_transforms)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		sample = self.samples[index]

		image = Image.open(sample.image_path).convert("RGB")
		mask = Image.open(sample.mask_path).convert("L")

		if self.augment:
			image, mask = self._apply_augmentations(image, mask)

		image_tensor = self.image_transform(image)
		mask_tensor = self.mask_transform(mask)

		# Ensure mask is (H, W) and binary/integer class labels.
		if mask_tensor.ndim == 3:
			mask_tensor = mask_tensor.squeeze(0)
		mask_tensor = (mask_tensor >= self.mask_threshold).long()

		return image_tensor, mask_tensor

	def _apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
		"""Apply deterministic augmentations to the image/mask pair."""

		# Horizontal flip
		if random.random() < 0.5:
			image = F.hflip(image)
			mask = F.hflip(mask)

		# Vertical flip
		if random.random() < 0.5:
			image = F.vflip(image)
			mask = F.vflip(mask)

		# Brightness jitter (image only)
		if self.brightness_delta > 0:
			factor = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
			image = F.adjust_brightness(image, max(factor, 0.0))

		return image, mask


def build_segmentation_dataloader(
	dataset: str,
	*,
	split: str = "train",
	root: Optional[Path] = None,
	batch_size: int = 4,
	shuffle: bool = True,
	num_workers: int = os.cpu_count() or 4,
	image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
	mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
	pin_memory: bool = True,
	resize_to: Optional[Tuple[int, int]] = None,
	augment: bool = False,
	brightness_delta: float = 0.15,
) -> DataLoader:
	"""Create a DataLoader for the requested segmentation dataset."""

	dataset_obj = SegmentationDataset(
		dataset=dataset,
		split=split,
		root=root,
		image_transform=image_transform,
		mask_transform=mask_transform,
		resize_to=resize_to,
		augment=augment,
		brightness_delta=brightness_delta,
	)

	return DataLoader(
		dataset_obj,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


__all__ = [
	"SegmentationDataset",
	"build_segmentation_dataloader",
]

