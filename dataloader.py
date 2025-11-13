"""Segmentation dataset utilities for DRIVE and PH2 datasets."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F


# Default dataset roots provided by the user.
DEFAULT_DATA_ROOTS = {
	"drive": Path("/dtu/datasets1/02516/DRIVE"),
	"ph2": Path("/dtu/datasets1/02516/PH2_Dataset_images"),
	"weak_labels": Path("./region_growing_masks/"),
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
    
    # IMPORTANT: DRIVE test folder has NO ground truth labels (1st_manual)!
    # We MUST use training folder for all train/val/test splits
    if split not in {"training", "train", "val", "validation", "test", "testing"}:
        raise ValueError("DRIVE split must be 'train', 'val', or 'test'.")

    # Always use training folder because it's the only one with 1st_manual labels
    split_dir = "training"

    images_dir = root / split_dir / "images"
    masks_dir = root / split_dir / "1st_manual"  # Ground truth labels

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


def _discover_weak_label_samples(root: Path, subfolder: str) -> List[SegmentationSample]:
    """Collect (image, mask) pairs from weak labels folder structure.
    
    Masks are in region_growing_masks/<subfolder>/
    Images come from the PH2 dataset.
    
    Expected mask naming: IMD002_mask.bmp, IMD003_mask.bmp, etc.
    We extract the patient ID (e.g., IMD002) and find the corresponding image in PH2 dataset.
    """
    mask_folder = root / subfolder
    
    if not mask_folder.is_dir():
        raise FileNotFoundError(f"Weak labels folder not found: {mask_folder}")
    
    # Get PH2 dataset root
    ph2_root = DEFAULT_DATA_ROOTS['ph2']
    if not ph2_root.is_dir():
        raise FileNotFoundError(f"PH2 dataset not found at: {ph2_root}")
    
    # Collect all mask files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    mask_files = sorted([f for f in mask_folder.iterdir() 
                        if f.suffix.lower() in image_extensions 
                        and 'mask' in f.stem.lower()])
    
    if not mask_files:
        raise ValueError(f"No mask files found in {mask_folder}")
    
    samples: List[SegmentationSample] = []
    
    for mask_path in mask_files:
        # Extract patient ID from mask filename
        # Example: IMD002_mask.bmp -> IMD002
        mask_stem = mask_path.stem
        patient_id = mask_stem.replace('_mask', '').replace('_label', '').upper()
        
        # Find corresponding image in PH2 dataset
        # Structure: PH2_Dataset_images/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp
        patient_dir = ph2_root / patient_id
        
        if not patient_dir.is_dir():
            print(f"Warning: Patient directory not found for {patient_id}, skipping...")
            continue
        
        # Find dermoscopic image folder
        image_dir = next(
            (p for p in patient_dir.iterdir() 
             if p.is_dir() and 'dermoscopic' in p.name.lower()),
            None,
        )
        
        if image_dir is None:
            print(f"Warning: Dermoscopic image folder not found for {patient_id}, skipping...")
            continue
        
        # Find the image file
        image_path = None
        for ext in image_extensions:
            candidates = list(image_dir.glob(f"{patient_id}{ext}"))
            if candidates:
                image_path = candidates[0]
                break
        
        if image_path is None:
            print(f"Warning: Image file not found for {patient_id}, skipping...")
            continue
        
        samples.append(SegmentationSample(image_path=image_path, mask_path=mask_path))
    
    if not samples:
        raise ValueError(f"No valid image-mask pairs found in {mask_folder}")
    
    print(f"Found {len(samples)} weak label samples in {subfolder}/")
    return samples

def _split_samples(
	samples: List[SegmentationSample],
	split: str,
	train_ratio: float = 0.7,
	val_ratio: float = 0.15,
	seed: int = 21,
) -> List[SegmentationSample]:
	
	split_lower = split.lower()
	
	# Special case: return all samples without splitting
	if split_lower == 'all':
		return samples
	
	n_total = len(samples)
	n_train = int(n_total * train_ratio)
	n_val = int(n_total * val_ratio)
	n_test = n_total - n_train - n_val

	indices = list(range(n_total))
	generator = torch.Generator().manual_seed(seed)
	shuffled_indices = torch.randperm(n_total, generator=generator).tolist()

	train_indices = shuffled_indices[:n_train]
	val_indices = shuffled_indices[n_train:n_train + n_val]
	test_indices = shuffled_indices[n_train + n_val:]
	
	if split_lower in {"train", "training"}:
		return [samples[i] for i in train_indices]
	elif split_lower in {"val", "validation"}:
		return [samples[i] for i in val_indices]
	elif split_lower in {"test", "testing"}:
		return [samples[i] for i in test_indices]
	else:
		raise ValueError(f"Unknown split {split}.")
	

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
		seed: int = 21,
		weak_label_subfolder: Optional[str] = None,
	) -> None:
		dataset_key = dataset.lower()
		if dataset_key not in {"drive", "ph2", "weak_labels"}:
			raise ValueError("dataset must be 'drive', 'ph2', or 'weak_labels'.")

		root_path = Path(root) if root is not None else DEFAULT_DATA_ROOTS[dataset_key]

		if dataset_key == "weak_labels":
			# Weak labels: load from specific subfolder
			if weak_label_subfolder is None:
				raise ValueError("weak_label_subfolder must be specified for weak_labels dataset")
			all_samples = _discover_weak_label_samples(root_path, weak_label_subfolder)
			samples = _split_samples(all_samples, split, seed=seed)
			if split.lower() == 'all':
				print(f"Weak labels ({weak_label_subfolder}): Using ALL {len(samples)} samples (no split)")
			else:
				print(f"Weak labels ({weak_label_subfolder}) {split} split: {len(samples)} samples (seed={seed})")
		elif dataset_key == "ph2":
            # PH2: Get all samples, then split train/val/test
			all_samples = _discover_ph2_samples(root_path)
			samples = _split_samples(all_samples, split, seed=seed)
			print(f"PH2 {split} split: {len(samples)} samples (seed={seed})")
		else:
            # DRIVE: Only training/ has ground truth (1st_manual)
            # We split the 20 training images into train/val/test
			if split.lower() in {'train', 'training'}:
				all_train_samples = _discover_drive_samples(root_path, 'train')
				samples = _split_samples(all_train_samples, 'train', train_ratio=0.7, val_ratio=0.15, seed=seed)
				print(f"DRIVE train split: {len(samples)}/{len(all_train_samples)} samples (seed={seed})")
			elif split.lower() in {'val', 'validation'}:
				all_train_samples = _discover_drive_samples(root_path, 'train')
				samples = _split_samples(all_train_samples, 'val', train_ratio=0.7, val_ratio=0.15, seed=seed)
				print(f"DRIVE val split: {len(samples)}/{len(all_train_samples)} samples (seed={seed})")
			elif split.lower() in {'test', 'testing'}:
				all_train_samples = _discover_drive_samples(root_path, 'train')
				samples = _split_samples(all_train_samples, 'test', train_ratio=0.7, val_ratio=0.15, seed=seed)
				print(f"DRIVE test split: {len(samples)}/{len(all_train_samples)} samples (seed={seed})")
			else:
				raise ValueError(f"Unknown split: {split}")

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

		# Type assertions to ensure tensors after transforms
		assert isinstance(image_tensor, torch.Tensor), "image_transform must return a Tensor"
		assert isinstance(mask_tensor, torch.Tensor), "mask_transform must return a Tensor"

		# Ensure mask is (H, W) and binary/integer class labels.
		if mask_tensor.ndim == 3: 
			mask_tensor = mask_tensor.squeeze(0)
		mask_tensor = (mask_tensor >= self.mask_threshold).long()

		return image_tensor, mask_tensor

	def _apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
		"""Apply deterministic augmentations to the image/mask pair."""

		# Horizontal flip
		if random.random() < 0.5:
			image = F.hflip(image)  # type: ignore[assignment]
			mask = F.hflip(mask)  # type: ignore[assignment]

		# Vertical flip
		if random.random() < 0.5:
			image = F.vflip(image)  # type: ignore[assignment]
			mask = F.vflip(mask)  # type: ignore[assignment]

		# Brightness jitter (image only)
		if self.brightness_delta > 0:
			factor = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
			image = F.adjust_brightness(image, max(factor, 0.0))  # type: ignore[assignment]

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
	seed: int = 21,
	weak_label_subfolder: Optional[str] = None,
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
		seed=seed,
		weak_label_subfolder=weak_label_subfolder,
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

