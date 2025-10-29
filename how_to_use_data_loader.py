"""Example script for inspecting the segmentation dataloaders."""

from __future__ import annotations

from pathlib import Path

import torch

from dataloader import build_segmentation_dataloader


# ---------------------------------------------------------------------------
# Configuration (edit these to test different settings)
# ---------------------------------------------------------------------------
DATASET_NAME = "ph2"  # options: "drive", "ph2"
SPLIT = "train"  # drive-only options: "train"/"training", "test"/"testing"; ignored for ph2

# Set to None to use the defaults embedded in dataloader.py
DRIVE_ROOT = Path("/dtu/datasets1/02516/DRIVE")  # options: Path(...) or None
PH2_ROOT = Path("/dtu/datasets1/02516/PH2_Dataset_images")  # options: Path(...) or None

# Resize target as (height, width). None keeps native resolution.
# Typical native sizes: DRIVE ≈ (584, 565); PH2 ≈ (576, 768) but varies slightly.
RESIZE_TO = (572, 765)  # options: (H, W) tuple or None

# Augmentation controls: apply random flips (horizontal/vertical) and brightness jitter.
AUGMENT = True  # options: True, False
BRIGHTNESS_DELTA = 0.15  # options: non-negative float (~0.0-0.3 works well)

# Visualization toggle: save first image/mask pair from the first batch.
SAVE_FIRST_SAMPLE = True  # options: True, False
OUTPUT_DIR = Path("./tmp/first_batch")  # options: Path(...)

BATCH_SIZE = 4  # options: any positive int
SHUFFLE = False  # options: True, False
NUM_WORKERS = 2  # options: non-negative integers
PIN_MEMORY = torch.cuda.is_available()  # options: True, False


def resolve_root(dataset: str) -> Path | None:
	dataset = dataset.lower()
	if dataset == "drive":
		return DRIVE_ROOT
	if dataset == "ph2":
		return PH2_ROOT
	raise ValueError(f"Unsupported dataset '{dataset}'. Choose 'drive' or 'ph2'.")


def main() -> None:
	dataset_key = DATASET_NAME.lower()
	root_override = resolve_root(dataset_key)

	data_loader = build_segmentation_dataloader(
		dataset=dataset_key,
		split=SPLIT,
		root=root_override,
		batch_size=BATCH_SIZE,
		shuffle=SHUFFLE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		resize_to=RESIZE_TO,
		augment=AUGMENT,
		brightness_delta=BRIGHTNESS_DELTA,
	)

	print(f"Loaded dataset '{dataset_key}' with {len(data_loader.dataset)} samples.")
	print(
		"Parameters -> "
		f"batch_size: {BATCH_SIZE}, shuffle: {SHUFFLE}, num_workers: {NUM_WORKERS}, pin_memory: {PIN_MEMORY}, "
		f"resize_to: {RESIZE_TO}, augment: {AUGMENT}, brightness_delta: {BRIGHTNESS_DELTA}"
	)

	images, masks = next(iter(data_loader))
	print(f"Image batch shape: {tuple(images.shape)} | dtype: {images.dtype}")
	print(f"Mask batch shape:  {tuple(masks.shape)} | dtype: {masks.dtype} | unique values: {masks.unique().tolist()}")

	if SAVE_FIRST_SAMPLE:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		img0 = images[0].detach().cpu()
		mask0 = masks[0].detach().cpu()
		if mask0.ndim == 3:
			mask0 = mask0.squeeze(0)

		# Convert to numpy arrays for saving
		img_arr = img0.permute(1, 2, 0).numpy()
		mask_arr = mask0.numpy()

		# Clamp image to [0, 1] range for correctness then save as PNG
		img_arr = img_arr.clip(0.0, 1.0)

		from PIL import Image

		Image.fromarray((img_arr * 255).astype("uint8")).save(OUTPUT_DIR / "sample0_image.png")

		# Scale mask to [0, 255] for visualization
		mask_min = mask_arr.min()
		mask_max = mask_arr.max() if mask_arr.max() != mask_min else mask_min + 1
		mask_scaled = ((mask_arr - mask_min) / (mask_max - mask_min) * 255.0).astype("uint8")
		Image.fromarray(mask_scaled).save(OUTPUT_DIR / "sample0_mask.png")

		print(f"Saved first batch sample to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
	main()

