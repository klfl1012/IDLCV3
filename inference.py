from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import build_segmentation_dataloader
from model_registry import rebuild_model_from_config
from trainer import LitSegmenter
import torch.nn.functional as F
from losses import *
import torchvision


def _save_prediction_example(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    save_path: Path,
    batch_idx: int = 0,
    sample_idx: int = 0,
) -> None:
    img_np = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    
    # Clip to valid range (in case of normalization artifacts)
    img_np = np.clip(img_np, 0, 1)
    
    gt_np = ground_truth.cpu().numpy()
    pred_np = prediction.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_np)
    axes[1].imshow(gt_np, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img_np)
    axes[2].imshow(pred_np, alpha=0.5, cmap='jet')
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    output_path = save_path.parent / f'{save_path.stem}_batch{batch_idx}_sample{sample_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f'  Saved visualization: {output_path.name}')


def generate_and_save_predictions(
    model: nn.Module,
    dataloader,
    mask_folder: Path,
    device: torch.device,
    use_amp: bool = False,
) -> None:
    """
    Generate predictions and overwrite masks on disk.
    
    Args:
        model: Trained model
        dataloader: DataLoader with all samples
        mask_folder: Path to folder where masks should be saved (e.g., region_growing_masks/10point/)
        device: Device to run inference on
        use_amp: Whether to use mixed precision
    """
    from PIL import Image
    
    model.eval()
    model.to(device)
    
    print(f'\n{"="*60}')
    print(f'Generating predictions and updating masks...')
    print(f'{"="*60}')
    
    with torch.inference_mode():
        for batch_idx, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            
            # Run inference
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(imgs)
            else:
                logits = model(imgs)
            
            # Convert logits to binary predictions
            if logits.dim() == 4:
                logits = logits.squeeze(1)  # Remove channel dimension
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()  # [B, H, W]
            
            # Save each prediction as mask
            for i in range(preds.shape[0]):
                sample_idx = batch_idx * dataloader.batch_size + i
                if sample_idx >= len(dataloader.dataset.samples):
                    break
                
                # Get original mask path
                original_mask_path = dataloader.dataset.samples[sample_idx].mask_path
                
                # Convert prediction to image (0-255)
                pred_mask = (preds[i] * 255).astype(np.uint8)
                
                # Save as image (overwrite old mask)
                mask_img = Image.fromarray(pred_mask, mode='L')
                mask_img.save(original_mask_path)
                
                if (sample_idx + 1) % 10 == 0:
                    print(f'  Updated {sample_idx + 1} masks...')
    
    print(f'✓ Updated all {len(dataloader.dataset)} masks in {mask_folder}')
    print(f'{"="*60}\n')


def log_weak_label_progress(
    model: nn.Module,
    dataloader,
    tensorboard_logger: TensorBoardLogger,
    device: torch.device,
    iteration: int,
    use_amp: bool = False,
    num_samples: int = 8,
    seed: int = 21,
) -> None:
    """
    Log weak label predictions to TensorBoard to visualize progress across iterations.
    
    Args:
        model: Trained model
        dataloader: DataLoader with weak label samples
        tensorboard_logger: TensorBoard logger instance
        device: Device to run inference on
        iteration: Current iteration number
        use_amp: Whether to use mixed precision
        num_samples: Number of random samples to visualize
        seed: Random seed for reproducibility
    """
    
    model.eval()
    model.to(device)
    
    all_imgs = []
    all_masks = []
    all_preds = []
    
    print(f'\nGenerating predictions for TensorBoard logging...')
    
    with torch.inference_mode():
        for imgs, masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Run inference
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(imgs)
            else:
                logits = model(imgs)
            
            # Convert to predictions
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_imgs.append(imgs.cpu())
            all_masks.append(masks.cpu())
            all_preds.append(preds.cpu())
    
    # Concatenate all batches
    all_imgs = torch.cat(all_imgs, dim=0)  # [N, 3, H, W]
    all_masks = torch.cat(all_masks, dim=0)  # [N, H, W]
    all_preds = torch.cat(all_preds, dim=0)  # [N, H, W]
    
    # Randomly select samples
    total_samples = all_imgs.size(0)
    num_to_log = min(num_samples, total_samples)
    
    # Random selection with FIXED seed for reproducibility (same samples across all iterations)
    torch.manual_seed(seed)  # Same seed → same images every iteration
    indices = torch.randperm(total_samples)[:num_to_log]
    
    imgs_viz = all_imgs[indices]
    masks_viz = all_masks[indices].unsqueeze(1).float()  # [N, 1, H, W]
    preds_viz = all_preds[indices].unsqueeze(1).float()  # [N, 1, H, W]
    
    # Convert to 3-channel for visualization
    masks_rgb = masks_viz.repeat(1, 3, 1, 1)
    preds_rgb = preds_viz.repeat(1, 3, 1, 1)
    
    # Create grid: [Original | Weak Labels | Predictions]
    grid = torchvision.utils.make_grid(
        torch.cat([imgs_viz, masks_rgb, preds_rgb], dim=0),
        nrow=num_to_log,
        normalize=True,
        scale_each=True,
    )
    
    # Log to TensorBoard
    tensorboard_logger.experiment.add_image(
        f'weak_labels_progress/iteration_{iteration}',
        grid,
        global_step=iteration,
    )
    
    print(f'✓ Logged {num_to_log} weak label predictions to TensorBoard (iteration {iteration})')


def evaluate_on_ph2_test_set(
    checkpoint_path: Path,
    model: nn.Module,
    criterion: nn.Module,
    optimizer_class: type,
    optimizer_kwargs: dict,
    batch_size: int,
    resize_to: tuple[int, int],
    device: torch.device,
    use_amp: bool = False,
    seed: int = 21,
    log_images_to_tensorboard: bool = True,
    tensorboard_logger: Optional[TensorBoardLogger] = None,
    experiment_name: str = 'final_evaluation',
    num_samples_to_log: int = 10,
) -> dict[str, float]:
    """
    Evaluate trained model on real PH2 test set and optionally log predictions to TensorBoard.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model: Model architecture (for loading checkpoint)
        criterion: Loss criterion
        optimizer_class: Optimizer class
        optimizer_kwargs: Optimizer kwargs
        batch_size: Batch size for evaluation
        resize_to: Image resize dimensions (H, W)
        device: Device to run on
        use_amp: Use mixed precision
        seed: Random seed
        log_images_to_tensorboard: Whether to log prediction images
        tensorboard_logger: TensorBoard logger instance (required if log_images_to_tensorboard=True)
        experiment_name: Name for TensorBoard logging
        num_samples_to_log: Number of random samples to log to TensorBoard (default: 10)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f'\n{"="*80}')
    print(f'FINAL EVALUATION ON REAL PH2 TEST SPLIT')
    print(f'{"="*80}\n')
    
    # Build real PH2 test dataloader
    ph2_test_loader = build_segmentation_dataloader(
        dataset='ph2',
        split='test',
        batch_size=batch_size,
        shuffle=False,
        resize_to=resize_to,
        augment=False,
        num_workers=4,
        seed=seed,
    )
    
    # Load best model
    best_model = LitSegmenter.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        strict=False,
    )
    best_model.to(device)
    best_model.eval()
    
    # Metrics
    total_correct = 0
    total_pixels = 0
    all_imgs = []
    all_masks = []
    all_preds = []
    
    print(f'Running inference on {len(str(ph2_test_loader.dataset))} test samples...')
    
    with torch.inference_mode():
        for imgs, masks in ph2_test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = best_model(imgs)
            else:
                logits = best_model(imgs)
            
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).long()
            
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            
            # Store for visualization
            if log_images_to_tensorboard:
                all_imgs.append(imgs.cpu())
                all_masks.append(masks.cpu())
                all_preds.append(preds.cpu())
    
    accuracy = total_correct / total_pixels
    
    print(f'\n{"="*80}')
    print(f'PH2 Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'{"="*80}\n')
    
    # Log images to TensorBoard
    if log_images_to_tensorboard and tensorboard_logger is not None:
        import torchvision
        
        # Concatenate all batches
        all_imgs = torch.cat(all_imgs, dim=0)  # [N, 3, H, W]
        all_masks = torch.cat(all_masks, dim=0)  # [N, H, W]
        all_preds = torch.cat(all_preds, dim=0)  # [N, H, W]
        
        # Randomly select num_samples_to_log samples
        total_samples = all_imgs.size(0)
        num_to_log = min(num_samples_to_log, total_samples)
        
        # Random selection with seed for reproducibility
        torch.manual_seed(seed)
        indices = torch.randperm(total_samples)[:num_to_log]
        
        imgs_viz = all_imgs[indices]
        masks_viz = all_masks[indices].unsqueeze(1).float()  # [N, 1, H, W]
        preds_viz = all_preds[indices].unsqueeze(1).float()  # [N, 1, H, W]
        
        # Convert to 3-channel for visualization
        masks_rgb = masks_viz.repeat(1, 3, 1, 1)
        preds_rgb = preds_viz.repeat(1, 3, 1, 1)
        
        # Create grid: [Original | Ground Truth | Prediction]
        grid = torchvision.utils.make_grid(
            torch.cat([imgs_viz, masks_rgb, preds_rgb], dim=0),
            nrow=num_to_log,
            normalize=True,
            scale_each=True,
        )
        
        # Log to TensorBoard
        tensorboard_logger.experiment.add_image(
            f'{experiment_name}/ph2_test_predictions',
            grid,
            global_step=0,
        )
        
        print(f'✓ Logged {num_to_log} randomly selected prediction examples to TensorBoard')
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_pixels': total_pixels,
    }


def _rebuild_criterion_from_hparams(criterion_class, criterion_kwargs: dict) -> nn.Module:
    '''
    Rebuild criterion from checkpoint hyperparameters using dictionary mapping.
    
    Args:
        criterion_class: The class or class name of the criterion
        criterion_kwargs: Keyword arguments for criterion initialization
        
    Returns:
        Reconstructed criterion module
    '''
    if criterion_class is None:
        return nn.BCEWithLogitsLoss()
    
    criterion_name = criterion_class.__name__ if hasattr(criterion_class, '__name__') else str(criterion_class)
    criterion_name_lower = criterion_name.lower() 
    
    if 'bcewithlogitsloss' in criterion_name_lower:
        pos_weight = criterion_kwargs.get('pos_weight', None)
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([float(pos_weight)])
            print(f'Reconstructing BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f}')
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print('Reconstructing BCEWithLogitsLoss (no pos_weight)')
            return nn.BCEWithLogitsLoss()
    
    # Criterion mapping with factory functions
    criterion_map = {
        'focal': lambda alpha=1.0, gamma=2.0, epsilon=1e-7, from_logits=True: FocalLoss(alpha=alpha, gamma=gamma, epsilon=epsilon, from_logits=from_logits),
    }
    
    # Find matching criterion
    for key, factory in criterion_map.items():
        if key in criterion_name:
            return factory(criterion_kwargs) # pyright: ignore[reportArgumentType]
    
    # Fallback
    print(f'Unknown criterion "{criterion_name}", using BCE as fallback')
    return nn.BCEWithLogitsLoss()


@torch.inference_mode()
def _sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int],
    overlap: int = 64,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    uses_channels_last: bool = False,
) -> torch.Tensor:
    """
    Sliding window inference for arbitrary image sizes.
    
    Args:
        model: The neural network model
        image: Input image tensor [1, C, H, W]
        patch_size: Size of patches to process (H, W)
        overlap: Overlap between patches in pixels
        device: Device to run inference on
        use_amp: Whether to use mixed precision
        amp_dtype: Data type for mixed precision
        uses_channels_last: Whether model uses channels_last format
    
    Returns:
        Prediction tensor [1, 1, H, W] for binary segmentation
    """
    if device is None:
        device = image.device
    
    C, H, W = image.shape[1], image.shape[2], image.shape[3]
    patch_h, patch_w = patch_size
    
    # Create output accumulators
    predictions = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)
    weights = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)
    
    # Create Gaussian weight for smooth blending
    def gaussian_window(h: int, w: int) -> torch.Tensor:
        y = torch.linspace(-1, 1, h)
        x = torch.linspace(-1, 1, w)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        weight = torch.exp(-(xx**2 + yy**2))
        return weight.unsqueeze(0).unsqueeze(0).to(device)
    
    weight_map = gaussian_window(patch_h, patch_w)
    
    # Calculate stride
    stride_h = patch_h - overlap
    stride_w = patch_w - overlap
    
    # Sliding window
    y_positions = list(range(0, H - patch_h + 1, stride_h))
    x_positions = list(range(0, W - patch_w + 1, stride_w))
    
    # Add last positions if image is larger
    if y_positions[-1] + patch_h < H:
        y_positions.append(H - patch_h)
    if x_positions[-1] + patch_w < W:
        x_positions.append(W - patch_w)
    
    total_patches = len(y_positions) * len(x_positions)
    print(f'Sliding window: Processing {total_patches} patches ({len(y_positions)}×{len(x_positions)}) with overlap={overlap}px')
    
    for y in y_positions:
        for x in x_positions:
            # Extract patch
            patch = image[:, :, y:y+patch_h, x:x+patch_w]
            
            # Apply channels_last if needed
            if uses_channels_last:
                patch = patch.contiguous(memory_format=torch.channels_last)
            
            # Inference
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(patch)
            else:
                logits = model(patch)
            
            # Convert logits to probabilities
            if logits.dim() == 4:
                logits = logits.squeeze(1)  # Remove channel dimension if present
            probs = torch.sigmoid(logits).unsqueeze(1)  # [1, 1, H, W]
            
            # Accumulate with Gaussian weighting
            predictions[:, :, y:y+patch_h, x:x+patch_w] += probs * weight_map
            weights[:, :, y:y+patch_h, x:x+patch_w] += weight_map
    
    # Normalize by weights
    predictions = predictions / (weights + 1e-8)
    
    return predictions


def _is_divisible_by_16(height: int, width: int) -> bool:
    """Check if dimensions are divisible by 16."""
    return (height % 16 == 0) and (width % 16 == 0)


@torch.inference_mode()
def _evaluate_segmentation(
    checkpoint_path: Path,
    dataset: str,
    split: str = 'test',
    dataset_root: Optional[Path] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    resize_to: Optional[tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
    precision: Optional[str] = None,
    save_examples: bool = True,
) -> dict:
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found at "{checkpoint_path}".')

    # Load checkpoint to inspect hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # Automatically detect precision used in training from checkpoint if not provided
    if precision is None:
        precision = hparams.get('precision', '32')
        print(f'Auto-detected precision from checkpoint: {precision}')
    
    # Check if model was trained with channels_last memory format (compiled models)
    uses_channels_last = hparams.get('model_config', {}).get('uses_channels_last', False)
    
    print(f'Loading model from checkpoint:')
    print(f'  Model: {hparams.get("model_name", "unknown")}')
    print(f'  Dataset: {hparams.get("model_config", {}).get("dataset", "unknown")}')
    print(f'  Criterion: {hparams.get("criterion_class", "unknown")}')
    print(f'  Precision: {precision}')
    print(f'  Channels Last: {uses_channels_last}')

    # Simplified precision check - supports both FP16 (V100) and BF16 (A100+)
    use_amp = precision in ('16-mixed', 'bf16-mixed') and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if precision == 'bf16-mixed' else torch.float16
    
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f'Using mixed precision inference: {precision}')

    # Rebuild model from saved config
    model = rebuild_model_from_config(hparams['model_config'])
    
    # Rebuild criterion from saved config
    criterion = _rebuild_criterion_from_hparams(
        hparams.get('criterion_class'),
        hparams.get('criterion_kwargs', {})
    )
    
    # Load LitSegmenter with reconstructed components
    lit_model = LitSegmenter.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1e-4},
        strict=False,
    )

    lit_model.to(device)

    # Apply channels_last memory format to convert Conv2d weights
    if uses_channels_last and device.type == 'cuda':
        for module in lit_model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if module.weight.dim() == 4: 
                    module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)
        print('Applied channels_last memory format for inference')
    
    lit_model.eval()

    # Build dataloader
    test_loader = build_segmentation_dataloader(
        dataset=dataset,
        split=split,
        root=dataset_root,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        resize_to=resize_to,
        augment=False,
    )
    
    # Determine if we need sliding window inference
    # Check if original dataset resolution is divisible by 16
    sample_batch = next(iter(test_loader))
    sample_img = sample_batch[0][0]  # Get first image
    img_h, img_w = sample_img.shape[-2], sample_img.shape[-1]
    
    use_sliding_window = not _is_divisible_by_16(img_h, img_w)
    
    if use_sliding_window:
        print(f'\n{"="*60}')
        print(f'Image size {img_h}×{img_w} is NOT divisible by 16.')
        print(f'Enabling Sliding Window Inference for optimal quality.')
        print(f'{"="*60}\n')
        
        # Determine patch size (closest divisible by 16, smaller than image)
        patch_h = min((img_h // 16) * 16, 576)
        patch_w = min((img_w // 16) * 16, 576)
        if patch_h == 0:
            patch_h = 16
        if patch_w == 0:
            patch_w = 16
        patch_size = (patch_h, patch_w)
        overlap = min(64, patch_h // 4, patch_w // 4)
        print(f'Patch size: {patch_size}, Overlap: {overlap}px')
    else:
        print(f'\n{"="*60}')
        print(f'Image size {img_h}×{img_w} is divisible by 16.')
        print(f'Using standard inference (no sliding window needed).')
        print(f'{"="*60}\n')
        patch_size = None
        overlap = None

    # Initialize metrics
    task = 'binary' if lit_model.num_classes == 2 else 'multiclass'
    metrics = {
        'iou': tm.JaccardIndex(task=task, num_classes=lit_model.num_classes).to(device),
        'accuracy': tm.Accuracy(task=task, num_classes=lit_model.num_classes).to(device),
        'sensitivity': tm.Recall(task=task, num_classes=lit_model.num_classes).to(device),
        'specificity': tm.Specificity(task=task, num_classes=lit_model.num_classes).to(device),
        'precision': tm.Precision(task=task, num_classes=lit_model.num_classes).to(device),
        'dice': DiceCoefficient(num_classes=lit_model.num_classes).to(device),
    }

    total_loss = 0.0
    total_samples = 0

    print(f'Evaluating on {dataset.upper()} {split} split...')
    
    for batch_idx, (imgs, masks) in enumerate(test_loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Choose inference method based on image dimensions
        if use_sliding_window:
            # Process each image in batch separately with sliding window
            batch_preds = []
            for i in range(imgs.shape[0]):
                single_img = imgs[i:i+1]  # [1, C, H, W]
                
                # Sliding window inference returns probabilities
                pred_probs = _sliding_window_inference(
                    model=lit_model,
                    image=single_img,
                    patch_size=patch_size, # pyright: ignore[reportArgumentType]
                    overlap=overlap, # pyright: ignore[reportArgumentType]
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    uses_channels_last=uses_channels_last,
                )
                
                # Convert to binary predictions
                pred_binary = (pred_probs > 0.5).squeeze(1).long()  # [1, H, W]
                batch_preds.append(pred_binary)
            
            preds = torch.cat(batch_preds, dim=0)  # [B, H, W]
            
            # Calculate loss (use first image for demonstration)
            # Note: For sliding window, we use the averaged probabilities
            logits = torch.logit(pred_probs.squeeze(0), eps=1e-6)  # Convert back to logits for loss
            if lit_model.num_classes == 2:
                loss = lit_model.criterion(logits, masks[0:1].float())
            else:
                loss = lit_model.criterion(logits.unsqueeze(0), masks[0:1])
                
        else:
            # Standard inference (no sliding window)
            # Apply channels_last to input if model uses it
            if uses_channels_last and device.type == 'cuda':
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            # Use autocast for mixed precision if enabled
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = lit_model(imgs)
                    
                    if lit_model.num_classes == 2:
                        if logits.dim() == 4:
                            logits = logits.squeeze(1)
                        loss = lit_model.criterion(logits, masks.float())
                        preds = (torch.sigmoid(logits) > 0.5).long()
                    else:
                        loss = lit_model.criterion(logits, masks)
                        preds = logits.argmax(dim=1)
            else:
                logits = lit_model(imgs)
                
                if lit_model.num_classes == 2:
                    if logits.dim() == 4:
                        logits = logits.squeeze(1)
                    loss = lit_model.criterion(logits, masks.float())
                    preds = (torch.sigmoid(logits) > 0.5).long()
                else:
                    loss = lit_model.criterion(logits, masks)
                    preds = logits.argmax(dim=1)

        for metric in metrics.values():
            metric(preds, masks)

        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
        
        # Save example visualizations (one per batch)
        if save_examples and save_path is not None:
            # Save first sample from this batch
            _save_prediction_example(
                image=imgs[0],
                ground_truth=masks[0],
                prediction=preds[0],
                save_path=save_path,
                batch_idx=batch_idx,
                sample_idx=0,
            )
        
        # Progress indicator
        if (batch_idx + 1) % 10 == 0:
            print(f'  Processed {total_samples} samples...')

    avg_loss = total_loss / max(total_samples, 1)
    results = {
        'loss': avg_loss,
        'iou': metrics['iou'].compute().item(),
        'accuracy': metrics['accuracy'].compute().item(),
        'sensitivity': metrics['sensitivity'].compute().item(),
        'specificity': metrics['specificity'].compute().item(),
        'precision': metrics['precision'].compute().item(),
        'dice': metrics['dice'].compute().item(),
        'samples': total_samples,
    }

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'metrics': results,
            'checkpoint': str(checkpoint_path),
            'model': hparams.get('model_name', 'unknown'),
            'dataset': dataset,
            'split': split,
            'batch_size': batch_size,
            'resize_to': resize_to,
        }
        with save_path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, indent=2)

    return results


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation model on DRIVE or PH2 dataset.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to trained Lightning checkpoint (.ckpt)')
    parser.add_argument('--dataset', type=str, required=True, choices=['drive', 'ph2'], help='Target dataset')
    parser.add_argument('--split', type=str, default='test', help='Data split to evaluate (train/test/val)')
    parser.add_argument('--dataset-root', type=Path, default=None, help='Override default dataset root path')
    parser.add_argument('--batch-size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker count')
    parser.add_argument('--resize', type=int, nargs=2, default=None, metavar=('HEIGHT', 'WIDTH'), help='Resize images to (H, W). Example: --resize 572 572')
    parser.add_argument('--metrics-out', type=Path, default=None, help='Optional path to save evaluation metrics as JSON')
    parser.add_argument('--precision', type=str, choices=['32', '16-mixed', 'bf16-mixed'], default=None, help='Precision mode for evaluation (32, 16-mixed, bf16-mixed). Auto-detected from checkpoint if not specified.')
    parser.add_argument('--no-save-examples', action='store_true', help='Disable saving example prediction visualizations')
    return parser.parse_args()



def main():
    args = _build_args()
    
    # Convert resize_to to tuple if provided
    resize_to = tuple(args.resize) if args.resize else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    metrics = _evaluate_segmentation(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_to=resize_to,
        device=device,
        save_path=args.metrics_out,
        save_examples=not args.no_save_examples,
    )

    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Dataset:      {args.dataset.upper()} ({args.split} split)')
    print(f'Samples:      {metrics['samples']}')
    print(f'Loss:         {metrics['loss']:.4f}')
    print(f'IoU:          {metrics['iou']:.4f}')
    print(f'Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)')
    print(f'Sensitivity:  {metrics['sensitivity']:.4f}')
    print(f'Specificity:  {metrics['specificity']:.4f}')
    print(f'Precision:    {metrics['precision']:.4f}')
    print(f'Dice:         {metrics['dice']:.4f}')
    print('='*60)
    
    if args.metrics_out is not None:
        print(f'\nMetrics saved to: {args.metrics_out}')


if __name__ == '__main__':
    main()