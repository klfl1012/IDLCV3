from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchmetrics as tm
from torch.utils.data import DataLoader

from dataloader import build_segmentation_dataloader
from model_registry import rebuild_model_from_config
from trainer import LitSegmenter
import torch.nn.functional as F
from losses import *

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