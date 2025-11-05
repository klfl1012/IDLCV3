from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchmetrics as tm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from dataloader import build_segmentation_dataloader
from model_registry import rebuild_model_from_config
from trainer import LitSegmenter


class DiceCoefficient(tm.Metric):
    """
    Manual implementation of Dice coefficient (F1 score) for segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Ensure same shape
        if preds.shape != targets.shape:
            raise ValueError(f"Predictions shape {preds.shape} != targets shape {targets.shape}")
        
        # Flatten tensors
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        
        # Calculate intersection and union
        intersection = torch.sum(preds_flat * targets_flat)
        union = torch.sum(preds_flat) + torch.sum(targets_flat)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum += dice
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.dice_sum / self.count if self.count > 0 else torch.tensor(0.0)


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
    
    # Combined loss helper class
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
            self.bce = nn.BCEWithLogitsLoss()
        
        def forward(self, logits, targets):
            return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)
    
    # Criterion mapping with factory functions
    criterion_map = {
        'BCEWithLogitsLoss': lambda kwargs: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([kwargs['pos_weight']]) if 'pos_weight' in kwargs else None
        ),
        'DiceLoss': lambda kwargs: smp.losses.DiceLoss(**kwargs),
        'FocalLoss': lambda kwargs: smp.losses.FocalLoss(**kwargs),
        'LovaszLoss': lambda kwargs: smp.losses.LovaszLoss(**kwargs),
        'DiceBCELoss': lambda kwargs: CombinedLoss(),
        'CombinedLoss': lambda kwargs: CombinedLoss(),
    }
    
    # Find matching criterion
    for key, factory in criterion_map.items():
        if key in criterion_name:
            return factory(criterion_kwargs)
    
    # Fallback
    print(f'Unknown criterion "{criterion_name}", using BCE as fallback')
    return nn.BCEWithLogitsLoss()


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
    
    for imgs, masks in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
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