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
from model_registry import available_models, build_model, rebuild_model_from_config
from trainer import LitSegmenter


def _rebuild_criterion_from_hparams(criterion_class, criterion_kwargs: dict) -> nn.Module:
    """
    Rebuild criterion from checkpoint hyperparameters using dictionary mapping.
    
    Args:
        criterion_class: The class or class name of the criterion
        criterion_kwargs: Keyword arguments for criterion initialization
        
    Returns:
        Reconstructed criterion module
    """
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
    print(f"Unknown criterion '{criterion_name}', using BCE as fallback")
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
) -> dict:
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found at "{checkpoint_path}".')

    # Load checkpoint to inspect hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    print(f"Loading model from checkpoint:")
    print(f"  Model: {hparams.get('model_name', 'unknown')}")
    print(f"  Dataset: {hparams.get('model_config', {}).get('dataset', 'unknown')}")
    print(f"  Criterion: {hparams.get('criterion_class', 'unknown')}")

    # Rebuild model from saved config
    from model_registry import rebuild_model_from_config
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
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-4},
    )

    lit_model.to(device)
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
    }

    total_loss = 0.0
    total_samples = 0

    print(f'Evaluating on {dataset.upper()} {split} split...')
    
    for imgs, masks in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

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
    print(f'Samples:      {metrics["samples"]}')
    print(f'Loss:         {metrics["loss"]:.4f}')
    print(f'IoU:          {metrics["iou"]:.4f}')
    print(f'Accuracy:     {metrics["accuracy"]:.4f} ({metrics["accuracy"]*100:.2f}%)')
    print(f'Sensitivity:  {metrics["sensitivity"]:.4f}')
    print(f'Specificity:  {metrics["specificity"]:.4f}')
    print(f'Precision:    {metrics["precision"]:.4f}')
    print('='*60)
    
    if args.metrics_out is not None:
        print(f'\nMetrics saved to: {args.metrics_out}')


if __name__ == '__main__':
    main()