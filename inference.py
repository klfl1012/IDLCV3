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
from model_registry import available_models, build_model
from trainer import LitSegmenter


@torch.inference_mode()
def evaluate_segmentation(
    checkpoint_path: Path,
    model_name: str,
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
        augment=False,  # No augmentation during evaluation
    )

    # Build model
    model = build_model(model_name, dataset=dataset)
    
    # Load from checkpoint
    lit_model = LitSegmenter.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
        criterion=nn.BCEWithLogitsLoss(),  # Dummy, not used during eval
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-4},
    )

    lit_model.to(device)
    lit_model.eval()

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

    print(f'Evaluating {model_name} on {dataset.upper()} {split} split...')
    
    for imgs, masks in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward pass
        logits = lit_model(imgs)
        
        # Handle binary segmentation
        if lit_model.num_classes == 2:
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            loss = lit_model.criterion(logits, masks.float())
            preds = (torch.sigmoid(logits) > 0.5).long()
        else:
            loss = lit_model.criterion(logits, masks)
            preds = logits.argmax(dim=1)

        # Update metrics
        for metric in metrics.values():
            metric(preds, masks)

        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

    # Compute final metrics
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

    # Save results
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'metrics': results,
            'checkpoint': str(checkpoint_path),
            'model': model_name,
            'dataset': dataset,
            'split': split,
            'batch_size': batch_size,
            'resize_to': resize_to,
        }
        with save_path.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained segmentation model on DRIVE or PH2 dataset.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to trained Lightning checkpoint (.ckpt)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='unet',
        choices=available_models(),
        help='Model architecture to evaluate',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['drive', 'ph2'],
        help='Target dataset',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Data split to evaluate (train/test/val)',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=None,
        help='Override default dataset root path',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Evaluation batch size',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader worker count',
    )
    parser.add_argument(
        '--resize-to',
        type=int,
        nargs=2,
        default=None,
        metavar=('HEIGHT', 'WIDTH'),
        help='Resize images to (H, W). Example: --resize-to 572 572',
    )
    parser.add_argument(
        '--metrics-out',
        type=Path,
        default=None,
        help='Optional path to save evaluation metrics as JSON',
    )

    args = parser.parse_args()

    # Convert resize_to to tuple if provided
    resize_to = tuple(args.resize_to) if args.resize_to else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    metrics = evaluate_segmentation(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
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
    print(f'Model:        {args.model}')
    print(f'Samples:      {metrics["samples"]}')
    print(f'Loss:         {metrics["loss"]:.4f}')
    print(f'IoU:          {metrics["iou"]:.4f}')
    print(f'Accuracy:     {metrics["accuracy"]:.4f} ({metrics["accuracy"]*100:.2f}%)')
    print(f'Sensitivity:  {metrics['sensitivity']:.4f}')
    print(f'Specificity:  {metrics['specificity']:.4f}')
    print(f'Precision:    {metrics['precision']:.4f}')
    print('='*60)
    
    if args.metrics_out is not None:
        print(f'\nMetrics saved to: {args.metrics_out}')


if __name__ == '__main__':
    main()