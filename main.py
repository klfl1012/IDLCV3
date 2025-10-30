import argparse
from pathlib import Path
import ast
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from dataloader import *
from model_registry import available_models, build_model
from trainer import get_segm_trainer, LitSegmenter
import torchmetrics as tm


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate segmentation model on dataset split.')
    parser.add_argument('--model', type=str, required=True, choices=available_models())
    parser.add_argument('--criterion', type=str, default='dice', help='Loss criterion to use (dice, bce, focal, jaccard).')
    parser.add_argument('--model_kwargs', type=str, default='{}', help='Model keyword arguments as a dictionary string.')
    parser.add_argument('--dataset', type=str, required=True, choices=['drive', 'ph2'])
    parser.add_argument('--resize', type=tuple, default=(572, 572), metavar=('HEIGHT', 'WIDTH'), help='Resize images to (H, W). Example: --resize 572 572')
    parser.add_argument('--outdir', type=Path, default=Path('./outputs'), help='Output directory for checkpoints and logs.')
    parser.add_argument('--seed', type=int, default=21, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use (adam, adamw, sgd).')
    parser.add_argument('--optimizer_kwargs', type=str, default='{"lr": 1e-4, "weight_decay": 1e-2}', help='Optimizer keyword arguments as a dictionary string.')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help='Learning rate scheduler to use (optional).')
    parser.add_argument('--scheduler_kwargs', type=str, default='{}', help='Scheduler keyword arguments as a dictionary string.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience (in epochs).')
    parser.add_argument('--no_preview', action='store_true', help='Disable printing example batch shapes before training.')

    return parser.parse_args()


def _describe_batch(inputs, labels, prefix: str) -> None:
    print(prefix)

    if isinstance(inputs, dict):
        for key, tensor in inputs.items():
            if hasattr(tensor, 'shape'):
                print(f'  Input "{key}": shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}')
            
            else:
                print(f'  Input "{key}": type={type(tensor)}')

    elif isinstance(inputs, torch.Tensor):
        for idx, tensor in enumerate(inputs):
            print(f'  Input[{idx}]: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}')
    
    else:
        print(f'  inputs: shape={tuple(inputs.shape)}, dtype={inputs.dtype}')
    print(f'  labels: shape={tuple(labels.shape)}, dtype={labels.dtype}')

def _parse_dict_arg(arg_value: str, arg_name: str) -> dict:
    try:
        result = ast.literal_eval(arg_value)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except Exception:
        raise ValueError(f'Invalid format for {arg_name}. Expected a dictionary string, got: {arg_value}')
    

def _get_optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    name_lower = name.lower()
    mapping = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
    }
    if name_lower not in mapping:
        raise ValueError(f'Unknown optimizer "{name}". Supported: {list(mapping.keys())}')
    return mapping[name_lower]

def _get_scheduler_class(name: str) -> type[torch.optim.lr_scheduler._LRScheduler]:
    name_lower = name.lower()
    mapping = {
        'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
        'step_lr': torch.optim.lr_scheduler.StepLR,
    }
    if name_lower not in mapping:
        raise ValueError(f'Unknown scheduler "{name}". Supported: {list(mapping.keys())}')
    return mapping[name_lower]


# TODO: Implement actual loss functions here!
def _get_criterion(name: str) -> nn.Module:
    name_lower = name.lower()
    mapping = {
        'dice': nn.Module(), 
        'bce': nn.Module(),
        'focal': nn.Module(),
        'jaccard': nn.Module(),
    }
    if name_lower not in mapping:
        raise ValueError(f'Unknown criterion "{name}". Supported: {list(mapping.keys())}')
    return mapping[name_lower]
    

def main():

    args = _build_args()

    model = build_model(
        name=args.model,
        model_kwargs=_parse_dict_arg(args.model_kwargs, 'model_kwargs'),
        dataset=args.dataset,
    )

    train_loader = build_segmentation_dataloader(
        dataset=args.dataset,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        resize_to=args.resize,
        augment=True,
    )

    val_loader = build_segmentation_dataloader(
        dataset=args.dataset,
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        resize_to=args.resize,
        augment=False,
    )

    if not args.no_preview:
        train_inputs, train_labels = next(iter(train_loader))
        _describe_batch(train_inputs, train_labels, prefix='Example training batch:')
        val_inputs, val_labels = next(iter(val_loader))
        _describe_batch(val_inputs, val_labels, prefix='Example validation batch:')

    optimizer_kwargs = _parse_dict_arg(args.optimizer_kwargs, 'optimizer_kwargs')
    optimizer_class = _get_optimizer_class(args.optimizer)
    scheduler_kwargs = _parse_dict_arg(args.scheduler_kwargs, 'scheduler_kwargs')
    scheduler_class = _get_scheduler_class(args.scheduler) if args.scheduler else None


    criterion = _get_criterion(args.criterion)


    litmodel = LitSegmenter(
        model=model,
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        outdir=args.outdir,
    )

    trainer = get_segm_trainer(
        max_epochs=args.max_epochs,
        outdir=args.outdir,
        early_stopping_patience=args.early_stopping_patience,
    )

    trainer.fit(litmodel, train_loader, val_loader)





if __name__ == '__main__':
    main()

    