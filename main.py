import argparse
from pathlib import Path
import ast
import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import dataloader
from dataloader import *
from model_registry import available_models, build_model
from trainer import get_segm_trainer, LitSegmenter
import torchmetrics as tm


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train segmentation model with optional ablation study.')
    parser.add_argument('--model', type=str, required=True, choices=available_models())
    parser.add_argument('--criterion', type=str, default='bce', choices=['bce', 'weighted_bce', 'dice', 'focal', 'lovasz', 'dice_bce'], help='Loss criterion to use.')
    parser.add_argument('--criterion_kwargs', type=str, default='{}', help='Criterion keyword arguments as a dictionary string (e.g., \'{"pos_weight": 10.0}\').')
    parser.add_argument('--model_kwargs', type=str, default='{}', help='Model keyword arguments as a dictionary string.')
    parser.add_argument('--dataset', type=str, required=True, choices=['drive', 'ph2'])
    parser.add_argument('--resize', type=int, nargs=2, default=(572, 572), metavar=('HEIGHT', 'WIDTH'), help='Resize images to (H, W). Example: --resize 572 572')
    parser.add_argument('--outdir', type=Path, default=Path('./outputs'), help='Output directory for checkpoints and logs.')
    parser.add_argument('--seed', type=int, default=21, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use (adam, adamw, sgd).')
    parser.add_argument('--optimizer_kwargs', type=str, default='{"lr": 1e-4, "weight_decay": 1e-5}', help='Optimizer keyword arguments as a dictionary string.')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help='Learning rate scheduler to use (optional).')
    parser.add_argument('--scheduler_kwargs', type=str, default='{"mode": "min", "factor": 0.5, "patience": 5}', help='Scheduler keyword arguments as a dictionary string.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience (in epochs).')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Enable mixed precision training (FP16).')
    parser.add_argument('--ablation_study', action='store_true', help='Run ablation study with multiple loss functions.')
    parser.add_argument('--no_preview', action='store_true', help='Disable printing example batch shapes before training.')

    return parser.parse_args()


def _describe_batch(inputs, labels, prefix: str) -> None:
    print(prefix)

    if isinstance(inputs, dict):
        for key, tensor in inputs.items():
            if hasattr(tensor, 'shape'):
                print(f'    Input "{key}": shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}')
            else:
                print(f'    Input "{key}": type={type(tensor)}')
    elif isinstance(inputs, torch.Tensor):
        print(f'    Inputs: shape={tuple(inputs.shape)}, dtype={inputs.dtype}')
    else:
        for idx, tensor in enumerate(inputs):
            print(f'    Input[{idx}]: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}')

    print(f'    Labels: shape={tuple(labels.shape)}, dtype={labels.dtype}')


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
        'sgd': torch.optim.SGD,
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


def _get_criterion(name: str, criterion_kwargs: dict) -> tuple[nn.Module, str, dict]:
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
            self.bce = nn.BCEWithLogitsLoss()
        
        def forward(self, logits, targets):
            return 0.5 * self.dice(logits, targets) + 0.5 * self.bce(logits, targets)
    
    # Criterion mapping: name -> (factory_function, criterion_name, default_config)
    criterion_map = {
        'bce': (lambda kwargs: nn.BCEWithLogitsLoss(), 'BCEWithLogitsLoss', {}),
        'weighted_bce': (lambda kwargs: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([kwargs.get('pos_weight', 10.0)])), 'BCEWithLogitsLoss', {'pos_weight': 10.0}),  # Will be overridden if provided in kwargs
        'dice': (lambda kwargs: smp.losses.DiceLoss(mode='binary', from_logits=True), 'DiceLoss', {'mode': 'binary', 'from_logits': True}),
        'focal': (lambda kwargs: smp.losses.FocalLoss(mode='binary'), 'FocalLoss', {'mode': 'binary'}),
        'lovasz': (lambda kwargs: smp.losses.LovaszLoss(mode='binary', from_logits=True), 'LovaszLoss', {'mode': 'binary', 'from_logits': True}),
        'dice_bce': (lambda kwargs: CombinedLoss(), 'DiceBCELoss', {'mode': 'binary', 'from_logits': True}),
    }
    
    name_lower = name.lower()
    if name_lower not in criterion_map:
        raise ValueError(f'Unknown criterion "{name}". Supported: {list(criterion_map.keys())}')
    
    factory, criterion_name, default_config = criterion_map[name_lower]
    
    # Merge default config with user-provided kwargs for weighted_bce
    final_config = {**default_config, **criterion_kwargs}
    
    return factory(criterion_kwargs), criterion_name, final_config


def _train_single_experiment(
    args: argparse.Namespace,
    criterion_name: str,
    criterion_kwargs: dict,
    train_loader,
    val_loader,
) -> None:
    """Train a single experiment with given criterion."""
    
    print(f"\n{'='*60}")
    print(f"Training with {criterion_name} loss")
    print(f"Configuration: {criterion_kwargs}")
    print(f"{'='*60}\n")
    
    # Build model with config
    model, model_config = build_model(
        name=args.model,
        dataset=args.dataset,
        **_parse_dict_arg(args.model_kwargs, 'model_kwargs'),
    )
    
    # Get criterion with config
    criterion, crit_name, crit_config = _get_criterion(criterion_name, criterion_kwargs)
    
    # Setup optimizer and scheduler
    optimizer_class = _get_optimizer_class(args.optimizer)
    optimizer_kwargs = _parse_dict_arg(args.optimizer_kwargs, 'optimizer_kwargs')
    scheduler_class = _get_scheduler_class(args.scheduler) if args.scheduler else None
    scheduler_kwargs = _parse_dict_arg(args.scheduler_kwargs, 'scheduler_kwargs')
    
    # Create output directory for this experiment
    exp_outdir = args.outdir / f'_{criterion_name}'
    
    # Determine precision
    # precision = 'bf16-mixed' if args.use_mixed_precision and torch.cuda.is_available() else '32'
    precision = '16-mixed' if args.use_mixed_precision and torch.cuda.is_available() else '32'
    
    # Create LitSegmenter with full config
    litmodel = LitSegmenter(
        model=model,
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        outdir=str(exp_outdir),
        num_classes=2,
        model_name=model_config['name'],
        model_config=model_config,
        criterion_class=type(criterion),
        criterion_kwargs=crit_config,
        precision=precision,  
    )
    
    trainer = get_segm_trainer(
        max_epochs=args.max_epochs,
        outdir=str(exp_outdir),
        experiment_name=f'{args.dataset}_{criterion_name}',
        early_stopping_patience=args.early_stopping_patience,
        use_mixed_precision=args.use_mixed_precision,
    )
    
    trainer.fit(litmodel, train_loader, val_loader)
    

def main():
    args = _build_args()
    
    # Set random seed
    pl.seed_everything(args.seed, workers=True)
    
    # Build dataloaders (shared for all experiments)
    train_loader = build_segmentation_dataloader(
        dataset=args.dataset,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        resize_to=tuple(args.resize),
        augment=True,
        num_workers=4,
        seed=args.seed,
    )

    val_loader = build_segmentation_dataloader(
        dataset=args.dataset,
        split='val',  # DRIVE: use 'val', PH2: no official split
        batch_size=args.batch_size,
        shuffle=False,
        resize_to=tuple(args.resize),
        augment=False,
        num_workers=4,
        seed=args.seed,
    )

    # Preview batch
    if not args.no_preview:
        train_inputs, train_labels = next(iter(train_loader))
        _describe_batch(train_inputs, train_labels, prefix='Example training batch:')
        val_inputs, val_labels = next(iter(val_loader))
        _describe_batch(val_inputs, val_labels, prefix='Example validation batch:')
    
    # Run ablation study or single training
    if args.ablation_study:
        print("\n" + "="*60)
        print("RUNNING ABLATION STUDY")
        print("="*60)
        
        # Define experiments for ablation study
        experiments = [
            {'name': 'bce', 'kwargs': {}},
            {'name': 'weighted_bce', 'kwargs': {'pos_weight': 10.0}},
            {'name': 'dice', 'kwargs': {}},
            {'name': 'focal', 'kwargs': {}},
            {'name': 'lovasz', 'kwargs': {}},
            {'name': 'dice_bce', 'kwargs': {}},
        ]
        
        for exp in experiments:
            _train_single_experiment(
                args=args,
                criterion_name=exp['name'],
                criterion_kwargs=exp['kwargs'],
                train_loader=train_loader,
                val_loader=val_loader,
            )

        print("Ablation study completed!")

    else:
        # Single training run
        criterion_kwargs = _parse_dict_arg(args.criterion_kwargs, 'criterion_kwargs')
        _train_single_experiment(
            args=args,
            criterion_name=args.criterion,
            criterion_kwargs=criterion_kwargs,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        print("Training completed!")


if __name__ == '__main__':
    main()

    