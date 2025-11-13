from __future__ import annotations
import argparse
from pathlib import Path
import ast
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataloader import *
from model_registry import available_models, build_model
from trainer import get_segm_trainer, LitSegmenter
from losses import FocalLoss
from inference import generate_and_save_predictions, evaluate_on_ph2_test_set, log_weak_label_progress
from pytorch_lightning.loggers import TensorBoardLogger


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train segmentation model with optional ablation study.')
    parser.add_argument('--model', type=str, required=True, choices=available_models())
    parser.add_argument('--criterion', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal'], help='Loss criterion to use.')
    parser.add_argument('--criterion_kwargs', type=str, default='{}', help='Criterion keyword arguments as a dictionary string (e.g., \'{"pos_weight": 10.0}\').')
    parser.add_argument('--model_kwargs', type=str, default='{}', help='Model keyword arguments as a dictionary string.')
    parser.add_argument('--dataset', type=str, required=True, choices=['drive', 'ph2', 'weak_labels'])
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
    parser.add_argument('--weak_iterations', type=int, default=5, help='Number of self-training iterations for weak labels.')
    parser.add_argument('--weak_epochs_per_iter', type=int, default=5, help='Training epochs per weak label iteration.')
    parser.add_argument('--weak_folders', nargs='*', default=None, help='Optional list of weak-label subfolders to run the ablation on (e.g., 10point 2point). If not set, all subfolders are used.')
    parser.add_argument('--num_samples_to_log', type=int, default=8, help='Number of random samples to log to TensorBoard during each iteration and final evaluation.')
    parser.add_argument('--weak_final_epochs', type=int, default=200, help='Number of epochs to train on weak labels in the final training phase after all self-training iterations.')

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

    criterion_map = {
        'bce': (
            lambda: nn.BCEWithLogitsLoss(), 'BCEWithLogitsLoss', {}
        ),
        'weighted_bce': (
            lambda pos_weight=10.0: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])), 
            'BCEWithLogitsLoss', 
            {'pos_weight': 10.0}
        ),
        'focal': (
            lambda alpha=1.0, gamma=2.0, epsilon=1e-7, from_logits=True: FocalLoss(alpha=alpha, gamma=gamma, epsilon=epsilon, from_logits=from_logits),
            'FocalLoss',
            {'alpha': 1.0, 'gamma': 2.0, 'epsilon': 1e-7 , 'from_logits': True}
        ),
    }

    name_lower = name.lower()
    if name_lower not in criterion_map:
        raise ValueError(f'Unknown criterion "{name}". Supported: {list(criterion_map.keys())}')
    
    factory, criterion_name, default_config = criterion_map[name_lower]
    
    final_config = {**default_config, **criterion_kwargs}
    
    return factory(**final_config), criterion_name, final_config


def _train_single_experiment(
    args: argparse.Namespace,
    criterion_name: str,
    criterion_kwargs: dict,
    train_loader,
    val_loader,
) -> None:

    print(f'\n{"="*60}')
    print(f'Training with {criterion_name} loss')
    print(f'Configuration: {criterion_kwargs}')
    print(f'{"="*60}\n')

    model, model_config = build_model(
        name=args.model,
        dataset=args.dataset,
        **_parse_dict_arg(args.model_kwargs, 'model_kwargs'),
    )
    
    criterion, crit_name, crit_config = _get_criterion(criterion_name, criterion_kwargs)
    
    optimizer_class = _get_optimizer_class(args.optimizer)
    optimizer_kwargs = _parse_dict_arg(args.optimizer_kwargs, 'optimizer_kwargs')
    scheduler_class = _get_scheduler_class(args.scheduler) if args.scheduler else None
    scheduler_kwargs = _parse_dict_arg(args.scheduler_kwargs, 'scheduler_kwargs')
    
    exp_outdir = args.outdir / f'_{criterion_name}'
    
    precision = '16-mixed' if args.use_mixed_precision and torch.cuda.is_available() else '32'
    
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


def _iterative_weak_label_training(
    args: argparse.Namespace,
    weak_label_subfolder: str,
    num_iterations: int = 5,
    epochs_per_iteration: int = 5,
    final_epochs: int = 200,
) -> Path | None:
    
    print(f'\n{"="*80}')
    print(f'ITERATIVE SELF-TRAINING: {weak_label_subfolder}')
    print(f'Iterations: {num_iterations}, Epochs per iteration: {epochs_per_iteration}')
    print(f'Final training: {final_epochs} epochs on best pseudo-labels')
    print(f'Mode: Train & test on ALL samples (no split), overwrite masks after each iteration')
    print(f'{"="*80}\n')
    
    exp_outdir = args.outdir / 'weak_labels_ablation' / weak_label_subfolder
    exp_outdir.mkdir(parents=True, exist_ok=True)
    
    # Path to mask folder
    mask_folder = Path('./region_growing_masks') / weak_label_subfolder
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer_class = _get_optimizer_class(args.optimizer)
    optimizer_kwargs = _parse_dict_arg(args.optimizer_kwargs, 'optimizer_kwargs')
    scheduler_class = _get_scheduler_class(args.scheduler) if args.scheduler else None
    scheduler_kwargs = _parse_dict_arg(args.scheduler_kwargs, 'scheduler_kwargs')
    precision = '16-mixed' if args.use_mixed_precision and torch.cuda.is_available() else '32'
    use_amp = args.use_mixed_precision and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    final_checkpoint_path = None
    
    # Create TensorBoard logger for progress tracking across all iterations
    progress_logger = TensorBoardLogger(
        save_dir=str(exp_outdir),
        name='progress',
    )
    
    for iteration in range(1, num_iterations + 1):
        print(f'\n{"-"*60}')
        print(f'SELF-TRAINING ITERATION {iteration}/{num_iterations}')
        print(f'{"-"*60}\n')
        
        # Create iteration directory
        iter_outdir = exp_outdir / f'iteration_{iteration}_temp'
        iter_outdir.mkdir(parents=True, exist_ok=True)
        
        # Build dataloader with ALL samples (no split)
        train_loader = build_segmentation_dataloader(
            dataset='weak_labels',
            split='all',  # Use all samples!
            batch_size=args.batch_size,
            shuffle=True,
            resize_to=tuple(args.resize),
            augment=True,
            num_workers=4,
            seed=args.seed + iteration, 
            weak_label_subfolder=weak_label_subfolder,
        )
        
        # Use same loader for validation (train & val on same data)
        val_loader = build_segmentation_dataloader(
            dataset='weak_labels',
            split='all',  # Use all samples!
            batch_size=args.batch_size,
            shuffle=False,
            resize_to=tuple(args.resize),
            augment=False,
            num_workers=4,
            seed=args.seed + iteration,
            weak_label_subfolder=weak_label_subfolder,
        )
        
        # Build fresh model for this iteration
        model, model_config = build_model(
            name=args.model,
            dataset='weak_labels',
            **_parse_dict_arg(args.model_kwargs, 'model_kwargs'),
        )
        
        # Create LitModule
        litmodel = LitSegmenter(
            model=model,
            criterion=criterion,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            outdir=str(exp_outdir / f'iteration_{iteration}_temp'),
            num_classes=2,
            model_name=model_config['name'],
            model_config=model_config,
            criterion_class=type(criterion),
            criterion_kwargs={},
            precision=precision,
            log_images=False,
        )
        
        # Create trainer for this iteration (NO checkpointing)
        trainer = get_segm_trainer(
            max_epochs=epochs_per_iteration,
            outdir=str(exp_outdir / f'iteration_{iteration}_temp'),
            experiment_name=f'{weak_label_subfolder}_iter{iteration}',
            early_stopping_patience=epochs_per_iteration,
            use_mixed_precision=args.use_mixed_precision,
            log_images=False,
            enable_checkpointing=False,  # No checkpoints during iterations!
        )
        
        # Train
        print(f'Training on {len(str(train_loader.dataset))} samples for {epochs_per_iteration} epochs...')
        trainer.fit(litmodel, train_loader, val_loader)
        
        # Log predictions to TensorBoard after each iteration
        print(f'\nLogging predictions to TensorBoard...')
        log_weak_label_progress(
            model=litmodel.model,
            dataloader=val_loader,
            tensorboard_logger=progress_logger,
            device=device,
            iteration=iteration,
            use_amp=use_amp,
            num_samples=args.num_samples_to_log,
            seed=args.seed,
        )
        
        # Generate predictions and overwrite masks for ALL iterations
        print(f'\nGenerating new pseudo-labels...')
        generate_and_save_predictions(
            model=litmodel.model,
            dataloader=val_loader,
            mask_folder=mask_folder,
            device=device,
            use_amp=use_amp,
        )
        
        print(f'\nCompleted iteration {iteration}/{num_iterations}')
    
    # Final training on refined weak labels after all iterations
    print(f'\n{"="*80}')
    print(f'FINAL TRAINING: {weak_label_subfolder}')
    print(f'Training for up to {final_epochs} epochs on refined pseudo-labels')
    print(f'{"="*80}\n')
    
    # Create final model directory
    final_model_dir = exp_outdir / 'final_model'
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Build dataloaders for final training (with best pseudo-labels from iteration 5)
    final_train_loader = build_segmentation_dataloader(
        dataset='weak_labels',
        split='all',
        batch_size=args.batch_size,
        shuffle=True,
        resize_to=tuple(args.resize),
        augment=True,
        num_workers=4,
        seed=args.seed,
        weak_label_subfolder=weak_label_subfolder,
    )
    
    final_val_loader = build_segmentation_dataloader(
        dataset='weak_labels',
        split='all',
        batch_size=args.batch_size,
        shuffle=False,
        resize_to=tuple(args.resize),
        augment=False,
        num_workers=4,
        seed=args.seed,
        weak_label_subfolder=weak_label_subfolder,
    )
    
    # Build fresh model for final training
    final_model, final_model_config = build_model(
        name=args.model,
        dataset='weak_labels',
        **_parse_dict_arg(args.model_kwargs, 'model_kwargs'),
    )
    
    # Create LitModule for final training
    final_litmodel = LitSegmenter(
        model=final_model,
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        outdir=str(exp_outdir / 'final_model'),
        num_classes=2,
        model_name=final_model_config['name'],
        model_config=final_model_config,
        criterion_class=type(criterion),
        criterion_kwargs={},
        precision=precision,
        log_images=False,
    )
    
    # Create trainer for final training (WITH checkpointing and early stopping!)
    final_trainer = get_segm_trainer(
        max_epochs=final_epochs,
        outdir=str(exp_outdir / 'final_model'),
        experiment_name=f'{weak_label_subfolder}_final',
        early_stopping_patience=10,  # Enable early stopping
        use_mixed_precision=args.use_mixed_precision,
        log_images=False,
        enable_checkpointing=True,  # Save best checkpoint!
    )
    
    # Final training
    print(f'Final training on {len(str(final_train_loader.dataset))} samples...')
    final_trainer.fit(final_litmodel, final_train_loader, final_val_loader)
    
    # Find best checkpoint from final training
    ckpt_dir = exp_outdir / 'final_model' / 'checkpoints'
    final_checkpoint_path = None
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.glob('*.ckpt'))
        if ckpt_files:
            best_ckpt = next((f for f in ckpt_files if 'best' in f.name.lower()), ckpt_files[-1])
            print(f'\nFinal model checkpoint: {best_ckpt}')
            final_checkpoint_path = best_ckpt
    
    # Log final predictions on weak labels
    print(f'\nLogging final predictions on weak labels...')
    log_weak_label_progress(
        model=final_litmodel.model,
        dataloader=final_val_loader,
        tensorboard_logger=progress_logger,
        device=device,
        iteration=num_iterations + 1,  # Iteration 6 = final
        use_amp=use_amp,
        num_samples=args.num_samples_to_log,
        seed=args.seed,
    )
    
    print(f'\n{"="*80}')
    print(f'EVALUATING FINAL MODEL ON PH2 TEST SET')
    print(f'{"="*80}\n')
    
    # Evaluate on real PH2 test set
    if final_checkpoint_path is not None:
        evaluate_on_ph2_test_set(
            checkpoint_path=final_checkpoint_path,
            model=final_model,
            criterion=criterion,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=args.batch_size,
            resize_to=tuple(args.resize),
            device=device,
            use_amp=use_amp,
            seed=args.seed,
            log_images_to_tensorboard=True,
            tensorboard_logger=progress_logger,
            experiment_name=f'{weak_label_subfolder}_final_eval',
            num_samples_to_log=args.num_samples_to_log,
        )
    
    print(f'\nCompleted iterative weak label training for: {weak_label_subfolder}')
    print(f'Results saved in: {exp_outdir}\n')
    
    return final_checkpoint_path
    

def main():
    args = _build_args()
    
    pl.seed_everything(args.seed, workers=True)
    
    if args.dataset == 'weak_labels':
        print("\n" + "="*80)
        print("WEAK LABELS ABLATION STUDY")
        print("="*80)
        
        # Find all subfolders in region_growing_masks/
        weak_labels_root = Path('./region_growing_masks')
        if not weak_labels_root.exists():
            raise FileNotFoundError(f"Weak labels directory not found: {weak_labels_root}")
        
        all_subfolders = sorted([d.name for d in weak_labels_root.iterdir() if d.is_dir()])
        
        if not all_subfolders:
            raise ValueError(f"No subfolders found in {weak_labels_root}")
        
        if args.weak_folders:
            wanted = [w.lower() for w in args.weak_folders]
            subfolders = [f for f in all_subfolders if f.lower() in wanted]
            if not subfolders:
                raise ValueError(
                    f"No matching weak-label subfolders found for {args.weak_folders}. "
                    f"Available: {all_subfolders}"
                )
            print(f"\nFiltered to {len(subfolders)} weak label folders: {subfolders}")
        else:
            subfolders = all_subfolders
            print(f"\nFound {len(subfolders)} weak label folders: {subfolders}")


        print(f"Training config: {args.weak_iterations} iterations x {args.weak_epochs_per_iter} epochs x {args.weak_final_epochs} final epochs\n")

        final_models = {}
        
        for subfolder in subfolders:
            final_ckpt = _iterative_weak_label_training(
                args=args,
                weak_label_subfolder=subfolder,
                num_iterations=args.weak_iterations,
                epochs_per_iteration=args.weak_epochs_per_iter,
                final_epochs=args.weak_final_epochs,
            )
            final_models[subfolder] = final_ckpt
        
        print("\n" + "="*80)
        print("WEAK LABELS ABLATION STUDY COMPLETED")
        print("="*80)
        print("\nFinal models:")
        for subfolder, ckpt_path in final_models.items():
            print(f"  {subfolder}: {ckpt_path}")
        
        return
    
    # Standard training for DRIVE/PH2, Build dataloaders (shared for all experiments)
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
        split='val',  
        batch_size=args.batch_size,
        shuffle=False,
        resize_to=tuple(args.resize),
        augment=False,
        num_workers=4,
        seed=args.seed,
    )

    # Preview logging of example batch
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
        
        experiments = [
            {'name': 'bce', 'kwargs': {}},
            {'name': 'weighted_bce', 'kwargs': {'pos_weight': 10.0}},
            {'name': 'focal', 'kwargs': {}},
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

    