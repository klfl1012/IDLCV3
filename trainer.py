import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm
from pathlib import Path
from typing import Optional, Type, Any
from torch.profiler import ProfilerActivity, schedule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchvision.utils import make_grid
import segmentation_models_pytorch as smp


class LitSegmenter(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[dict] = None,
        outdir: str = './results',
        num_classes: int = 2,
        log_imgs_every_n_epochs: int = 5,
        model_name: Optional[str] = None,
        model_config: Optional[dict] = None,
        criterion_class: Optional[Type[nn.Module]] = None,
        criterion_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])

        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.log_imgs_every_n_epochs = log_imgs_every_n_epochs
        self.num_classes = num_classes
        self.outdir = Path(outdir)
        os.makedirs(self.outdir, exist_ok=True)

        # metrics
        task = 'binary' if num_classes == 2 else 'multiclass'
        self.train_iou = tm.JaccardIndex(task=task, num_classes=num_classes)
        self.val_iou = tm.JaccardIndex(task=task, num_classes=num_classes)
        self.train_acc = tm.Accuracy(task=task, num_classes=num_classes)
        self.val_acc = tm.Accuracy(task=task, num_classes=num_classes)
        self.train_sensitivity = tm.Recall(task=task, num_classes=num_classes)  # True Positive Rate
        self.val_sensitivity = tm.Recall(task=task, num_classes=num_classes)
        self.train_specificity = tm.Specificity(task=task, num_classes=num_classes)  # True Negative Rate
        self.val_specificity = tm.Specificity(task=task, num_classes=num_classes)
        self.train_precision = tm.Precision(task=task, num_classes=num_classes)
        self.val_precision = tm.Precision(task=task, num_classes=num_classes)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_acc': [],
            'val_acc': [],
            'train_sensitivity': [],
            'val_sensitivity': [],
            'train_specificity': [],
            'val_specificity': [],
            'train_precision': [],
            'val_precision': []
        }

        self._train_epoch_outputs = []
        self._val_epoch_outputs = []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    
    def _compute_loss_and_metrics(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str
    ) -> dict[str, torch.Tensor]:
        
        imgs, masks = batch
        logits = self(imgs)

        if self.num_classes == 2:
            logits = logits.squeeze(1)
            loss = self.criterion(logits, masks.float())
            preds = (torch.sigmoid(logits) > 0.5).long()

        else:
            loss = self.criterion(logits, masks)
            preds = logits.argmax(dim=1)

        metrics = (
            getattr(self, f'{stage}_iou'), 
            getattr(self, f'{stage}_acc'),
            getattr(self, f'{stage}_sensitivity'),
            getattr(self, f'{stage}_specificity'),
            getattr(self, f'{stage}_precision')
        )
        for metric in metrics:
            metric(preds, masks)

        return {
            'loss': loss,
            'preds': preds,
            'masks': masks,
            'imgs': imgs,
            'logits': logits,
        }

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        outputs = self._compute_loss_and_metrics(batch, 'train')
        self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0))
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_sensitivity', self.train_sensitivity, on_step=False, on_epoch=True)
        self.log('train_specificity', self.train_specificity, on_step=False, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True)

        self._train_epoch_outputs.append(outputs)

        if batch_idx == 0 and self.current_epoch % self.log_imgs_every_n_epochs == 0:
            self._train_vis_batch = {
                'imgs': outputs['imgs'][:4].detach().cpu(),
                'masks': outputs['masks'][:4].detach().cpu(),
                'preds': outputs['preds'][:4].detach().cpu(),
            }
        
        return outputs['loss']
    

    def validation_step(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        
        outputs = self._compute_loss_and_metrics(batch, 'val')
        self.log('val_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0))
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_sensitivity', self.val_sensitivity, on_step=False, on_epoch=True)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        
        self._val_epoch_outputs.append(outputs['loss'].detach())
        
        if batch_idx == 0 and self.current_epoch % self.log_imgs_every_n_epochs == 0:
            self._val_vis_batch = {
                'imgs': outputs['imgs'][:4].detach().cpu(), 
                'preds': outputs['preds'][:4].detach().cpu(),
                'masks': outputs['masks'][:4].detach().cpu(),
            }
        
        return outputs['loss']
    
    def on_train_epoch_end(self) -> None:
        if self._train_epoch_outputs:
            losses = [out['loss'] if isinstance(out, dict) else out for out in self._train_epoch_outputs]
            avg_loss = torch.stack(losses).mean().item()
            self.history['train_loss'].append(avg_loss)
            self.history['train_iou'].append(self.train_iou.compute().item())
            self.history['train_acc'].append(self.train_acc.compute().item())
            self.history['train_sensitivity'].append(self.train_sensitivity.compute().item())
            self.history['train_specificity'].append(self.train_specificity.compute().item())
            self.history['train_precision'].append(self.train_precision.compute().item())
            self._train_epoch_outputs.clear()

        if hasattr(self, '_train_vis_batch') and self.current_epoch % self.log_imgs_every_n_epochs == 0:
            self._log_segmentations_imgs(self._train_vis_batch, 'train')
            delattr(self, '_train_vis_batch')

    
    def on_validation_epoch_end(self) -> None:
        if self._val_epoch_outputs:
            avg_loss = torch.stack(self._val_epoch_outputs).mean().item()
            self.history['val_loss'].append(avg_loss)
            self.history['val_iou'].append(self.val_iou.compute().item())
            self.history['val_acc'].append(self.val_acc.compute().item())
            self.history['val_sensitivity'].append(self.val_sensitivity.compute().item())
            self.history['val_specificity'].append(self.val_specificity.compute().item())
            self.history['val_precision'].append(self.val_precision.compute().item())
            self._val_epoch_outputs.clear()

        if hasattr(self, '_val_vis_batch') and self.current_epoch % self.log_imgs_every_n_epochs == 0:
            self._log_segmentations_imgs(self._val_vis_batch, 'val')
            delattr(self, '_val_vis_batch')

    def _log_segmentations_imgs(
        self,
        batch_data: dict[str, torch.Tensor],
        stage: str
    ) -> None:
        
        imgs = batch_data['imgs']
        masks = batch_data['masks']
        preds = batch_data['preds']

        preds_viz = preds.unsqueeze(1).float() / max(self.num_classes - 1, 1)
        masks_viz = masks.unsqueeze(1).float() / max(self.num_classes - 1, 1)

        grid = torch.cat([imgs, preds_viz.repeat(1, 3, 1, 1), masks_viz.repeat(1, 3, 1, 1)], dim=0)
        grid = make_grid(grid, nrow=len(imgs), normalize=True, value_range=(0, 1))

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f'{stage}_preds',
                grid,
                global_step=self.current_epoch
            )
        
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

        if self.scheduler_class is None:
            return {'optimizer': optimizer}
        
        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        
        config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            },
        }

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            config['lr_scheduler']['monitor'] = 'val_loss'

        return config
    


def get_segm_trainer(
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
    outdir: str = './results',
    experiment_name: str = 'segmentation_experiment',
    enable_profiler: bool = False,
    gradient_clip_val: Optional[float] = 1.0,
    accumulate_grad_batches: int = 1,
    use_mixed_precision: bool = False,
) -> pl.Trainer:
    
    outdir_path = Path(outdir)
    os.makedirs(outdir_path, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            monitor='val_iou',
            dirpath=outdir_path / 'checkpoints',
            filename=f'{experiment_name}_{{epoch:02d}}_{{val_iou:.4f}}',
            save_top_k=3,
            mode='max',
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    loggers = [
        TensorBoardLogger(
            save_dir=outdir_path / 'logs',
            name=experiment_name,
        ),
        CSVLogger(
            save_dir=outdir_path / 'logs',
            name=experiment_name,
        ),
    ]

    profiler = None
    if enable_profiler:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        profiler = PyTorchProfiler(
            dirpath=outdir_path,
            filename='profiler_trace.json',
            schedule=schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1,
            ),
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        logger=loggers,
        profiler=profiler,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed' if use_mixed_precision and torch.cuda.is_available() else 32
    )

    return trainer