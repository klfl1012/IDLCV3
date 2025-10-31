from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch.nn as nn
from unet import UNet


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build_fn: Callable[..., nn.Module]
    in_channels: int
    out_channels: int
    description: str = ''
    default_params: Optional[dict] = None


# default configs for models
DRIVE_CONFIG = {
    'in_channels': 3,      # RGB images
    'out_channels': 1,     # Binary segmentation (vessels)
    'features': [64, 128, 256, 512],
}

PH2_CONFIG = {
    'in_channels': 3,      # RGB images
    'out_channels': 1,     # Binary segmentation (lesions)
    'features': [64, 128, 256, 512],
}


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    'unet': ModelSpec(
        name='UNet',
        build_fn=lambda **kwargs: UNet(**kwargs),
        in_channels=3,
        out_channels=1,
        description='Standard U-Net architecture for medical image segmentation.',
        default_params={
            'in_channels': 3,
            'out_channels': 1,
            'features': [64, 128, 256, 512],
            'activation': nn.SiLU,
            'norm_groups': 8,
            'dropout': 0.1,
        },
    ),
    'unet_small': ModelSpec(
        name='UNet-Small',
        build_fn=lambda **kwargs: UNet(**kwargs),
        in_channels=3,
        out_channels=1,
        description='Smaller U-Net for faster training and lower memory usage.',
        default_params={
            'in_channels': 3,
            'out_channels': 1,
            'features': [32, 64, 128, 256],
            'activation': nn.SiLU,
            'norm_groups': 8,
            'dropout': 0.1,
        },
    ),
    'unet_large': ModelSpec(
        name='UNet-Large',
        build_fn=lambda **kwargs: UNet(**kwargs),
        in_channels=3,
        out_channels=1,
        description='Larger U-Net with more capacity for complex segmentation tasks.',
        default_params={
            'in_channels': 3,
            'out_channels': 1,
            'features': [64, 128, 256, 512, 1024],
            'activation': nn.SiLU,
            'norm_groups': 8,
            'dropout': 0.2,
        },
    ),
}


def available_models() -> Tuple[str, ...]:
    
    return tuple(MODEL_REGISTRY.keys())


def resolve_model(name: str) -> ModelSpec:

    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise KeyError(
            f'Unknown model "{name}". Available models: {available}'
        )
    return MODEL_REGISTRY[key]


def build_model(
    name: str,
    dataset: str = 'drive',
    **model_kwargs
) -> tuple[nn.Module, dict]:
    spec = resolve_model(name)
    
    # default parameters
    params = spec.default_params.copy() if spec.default_params else {}
    
    # apply dataset-specific config
    dataset_key = dataset.lower()
    if dataset_key == 'drive':
        params.update(DRIVE_CONFIG)
    elif dataset_key == 'ph2':
        params.update(PH2_CONFIG)
    else:
        raise ValueError(f'Unknown dataset "{dataset}". Choose "drive" or "ph2".')
    
    # override with user-provided kwargs
    params.update(model_kwargs)
    
    model = spec.build_fn(**params)
    
    # Config für Reconstruction
    model_config = {
        'name': name,
        'dataset': dataset,
        'kwargs': params,
    }
    
    print(f'Built {spec.name} for {dataset.upper()} dataset')
    print(f'  Input channels: {params["in_channels"]}, Output channels: {params["out_channels"]}')
    if 'features' in params:
        print(f'  Feature channels: {params["features"]}')
    
    return model, model_config


def rebuild_model_from_config(model_config: dict) -> nn.Module:
    """Rebuild model from saved config."""
    return build_model(
        name=model_config['name'],
        dataset=model_config['dataset'],
        **model_config['kwargs']
    )[0]


__all__ = [
    'ModelSpec',
    'MODEL_REGISTRY',
    'available_models',
    'resolve_model',
    'build_model',
    'rebuild_model_from_config',
    'DRIVE_CONFIG',
    'PH2_CONFIG',
]