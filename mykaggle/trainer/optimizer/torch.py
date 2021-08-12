from typing import Any, Dict
import torch
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def get_optimizer(name: str, lr: float, weight_decay: float, parameters, **kwargs) -> torch.optim.Optimizer:
    optimizer: torch.optim.Optimizer
    if name == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    return optimizer


def get_scheduler(settings: Dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_name = settings.get('scheduler')
    scheduler: _LRScheduler
    if scheduler_name is None:
        return None
    elif scheduler_name == 'ExponentialDecay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 1 / (1 + e), last_epoch=-1
        )
    elif scheduler_name == 'linear_decay':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            0,
            settings['num_total_steps'],
        )
    elif scheduler_name == 'linear_decay_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            settings['warmup_epochs'] * settings['num_batches'],
            settings['num_total_steps'],
        )
    elif scheduler_name == 'cosine_decay':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            0,
            settings['num_total_steps'],
        )
    elif scheduler_name == 'cosine_decay_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            settings['warmup_epochs'] * settings['num_batches'],
            settings['num_total_steps'],
        )
    return scheduler
