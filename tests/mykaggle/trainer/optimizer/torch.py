from typing import Any, Dict
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_optimizer(settings: Dict[str, Any], parameters) -> torch.optim.Optimizer:
    name = settings.get('optimizer')
    if name == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=settings['learning_rate'],
            weight_decay=settings['weight_decay']
        )
    else:
        optimizer = torch.optim.Adam(
            parameters,
            lr=settings['learning_rate'],
            weight_decay=settings['weight_decay']
        )
    return optimizer


def get_scheduler(settings: Dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler = settings.get('scheduler')
    if scheduler is None:
        return None
    elif scheduler == 'CosineAnnealingWarmupRestarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            T_0=settings['scheduler_T_0'],
            T_multi=1,
            eta_min=settings['scheduler_min_lr'],
            last_epoch=-1,
            warmup_epochs=settings['warmup_epochs'],
        )
    elif scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=settings['scheduler_T_0'],
            T_mult=1,
            eta_min=settings['scheduler_min_lr'],
            last_epoch=-1
        )
    elif scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=settings['learning_rate'],
            max_lr=settings['scheduler_max_lr'],
            step_size_up=settings['step_size_up'],
            mode='exp_range',
            gamma=0.995,
            cycle_momentum=False
        )
    return scheduler


class CosineAnnealingWarmupRestarts(CosineAnnealingWarmRestarts):
    ''' extension of official CosineAnnealingWarmRestarts for warmup '''
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_multi: int = 1,
        eta_min: float = 0.000001,  # type: ignore
        last_epoch: int = -1,
        verbose: bool = False,
        warmup_epochs: float = 0.,
    ) -> None:
        '''
        Args:
            T_0: 半波長の epoch 数
        Property:
            T_cur: 現在のサイクルでの epoch 数 (float, 1.5 だったら 1.5 epoch 分)
        '''
        super(CosineAnnealingWarmupRestarts, self).__init__(
            optimizer, T_0, T_multi, eta_min, last_epoch, verbose  # type: ignore
        )
        self.warmup_epochs = warmup_epochs  # warmup step size

    def get_lr(self):
        if self.T_cur < self.warmup_epochs:
            warmup_lr = self.get_lr_in_warmup(self.T_cur, self.warmup_epochs, self.eta_min)
            original_lr = super(CosineAnnealingWarmupRestarts, self).get_lr()
            return min(warmup_lr, original_lr)
        else:
            return super(CosineAnnealingWarmupRestarts, self).get_lr()

    def get_lr_in_warmup(self, cur: float, warmup_epochs: float, min_lr: float):
        '''
        Args:
            cur: epoch now (float; 1.5 = 1 epoch and half)
            warmup_epochs: do warmup in this epochs
        '''
        diff = cur / warmup_epochs
        lrs = [
            max(base_lr * diff, min_lr)
            for base_lr in self.base_lrs  # type: ignore
        ]
        return lrs
