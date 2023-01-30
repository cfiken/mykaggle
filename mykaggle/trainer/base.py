from typing import Tuple
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Mode(Enum):
    TRAIN = 'TRAIN'
    VALID = 'VALID'
    TEST = 'TEST'


@dataclass
class TrainingStates:
    batch_step: int = 0
    optimizer_step: int = 0
    metric: float = 0
    loss: float = 0
    oof_preds: np.ndarray = np.zeros(1)
    best_preds: np.ndarray = np.zeros(1)

    def batch(self):
        self.batch_step += 1

    def step(self):
        self.optimizer_step += 1

    def update(self, preds: np.ndarray, metric: float, loss: float):
        self.oof_preds = preds
        self.metric = metric
        self.loss = loss


class TrainerBase:
    @abstractmethod
    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        *args, **kwargs
    ) -> np.ndarray:
        pass

    @abstractmethod
    def validation(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        *args, **kwargs
    ) -> Tuple[np.ndarray, ...]:
        pass
