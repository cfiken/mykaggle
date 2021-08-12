
from typing import Any, Dict
from torch.utils.data import DataLoader, Dataset

from mykaggle.trainer.base import Mode


def get_dataloader(
    settings: Dict[str, Any],
    dataset: Dataset,
    mode: Mode,
    fold: int,
    *args, **kwargs
) -> DataLoader:
    batch_size = settings['batch_size'] if mode == Mode.TRAIN else settings['test_batch_size']
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=mode == Mode.TRAIN,
        num_workers=settings['num_workers'],
    )
    return dataloader
