from typing import Any, Dict
import pandas as pd
from torch.utils.data import Dataset

from mykaggle.trainer.base import Mode


class DatasetBase(Dataset):
    def __init__(
        self,
        settings: Dict[str, Any],
        mode: Mode,
        df: pd.DataFrame,
        *args, **kwargs,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.mode = mode
        self.df = df.copy()


def get_dataset(
    settings: Dict[str, Any],
    mode: Mode,
    df: pd.DataFrame,
    *args, **kwargs
) -> DatasetBase:
    return DatasetBase(settings, mode, df, *args, **kwargs)
