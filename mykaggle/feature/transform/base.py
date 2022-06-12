from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransform(ABC, BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def fit(self, X: pd.DataFrame):
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
