from typing import Any, Union, List
import pandas as pd

from mykaggle.transform.base import BaseTransform


class BaseGroupByTransform(BaseTransform):
    '''
    GroupBy 系の変換をかける Transform の Base クラス。
    aggregate の実装が必要。
    '''

    def __init__(self, keys: List[str], targets: List[str], aggs: List[str]) -> None:
        '''
        Args:
          keys: 集約キー
          targets: 集約後に計算を行うカラム
          aggs: 集約後に行う演算
        '''
        self.keys = keys
        self.targets = targets
        self.aggs = aggs

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def fit(self, X: pd.DataFrame) -> 'BaseGroupByTransform':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate(df, self.keys, self.targets, self.aggs)

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        targets: List[str],
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def _prepare_aggregated_columns(
        self,
        keys: List[str],
        targets: List[str],
        aggs: List[Union[str, Any]]
    ) -> List[str]:
        aggs = [a if isinstance(a, str) else a.__name__ for a in aggs]
        return ['_'.join([a, target, 'groupby'] + keys) for target in targets for a in aggs]


class BasicGroupByTransform(BaseGroupByTransform):
    '''
    mean/median/sum/std など、基本的な演算を行う GroupByTransform.
    '''

    def __init__(self, keys: List[str], targets: List[str], aggs: List[str]) -> None:
        self.keys = keys
        self.targets = targets
        self.aggs = aggs

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        targets: List[str],
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        columns = keys + targets
        df_output = df.loc[:, columns].groupby(keys)[targets].agg(aggs).reset_index()
        df_output.columns = keys + self._prepare_aggregated_columns(keys, targets, aggs)
        return df_output
