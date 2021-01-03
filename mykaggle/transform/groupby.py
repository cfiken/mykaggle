from typing import Any, Union, List, Optional
import pandas as pd

from mykaggle.transform.base import BaseTransform
from mykaggle.lib.pandas_util import change_dtype


class BaseGroupByTransform(BaseTransform):
    '''
    GroupBy 系の変換をかける Transform の Base クラス。
    aggregate の実装が必要。
    '''

    def __init__(self, keys: List[str], values: List[str], aggs: List[str], *args, **kwargs) -> None:
        '''
        Args:
            keys: 集約キー
            targets: 集約後に計算を行うカラム
            aggs: 集約後に行う演算
        '''
        self.keys = keys
        self.values = values
        self.aggs = aggs
        self.features: List[pd.DataFrame] = []

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def fit(self, X: pd.DataFrame) -> 'BaseGroupByTransform':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate(df, self.keys, self.values, self.aggs)

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        values: List[str],
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def _get_column_names(
        self,
        keys: List[str],
        values: List[str],
        aggs: Optional[List[Union[str, Any]]] = None,
        prefix: Optional[str] = None
    ) -> List[str]:
        if aggs is None:
            if prefix:
                return ['_'.join([prefix, v, 'groupby'] + keys) for v in values]
            else:
                return ['_'.join([v, 'groupby'] + keys) for v in values]

        aggs = [a if isinstance(a, str) else a.__name__ for a in aggs]
        if prefix:
            return ['_'.join([prefix, a, v, 'groupby'] + keys) for v in values for a in aggs]
        else:
            return ['_'.join([a, v, 'groupby'] + keys) for v in values for a in aggs]


class GroupByTransform(BaseGroupByTransform):
    '''
    mean/median/sum/std など、基本的な演算を行う GroupByTransform.
    '''

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        values: List[str],
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        columns = list(set(keys + values))
        df_output = df.loc[:, columns].groupby(keys)[values].agg(aggs).reset_index()
        new_columns = self._get_column_names(keys, values, aggs)
        df_output.columns = keys + new_columns
        df_output = change_dtype(df_output, columns=new_columns)
        self.features.append(df_output)
        return df_output


class DiffGroupByTransform(BaseGroupByTransform):

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        values: List[str],
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        all_features = list(set(keys + values))

        base_column_names = self._get_column_names(keys, values, aggs)
        new_column_names = self._get_column_names(keys, values, aggs, prefix='diff')

        df_base = df[all_features].groupby(keys)[values].agg(aggs).reset_index()
        df_base.columns = keys + base_column_names
        df_base = df[all_features].merge(df_base, on=keys, how='left')

        for i, (base_feature, new_feature) in enumerate(zip(base_column_names, new_column_names)):
            v = values[i // len(aggs)]
            df_base[new_feature] = df_base[base_feature] - df_base[v]

        df_output = df_base[keys + new_column_names]
        self.features.append(df_output)
        return df_output

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate(df, self.keys, self.values, self.aggs)


class RatioGroupByTransform(BaseGroupByTransform):

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        values: List[str],
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        all_features = list(set(keys + values))

        base_column_names = self._get_column_names(keys, values, aggs)
        new_column_names = self._get_column_names(keys, values, aggs, prefix='ratio')

        df_base = df[all_features].groupby(keys)[values].agg(aggs).reset_index()
        df_base.columns = keys + base_column_names
        df_base = df[all_features].merge(df_base, on=keys, how='left')

        for i, (base_feature, new_feature) in enumerate(zip(base_column_names, new_column_names)):
            v = values[i // len(aggs)]
            df_base[new_feature] = df_base[v] / df_base[base_feature]

        df_output = df_base[keys + new_column_names]
        self.features.append(df_output)
        return df_output

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate(df, self.keys, self.values, self.aggs)


class ShiftGroupByTransform(BaseGroupByTransform):

    def __init__(
        self,
        keys: List[str],
        values: List[str],
        shift: List[int],
        fillna: Union[None, int, str] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(keys, values, [], *args, **kwargs)
        self.shift = shift
        self.fillna = fillna

        aggs = kwargs.get('aggs')
        if aggs is not None:
            raise ValueError(f'Argument `aggs` is not used in this class. Now {aggs} is specified.')

    def aggregate(
        self,
        df: pd.DataFrame,
        keys: List[str],
        values: List[str],
        shift: List[int],
        fillna: Union[None, int, str],
        *args, **kwargs
    ) -> pd.DataFrame:
        all_features = list(set(keys + values))
        all_new_column_names = []
        for s in shift:
            new_column_names = self._get_column_names(keys, values, prefix=f'lag_{s}')
            all_new_column_names.extend(new_column_names)
            df_shift = df[all_features].groupby(keys)[values].shift(s)
            df[new_column_names] = df_shift[values] - df[values]
        if fillna is not None:
            df = df.fillna(fillna)
        df_output = df[keys + all_new_column_names]
        self.features.append(df_output)
        return df_output

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate(df, self.keys, self.values, self.shift)
