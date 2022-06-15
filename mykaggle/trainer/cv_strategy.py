from typing import List, Dict, Tuple, Union, Optional, Type
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


class CVStrategy:

    @classmethod
    def create(cls, name: str, num_splits: int) -> 'CVStrategy':
        StrategyMap: Dict[str, Type[CVStrategy]] = {
            'random': Random,
            'stratified': Stratified,
            'group': Group,
        }
        target_cls = StrategyMap[name]
        return target_cls(num_splits)

    def __init__(self, num_splits: int) -> None:
        self.num_splits = num_splits

    def preprocess(self, df: pd.DataFrame, *args, **kwargs) -> None:
        return df

    def split(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        *,
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_column: Optional[str] = None,
        group: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[List[int], List[int]]]:
        '''
        前処理し、各ストラテジーのアルゴリズムで分割を行う関数
        '''
        df = self.preprocess(df)
        return self._split(df, y, y_column, group, group_column)

    def _split(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_column: Optional[str] = None,
        group: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[List[int], List[int]]]:
        '''
        各ストラテジーの実装を書く関数
        '''
        raise NotImplementedError('You must implement a split method.')

    def split_and_set(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_column: Optional[str] = None,
        group: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        '''
        各ストラテジーの分割を元の DataFrame に 'fold' カラムとして入れる関数
        '''
        df = self.preprocess(df)
        splits = self._split(df, y, group, *args, **kwargs)
        for i, (_, valid_idx) in enumerate(splits):
            df.loc[valid_idx, kwargs.get('fold_column', 'fold')] = i
        return df


class Random(CVStrategy):

    def _split(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_column: Optional[str] = None,
        group: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[List[int], List[int]]]:
        splitter = KFold(self.num_splits)
        splits = list(splitter.split(df))
        return splits


class Stratified(CVStrategy):

    def _split(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_column: Optional[str] = None,
        group: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[List[int], List[int]]]:
        if y is None and y_column is None:
            raise ValueError('You need pass either `y` or `y_column`')
        df = self.preprocess(df)
        y = y if y is not None else df[y_column]
        splitter = StratifiedKFold(self.num_splits)
        splits = list(splitter.split(df, y=y))
        return splits


class Group(CVStrategy):

    def _split(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_column: Optional[str] = None,
        group: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[List[int], List[int]]]:
        if y is None and y_column is None:
            raise ValueError('You need pass either `y` or `y_column`')
        y = y if y is not None else df[y_column]
        if group is None and group_column is None:
            raise ValueError('You need pass either `group` or `group_column`')
        group = group if group is not None else df[group_column]
        splitter = GroupKFold(self.num_splits)
        splits = list(splitter.split(df, group=group))
        return splits
