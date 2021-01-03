from typing import Union, List, Optional
import numpy as np
import pandas as pd


def change_column_name(df: Union[pd.DataFrame, pd.Series],
                       old: Union[str, List[str]],
                       new: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(old, str) and isinstance(new, str):
        return df.rename(columns={old: new})

    if len(old) != len(new):
        raise ValueError(f'The length of names are different: old={old}, new={new}')
    name_map = {}
    for o, n in zip(old, new):
        name_map[o] = n
    return df.rename(columns=name_map)


def change_dtype(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    '''
    DataFrame の dtype を最適化
    ref: https://github.com/Ynakatsuka/kaggle_utils/blob/master/kaggle_utils/utils/__init__.py
    '''
    if columns is None:
        columns = df.columns

    for col in columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:3] == 'flo':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # to avoid feather writting error
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df
