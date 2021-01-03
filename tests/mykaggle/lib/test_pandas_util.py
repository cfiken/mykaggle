import pytest

import numpy as np
import pandas as pd

from mykaggle.lib import pandas_util as util


class TestPandasUtil:

    def test_change_column_name(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        df = util.change_column_name(df, 'a', 'aaa')
        assert 'aaa' in list(df.columns)
        assert 'a' not in list(df.columns)
        assert 'b' in list(df.columns)  # remain other columns

        df = util.change_column_name(df, ['b'], ['bbb'])
        assert 'bbb' in list(df.columns)
        assert 'b' not in list(df.columns)

        df = util.change_column_name(df, ['x', 'y'], ['xxx', 'yyy'])
        assert 'xxx' in list(df.columns)
        assert 'yyy' in list(df.columns)
        assert 'x' not in list(df.columns)
        assert 'y' not in list(df.columns)

        with pytest.raises(ValueError):
            df = util.change_column_name(df, ['x', 'y'], ['xxx', 'yyy', 'zzz'])

    def test_change_dtype(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        x_dtype = df['x'].dtype
        expected_x_dtype = np.float32
        y_dtype = df['y'].dtype
        print(x_dtype, y_dtype)
        df = util.change_dtype(df, columns=['x'])
        assert df['x'].dtype != x_dtype
        assert df['x'].dtype == expected_x_dtype
        assert df['y'].dtype == y_dtype

        df = util.change_dtype(df)
        expected_y_dtype = np.int8
        assert df['y'].dtype != y_dtype
        assert df['y'].dtype == expected_y_dtype
