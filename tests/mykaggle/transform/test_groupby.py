from pytest import approx
import numpy as np
import pandas as pd

from mykaggle.transform.groupby import (
    BaseGroupByTransform, GroupByTransform,
    DiffGroupByTransform, RatioGroupByTransform,
    ShiftGroupByTransform,
)


class TestBaseGroupByTransform:
    def test_prepare_columns(self):
        keys = ['a', 'b']
        values = ['x', 'y']
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_groupby_a_b', 'sum_x_groupby_a_b',
            'mean_y_groupby_a_b', 'sum_y_groupby_a_b'
        ]
        transform = BaseGroupByTransform(keys, values, aggs)
        columns = transform._get_column_names(keys, values, aggs)

        assert expected_columns == columns


class TestGroupByTransform:

    def test_aggregate(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        keys = ['a', 'b']
        values = ['x', 'y']
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_groupby_a_b', 'sum_x_groupby_a_b',
            'mean_y_groupby_a_b', 'sum_y_groupby_a_b'
        ]
        expected_mean_x_groupby_a_b_1 = (0.4 + 0.1 + 0.2) / 3
        expected_mean_x_groupby_a_b_2 = 0.3
        sum_y_groupby_a_b_1 = 0.4 + 0.1 + 0.2
        sum_y_groupby_a_b_2 = 0.3

        transform = GroupByTransform(keys, values, aggs)
        df_output = transform.aggregate(df, keys, values, aggs)

        assert df_output.columns.tolist() == keys + expected_columns
        assert df_output.loc[0, 'mean_x_groupby_a_b'] == approx(expected_mean_x_groupby_a_b_1)
        assert df_output.loc[1, 'mean_x_groupby_a_b'] == approx(expected_mean_x_groupby_a_b_2)
        assert df_output.loc[0, 'sum_x_groupby_a_b'] == approx(sum_y_groupby_a_b_1)
        assert df_output.loc[1, 'sum_x_groupby_a_b'] == approx(sum_y_groupby_a_b_2)


class TestDiffGroupByTransform:

    def test_aggregate(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        keys = ['b']
        values = ['x', 'y']
        aggs = ['mean']

        expected_columns = [
            'diff_mean_x_groupby_b',
            'diff_mean_y_groupby_b',
        ]
        expected_mean_x_groupby_b_aa = (0.1 + 0.2 + 0.4 + 0.4) / 4
        expected_mean_x_groupby_b_bb = (0.3 + 0.6) / 2
        expected_mean_x_groupby_b_cc = (0.2 + 0.3 + 0.1 + 0.7 + 0.2 + 0.3) / 6
        expected_mean_x_groupby_b_dd = (0.4 + 0.3 + 0.2) / 3
        expected_mean_y_groupby_b_aa = (2 + 2 + 2 + 2) / 4
        expected_mean_y_groupby_b_bb = (2 + 5) / 2
        expected_mean_y_groupby_b_cc = (2 + 2 + 2 + 5 + 1 + 3) / 6
        expected_mean_y_groupby_b_dd = (2 + 2 + 1) / 3

        transform = DiffGroupByTransform(keys, values, aggs)
        df_output = transform.aggregate(df, keys, values, aggs)

        assert df_output.columns.tolist() == keys + expected_columns
        assert df_output.loc[0, expected_columns[0]] == approx(expected_mean_x_groupby_b_aa - df.loc[0, 'x'])
        assert df_output.loc[0, expected_columns[1]] == approx(expected_mean_y_groupby_b_aa - df.loc[0, 'y'])
        assert df_output.loc[1, expected_columns[0]] == approx(expected_mean_x_groupby_b_bb - df.loc[1, 'x'])
        assert df_output.loc[1, expected_columns[1]] == approx(expected_mean_y_groupby_b_bb - df.loc[1, 'y'])
        assert df_output.loc[2, expected_columns[0]] == approx(expected_mean_x_groupby_b_aa - df.loc[2, 'x'])
        assert df_output.loc[2, expected_columns[1]] == approx(expected_mean_y_groupby_b_aa - df.loc[2, 'y'])
        assert df_output.loc[6, expected_columns[0]] == approx(expected_mean_x_groupby_b_cc - df.loc[6, 'x'])
        assert df_output.loc[6, expected_columns[1]] == approx(expected_mean_y_groupby_b_cc - df.loc[6, 'y'])
        assert df_output.loc[10, expected_columns[0]] == approx(expected_mean_x_groupby_b_dd - df.loc[10, 'x'])
        assert df_output.loc[10, expected_columns[1]] == approx(expected_mean_y_groupby_b_dd - df.loc[10, 'y'])


class TestRatioGroupByTransform:

    def test_aggregate(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        keys = ['b']
        values = ['x', 'y']
        aggs = ['mean']

        expected_columns = [
            'ratio_mean_x_groupby_b',
            'ratio_mean_y_groupby_b',
        ]
        expected_mean_x_groupby_b_aa = (0.1 + 0.2 + 0.4 + 0.4) / 4
        expected_mean_x_groupby_b_bb = (0.3 + 0.6) / 2
        expected_mean_x_groupby_b_cc = (0.2 + 0.3 + 0.1 + 0.7 + 0.2 + 0.3) / 6
        expected_mean_x_groupby_b_dd = (0.4 + 0.3 + 0.2) / 3
        expected_mean_y_groupby_b_aa = (2 + 2 + 2 + 2) / 4
        expected_mean_y_groupby_b_bb = (2 + 5) / 2
        expected_mean_y_groupby_b_cc = (2 + 2 + 2 + 5 + 1 + 3) / 6
        expected_mean_y_groupby_b_dd = (2 + 2 + 1) / 3

        transform = RatioGroupByTransform(keys, values, aggs)
        df_output = transform.aggregate(df, keys, values, aggs)

        assert df_output.columns.tolist() == keys + expected_columns
        assert df_output.loc[0, expected_columns[0]] == approx(df.loc[0, 'x'] / expected_mean_x_groupby_b_aa)
        assert df_output.loc[0, expected_columns[1]] == approx(df.loc[0, 'y'] / expected_mean_y_groupby_b_aa)
        assert df_output.loc[1, expected_columns[0]] == approx(df.loc[1, 'x'] / expected_mean_x_groupby_b_bb)
        assert df_output.loc[1, expected_columns[1]] == approx(df.loc[1, 'y'] / expected_mean_y_groupby_b_bb)
        assert df_output.loc[2, expected_columns[0]] == approx(df.loc[2, 'x'] / expected_mean_x_groupby_b_aa)
        assert df_output.loc[2, expected_columns[1]] == approx(df.loc[2, 'y'] / expected_mean_y_groupby_b_aa)
        assert df_output.loc[6, expected_columns[0]] == approx(df.loc[6, 'x'] / expected_mean_x_groupby_b_cc)
        assert df_output.loc[6, expected_columns[1]] == approx(df.loc[6, 'y'] / expected_mean_y_groupby_b_cc)
        assert df_output.loc[10, expected_columns[0]] == approx(df.loc[10, 'x'] / expected_mean_x_groupby_b_dd)
        assert df_output.loc[10, expected_columns[1]] == approx(df.loc[10, 'y'] / expected_mean_y_groupby_b_dd)


class TestShiftGroupByTransform:

    def test_aggregate(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        keys = ['b']
        values = ['x', 'y']

        expected_columns = [
            'lag_1_x_groupby_b',
            'lag_1_y_groupby_b',
            'lag_2_x_groupby_b',
            'lag_2_y_groupby_b',
        ]
        df_shift1 = df.groupby(keys)[values].shift(1)
        expected_x_s1 = (df_shift1['x'] - df['x']).fillna(-1)
        expected_y_s1 = (df_shift1['y'] - df['y']).fillna(-1)
        df_shift2 = df.groupby(keys)[values].shift(2)
        expected_x_s2 = (df_shift2['x'] - df['x']).fillna(-1)
        expected_y_s2 = (df_shift2['y'] - df['y']).fillna(-1)

        transform = ShiftGroupByTransform(keys, values, shift=[1, 2], fillna=-1)
        df_output = transform.aggregate(df, keys, values, shift=[1, 2], fillna=-1)

        assert df_output.columns.tolist() == keys + expected_columns
        assert (df_output[expected_columns[0]] == expected_x_s1).all()
        assert (df_output[expected_columns[1]] == expected_y_s1).all()
        assert (df_output[expected_columns[2]] == expected_x_s2).all()
        assert (df_output[expected_columns[3]] == expected_y_s2).all()
