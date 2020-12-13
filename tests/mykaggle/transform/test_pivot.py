from pytest import approx
import pandas as pd
import numpy as np

from mykaggle.transform.pivot import PivotTransform


class TestPivotTransform:
    def test_prepare_columns(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        indices = ['a']
        target = 'x'
        column = 'b'
        column_values = df[column].unique().tolist()
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_pivot_by_a_for_b_aa', 'mean_x_pivot_by_a_for_b_bb',
            'sum_x_pivot_by_a_for_b_aa', 'sum_x_pivot_by_a_for_b_bb',
        ]
        transform = PivotTransform(indices, target, column, aggs)
        columns = transform._prepare_columns(indices, column, column_values, target, aggs)

        assert expected_columns == columns

    def test_prepare_columns_with_multi_indices(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        indices = ['a', 'c']
        target = 'x'
        column = 'b'
        column_values = df[column].unique().tolist()
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_pivot_by_a_c_for_b_aa', 'mean_x_pivot_by_a_c_for_b_bb',
            'sum_x_pivot_by_a_c_for_b_aa', 'sum_x_pivot_by_a_c_for_b_bb',
        ]
        transform = PivotTransform(indices, target, column, aggs)
        columns = transform._prepare_columns(indices, column, column_values, target, aggs)

        assert expected_columns == columns

    def test_pivot(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        indices = ['a']
        target = 'x'
        column = 'b'
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_pivot_by_a_for_b_aa', 'mean_x_pivot_by_a_for_b_bb',
            'sum_x_pivot_by_a_for_b_aa', 'sum_x_pivot_by_a_for_b_bb',
        ]
        expected_mean_x_pivot_by_a_for_b_aa = [(0.1 + 0.2 + 0.4) / 3, 0.4]
        expected_sum_x_pivot_by_a_for_b_aa = [0.1 + 0.2 + 0.4, 0.4]
        expected_mean_x_pivot_by_a_for_b_bb = [0.3, 0.6]
        expected_sum_x_pivot_by_a_for_b_bb = [0.3, 0.6]

        expected_values = [
            expected_mean_x_pivot_by_a_for_b_aa, expected_mean_x_pivot_by_a_for_b_bb,
            expected_sum_x_pivot_by_a_for_b_aa, expected_sum_x_pivot_by_a_for_b_bb
        ]

        transform = PivotTransform(indices, column, target, aggs)
        df_output = transform.pivot(df, indices, column, target, aggs)

        assert df_output.columns.tolist() == indices + expected_columns
        for c, v in zip(expected_columns, expected_values):
            print(c, v)
            assert df_output.loc[0, c] == approx(v[0])
            assert df_output.loc[1, c] == approx(v[1])

    def test_pivot_multi_indices(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        indices = ['a', 'c']
        target = 'x'
        column = 'b'
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_pivot_by_a_c_for_b_aa', 'mean_x_pivot_by_a_c_for_b_bb',
            'sum_x_pivot_by_a_c_for_b_aa', 'sum_x_pivot_by_a_c_for_b_bb',
        ]
        expected_mean_x_pivot_by_a_c_for_b_aa = [0.4, 0.2, 0.1, np.nan, 0.4]
        expected_mean_x_pivot_by_a_c_for_b_bb = [np.nan, 0.3, np.nan, 0.6, np.nan]
        expected_sum_x_pivot_by_a_c_for_b_aa = [0.4, 0.2, 0.1, np.nan, 0.4]
        expected_sum_x_pivot_by_a_c_for_b_bb = [np.nan, 0.3, np.nan, 0.6, np.nan]

        expected_values = [
            expected_mean_x_pivot_by_a_c_for_b_aa, expected_mean_x_pivot_by_a_c_for_b_bb,
            expected_sum_x_pivot_by_a_c_for_b_aa, expected_sum_x_pivot_by_a_c_for_b_bb
        ]

        transform = PivotTransform(indices, column, target, aggs)
        df_output = transform.pivot(df, indices, column, target, aggs)

        assert df_output.columns.tolist() == indices + expected_columns
        for c, v in zip(expected_columns, expected_values):
            print(c, v)
            for i in range(5):
                if np.isnan(v[i]):
                    assert np.isnan(df_output.loc[i, c])
                else:
                    assert df_output.loc[i, c] == approx(v[i])
