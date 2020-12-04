from pytest import approx
import pandas as pd

from mykaggle.transform.groupby import BaseGroupByTransform, GroupByTransform


class TestBaseGroupByTransform:
    def test_prepare_columns(self):
        keys = ['a', 'b']
        targets = ['x', 'y']
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_groupby_a_b', 'sum_x_groupby_a_b',
            'mean_y_groupby_a_b', 'sum_y_groupby_a_b'
        ]
        transform = BaseGroupByTransform(keys, targets, aggs)
        columns = transform._prepare_aggregated_columns(keys, targets, aggs)

        assert expected_columns == columns


class TestGroupByTransform:

    def test_aggregate(self):
        df = pd.read_csv('./tests/data/dummy.csv')
        keys = ['a', 'b']
        targets = ['x', 'y']
        aggs = ['mean', 'sum']

        expected_columns = [
            'mean_x_groupby_a_b', 'sum_x_groupby_a_b',
            'mean_y_groupby_a_b', 'sum_y_groupby_a_b'
        ]
        expected_mean_x_groupby_a_b_1 = (0.4 + 0.1 + 0.2) / 3
        expected_mean_x_groupby_a_b_2 = 0.3
        sum_y_groupby_a_b_1 = 0.4 + 0.1 + 0.2
        sum_y_groupby_a_b_2 = 0.3

        transform = GroupByTransform(keys, targets, aggs)
        df_output = transform.aggregate(df, keys, targets, aggs)

        assert df_output.columns.tolist() == keys + expected_columns
        assert df_output.loc[0, 'mean_x_groupby_a_b'] == approx(expected_mean_x_groupby_a_b_1)
        assert df_output.loc[1, 'mean_x_groupby_a_b'] == approx(expected_mean_x_groupby_a_b_2)
        assert df_output.loc[0, 'sum_x_groupby_a_b'] == approx(sum_y_groupby_a_b_1)
        assert df_output.loc[1, 'sum_x_groupby_a_b'] == approx(sum_y_groupby_a_b_2)
