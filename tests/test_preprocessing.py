import unittest
import polars as pl
from src.utils import read_all_csvs
from src.preprocessing import NetFlowBinner


class TestPreprocessing(unittest.TestCase):

    def test_netflow_binner(self):

        # Arrange
        expected_result_length = 1_623_132
        bucket_interval = "1h"
        df = read_all_csvs('./data')

        binner = NetFlowBinner(
            df,
            bucket_interval
        )

        # Act
        result = binner.process()
        total_outbound_events = result.filter(pl.col(binner.id_col_name) == 6004.07)[binner.departure_col_name].sum()


        # Assert
        self.assertEqual(expected_result_length, result.height)
        self.assertEqual(2333, total_outbound_events)