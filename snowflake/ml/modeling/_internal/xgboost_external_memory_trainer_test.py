import math

import inflection
import pandas as pd
from absl.testing import absltest
from sklearn.datasets import load_iris

from snowflake.ml._internal.utils.temp_file_utils import (
    cleanup_temp_files,
    get_temp_file_path,
)
from snowflake.ml.modeling._internal.xgboost_external_memory_trainer import (
    get_data_iterator,
)


class XGBoostExternalMemoryTrainerTest(absltest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def get_dataset(self) -> pd.DataFrame:
        input_df_pandas = load_iris(as_frame=True).frame
        input_df_pandas.columns = [inflection.parameterize(c, "_").upper() for c in input_df_pandas.columns]
        input_cols = [c for c in input_df_pandas.columns if not c.startswith("TARGET")]
        label_col = [c for c in input_df_pandas.columns if c.startswith("TARGET")]
        return (input_df_pandas, input_cols, label_col)

    def test_data_iterator_single_file(self) -> None:
        df, input_cols, label_col = self.get_dataset()

        num_rows_in_original_dataset = df.shape[0]
        batch_size = 20

        temp_file = get_temp_file_path()
        df.to_parquet(temp_file)

        it = get_data_iterator(
            file_paths=[temp_file],
            batch_size=20,
            input_cols=input_cols,
            label_cols=label_col,
        )

        num_rows = 0
        num_batches = 0

        def consumer_func(data: pd.DataFrame, label: pd.DataFrame) -> None:
            nonlocal num_rows
            nonlocal num_batches
            num_rows += data.shape[0]
            num_batches += 1

        while it.next(consumer_func):
            pass

        self.assertEqual(num_rows, num_rows_in_original_dataset)
        self.assertEqual(num_batches, math.ceil(float(num_rows_in_original_dataset) / float(batch_size)))
        cleanup_temp_files(temp_file)

    def test_data_iterator_multiple_file(self) -> None:
        df, input_cols, label_col = self.get_dataset()

        num_rows_in_original_dataset = df.shape[0]
        batch_size = 20

        temp_file1 = get_temp_file_path()
        temp_file2 = get_temp_file_path()
        df1, df2 = df.iloc[:70], df.iloc[70:]
        df1.to_parquet(temp_file1)
        df2.to_parquet(temp_file2)

        it = get_data_iterator(
            file_paths=[temp_file1, temp_file2],
            batch_size=20,
            input_cols=input_cols,
            label_cols=label_col,
        )

        num_rows = 0
        num_batches = 0

        def consumer_func(data: pd.DataFrame, label: pd.DataFrame) -> None:
            nonlocal num_rows
            nonlocal num_batches
            num_rows += data.shape[0]
            num_batches += 1

        while it.next(consumer_func):
            pass

        self.assertEqual(num_rows, num_rows_in_original_dataset)
        self.assertEqual(num_batches, math.ceil(float(num_rows_in_original_dataset) / float(batch_size)))
        cleanup_temp_files([temp_file1, temp_file2])


if __name__ == "__main__":
    absltest.main()
