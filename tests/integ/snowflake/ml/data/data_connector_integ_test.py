import random
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from numpy import typing as npt

from snowflake import snowpark
from snowflake.ml import data, dataset
from snowflake.ml.utils import sql_client
from snowflake.snowpark._internal import utils as snowpark_utils
from tests.integ.snowflake.ml.fileset import fileset_integ_utils
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)

DC_INTEG_TEST_DB = "DC_INTEG_TEST_DB"
DC_INTEG_TEST_SCHEMA = "DC_INTEG_TEST_SCHEMA"

np.random.seed(0)
random.seed(0)


def create_data_connectors(session: snowpark.Session, create: bool, num_rows: int) -> List[data.DataConnector]:
    rst = []

    # DataFrame connector
    query = fileset_integ_utils.get_fileset_query(num_rows)
    df = session.sql(query)
    rst.append(data.DataConnector.from_dataframe(df))

    # Dataset connector
    ds_name = "test_dataset"
    ds_version = "v1"
    if create:
        ds = dataset.create_from_dataframe(session, ds_name, ds_version, input_dataframe=df)
    else:
        ds = dataset.load_dataset(session, ds_name, ds_version)

    rst.append(data.DataConnector.from_dataset(ds))

    return rst


class TestDataConnector(common_test_base.CommonTestBase):
    """Integration tests for Snowflake Dataset."""

    def setUp(self) -> None:
        # Disable base class setup/teardown in favor of classmethods
        pass

    def tearDown(self) -> None:
        # Disable base class setup/teardown in favor of classmethods
        pass

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)

        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            cls.dbm.create_database(DC_INTEG_TEST_DB, creation_mode=sql_client.CreationMode(if_not_exists=True))
            cls.dbm.cleanup_schemas(DC_INTEG_TEST_SCHEMA, DC_INTEG_TEST_DB)
            cls.dbm.use_database(DC_INTEG_TEST_DB)

            cls.db = cls.session.get_current_database()
            cls.schema = cls.dbm.create_random_schema(DC_INTEG_TEST_SCHEMA)
            cls.schema = f'"{cls.schema}"'  # Need quotes around schema name for regex matches later
        else:
            cls.db = cls.session.get_current_database()
            cls.schema = cls.session.get_current_schema()

        cls.num_rows = 10000
        cls.suts = create_data_connectors(
            cls.session, create=(not snowpark_utils.is_in_stored_procedure()), num_rows=cls.num_rows
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            cls.dbm.drop_schema(cls.schema, if_exists=True)
            cls.session.close()
        super().tearDownClass()

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["tensorflow"])
    @parameterized.parameters(  # type: ignore[misc]
        {"batch_size": 2048, "shuffle": False, "drop_last_batch": False},
    )
    def test_to_tf_dataset(self, batch_size: int, shuffle: bool, drop_last_batch: bool) -> None:
        import tensorflow as tf

        def numpy_batch_generator(ds: tf.data.Dataset) -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            for batch in ds:
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, tf.Tensor)
                    self.assertEqual(1, v.shape.rank)
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        for sut in self.suts:
            with self.subTest(type(sut.data_sources[0]).__name__):
                self._validate_batches(
                    batch_size,
                    drop_last_batch,
                    numpy_batch_generator(
                        sut.to_tf_dataset(batch_size=batch_size, shuffle=shuffle, drop_last_batch=drop_last_batch)
                    ),
                )

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["pytorch", "torchdata"])
    @parameterized.parameters(  # type: ignore[misc]
        {"batch_size": 2048, "shuffle": False, "drop_last_batch": False},
    )
    def test_to_torch_datapipe(self, batch_size: int, shuffle: bool, drop_last_batch: bool) -> None:
        import torch
        import torch.utils.data as torch_data

        def numpy_batch_generator(dp: torch_data.IterDataPipe) -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            for batch in torch_data.DataLoader(dp, batch_size=None, num_workers=0):
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(2, v.dim())
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        for sut in self.suts:
            with self.subTest(type(sut.data_sources[0]).__name__):
                self._validate_batches(
                    batch_size,
                    drop_last_batch,
                    numpy_batch_generator(
                        sut.to_torch_datapipe(batch_size=batch_size, shuffle=shuffle, drop_last_batch=drop_last_batch)
                    ),
                )

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["pytorch", "torchdata"])
    @parameterized.parameters(  # type: ignore[misc]
        {"batch_size": 2048, "shuffle": False, "drop_last_batch": False},
    )
    def test_to_torch_dataset(self, batch_size: int, shuffle: bool, drop_last_batch: bool) -> None:
        import torch
        import torch.utils.data as torch_data

        def numpy_batch_generator(ds: torch_data.Dataset) -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            for batch in torch_data.DataLoader(ds, batch_size=batch_size, drop_last=drop_last_batch, num_workers=0):
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(2, v.dim())
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        for sut in self.suts:
            with self.subTest(type(sut.data_sources[0]).__name__):
                self._validate_batches(
                    batch_size,
                    drop_last_batch,
                    numpy_batch_generator(sut.to_torch_dataset(shuffle=shuffle)),
                )

    def test_to_pandas(self) -> None:
        for sut in self.suts:
            with self.subTest(type(sut.data_sources[0]).__name__):
                self._validate_pandas(sut.to_pandas())

    def _validate_batches(
        self,
        batch_size: int,
        drop_last_batch: bool,
        numpy_batch_generator: Generator[Dict[str, npt.NDArray[Any]], None, None],
    ) -> None:
        if drop_last_batch:
            expected_num_rows = self.num_rows - self.num_rows % batch_size
        else:
            expected_num_rows = self.num_rows

        actual_min_counter = {
            "NUMBER_INT_COL": float("inf"),
            "NUMBER_FIXED_POINT_COL": float("inf"),
        }
        actual_max_counter = {
            "NUMBER_INT_COL": 0.0,
            "NUMBER_FIXED_POINT_COL": 0.0,
        }
        actual_sum_counter = {
            "NUMBER_INT_COL": 0.0,
            "NUMBER_FIXED_POINT_COL": 0.0,
        }
        actual_num_rows = 0
        for iteration, batch in enumerate(numpy_batch_generator):
            # If drop_last_batch is False, the last batch might not have the same size as the other batches.
            if not drop_last_batch and iteration == self.num_rows // batch_size:
                expected_batch_size = self.num_rows % batch_size
            else:
                expected_batch_size = batch_size

            for col_name in ["NUMBER_INT_COL", "NUMBER_FIXED_POINT_COL"]:
                col = batch[col_name]
                self.assertEqual(col.size, expected_batch_size)

                actual_min_counter[col_name] = min(np.min(col), actual_min_counter[col_name])
                actual_max_counter[col_name] = max(np.max(col), actual_max_counter[col_name])
                actual_sum_counter[col_name] += np.sum(col)

            actual_num_rows += expected_batch_size

        self.assertEqual(actual_num_rows, expected_num_rows)
        actual_avg_counter = {"NUMBER_INT_COL": 0.0, "NUMBER_FIXED_POINT_COL": 0.0}
        for key, value in actual_sum_counter.items():
            actual_avg_counter[key] = value / actual_num_rows

        if not drop_last_batch:
            # We can only get the whole set of data for comparison if drop_last_batch is False.
            for key in ["NUMBER_INT_COL", "NUMBER_FIXED_POINT_COL"]:
                self.assertAlmostEqual(fileset_integ_utils.get_column_min(key), actual_min_counter[key], 1)
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_max(key, expected_num_rows), actual_max_counter[key], 1
                )
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_avg(key, expected_num_rows), actual_avg_counter[key], 1
                )

    def _validate_pandas(self, df: pd.DataFrame) -> None:
        for key in ["NUMBER_INT_COL", "FLOAT_COL"]:
            with self.subTest(key):
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_min(key),
                    df[key].min(),
                    1,
                )
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_max(key, self.num_rows),
                    df[key].max(),
                    1,
                )
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_avg(key, self.num_rows),
                    df[key].mean(),
                    delta=1,  # FIXME: We lose noticeable precision from data casting (~0.5 error)
                )


if __name__ == "__main__":
    absltest.main()
