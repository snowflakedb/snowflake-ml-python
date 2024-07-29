import random
from typing import Any, Callable, Dict, Generator

import numpy as np
from absl.testing import absltest
from numpy import typing as npt

from snowflake.ml import dataset
from snowflake.snowpark._internal import utils as snowpark_utils
from tests.integ.snowflake.ml.fileset import fileset_integ_utils
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)

np.random.seed(0)
random.seed(0)


class TestSnowflakeDatasetBase(common_test_base.CommonTestBase):
    """Integration tests for Snowflake Dataset."""

    DS_INTEG_TEST_DB: str
    DS_INTEG_TEST_SCHEMA: str

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
        cls.num_rows = 10000
        cls.query = fileset_integ_utils.get_fileset_query(cls.num_rows)
        cls.test_table = "test_table"
        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            cls.dbm.create_database(cls.DS_INTEG_TEST_DB, if_not_exists=True)
            cls.dbm.cleanup_schemas(cls.DS_INTEG_TEST_SCHEMA, cls.DS_INTEG_TEST_DB)
            cls.dbm.use_database(cls.DS_INTEG_TEST_DB)

            cls.db = cls.session.get_current_database()
            cls.schema = cls.dbm.create_random_schema(cls.DS_INTEG_TEST_SCHEMA)
            cls.schema = f'"{cls.schema}"'  # Need quotes around schema name for regex matches later
        else:
            cls.db = cls.session.get_current_database()
            cls.schema = cls.session.get_current_schema()
        cls.session.sql(f"create table if not exists {cls.test_table} as ({cls.query})").collect()

    @classmethod
    def tearDownClass(cls) -> None:
        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            cls.dbm.drop_schema(cls.schema, if_exists=True)
            cls.session.close()
        super().tearDownClass()

    def validate_dataset_connectors(
        self, datapipe_shuffle: bool, drop_last_batch: bool, batch_size: int, ds: dataset.Dataset
    ) -> None:
        raise NotImplementedError

    def _test_dataset_connectors(
        self,
        dataset_shuffle: bool,
        datapipe_shuffle: bool,
        drop_last_batch: bool,
        batch_size: int = 2048,
        dataset_prefix: str = "dataset_integ_connector",
    ) -> None:
        """Test if dataset create() can materialize a dataframe, and create a ready-to-use Dataset object."""
        dataset_name = f"{dataset_prefix}_{dataset_shuffle}_{datapipe_shuffle}_{drop_last_batch}"
        dataset_version = "test"
        created_ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table),
            shuffle=dataset_shuffle,
        )

        for file in created_ds.read.files():
            self.assertRegex(
                file, rf"snow://dataset/{self.db}.{self.schema}.{dataset_name}/versions/{dataset_version}/.*[.]parquet"
            )

        # Verify that we can restore a Dataset object
        ds = dataset.load_dataset(
            name=dataset_name,
            version=dataset_version,
            session=self.session,
        )
        for file in ds.read.files():
            self.assertRegex(
                file, rf"snow://dataset/{self.db}.{self.schema}.{dataset_name}/versions/{dataset_version}/.*[.]parquet"
            )

        self.validate_dataset_connectors(datapipe_shuffle, drop_last_batch, batch_size, ds)

    def _validate_batches(
        self,
        batch_size: int,
        drop_last_batch: bool,
        numpy_batch_generator: Callable[[], Generator[Dict[str, npt.NDArray[Any]], None, None]],
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
        for iteration, batch in enumerate(numpy_batch_generator()):
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


if __name__ == "__main__":
    absltest.main()
