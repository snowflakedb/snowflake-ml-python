import random
import uuid
from typing import Any, Callable, Generator

import numpy as np
from absl.testing import absltest
from numpy import typing as npt

from snowflake.ml._internal.exceptions import fileset_errors
from snowflake.ml.fileset import fileset
from tests.integ.snowflake.ml.fileset import fileset_integ_utils
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)

np.random.seed(0)
random.seed(0)

FILESET_INTEG_SCHEMA = "FILESET_INTEG_TEST"


class TestSnowflakeFileSetBase(common_test_base.CommonTestBase):
    """Integration tests for Snowflake FileSet."""

    table_name: str

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.snowpark_session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.snowpark_session)
        cls.dbm.cleanup_schemas(FILESET_INTEG_SCHEMA)  # Clean up any old schemas
        cls.sf_connection = cls.snowpark_session._conn._conn
        cls.db = cls.snowpark_session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(FILESET_INTEG_SCHEMA)
        cls.num_rows = 1500000
        cls.query = fileset_integ_utils.get_fileset_query(cls.num_rows)
        cls.stage = f"{cls.db}.{cls.schema}.{cls.table_name}_{uuid.uuid4().hex}"
        fileset_integ_utils.create_snowflake_stage_if_not_exists(cls.snowpark_session, cls.stage, temp=False)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.dbm.drop_stage(cls.stage, if_exists=True)

    def _test_fileset_make_and_call(
        self,
        use_snowpark: bool,
        fileset_shuffle: bool,
        datapipe_shuffle: bool,
        drop_last_batch: bool,
        test_delete: bool = True,
        fileset_prefix: str = "fileset_integ_make",
    ) -> None:
        """Test if fileset make() can materialize a query or a dataframe, and create a ready-to-use FileSet object."""
        if use_snowpark:
            df = self.snowpark_session.sql(self.query)
            fs_input_kwargs = {
                "snowpark_dataframe": df,
            }
        else:
            fs_input_kwargs = {
                "sf_connection": self.sf_connection,
                "query": self.query,
            }

        batch_size = 2048

        fileset_name = f"{fileset_prefix}_{fileset_shuffle}_{datapipe_shuffle}_{drop_last_batch}"
        fs = fileset.FileSet.make(
            target_stage_loc=f"@{self.stage}",
            name=fileset_name,
            shuffle=fileset_shuffle,
            **fs_input_kwargs,
        )
        files = fs.files()

        # Validate that FileSet is able to generated multiple stage files.
        # If it fails. increasing self.num_rows might help fix the issue.
        self.assertGreater(len(files), 1)

        self.assertEqual(fs.name, fileset_name)
        self.assertEqual(fs.fileset_stage_location(), f"sfc://@{self.stage}/{fileset_name}/")

        self.validate_fileset(datapipe_shuffle, drop_last_batch, batch_size, fs)

        if test_delete:
            fs.delete()
            with self.assertRaises(fileset_errors.FileSetAlreadyDeletedError):
                fs.files()

    def validate_fileset(
        self, datapipe_shuffle: bool, drop_last_batch: bool, batch_size: int, fs: fileset.FileSet
    ) -> None:
        raise NotImplementedError

    def _validate_batches(
        self,
        batch_size: int,
        drop_last_batch: bool,
        numpy_batch_generator: Callable[[], Generator[dict[str, npt.NDArray[Any]], None, None]],
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
