#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import os
import random
import tempfile
from typing import Any, Callable, Dict, Generator

import numpy as np
import tensorflow as tf
import torch
from absl.testing import absltest, parameterized
from numpy import typing as npt
from torch.utils import data

from snowflake import connector, snowpark
from snowflake.ml.fileset import fileset, fileset_errors
from snowflake.ml.utils import connection_params
from tests.integ.snowflake.ml.fileset import fileset_integ_utils

np.random.seed(0)
random.seed(0)


class TestSnowflakeFileSet(parameterized.TestCase):
    """Integration tests for Snowflake FileSet."""

    connection_parameters = connection_params.SnowflakeLoginOptions()
    sf_connection = connector.connect(**connection_parameters)
    snowpark_session = snowpark.Session.builder.config("connection", sf_connection).create()
    db = snowpark_session.get_current_database()
    schema = snowpark_session.get_current_schema()
    table_name = "FILESET_INTEG"
    stage = f"{db}.{schema}.fileset_integ"
    num_rows = 1500000
    query = fileset_integ_utils.get_fileset_query(num_rows)

    @classmethod
    def setUpClass(cls) -> None:
        fileset_integ_utils.create_tmp_snowflake_stage_if_not_exists(cls.snowpark_session, cls.stage)

    @parameterized.parameters(  # type: ignore[misc]
        {"use_snowpark": True, "fileset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"use_snowpark": True, "fileset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": False},
        {"use_snowpark": False, "fileset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"use_snowpark": False, "fileset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": False},
    )
    def test_fileset_make_and_call(
        self, use_snowpark: bool, fileset_shuffle: bool, datapipe_shuffle: bool, drop_last_batch: bool
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

        fileset_name = f"fileset_integ_make_{fileset_shuffle}_{datapipe_shuffle}_{drop_last_batch}"
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

        dp = fs.to_torch_datapipe(batch_size=batch_size, shuffle=datapipe_shuffle, drop_last_batch=drop_last_batch)
        self._validate_torch_datapipe(dp, batch_size, drop_last_batch)

        ds = fs.to_tf_dataset(batch_size=batch_size, shuffle=datapipe_shuffle, drop_last_batch=drop_last_batch)
        self._validate_tf_dataset(ds, batch_size, drop_last_batch)

        fs.delete()
        with self.assertRaises(fileset_errors.FileSetAlreadyDeletedError):
            fs.files()

    def _validate_torch_datapipe(
        self, datapipe: data.IterDataPipe[Dict[str, npt.NDArray[Any]]], batch_size: int, drop_last_batch: bool
    ) -> None:
        def numpy_batch_generator() -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            for batch in data.DataLoader(datapipe, batch_size=None, num_workers=0):
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(1, v.dim())
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        self._validate_batches(batch_size, drop_last_batch, numpy_batch_generator)

    def _validate_tf_dataset(self, dataset: tf.data.Dataset, batch_size: int, drop_last_batch: bool) -> None:
        def numpy_batch_generator() -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            for batch in dataset:
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, tf.Tensor)
                    self.assertEqual(1, v.shape.rank)
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        self._validate_batches(batch_size, drop_last_batch, numpy_batch_generator)

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
            # We can only get the whole set of data for comparision if drop_last_batch is False.
            for key in ["NUMBER_INT_COL", "NUMBER_FIXED_POINT_COL"]:
                self.assertAlmostEqual(fileset_integ_utils.get_column_min(key), actual_min_counter[key], 1)
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_max(key, expected_num_rows), actual_max_counter[key], 1
                )
                self.assertAlmostEqual(
                    fileset_integ_utils.get_column_avg(key, expected_num_rows), actual_avg_counter[key], 1
                )

    def test_restore_fileset(self) -> None:
        """Test if a FileSet can be restored if it's not deleted before."""
        fileset_name = "fileset_integ_restore"
        fs = fileset.FileSet.make(
            target_stage_loc=f"@{self.stage}",
            name=fileset_name,
            shuffle=True,
            sf_connection=self.sf_connection,
            query=self.query,
        )

        # Verify that duplicate fileset is not allowed to be created.
        with self.assertRaises(fileset_errors.FileSetExistError):
            fileset.FileSet.make(
                target_stage_loc=f"@{self.stage}",
                name=fileset_name,
                shuffle=True,
                sf_connection=self.sf_connection,
                query=self.query,
            )

        # Verify that we can restore a FileSet object
        fs2 = fileset.FileSet(
            target_stage_loc=f"@{self.stage}",
            name=fileset_name,
            sf_connection=self.sf_connection,
        )
        self.assertEqual(fs.files(), fs2.files())

        # Verify that the FileSet restore could fail if the target stage file are from more than one query.
        with self.assertRaises(fileset_errors.MoreThanOneQuerySourceError):
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, "f")
                with open(local_path, "w") as f:
                    f.write("nothing")
                fileset_integ_utils.upload_file_to_snowflake(
                    self.snowpark_session, local_path, self.stage, fileset_name
                )
            fileset.FileSet(
                target_stage_loc=f"@{self.stage}",
                name=fileset_name,
                sf_connection=self.sf_connection,
            )

        fs.delete()

    def test_stage_encryption_validation(self) -> None:
        """Validated that FileSet will raise an error if the target stage is not server-side encrypted."""
        stage_no_sse = f"{self.db}.{self.schema}.no_sse"
        fileset_integ_utils.create_tmp_snowflake_stage_if_not_exists(self.snowpark_session, stage_no_sse, False)
        with self.assertRaises(fileset_errors.FileSetLocationError):
            fileset.FileSet.make(
                target_stage_loc=f"@{stage_no_sse}",
                name="stage_encryption",
                shuffle=True,
                sf_connection=self.sf_connection,
                query=self.query,
            )


if __name__ == "__main__":
    absltest.main()
