import os
import random
import tempfile
from typing import Any, Dict, Generator

import numpy as np
import torch
from absl.testing import absltest, parameterized
from numpy import typing as npt
from torch.utils import data

from snowflake import snowpark
from snowflake.ml._internal.exceptions import fileset_errors
from snowflake.ml.fileset import fileset
from snowflake.snowpark import functions
from tests.integ.snowflake.ml.fileset import (
    fileset_integ_test_base,
    fileset_integ_utils,
)
from tests.integ.snowflake.ml.test_utils import common_test_base

np.random.seed(0)
random.seed(0)


class TestSnowflakeFileSet(fileset_integ_test_base.TestSnowflakeFileSetBase):
    """Integration tests for Snowflake FileSet."""

    table_name = "FILESET_INTEG"

    @parameterized.parameters(  # type: ignore[misc]
        {"use_snowpark": True, "fileset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"use_snowpark": True, "fileset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": False},
        {"use_snowpark": False, "fileset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"use_snowpark": False, "fileset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": False},
    )
    def test_fileset_make_and_call(
        self, use_snowpark: bool, fileset_shuffle: bool, datapipe_shuffle: bool, drop_last_batch: bool
    ) -> None:
        self._test_fileset_make_and_call(use_snowpark, fileset_shuffle, datapipe_shuffle, drop_last_batch)

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["pytorch", "torchdata"])
    @parameterized.parameters(  # type: ignore[misc]
        {"fileset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"fileset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": False},
    )
    def test_fileset_make_and_call_sproc(
        self, fileset_shuffle: bool, datapipe_shuffle: bool, drop_last_batch: bool
    ) -> None:
        self._test_fileset_make_and_call(
            True,
            fileset_shuffle,
            datapipe_shuffle,
            drop_last_batch,
            test_delete=False,
            fileset_prefix="fileset_integ_sproc",
        )

    def validate_fileset(
        self, datapipe_shuffle: bool, drop_last_batch: bool, batch_size: int, fs: fileset.FileSet
    ) -> None:
        dp = fs.to_torch_datapipe(batch_size=batch_size, shuffle=datapipe_shuffle, drop_last_batch=drop_last_batch)
        self._validate_torch_datapipe(dp, batch_size, drop_last_batch)

        df = fs.to_snowpark_dataframe()
        self._validate_snowpark_dataframe(df)

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

    def _validate_snowpark_dataframe(self, df: snowpark.DataFrame) -> None:
        for key in ["NUMBER_INT_COL", "NUMBER_FIXED_POINT_COL"]:
            self.assertAlmostEqual(
                fileset_integ_utils.get_column_min(key),
                df.select(functions.min(key)).collect()[0][0],
                1,
            )
            self.assertAlmostEqual(
                fileset_integ_utils.get_column_max(key, self.num_rows),
                df.select(functions.max(key)).collect()[0][0],
                1,
            )
            self.assertAlmostEqual(
                fileset_integ_utils.get_column_avg(key, self.num_rows),
                df.select(functions.avg(key)).collect()[0][0],
                1,
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
        fileset_integ_utils.create_snowflake_stage_if_not_exists(self.snowpark_session, stage_no_sse, False)
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
