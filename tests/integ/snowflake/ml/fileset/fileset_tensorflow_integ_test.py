import random
from typing import Any, Dict, Generator

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from numpy import typing as npt

from snowflake.ml.fileset import fileset
from tests.integ.snowflake.ml.fileset import fileset_integ_test_base
from tests.integ.snowflake.ml.test_utils import common_test_base

np.random.seed(0)
random.seed(0)


class TestSnowflakeFileSetTensorflow(fileset_integ_test_base.TestSnowflakeFileSetBase):
    """Integration tests for Snowflake FileSet."""

    table_name = "FILESET_TF_INTEG"

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

    @common_test_base.CommonTestBase.sproc_test(additional_packages=["tensorflow"])
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
            fileset_prefix="fileset_tf_integ_sproc",
        )

    def validate_fileset(
        self, datapipe_shuffle: bool, drop_last_batch: bool, batch_size: int, fs: fileset.FileSet
    ) -> None:
        ds = fs.to_tf_dataset(batch_size=batch_size, shuffle=datapipe_shuffle, drop_last_batch=drop_last_batch)
        self._validate_tf_dataset(ds, batch_size, drop_last_batch)

    def _validate_tf_dataset(self, dataset: "tf.data.Dataset", batch_size: int, drop_last_batch: bool) -> None:
        def numpy_batch_generator() -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            for batch in dataset:
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, tf.Tensor)
                    self.assertEqual(1, v.shape.rank)
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        self._validate_batches(batch_size, drop_last_batch, numpy_batch_generator)


if __name__ == "__main__":
    absltest.main()
