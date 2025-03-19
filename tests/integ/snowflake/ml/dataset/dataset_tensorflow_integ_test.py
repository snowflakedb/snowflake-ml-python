import random
from typing import Any, Dict, Generator
from uuid import uuid4

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from numpy import typing as npt

from snowflake.ml import dataset
from tests.integ.snowflake.ml.dataset import dataset_integ_test_base
from tests.integ.snowflake.ml.test_utils import common_test_base

np.random.seed(0)
random.seed(0)


class TestSnowflakeDataseTensorflow(dataset_integ_test_base.TestSnowflakeDatasetBase):
    """Integration tests for Snowflake Dataset."""

    DS_INTEG_TEST_DB = "SNOWML_DATASET_TF_TEST_DB"
    DS_INTEG_TEST_SCHEMA = "DATASET_TF_TEST"

    @parameterized.parameters(  # type: ignore[misc]
        {"dataset_shuffle": True, "datapipe_shuffle": False, "drop_last_batch": False},
        {"dataset_shuffle": False, "datapipe_shuffle": True, "drop_last_batch": False},
        {"dataset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"dataset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": True},
    )
    def test_dataset_connectors(self, dataset_shuffle: bool, datapipe_shuffle: bool, drop_last_batch: bool) -> None:
        self._test_dataset_connectors(dataset_shuffle, datapipe_shuffle, drop_last_batch)

    @common_test_base.CommonTestBase.sproc_test(local=False, additional_packages=["tensorflow"])
    @parameterized.parameters(  # type: ignore[misc]
        {"dataset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": True},
    )
    def test_dataset_connectors_sproc(
        self, dataset_shuffle: bool, datapipe_shuffle: bool, drop_last_batch: bool
    ) -> None:
        # Generate random prefixes due to race condition between sprocs causing dataset collision
        self._test_dataset_connectors(
            dataset_shuffle, datapipe_shuffle, drop_last_batch, dataset_prefix=f"dataset_integ_sproc_{uuid4().hex}"
        )

    def validate_dataset_connectors(
        self, datapipe_shuffle: bool, drop_last_batch: bool, batch_size: int, ds: dataset.Dataset
    ) -> None:
        tf_ds = ds.read.to_tf_dataset(batch_size=batch_size, shuffle=datapipe_shuffle, drop_last_batch=drop_last_batch)
        self._validate_tf_dataset(tf_ds, batch_size, drop_last_batch)

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
