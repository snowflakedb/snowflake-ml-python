import logging
import os
import random
import unittest
from typing import Any, Dict, Generator
from uuid import uuid4

import cloudpickle as cp
import numpy as np
import pandas as pd
import torch
from absl.testing import absltest, parameterized
from numpy import typing as npt
from torch.utils import data

from snowflake import snowpark
from snowflake.ml import dataset
from snowflake.ml._internal.exceptions import dataset_errors
from snowflake.snowpark import functions
from tests.integ.snowflake.ml.dataset import dataset_integ_test_base
from tests.integ.snowflake.ml.fileset import fileset_integ_utils
from tests.integ.snowflake.ml.test_utils import common_test_base

np.random.seed(0)
random.seed(0)


class TestSnowflakeDataset(dataset_integ_test_base.TestSnowflakeDatasetBase):
    """Integration tests for Snowflake Dataset."""

    DS_INTEG_TEST_DB = "SNOWML_DATASET_TEST_DB"
    DS_INTEG_TEST_SCHEMA = "DATASET_TEST"

    @common_test_base.CommonTestBase.sproc_test(local=True, additional_packages=["pytorch"])
    def test_dataset_management(self) -> None:
        """Test Dataset management APIs"""
        dataset_name = f"dataset_integ_management_{uuid4().hex}"
        ds = dataset.Dataset.create(self.session, dataset_name)
        assert isinstance(ds, dataset.Dataset)  # Use plain assert so type inferencing works
        self.assertEmpty(ds.list_versions())

        with self.assertRaises(dataset_errors.DatasetExistError):
            dataset.Dataset.create(self.session, dataset_name)

        with self.assertRaises(dataset_errors.DatasetNotExistError):
            dataset.Dataset.load(self.session, "dataset_not_exist")

        loaded_ds = dataset.Dataset.load(self.session, dataset_name)
        self.assertEmpty(loaded_ds.list_versions())

        # Create version. Should be reflected in both
        dataset_version1 = "v1"
        ds.create_version(
            version=dataset_version1,
            input_dataframe=self.session.table(self.test_table).limit(1000),
        )
        self.assertListEqual([dataset_version1], ds.list_versions())
        self.assertListEqual([dataset_version1], loaded_ds.list_versions())

        # FIXME: Add DatasetVersionExistError
        with self.assertRaises(dataset_errors.DatasetExistError):
            ds.create_version(
                version=dataset_version1,
                input_dataframe=self.session.table(self.test_table).limit(1000),
            )

        # Validate with two versions (including version list ordering)
        dataset_version2 = "v2"
        ds.create_version(
            version=dataset_version2,
            input_dataframe=self.session.table(self.test_table).limit(1000),
        )
        self.assertListEqual([dataset_version1, dataset_version2], ds.list_versions())
        self.assertListEqual([dataset_version1, dataset_version2], loaded_ds.list_versions())

    # Don't run in sprocs to speed up tests
    def test_dataset_case_sensitivity(self) -> None:
        dataset_name = "dataset_integ_case_sensitive"
        dataset_version = "v1"
        ds = dataset.create_from_dataframe(
            self.session,
            f'"{dataset_name}"',
            dataset_version,
            self.session.table(self.test_table).limit(1000),
            comment="lowercase",
        )
        self.assertRegex(ds.selected_version.url(), rf"snow://dataset/[\"\w.]*\"{dataset_name}\"/versions/v1")

        # Test dataset name case sensitivity
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            dataset.Dataset.load(self.session, dataset_name)
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            dataset.Dataset.load(self.session, dataset_name)
        loaded_ds = dataset.load_dataset(self.session, f'"{dataset_name}"', dataset_version)

        self.assertEqual([dataset_version], loaded_ds.list_versions())
        self.assertEqual(ds.selected_version.url(), loaded_ds.selected_version.url())
        self.assertEqual("lowercase", loaded_ds.selected_version.comment)

        # Test version case sensitivity
        _ = ds.select_version("v1")
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            _ = ds.select_version("V1")

        uppercase_ds = ds.create_version("V1", self.session.table(self.test_table).limit(100), comment="uppercase")
        self.assertRegex(uppercase_ds.selected_version.url(), rf"snow://dataset/[\"\w.]*\"{dataset_name}\"/versions/V1")
        self.assertEqual("uppercase", uppercase_ds.selected_version.comment)

        # Make sure versions are kept distinct
        loaded_uppercase_ds = ds.select_version("V1")
        self.assertEqual("lowercase", loaded_ds.selected_version.comment)
        self.assertEqual("uppercase", loaded_uppercase_ds.selected_version.comment)
        self.assertEqual(1000, loaded_ds.read.to_snowpark_dataframe().count())
        self.assertEqual(100, loaded_uppercase_ds.read.to_snowpark_dataframe().count())

    def test_dataset_properties(self) -> None:
        """Test Dataset version property loading"""
        from datetime import datetime, timezone

        current_time = datetime.now(timezone.utc)
        dataset_name = "dataset_integ_metadata"
        dataset_version = "v1"
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.sql("SELECT 1"),
            exclude_cols=["timestamp"],
            comment="this is my dataset 'with quotes'",
        )

        self.assertListEqual(ds.read.data_sources[0].exclude_cols, ["timestamp"])
        self.assertEqual(ds.selected_version.comment, "this is my dataset 'with quotes'")
        self.assertGreaterEqual(ds.selected_version.created_on, current_time)

        ds1 = ds.create_version("no_comment", self.session.sql("SELECT 1"))
        self.assertEmpty(ds1.read.data_sources[0].exclude_cols)
        self.assertIsNone(ds1.selected_version.comment)

        ds_pickle = cp.dumps(ds)
        ds_unpickle = cp.loads(ds_pickle)
        self.assertListEqual(ds_unpickle.read.data_sources[0].exclude_cols, ["timestamp"])
        self.assertEqual(ds_unpickle.selected_version.comment, "this is my dataset 'with quotes'")
        self.assertEqual(1, len(ds_unpickle.read.to_pandas()))

    # Don't run in sprocs to speed up tests
    def test_dataset_partition_by(self) -> None:
        """Test Dataset creation from Snowpark DataFrame"""
        dataset_name = f"{self.db}.{self.schema}.dataset_integ_partition"
        ds1_version = "constant_partition"
        ds1 = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=ds1_version,
            input_dataframe=self.session.table(self.test_table).limit(1000),
            partition_by="'subdir'",
        )
        ds1_dirs = {os.path.dirname(f) for f in ds1.read.files()}
        self.assertListEqual([f"snow://dataset/{dataset_name}/versions/{ds1_version}/subdir"], sorted(ds1_dirs))

        ds2_version = "range_partition"
        ds2 = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=ds2_version,
            input_dataframe=self.session.sql(
                "select seq4() as ID, uniform(1, 4, random(42)) as part from table(generator(rowcount => 10000))"
            ),
            partition_by="to_varchar(PART)",
        )
        ds2_dirs = {os.path.dirname(f) for f in ds2.read.files()}
        self.assertListEqual(
            [
                f"snow://dataset/{dataset_name}/versions/{ds2_version}/1",
                f"snow://dataset/{dataset_name}/versions/{ds2_version}/2",
                f"snow://dataset/{dataset_name}/versions/{ds2_version}/3",
                f"snow://dataset/{dataset_name}/versions/{ds2_version}/4",
            ],
            sorted(ds2_dirs),
        )

    # Don't run in sprocs to speed up tests
    def test_create_from_dataframe(self) -> None:
        """Test Dataset creation from Snowpark DataFrame"""
        dataset_name = "dataset_integ_create_from_dataframe"
        dataset_version = "v1"
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table).limit(1000),
        )

        df = ds.read.to_snowpark_dataframe()
        df_count = df.count()

        # Verify that duplicate dataset is not allowed to be created.
        with self.assertRaises(dataset_errors.DatasetExistError):
            dataset.create_from_dataframe(
                session=self.session,
                name=dataset_name,
                version=dataset_version,
                input_dataframe=self.session.table(self.test_table).limit(1000),
            )

        # Verify that creating a different Dataset version works
        dataset_version2 = "v2"
        dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version2,
            input_dataframe=self.session.table(self.test_table).limit(1000),
        )

        # Ensure v1 contents unaffected by v2
        self.assertEqual(df_count, ds.read.to_snowpark_dataframe().count())

    # Don't run in sprocs due to quirky schema handling in sproc (can't use USE SCHEMA but CREATE SCHEMA changes schema)
    def test_create_from_dataframe_fqn(self) -> None:
        """Test Dataset creation with fully qualified name"""
        schema = self.dbm.create_random_schema(self.DS_INTEG_TEST_SCHEMA)
        self.session.use_schema(self.schema)  # Keep session on main test schema
        try:
            dataset_name = f"{self.db}.{schema}.dataset_integ_create_from_dataframe_fqn"
            dataset_version = "v1"
            ds = dataset.create_from_dataframe(
                session=self.session,
                name=dataset_name,
                version=dataset_version,
                input_dataframe=self.session.table(self.test_table).limit(1000),
            )

            self.assertGreater(len(ds.read.files()), 0)
            for file in ds.read.files():
                self.assertStartsWith(file, f"snow://dataset/{dataset_name}/versions/{dataset_version}/")
        finally:
            self.session.sql(f"drop schema {self.db}.{schema}").collect()

    def test_create_from_dataframe_timestamp(self) -> None:
        """Test Dataset creation from a time-series DataFrame. Should not print spurious warnings."""
        num_rows = 100
        df = self.session.sql(
            "select dateadd(day, seq4(), current_date()) as timestamp"
            ", uniform(1, 100, random()) as data_col"
            f" from table(generator(rowcount => {num_rows}))"
        )
        dataset_name = "test_create_from_dataframe_timestamp"
        dataset_version = "v1"
        with self.assertLogs(level=logging.WARNING) as logs:
            # Necessary to make assertLogs work
            # Starting in Python 3.10 there is a new assertNoLogs method which can be used instead
            logging.warning("Dummy warning")

            ds = dataset.create_from_dataframe(
                session=self.session,
                name=dataset_name,
                version=dataset_version,
                input_dataframe=df,
                exclude_cols=["timestamp"],
            )

            # No warnings besides our dummy warning should have been generated
            self.assertEqual(1, len(logs.output))

        self.assertEqual(num_rows, len(ds.read.to_pandas()))

    @unittest.skip("Fails due to server side issue. Need to be investigated SNOW-1862761")
    @common_test_base.CommonTestBase.sproc_test(local=True, additional_packages=["pytorch"])
    def test_dataset_from_dataset(self) -> None:
        # Generate random prefixes due to race condition between sprocs causing dataset collision
        dataset_name = f"dataset_integ_dataset_from_dataset_{uuid4().hex}"
        dataset_version = "v1"
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table),
        )

        ds_df = ds.read.to_snowpark_dataframe()
        dataset_version2 = "v2"
        ds2 = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version2,
            input_dataframe=ds_df,
        )

        self._validate_snowpark_dataframe(ds2.read.to_snowpark_dataframe())

    # Don't run in sprocs since sprocs don't have delete privilege
    def test_dataset_delete(self) -> None:
        """Test dataset deletion"""
        dataset_name = "dataset_integ_delete"
        dataset_version = "test"
        ds = dataset.create_from_dataframe(
            self.session,
            dataset_name,
            dataset_version,
            self.session.table(self.test_table).limit(1000),
        )
        dsv = ds.selected_version

        self.assertNotEmpty(ds.list_versions())
        self.assertNotEmpty(dsv.list_files())
        ds.delete_version(dataset_version)
        self.assertEmpty(ds.list_versions())

        # Delete dataset. Loaded Dataset should also be deleted
        loaded_ds = dataset.Dataset.load(self.session, dataset_name)
        ds.delete()
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            ds.list_versions()
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            loaded_ds.list_versions()

        # create_version should fail for deleted/nonexistent datasets
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            ds.create_version(
                version="new_version",
                input_dataframe=self.session.table(self.test_table).limit(1000),
            )

        # create_version should fail for deleted/nonexistent datasets
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            dataset.Dataset.load(self.session, dataset_name)

    # Don't run in sprocs to speed up tests
    def test_restore_nonexistent_dataset(self) -> None:
        """Test load of non-existent dataset"""
        # Dataset not exist
        dataset_name = "dataset_integ_notexist"
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            dataset.load_dataset(
                name=dataset_name,
                version="test",
                session=self.session,
            )

        # Version not exist
        dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version="test",
            input_dataframe=self.session.sql("select 1"),
        )
        with self.assertRaises(dataset_errors.DatasetNotExistError):
            dataset.load_dataset(
                name=dataset_name,
                version="not_exist",
                session=self.session,
            )

    @parameterized.parameters(
        '"name/slash"',
        '"lots/of///slashes"',
        '"trailing_slash/"',
        '"/leading_slash"',
        '"😃"',
        '"😃_with_text"',
        '"spaces   in name"',
        "versions",
        '"versions/v1"',
        '"my_ds/versions/"',
        '"i have\n\t\rspecial characters"',
        '"-=Σ(( つ•̀ω•́)つ"',
        ("versions", "versions"),
    )
    def test_dataset_names(self, dataset_name: str, dataset_version: str = "v1") -> None:
        """Test datasets with challenging names"""
        row_count = 10
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table).limit(row_count),
        )

        self.assertEqual(row_count, len(ds.read.to_pandas()))

    @common_test_base.CommonTestBase.sproc_test(local=True, additional_packages=["pytorch"])
    def test_file_access(self) -> None:
        import pyarrow.parquet as pq

        dataset_name = f"dataset_integ_file_access_{uuid4().hex}"
        dataset_version = "v1"
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table),
        )

        pq_ds = pq.ParquetDataset(ds.read.files(), filesystem=ds.read.filesystem())
        pq_table = pq_ds.read()
        self.assertEqual(self.num_rows, len(pq_table))
        self._validate_pandas(pq_table.to_pandas())

    @common_test_base.CommonTestBase.sproc_test(local=True, additional_packages=["pytorch"])
    def test_to_pandas(self) -> None:
        dataset_name = f"dataset_integ_pandas_{uuid4().hex}"
        dataset_version = "v1"
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table),
        )

        pd_df = ds.read.to_pandas()
        self._validate_pandas(pd_df)

        # FIXME: This currently fails due to float64 -> float32 cast during Dataset creation
        # Additionally may need to sort the Pandas DataFrame to align with Snowpark DataFrame
        # df = ds.to_snowpark_dataframe()
        # pd.testing.assert_frame_equal(df.to_pandas(), pd_df, check_index_type=False)

    @common_test_base.CommonTestBase.sproc_test(local=True, additional_packages=["pytorch"])
    def test_to_dataframe(self) -> None:
        all_columns = [col for col, _ in fileset_integ_utils._TEST_RESULTSET_SCHEMA]
        exclude_cols = all_columns[:2]
        label_cols = all_columns[1:3]  # Intentionally overlap with exclude_cols (unintended but likely common behavior)

        # Generate random prefixes due to race condition between sprocs causing dataset collision
        dataset_name = f"dataset_integ_to_dataframe_{uuid4().hex}"
        dataset_version = "test"
        ds = dataset.create_from_dataframe(
            session=self.session,
            name=dataset_name,
            version=dataset_version,
            input_dataframe=self.session.table(self.test_table),
            exclude_cols=exclude_cols,
            label_cols=label_cols,
        )

        df = ds.read.to_snowpark_dataframe()
        self._validate_snowpark_dataframe(df)
        self.assertSameElements(all_columns, df.columns)

        features_df = ds.read.to_snowpark_dataframe(only_feature_cols=True)
        non_feature_cols = set(exclude_cols + label_cols)
        feature_cols = [col for col in all_columns if col not in non_feature_cols]
        self.assertSameElements(feature_cols, features_df.columns)

    @parameterized.parameters(  # type: ignore[misc]
        {"dataset_shuffle": True, "datapipe_shuffle": False, "drop_last_batch": False},
        {"dataset_shuffle": False, "datapipe_shuffle": True, "drop_last_batch": False},
        {"dataset_shuffle": False, "datapipe_shuffle": False, "drop_last_batch": True},
        {"dataset_shuffle": True, "datapipe_shuffle": True, "drop_last_batch": True},
    )
    def test_dataset_connectors(self, dataset_shuffle: bool, datapipe_shuffle: bool, drop_last_batch: bool) -> None:
        self._test_dataset_connectors(dataset_shuffle, datapipe_shuffle, drop_last_batch)

    @common_test_base.CommonTestBase.sproc_test(local=False, additional_packages=["pytorch", "torchdata"])
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
        pt_dp = ds.read.to_torch_datapipe(
            batch_size=batch_size, shuffle=datapipe_shuffle, drop_last_batch=drop_last_batch
        )
        self._validate_torch_datapipe(pt_dp, batch_size, drop_last_batch)

        pt_ds = ds.read.to_torch_dataset(shuffle=datapipe_shuffle)
        self._validate_torch_dataset(pt_ds, batch_size, drop_last_batch)

        df = ds.read.to_snowpark_dataframe()
        self._validate_snowpark_dataframe(df)

    def _validate_torch_dataset(
        self, ds: "data.IterableDataset[Dict[str, Any]]", batch_size: int, drop_last_batch: bool
    ) -> None:
        def numpy_batch_generator() -> Generator[dict[str, npt.NDArray[Any]], None, None]:
            for batch in data.DataLoader(ds, batch_size=batch_size, drop_last=drop_last_batch, num_workers=0):
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(2, v.dim())
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        self._validate_batches(batch_size, drop_last_batch, numpy_batch_generator)

    def _validate_torch_datapipe(
        self, datapipe: "data.IterDataPipe[Dict[str, npt.NDArray[Any]]]", batch_size: int, drop_last_batch: bool
    ) -> None:
        def numpy_batch_generator() -> Generator[dict[str, npt.NDArray[Any]], None, None]:
            for batch in data.DataLoader(datapipe, batch_size=None, num_workers=0):
                numpy_batch = {}
                for k, v in batch.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(2, v.dim())
                    numpy_batch[k] = v.numpy()
                yield numpy_batch

        self._validate_batches(batch_size, drop_last_batch, numpy_batch_generator)

    def _validate_snowpark_dataframe(self, df: snowpark.DataFrame) -> None:
        for key in ["NUMBER_INT_COL", "NUMBER_FIXED_POINT_COL"]:
            self.assertAlmostEqual(
                fileset_integ_utils.get_column_min(key),
                float(df.select(functions.min(key)).collect()[0][0]),
                1,
            )
            self.assertAlmostEqual(
                fileset_integ_utils.get_column_max(key, self.num_rows),
                float(df.select(functions.max(key)).collect()[0][0]),
                1,
            )
            self.assertAlmostEqual(
                fileset_integ_utils.get_column_avg(key, self.num_rows),
                float(df.select(functions.avg(key)).collect()[0][0]),
                1,
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
