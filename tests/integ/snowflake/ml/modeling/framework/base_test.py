#!/usr/bin/env python3
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from absl.testing.absltest import TestCase, main
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor

from snowflake.ml._internal.exceptions.exceptions import SnowflakeMLException
from snowflake.ml.modeling.framework.base import BaseTransformer, _process_cols
from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    MinMaxScaler,
    StandardScaler,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark.exceptions import SnowparkColumnException
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    DATA_NONE_NAN,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class TestBaseFunctions(TestCase):
    """Test Base."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_set_params(self) -> None:
        class TestTransformer(BaseTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._sklearn_object: Optional[Any] = None

            def _fit(self, dataset: DataFrame) -> "TestTransformer":
                return self

        with self.subTest("Test with estimator."):
            estimator = TestTransformer()
            estimator._sklearn_object = XGBRegressor()
            estimator.set_input_cols(["COL_1", "COL_2"])
            estimator.set_label_cols("COL_3")

            estimator.set_params(**dict(max_depth=4, n_estimators=27))
            self.assertEqual(estimator.to_sklearn().max_depth, 4)
            self.assertEqual(estimator.to_sklearn().n_estimators, 27)

        with self.subTest("Test failure"):
            with self.assertRaises(SnowflakeMLException):
                estimator = TestTransformer()
                estimator._sklearn_object = XGBRegressor()
                estimator.set_params(**dict(made_up=4, n_estimators=27))

    def test_infer_input_output_cols(self) -> None:
        test_df = pd.DataFrame({"COL_1": [1, 2, 3], "COL_2": [4, 5, 6], "COL_3": [10, 11, 12]})

        class TestTransformer(BaseTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._sklearn_object: Optional[Any] = None

            def _fit(self, dataset: DataFrame) -> "TestTransformer":
                return self

        with self.subTest("Test with regressor"):
            estimator = TestTransformer()
            estimator._sklearn_object = XGBRegressor()

            estimator.set_input_cols(["COL_1", "COL_2"])
            estimator.set_label_cols("COL_3")

            estimator._infer_input_output_cols(dataset=test_df)

            self.assertEqual(estimator.output_cols, ["OUTPUT_COL_3"])

        with self.subTest("Test with unsupervised model"):
            estimator = TestTransformer()
            estimator._sklearn_object = GaussianMixture()

            estimator.set_input_cols(["COL_1", "COL_2"])
            estimator._infer_input_output_cols(dataset=test_df)

            self.assertEqual(estimator.output_cols, ["OUTPUT_0"])

        # Users must specify transformer output columns explicitly.
        with self.subTest("Test transformer raises `ValueError`"):
            with self.assertRaises(SnowflakeMLException):
                estimator = TestTransformer()
                estimator._sklearn_object = SimpleImputer()

                estimator.set_input_cols(["COL_1", "COL_2"])
                estimator._infer_input_output_cols(dataset=test_df)

    def test_fit_input_cols_check(self) -> None:
        """
        Verify the input columns check during fitting.

        Raises
        ------
        AssertionError
            If no error on input columns is raised during fitting.
        """
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        scaler = MinMaxScaler()
        with pytest.raises(RuntimeError) as excinfo:
            scaler.fit(df)
        assert "input_cols is not set." in excinfo.value.args[0]

    def test_transform_input_cols_check(self) -> None:
        """
        Verify the input columns check during transform.

        Raises
        ------
        AssertionError
            If no error on input columns is raised during transform.
        """
        input_cols, output_cols = NUMERIC_COLS, OUTPUT_COLS
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        scaler = MinMaxScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)
        scaler.set_input_cols([])
        with pytest.raises(RuntimeError) as excinfo:
            scaler.transform(df)
        assert "input_cols is not set." in excinfo.value.args[0]

    def test_transform_output_cols_check(self) -> None:
        """
        Verify the output columns check during transform.

        Raises
        ------
        AssertionError
            If no error on output columns is raised during transform.
        """
        input_cols, output_cols = NUMERIC_COLS, OUTPUT_COLS
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        # output_cols not set
        scaler = MinMaxScaler().set_input_cols(input_cols)
        scaler.fit(df)
        with pytest.raises(RuntimeError) as excinfo:
            scaler.transform(df)
        assert "output_cols is not set." in excinfo.value.args[0]

        # output_cols of mismatched sizes
        undersized_output_cols = output_cols[:-1]
        scaler = MinMaxScaler().set_input_cols(input_cols).set_output_cols(undersized_output_cols)
        scaler.fit(df)
        with pytest.raises(RuntimeError) as excinfo:
            scaler.transform(df)
        assert "Size mismatch" in excinfo.value.args[0]

        oversized_output_cols = output_cols.copy()
        oversized_output_cols.append("OUT_OVER")
        scaler = MinMaxScaler().set_input_cols(input_cols).set_output_cols(oversized_output_cols)
        scaler.fit(df)
        with pytest.raises(RuntimeError) as excinfo:
            scaler.transform(df)
        assert "Size mismatch" in excinfo.value.args[0]

    def test_to_sklearn(self) -> None:
        input_cols = ["F"]
        output_cols = ["F"]
        pandas_df_1 = pd.DataFrame(data={"F": [1, 2, 3]})
        pandas_df_2 = pd.DataFrame(data={"F": [4, 5, 6]})

        snow_df_1 = self._session.create_dataframe(pandas_df_1)
        snow_df_2 = self._session.create_dataframe(pandas_df_2)

        scaler = StandardScaler(input_cols=input_cols, output_cols=output_cols)
        scaler.fit(snow_df_1)
        sk_object_1 = scaler.to_sklearn()
        assert np.allclose([sk_object_1.mean_], [2.0])
        scaler.fit(snow_df_2)
        sk_object_2 = scaler.to_sklearn()
        assert np.allclose([sk_object_2.mean_], [5.0])

    def test_validate_data_has_no_nulls(self) -> None:
        input_cols = NUMERIC_COLS
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)
        _, df_with_nulls = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        class TestTransformer(BaseTransformer):
            pass

        # Assert that numeric data with no null data passes
        transformer = TestTransformer().set_input_cols(input_cols)  # type: ignore[abstract]
        transformer._validate_data_has_no_nulls(df)  # type: ignore[attr-defined]

        # Assert that numeric data with null data raises
        transformer = TestTransformer().set_input_cols(input_cols)  # type: ignore[abstract]
        with self.assertRaises(SnowflakeMLException):
            transformer._validate_data_has_no_nulls(df_with_nulls)  # type: ignore[attr-defined]

        # Assert that extra input columns raises
        transformer = TestTransformer().set_input_cols(input_cols + ["nonexistent_column"])  # type: ignore[abstract]
        with self.assertRaises(SnowparkColumnException):
            transformer._validate_data_has_no_nulls(df)  # type: ignore[attr-defined]

    def test_base_double_quoted_identifiers(self) -> None:
        """
        According to identifier syntax, double-quoted Identifiers should be case sensitive
        Doc: https://docs.snowflake.com/en/sql-reference/identifiers-syntax
        """
        input_cols = [
            "CARAT",
            "DEPTH",
            "TABLE_PCT",
            "PRICE",
            "X",
            "Y",
            "Z",
            '"CUT_OE_Fair"',
            '"CUT_OE_Good"',
            '"CUT_OE_Ideal"',
            '"CUT_OE_Premium"',
            '"CUT_OE_Very Good"',
            '"carat"',
        ]
        after_processed_input_cols = _process_cols(input_cols)
        self.assertEqual(input_cols, after_processed_input_cols)


if __name__ == "__main__":
    main()
