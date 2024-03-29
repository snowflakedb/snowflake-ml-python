import importlib
import os
import pickle
import sys
import tempfile
from typing import List
from unittest import TestCase

import cloudpickle
import joblib
import numpy as np
from absl.testing.absltest import main
from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from snowflake.ml.modeling.impute import SimpleImputer  # type: ignore[attr-defined]
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    CATEGORICAL_COLS,
    DATA,
    DATA_ALL_NONE,
    DATA_NAN,
    DATA_NONE_NAN,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class SimpleImputerTest(TestCase):
    """Test SimpleImputer."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_inconsistent_input_col_type(self) -> None:
        """
        Verify failure scenario of inconsistent input types.

        Raises
        ------
        AssertionError
            If the input columns do not have same type.
        """
        input_cols = ["FLOAT1", "STR1"]
        output_cols = input_cols
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        for strategy in ["mean", "constant", "median"]:
            simple_imputer = SimpleImputer(strategy=strategy, input_cols=input_cols, output_cols=output_cols)
            with self.assertRaisesRegex(TypeError, "Inconsistent input column types."):
                simple_imputer.fit(df)

    def test_fit(self) -> None:
        """
        Verify fitted categories.

        Raises
        ------
        AssertionError
            If the fit result differs from the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        simple_imputer_sklearn = SklearnSimpleImputer()
        simple_imputer_sklearn.fit(df_pandas[input_cols])

        statistics_numpy = np.array(list(simple_imputer.statistics_.values()))

        np.testing.assert_equal(statistics_numpy, simple_imputer_sklearn.statistics_)

    def test_fit_constant(self) -> None:
        """
        Verify constant fit statistics.

        Raises
        ------
        AssertionError
            If the fit result differs from the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        fill_value = 2
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(
            input_cols=input_cols, output_cols=output_cols, strategy="constant", fill_value=fill_value
        )
        simple_imputer.fit(df)

        simple_imputer_sklearn = SklearnSimpleImputer(strategy="constant", fill_value=fill_value)
        simple_imputer_sklearn.fit(df_pandas[input_cols])

        statistics_numpy = np.array(list(simple_imputer.statistics_.values()))

        np.testing.assert_equal(statistics_numpy, simple_imputer_sklearn.statistics_)

    def test_fit_constant_no_fill_numeric(self) -> None:
        """
        Verify constant fit statistics with no fill value specified.

        Raises
        ------
        AssertionError
            If the fit result differs from the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols, strategy="constant")
        simple_imputer.fit(df)

        simple_imputer_sklearn = SklearnSimpleImputer(strategy="constant")
        simple_imputer_sklearn.fit(df_pandas[input_cols])

        statistics_numpy = np.array(list(simple_imputer.statistics_.values()))

        np.testing.assert_equal(statistics_numpy, simple_imputer_sklearn.statistics_)

    def test_fit_all_missing(self) -> None:
        """
        Verify fit statistics when the data is missing.

        Raises
        ------
        AssertionError
            If the fit result differs from the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_ALL_NONE, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        simple_imputer_sklearn = SklearnSimpleImputer()
        simple_imputer_sklearn.fit(df_pandas[input_cols])

        statistics_numpy = np.array(list(simple_imputer.statistics_.values()))

        np.testing.assert_allclose(statistics_numpy, simple_imputer_sklearn.statistics_, equal_nan=True)

    def test_fit_all_missing_categorial_keep_empty_features_false(self) -> None:
        """
        Verify fit statistics when the data is missing.

        Raises
        ------
        AssertionError
            If the fit result differs from the one generated by Sklearn.
        """
        input_cols = CATEGORICAL_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_ALL_NONE, SCHEMA)

        # TODO(hayu): [SNOW-752265] Support SimpleImputer keep_empty_features.
        #  Add back `keep_empty_features=False` when supported.
        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        simple_imputer_sklearn = SklearnSimpleImputer()
        simple_imputer_sklearn.fit(df_pandas[input_cols])

        statistics_numpy = np.array(list(simple_imputer.statistics_.values()))

        np.testing.assert_allclose(statistics_numpy, simple_imputer_sklearn.statistics_, equal_nan=True)

    def test_fit_all_missing_keep_missing_false(self) -> None:
        """
        Verify fit statistics when the data is missing.

        Raises
        ------
        AssertionError
            If the fit result differs from the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_ALL_NONE, SCHEMA)

        # TODO(hayu): [SNOW-752265] Support SimpleImputer keep_empty_features.
        #  Add back `keep_empty_features=False` when supported.
        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        simple_imputer_sklearn = SklearnSimpleImputer()
        simple_imputer_sklearn.fit(df_pandas[input_cols])

        statistics_numpy = np.array(list(simple_imputer.statistics_.values()))

        np.testing.assert_allclose(statistics_numpy, simple_imputer_sklearn.statistics_, equal_nan=True)

    def test_reset(self) -> None:
        """
        Verify reset logic.

        Raises
        ------
        AssertionError
            If attribute is not deleted.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        simple_imputer._reset()
        with self.assertRaises(AttributeError):
            simple_imputer.statistics_

    def test_transform_snowpark(self) -> None:
        """
        Verify the transformed results for an imputation of mean values in numeric columns.

        Raises
        ------
        AssertionError
            If the transformed result does not equal the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan)

        sklearn_simple_imputer = SklearnSimpleImputer()
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_allclose(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_snowpark_output_columns_same_as_input_columns(self) -> None:
        """
        Verify the transformed results for an imputation of mean values in numeric columns.

        Raises
        ------
        AssertionError
            If the transformed result does not equal the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = input_cols
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan)

        sklearn_simple_imputer = SklearnSimpleImputer()
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_allclose(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_snowpark_output_columns_one_equal_to_input_column(self) -> None:
        """
        Verify the transformed results for an imputation of mean values in numeric columns.

        Raises
        ------
        AssertionError
            If the transformed result does not equal the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = [NUMERIC_COLS[0], OUTPUT_COLS[0]]
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan)

        sklearn_simple_imputer = SklearnSimpleImputer()
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_allclose(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_snowpark_missing_values_not_nan(self) -> None:
        """
        Verify imputed data when the missing value specified is not None or nan.

        Raises
        ------
        AssertionError
            If the transformed result does not equal the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        for strategy in ["mean", "median", "most_frequent", "constant"]:
            simple_imputer = SimpleImputer(
                strategy=strategy, input_cols=input_cols, output_cols=output_cols, missing_values=-1.0
            )
            simple_imputer.fit(df)

            transformed_df = simple_imputer.transform(df)

            sklearn_simple_imputer = SklearnSimpleImputer(strategy=strategy, missing_values=-1.0)
            sklearn_simple_imputer.fit(df_pandas[input_cols])
            sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_pandas[input_cols])

            np.testing.assert_allclose(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_snowpark_most_frequent_strategy_categorical(self) -> None:
        """
        Verify imputed data for categorical columns.

        Raises
        ------
        AssertionError
            If the transformed result does not match the expected result.
        """
        input_cols = CATEGORICAL_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(
            input_cols=input_cols, output_cols=output_cols, missing_values=None, strategy="most_frequent"
        )
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan)

        sklearn_simple_imputer = SklearnSimpleImputer(missing_values=None, strategy="most_frequent")
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_equal(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_snowpark_most_frequent_strategy_categorical_mixed_types(self) -> None:
        """
        Verify imputed data for categorical columns.

        Raises
        ------
        AssertionError
            If the transformed result does not match the expected result.
        """
        input_cols = [CATEGORICAL_COLS[0], NUMERIC_COLS[0]]
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols, strategy="most_frequent")
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan)

        sklearn_simple_imputer = SklearnSimpleImputer(strategy="most_frequent")
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_equal(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_snowpark_most_frequent_strategy_numeric(self) -> None:
        """
        Verify imputed data for "most frequent" strategy and numerical data.

        Raises
        ------
        AssertionError
            If the transformed result does not match the expected result.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols, strategy="most_frequent")
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan)

        simple_imputer_sklearn = SklearnSimpleImputer(strategy="most_frequent")
        simple_imputer_sklearn.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = simple_imputer_sklearn.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_allclose(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_transformed_dataset)

    def test_transform_sklearn(self) -> None:
        """
        Verify transform of pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed result does not match the one generated by Sklearn.
        """
        input_cols = NUMERIC_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        simple_imputer = SimpleImputer(input_cols=input_cols, output_cols=output_cols)
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df = simple_imputer.transform(df_none_nan_pandas)

        sklearn_simple_imputer = SklearnSimpleImputer()
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_allclose(transformed_df[output_cols].to_numpy(), sklearn_transformed_dataset)

    def test_transform_sklearn_constant_string(self) -> None:
        """
        Verify imputed data using a string constant.

        Raises
        ------
        AssertionError
            If the transformed data does not match the expected result.
        """
        input_cols = CATEGORICAL_COLS
        output_cols = OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        fill_value = "foo"
        simple_imputer = SimpleImputer(
            input_cols=input_cols,
            output_cols=output_cols,
            strategy="constant",
            fill_value=fill_value,
            missing_values=None,
        )
        simple_imputer.fit(df)

        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        transformed_df_sklearn = simple_imputer.transform(df_none_nan_pandas)

        sklearn_simple_imputer = SklearnSimpleImputer(
            strategy="constant",
            fill_value=fill_value,
            missing_values=None,
        )
        sklearn_simple_imputer.fit(df_pandas[input_cols])
        sklearn_transformed_dataset = sklearn_simple_imputer.transform(df_none_nan_pandas[input_cols])

        np.testing.assert_equal(transformed_df_sklearn[output_cols].to_numpy(), sklearn_transformed_dataset)

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        data, schema = DATA, SCHEMA
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        simple_imputer = SimpleImputer().set_input_cols(input_cols).set_output_cols(output_cols)
        simple_imputer.fit(df1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            self._to_be_deleted_files.append(file.name)
            simple_imputer_dump_cloudpickle = cloudpickle.dumps(simple_imputer)
            simple_imputer_dump_pickle = pickle.dumps(simple_imputer)
            joblib.dump(simple_imputer, file.name)

            self._session.close()

            # transform in session 2
            self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
            _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
            input_cols_extended = input_cols.copy()
            input_cols_extended.append(id_col)

            importlib.reload(sys.modules["snowflake.ml.modeling.impute.simple_imputer"])

            # cloudpickle
            simple_imputer_load_cloudpickle = cloudpickle.loads(simple_imputer_dump_cloudpickle)
            transformed_df_cloudpickle = simple_imputer_load_cloudpickle.transform(df2[input_cols_extended])
            actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # pickle
            simple_imputer_load_pickle = pickle.loads(simple_imputer_dump_pickle)
            transformed_df_pickle = simple_imputer_load_pickle.transform(df2[input_cols_extended])
            actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # joblib
            simple_imputer_load_joblib = joblib.load(file.name)
            transformed_df_joblib = simple_imputer_load_joblib.transform(df2[input_cols_extended])
            actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

            # sklearn
            simple_imputer_sklearn = SklearnSimpleImputer()
            simple_imputer_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = simple_imputer_sklearn.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
