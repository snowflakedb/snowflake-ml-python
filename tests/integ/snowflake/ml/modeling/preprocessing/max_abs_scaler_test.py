#!/usr/bin/env python3
import importlib
import os
import pickle
import sys
import tempfile
from typing import List

import cloudpickle
import joblib
import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler

from snowflake.ml.modeling.preprocessing import (
    MaxAbsScaler,  # type: ignore[attr-defined]
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class MaxAbsScalerTest(TestCase):
    """Test MaxAbsScaler."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_fit(self) -> None:
        """
        Verify fitted states.

        Raises
        ------
        AssertionError
            If the fitted states do not match those of the sklearn scaler.
        """
        input_cols = NUMERIC_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        for _df in [df_pandas, df]:
            scaler = MaxAbsScaler().set_input_cols(input_cols)
            scaler.fit(_df)

            actual_max_abs = scaler._convert_attribute_dict_to_ndarray(scaler.max_abs_)
            actual_scale = scaler._convert_attribute_dict_to_ndarray(scaler.scale_)

            # sklearn
            scaler_sklearn = SklearnMaxAbsScaler()
            scaler_sklearn.fit(df_pandas[input_cols])

            np.testing.assert_allclose(actual_max_abs, scaler_sklearn.max_abs_)
            np.testing.assert_allclose(actual_scale, scaler_sklearn.scale_)

    def test_transform(self) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        input_df_pandas, input_df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Fit with snowpark dataframe, transform on snowpark dataframe
        scaler = MaxAbsScaler(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        scaler.fit(input_df)
        transformed_df = scaler.transform(input_df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # Fit with pandas dataframe, transform on snowpark dataframe
        scaler2 = MaxAbsScaler(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        scaler2.fit(input_df_pandas)
        transformed_df = scaler2.transform(input_df[input_cols_extended])
        actual_arr2 = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnMaxAbsScaler()
        scaler_sklearn.fit(input_df_pandas[input_cols])
        expected_arr = scaler_sklearn.transform(input_df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, expected_arr)
        np.testing.assert_allclose(actual_arr2, expected_arr)

    def test_transform_pandas_input(self) -> None:
        data, schema = DATA, SCHEMA
        input_cols, output_cols = NUMERIC_COLS, OUTPUT_COLS
        input_df_pandas = pd.DataFrame(data, columns=schema)
        input_df = self._session.create_dataframe(input_df_pandas)

        scaler = MaxAbsScaler(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        scaler.fit(input_df)
        transformed_df = scaler.transform(input_df_pandas[input_cols])
        actual_arr = transformed_df[output_cols].to_numpy()

        scaler_sklearn = SklearnMaxAbsScaler()
        scaler_sklearn.fit(input_df_pandas[input_cols])
        transformed_df_sklearn = scaler_sklearn.transform(input_df_pandas[input_cols])
        expected_arr = transformed_df_sklearn

        np.testing.assert_allclose(actual_arr, expected_arr)

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

        scaler = MaxAbsScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            self._to_be_deleted_files.append(file.name)
            scaler_dump_cloudpickle = cloudpickle.dumps(scaler)
            scaler_dump_pickle = pickle.dumps(scaler)
            joblib.dump(scaler, file.name)

            self._session.close()

            # transform in session 2
            self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
            _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
            input_cols_extended = input_cols.copy()
            input_cols_extended.append(id_col)

            importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.max_abs_scaler"])

            # cloudpickle
            scaler_load_cloudpickle = cloudpickle.loads(scaler_dump_cloudpickle)
            transformed_df_cloudpickle = scaler_load_cloudpickle.transform(df2[input_cols_extended])
            actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # pickle
            scaler_load_pickle = pickle.loads(scaler_dump_pickle)
            transformed_df_pickle = scaler_load_pickle.transform(df2[input_cols_extended])
            actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # joblib
            scaler_load_joblib = joblib.load(file.name)
            transformed_df_joblib = scaler_load_joblib.transform(df2[input_cols_extended])
            actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

            # sklearn
            scaler_sklearn = SklearnMaxAbsScaler()
            scaler_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = scaler_sklearn.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
