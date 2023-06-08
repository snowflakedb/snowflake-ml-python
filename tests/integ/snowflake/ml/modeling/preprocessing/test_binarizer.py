#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import importlib
import os
import pickle
import sys
import tempfile

import cloudpickle
import joblib
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.preprocessing import Binarizer as SklearnBinarizer

from snowflake.ml.modeling.preprocessing import Binarizer  # type: ignore[attr-defined]
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    DATA_NONE_NAN,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class BinarizerTest(TestCase):
    """Test Binarizer."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_transform(self) -> None:
        threshold = 2.0

        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        input_df_pandas, input_df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        binarizer = Binarizer(
            threshold=threshold,
            input_cols=input_cols,
            output_cols=output_cols,
        )

        binarizer.fit(input_df)

        transformed_df = binarizer.transform(input_df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        sklearn_binarizer = SklearnBinarizer(threshold=threshold)
        sklearn_binarizer.fit(input_df_pandas[input_cols])
        expected_arr = sklearn_binarizer.transform(input_df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, expected_arr)

    def test_transform_pandas_input(self) -> None:
        threshold = 2.0

        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        binarizer = Binarizer(
            threshold=threshold,
            input_cols=input_cols,
            output_cols=output_cols,
        )
        binarizer.fit(df)

        transformed_sklearn_df = binarizer.transform(input_df_pandas[input_cols])
        actual_arr = transformed_sklearn_df[output_cols].to_numpy()

        sklearn_binarizer = SklearnBinarizer(threshold=threshold)
        sklearn_binarizer.fit(input_df_pandas[input_cols])
        expected_arr = sklearn_binarizer.transform(input_df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, expected_arr)

    def test_transform_raises_on_nulls(self) -> None:
        threshold = 2.0

        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        _, input_df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        binarizer = Binarizer(
            threshold=threshold,
            input_cols=input_cols,
            output_cols=output_cols,
        )

        binarizer.fit(input_df)

        with self.assertRaises(ValueError):
            binarizer.transform(input_df[input_cols_extended])

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.
        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn binarizer.
        """
        data, schema = DATA, SCHEMA
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        threshold = 2.0

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        binarizer = Binarizer(threshold=threshold).set_input_cols(input_cols).set_output_cols(output_cols)
        binarizer.fit(df1)
        filepath = os.path.join(tempfile.gettempdir(), "test_serialization.pkl")
        self._to_be_deleted_files.append(filepath)
        binarizer_dump_cloudpickle = cloudpickle.dumps(binarizer)
        binarizer_dump_pickle = pickle.dumps(binarizer)
        joblib.dump(binarizer, filepath)

        self._session.close()

        # transform in session 2
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)

        importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.binarizer"])

        # cloudpickle
        binarizer_load_cloudpickle = cloudpickle.loads(binarizer_dump_cloudpickle)
        transformed_df_cloudpickle = binarizer_load_cloudpickle.transform(df2[input_cols_extended])
        actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

        # pickle
        binarizer_load_pickle = pickle.loads(binarizer_dump_pickle)
        transformed_df_pickle = binarizer_load_pickle.transform(df2[input_cols_extended])
        actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

        # joblib
        binarizer_load_joblib = joblib.load(filepath)
        transformed_df_joblib = binarizer_load_joblib.transform(df2[input_cols_extended])
        actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        binarizer_sklearn = SklearnBinarizer(threshold=threshold)
        binarizer_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = binarizer_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
        np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
        np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
