#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import os
import pickle
import tempfile
from typing import Any, Dict, List

import cloudpickle
import joblib
import numpy as np
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn.preprocessing import Normalizer as SklearnNormalizer

from snowflake.ml.preprocessing import Normalizer
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from tests.integ.snowflake.ml.framework import utils as framework_utils
from tests.integ.snowflake.ml.framework.utils import (
    CATEGORICAL_COLS,
    DATA_NONE_NAN,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)

_DATA_NORMALIZE = [
    ["1", "c", "zeroes", 0.0, 0.0],
    ["2", "a", "norm:l2", -3.0, -4.0],
    ["3", "a", "norm:l2", 3.0, 4.0],
    ["4", "A", "big_number", 3.5, -123456789.9999],
    ["5", "A", "small_number", 3.0, 0.00000000012],
    ["6", "b", "norm:max", -20.0, -100.0],
    ["7", "b", "norm:max", -10.0, 100],
]


class NormalizerTest(parameterized.TestCase):
    """Test Normalizer."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    @parameterized.parameters({"norm": "l2"}, {"norm": "l1"}, {"norm": "max"})  # type: ignore [misc]
    def test_transform(self, norm: Dict[str, Any]) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, _DATA_NORMALIZE, SCHEMA, np.nan)

        normalizer = Normalizer(
            norm=norm,
            input_cols=input_cols,
            output_cols=output_cols,
        )
        normalizer.fit(df)

        transformed_df = normalizer.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        normalizer_sklearn = SklearnNormalizer(norm=norm)
        normalizer_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = normalizer_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        assert np.allclose(actual_arr, sklearn_arr)

    def test_transform_raises_on_unsupported_norm(self) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        _, df = framework_utils.get_df(self._session, _DATA_NORMALIZE, SCHEMA, np.nan)

        normalizer = Normalizer(
            norm="an unsupported norm",
            input_cols=input_cols,
            output_cols=output_cols,
        )

        normalizer.fit(df)

        with self.assertRaises(ValueError):
            normalizer.transform(df[input_cols_extended])

    def test_transform_raises_with_null_inputs(self) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        _, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        normalizer = Normalizer(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        normalizer.fit(df)

        with self.assertRaises(ValueError):
            normalizer.transform(df[input_cols_extended])

    def test_transform_raises_on_3d_data(self) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)

        _, df = framework_utils.get_df(
            self._session, [["1", "a", "a", [1], [2]], ["2", "a", "a", [3], [4]]], SCHEMA, np.nan
        )
        normalizer = Normalizer(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        normalizer.fit(df)
        transformed_df = normalizer.transform(df[input_cols_extended])

        with self.assertRaises(SnowparkSQLException):
            transformed_df.collect()

    def test_transform_raises_with_nonnumerical_columns(self) -> None:
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        _, df = framework_utils.get_df(self._session, _DATA_NORMALIZE, SCHEMA, np.nan)

        normalizer = Normalizer(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        normalizer.fit(df)
        transformed_df = normalizer.transform(df[input_cols_extended])

        with self.assertRaises(SnowparkSQLException):
            transformed_df.collect()

    @parameterized.parameters({"norm": "l2"}, {"norm": "l1"}, {"norm": "max"})  # type: ignore [misc]
    def test_transform_pandas(self, norm: Dict[str, Any]) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        df_pandas, df = framework_utils.get_df(self._session, _DATA_NORMALIZE, SCHEMA, np.nan)

        normalizer = Normalizer(
            norm=norm,
            input_cols=input_cols,
            output_cols=output_cols,
        )
        normalizer.fit(df)

        transformed_df = normalizer.transform(df_pandas[input_cols])
        actual_arr = transformed_df[output_cols].to_numpy()

        normalizer_sklearn = SklearnNormalizer(norm=norm)
        normalizer_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = normalizer_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        assert np.allclose(actual_arr, sklearn_arr)

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn normalizer.
        """
        data, schema = _DATA_NORMALIZE, SCHEMA
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        normalizer = Normalizer().set_input_cols(input_cols).set_output_cols(output_cols)
        normalizer.fit(df1)
        filepath = os.path.join(tempfile.gettempdir(), "test_standard_normalizer.pkl")
        self._to_be_deleted_files.append(filepath)
        normalizer_dump_cloudpickle = cloudpickle.dumps(normalizer)
        normalizer_dump_pickle = pickle.dumps(normalizer)
        joblib.dump(normalizer, filepath)

        self._session.close()

        # transform in session 2
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)

        # cloudpickle
        normalizer_load_cloudpickle = cloudpickle.loads(normalizer_dump_cloudpickle)
        transformed_df_cloudpickle = normalizer_load_cloudpickle.transform(df2[input_cols_extended])
        actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

        # pickle
        normalizer_load_pickle = pickle.loads(normalizer_dump_pickle)
        transformed_df_pickle = normalizer_load_pickle.transform(df2[input_cols_extended])
        actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

        # joblib
        normalizer_load_joblib = joblib.load(filepath)
        transformed_df_joblib = normalizer_load_joblib.transform(df2[input_cols_extended])
        actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        normalizer_sklearn = SklearnNormalizer()
        normalizer_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = normalizer_sklearn.transform(df_pandas[input_cols])

        assert np.allclose(actual_arr_cloudpickle, sklearn_arr)
        assert np.allclose(actual_arr_pickle, sklearn_arr)
        assert np.allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
