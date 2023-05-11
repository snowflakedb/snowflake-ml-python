#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import importlib
import os
import pickle
import sys
import tempfile
from typing import List

import cloudpickle
import joblib
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

from snowflake.ml.preprocessing import MinMaxScaler  # type: ignore[attr-defined]
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.framework import utils as framework_utils
from tests.integ.snowflake.ml.framework.utils import (
    DATA,
    DATA_CLIP,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class MinMaxScalerTest(TestCase):
    """Test MinMaxScaler."""

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
            scaler = MinMaxScaler().set_input_cols(input_cols)
            scaler.fit(_df)

            actual_min = scaler._convert_attribute_dict_to_ndarray(scaler.min_)
            actual_scale = scaler._convert_attribute_dict_to_ndarray(scaler.scale_)
            actual_data_min = scaler._convert_attribute_dict_to_ndarray(scaler.data_min_)
            actual_data_max = scaler._convert_attribute_dict_to_ndarray(scaler.data_max_)
            actual_data_range = scaler._convert_attribute_dict_to_ndarray(scaler.data_range_)

            # sklearn
            scaler_sklearn = SklearnMinMaxScaler()
            scaler_sklearn.fit(df_pandas[input_cols])

            np.testing.assert_allclose(actual_min, scaler_sklearn.min_)
            np.testing.assert_allclose(actual_scale, scaler_sklearn.scale_)
            np.testing.assert_allclose(actual_data_min, scaler_sklearn.data_min_)
            np.testing.assert_allclose(actual_data_max, scaler_sklearn.data_max_)
            np.testing.assert_allclose(actual_data_range, scaler_sklearn.data_range_)

    def test_transform(self) -> None:
        """
        Verify transformed results.

        Fitted dataset:
        --------------------------------------------------------
        |"ID"  |"STR1"  |"STR2"      |"FLOAT1"  |"FLOAT2"      |
        --------------------------------------------------------
        |1     |c       |g1ehQlL80t  |-1.0      |0.0           |
        |2     |a       |zOyDvcyZ2s  |8.3       |124.6         |
        |3     |b       |zOyDvcyZ2s  |2.0       |2253463.5342  |
        |4     |A       |TuDOcLxToB  |3.5       |-1350.6407    |
        |5     |d       |g1ehQlL80t  |2.5       |-1.0          |
        |6     |b       |g1ehQlL80t  |4.0       |457946.23462  |
        |7     |b       |g1ehQlL80t  |-10.0     |-12.564       |
        --------------------------------------------------------

        Source SQL query:
        SELECT  *  FROM TEMP_TABLE

        Dataset to transform:
        ---------------------------
        |"FLOAT1"  |"FLOAT2"      |
        ---------------------------
        |-1.0      |0.0           |
        |8.3       |124.6         |
        |2.0       |2253463.5342  |
        |3.5       |-1350.6407    |
        |2.5       |-1.0          |
        |4.0       |457946.23462  |
        |-10.0     |-12.564       |
        ---------------------------

        Transformed SQL query:
        SELECT "FLOAT1", "FLOAT2", "ID", (("FLOAT1" * 'scale_' :: FLOAT) + 'min_' :: FLOAT) AS "OUTPUT1",
        (("FLOAT2" * 'scale_' :: FLOAT) + 'min_' :: FLOAT) AS "OUTPUT2" FROM ( SELECT "FLOAT1", "FLOAT2", "ID"
        FROM (SOURCE_SQL_QUERY))

        Transformed dataset:
        --------------------------------------------------------------------------------
        |"FLOAT1"  |"FLOAT2"      |"ID"  |"OUTPUT1"              |"OUTPUT2"            |
        --------------------------------------------------------------------------------
        |-1.0      |0.0           |1     |-0.016393442622950935  |-0.9988019937828714  |
        |8.3       |124.6         |2     |0.9999999999999999     |-0.9986914746976296  |
        |2.0       |2253463.5342  |3     |0.3114754098360655     |1.0000000000000002   |
        |3.5       |-1350.6407    |4     |0.47540983606557363    |-1.0                 |
        |2.5       |-1.0          |5     |0.3661202185792348     |-0.9988028807739247  |
        |4.0       |457946.23462  |6     |0.5300546448087431     |-0.592607780780543   |
        |-10.0     |-12.564       |7     |-1.0                   |-0.998813137938465   |
        --------------------------------------------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        feature_range = (-1, 1)

        # Fit with snowpark dataframe, transformed on snowpark dataframe
        scaler = MinMaxScaler(feature_range=feature_range).set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)

        transformed_df = scaler.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # Fit with pandas dataframe, transformed on snowpark dataframe
        scaler2 = MinMaxScaler(feature_range=feature_range).set_input_cols(input_cols).set_output_cols(output_cols)
        scaler2.fit(df_pandas)

        transformed_df = scaler2.transform(df[input_cols_extended])
        actual_arr2 = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnMinMaxScaler(feature_range=feature_range)
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)
        np.testing.assert_allclose(actual_arr2, sklearn_arr)

    def test_transform_clip(self) -> None:
        """
        Test `clip=True`.

        Fitted dataset:
        ---------------------------------------------------------
        |"ID"  |"STR1"  |"STR2"      |"FLOAT1"  |"FLOAT2"       |
        ---------------------------------------------------------
        |1     |c       |g1ehQlL80t  |-1.0      |0.0            |
        |2     |a       |zOyDvcyZ2s  |8.3       |124.6          |
        |3     |b       |zOyDvcyZ2s  |2.0       |230385.5342    |
        |4     |A       |TuDOcLxToB  |3.5       |-1350.6407     |
        |5     |d       |g1ehQlL80t  |2.5       |-1.0           |
        |6     |b       |g1ehQlL80t  |4.0       |5304.23920462  |
        |7     |b       |g1ehQlL80t  |-10.0     |-12.564        |
        ---------------------------------------------------------

        Source SQL query:
        SELECT  *  FROM TEMP_TABLE

        Dataset to transform:
        ----------------------------
        |"FLOAT1"  |"FLOAT2"       |
        ----------------------------
        |-1.0      |0.0            |
        |8.3       |124.6          |
        |2.0       |230385.5342    |
        |3.5       |-1350.6407     |
        |2.5       |-1.0           |
        |4.0       |5304.23920462  |
        |-10.0     |-12.564        |
        ----------------------------

        Transformed SQL query:
        SELECT "FLOAT1", "FLOAT2", "ID", least(greatest((("FLOAT1" * 'scale_' :: FLOAT) + 'min_' :: FLOAT),
        feature_range[0]), feature_range[1]) AS "OUTPUT1",
        least(greatest((("FLOAT2" * 'scale_' :: FLOAT) + 'min_' :: FLOAT),
        feature_range[0]), feature_range[1]) AS "OUTPUT2" FROM ( SELECT "FLOAT1", "FLOAT2", "ID"
        FROM (SOURCE_SQL_QUERY))

        Transformed dataset:
        --------------------------------------------------------------------------------
        |"FLOAT1"  |"FLOAT2"       |"ID"  |"OUTPUT1"             |"OUTPUT2"            |
        --------------------------------------------------------------------------------
        |-1.0      |0.0            |1     |-0.03278688524590187  |-1.9766865798905529  |
        |8.3       |124.6          |2     |1.9999999999999998    |-1.9745358582769978  |
        |2.0       |230385.5342    |3     |0.622950819672131     |2.0                  |
        |3.5       |-1350.6407     |4     |0.9508196721311473    |-2.0                 |
        |2.5       |-1.0           |5     |0.7322404371584696    |-1.9767038408986872  |
        |4.0       |5304.23920462  |6     |1.0601092896174862    |-1.885130063832429   |
        |-10.0     |-12.564        |7     |-2.0                  |-1.9769034471967544  |
        --------------------------------------------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_CLIP, SCHEMA, np.nan)

        feature_range = (-2, 2)
        scaler = (
            MinMaxScaler(feature_range=feature_range, clip=True).set_input_cols(input_cols).set_output_cols(output_cols)
        )
        scaler.fit(df)

        transformed_df = scaler.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnMinMaxScaler(feature_range=feature_range, clip=True)
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_pandas(self) -> None:
        """
        Verify transformed results on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols = NUMERIC_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        feature_range = (-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range).set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)

        transformed_df = scaler.transform(df_pandas[input_cols])
        actual_arr = transformed_df[output_cols].to_numpy()

        # sklearn
        scaler_sklearn = SklearnMinMaxScaler(feature_range=feature_range)
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_clip_pandas(self) -> None:
        """
        Test `clip=True` on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols = NUMERIC_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_CLIP, SCHEMA, np.nan)

        feature_range = (-2, 2)
        scaler = (
            MinMaxScaler(feature_range=feature_range, clip=True).set_input_cols(input_cols).set_output_cols(output_cols)
        )
        scaler.fit(df)

        transformed_df = scaler.transform(df_pandas[input_cols])
        actual_arr = transformed_df[output_cols].to_numpy()

        # sklearn
        scaler_sklearn = SklearnMinMaxScaler(feature_range=feature_range, clip=True)
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_args_with_cols(self) -> None:
        """
        Verify that queries via different ways of setting input & output columns are identical.

        Raises
        ------
        AssertionError
            If the queries are not identical.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        scaler1 = MinMaxScaler()
        scaler1.set_input_cols(input_cols)
        scaler1.set_output_cols(output_cols)
        transformed_df1 = scaler1.fit(df).transform(df[input_cols_extended])

        scaler2 = MinMaxScaler(input_cols=input_cols, output_cols=output_cols)
        transformed_df2 = scaler2.fit(df).transform(df[input_cols_extended])

        scaler3 = MinMaxScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        transformed_df3 = scaler3.fit(df).transform(df[input_cols_extended])

        self.assertEqual(transformed_df1.queries["queries"][-1], transformed_df2.queries["queries"][-1])
        self.assertEqual(transformed_df1.queries["queries"][-1], transformed_df3.queries["queries"][-1])

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

        scaler = MinMaxScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df1)

        filepath = os.path.join(tempfile.gettempdir(), "test_min_max_scaler.pkl")
        self._to_be_deleted_files.append(filepath)
        scaler_dump_cloudpickle = cloudpickle.dumps(scaler)
        scaler_dump_pickle = pickle.dumps(scaler)
        joblib.dump(scaler, filepath)

        self._session.close()

        # transform in session 2
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)

        importlib.reload(sys.modules["snowflake.ml.preprocessing.min_max_scaler"])

        # cloudpickle
        scaler_load_cloudpickle = cloudpickle.loads(scaler_dump_cloudpickle)
        transformed_df_cloudpickle = scaler_load_cloudpickle.transform(df2[input_cols_extended])
        actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

        # pickle
        scaler_load_pickle = pickle.loads(scaler_dump_pickle)
        transformed_df_pickle = scaler_load_pickle.transform(df2[input_cols_extended])
        actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

        # joblib
        scaler_load_joblib = joblib.load(filepath)
        transformed_df_joblib = scaler_load_joblib.transform(df2[input_cols_extended])
        actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnMinMaxScaler()
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
        np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
        np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
