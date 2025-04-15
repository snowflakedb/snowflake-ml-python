#!/usr/bin/env python3
import importlib
import os
import pickle
import sys
import tempfile

import cloudpickle
import joblib
import numpy as np
from absl.testing.absltest import TestCase, main
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    StandardScaler,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session, functions, types
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class StandardScalerTest(TestCase):
    """Test StandardScaler."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: list[str] = []

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
            scaler = StandardScaler().set_input_cols(input_cols)
            scaler.fit(_df)

            actual_scale = scaler._convert_attribute_dict_to_ndarray(scaler.scale_)
            actual_mean = scaler._convert_attribute_dict_to_ndarray(scaler.mean_)
            actual_var = scaler._convert_attribute_dict_to_ndarray(scaler.var_)

            # sklearn
            scaler_sklearn = SklearnStandardScaler()
            scaler_sklearn.fit(df_pandas[input_cols])

            np.testing.assert_allclose(actual_scale, scaler_sklearn.scale_)
            np.testing.assert_allclose(actual_mean, scaler_sklearn.mean_)
            np.testing.assert_allclose(actual_var, scaler_sklearn.var_)

    def test_fit_decimal(self) -> None:
        """
        Verify fitted states with DecimalType

        Raises
        ------
        AssertionError
            If the fitted states do not match those of the sklearn scaler.
        """
        input_cols = NUMERIC_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Map DoubleType to DecimalType
        fields = df.schema.fields
        selected_cols = []
        for field in fields:
            src = field.column_identifier.quoted_name
            if isinstance(field.datatype, types.DoubleType):
                dest = types.DecimalType(38, 10)
                selected_cols.append(functions.cast(functions.col(src), dest).alias(src))
            else:
                selected_cols.append(functions.col(src))
        df = df.select(selected_cols)

        for _df in [df_pandas, df]:
            scaler = StandardScaler().set_input_cols(input_cols)
            scaler.fit(_df)

            actual_scale = scaler._convert_attribute_dict_to_ndarray(scaler.scale_)
            actual_mean = scaler._convert_attribute_dict_to_ndarray(scaler.mean_)
            actual_var = scaler._convert_attribute_dict_to_ndarray(scaler.var_)

            # sklearn
            scaler_sklearn = SklearnStandardScaler()
            scaler_sklearn.fit(df_pandas[input_cols])

            np.testing.assert_allclose(actual_scale, scaler_sklearn.scale_)
            np.testing.assert_allclose(actual_mean, scaler_sklearn.mean_)
            np.testing.assert_allclose(actual_var, scaler_sklearn.var_)

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
        SELECT "FLOAT1", "FLOAT2", "ID", (("FLOAT1" - 'mean_' :: FLOAT) / 'scale_' :: FLOAT) AS "OUTPUT1",
        (("FLOAT2" - 'mean_' :: FLOAT) / 'scale_' :: FLOAT) AS "OUTPUT2" FROM ( SELECT "FLOAT1", "FLOAT2", "ID"
        FROM (SOURCE_SQL_QUERY))

        Transformed dataset:
        -------------------------------------------------------------------------------
        |"FLOAT1"  |"FLOAT2"      |"ID"  |"OUTPUT1"            |"OUTPUT2"             |
        -------------------------------------------------------------------------------
        |-1.0      |0.0           |1     |-0.4400201523874598  |-0.4975539003584193   |
        |8.3       |124.6         |2     |1.317360947025033    |-0.4973937751686294   |
        |2.0       |2253463.5342  |3     |0.1268769764552798   |2.3984033716991413    |
        |3.5       |-1350.6407    |4     |0.4103255408766496   |-0.4992896274725882   |
        |2.5       |-1.0          |5     |0.22135983126240305  |-0.49755518547230204  |
        |4.0       |457946.23462  |6     |0.5048083956837728   |0.09095916330203942   |
        |-10.0     |-12.564       |7     |-2.140711538915679   |-0.4975700465292421   |
        -------------------------------------------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Fit with snowpark dataframe, transformed on snowpark dataframe
        scaler = StandardScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)

        transformed_df = scaler.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # Fit with pandas dataframe, transformed on snowpark dataframe
        scaler2 = StandardScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler2.fit(df_pandas)

        transformed_df = scaler2.transform(df[input_cols_extended])
        actual_arr2 = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnStandardScaler()
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)
        np.testing.assert_allclose(actual_arr2, sklearn_arr)

    def test_transform_without_mean(self) -> None:
        """
        Test `with_mean=False`.

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
        SELECT "FLOAT1", "FLOAT2", "ID", ("FLOAT1" / 'scale_' :: FLOAT) AS "OUTPUT1",
        ("FLOAT2" / 'scale_' :: FLOAT) AS "OUTPUT2" FROM ( SELECT "FLOAT1", "FLOAT2", "ID"
        FROM (SOURCE_SQL_QUERY))

        Transformed dataset:
        -----------------------------------------------------------------------------------
        |"FLOAT1"  |"FLOAT2"      |"ID"  |"OUTPUT1"             |"OUTPUT2"                |
        -----------------------------------------------------------------------------------
        |-1.0      |0.0           |1     |-0.18896570961424652  |0.0                      |
        |8.3       |124.6         |2     |1.5684153897982462    |0.00016012518978989035   |
        |2.0       |2253463.5342  |3     |0.37793141922849305   |2.895957272057561        |
        |3.5       |-1350.6407    |4     |0.6613799836498628    |-0.0017357271141689435   |
        |2.5       |-1.0          |5     |0.4724142740356163    |-1.2851138827439032e-06  |
        |4.0       |457946.23462  |6     |0.7558628384569861    |0.5885130636604587       |
        |-10.0     |-12.564       |7     |-1.8896570961424652   |-1.61461708227944e-05    |
        -----------------------------------------------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        scaler = StandardScaler(with_mean=False).set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)

        transformed_df = scaler.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnStandardScaler(with_mean=False)
        scaler_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = scaler_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_without_std(self) -> None:
        """
        Test `with_std=False`.

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
        SELECT "FLOAT1", "FLOAT2", "ID", ("FLOAT1" - 'mean_' :: FLOAT) AS "OUTPUT1",
        ("FLOAT2" - 'mean_' :: FLOAT) AS "OUTPUT2" FROM ( SELECT "FLOAT1", "FLOAT2", "ID"
        FROM (SOURCE_SQL_QUERY))

        Transformed dataset:
        ------------------------------------------------------------------------------
        |"FLOAT1"  |"FLOAT2"      |"ID"  |"OUTPUT1"            |"OUTPUT2"            |
        ------------------------------------------------------------------------------
        |-1.0      |0.0           |1     |-2.3285714285714287  |-387167.16630285716  |
        |8.3       |124.6         |2     |6.9714285714285715   |-387042.5663028572   |
        |2.0       |2253463.5342  |3     |0.6714285714285713   |1866296.367897143    |
        |3.5       |-1350.6407    |4     |2.1714285714285713   |-388517.80700285715  |
        |2.5       |-1.0          |5     |1.1714285714285713   |-387168.16630285716  |
        |4.0       |457946.23462  |6     |2.6714285714285713   |70779.06831714284    |
        |-10.0     |-12.564       |7     |-11.32857142857143   |-387179.7303028572   |
        ------------------------------------------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn scaler.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        scaler = StandardScaler(with_std=False).set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)

        transformed_df = scaler.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnStandardScaler(with_std=False)
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

        scaler = StandardScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df)

        transformed_df = scaler.transform(df_pandas[input_cols])
        actual_arr = transformed_df[output_cols].to_numpy()

        # sklearn
        scaler_sklearn = SklearnStandardScaler()
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

        scaler1 = StandardScaler()
        scaler1.set_input_cols(input_cols)
        scaler1.set_output_cols(output_cols)
        transformed_df1 = scaler1.fit(df).transform(df[input_cols_extended])

        scaler2 = StandardScaler(input_cols=input_cols, output_cols=output_cols)
        transformed_df2 = scaler2.fit(df).transform(df[input_cols_extended])

        scaler3 = StandardScaler().set_input_cols(input_cols).set_output_cols(output_cols)
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

        scaler = StandardScaler().set_input_cols(input_cols).set_output_cols(output_cols)
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

            importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.standard_scaler"])

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
            scaler_sklearn = SklearnStandardScaler()
            scaler_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = scaler_sklearn.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
