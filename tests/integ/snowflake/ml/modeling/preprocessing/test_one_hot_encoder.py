#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import importlib
import json
import os
import pickle
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import pytest
from absl.testing import parameterized
from absl.testing.absltest import main
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from snowflake.ml._internal.utils import identifier
from snowflake.ml.modeling.preprocessing import (
    OneHotEncoder,  # type: ignore[attr-defined]
)
from snowflake.ml.utils import sparse as utils_sparse
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import DataFrame, Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    BOOLEAN_COLS,
    CATEGORICAL_COLS,
    DATA,
    DATA_BOOLEAN,
    DATA_NONE_NAN,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
    SCHEMA_BOOLEAN,
    UNKNOWN_CATEGORICAL_VALUES_LIST,
    _EqualityFunc,
    equal_default,
    equal_list_of,
    equal_np_array,
    equal_optional_of,
    equal_pandas_df_ignore_row_order,
)

_DATA_QUOTES = [
    ["1", '"A"', "'g1ehQlL80t'", -1.0, 0.0],
    ["2", '"a"', '"zOyDv"cyZ2s"', 8.3, 124.6],
    ["3", '""a""', '"zOyDv""cyZ2s', 2.0, 2253463.5342],
    ["4", "''B''", '""zOyDvcyZ2s', 3.5, -1350.6407],
    ["5", "''b''", "'g1ehQl''L80t", 2.5, -1.0],
    ["6", "'b\"", "''g1ehQlL80t", 4.0, 457946.23462],
    ["7", "'b", "\"'g1ehQl'L80t'", -10.0, -12.564],
]


class OneHotEncoderTest(parameterized.TestCase):
    """Test OneHotEncoder."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    @staticmethod
    def compare_sparse_transform_results(native_arr: np.ndarray, sklearn_res: np.ndarray) -> bool:
        """Compare native and sklearn sparse transform results."""
        # if the numbers of stored values are equal
        is_nnz_equal = native_arr.shape[0] == sklearn_res.nnz
        sklearn_coo = sklearn_res.tocoo()
        sklearn_values = []
        for coo_row, coo_col in zip(sklearn_coo.row, sklearn_coo.col):
            sklearn_values.append([coo_row, coo_col])
        sklearn_arr = np.array(sklearn_values)
        return bool(is_nnz_equal) and np.allclose(native_arr, sklearn_arr)

    @staticmethod
    def convert_sparse_df_to_arr(sparse_df: DataFrame, output_cols: List[str], id_col: str) -> np.ndarray:
        """Convert sparse transformed dataframe to an array of [row, column]."""
        sparse_output_pandas = (
            sparse_df.sort(id_col)[output_cols].to_pandas().applymap(lambda x: json.loads(x) if x else None)
        )

        def map_output(x: Optional[Dict[str, Any]]) -> Optional[List[int]]:
            """Map {encoding: 1, "array_length": length} to [encoding, length]."""
            if x is None:
                return None
            res = [-1, -1]
            for key in x:
                if key == "array_length":
                    res[1] = x[key]
                else:
                    res[0] = int(key)
            return res

        mapped_pandas = sparse_output_pandas.applymap(map_output)
        base_encoding = 0
        for idx in range(1, len(mapped_pandas.columns)):
            base_encoding += mapped_pandas.iloc[0, idx - 1][1]
            mapped_pandas.iloc[:, idx] = mapped_pandas.iloc[:, idx].apply(lambda x: [x[0] + base_encoding, x[1]])

        values = []
        for row_idx, row in mapped_pandas.iterrows():
            for _, item in row.items():
                if item is not None and not np.isnan(item[0]):
                    values.append([row_idx, item[0]])
        return np.array(values)

    def test_fit(self) -> None:
        """
        Verify fitted categories.

        Raises
        ------
        AssertionError
            If the fitted categories do not match those of the sklearn encoder.
        """
        input_cols = CATEGORICAL_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = OneHotEncoder().set_input_cols(input_cols)
        encoder.fit(df)

        actual_categories = encoder._categories_list

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])

        for actual_cats, sklearn_cats in zip(actual_categories, encoder_sklearn.categories_):
            self.assertEqual(sklearn_cats.tolist(), actual_cats.tolist())

    @parameterized.parameters(  # type: ignore
        {"params": {}},
        {"params": {"min_frequency": 2}},
        {"params": {"max_categories": 2}},
    )
    def test_fit_pandas(self, params: Dict[str, Any]) -> None:
        """
        Verify that:
            (1) an encoder fit on a pandas dataframe matches its sklearn counterpart on `categories_`, and
            (2) the encoder fit on a pandas dataframe matches the encoder fit on a Snowpark DataFrame on all
                relevant stateful attributes.

        Args:
            params: A parameter combination for the encoder's constructor.

        Raises:
            AssertionError: If either condition does not hold.
        """
        input_cols = CATEGORICAL_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Fit with SnowPark DF
        encoder1 = OneHotEncoder(**params).set_input_cols(input_cols)
        encoder1.fit(df)

        # Fit with Pandas DF
        encoder2 = OneHotEncoder(**params).set_input_cols(input_cols)
        encoder2.fit(df_pandas)

        # Fit with an SKLearn encoder
        encoder_sklearn = SklearnOneHotEncoder(**params)
        encoder_sklearn.fit(df_pandas[input_cols])

        # Validate that SnowML transformer state is equivalent to SKLearn transformer state
        for pandas_cats, sklearn_cats in zip(encoder2._categories_list, encoder_sklearn.categories_):
            self.assertEqual(sklearn_cats.tolist(), pandas_cats.tolist())

        # Validate that transformer state is equivalent whether fitted with pandas or Snowpark DataFrame.
        attrs: Dict[str, _EqualityFunc] = {
            "categories_": equal_list_of(equal_np_array),
            "drop_idx_": equal_np_array,
            "_n_features_outs": equal_default,
            "_dense_output_cols": equal_default,
            "_infrequent_indices": equal_list_of(equal_np_array),
            "_default_to_infrequent_mappings": equal_list_of(equal_optional_of(equal_np_array)),
            "_state_pandas": equal_pandas_df_ignore_row_order,
        }

        mismatched_attributes: Dict[str, Tuple[Any, Any]] = {}
        for attr, equality_func in attrs.items():
            attr1 = getattr(encoder1, attr, None)
            attr2 = getattr(encoder2, attr, None)
            if not equal_optional_of(equality_func)(attr1, attr2):
                mismatched_attributes[attr] = (attr1, attr2)

        if mismatched_attributes:
            raise AssertionError(
                f"Attributes\n{mismatched_attributes}\ndo not match on the transformers fitted with "
                f"a snowpark.DataFrame and a pd.DataFrame."
            )

    def test_fit_pandas_bad_input_cols(self) -> None:
        input_cols = CATEGORICAL_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Fit with Pandas DF
        encoder = OneHotEncoder().set_input_cols(input_cols + ["nonexistent_column"])
        with self.assertRaises(KeyError):
            encoder.fit(df_pandas)

    def test_transform_dense(self) -> None:
        """
        Verify dense transformed results.

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
        -----------------------
        |"STR1"  |"STR2"      |
        -----------------------
        |c       |g1ehQlL80t  |
        |a       |zOyDvcyZ2s  |
        |b       |zOyDvcyZ2s  |
        |A       |TuDOcLxToB  |
        |d       |g1ehQlL80t  |
        |b       |g1ehQlL80t  |
        |b       |g1ehQlL80t  |
        -----------------------

        Transformed SQL query:
        SELECT "STR1", "STR2", "ID", "OUTPUT1_A", "OUTPUT1_a", "OUTPUT1_b", "OUTPUT1_c", "OUTPUT1_d",
        "OUTPUT2_TuDOcLxToB", "OUTPUT2_g1ehQlL80t", "OUTPUT2_zOyDvcyZ2s"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "ID" AS "ID", "OUTPUT1_A" AS "OUTPUT1_A",
        "OUTPUT1_a" AS "OUTPUT1_a", "OUTPUT1_b" AS "OUTPUT1_b", "OUTPUT1_c" AS "OUTPUT1_c", "OUTPUT1_d" AS "OUTPUT1_d"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "ID" AS "ID" FROM <TEMP_TABLE>)
        AS SNOWPARK_LEFT LEFT OUTER JOIN ( SELECT "OUTPUT1_A" AS "OUTPUT1_A", "OUTPUT1_a" AS "OUTPUT1_a",
        "OUTPUT1_b" AS "OUTPUT1_b", "OUTPUT1_c" AS "OUTPUT1_c", "OUTPUT1_d" AS "OUTPUT1_d", "_CATEGORY" AS "_CATEGORY"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE ("_COLUMN_NAME" = 'STR1'))) AS SNOWPARK_RIGHT
        ON EQUAL_NULL("STR1", "_CATEGORY")))) AS SNOWPARK_LEFT
        LEFT OUTER JOIN ( SELECT "OUTPUT2_TuDOcLxToB" AS "OUTPUT2_TuDOcLxToB",
        "OUTPUT2_g1ehQlL80t" AS "OUTPUT2_g1ehQlL80t", "OUTPUT2_zOyDvcyZ2s" AS "OUTPUT2_zOyDvcyZ2s",
        "_CATEGORY" AS "_CATEGORY" FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE ("_COLUMN_NAME" = 'STR2'))) AS SNOWPARK_RIGHT
        ON EQUAL_NULL("STR2", "_CATEGORY")))

        Transformed dataset:
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------ # noqa
        |"STR1"  |"STR2"      |"OUTPUT1_A"  |"OUTPUT1_a"  |"OUTPUT1_b"  |"OUTPUT1_c"  |"OUTPUT1_d"  |"OUTPUT2_TuDOcLxToB"  |"OUTPUT2_g1ehQlL80t"  |"OUTPUT2_zOyDvcyZ2s"  | # noqa
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------ # noqa
        |c       |g1ehQlL80t  |0.0          |0.0          |0.0          |1.0          |0.0          |0.0                   |1.0                   |0.0                   | # noqa
        |a       |zOyDvcyZ2s  |0.0          |1.0          |0.0          |0.0          |0.0          |0.0                   |0.0                   |1.0                   | # noqa
        |b       |zOyDvcyZ2s  |0.0          |0.0          |1.0          |0.0          |0.0          |0.0                   |0.0                   |1.0                   | # noqa
        |A       |TuDOcLxToB  |1.0          |0.0          |0.0          |0.0          |0.0          |1.0                   |0.0                   |0.0                   | # noqa
        |d       |g1ehQlL80t  |0.0          |0.0          |0.0          |0.0          |1.0          |0.0                   |1.0                   |0.0                   | # noqa
        |b       |g1ehQlL80t  |0.0          |0.0          |1.0          |0.0          |0.0          |0.0                   |1.0                   |0.0                   | # noqa
        |b       |g1ehQlL80t  |0.0          |0.0          |1.0          |0.0          |0.0          |0.0                   |1.0                   |0.0                   | # noqa
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------ # noqa

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False

        # Fit with snowpark dataframe
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # Fit with pandas dataframe, transformed on snowpark dataframe
        encoder2 = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder2.fit(df_pandas)

        transformed_df = encoder2.transform(df[input_cols_extended])
        actual_arr2 = transformed_df.sort(id_col)[encoder2.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)
        np.testing.assert_allclose(actual_arr2, sklearn_arr)

    def test_transform_sparse(self) -> None:
        """
        Verify sparse transformed results.

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
        -----------------------
        |"STR1"  |"STR2"      |
        -----------------------
        |c       |g1ehQlL80t  |
        |a       |zOyDvcyZ2s  |
        |b       |zOyDvcyZ2s  |
        |A       |TuDOcLxToB  |
        |d       |g1ehQlL80t  |
        |b       |g1ehQlL80t  |
        |b       |g1ehQlL80t  |
        -----------------------

        Transformed SQL query:
        SELECT "STR1", "STR2", "ID", "OUTPUT1", "OUTPUT2"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "ID" AS "ID", "OUTPUT1" AS "OUTPUT1"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "ID" AS "ID" FROM <TEMP_TABLE>)
        AS SNOWPARK_LEFT LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT1" AS "OUTPUT1"
        FROM ( SELECT "_CATEGORY", "_ENCODED_VALUE" AS "OUTPUT1"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE ("_COLUMN_NAME" = 'STR1')))) AS SNOWPARK_RIGHT
        ON EQUAL_NULL("STR1", "_CATEGORY")))) AS SNOWPARK_LEFT
        LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT2" AS "OUTPUT2"
        FROM ( SELECT "_CATEGORY", "_ENCODED_VALUE" AS "OUTPUT2"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE ("_COLUMN_NAME" = 'STR2')))) AS SNOWPARK_RIGHT
        ON EQUAL_NULL("STR2", "_CATEGORY")))

        Transformed dataset:
        -------------------------------------------------------------------
        |"STR1"  |"STR2"      |"OUTPUT1"            |"OUTPUT2"            |
        -------------------------------------------------------------------
        |c       |g1ehQlL80t  |{                    |{                    |
        |        |            |  "3": 1,            |  "1": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        |a       |zOyDvcyZ2s  |{                    |{                    |
        |        |            |  "1": 1,            |  "2": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        |b       |zOyDvcyZ2s  |{                    |{                    |
        |        |            |  "2": 1,            |  "2": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        |A       |TuDOcLxToB  |{                    |{                    |
        |        |            |  "0": 1,            |  "0": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        |d       |g1ehQlL80t  |{                    |{                    |
        |        |            |  "4": 1,            |  "1": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        |b       |g1ehQlL80t  |{                    |{                    |
        |        |            |  "2": 1,            |  "1": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        |b       |g1ehQlL80t  |{                    |{                    |
        |        |            |  "2": 1,            |  "1": 1,            |
        |        |            |  "array_length": 5  |  "array_length": 3  |
        |        |            |}                    |}                    |
        -------------------------------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = True
        # Fit with snowpark dataframe
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # Fit with pandas dataframe, transform with snowpark dataframe
        encoder2 = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder2.fit(df_pandas)

        transformed_df = encoder2.transform(df[input_cols_extended])
        actual_arr2 = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))
        self.assertTrue(self.compare_sparse_transform_results(actual_arr2, sklearn_arr))

        # assert array length
        output_pandas = transformed_df[output_cols].to_pandas().applymap(lambda x: json.loads(x) if x else None)
        for idx, n_features_out in enumerate(encoder_sklearn._n_features_outs):
            array_length = output_pandas.iloc[0, idx]["array_length"]
            self.assertEqual(array_length, n_features_out)

        # loading into memory with `to_pandas_with_sparse`
        df_pandas_output = utils_sparse.to_pandas_with_sparse(transformed_df.sort(id_col)[output_cols], output_cols)
        np.testing.assert_allclose(df_pandas_output.to_numpy(), sklearn_arr.toarray())

    def test_transform_null_dense(self) -> None:
        """
        Verify dense transformed results when the NULL category exists.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_null_sparse(self) -> None:
        """
        Verify sparse transformed results when the NULL category exists.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = True
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

    def test_transform_boolean_dense(self) -> None:
        """
        Verify dense transformed results on boolean categories.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = BOOLEAN_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_BOOLEAN, SCHEMA_BOOLEAN, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_boolean_sparse(self) -> None:
        """
        Verify sparse transformed results on boolean categories.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = BOOLEAN_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_BOOLEAN, SCHEMA_BOOLEAN, np.nan)

        sparse = True
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

    def test_transform_numeric_dense(self) -> None:
        """
        Verify dense transformed results on numeric categories.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        # compare the frequency of each category
        np.testing.assert_allclose(
            np.sort(actual_arr.sum(axis=0)),
            np.sort(sklearn_arr.sum(axis=0)),
        )

    def test_transform_numeric_sparse(self) -> None:
        """
        Verify sparse transformed results on numeric categories.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = True
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])
        sklearn_arr_dense = sklearn_arr.toarray()

        # convert the array of coordinates to a csr_matrix and then to a dense array
        row, col, data = [], [], []
        for vals in actual_arr:
            row.append(vals[0])
            col.append(vals[1])
            data.append(1)
        actual_arr_dense = csr_matrix((data, (row, col)), shape=sklearn_arr_dense.shape).toarray()

        # compare the frequency of each category
        np.testing.assert_allclose(
            np.sort(actual_arr_dense.sum(axis=0)),
            np.sort(sklearn_arr_dense.sum(axis=0)),
        )

    def test_transform_quotes_dense(self) -> None:
        """
        Verify dense transformed results on categories with single and double quotes.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        id_col = ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, _DATA_QUOTES, SCHEMA_BOOLEAN, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_categories(self) -> None:
        """
        Test that `categories` is provided.

        Raises
        ------
        AssertionError
            If the fitted categories do not match those of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        categories = {}
        categories_list = []
        for idx, input_col in enumerate(input_cols):
            cats = list(set(framework_utils.get_pandas_feature(df_pandas[input_cols], feature_idx=idx)))
            # test excessive given categories
            cats.append(f"extra_cat_{input_col}")
            # sort categories in descending order to test if orders are preserved
            cats.sort(reverse=True)
            cats_arr = np.array(cats)
            categories[input_col] = cats_arr
            categories_list.append(cats_arr)

        sparse = False
        encoder = (
            OneHotEncoder(sparse=sparse, categories=categories).set_input_cols(input_cols).set_output_cols(output_cols)
        )
        encoder.fit(df)
        actual_categories = encoder._categories_list

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, categories=categories_list)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        # verify all given categories including excessive ones are in `encoder._state_pandas`
        category_col = [c for c in encoder._state_pandas.columns if "_CATEGORY" in c][0]
        actual_state_categories = encoder._state_pandas.groupby("_COLUMN_NAME")[category_col].apply(np.array).to_dict()
        self.assertEqual(set(categories.keys()), set(actual_state_categories.keys()))
        for key in categories.keys():
            self.assertEqual(set(categories[key]), set(actual_state_categories[key]))

        for actual_cats, sklearn_cats in zip(actual_categories, encoder_sklearn.categories_):
            self.assertEqual(sklearn_cats.tolist(), actual_cats.tolist())
        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_categories_unknown(self) -> None:
        """
        Test that `categories` is provided and unknown categories exist.

        Raises
        ------
        AssertionError
            If no error on unknown categories is raised during fit.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols = CATEGORICAL_COLS
        df_pandas, _ = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        categories = {}
        for idx, input_col in enumerate(input_cols):
            cats = set(framework_utils.get_pandas_feature(df_pandas[input_cols], feature_idx=idx))
            categories[input_col] = np.array(cats)

        encoder = OneHotEncoder(categories=categories).set_input_cols(input_cols)

        with pytest.raises(ValueError) as excinfo:
            encoder.fit(unknown_df)

        self.assertIn("Found unknown categories during fit", excinfo.value.args[0])

    def test_drop_first(self) -> None:
        """
        Test `drop="first"`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="ignore", drop="first")
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, handle_unknown="ignore", drop="first")
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_drop_if_binary(self) -> None:
        """
        Test `drop="if_binary"`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="ignore", drop="if_binary")
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, handle_unknown="ignore", drop="if_binary")
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_drop_idx_infrequent_categories(self) -> None:
        """
        Test drop_idx is defined correctly with infrequent categories.

        Raises
        ------
        AssertionError
            - If drop_idx does not match that of the sklearn encoder.
            - If the transformed output does not match that of the sklearn encoder.
        """
        schema = ["COL1"]
        input_cols, output_cols = ["COL1"], ["OUT1"]
        sparse = False

        data = np.array([["a"] * 2 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4]).T.tolist()
        df_pandas, df = framework_utils.get_df(self._session, data, schema)
        for drop in ["first", ["d"]]:
            # sklearn
            encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=4, handle_unknown="ignore", drop=drop)
            encoder_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=input_cols)[input_cols])
            for _df in [df, df_pandas]:
                encoder = (
                    OneHotEncoder(sparse=sparse, min_frequency=4, handle_unknown="ignore", drop=drop)
                    .set_input_cols(input_cols)
                    .set_output_cols(output_cols)
                )
                encoder.fit(_df)
                transformed_df = encoder.transform(_df)
                if isinstance(transformed_df, DataFrame):
                    actual_arr = transformed_df.sort(input_cols)[encoder.get_output_cols()].to_pandas().to_numpy()
                else:
                    actual_arr = transformed_df[encoder.get_output_cols()].to_numpy()

                self.assertEqual(
                    encoder_sklearn.categories_[0][encoder_sklearn.drop_idx_[0]],
                    encoder.categories_[input_cols[0]][encoder.drop_idx_[0]],
                )
                np.testing.assert_allclose(actual_arr, sklearn_arr)

        data = np.array([["a"] * 2 + ["b"] * 2 + ["c"] * 10]).T.tolist()
        df_pandas, df = framework_utils.get_df(self._session, data, schema)
        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(
            sparse=sparse, min_frequency=4, handle_unknown="ignore", drop="if_binary"
        )
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=input_cols)[input_cols])
        for _df in [df, df_pandas]:
            encoder = (
                OneHotEncoder(sparse=sparse, min_frequency=4, handle_unknown="ignore", drop="if_binary")
                .set_input_cols(input_cols)
                .set_output_cols(output_cols)
            )
            encoder.fit(_df)
            transformed_df = encoder.transform(_df)
            if isinstance(transformed_df, DataFrame):
                actual_arr = transformed_df.sort(input_cols)[encoder.get_output_cols()].to_pandas().to_numpy()
            else:
                actual_arr = transformed_df[encoder.get_output_cols()].to_numpy()

            self.assertEqual(
                encoder_sklearn.categories_[0][encoder_sklearn.drop_idx_[0]],
                encoder.categories_[input_cols[0]][encoder.drop_idx_[0]],
            )
            np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_handle_unknown_error(self) -> None:
        """
        Test `handle_unknown="error"`.

        Raises
        ------
        AssertionError
            If no error on unknown categories is raised during transform.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        # sparse
        sparse = True
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="error").set_input_cols(input_cols).set_output_cols(output_cols)
        )
        encoder.fit(df)

        with pytest.raises(ValueError) as excinfo:
            encoder.transform(unknown_df).collect()

        self.assertIn("Found unknown categories during transform", excinfo.value.args[0])

        # dense
        sparse = False
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="error").set_input_cols(input_cols).set_output_cols(output_cols)
        )
        encoder.fit(df)

        with pytest.raises(ValueError) as excinfo:
            encoder.transform(unknown_df).collect()

        self.assertIn("Found unknown categories during transform", excinfo.value.args[0])

    def test_handle_unknown_ignore_dense(self) -> None:
        """
        Test dense `handle_unknown="ignore"`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="ignore")
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        transformed_df = encoder.transform(unknown_df)
        actual_arr = transformed_df.sort(input_cols[0])[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, handle_unknown="ignore")
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(unknown_pandas.sort_values(by=[input_cols[0]])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_handle_unknown_ignore_sparse(self) -> None:
        """
        Test sparse `handle_unknown="ignore"`.

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
        -----------------------
        |"STR1"  |"STR2"      |
        -----------------------
        |z       |zOyDvcyZ2s  |
        |a       |g1ehQlL80t  |
        |b       |g1ehQlL80t  |
        -----------------------

        Transformed SQL query:
        SELECT "STR1", "STR2", "OUTPUT1", "OUTPUT2"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "OUTPUT1" AS "OUTPUT1"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2" FROM <TEMP_TABLE>) AS SNOWPARK_LEFT
        LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT1" AS "OUTPUT1"
        FROM ( SELECT "_CATEGORY", "_ENCODED_VALUE" AS "OUTPUT1"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE ("_COLUMN_NAME" = 'STR1')))) AS SNOWPARK_RIGHT
        ON EQUAL_NULL("STR1", "_CATEGORY")))) AS SNOWPARK_LEFT
        LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT2" AS "OUTPUT2"
        FROM ( SELECT "_CATEGORY", "_ENCODED_VALUE" AS "OUTPUT2"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE ("_COLUMN_NAME" = 'STR2')))) AS SNOWPARK_RIGHT
        ON EQUAL_NULL("STR2", "_CATEGORY")))

        Transformed dataset:
        -----------------------------------------------
        |"STR1"  |"STR2"      |"OUTPUT1"  |"OUTPUT2"  |
        -----------------------------------------------
        |a       |g1ehQlL80t  |{          |{          |
        |        |            |  "1": 1   |  "1": 1   |
        |        |            |}          |}          |
        |b       |g1ehQlL80t  |{          |{          |
        |        |            |  "2": 1   |  "1": 1   |
        |        |            |}          |}          |
        |z       |zOyDvcyZ2s  |NULL       |{          |
        |        |            |           |  "2": 1   |
        |        |            |           |}          |
        -----------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = True
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="ignore")
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        transformed_df = encoder.transform(unknown_df)
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, input_cols[0])

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, handle_unknown="ignore")
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(unknown_pandas.sort_values(by=[input_cols[0]])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

    # TODO(hayu): [SNOW-752263] Support OneHotEncoder handle_unknown="infrequent_if_exist".
    #  Add back when `handle_unknown="infrequent_if_exist"` is supported.
    # def test_handle_unknown_infrequent_if_exist_dense(self):
    #     """
    #     Test dense `handle_unknown="infrequent_if_exist"` with `min_frequency` set.

    #     Raises
    #     ------
    #     AssertionError
    #         If the transformed output does not match that of the sklearn encoder.
    #     """
    #     values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
    #     input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
    #     df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

    #     sparse = False
    #     encoder = (
    #         OneHotEncoder(sparse=sparse, handle_unknown="infrequent_if_exist", min_frequency=2)
    #         .set_input_cols(input_cols)
    #         .set_output_cols(output_cols)
    #     )
    #     encoder.fit(df)

    #     unknown_data = list(zip(*values_list))
    #     unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
    #     unknown_df = self._session.create_dataframe(unknown_pandas)

    #     transformed_df = encoder.transform(unknown_df)
    #     actual_arr = transformed_df.sort(input_cols[0])[encoder.get_output_cols()].to_pandas().to_numpy()

    #     # sklearn
    #     encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, handle_unknown="infrequent_if_exist", min_frequency=2)
    #     encoder_sklearn.fit(df_pandas[input_cols])
    #     sklearn_arr = encoder_sklearn.transform(unknown_pandas.sort_values(by=[input_cols[0]])[input_cols])

    #     np.testing.assert_allclose(actual_arr, sklearn_arr)

    # TODO(hayu): [SNOW-752263] Support OneHotEncoder handle_unknown="infrequent_if_exist".
    #  Add back when `handle_unknown="infrequent_if_exist"` is supported.
    # def test_handle_unknown_infrequent_if_exist_sparse(self):
    #     """
    #     Test sparse `handle_unknown="infrequent_if_exist"` with `min_frequency` set.

    #     Raises
    #     ------
    #     AssertionError
    #         If the transformed output does not match that of the sklearn encoder.
    #     """
    #     values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
    #     input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
    #     df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

    #     sparse = True
    #     encoder = (
    #         OneHotEncoder(sparse=sparse, handle_unknown="infrequent_if_exist", min_frequency=2)
    #         .set_input_cols(input_cols)
    #         .set_output_cols(output_cols)
    #     )
    #     encoder.fit(df)

    #     unknown_data = list(zip(*values_list))
    #     unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
    #     unknown_df = self._session.create_dataframe(unknown_pandas)

    #     transformed_df = encoder.transform(unknown_df)
    #     actual_arr = self.convert_sparse_df_to_arr(
    #         transformed_df, output_cols, input_cols[0]
    #     )

    #     # sklearn
    #     encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, handle_unknown="infrequent_if_exist", min_frequency=2)
    #     encoder_sklearn.fit(df_pandas[input_cols])
    #     sklearn_arr = encoder_sklearn.transform(unknown_pandas.sort_values(by=[input_cols[0]])[input_cols])

    #     assert self.compare_sparse_transform_results(actual_arr, sklearn_arr)

    def test_min_frequency_dense(self) -> None:
        """
        Test dense `min_frequency`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False

        # min_frequency: numbers.Integral
        min_frequency_int: int = 2
        encoder = (
            OneHotEncoder(sparse=sparse, min_frequency=min_frequency_int)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=min_frequency_int)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

        # min_frequency: numbers.Real
        min_frequency_float: float = 0.3
        encoder = (
            OneHotEncoder(sparse=sparse, min_frequency=min_frequency_float)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=min_frequency_float)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_min_frequency_sparse(self) -> None:
        """
        Test sparse `min_frequency`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = True

        # min_frequency: numbers.Integral
        min_frequency_int: int = 2
        encoder = (
            OneHotEncoder(sparse=sparse, min_frequency=min_frequency_int)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=min_frequency_int)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

        # min_frequency: numbers.Real
        min_frequency_float: float = 0.3
        encoder = (
            OneHotEncoder(sparse=sparse, min_frequency=min_frequency_float)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=min_frequency_float)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

    def test_min_frequency_null_dense(self) -> None:
        """
        Test dense `min_frequency` when the NULL category exists.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse, min_frequency=2).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=2)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_min_frequency_null_sparse(self) -> None:
        """
        Test sparse `min_frequency` when the NULL category exists.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = True
        encoder = OneHotEncoder(sparse=sparse, min_frequency=2).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, min_frequency=2)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

    def test_max_categories_dense(self) -> None:
        """
        Test dense `max_categories`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse, max_categories=2).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[encoder.get_output_cols()].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, max_categories=2)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_max_categories_sparse(self) -> None:
        """
        Test sparse `max_categories`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = True
        encoder = OneHotEncoder(sparse=sparse, max_categories=2).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = self.convert_sparse_df_to_arr(transformed_df, output_cols, id_col)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse, max_categories=2)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        self.assertTrue(self.compare_sparse_transform_results(actual_arr, sklearn_arr))

    def test_transform_pandas_dense(self) -> None:
        """
        Verify dense transformed results on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df_pandas)
        actual_arr = transformed_df[encoder.get_output_cols()].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_pandas_sparse(self) -> None:
        """
        Verify sparse transformed results on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        sparse = True
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_matrix = encoder.transform(df_pandas)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_matrix = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(transformed_matrix.toarray(), sklearn_matrix.toarray())

    def test_transform_null_pandas_dense(self) -> None:
        """
        Verify dense transformed results on pandas dataframe when the NULL category exists.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)
        converted_pandas = df.to_pandas()

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(converted_pandas)
        actual_arr = transformed_df[encoder.get_output_cols()].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_transform_null_pandas_sparse(self) -> None:
        """
        Verify sparse transformed results on pandas dataframe when the NULL category exists.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)
        converted_pandas = df.to_pandas()

        sparse = True
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_matrix = encoder.transform(converted_pandas)

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_matrix = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(transformed_matrix.toarray(), sklearn_matrix.toarray())

    def test_fit_transform_null_pandas(self) -> None:
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df_pandas)

        transformed_df = encoder.transform(df_pandas)
        actual_arr = transformed_df[encoder.get_output_cols()].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr)

    def test_handle_unknown_error_pandas(self) -> None:
        """
        Test `handle_unknown="error"` on pandas dataframe.

        Raises
        ------
        AssertionError
            If no error on unknown categories is raised during transform.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)
        converted_unknown_pandas = unknown_df.to_pandas()

        # sparse
        sparse = True
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="error").set_input_cols(input_cols).set_output_cols(output_cols)
        )
        encoder.fit(df)

        with pytest.raises(ValueError) as excinfo:
            encoder.transform(converted_unknown_pandas)

        self.assertIn("Found unknown categories", excinfo.value.args[0])

        # dense
        sparse = False
        encoder = (
            OneHotEncoder(sparse=sparse, handle_unknown="error").set_input_cols(input_cols).set_output_cols(output_cols)
        )
        encoder.fit(df)

        with pytest.raises(ValueError) as excinfo:
            encoder.transform(converted_unknown_pandas)

        self.assertIn("Found unknown categories", excinfo.value.args[0])

    @parameterized.parameters(  # type: ignore
        {"params": {"sparse": False}},
        {"params": {"sparse": False, "min_frequency": 2}},
        {"params": {"sparse": False, "max_categories": 2}},
    )
    def test_get_output_cols_dense(self, params: Dict[str, Any]) -> None:
        """
        Test dense output columns getter.

        Args:
            params: A parameter combination for the encoder's constructor.

        Raises
        ------
        AssertionError
            If the getter returns incorrect dense output columns.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = OneHotEncoder(**params).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)
        expected_output_cols = []
        for input_col in input_cols:
            expected_output_cols.extend(
                [identifier.get_inferred_name(col) for col in encoder._dense_output_cols_mappings[input_col]]
            )

        # output columns are set before fitting
        # fit Snowpark dataframe
        encoder11 = OneHotEncoder(**params).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder11.fit(df)
        self.assertListEqual(encoder11.get_output_cols(), expected_output_cols)

        # fit pandas dataframe
        encoder12 = OneHotEncoder(**params).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder12.fit(df_pandas)
        self.assertListEqual(encoder12.get_output_cols(), expected_output_cols)

        # output columns are unset before fitting
        # fit and transform Snowpark dataframe
        encoder21 = OneHotEncoder(**params).set_input_cols(input_cols)
        encoder21.fit(df)
        self.assertListEqual(encoder21.get_output_cols(), [])
        encoder21.set_output_cols(output_cols).transform(df)
        self.assertListEqual(encoder21.get_output_cols(), expected_output_cols)

        # fit and transform pandas dataframe
        encoder22 = OneHotEncoder(**params).set_input_cols(input_cols)
        encoder22.fit(df_pandas)
        self.assertListEqual(encoder22.get_output_cols(), [])
        encoder22.set_output_cols(output_cols).transform(df_pandas)
        self.assertListEqual(encoder22.get_output_cols(), expected_output_cols)

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        data, schema = DATA_NONE_NAN, SCHEMA
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        sparse = False
        encoder = OneHotEncoder(sparse=sparse).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df1)
        filepath = os.path.join(tempfile.gettempdir(), "test_one_hot_encoder.pkl")
        self._to_be_deleted_files.append(filepath)
        encoder_dump_cloudpickle = cloudpickle.dumps(encoder)
        encoder_dump_pickle = pickle.dumps(encoder)
        joblib.dump(encoder, filepath)

        self._session.close()

        # transform in session 2
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)

        importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.one_hot_encoder"])

        # cloudpickle
        encoder_load_cloudpickle = cloudpickle.loads(encoder_dump_cloudpickle)
        transformed_df_cloudpickle = encoder_load_cloudpickle.transform(df2[input_cols_extended])
        actual_arr_cloudpickle = (
            transformed_df_cloudpickle.sort(id_col)[encoder_load_cloudpickle.get_output_cols()].to_pandas().to_numpy()
        )

        # pickle
        encoder_load_pickle = pickle.loads(encoder_dump_pickle)
        transformed_df_pickle = encoder_load_pickle.transform(df2[input_cols_extended])
        actual_arr_pickle = (
            transformed_df_pickle.sort(id_col)[encoder_load_pickle.get_output_cols()].to_pandas().to_numpy()
        )

        # joblib
        encoder_load_joblib = joblib.load(filepath)
        transformed_df_joblib = encoder_load_joblib.transform(df2[input_cols_extended])
        actual_arr_joblib = (
            transformed_df_joblib.sort(id_col)[encoder_load_joblib.get_output_cols()].to_pandas().to_numpy()
        )

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder(sparse=sparse)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
        np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
        np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)

    def test_drop_input_cols(self) -> None:
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        input_cols, output_cols = CATEGORICAL_COLS, CATEGORICAL_COLS

        encoder = OneHotEncoder(input_cols=input_cols, output_cols=output_cols, drop_input_cols=True)
        transformed_df = encoder.fit(df).transform(df)

        self.assertEqual(0, len(set(input_cols) & set(transformed_df.to_pandas().columns)))

    def test_fit_empty(self) -> None:
        data = [[]]
        df = self._session.create_dataframe(data, ["CAT"]).na.drop()

        encoder = OneHotEncoder(input_cols=["CAT"], output_cols=["CAT"], drop_input_cols=True)
        with self.assertRaises(ValueError) as ex:
            encoder.fit(df)
        self.assertIn("Empty data while a minimum of 1 sample is required.", str(ex.exception))

    def test_fit_snowpark_transform_numeric_data(self) -> None:
        snow_df = self._session.sql(
            """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
            FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS
            LIMIT 2000"""
        ).drop("Y")
        input_cols = [c for c in snow_df.columns if c != "LABEL"]
        # contains dtype as int, object, float.
        output_cols = [f"OHE_{c}" for c in input_cols]

        snow_df_single_feature = snow_df[input_cols]
        ohe = OneHotEncoder(input_cols=input_cols, output_cols=output_cols)
        ohe.fit(snow_df_single_feature)
        ohe.transform(snow_df_single_feature.to_pandas())

    def test_fit_snowpark_transform_everydtypes(self) -> None:
        x = np.ones(
            (10,),
            dtype=[
                ("X", np.uint8),
                ("Y", np.float64),
                ("Z", np.str_),
                ("A", np.bool8),
                ("B", np.bytes0),
                ("C", np.object0),
            ],
        )
        pd_df = pd.DataFrame(x)
        df = self._session.create_dataframe(pd_df)
        input_cols = ["A", "B", "C", "X", "Y", "Z"]
        output_cols = [f"OHE_{c}" for c in input_cols]

        ohe = OneHotEncoder(input_cols=input_cols, output_cols=output_cols)
        ohe.fit(df)
        actual_arr = ohe.transform(pd_df)[ohe.get_output_cols()].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOneHotEncoder()
        encoder_sklearn.fit(pd_df[input_cols])
        sklearn_arr = encoder_sklearn.transform(pd_df[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr.toarray())

    def test_identical_snowpark_vs_pandas_output_column_names(self) -> None:
        snow_df = self._session.sql(
            """SELECT *, IFF(Y = 'yes', 1.0, 0.0) as LABEL
            FROM ML_DATASETS.PUBLIC.UCI_BANK_MARKETING_20COLUMNS
            LIMIT 1000"""
        ).drop("Y")
        pd_df = snow_df.to_pandas()
        cols = [
            "AGE",
            "CAMPAIGN",
            "CONTACT",
            "DAY_OF_WEEK",
            "EDUCATION",
            "JOB",
            "MONTH",
            "DURATION",
        ]

        ohe = OneHotEncoder(input_cols=cols, output_cols=cols).fit(snow_df)
        snow_cols = ohe.transform(snow_df).columns
        pd_cols = ohe.transform(pd_df).columns.tolist()

        self.assertCountEqual(snow_cols, pd_cols)


if __name__ == "__main__":
    main()
