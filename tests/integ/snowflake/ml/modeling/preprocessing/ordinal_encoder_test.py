#!/usr/bin/env python3
import importlib
import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder

from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    OrdinalEncoder,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
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
)


class OrdinalEncoderTest(parameterized.TestCase):
    """Test OrdinalEncoder."""

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
        Verify fitted categories.

        Raises
        ------
        AssertionError
            If the fitted categories do not match those of the sklearn encoder.
        """
        input_cols = CATEGORICAL_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = OrdinalEncoder().set_input_cols(input_cols)
        encoder.fit(df)

        actual_categories = encoder._categories_list

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])

        for actual_cats, sklearn_cats in zip(actual_categories, encoder_sklearn.categories_):
            self.assertEqual(sklearn_cats.tolist(), actual_cats.tolist())

    def test_fit_index_consecutiveness(self) -> None:
        """
        Verify if fitted indices are consecutive.

        Raises
        ------
        AssertionError
            If the fitted indices are not consecutive.
        """
        input_col = NUMERIC_COLS[0]
        seed, rowcount = 0, 10 * 1000 * 1000
        df = self._session.sql(
            f"select uniform(1, 100, random({seed})) as {input_col} " f"from table(generator(rowcount => {rowcount}))"
        )

        encoder = OrdinalEncoder().set_input_cols(input_col)
        encoder.fit(df)

        max_index = encoder._state_pandas["_INDEX"].max()
        distinct_count = df[[input_col]].distinct().count()
        self.assertEqual(distinct_count - 1, max_index)

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {}},
        {"params": {"handle_unknown": "use_encoded_value", "unknown_value": -1}},
        {"params": {"encoded_missing_value": -1}},
    )
    def test_fit_pandas(self, params: Dict[str, Any]) -> None:
        """
        Verify that:
            (1) an encoder fit on a pandas dataframe matches its sklearn counterpart on `categories_`, and
            (2) the encoder fit on a pandas dataframe matches the encoder fit on a Snowpark DataFrame on all
                relevant stateful attributes. (Note: This does not check for a _vocab_table_name match.)

        Args:
            params: A parameter combination for the encoder's constructor.

        Raises:
            AssertionError: If either condition does not hold.
        """

        input_cols = CATEGORICAL_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        # Fit with SnowPark DF
        encoder1 = OrdinalEncoder(**params).set_input_cols(input_cols)
        encoder1.fit(df)

        # Fit with Pandas DF
        encoder2 = OrdinalEncoder(**params).set_input_cols(input_cols)
        encoder2.fit(df_pandas)

        # Fit with an SKLearn encoder
        encoder_sklearn = SklearnOrdinalEncoder(**params)
        encoder_sklearn.fit(df_pandas[input_cols])

        # Validate that SnowML transformer state is equivalent to SKLearn transformer state
        for pandas_cats, sklearn_cats in zip(encoder2._categories_list, encoder_sklearn.categories_):
            self.assertEqual(sklearn_cats.tolist(), pandas_cats.tolist())

        # Validate that transformer state is equivalent whether fitted with pandas or Snowpark DataFrame.
        attrs: Dict[str, _EqualityFunc] = {
            "categories_": equal_list_of(equal_np_array),
            "_missing_indices": equal_default,
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
        FROM ( SELECT "_CATEGORY", "_INDEX" AS "OUTPUT1"
        FROM ( SELECT  *  FROM (( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NOT NULL)
        UNION ( SELECT "_COLUMN_NAME", "_CATEGORY", 'NAN' :: FLOAT AS "_INDEX"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NULL)))
        WHERE ("_COLUMN_NAME" = 'STR1')))) AS SNOWPARK_RIGHT ON EQUAL_NULL("STR1", "_CATEGORY")))) AS SNOWPARK_LEFT
        LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT2" AS "OUTPUT2"
        FROM ( SELECT "_CATEGORY", "_INDEX" AS "OUTPUT2"
        FROM ( SELECT  *  FROM (( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NOT NULL)
        UNION ( SELECT "_COLUMN_NAME", "_CATEGORY", 'NAN' :: FLOAT AS "_INDEX"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NULL)))
        WHERE ("_COLUMN_NAME" = 'STR2')))) AS SNOWPARK_RIGHT ON EQUAL_NULL("STR2", "_CATEGORY")))

        Transformed dataset:
        -----------------------------------------------
        |"STR1"  |"STR2"      |"OUTPUT1"  |"OUTPUT2"  |
        -----------------------------------------------
        |c       |g1ehQlL80t  |0.0        |0.0        |
        |a       |zOyDvcyZ2s  |1.0        |1.0        |
        |b       |zOyDvcyZ2s  |2.0        |1.0        |
        |A       |TuDOcLxToB  |4.0        |2.0        |
        |d       |g1ehQlL80t  |3.0        |0.0        |
        |b       |g1ehQlL80t  |2.0        |0.0        |
        |b       |g1ehQlL80t  |2.0        |0.0        |
        -----------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Fit with snowpark dataframe, transformed on snowpark dataframe
        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # Fit with pandas dataframe, transformed on snowpark dataframe
        encoder2 = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder2.fit(df_pandas)

        transformed_df = encoder2.transform(df[input_cols_extended])
        actual_arr2 = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)
        np.testing.assert_allclose(actual_arr2, sklearn_arr, equal_nan=True)

    def test_transform_null(self) -> None:
        """
        Verify transformed results when the NULL category exists.

        Fitted dataset:
        --------------------------------------------------------
        |"ID"  |"STR1"  |"STR2"      |"FLOAT1"  |"FLOAT2"      |
        --------------------------------------------------------
        |1     |a       |g1ehQlL80t  |-1.0      |0.0           |
        |2     |a       |zOyDvcyZ2s  |NULL      |NULL          |
        |3     |b       |NULL        |2.0       |2253463.5342  |
        |4     |A       |TuDOcLxToB  |NULL      |-1350.6407    |
        |5     |NULL    |g1ehQlL80t  |2.5       |-1.0          |
        |6     |NULL    |g1ehQlL80t  |4.0       |NULL          |
        |7     |b       |TuDOcLxToB  |-10.0     |-12.564       |
        --------------------------------------------------------

        Source SQL query:
        SELECT  *  FROM TEMP_TABLE

        Dataset to transform:
        -----------------------
        |"STR1"  |"STR2"      |
        -----------------------
        |a       |g1ehQlL80t  |
        |a       |zOyDvcyZ2s  |
        |b       |NULL        |
        |A       |TuDOcLxToB  |
        |NULL    |g1ehQlL80t  |
        |NULL    |g1ehQlL80t  |
        |b       |TuDOcLxToB  |
        -----------------------

        Transformed SQL query:
        SELECT "STR1", "STR2", "ID", "OUTPUT1", "OUTPUT2"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "ID" AS "ID", "OUTPUT1" AS "OUTPUT1"
        FROM ( SELECT  *  FROM (( SELECT "STR1" AS "STR1", "STR2" AS "STR2", "ID" AS "ID" FROM <TEMP_TABLE>)
        AS SNOWPARK_LEFT LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT1" AS "OUTPUT1"
        FROM ( SELECT "_CATEGORY", "_INDEX" AS "OUTPUT1"
        FROM ( SELECT  *  FROM (( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NOT NULL)
        UNION ( SELECT "_COLUMN_NAME", "_CATEGORY", 'NAN' :: FLOAT AS "_INDEX"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NULL)))
        WHERE ("_COLUMN_NAME" = 'STR1')))) AS SNOWPARK_RIGHT ON EQUAL_NULL("STR1", "_CATEGORY")))) AS SNOWPARK_LEFT
        LEFT OUTER JOIN ( SELECT "_CATEGORY" AS "_CATEGORY", "OUTPUT2" AS "OUTPUT2"
        FROM ( SELECT "_CATEGORY", "_INDEX" AS "OUTPUT2" FROM ( SELECT  *  FROM (( SELECT  *  FROM <TEMP_TABLE>
        WHERE "_CATEGORY" IS NOT NULL) UNION ( SELECT "_COLUMN_NAME", "_CATEGORY", 'NAN' :: FLOAT AS "_INDEX"
        FROM ( SELECT  *  FROM <TEMP_TABLE> WHERE "_CATEGORY" IS NULL)))
        WHERE ("_COLUMN_NAME" = 'STR2')))) AS SNOWPARK_RIGHT ON EQUAL_NULL("STR2", "_CATEGORY")))

        Transformed dataset:
        -----------------------------------------------
        |"STR1"  |"STR2"      |"OUTPUT1"  |"OUTPUT2"  |
        -----------------------------------------------
        |a       |g1ehQlL80t  |0.0        |0.0        |
        |a       |zOyDvcyZ2s  |0.0        |1.0        |
        |b       |NULL        |1.0        |nan        |
        |A       |TuDOcLxToB  |2.0        |2.0        |
        |NULL    |g1ehQlL80t  |nan        |0.0        |
        |NULL    |g1ehQlL80t  |nan        |0.0        |
        |b       |TuDOcLxToB  |1.0        |2.0        |
        -----------------------------------------------

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_transform_boolean(self) -> None:
        """
        Verify transformed results on boolean categories.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = BOOLEAN_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_BOOLEAN, SCHEMA_BOOLEAN, np.nan)

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_transform_numeric(self) -> None:
        """
        Verify transformed results on numeric categories.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

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

        encoder = OrdinalEncoder(categories=categories).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(categories=categories_list)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        # verify matching categories values
        self.assertDictEqual(
            {col: cat_values.tolist() for col, cat_values in encoder.categories.items()},
            {col: encoder_sklearn.categories_[i].tolist() for i, col in enumerate(input_cols)},
        )
        self.assertDictEqual(
            {col: cat_values.tolist() for col, cat_values in encoder.categories_.items()},
            {col: encoder_sklearn.categories_[i].tolist() for i, col in enumerate(input_cols)},
        )
        self.assertSequenceEqual(
            [cats.tolist() for cats in encoder._categories_list],
            [cats.tolist() for cats in encoder_sklearn.categories_],
        )
        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_categories_list(self) -> None:
        """
        Test that `categories` as a list of array-like is accepted.

        Raises
        ------
        AssertionError
            If the fitted categories do not match those of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        categories = []
        for idx, input_col in enumerate(input_cols):
            cats = list(set(framework_utils.get_pandas_feature(df_pandas[input_cols], feature_idx=idx)))
            # test excessive given categories
            cats.append(f"extra_cat_{input_col}")
            # sort categories in descending order to test if orders are preserved
            cats.sort(reverse=True)
            cats_arr = np.array(cats)
            categories.append(cats_arr)

        encoder = OrdinalEncoder(categories=categories).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(categories=categories)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        # verify matching categories values
        self.assertSequenceEqual(
            [cats.tolist() for cats in encoder.categories], [cats.tolist() for cats in encoder_sklearn.categories_]
        )
        self.assertDictEqual(
            {col: cat_values.tolist() for col, cat_values in encoder.categories_.items()},
            {col: encoder_sklearn.categories_[i].tolist() for i, col in enumerate(input_cols)},
        )
        self.assertSequenceEqual(
            [cats.tolist() for cats in encoder._categories_list],
            [cats.tolist() for cats in encoder_sklearn.categories_],
        )
        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

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

        encoder = OrdinalEncoder(categories=categories).set_input_cols(input_cols)

        with pytest.raises(ValueError) as excinfo:
            encoder.fit(unknown_df)

        self.assertTrue("Found unknown categories during fit" in excinfo.value.args[0])

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

        encoder = OrdinalEncoder(handle_unknown="error").set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        with pytest.raises(ValueError) as excinfo:
            encoder.transform(unknown_df)

        self.assertTrue("Found unknown categories during transform" in excinfo.value.args[0])

    def test_handle_unknown_use_encoded_value_nan(self) -> None:
        """
        Test `handle_unknown="use_encoded_value"` and `unknown_value=np.nan`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        transformed_df = encoder.transform(unknown_df)
        actual_arr = transformed_df.sort(input_cols[0])[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(unknown_pandas.sort_values(by=[input_cols[0]])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_handle_unknown_use_encoded_value_int(self) -> None:
        """
        Test `handle_unknown="use_encoded_value"` and `unknown_value` being set to an integer.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)
        unknown_df = self._session.create_dataframe(unknown_pandas)

        transformed_df = encoder.transform(unknown_df)
        actual_arr = transformed_df.sort(input_cols[0])[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(unknown_pandas.sort_values(by=[input_cols[0]])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_encoded_missing_value_nan(self) -> None:
        """
        Test `encoded_missing_value=np.nan`.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        encoded_missing_value = np.nan
        encoder = (
            OrdinalEncoder(encoded_missing_value=encoded_missing_value)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(encoded_missing_value=encoded_missing_value)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_encoded_missing_value_int(self) -> None:
        """
        Test `encoded_missing_value` being set to an integer.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        encoded_missing_value = -1
        encoder = (
            OrdinalEncoder(encoded_missing_value=encoded_missing_value)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(encoded_missing_value=encoded_missing_value)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_invalid_encoded_missing_value(self) -> None:
        """
        Test `encoded_missing_value` being set to an encoding of known categories.

        Raises
        ------
        AssertionError
            If no error on `encoded_missing_value` is raised during fit.
        """
        input_cols, output_cols, id_col = CATEGORICAL_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        encoded_missing_value = 1
        encoder = (
            OrdinalEncoder(encoded_missing_value=encoded_missing_value)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )

        with self.assertRaisesRegex(
            ValueError, r"encoded_missing_value \(.+\) is already used to encode a known category in features: .+"
        ):
            encoder.fit(df)

    def test_transform_pandas(self) -> None:
        """
        Verify transformed results on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df_pandas[input_cols])
        actual_output_cols = [c for c in transformed_df.columns if c not in input_cols]
        actual_arr = transformed_df[actual_output_cols].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_transform_null_pandas(self) -> None:
        """
        Verify transformed results on pandas dataframe with null categories.
        None is treated as a missing value, while in sklearn only NaN is
        considered missing.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)
        converted_pandas = df.to_pandas()

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(converted_pandas[input_cols])
        actual_output_cols = [c for c in transformed_df.columns if c not in input_cols]
        actual_arr = transformed_df[actual_output_cols].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_handle_unknown_use_encoded_value_int_pandas(self) -> None:
        """
        Test `handle_unknown="use_encoded_value"` and `unknown_value` being set
        to an integer on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        values_list = UNKNOWN_CATEGORICAL_VALUES_LIST
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        unknown_data = list(zip(*values_list))
        unknown_pandas = pd.DataFrame(unknown_data, columns=input_cols)

        transformed_df = encoder.transform(unknown_pandas[input_cols])
        actual_output_cols = [c for c in transformed_df.columns if c not in input_cols]
        actual_arr = transformed_df[actual_output_cols].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(unknown_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

    def test_encoded_missing_value_int_pandas(self) -> None:
        """
        Test `encoded_missing_value` being set to an integer on pandas dataframe.

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, OUTPUT_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)
        converted_pandas = df.to_pandas()

        encoded_missing_value = -1
        encoder = (
            OrdinalEncoder(encoded_missing_value=encoded_missing_value)
            .set_input_cols(input_cols)
            .set_output_cols(output_cols)
        )
        encoder.fit(df)

        transformed_df = encoder.transform(converted_pandas[input_cols])
        actual_output_cols = [c for c in transformed_df.columns if c not in input_cols]
        actual_arr = transformed_df[actual_output_cols].to_numpy()

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder(encoded_missing_value=encoded_missing_value)
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        np.testing.assert_allclose(actual_arr, sklearn_arr, equal_nan=True)

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

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            self._to_be_deleted_files.append(file.name)
            encoder_dump_cloudpickle = cloudpickle.dumps(encoder)
            # TODO(SNOW-1704904): Disabling pickle and joblib serde due to the below error
            # encoder_dump_pickle = pickle.dumps(encoder)
            # joblib.dump(encoder, file.name)

            self._session.close()

            # transform in session 2
            self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
            _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
            input_cols_extended = input_cols.copy()
            input_cols_extended.append(id_col)

            importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.ordinal_encoder"])

            # cloudpickle
            encoder_load_cloudpickle = cloudpickle.loads(encoder_dump_cloudpickle)
            transformed_df_cloudpickle = encoder_load_cloudpickle.transform(df2[input_cols_extended])
            actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # pickle
            # encoder_load_pickle = pickle.loads(encoder_dump_pickle)
            # transformed_df_pickle = encoder_load_pickle.transform(df2[input_cols_extended])
            # actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # joblib
            # encoder_load_joblib = joblib.load(file.name)
            # transformed_df_joblib = encoder_load_joblib.transform(df2[input_cols_extended])
            # actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

            # sklearn
            encoder_sklearn = SklearnOrdinalEncoder()
            encoder_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr, equal_nan=True)
            # np.testing.assert_allclose(actual_arr_pickle, sklearn_arr, equal_nan=True)
            # np.testing.assert_allclose(actual_arr_joblib, sklearn_arr, equal_nan=True)

    def test_same_input_output_cols(self) -> None:
        """
        Test input columns are the same as output columns, the output columns should overwrite the same input columns

        Raises
        ------
        AssertionError
            If the transformed output does not match that of the sklearn encoder.
        """
        input_cols, output_cols = CATEGORICAL_COLS, CATEGORICAL_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA, np.nan)

        encoder = OrdinalEncoder(drop_input_cols=True).set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)

        transformed_df = encoder.transform(df)

        # sklearn
        encoder_sklearn = SklearnOrdinalEncoder()
        encoder_sklearn.fit(df_pandas[input_cols])
        sklearn_arr = encoder_sklearn.transform(df_pandas[input_cols])

        self.assertEqual(len(set(input_cols) & set(transformed_df.to_pandas().columns)), len(set(input_cols)))
        np.testing.assert_allclose(transformed_df[output_cols].to_pandas().to_numpy(), sklearn_arr, equal_nan=True)

    def test_double_quoted_same_input_output_cols(self) -> None:
        data = [["a", "b"]]
        cat_cols = ['"col1"', '"col2"']
        df_pandas, df = framework_utils.get_df(self._session, data, cat_cols)

        encoder = OrdinalEncoder(drop_input_cols=True).set_input_cols(cat_cols).set_output_cols(cat_cols)
        transformed_df = encoder.fit(df).transform(df)

        self.assertEqual(cat_cols, transformed_df.columns)

    def test_mixed_column_types(self) -> None:
        data = [["a", 1]]
        cat_cols = ['"col1"', '"col2"']
        df_pandas, df = framework_utils.get_df(self._session, data, cat_cols)

        encoder = OrdinalEncoder(drop_input_cols=True).set_input_cols(cat_cols).set_output_cols(cat_cols)
        transformed_df = encoder.fit(df).transform(df)

        self.assertEqual(cat_cols, transformed_df.columns)

        data = [[1.0, True]]
        cat_cols = ['"col1"', '"col2"']
        df_pandas, df = framework_utils.get_df(self._session, data, cat_cols)

        encoder = OrdinalEncoder(drop_input_cols=True).set_input_cols(cat_cols).set_output_cols(cat_cols)
        transformed_df = encoder.fit(df).transform(df)

        self.assertEqual(cat_cols, transformed_df.columns)

        data = [[True, "a"]]
        cat_cols = ['"col1"', '"col2"']
        df_pandas, df = framework_utils.get_df(self._session, data, cat_cols)

        encoder = OrdinalEncoder(drop_input_cols=True).set_input_cols(cat_cols).set_output_cols(cat_cols)
        transformed_df = encoder.fit(df).transform(df)

        self.assertEqual(cat_cols, transformed_df.columns)

    def test_large_num_cols(self) -> None:
        num_cols = 300
        input_cols = [f"COL{i}" for i in range(1, num_cols + 1)]
        output_cols = [f"OUT{i}" for i in range(1, num_cols + 1)]
        col_cardinality = {col: 2 for col in input_cols}
        data = {}
        for col, cardinality in col_cardinality.items():
            data[col] = np.random.randint(0, cardinality, size=100)
        df = self._session.create_dataframe(pd.DataFrame(data))

        encoder = OrdinalEncoder(input_cols=input_cols, output_cols=output_cols)
        encoder.fit(df)
        res = encoder.transform(df)
        res.collect()

    def test_lowercase_output_cols_set_input_output(self) -> None:
        input_cols = "COL"
        output_cols = "OUT_foo"
        data = {"COL": np.random.randint(0, 2, size=10)}
        df = self._session.create_dataframe(pd.DataFrame(data))

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)
        res = encoder.transform(df)
        res.collect()
        assert "OUT_FOO" in res.columns

    def test_quoted_output_cols_set_input_output(self) -> None:
        input_cols = "COL"
        output_cols = '"OUT_foo"'
        data = {"COL": np.random.randint(0, 2, size=10)}
        df = self._session.create_dataframe(pd.DataFrame(data))

        encoder = OrdinalEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        encoder.fit(df)
        res = encoder.transform(df)
        res.collect()
        assert output_cols in res.columns

    def test_large_num_cols_unknown(self) -> None:
        num_cols = 300
        input_cols = [f"COL{i}" for i in range(1, num_cols + 1)]
        output_cols = [f"OUT{i}" for i in range(1, num_cols + 1)]
        col_cardinality = {col: 2 for col in input_cols}
        data = {}
        for col, cardinality in col_cardinality.items():
            data[col] = np.random.randint(0, cardinality, size=100)
        df = self._session.create_dataframe(pd.DataFrame(data))

        # unknown categories exist
        unknown_col_cardinality = {col: 3 for col in input_cols}
        unknown_data = {}
        for col, cardinality in unknown_col_cardinality.items():
            unknown_data[col] = np.random.randint(0, cardinality, size=100)
        unknown_df = self._session.create_dataframe(pd.DataFrame(unknown_data))

        encoder = OrdinalEncoder(input_cols=input_cols, output_cols=output_cols)
        encoder.fit(df)

        with pytest.raises(ValueError):
            encoder.transform(unknown_df)


if __name__ == "__main__":
    main()
