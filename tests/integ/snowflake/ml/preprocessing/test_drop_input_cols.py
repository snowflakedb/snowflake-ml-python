#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import os
from typing import Any, List

import numpy as np
from absl.testing.absltest import TestCase

from snowflake.ml.framework.pipeline import Pipeline
from snowflake.ml.preprocessing import (
    Binarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.framework import utils as framework_utils
from tests.integ.snowflake.ml.framework.utils import (
    CATEGORICAL_COLS,
    DATA,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
)


class DropInputColsTest(TestCase):
    """Test DropInputCols."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def _run_and_compare_result(
        self,
        transformer: Any,
        input_cols: List[str],
        output_cols: List[str],
        id_col: str,
        data: List[List[Any]],
    ) -> None:
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        input_pandas_df, input_df = framework_utils.get_df(self._session, data, SCHEMA, np.nan)
        transformer.set_input_cols(input_cols=input_cols).set_output_cols(output_cols=output_cols)
        transformer.fit(input_df)
        # Transform Snowpark DataFrame
        output_df = transformer.transform(input_df[input_cols_extended])
        pruned_output_cols = output_df.columns.copy()
        pruned_output_cols.remove(id_col)
        assert all(elem not in input_df.columns for elem in pruned_output_cols)
        # Transform Pandas DataFrame
        pandas_output_df = transformer.transform(input_pandas_df[input_cols_extended])
        pandas_pruned_output_cols = pandas_output_df.columns.tolist()
        pandas_pruned_output_cols.remove(id_col)
        assert all(elem not in input_pandas_df.columns for elem in pandas_pruned_output_cols)

    def test_one_hot_encoder(self) -> None:
        self._run_and_compare_result(
            OneHotEncoder(sparse=False, drop_input_cols=True), CATEGORICAL_COLS, OUTPUT_COLS, ID_COL, DATA
        )

    def test_min_max_scaler(self) -> None:
        self._run_and_compare_result(
            MinMaxScaler(feature_range=(-1, 1), drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, DATA
        )

    def test_label_encoder(self) -> None:
        input_col = SCHEMA[1]
        output_col = "_TEST"
        self._run_and_compare_result(LabelEncoder(drop_input_cols=True), [input_col], [output_col], ID_COL, DATA)

    def test_ordinal_encoder(self) -> None:
        self._run_and_compare_result(OrdinalEncoder(drop_input_cols=True), CATEGORICAL_COLS, OUTPUT_COLS, ID_COL, DATA)

    def test_robust_scaler(self) -> None:
        self._run_and_compare_result(RobustScaler(drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, DATA)

    def test_simple_imputer(self) -> None:
        self._run_and_compare_result(SimpleImputer(drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, DATA)

    def test_max_abs_scaler(self) -> None:
        self._run_and_compare_result(MaxAbsScaler(drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, DATA)

    def test_standard_scaler(self) -> None:
        self._run_and_compare_result(StandardScaler(drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, DATA)

    def test_normalizer(self) -> None:
        data_normalize = [
            ["1", "c", "zeroes", 0.0, 0.0],
            ["2", "a", "norm:l2", -3.0, -4.0],
            ["3", "a", "norm:l2", 3.0, 4.0],
            ["4", "A", "big_number", 3.5, -123456789.9999],
            ["5", "A", "small_number", 3.0, 0.00000000012],
            ["6", "b", "norm:max", -20.0, -100.0],
            ["7", "b", "norm:max", -10.0, 100],
        ]
        self._run_and_compare_result(
            Normalizer(norm="l2", drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, data_normalize
        )

    def test_binarizer(self) -> None:
        self._run_and_compare_result(
            Binarizer(threshold=0.2, drop_input_cols=True), NUMERIC_COLS, OUTPUT_COLS, ID_COL, DATA
        )

    def test_pipeline(self) -> None:
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)
        ohe = (
            OneHotEncoder(sparse=False, drop_input_cols=True)
            .set_input_cols(CATEGORICAL_COLS)
            .set_output_cols(OUTPUT_COLS)
        )
        mms = MinMaxScaler(drop_input_cols=True).set_input_cols(NUMERIC_COLS).set_output_cols(OUTPUT_COLS)
        ppl = Pipeline([("ohe", ohe), ("mms", mms)])
        ppl.fit(df)
        transformed_df = ppl.transform(df)
        assert all(elem not in [CATEGORICAL_COLS + NUMERIC_COLS] for elem in transformed_df.columns)
