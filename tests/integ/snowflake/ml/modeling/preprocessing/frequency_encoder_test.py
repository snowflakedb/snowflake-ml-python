#!/usr/bin/env python3
from typing import Any

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from category_encoders import CountEncoder

from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml._internal.utils.connection_params import SnowflakeLoginOptions
from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    FrequencyEncoder,
)
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    CATEGORICAL_COLS,
    DATA,
    DATA_NONE_NAN,
    ID_COL,
    OUTPUT_COLS,
    SCHEMA,
)


def _count_encoder(normalize: bool, unknown_value: int = 0) -> CountEncoder:
    return CountEncoder(
        cols=CATEGORICAL_COLS,
        normalize=normalize,
        handle_unknown=unknown_value,
        handle_missing="value",
        return_df=True,
    )


class FrequencyEncoderTest(parameterized.TestCase):
    """Integration tests for FrequencyEncoder."""

    def setUp(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    @parameterized.parameters({"normalize": True}, {"normalize": False})  # type: ignore[misc]
    def test_fit_snowpark_frequencies(self, normalize: bool) -> None:
        """The fitted frequencies_ match category_encoders.CountEncoder's per-category values."""
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = FrequencyEncoder(normalize=normalize, input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS).fit(df)

        reference = _count_encoder(normalize)
        reference.fit(df_pandas[CATEGORICAL_COLS])
        for input_col in CATEGORICAL_COLS:
            expected = {str(category): float(value) for category, value in reference.mapping[input_col].items()}
            actual = {str(category): value for category, value in encoder.frequencies_[input_col].items()}
            self.assertEqual(expected, actual)

    @parameterized.parameters({"normalize": True}, {"normalize": False})  # type: ignore[misc]
    def test_transform_snowpark_matches_count_encoder(self, normalize: bool) -> None:
        """Snowpark-side encoding matches category_encoders.CountEncoder applied to the same data."""
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        encoder = FrequencyEncoder(normalize=normalize, input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS).fit(df)
        actual = encoder.transform(df).to_pandas().sort_values(ID_COL).reset_index(drop=True)

        expected = _count_encoder(normalize).fit_transform(df_pandas[CATEGORICAL_COLS])
        for input_col, output_col in zip(CATEGORICAL_COLS, OUTPUT_COLS):
            np.testing.assert_allclose(actual[output_col].to_numpy(), expected[input_col].to_numpy())

    def test_transform_snowpark_handles_nulls_and_unknowns(self) -> None:
        """Null is encoded as its own category; values unseen at fit get unknown_value."""
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)

        encoder = FrequencyEncoder(input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS).fit(df)

        # Null (seen during fit) is encoded with its own frequency, matching CountEncoder.
        expected = _count_encoder(normalize=True).fit_transform(df_pandas[CATEGORICAL_COLS])
        actual = encoder.transform(df).to_pandas().sort_values(ID_COL).reset_index(drop=True)
        for input_col, output_col in zip(CATEGORICAL_COLS, OUTPUT_COLS):
            np.testing.assert_allclose(actual[output_col].to_numpy(), expected[input_col].to_numpy())

        # A category not present during fit is encoded with unknown_value (default 0).
        unknown_pandas = pd.DataFrame([["99", "NOT_SEEN", "NOT_SEEN"]], columns=[ID_COL] + CATEGORICAL_COLS)
        unknown_df = self._session.create_dataframe(unknown_pandas)
        unknown_out = encoder.transform(unknown_df).to_pandas()
        for output_col in OUTPUT_COLS:
            self.assertEqual(unknown_out[output_col].iloc[0], 0.0)

    def test_custom_unknown_value(self) -> None:
        """unknown_value controls the encoding of categories unseen at fit time."""
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)
        encoder = FrequencyEncoder(unknown_value=-1, input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS).fit(df)

        unknown_pandas = pd.DataFrame([["99", "NOT_SEEN", "NOT_SEEN"]], columns=[ID_COL] + CATEGORICAL_COLS)
        unknown_df = self._session.create_dataframe(unknown_pandas)
        unknown_out = encoder.transform(unknown_df).to_pandas()
        for output_col in OUTPUT_COLS:
            self.assertEqual(unknown_out[output_col].iloc[0], -1.0)

    @parameterized.parameters({"normalize": True}, {"normalize": False})  # type: ignore[misc]
    def test_pandas_transform_matches_snowpark(self, normalize: bool) -> None:
        """A pandas fit/transform produces the same encoding as a Snowpark fit/transform."""
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        pandas_encoder = FrequencyEncoder(
            normalize=normalize, input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS
        ).fit(df_pandas)
        pandas_out = pandas_encoder.transform(df_pandas)

        snowpark_encoder = FrequencyEncoder(
            normalize=normalize, input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS
        ).fit(df)
        snowpark_out = snowpark_encoder.transform(df).to_pandas().sort_values(ID_COL).reset_index(drop=True)

        for output_col in OUTPUT_COLS:
            np.testing.assert_allclose(pandas_out[output_col].to_numpy(), snowpark_out[output_col].to_numpy())

    def test_to_sklearn(self) -> None:
        """to_sklearn returns the fitted CountEncoder after a pandas fit and errors after a Snowpark fit."""
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        pandas_encoder = FrequencyEncoder(input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS).fit(df_pandas)
        self.assertIsInstance(pandas_encoder.to_sklearn(), CountEncoder)

        snowpark_encoder = FrequencyEncoder(input_cols=CATEGORICAL_COLS, output_cols=OUTPUT_COLS).fit(df)
        with self.assertRaises(exceptions.SnowflakeMLException):
            snowpark_encoder.to_sklearn()

    def test_drop_input_cols(self) -> None:
        """drop_input_cols removes the original categorical columns from the output."""
        _, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)
        encoder = FrequencyEncoder(
            input_cols=CATEGORICAL_COLS,
            output_cols=OUTPUT_COLS,
            drop_input_cols=True,
        ).fit(df)
        transformed: Any = encoder.transform(df)
        for input_col in CATEGORICAL_COLS:
            self.assertNotIn(input_col, transformed.columns)
        for output_col in OUTPUT_COLS:
            self.assertIn(output_col, transformed.columns)


if __name__ == "__main__":
    absltest.main()
