import pandas as pd
from absl.testing import absltest

from snowflake.ml.modeling.preprocessing import (
    label_encoder,
    one_hot_encoder,
    ordinal_encoder,
)


def _string_frame(*values: str | None) -> pd.DataFrame:
    return pd.DataFrame({"c": pd.Series(list(values), dtype="string")})


def _one_hot_encoder() -> one_hot_encoder.OneHotEncoder:
    return one_hot_encoder.OneHotEncoder(handle_unknown="error", sparse=False, input_cols=["c"], output_cols=["c_out"])


def _ordinal_encoder() -> ordinal_encoder.OrdinalEncoder:
    return ordinal_encoder.OrdinalEncoder(handle_unknown="error", input_cols=["c"], output_cols=["c_out"])


class EncoderStringNaTest(absltest.TestCase):
    def test_one_hot_encoder_object_none_fit_string_transform(self) -> None:
        encoder = _one_hot_encoder()
        encoder.fit(pd.DataFrame({"c": ["a", None]}))
        transformed = encoder.transform(_string_frame("a", None, "a"))
        self.assertEqual(transformed[encoder.get_output_cols()].to_numpy().shape[0], 3)

    def test_one_hot_encoder_string_missing_is_known(self) -> None:
        encoder = _one_hot_encoder()
        encoder.fit(_string_frame("a", None))
        transformed = encoder.transform(_string_frame("a", None, "a"))
        self.assertEqual(transformed[encoder.get_output_cols()].to_numpy().shape[0], 3)

    def test_one_hot_encoder_unknown_string_missing_raises(self) -> None:
        encoder = _one_hot_encoder()
        encoder.fit(_string_frame("a", "b"))
        with self.assertRaisesRegex(ValueError, "Found unknown categories"):
            encoder.transform(_string_frame("a", None))

    def test_ordinal_encoder_object_none_fit_string_transform(self) -> None:
        encoder = _ordinal_encoder()
        encoder.fit(pd.DataFrame({"c": ["a", None]}))
        transformed = encoder.transform(_string_frame(None, "a"))
        self.assertEqual(len(transformed["c_out"]), 2)

    def test_ordinal_encoder_string_missing_is_known(self) -> None:
        encoder = _ordinal_encoder()
        encoder.fit(_string_frame("a", None))
        transformed = encoder.transform(_string_frame(None, "a"))
        self.assertEqual(len(transformed["c_out"]), 2)

    def test_ordinal_encoder_unknown_string_missing_raises(self) -> None:
        encoder = _ordinal_encoder()
        encoder.fit(_string_frame("a", "b"))
        with self.assertRaisesRegex(ValueError, "Found unknown categories"):
            encoder.transform(_string_frame("a", None))

    def test_label_encoder_string_missing_is_known(self) -> None:
        encoder = label_encoder.LabelEncoder(input_cols=["c"], output_cols=["c_out"])
        encoder.fit(_string_frame("a", None))
        transformed = encoder.transform(_string_frame(None, "a"))
        self.assertEqual(len(transformed["c_out"]), 2)


if __name__ == "__main__":
    absltest.main()
