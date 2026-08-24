"""Unit tests for imperative FV_SOURCE_REFS auto-stamping.

``FeatureStore.register_feature_view`` now derives a best-effort source-ref
list from the raw ``feature_df`` schema when the caller did not supply one, so
imperatively-registered batch feature views carry ``FV_SOURCE_REFS`` metadata
and round-trip cleanly through declarative state recovery. These tests pin the
pure derivation helper ``_derive_source_refs_from_feature_df``.
"""

from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml.feature_store.feature_store import _derive_source_refs_from_feature_df
from snowflake.snowpark.types import (
    BinaryType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _fv(
    *,
    schema: object = None,
    is_streaming: bool = False,
    is_realtime: bool = False,
    feature_df_none: bool = False,
    name: str = "MY_BFV",
) -> MagicMock:
    """Build a MagicMock standing in for a FeatureView.

    The derivation helper only touches ``is_streaming``,
    ``is_realtime_feature_view``, ``feature_df`` (and its ``schema``), and
    ``name.resolved()``, so a MagicMock isolates the helper from the heavy
    FeatureView constructor.
    """
    fv = MagicMock()
    fv.is_streaming = is_streaming
    fv.is_realtime_feature_view = is_realtime
    fv.name.resolved.return_value = name
    if feature_df_none:
        fv.feature_df = None
    else:
        fv.feature_df.schema = schema
    return fv


class DeriveSourceRefsTest(absltest.TestCase):
    """Pin ``_derive_source_refs_from_feature_df`` behavior."""

    def test_batch_fv_captures_raw_schema_columns(self) -> None:
        schema = StructType(
            [
                StructField("USER_ID", StringType()),
                StructField("AMOUNT", DoubleType()),
                StructField("EVENT_TIME", TimestampType()),
            ]
        )
        refs = _derive_source_refs_from_feature_df(_fv(schema=schema))
        assert refs is not None
        self.assertEqual(len(refs), 1)
        # Synthetic ``<FV>__SOURCE`` name avoids self-colliding with the FV's
        # own identifier in the shared source/FV dependency namespace.
        self.assertEqual(refs[0]["name"], "MY_BFV__SOURCE")
        self.assertEqual(refs[0]["source_type"], "Batch")
        # Columns-only merge contract: the row must never carry binding
        # identity (``table``/``query`` are owned by the DT-text classifier).
        self.assertNotIn("table", refs[0])
        self.assertNotIn("query", refs[0])
        self.assertEqual(
            refs[0]["columns"],
            [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "AMOUNT", "type": "DoubleType"},
                {"name": "EVENT_TIME", "type": "TimestampType"},
            ],
        )

    def test_binary_column_is_captured(self) -> None:
        """BinaryType raw columns (e.g. HLL sketch inputs) must serialize."""
        schema = StructType(
            [
                StructField("KEY", StringType()),
                StructField("SKETCH", BinaryType()),
            ]
        )
        refs = _derive_source_refs_from_feature_df(_fv(schema=schema))
        assert refs is not None
        self.assertEqual(refs[0]["columns"][1], {"name": "SKETCH", "type": "BinaryType"})

    def test_streaming_fv_returns_none(self) -> None:
        schema = StructType([StructField("USER_ID", StringType())])
        self.assertIsNone(_derive_source_refs_from_feature_df(_fv(schema=schema, is_streaming=True)))

    def test_realtime_fv_returns_none(self) -> None:
        schema = StructType([StructField("USER_ID", StringType())])
        self.assertIsNone(_derive_source_refs_from_feature_df(_fv(schema=schema, is_realtime=True)))

    def test_missing_feature_df_returns_none(self) -> None:
        self.assertIsNone(_derive_source_refs_from_feature_df(_fv(feature_df_none=True)))

    def test_empty_schema_returns_none(self) -> None:
        self.assertIsNone(_derive_source_refs_from_feature_df(_fv(schema=StructType([]))))

    def test_schema_read_failure_returns_none(self) -> None:
        """A schema read that raises must never break registration."""
        fv = MagicMock()
        fv.is_streaming = False
        fv.is_realtime_feature_view = False
        fv.name.resolved.return_value = "MY_BFV"
        type(fv.feature_df).schema = property(lambda _self: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertIsNone(_derive_source_refs_from_feature_df(fv))


if __name__ == "__main__":
    absltest.main()
