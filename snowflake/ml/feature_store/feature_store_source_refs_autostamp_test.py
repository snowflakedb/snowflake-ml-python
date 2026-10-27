"""Unit tests for imperative FV_SOURCE_REFS auto-stamping.

``FeatureStore.register_feature_view`` now derives a best-effort source-ref
list from the raw ``feature_df`` schema when the caller did not supply one, so
imperatively-registered batch feature views carry ``FV_SOURCE_REFS`` metadata
and round-trip cleanly through declarative state recovery. These tests pin the
pure derivation helper ``_derive_source_refs_from_feature_df``.
"""

from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
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
    version: str | None = "V1",
    query: str = "SELECT USER_ID, AMOUNT FROM EVENTS",
) -> MagicMock:
    """Build a MagicMock standing in for a FeatureView.

    The derivation helper only touches ``is_streaming``,
    ``is_realtime_feature_view``, ``feature_df`` (and its ``schema``), ``query``,
    ``version``, and ``name`` (resolved and fed to
    ``FeatureView._get_physical_name``), so a MagicMock with a real
    ``SqlIdentifier`` name isolates the helper from the heavy FeatureView
    constructor while keeping the physical-name derivation faithful.
    """
    fv = MagicMock()
    fv.is_streaming = is_streaming
    fv.is_realtime_feature_view = is_realtime
    # A real SqlIdentifier so ``FeatureView._get_physical_name`` (which reads the
    # name object, not a mocked ``.resolved()``) produces the true ``name$version``.
    fv.name = SqlIdentifier(name)
    fv.version = version
    fv.query = query
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
        # Synthetic ``<FV>$<VERSION>__SOURCE`` name (the FV physical name) avoids
        # self-colliding with the FV's own identifier and keeps two versions of
        # one FV name from collapsing to a single virtual source during recovery.
        # ``$`` is an injective delimiter (the version regex forbids it).
        self.assertEqual(refs[0]["name"], "MY_BFV$V1__SOURCE")
        self.assertEqual(refs[0]["source_type"], "Batch")
        # The row carries the FV's query so recovery rebuilds a valid BatchSource
        # (DT ``FROM``-clause recovery was removed, so this is the only binding).
        self.assertEqual(refs[0]["query"], "SELECT USER_ID, AMOUNT FROM EVENTS")
        self.assertNotIn("table", refs[0])
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

    def test_version_qualified_name_distinct_per_version(self) -> None:
        """Two versions of one FV name yield distinct source names, so recovery
        does not dedupe them into one virtual source (the RECREATE_SOURCE loop).

        The explicit ``version`` argument mirrors the register call site, where
        the draft FV's own ``version`` may still be ``None``.
        """
        schema = StructType([StructField("USER_ID", StringType())])
        v1 = _derive_source_refs_from_feature_df(_fv(schema=schema), version="V1")
        v2 = _derive_source_refs_from_feature_df(_fv(schema=schema), version="V2")
        assert v1 is not None and v2 is not None
        self.assertEqual(v1[0]["name"], "MY_BFV$V1__SOURCE")
        self.assertEqual(v2[0]["name"], "MY_BFV$V2__SOURCE")

    def test_name_version_boundary_is_injective(self) -> None:
        """``$`` separates name from version unambiguously, so (name, version)
        pairs that the old ``_``-joined form collapsed now stay distinct.

        FV ``A_B`` version ``C`` and FV ``A`` version ``B_C`` both produced
        ``A_B_C__SOURCE`` under the old scheme; the physical-name ``$`` delimiter
        keeps them apart (``A_B$C`` vs ``A$B_C``).
        """
        schema = StructType([StructField("USER_ID", StringType())])
        a_b__c = _derive_source_refs_from_feature_df(_fv(schema=schema, name="A_B"), version="C")
        a__b_c = _derive_source_refs_from_feature_df(_fv(schema=schema, name="A"), version="B_C")
        assert a_b__c is not None and a__b_c is not None
        self.assertEqual(a_b__c[0]["name"], "A_B$C__SOURCE")
        self.assertEqual(a__b_c[0]["name"], "A$B_C__SOURCE")
        self.assertNotEqual(a_b__c[0]["name"], a__b_c[0]["name"])

    def test_dotted_version_name(self) -> None:
        """A version containing ``.`` (``1.0``) rides into the name verbatim; the
        exporter makes it a safe filename, so the stamp must not sanitise it."""
        schema = StructType([StructField("USER_ID", StringType())])
        refs = _derive_source_refs_from_feature_df(_fv(schema=schema), version="1.0")
        assert refs is not None
        self.assertEqual(refs[0]["name"], "MY_BFV$1.0__SOURCE")

    def test_unversioned_fallback_when_no_version(self) -> None:
        """No resolvable version falls back to the bare ``<FV>__SOURCE`` name."""
        schema = StructType([StructField("USER_ID", StringType())])
        refs = _derive_source_refs_from_feature_df(_fv(schema=schema, version=None))
        assert refs is not None
        self.assertEqual(refs[0]["name"], "MY_BFV__SOURCE")

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

    def test_missing_query_returns_none(self) -> None:
        """Without a resolvable query there is no binding to stamp."""
        schema = StructType([StructField("USER_ID", StringType())])
        self.assertIsNone(_derive_source_refs_from_feature_df(_fv(schema=schema, query="")))

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
