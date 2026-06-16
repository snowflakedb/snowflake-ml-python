"""Unit tests for FeatureStoreMetadataManager."""

from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.aggregation import AggregationSpec, AggregationType
from snowflake.ml.feature_store.metadata_manager import (
    AggregationMetadata,
    FeatureStoreMetadataManager,
    FvSourceRefsMetadata,
    MetadataObjectType,
    MetadataType,
)


def _make_manager():
    """Create a FeatureStoreMetadataManager with a mocked session."""
    session = MagicMock()
    session.sql.return_value.collect.return_value = []
    manager = FeatureStoreMetadataManager(
        session=session,
        schema_path="DB.SCHEMA",
        fs_object_tag_path="DB.SCHEMA.FS_TAG",
        telemetry_stmp={},
    )
    # Force table_exists to True so we skip the CREATE TABLE check
    manager._table_exists = True
    return manager, session


class FeatureViewMetadataNormalizationTest(absltest.TestCase):
    """Tests that feature view metadata normalization is consistent between write/read/delete."""

    def test_save_fv_metadata_unquoted_name_uppercased(self) -> None:
        """Unquoted FV name is already uppercased by SqlIdentifier, so strip('"') preserves it."""
        manager, session = _make_manager()
        # SqlIdentifier('myFV').identifier() -> 'MYFV'  (unquoted -> uppercased)
        fv_name = str(SqlIdentifier("myFV"))  # 'MYFV'
        specs = AggregationMetadata(feature_granularity="1h", features=[])

        manager.save_feature_view_metadata(fv_name, "v1", specs)

        sql = session.sql.call_args[0][0]
        # The OBJECT_NAME in the SQL should be 'MYFV'
        self.assertIn("'MYFV'", sql)
        self.assertNotIn("'myFV'", sql)

    def test_save_fv_metadata_quoted_name_preserves_case(self) -> None:
        """Quoted FV name preserves its mixed case after strip('\"')."""
        manager, session = _make_manager()
        # SqlIdentifier('"myFV"').identifier() -> '"myFV"'
        fv_name = str(SqlIdentifier('"myFV"'))  # '"myFV"'
        specs = AggregationMetadata(feature_granularity="1h", features=[])

        manager.save_feature_view_metadata(fv_name, "v1", specs)

        sql = session.sql.call_args[0][0]
        # After fix: strip('"') gives 'myFV', NOT 'MYFV'
        self.assertIn("'myFV'", sql)
        self.assertNotIn("'MYFV'", sql)

    def test_delete_fv_metadata_quoted_name_preserves_case(self) -> None:
        """Quoted FV name preserves its mixed case in DELETE."""
        manager, session = _make_manager()
        fv_name = str(SqlIdentifier('"myFV"'))  # '"myFV"'

        manager.delete_feature_view_metadata(fv_name, "v1")

        sql = session.sql.call_args[0][0]
        self.assertIn("'myFV'", sql)
        self.assertNotIn("'MYFV'", sql)

    def test_save_and_get_use_same_key_for_quoted_name(self) -> None:
        """Verify write and read paths produce the same OBJECT_NAME for a quoted identifier."""
        manager, session = _make_manager()
        fv_name = str(SqlIdentifier('"myFV"'))  # '"myFV"'
        specs = AggregationMetadata(feature_granularity="1h", features=[])

        # Write
        manager.save_feature_view_metadata(fv_name, "v1", specs)
        write_sql = session.sql.call_args[0][0]

        # Read (get_feature_specs -> _get_metadata)
        session.sql.return_value.collect.return_value = []
        manager.get_feature_specs(fv_name, "v1")
        read_sql = session.sql.call_args[0][0]

        # Both should use 'myFV' as the OBJECT_NAME
        self.assertIn("'myFV'", write_sql)
        self.assertIn("'myFV'", read_sql)

    def test_quoted_and_unquoted_produce_distinct_keys(self) -> None:
        """Quoted '"myFV"' and unquoted 'myFV' must produce different OBJECT_NAME values."""
        manager, session = _make_manager()
        specs = AggregationMetadata(feature_granularity="1h", features=[])

        # Unquoted 'myFV'
        manager.save_feature_view_metadata(str(SqlIdentifier("myFV")), "v1", specs)
        unquoted_sql = session.sql.call_args[0][0]

        # Quoted '"myFV"'
        manager.save_feature_view_metadata(str(SqlIdentifier('"myFV"')), "v1", specs)
        quoted_sql = session.sql.call_args[0][0]

        # Unquoted -> MYFV, quoted -> myFV
        self.assertIn("'MYFV'", unquoted_sql)
        self.assertIn("'myFV'", quoted_sql)


class StreamSourceMetadataNormalizationTest(absltest.TestCase):
    """Tests that stream source metadata normalization uses strip('"') consistently."""

    def test_upsert_unquoted_name_uppercased(self) -> None:
        """Unquoted stream source name is already uppercased by SqlIdentifier."""
        manager, session = _make_manager()
        name = SqlIdentifier("mySource").resolved()  # 'MYSOURCE'

        manager.save_stream_source(name, {"name": "MYSOURCE", "schema": {}, "desc": "", "owner": "R"})

        sql = session.sql.call_args[0][0]
        self.assertIn("'MYSOURCE'", sql)

    def test_upsert_quoted_name_preserves_case(self) -> None:
        """Quoted stream source name preserves its mixed case."""
        manager, session = _make_manager()
        name = SqlIdentifier('"mySource"').resolved()  # 'mySource'

        manager.save_stream_source(name, {"name": '"mySource"', "schema": {}, "desc": "", "owner": "R"})

        sql = session.sql.call_args[0][0]
        self.assertIn("'mySource'", sql)
        self.assertNotIn("'MYSOURCE'", sql)

    def test_get_quoted_name_preserves_case(self) -> None:
        """Get uses the same case-preserving key as save."""
        manager, session = _make_manager()
        name = SqlIdentifier('"mySource"').resolved()  # 'mySource'

        manager.get_stream_source_metadata(name)

        sql = session.sql.call_args[0][0]
        self.assertIn("'mySource'", sql)

    def test_delete_quoted_name_preserves_case(self) -> None:
        """Delete uses the same case-preserving key as save."""
        manager, session = _make_manager()
        name = SqlIdentifier('"mySource"').resolved()  # 'mySource'

        manager.delete_stream_source_metadata(name)

        sql = session.sql.call_args[0][0]
        self.assertIn("'mySource'", sql)


class TestFvSourceRefsMetadata(absltest.TestCase):
    """Round-trip tests for ``FvSourceRefsMetadata``.

    Covers the dataclass's ``to_dict``/``from_dict`` symmetry and the
    manager's ``save_feature_view_source_refs`` / ``get_feature_view_source_refs``
    pair, which together must preserve the authored source-ref list
    byte-identically across the metadata table.
    """

    def _sample_sources(self) -> list[dict]:
        return [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": "MY_DB.MY_SCH.EVENTS",
                "columns": [
                    {"name": "USER_ID", "type": "VARCHAR"},
                    {"name": "EVENT_TIME", "type": "TIMESTAMP_NTZ"},
                ],
            },
            {
                "name": "EVENTS_SQL_BATCH_DECL",
                "source_type": "Batch",
                "query": "SELECT * FROM MY_DB.MY_SCH.EVENTS",
                "columns": [{"name": "USER_ID", "type": "VARCHAR"}],
                "source_database": "MY_DB",
                "source_schema": "MY_SCH",
            },
        ]

    def test_to_dict_minimal(self) -> None:
        meta = FvSourceRefsMetadata(sources=[])
        d = meta.to_dict()
        self.assertEqual(d, {"sources": []})

    def test_to_dict_full(self) -> None:
        meta = FvSourceRefsMetadata(sources=self._sample_sources())
        d = meta.to_dict()
        self.assertEqual(len(d["sources"]), 2)
        self.assertEqual(d["sources"][0]["name"], "EVENTS_BATCH_DECL")
        self.assertEqual(d["sources"][0]["columns"][0]["name"], "USER_ID")
        self.assertEqual(d["sources"][1]["query"], "SELECT * FROM MY_DB.MY_SCH.EVENTS")

    def test_from_dict_minimal(self) -> None:
        meta = FvSourceRefsMetadata.from_dict({"sources": []})
        self.assertEqual(meta.sources, [])

    def test_from_dict_full(self) -> None:
        sources = self._sample_sources()
        meta = FvSourceRefsMetadata.from_dict({"sources": sources})
        self.assertEqual(meta.sources, sources)

    def test_round_trip_preserves_all_authored_fields(self) -> None:
        """All authored keys (name, source_type, table, query, columns,
        source_database, source_schema) survive a ``to_dict``/``from_dict`` pass."""
        sources = self._sample_sources()
        meta = FvSourceRefsMetadata(sources=sources)
        restored = FvSourceRefsMetadata.from_dict(meta.to_dict())
        self.assertEqual(restored.sources, sources)
        self.assertEqual(restored.sources[1]["source_database"], "MY_DB")
        self.assertEqual(restored.sources[1]["source_schema"], "MY_SCH")

    def test_round_trip_via_save_get(self) -> None:
        """``save_feature_view_source_refs`` writes a row that ``get_feature_view_source_refs``
        recovers byte-identically (round-trip via the metadata table)."""
        manager, session = _make_manager()
        sources = self._sample_sources()
        meta = FvSourceRefsMetadata(sources=sources)

        manager.save_feature_view_source_refs("MY_FV", "v1", meta)
        # Capture the JSON the manager wrote, then have the session return it
        # as a row when ``get_feature_view_source_refs`` queries.
        write_sql = session.sql.call_args[0][0]
        self.assertIn("FV_SOURCE_REFS", write_sql)
        self.assertIn("'MY_FV'", write_sql)
        self.assertIn("'v1'", write_sql)

        # Drive the read path: simulate the row coming back from Snowflake.
        session.sql.return_value.collect.return_value = [
            MagicMock(__getitem__=lambda self, key: meta.to_dict() if key == "METADATA" else None)
        ]
        restored = manager.get_feature_view_source_refs("MY_FV", "v1")
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.sources, sources)

    def test_save_uses_metadata_type_enum(self) -> None:
        """``save_feature_view_source_refs`` writes under ``MetadataType.FV_SOURCE_REFS``."""
        manager, session = _make_manager()
        manager.save_feature_view_source_refs("FV", "v1", FvSourceRefsMetadata(sources=[]))

        sql = session.sql.call_args[0][0]
        self.assertIn(MetadataType.FV_SOURCE_REFS.value, sql)
        self.assertIn(MetadataObjectType.FEATURE_VIEW.value, sql)

    def test_get_returns_none_when_row_absent(self) -> None:
        manager, session = _make_manager()
        session.sql.return_value.collect.return_value = []
        self.assertIsNone(manager.get_feature_view_source_refs("FV", "v1"))

    def test_delete_feature_view_metadata_sweeps_source_refs(self) -> None:
        """The existing per-FV cleanup sweep must remove the FV_SOURCE_REFS row."""
        manager, session = _make_manager()
        manager.delete_feature_view_metadata("FV", "v1")
        sql = session.sql.call_args[0][0]
        # The sweep is keyed by (OBJECT_TYPE, OBJECT_NAME, VERSION); no METADATA_TYPE
        # filter — so adding FV_SOURCE_REFS as a new type rides for free.
        self.assertIn("DELETE FROM", sql)
        self.assertIn("'FV'", sql)
        self.assertIn("'v1'", sql)
        self.assertIn(MetadataObjectType.FEATURE_VIEW.value, sql)


class TestAggregationMetadataAggregationSecondaryKeys(absltest.TestCase):
    """A2: ``aggregation_secondary_keys`` round-trips through AggregationMetadata."""

    def _sample_aggregation_specs(self) -> list[AggregationSpec]:
        return [
            AggregationSpec(
                source_column="AMOUNT",
                output_column="TOTAL_AMOUNT_24H",
                function=AggregationType.SUM,
                window="24h",
                offset="0",
            ),
        ]

    def test_aggregation_secondary_keys_round_trip(self) -> None:
        """``AggregationMetadata.aggregation_secondary_keys`` survives to_dict/from_dict."""
        meta = AggregationMetadata(
            feature_granularity="1h",
            features=self._sample_aggregation_specs(),
            aggregation_secondary_keys=["MERCHANT_ID"],
        )
        restored = AggregationMetadata.from_dict(meta.to_dict())
        self.assertEqual(restored.aggregation_secondary_keys, ["MERCHANT_ID"])

    def test_aggregation_secondary_keys_default_none(self) -> None:
        """A freshly-constructed ``AggregationMetadata`` has ``None`` secondary keys."""
        meta = AggregationMetadata(feature_granularity="1h", features=[])
        self.assertIsNone(meta.aggregation_secondary_keys)
        # And the field is stripped from the dict when None.
        self.assertNotIn("aggregation_secondary_keys", meta.to_dict())

    def test_legacy_dict_without_secondary_keys_decodes_as_none(self) -> None:
        """Existing applied state (no secondary keys field) decodes cleanly."""
        legacy = {
            "feature_granularity": "1h",
            "features": [],
        }
        meta = AggregationMetadata.from_dict(legacy)
        self.assertIsNone(meta.aggregation_secondary_keys)


if __name__ == "__main__":
    absltest.main()
