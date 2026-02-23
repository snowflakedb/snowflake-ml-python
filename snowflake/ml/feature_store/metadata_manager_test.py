"""Unit tests for FeatureStoreMetadataManager."""

from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.metadata_manager import (
    AggregationMetadata,
    FeatureStoreMetadataManager,
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


if __name__ == "__main__":
    absltest.main()
