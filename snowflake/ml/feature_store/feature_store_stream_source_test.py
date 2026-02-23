"""Unit tests for FeatureStore stream source CRUD methods."""

from typing import Any
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.stream_source import StreamSource, _schema_to_dict
from snowflake.snowpark.types import (
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)


def _make_schema() -> StructType:
    return StructType(
        [
            StructField("user_id", StringType()),
            StructField("amount", FloatType()),
            StructField("event_time", TimestampType(TimestampTimeZone.NTZ)),
        ]
    )


def _make_stream_source(name: str = "txn_events", desc: str = "Test") -> StreamSource:
    return StreamSource(
        name=name,
        schema=_make_schema(),
        desc=desc,
    )


def _make_metadata(name: str = "TXN_EVENTS", desc: str = "Test", owner: str = "ROLE_1") -> dict[str, Any]:
    """Build a metadata dict as returned by metadata_manager."""
    ss = _make_stream_source(name, desc=desc)
    ss.owner = owner
    return ss._to_dict()


def _create_feature_store_with_mocks() -> Any:
    """Create a FeatureStore with mocked dependencies (bypassing __init__)."""
    from snowflake.ml.feature_store.feature_store import (
        FeatureStore,
        _FeatureStoreConfig,
    )

    fs = object.__new__(FeatureStore)
    fs._session = MagicMock()
    fs._session.get_current_role.return_value = "ROLE_1"
    fs._session.get_current_warehouse.return_value = "WH_1"
    fs._metadata_manager = MagicMock()
    fs._config = _FeatureStoreConfig(
        database=SqlIdentifier("TEST_DB"),
        schema=SqlIdentifier("TEST_SCHEMA"),
    )
    fs._default_warehouse = SqlIdentifier("WH_1")
    fs._telemetry_stmp = {}
    fs._default_iceberg_external_volume = None
    fs._asof_join_enabled = None
    return fs


class RegisterStreamSourceTest(absltest.TestCase):
    """Tests for FeatureStore.register_stream_source."""

    def test_register_new_stream_source(self) -> None:
        """Test registering a new stream source succeeds."""
        fs = _create_feature_store_with_mocks()
        ss = _make_stream_source()

        # Stream source does not exist yet
        fs._metadata_manager.stream_source_exists.return_value = False
        # get_stream_source_metadata returns the saved metadata (after save)
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata()

        result = fs.register_stream_source(ss)

        fs._metadata_manager.save_stream_source.assert_called_once()
        call_args = fs._metadata_manager.save_stream_source.call_args
        self.assertEqual(call_args.kwargs["name"], "TXN_EVENTS")
        self.assertIsInstance(result, StreamSource)
        self.assertEqual(result.name, SqlIdentifier("TXN_EVENTS"))
        self.assertEqual(result.owner, "ROLE_1")

    def test_register_existing_stream_source_warns(self) -> None:
        """Test registering an already existing stream source raises UserWarning."""
        fs = _create_feature_store_with_mocks()
        ss = _make_stream_source()

        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata()

        with self.assertWarnsRegex(UserWarning, "already exists"):
            result = fs.register_stream_source(ss)

        # Should not save again
        fs._metadata_manager.save_stream_source.assert_not_called()
        self.assertIsInstance(result, StreamSource)

    def test_register_sets_owner_from_session(self) -> None:
        """Test that registration sets the owner from the current session role."""
        fs = _create_feature_store_with_mocks()
        fs._session.get_current_role.return_value = "MY_SPECIAL_ROLE"
        ss = _make_stream_source()

        fs._metadata_manager.stream_source_exists.return_value = False
        metadata = _make_metadata(owner="MY_SPECIAL_ROLE")
        fs._metadata_manager.get_stream_source_metadata.return_value = metadata

        result = fs.register_stream_source(ss)
        self.assertEqual(result.owner, "MY_SPECIAL_ROLE")

    def test_register_save_failure_raises(self) -> None:
        """Test that a save failure is raised as RuntimeError (unwrapped by dispatch_decorator)."""
        fs = _create_feature_store_with_mocks()
        ss = _make_stream_source()

        fs._metadata_manager.stream_source_exists.return_value = False
        fs._metadata_manager.save_stream_source.side_effect = RuntimeError("DB error")

        with self.assertRaises(RuntimeError) as cm:
            fs.register_stream_source(ss)
        self.assertIn("Failed to register stream source", str(cm.exception))


class GetStreamSourceTest(absltest.TestCase):
    """Tests for FeatureStore.get_stream_source."""

    def test_get_existing_stream_source(self) -> None:
        """Test retrieving an existing stream source."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata()

        result = fs.get_stream_source("txn_events")

        self.assertIsInstance(result, StreamSource)
        self.assertEqual(result.name, SqlIdentifier("TXN_EVENTS"))
        self.assertEqual(result.owner, "ROLE_1")
        self.assertEqual(result.desc, "Test")

    def test_get_nonexistent_stream_source_raises(self) -> None:
        """Test that getting a non-existent stream source raises ValueError (unwrapped by dispatch_decorator)."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_stream_source_metadata.return_value = None

        with self.assertRaises(ValueError) as cm:
            fs.get_stream_source("nonexistent")
        self.assertIn("Cannot find StreamSource", str(cm.exception))

    def test_get_stream_source_db_failure_raises(self) -> None:
        """Test that a database failure is raised as RuntimeError (unwrapped by dispatch_decorator)."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_stream_source_metadata.side_effect = RuntimeError("DB error")

        with self.assertRaises(RuntimeError) as cm:
            fs.get_stream_source("txn_events")
        self.assertIn("Failed to retrieve stream source", str(cm.exception))

    def test_get_case_insensitive(self) -> None:
        """Test that get_stream_source resolves names case-insensitively."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata()

        # Both should resolve to the same name
        result1 = fs.get_stream_source("txn_events")
        result2 = fs.get_stream_source("TXN_EVENTS")

        self.assertEqual(result1, result2)


class DeleteStreamSourceTest(absltest.TestCase):
    """Tests for FeatureStore.delete_stream_source."""

    def test_delete_existing_with_no_references(self) -> None:
        """Test deleting a stream source with no active references."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_ref_count.return_value = 0

        fs.delete_stream_source("txn_events")

        fs._metadata_manager.delete_stream_source_metadata.assert_called_once_with("TXN_EVENTS")

    def test_delete_nonexistent_raises(self) -> None:
        """Test that deleting a non-existent stream source raises ValueError (unwrapped by dispatch_decorator)."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = False

        with self.assertRaises(ValueError) as cm:
            fs.delete_stream_source("nonexistent")
        self.assertIn("does not exist", str(cm.exception))

    def test_delete_with_active_references_raises(self) -> None:
        """Test that deleting a stream source with active references fails."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_ref_count.return_value = 2

        with self.assertRaises(ValueError) as cm:
            fs.delete_stream_source("txn_events")
        self.assertIn("2 active reference(s)", str(cm.exception))

        # Should not reach the actual delete
        fs._metadata_manager.delete_stream_source_metadata.assert_not_called()

    def test_delete_db_failure_raises(self) -> None:
        """Test that a database failure during delete is raised as RuntimeError (unwrapped by dispatch_decorator)."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_ref_count.return_value = 0
        fs._metadata_manager.delete_stream_source_metadata.side_effect = RuntimeError("DB error")

        with self.assertRaises(RuntimeError) as cm:
            fs.delete_stream_source("txn_events")
        self.assertIn("Failed to delete stream source", str(cm.exception))


class UpdateStreamSourceTest(absltest.TestCase):
    """Tests for FeatureStore.update_stream_source."""

    def test_update_desc(self) -> None:
        """Test updating the description of a stream source."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata(desc="New desc")

        result = fs.update_stream_source("txn_events", desc="New desc")

        fs._metadata_manager.update_stream_source_desc.assert_called_once_with("TXN_EVENTS", "New desc")
        self.assertIsNotNone(result)
        self.assertEqual(result.desc, "New desc")

    def test_update_nonexistent_warns(self) -> None:
        """Test that updating a non-existent stream source returns None with warning."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = False

        with self.assertWarnsRegex(UserWarning, "does not exist"):
            result = fs.update_stream_source("nonexistent", desc="New desc")

        self.assertIsNone(result)
        fs._metadata_manager.update_stream_source_desc.assert_not_called()

    def test_update_no_desc_change(self) -> None:
        """Test that calling update without desc doesn't update the metadata."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata()

        result = fs.update_stream_source("txn_events")

        fs._metadata_manager.update_stream_source_desc.assert_not_called()
        self.assertIsNotNone(result)

    def test_update_db_failure_raises(self) -> None:
        """Test that a database failure during update is raised as RuntimeError (unwrapped by dispatch_decorator)."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.update_stream_source_desc.side_effect = RuntimeError("DB error")

        with self.assertRaises(RuntimeError) as cm:
            fs.update_stream_source("txn_events", desc="fail")
        self.assertIn("Failed to update stream source", str(cm.exception))


class ListStreamSourcesTest(absltest.TestCase):
    """Tests for FeatureStore.list_stream_sources."""

    def test_list_when_table_not_exists(self) -> None:
        """Test list_stream_sources returns empty DataFrame when metadata table doesn't exist."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.table_exists.return_value = False

        # Mock create_dataframe to return a mock DataFrame
        mock_df = MagicMock()
        fs._session.create_dataframe.return_value = mock_df

        result = fs.list_stream_sources()

        fs._session.create_dataframe.assert_called_once()
        self.assertEqual(result, mock_df)

    def test_list_when_table_exists(self) -> None:
        """Test list_stream_sources issues correct SQL when metadata table exists."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.table_exists.return_value = True
        fs._metadata_manager.table_path = "TEST_DB.TEST_SCHEMA.__FEATURE_STORE_METADATA__"

        mock_df = MagicMock()
        fs._session.sql.return_value = mock_df

        fs.list_stream_sources()

        # Verify SQL was called
        fs._session.sql.assert_called_once()
        sql = fs._session.sql.call_args[0][0]

        # Check that the SQL references the metadata table and correct object type
        self.assertIn("STREAM_SOURCE", sql)
        self.assertIn("STREAM_SOURCE_CONFIG", sql)
        self.assertIn("TEST_DB.TEST_SCHEMA.__FEATURE_STORE_METADATA__", sql)
        self.assertIn("NAME", sql)
        self.assertNotIn("TIMESTAMP_COL", sql)


class StreamSourceCaseSensitivityTest(absltest.TestCase):
    """Tests for case sensitivity of stream source names through the FeatureStore layer."""

    def test_register_quoted_name_preserves_case_in_metadata_key(self) -> None:
        """Test that a quoted stream source name preserves its case in the metadata manager call."""
        fs = _create_feature_store_with_mocks()
        ss = StreamSource(
            '"myMixed"',
            _make_schema(),
            desc="quoted",
        )

        fs._metadata_manager.stream_source_exists.return_value = False
        # The name stored in JSON is the str(SqlIdentifier) which includes quotes for quoted ids
        metadata = ss._to_dict()
        metadata["owner"] = "ROLE_1"
        fs._metadata_manager.get_stream_source_metadata.return_value = metadata

        result = fs.register_stream_source(ss)

        # save_stream_source should receive the resolved() name: "myMixed" (case-preserving, no quotes)
        call_args = fs._metadata_manager.save_stream_source.call_args
        self.assertEqual(call_args.kwargs["name"], "myMixed")

        # The returned StreamSource should preserve the quoted identity
        self.assertEqual(result.name.resolved(), "myMixed")

    def test_register_unquoted_name_uppercased_in_metadata_key(self) -> None:
        """Test that an unquoted stream source name is uppercased in the metadata manager call."""
        fs = _create_feature_store_with_mocks()
        ss = _make_stream_source("mySource")

        fs._metadata_manager.stream_source_exists.return_value = False
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata(name="MYSOURCE")

        fs.register_stream_source(ss)

        # save_stream_source should receive the uppercased resolved() name
        call_args = fs._metadata_manager.save_stream_source.call_args
        self.assertEqual(call_args.kwargs["name"], "MYSOURCE")

    def test_get_quoted_name_uses_case_preserving_key(self) -> None:
        """Test that get_stream_source uses the case-preserving resolved name for lookup."""
        fs = _create_feature_store_with_mocks()

        # The stored JSON name field contains the str(SqlIdentifier) value — WITH quotes for quoted ids
        metadata = {
            "name": '"myMixed"',
            "schema": _schema_to_dict(_make_schema()),
            "desc": "quoted",
            "owner": "ROLE_1",
        }
        fs._metadata_manager.get_stream_source_metadata.return_value = metadata

        result = fs.get_stream_source('"myMixed"')

        # Verify the metadata lookup used the resolved quoted name (no quotes, case-preserved)
        call_args = fs._metadata_manager.get_stream_source_metadata.call_args[0]
        self.assertEqual(call_args[0], "myMixed")

        self.assertEqual(result.name.resolved(), "myMixed")

    def test_delete_quoted_name_uses_case_preserving_key(self) -> None:
        """Test that delete_stream_source uses the case-preserving resolved name for lookup."""
        fs = _create_feature_store_with_mocks()
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_ref_count.return_value = 0

        fs.delete_stream_source('"myMixed"')

        # delete_stream_source_metadata should receive the case-preserving resolved name
        fs._metadata_manager.delete_stream_source_metadata.assert_called_once_with("myMixed")

    def test_quoted_and_unquoted_resolve_differently(self) -> None:
        """Test that quoted '"abc"' and unquoted 'abc' resolve to different lookup keys."""
        # Unquoted: "abc" -> SqlIdentifier("abc").resolved() -> "ABC"
        unquoted_key = SqlIdentifier("abc").resolved()
        self.assertEqual(unquoted_key, "ABC")

        # Quoted: '"abc"' -> SqlIdentifier('"abc"').resolved() -> "abc"
        quoted_key = SqlIdentifier('"abc"').resolved()
        self.assertEqual(quoted_key, "abc")

        # They are distinct
        self.assertNotEqual(unquoted_key, quoted_key)


class StreamSourceCRUDIntegrationTest(absltest.TestCase):
    """End-to-end CRUD flow tests using mocked dependencies."""

    def test_full_lifecycle(self) -> None:
        """Test the complete lifecycle: register -> get -> update -> delete."""
        fs = _create_feature_store_with_mocks()
        ss = _make_stream_source()

        # 1. Register
        fs._metadata_manager.stream_source_exists.return_value = False
        metadata_v1 = _make_metadata(owner="ROLE_1")
        fs._metadata_manager.get_stream_source_metadata.return_value = metadata_v1

        result = fs.register_stream_source(ss)
        self.assertEqual(result.owner, "ROLE_1")

        # 2. Get
        result = fs.get_stream_source("txn_events")
        self.assertEqual(result.name, SqlIdentifier("TXN_EVENTS"))

        # 3. Update description
        fs._metadata_manager.stream_source_exists.return_value = True
        metadata_v2 = _make_metadata(desc="Updated", owner="ROLE_1")
        fs._metadata_manager.get_stream_source_metadata.return_value = metadata_v2

        updated = fs.update_stream_source("txn_events", desc="Updated")
        self.assertEqual(updated.desc, "Updated")

        # 4. Delete (no references)
        fs._metadata_manager.get_stream_source_ref_count.return_value = 0
        fs.delete_stream_source("txn_events")
        fs._metadata_manager.delete_stream_source_metadata.assert_called_once_with("TXN_EVENTS")

    def test_register_then_delete_blocked_by_reference(self) -> None:
        """Test that delete is blocked when ref_count > 0."""
        fs = _create_feature_store_with_mocks()
        ss = _make_stream_source()

        # Register
        fs._metadata_manager.stream_source_exists.return_value = False
        fs._metadata_manager.get_stream_source_metadata.return_value = _make_metadata()
        fs.register_stream_source(ss)

        # Try to delete with active reference
        fs._metadata_manager.stream_source_exists.return_value = True
        fs._metadata_manager.get_stream_source_ref_count.return_value = 1

        with self.assertRaises(ValueError) as cm:
            fs.delete_stream_source("txn_events")
        self.assertIn("1 active reference(s)", str(cm.exception))

        # Remove reference, then delete succeeds
        fs._metadata_manager.get_stream_source_ref_count.return_value = 0
        fs.delete_stream_source("txn_events")


if __name__ == "__main__":
    absltest.main()
