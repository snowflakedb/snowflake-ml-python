"""Integration tests for FeatureStore stream source CRUD operations."""

from absl.testing import absltest, parameterized
from common_utils import create_random_schema
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark.types import (
    BooleanType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)


class FeatureStoreStreamSourceTest(FeatureStoreIntegTestBase, parameterized.TestCase):
    """Integration tests for stream source CRUD operations."""

    def setUp(self) -> None:
        super().setUp()
        self._active_feature_stores: list[FeatureStore] = []

    def tearDown(self) -> None:
        for fs in self._active_feature_stores:
            try:
                fs._clear(dryrun=False)
            except Exception:
                pass
            try:
                self._session.sql(f"DROP SCHEMA IF EXISTS {fs._config.full_schema_path}").collect()
            except Exception:
                pass
        super().tearDown()

    def _create_feature_store(self, name: str | None = None) -> FeatureStore:
        current_schema = create_random_schema(self._session, "SS_TEST", database=self.test_db) if name is None else name
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._active_feature_stores.append(fs)
        return fs

    def _make_stream_source(
        self,
        name: str = "txn_events",
        desc: str = "",
    ) -> StreamSource:
        return StreamSource(
            name=name,
            schema=StructType(
                [
                    StructField("user_id", StringType()),
                    StructField("amount", FloatType()),
                    StructField("merchant_id", StringType()),
                    StructField("event_time", TimestampType(TimestampTimeZone.NTZ)),
                ]
            ),
            desc=desc,
        )

    # =========================================================================
    # register_stream_source
    # =========================================================================

    def test_register_stream_source(self) -> None:
        """Test basic stream source registration."""
        fs = self._create_feature_store()
        ss = self._make_stream_source(desc="Real-time txn events")

        result = fs.register_stream_source(ss)

        self.assertIsInstance(result, StreamSource)
        self.assertEqual(result.name, SqlIdentifier("txn_events"))
        self.assertEqual(result.desc, "Real-time txn events")
        self.assertIsNotNone(result.owner)

    def test_register_duplicate_warns(self) -> None:
        """Test that registering a stream source with the same name warns and returns existing."""
        fs = self._create_feature_store()
        ss = self._make_stream_source()

        fs.register_stream_source(ss)

        # Register again with different schema — should warn and return existing
        ss2 = StreamSource(
            name="txn_events",
            schema=StructType(
                [
                    StructField("col_a", FloatType()),
                    StructField("ts", TimestampType(TimestampTimeZone.NTZ)),
                ]
            ),
            desc="different",
        )
        with self.assertWarnsRegex(UserWarning, "already exists"):
            result = fs.register_stream_source(ss2)

        # Original description should be preserved
        self.assertEqual(result.desc, "")

    def test_register_multiple_stream_sources(self) -> None:
        """Test registering multiple stream sources."""
        fs = self._create_feature_store()

        sources = [
            self._make_stream_source("src_a", desc="Source A"),
            self._make_stream_source("src_b", desc="Source B"),
            self._make_stream_source("src_c", desc="Source C"),
        ]
        for ss in sources:
            fs.register_stream_source(ss)

        result_df = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(result_df), 3)

    # =========================================================================
    # get_stream_source
    # =========================================================================

    def test_get_stream_source(self) -> None:
        """Test retrieving a registered stream source."""
        fs = self._create_feature_store()
        ss = self._make_stream_source(desc="Test desc")
        fs.register_stream_source(ss)

        result = fs.get_stream_source("txn_events")

        self.assertEqual(result.name, SqlIdentifier("txn_events"))
        self.assertEqual(result.desc, "Test desc")
        self.assertIsNotNone(result.owner)
        self.assertEqual(len(result.schema.fields), 4)

    def test_get_nonexistent_raises(self) -> None:
        """Test that getting a non-existent stream source raises ValueError."""
        fs = self._create_feature_store()

        with self.assertRaisesRegex(ValueError, "Cannot find StreamSource"):
            fs.get_stream_source("no_such_source")

    def test_get_case_insensitive(self) -> None:
        """Test that get resolves names case-insensitively."""
        fs = self._create_feature_store()
        fs.register_stream_source(self._make_stream_source("my_source"))

        # All of these should resolve to the same stream source
        r1 = fs.get_stream_source("my_source")
        r2 = fs.get_stream_source("MY_SOURCE")
        r3 = fs.get_stream_source("My_Source")

        self.assertEqual(r1, r2)
        self.assertEqual(r2, r3)

    # =========================================================================
    # list_stream_sources
    # =========================================================================

    def test_list_stream_sources_empty(self) -> None:
        """Test listing when no stream sources exist."""
        fs = self._create_feature_store()

        result_df = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(result_df), 0)
        # Verify schema
        expected_cols = {"NAME", "SCHEMA", "DESC", "OWNER"}
        self.assertEqual(set(result_df.columns), expected_cols)

    def test_list_stream_sources(self) -> None:
        """Test listing multiple registered stream sources."""
        fs = self._create_feature_store()

        fs.register_stream_source(self._make_stream_source("alpha", desc="First"))
        fs.register_stream_source(self._make_stream_source("beta", desc="Second"))

        result_df = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(result_df), 2)

        # Results should be ordered by NAME
        names = result_df["NAME"].tolist()
        self.assertEqual(names, ["ALPHA", "BETA"])

        # Check descriptions
        descs = dict(zip(result_df["NAME"], result_df["DESC"]))
        self.assertEqual(descs["ALPHA"], "First")
        self.assertEqual(descs["BETA"], "Second")

        # Owners should be non-empty
        for owner in result_df["OWNER"]:
            self.assertTrue(len(owner) > 0)

    # =========================================================================
    # delete_stream_source
    # =========================================================================

    def test_delete_stream_source(self) -> None:
        """Test deleting a stream source."""
        fs = self._create_feature_store()
        fs.register_stream_source(self._make_stream_source("to_delete"))

        # Verify it exists
        self.assertEqual(len(fs.list_stream_sources().collect()), 1)

        fs.delete_stream_source("to_delete")

        # Verify it's gone
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)
        with self.assertRaisesRegex(ValueError, "Cannot find StreamSource"):
            fs.get_stream_source("to_delete")

    def test_delete_nonexistent_raises(self) -> None:
        """Test that deleting a non-existent stream source raises ValueError."""
        fs = self._create_feature_store()

        with self.assertRaisesRegex(ValueError, "does not exist"):
            fs.delete_stream_source("no_such_source")

    def test_delete_with_active_references_blocked(self) -> None:
        """Test that deletion is blocked when ref_count > 0."""
        fs = self._create_feature_store()
        ss = self._make_stream_source("ref_source")
        fs.register_stream_source(ss)

        # Simulate a FeatureView registration incrementing the ref count
        fs._metadata_manager.increment_stream_source_ref_count("REF_SOURCE")

        # Verify the ref count
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("REF_SOURCE"), 1)

        # Delete should be blocked
        with self.assertRaisesRegex(ValueError, "active reference"):
            fs.delete_stream_source("ref_source")

        # Simulate FeatureView deletion decrementing the ref count
        fs._metadata_manager.decrement_stream_source_ref_count("REF_SOURCE")

        # Verify ref count is back to 0
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("REF_SOURCE"), 0)

        # Delete should now succeed
        fs.delete_stream_source("ref_source")
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)

    def test_increment_and_decrement_ref_count(self) -> None:
        """Test incrementing and decrementing the ref count on a stream source."""
        fs = self._create_feature_store()
        fs.register_stream_source(self._make_stream_source("multi_ref"))

        # Initially ref_count is 0
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("MULTI_REF"), 0)

        # Increment three times (simulating 3 FeatureView registrations)
        fs._metadata_manager.increment_stream_source_ref_count("MULTI_REF")
        fs._metadata_manager.increment_stream_source_ref_count("MULTI_REF")
        fs._metadata_manager.increment_stream_source_ref_count("MULTI_REF")

        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("MULTI_REF"), 3)

        # Delete should be blocked
        with self.assertRaisesRegex(ValueError, "active reference"):
            fs.delete_stream_source("multi_ref")

        # Decrement once
        fs._metadata_manager.decrement_stream_source_ref_count("MULTI_REF")
        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("MULTI_REF"), 2)

        # Still blocked
        with self.assertRaisesRegex(ValueError, "active reference"):
            fs.delete_stream_source("multi_ref")

        # Decrement remaining
        fs._metadata_manager.decrement_stream_source_ref_count("MULTI_REF")
        fs._metadata_manager.decrement_stream_source_ref_count("MULTI_REF")

        self.assertEqual(fs._metadata_manager.get_stream_source_ref_count("MULTI_REF"), 0)

        # Now delete succeeds
        fs.delete_stream_source("multi_ref")
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)

    # =========================================================================
    # update_stream_source
    # =========================================================================

    def test_update_stream_source_desc(self) -> None:
        """Test updating a stream source description."""
        fs = self._create_feature_store()
        fs.register_stream_source(self._make_stream_source("updatable", desc="Old desc"))

        result = fs.update_stream_source("updatable", desc="New desc")

        self.assertIsNotNone(result)
        self.assertEqual(result.desc, "New desc")

        # Verify via get
        fetched = fs.get_stream_source("updatable")
        self.assertEqual(fetched.desc, "New desc")

    def test_update_nonexistent_warns(self) -> None:
        """Test that updating a non-existent stream source warns and returns None."""
        fs = self._create_feature_store()

        with self.assertWarnsRegex(UserWarning, "does not exist"):
            result = fs.update_stream_source("no_such_source", desc="New")

        self.assertIsNone(result)

    def test_update_no_change(self) -> None:
        """Test update with no desc change returns the stream source unchanged."""
        fs = self._create_feature_store()
        fs.register_stream_source(self._make_stream_source("noop", desc="Original"))

        result = fs.update_stream_source("noop")

        self.assertIsNotNone(result)
        self.assertEqual(result.desc, "Original")

    # =========================================================================
    # Full lifecycle
    # =========================================================================

    def test_full_lifecycle(self) -> None:
        """Test the complete lifecycle: register → get → list → update → delete."""
        fs = self._create_feature_store()

        # 1. Register
        ss = self._make_stream_source("lifecycle", desc="v1")
        registered = fs.register_stream_source(ss)
        self.assertEqual(registered.name, SqlIdentifier("lifecycle"))
        self.assertEqual(registered.desc, "v1")
        self.assertIsNotNone(registered.owner)

        # 2. Get
        fetched = fs.get_stream_source("lifecycle")
        self.assertEqual(fetched.name, registered.name)
        self.assertEqual(fetched.desc, "v1")

        # 3. List — should have exactly one
        list_df = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(list_df), 1)
        self.assertEqual(list_df.iloc[0]["NAME"], "LIFECYCLE")

        # 4. Update description
        updated = fs.update_stream_source("lifecycle", desc="v2")
        self.assertEqual(updated.desc, "v2")

        # Verify the update persisted
        fetched2 = fs.get_stream_source("lifecycle")
        self.assertEqual(fetched2.desc, "v2")

        # 5. Delete
        fs.delete_stream_source("lifecycle")
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)
        with self.assertRaisesRegex(ValueError, "Cannot find StreamSource"):
            fs.get_stream_source("lifecycle")

    def test_case_sensitivity(self) -> None:
        """Test that stream source names are handled case-insensitively."""
        fs = self._create_feature_store()

        # Register with lowercase
        fs.register_stream_source(self._make_stream_source("my_src", desc="original"))

        # Duplicate with uppercase should warn
        with self.assertWarnsRegex(UserWarning, "already exists"):
            fs.register_stream_source(self._make_stream_source("MY_SRC", desc="dup"))

        # Get with mixed case should work
        r = fs.get_stream_source("My_Src")
        self.assertEqual(r.desc, "original")

        # Update with different casing should work
        updated = fs.update_stream_source("MY_SRC", desc="updated")
        self.assertEqual(updated.desc, "updated")

        # Delete with lowercase
        fs.delete_stream_source("my_src")
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)

    def test_quoted_identifier_stream_source(self) -> None:
        """Test stream source with quoted identifier preserves case.

        Quoted identifiers (e.g., '"myMixedCase"') preserve their original casing
        and are stored distinctly from unquoted identifiers, matching Snowflake's
        SQL identifier semantics.
        """
        fs = self._create_feature_store()

        ss = self._make_stream_source('"myMixedCase"')
        fs.register_stream_source(ss)

        # Get with quoted name works — case is preserved
        result = fs.get_stream_source('"myMixedCase"')
        self.assertIsNotNone(result)
        self.assertEqual(result.name.resolved(), "myMixedCase")

        # Unquoted "myMixedCase" resolves to "MYMIXEDCASE" — a different identifier
        with self.assertRaisesRegex(ValueError, "Cannot find StreamSource"):
            fs.get_stream_source("myMixedCase")

        # List shows the name from the metadata JSON (str(SqlIdentifier) includes quotes)
        list_df = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(list_df), 1)
        # SqlIdentifier('"myMixedCase"') serializes as "myMixedCase" in the JSON name field
        self.assertIn("myMixedCase", list_df.iloc[0]["NAME"])

        # Delete with quoted name
        fs.delete_stream_source('"myMixedCase"')
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)

    def test_quoted_and_unquoted_are_distinct(self) -> None:
        """Test that a quoted name and its unquoted equivalent are two separate stream sources."""
        fs = self._create_feature_store()

        # Register unquoted — stored as MYSRC
        fs.register_stream_source(self._make_stream_source("mysrc", desc="unquoted"))
        # Register quoted — stored as mysrc (case-preserved, distinct)
        fs.register_stream_source(self._make_stream_source('"mysrc"', desc="quoted"))

        # Both exist independently
        list_df = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(list_df), 2)

        # Retrieve each one
        r_unquoted = fs.get_stream_source("mysrc")
        self.assertEqual(r_unquoted.desc, "unquoted")
        self.assertEqual(r_unquoted.name.resolved(), "MYSRC")

        r_quoted = fs.get_stream_source('"mysrc"')
        self.assertEqual(r_quoted.desc, "quoted")
        self.assertEqual(r_quoted.name.resolved(), "mysrc")

        # They are not equal
        self.assertNotEqual(r_unquoted, r_quoted)

        # Delete unquoted, quoted should remain
        fs.delete_stream_source("mysrc")
        list_df2 = fs.list_stream_sources().to_pandas()
        self.assertEqual(len(list_df2), 1)

        r_remaining = fs.get_stream_source('"mysrc"')
        self.assertEqual(r_remaining.desc, "quoted")

        # Clean up
        fs.delete_stream_source('"mysrc"')
        self.assertEqual(len(fs.list_stream_sources().collect()), 0)

    def test_schema_roundtrip(self) -> None:
        """Test that the schema is preserved through register → get roundtrip."""
        fs = self._create_feature_store()
        original_schema = StructType(
            [
                StructField("user_id", StringType()),
                StructField("amount", FloatType()),
                StructField("is_active", BooleanType()),
                StructField("event_time", TimestampType(TimestampTimeZone.NTZ)),
            ]
        )
        ss = StreamSource("roundtrip", original_schema)
        fs.register_stream_source(ss)

        fetched = fs.get_stream_source("roundtrip")

        # Verify each field
        self.assertEqual(len(fetched.schema.fields), 4)
        for orig_field, fetched_field in zip(original_schema.fields, fetched.schema.fields):
            self.assertEqual(orig_field.name, fetched_field.name)
            self.assertEqual(type(orig_field.datatype), type(fetched_field.datatype))

    def test_independent_feature_stores_isolation(self) -> None:
        """Test that stream sources in different schemas are isolated."""
        fs1 = self._create_feature_store()
        fs2 = self._create_feature_store()

        fs1.register_stream_source(self._make_stream_source("shared_name", desc="from fs1"))
        fs2.register_stream_source(self._make_stream_source("shared_name", desc="from fs2"))

        r1 = fs1.get_stream_source("shared_name")
        r2 = fs2.get_stream_source("shared_name")

        self.assertEqual(r1.desc, "from fs1")
        self.assertEqual(r2.desc, "from fs2")

        # Each should have exactly one
        self.assertEqual(len(fs1.list_stream_sources().collect()), 1)
        self.assertEqual(len(fs2.list_stream_sources().collect()), 1)


if __name__ == "__main__":
    absltest.main()
