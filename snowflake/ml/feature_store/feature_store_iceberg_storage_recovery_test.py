"""Unit tests for recovering an Iceberg feature view's storage config on the read path.

``SHOW ICEBERG TABLES`` reports the physical path Snowflake wrote to — the requested
``BASE_LOCATION`` with a ``.<randomId>`` appended — rather than the value the feature
view was registered with. The read path copies that SHOW value as-is.

For a streaming feature view without a refresh cadence the offline object is a View,
which has no Iceberg form and never appears in that listing, so the fallback reads
the ``$UDF_TRANSFORMED`` landing table and reports its SHOW path
(``{registered}_UDF_TRANSFORMED.{randomId}``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    StorageConfig,
    StorageFormat,
    _FeatureViewMetadata,
)

_EXTERNAL_VOLUME = "MY_VOLUME"
_FV_BASE_LOCATION = "snowflake/feature_store/iceberg/DB/SCH/STREAM_FV$V1"
_LANDING_PHYSICAL_LOCATION = f"{_FV_BASE_LOCATION}_UDF_TRANSFORMED.a1B2c3D4"


def _create_feature_store_with_mocks() -> Any:
    """Create a FeatureStore with mocked dependencies (bypassing __init__).

    Note: This manually sets internal attributes. Update if FeatureStore.__init__ changes.

    Returns:
        A FeatureStore instance wired with mocks.
    """
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


def _landing_table_config() -> StorageConfig:
    """Return the config as ``SHOW ICEBERG TABLES`` reports it for ``$UDF_TRANSFORMED``.

    Returns:
        An Iceberg StorageConfig carrying the landing table's physical base location.
    """
    return StorageConfig(
        format=StorageFormat.ICEBERG,
        external_volume=_EXTERNAL_VOLUME,
        base_location=_LANDING_PHYSICAL_LOCATION,
    )


class StorageConfigBaseLocationCanonicalizationTest(absltest.TestCase):
    """Tests that ``base_location`` is canonical, which is what makes the round trip work."""

    def test_strips_trailing_separator(self) -> None:
        """``"a/b/"`` and ``"a/b"`` name the same storage, so they must compare equal."""
        with_separator = StorageConfig(format=StorageFormat.ICEBERG, base_location="a/b/")
        without_separator = StorageConfig(format=StorageFormat.ICEBERG, base_location="a/b")

        self.assertEqual(with_separator.base_location, "a/b")
        self.assertEqual(with_separator, without_separator)

    def test_strips_repeated_trailing_separators(self) -> None:
        """Every trailing separator is removed, not just the last one."""
        self.assertEqual(StorageConfig(base_location="a/b///").base_location, "a/b")

    def test_leaves_separator_only_base_location_alone(self) -> None:
        """A separator-only value is not rewritten to an empty string."""
        self.assertEqual(StorageConfig(base_location="/").base_location, "/")

    def test_leaves_missing_base_location_alone(self) -> None:
        """A config with no base location stays ``None``."""
        self.assertIsNone(StorageConfig(format=StorageFormat.ICEBERG).base_location)

    def test_from_json_canonicalizes(self) -> None:
        """A config read back from a metadata tag is canonical too."""
        restored = StorageConfig.from_json('{"format": "iceberg", "base_location": "a/b/"}')

        self.assertEqual(restored.base_location, "a/b")


class ComposeFeatureViewIcebergRecoveryTest(absltest.TestCase):
    """Tests for the ``get_feature_view`` recovery path."""

    def _run_compose(
        self,
        *,
        is_streaming: bool,
        landing_table_exists: bool,
    ) -> Any:
        """Reconstruct a View-backed Iceberg feature view through ``_compose_feature_view``.

        Args:
            is_streaming: Value recorded on the metadata tag.
            landing_table_exists: When True, ``SHOW ICEBERG TABLES`` finds ``$UDF_TRANSFORMED``.

        Returns:
            The reconstructed FeatureView.
        """
        from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

        fs = _create_feature_store_with_mocks()

        metadata = _FeatureViewMetadata(
            entities=["USER"],
            timestamp_col="EVENT_TIME",
            is_iceberg=True,
            is_streaming=is_streaming,
        )

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "name": "STREAM_FV$V1",
            "comment": "test",
            "owner": "ROLE_1",
        }[key]

        entity_row = MagicMock()
        entity_row.__getitem__ = lambda self, key: {
            "NAME": "USER",
            "JOIN_KEYS": '["USER_ID"]',
            "DESC": "",
        }[key]

        mock_df = MagicMock()
        mock_df.columns = ["USER_ID", "EVENT_TIME", "AMOUNT"]
        mock_df.queries = {"queries": ["SELECT 1"]}

        from snowflake.snowpark.types import TimestampType

        ts_field = MagicMock()
        ts_field.datatype = TimestampType()
        mock_df.schema.__getitem__ = lambda self, key: ts_field

        fs._session.sql.return_value = mock_df

        # The offline object is a View, so it is absent from SHOW ICEBERG TABLES; only the
        # landing table resolves, and only when the test says it exists.
        landing_name = FeatureView._get_udf_transformed_table_name(SqlIdentifier("STREAM_FV$V1")).resolved()

        def lookup(table_name: SqlIdentifier) -> StorageConfig | None:
            if landing_table_exists and table_name.resolved() == landing_name:
                return _landing_table_config()
            return None

        fs._get_iceberg_storage_config = MagicMock(side_effect=lookup)
        fs._lookup_feature_view_metadata = MagicMock(return_value=(metadata, "SELECT 1"))
        fs._determine_online_config_from_oft = MagicMock(return_value='{"enable": false}')
        fs._fetch_column_descs = MagicMock(return_value={})
        fs._hydrate_postgres_online_service = MagicMock()
        fs._extract_cluster_by_columns = MagicMock(return_value=None)
        fs._metadata_manager.get_feature_specs.return_value = None
        fs._metadata_manager.get_feature_view_source_refs.return_value = None
        fs._metadata_manager.get_streaming_metadata.return_value = None

        return fs._compose_feature_view(mock_row, _FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW, [entity_row])

    def test_view_backed_streaming_fv_recovers_config_from_landing_table(self) -> None:
        """A View-backed streaming Iceberg feature view reports the landing-table SHOW path.

        This is also what keeps such a feature view deletable: ``delete_feature_view`` reloads
        it first, so a failure here would strand it.
        """
        fv = self._run_compose(is_streaming=True, landing_table_exists=True)

        assert fv.storage_config is not None
        self.assertEqual(fv.storage_config.format, StorageFormat.ICEBERG)
        self.assertEqual(fv.storage_config.external_volume, _EXTERNAL_VOLUME)
        self.assertEqual(fv.storage_config.base_location, _LANDING_PHYSICAL_LOCATION)
        self.assertIsNone(fv.refresh_freq)

    def test_batch_fv_does_not_fall_back_to_landing_table(self) -> None:
        """A batch feature view has no landing table, so the miss stays an error.

        Falling back for batch would silently invent a storage config from an unrelated
        table that happened to match the naming pattern.
        """
        with self.assertRaises(Exception) as cm:
            self._run_compose(is_streaming=False, landing_table_exists=True)

        self.assertIn("Failed to retrieve Iceberg storage config", str(cm.exception))

    def test_streaming_fv_without_landing_table_still_raises(self) -> None:
        """When neither lookup resolves, the original error is still surfaced."""
        with self.assertRaises(Exception) as cm:
            self._run_compose(is_streaming=True, landing_table_exists=False)

        self.assertIn("Failed to retrieve Iceberg storage config", str(cm.exception))


class ListFeatureViewsIcebergRecoveryTest(absltest.TestCase):
    """Tests for the ``list_feature_views`` recovery path."""

    def _storage_config_json(
        self,
        *,
        is_streaming: bool,
        landing_table_exists: bool,
    ) -> str:
        """Return the STORAGE_CONFIG value ``_extract_feature_view_info`` emits.

        Args:
            is_streaming: Value recorded on the metadata tag.
            landing_table_exists: When True, the schema listing includes ``$UDF_TRANSFORMED``.

        Returns:
            The JSON-encoded storage config from the emitted listing row.
        """
        from snowflake.ml.feature_store.feature_store import (
            _LIST_FEATURE_VIEW_BASE_FIELDS,
        )

        storage_config_index = next(
            i for i, field in enumerate(_LIST_FEATURE_VIEW_BASE_FIELDS) if field.name.upper() == "STORAGE_CONFIG"
        )
        fs = _create_feature_store_with_mocks()

        metadata = _FeatureViewMetadata(
            entities=["USER"],
            timestamp_col="EVENT_TIME",
            is_iceberg=True,
            is_streaming=is_streaming,
        )

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "name": "STREAM_FV$V1",
            "database_name": "TEST_DB",
            "schema_name": "TEST_SCHEMA",
            "created_on": "2024-01-01",
            "owner": "ROLE_1",
            "comment": "test",
        }[key]

        # SHOW ICEBERG TABLES IN SCHEMA already covers the whole schema, so the landing table
        # is in the same cache the offline lookup missed on — the fallback costs no extra query.
        landing_name = FeatureView._get_udf_transformed_table_name(SqlIdentifier("STREAM_FV$V1")).resolved()
        schema_configs = {landing_name: _landing_table_config()} if landing_table_exists else {}

        fs._lookup_feature_view_metadata = MagicMock(return_value=(metadata, "SELECT 1"))
        fs._determine_online_config_from_oft = MagicMock(return_value='{"enable": false}')
        fs._get_all_iceberg_storage_configs = MagicMock(return_value=schema_configs)
        fs._metadata_manager.get_streaming_metadata.return_value = None
        fs._metadata_manager.get_feature_view_source_refs.return_value = None

        output_values: list[list[Any]] = []
        fs._extract_feature_view_info(mock_row, output_values, [], {})

        self.assertLen(output_values, 1)
        storage_config_json = output_values[0][storage_config_index]
        assert isinstance(storage_config_json, str)
        return storage_config_json

    def test_view_backed_streaming_fv_reports_landing_table_base_location(self) -> None:
        """The listing shows the landing-table SHOW path, not the registered folder."""
        storage_config_json = self._storage_config_json(is_streaming=True, landing_table_exists=True)

        storage_config = StorageConfig.from_json(storage_config_json)
        self.assertEqual(storage_config.format, StorageFormat.ICEBERG)
        self.assertEqual(storage_config.external_volume, _EXTERNAL_VOLUME)
        self.assertEqual(storage_config.base_location, _LANDING_PHYSICAL_LOCATION)

    def test_batch_fv_does_not_fall_back_to_landing_table(self) -> None:
        """A batch feature view keeps the existing degraded output instead of falling back."""
        with self.assertLogs(level="WARNING"):
            storage_config_json = self._storage_config_json(is_streaming=False, landing_table_exists=True)

        storage_config = StorageConfig.from_json(storage_config_json)
        self.assertEqual(storage_config.format, StorageFormat.ICEBERG)
        self.assertIsNone(storage_config.base_location)

    def test_streaming_fv_without_landing_table_stays_degraded(self) -> None:
        """When neither lookup resolves, the listing still warns and emits the default."""
        with self.assertLogs(level="WARNING"):
            storage_config_json = self._storage_config_json(is_streaming=True, landing_table_exists=False)

        self.assertIsNone(StorageConfig.from_json(storage_config_json).base_location)


if __name__ == "__main__":
    absltest.main()
