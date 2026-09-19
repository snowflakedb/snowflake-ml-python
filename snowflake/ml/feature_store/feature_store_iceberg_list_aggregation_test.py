"""Unit tests for Iceberg feature views on the create and update paths.

The compatibility gate refuses what Iceberg cannot express. ``last_n`` / ``first_n`` /
``last_distinct_n`` / ``first_distinct_n`` keep their per-tile state in ARRAY columns, which
an Iceberg table cannot store. The combination is refused when a feature view is created or
updated rather than when it is defined, so that an already registered feature view stays
readable. The same gate covers the other combinations Iceberg cannot express — rollup feature
views, ``approx_percentile``, and the streaming feature views whose Iceberg rules would
otherwise only be checked after the landing tables exist.

What survives the gate is created as a Dynamic Iceberg Table from the feature view's query
with no per-column types. Iceberg sources and streaming landing tables must already store
timestamps as ``TIMESTAMP_NTZ(6)``; the offline object infers that scale from the query.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
    StorageConfig,
    StorageFormat,
)
from snowflake.snowpark.types import (
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)

_LIST_FUNCTIONS = ("last_n", "first_n", "last_distinct_n", "first_distinct_n")

_DDL_EXTERNAL_VOLUME = "MY_VOLUME"
_DDL_BASE_LOCATION = "base/location"


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
    fs._asof_join_enabled = None
    return fs


def _make_tiled_fv(features: list[Feature], *, iceberg: bool) -> FeatureView:
    """Build a registered-looking tiled FeatureView, optionally Iceberg-backed.

    Args:
        features: Aggregation features for the tiled feature view.
        iceberg: When True, attach an Iceberg storage config.

    Returns:
        A tiled FeatureView with version/status/database/schema populated.
    """
    mock_df = MagicMock()
    mock_df.columns = ["USER_ID", "EVENT_TS", "AMOUNT"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}

    fv = FeatureView(
        name="TILED_FV",
        entities=[Entity(name="user", join_keys=["USER_ID"])],
        feature_df=mock_df,
        timestamp_col="EVENT_TS",
        refresh_freq="1h",
        feature_granularity="1h",
        features=features,
        storage_config=(
            StorageConfig(format=StorageFormat.ICEBERG, external_volume="VOL", base_location="loc/")
            if iceberg
            else None
        ),
    )
    fv._version = FeatureViewVersion("V1")
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("TEST_DB")
    fv._schema = SqlIdentifier("TEST_SCHEMA")
    return fv


def _distinct_n_features() -> list[Feature]:
    """Return the distinct-N feature pair used by most cases.

    Returns:
        A last/first distinct-N feature list.
    """
    return [
        Feature.last_distinct_n("AMOUNT", "24h", n=3).alias("RECENT"),
        Feature.first_distinct_n("AMOUNT", "24h", n=3).alias("EARLIEST"),
    ]


def _make_ddl_fv(
    schema: StructType,
    *,
    iceberg: bool,
    is_tiled: bool = False,
) -> MagicMock:
    """Build a feature-view stand-in exposing just what the DDL builders read.

    Every attribute the builders touch is set, including the ones that do not vary between
    cases, so the emitted statement can be compared in full — an attribute left unset would
    render as a mock repr in the middle of the DDL.

    Args:
        schema: Output schema of the offline object.
        iceberg: When True, attach an Iceberg storage config.
        is_tiled: Whether this feature view materializes from a tile query.

    Returns:
        A MagicMock standing in for a registered FeatureView.
    """
    fv = MagicMock(spec=FeatureView)
    fv.output_schema = schema
    fv.feature_descs = {}
    fv.is_tiled = is_tiled
    fv.desc = ""
    fv.refresh_freq = "1 minute"
    fv.refresh_mode = "AUTO"
    fv.initialize = "ON_CREATE"
    fv.initialization_warehouse = None
    fv.cluster_by = None
    fv.query = "SELECT * FROM source"
    fv._get_tile_query.return_value = "SELECT * FROM tiles"
    fv.storage_config = (
        StorageConfig(
            format=StorageFormat.ICEBERG,
            external_volume=_DDL_EXTERNAL_VOLUME,
            base_location=_DDL_BASE_LOCATION,
        )
        if iceberg
        else None
    )
    return fv


def _normalize_sql(query: str) -> str:
    """Collapse the DDL's layout whitespace so a whole statement fits one expected string.

    Args:
        query: Emitted SQL.

    Returns:
        The statement with every run of whitespace reduced to a single space.
    """
    return " ".join(query.split())


class RegisterFeatureViewIcebergListAggregationTest(parameterized.TestCase):
    """Tests for the register_feature_view gate."""

    def _install_registration_mocks(self, fs: Any, fv: FeatureView) -> None:
        """Patch the registration side effects so a gated call can be told apart from a real one.

        ``_get_feature_view_if_exists`` is made to raise so ``register_feature_view`` does not
        take its "already exists, skip registration" early return — otherwise a test that
        expects registration to proceed would pass without ever reaching the materialization
        step the gate is supposed to precede.

        Args:
            fs: Mocked feature store.
            fv: Feature view returned by the mocked getter.
        """
        fs._validate_entity_exists = MagicMock(return_value=True)
        fs._get_feature_view_if_exists = MagicMock(side_effect=RuntimeError("FeatureView does not exist."))
        fs._materialize_feature_view_resources = MagicMock()
        fs._create_offline_feature_view = MagicMock(return_value=[])
        fs._finalize_feature_view_registration = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

    @parameterized.parameters(*_LIST_FUNCTIONS)  # type: ignore[misc]
    def test_register_rejects_iceberg_list_aggregation(self, function_name: str) -> None:
        """Every ordered-N list aggregation is refused on Iceberg storage."""
        fs = _create_feature_store_with_mocks()
        feature = getattr(Feature, function_name)("AMOUNT", "24h", n=3).alias("RECENT")
        fv = _make_tiled_fv([feature], iceberg=True)
        fv._status = FeatureViewStatus.DRAFT
        self._install_registration_mocks(fs, fv)

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1")

        self.assertIn(f"Iceberg storage is not supported for the {function_name}", str(cm.exception))
        # The gate needs no round trip, so it must fire before the entity lookup and well
        # before anything is created in Snowflake.
        fs._validate_entity_exists.assert_not_called()
        fs._materialize_feature_view_resources.assert_not_called()
        fs._create_offline_feature_view.assert_not_called()
        fs._finalize_feature_view_registration.assert_not_called()

    def test_register_with_overwrite_rejects_iceberg_list_aggregation(self) -> None:
        """Re-registering with overwrite=True is the supported way to change storage format,
        so it must be gated too rather than dropping and recreating into an invalid table."""
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv(_distinct_n_features(), iceberg=True)
        fv._status = FeatureViewStatus.DRAFT
        self._install_registration_mocks(fs, fv)

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1", overwrite=True)

        self.assertIn(
            "Iceberg storage is not supported for the first_distinct_n, last_distinct_n aggregation",
            str(cm.exception),
        )
        fs._materialize_feature_view_resources.assert_not_called()
        fs._create_offline_feature_view.assert_not_called()

    def test_register_rejects_iceberg_rollup(self) -> None:
        """Rollup feature views are refused on Iceberg storage before anything is created.

        ``_create_rollup_feature_view`` emits a plain ``CREATE DYNAMIC TABLE`` and ignores
        ``storage_config``, so without the gate registration would succeed while producing a
        native Dynamic Table tagged ``is_iceberg=True`` in metadata — a state
        ``get_feature_view`` cannot load and ``delete_feature_view`` cannot clean up.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv([Feature.sum("AMOUNT", "24h").alias("SPEND")], iceberg=True)
        fv._status = FeatureViewStatus.DRAFT
        fv._rollup_metadata = MagicMock()
        self._install_registration_mocks(fs, fv)

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1")

        self.assertIn("Iceberg storage is not supported for rollup feature views", str(cm.exception))
        fs._validate_entity_exists.assert_not_called()
        fs._materialize_feature_view_resources.assert_not_called()
        fs._create_offline_feature_view.assert_not_called()
        fs._finalize_feature_view_registration.assert_not_called()

    def test_register_allows_native_list_aggregation(self) -> None:
        """The gate is Iceberg-specific: native storage still registers list aggregations.

        Asserts registration reaches ``_materialize_feature_view_resources`` — the shared
        dispatch point for the offline object and the online feature table — so the test
        would fail if the gate ever rejected native storage or moved below materialization.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv(_distinct_n_features(), iceberg=False)
        fv._status = FeatureViewStatus.DRAFT
        self._install_registration_mocks(fs, fv)

        fs.register_feature_view(feature_view=fv, version="V1")

        fs._validate_entity_exists.assert_called_once()
        fs._materialize_feature_view_resources.assert_called_once()

    def test_register_allows_iceberg_representable_aggregations(self) -> None:
        """Scalar tiles and the BINARY HLL sketch tile map to Iceberg types, so they register.

        ``approx_percentile`` is excluded — its t-Digest tile is an OBJECT; see
        ``test_register_rejects_iceberg_approx_percentile``.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv(
            [
                Feature.sum("AMOUNT", "24h").alias("SPEND"),
                Feature.count("AMOUNT", "24h").alias("TXN_COUNT"),
                Feature.avg("AMOUNT", "24h").alias("AVG_AMOUNT"),
                Feature.min("AMOUNT", "24h").alias("MIN_AMOUNT"),
                Feature.max("AMOUNT", "24h").alias("MAX_AMOUNT"),
                Feature.stddev("AMOUNT", "24h").alias("STD_AMOUNT"),
                Feature.var("AMOUNT", "24h").alias("VAR_AMOUNT"),
                Feature.approx_count_distinct("AMOUNT", "24h").alias("UNIQUES"),
            ],
            iceberg=True,
        )
        fv._status = FeatureViewStatus.DRAFT
        self._install_registration_mocks(fs, fv)

        fs.register_feature_view(feature_view=fv, version="V1")

        fs._validate_entity_exists.assert_called_once()
        fs._materialize_feature_view_resources.assert_called_once()

    def test_register_rejects_iceberg_approx_percentile(self) -> None:
        """``approx_percentile`` is refused on Iceberg before anything is created.

        Its t-Digest tile column is an OBJECT, which ``CREATE ICEBERG TABLE`` rejects with
        "Unsupported data type 'OBJECT' for iceberg tables". Gated so the caller gets a
        domain error rather than that compilation error mid-registration.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv(
            [Feature.approx_percentile("AMOUNT", "24h", percentile=0.5).alias("P50")],
            iceberg=True,
        )
        fv._status = FeatureViewStatus.DRAFT
        self._install_registration_mocks(fs, fv)

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1")

        self.assertIn("not supported for the approx_percentile aggregation", str(cm.exception))
        fs._validate_entity_exists.assert_not_called()
        fs._materialize_feature_view_resources.assert_not_called()
        fs._create_offline_feature_view.assert_not_called()
        fs._finalize_feature_view_registration.assert_not_called()


class RegisterStreamingFeatureViewIcebergTest(absltest.TestCase):
    """Tests that streaming Iceberg rules are enforced before the landing tables are created."""

    def _make_streaming_fv(self, **kwargs: Any) -> FeatureView:
        """Build a draft Iceberg-backed streaming FeatureView.

        Args:
            kwargs: Extra ``FeatureView`` arguments (e.g. ``refresh_freq``, ``online_config``).

        Returns:
            A streaming FeatureView with Iceberg storage.
        """
        from snowflake.ml.feature_store.stream_config import StreamConfig

        def _identity(df: Any) -> Any:
            return df

        backfill_df = MagicMock()
        backfill_df.columns = ["USER_ID", "EVENT_TS", "AMOUNT"]
        backfill_df.queries = {"queries": ["SELECT * FROM source"]}

        return FeatureView(
            name="STREAMING_FV",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            stream_config=StreamConfig(
                stream_source="txn_events",
                transformation_fn=_identity,
                backfill_df=backfill_df,
            ),
            timestamp_col="EVENT_TS",
            storage_config=StorageConfig(format=StorageFormat.ICEBERG, external_volume="VOL", base_location="loc/"),
            **kwargs,
        )

    def _assert_rejected_before_any_snowflake_work(self, fs: Any) -> None:
        """Assert the gate fired ahead of every step that touches Snowflake.

        ``run_streaming_preamble`` creates ``$UDF_TRANSFORMED`` and ``$BACKFILL``; asserting
        the session was never used covers the DDL it would issue and anything else on the way.

        Args:
            fs: Mocked feature store whose registration was expected to be gated.
        """
        fs._validate_entity_exists.assert_not_called()
        fs._materialize_feature_view_resources.assert_not_called()
        fs._session.sql.assert_not_called()

    def test_register_allows_iceberg_streaming_without_refresh_freq(self) -> None:
        """An Iceberg streaming feature view without a refresh cadence gets past the gate.

        Its offline object is a View, but its ``$UDF_TRANSFORMED`` and ``$BACKFILL`` landing
        tables still hold the rows and are created as Iceberg tables, so the request is
        meaningful. Entity validation is the first step after the gate, so failing it is how
        this asserts the gate was cleared without running the rest of registration.
        """
        fs = _create_feature_store_with_mocks()
        fv = self._make_streaming_fv()
        fs._validate_entity_exists = MagicMock(return_value=False)
        fs._materialize_feature_view_resources = MagicMock()

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1")

        self.assertIn("has not been registered", str(cm.exception))
        self.assertNotIn("Iceberg storage requires refresh_freq", str(cm.exception))
        fs._validate_entity_exists.assert_called_once()
        fs._materialize_feature_view_resources.assert_not_called()

    def test_register_rejects_iceberg_streaming_with_hybrid_table_online(self) -> None:
        """An Iceberg streaming feature view on a hybrid-table online store is refused up front."""
        fs = _create_feature_store_with_mocks()
        fv = self._make_streaming_fv(
            refresh_freq="1h",
            online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.HYBRID_TABLE),
        )
        fs._validate_entity_exists = MagicMock(return_value=True)
        fs._materialize_feature_view_resources = MagicMock()

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1")

        self.assertIn("Iceberg offline storage is only supported with Postgres online store type.", str(cm.exception))
        self._assert_rejected_before_any_snowflake_work(fs)


class UpdateFeatureViewIcebergListAggregationTest(absltest.TestCase):
    """Tests for the update_feature_view gate."""

    def test_update_rejects_feature_view_switched_to_iceberg(self) -> None:
        """A caller-supplied FeatureView is used as passed rather than reloaded from the backend,
        so an object mutated to Iceberg storage can still reach update and must be refused."""
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv(_distinct_n_features(), iceberg=False)
        # Switch the registered feature view over to Iceberg the only way a caller can.
        fv._storage_config = StorageConfig(format=StorageFormat.ICEBERG, external_volume="VOL")

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))

        with self.assertRaises(Exception) as cm:
            fs.update_feature_view(name=fv, desc="new description")

        self.assertIn(
            "Iceberg storage is not supported for the first_distinct_n, last_distinct_n aggregation",
            str(cm.exception),
        )
        fs._plan_feature_view_update_operations.assert_not_called()

    def test_update_allows_native_list_aggregation(self) -> None:
        """An unmodified native list-aggregation feature view still updates.

        Asserts the update reaches ``_plan_feature_view_update_operations`` rather than only
        that the gate does not raise, so the test would fail if the gate rejected native
        storage or moved below the planning step.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_tiled_fv(_distinct_n_features(), iceberg=False)

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))
        fs.get_feature_view = MagicMock(return_value=fv)

        fs.update_feature_view(name=fv, desc="new description")

        fs._plan_feature_view_update_operations.assert_called_once()


class CreateDynamicIcebergTableQueryTest(absltest.TestCase):
    """Tests the whole CREATE statement registration would send for each storage format."""

    _SCHEMA = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
        ]
    )

    def _query(self, *, iceberg: bool, is_tiled: bool = False) -> str:
        """Render the offline-object DDL.

        Args:
            iceberg: Whether the feature view uses Iceberg storage.
            is_tiled: Whether the feature view materializes from a tile query.

        Returns:
            The emitted CREATE statement, with layout whitespace collapsed.
        """
        from snowflake.ml.feature_store.feature_store import FeatureStore

        fs = _create_feature_store_with_mocks()
        fv = _make_ddl_fv(self._SCHEMA, iceberg=iceberg, is_tiled=is_tiled)
        # Registration passes the comment-only clause for both native and Iceberg DTs.
        # A tiled feature view is materialized without a column list at all.
        column_descs = "" if is_tiled else FeatureStore._build_column_descs(fv)
        return _normalize_sql(
            fs._create_dynamic_table_query(
                override_clause="",
                table_name="TEST_DB.TEST_SCHEMA.MY_FV$V1",
                column_descs=column_descs,
                schedule_task=False,
                feature_view=fv,
                tagging_clause="TAG_A = 'x'",
                warehouse="WH_1",
            )
        )

    def test_iceberg_ddl_leaves_columns_inferred(self) -> None:
        """The Dynamic Iceberg Table infers column types from the feature view query.

        Iceberg sources and streaming landing tables already store timestamps as
        ``TIMESTAMP_NTZ(6)``, so the offline object does not redeclare a scale.
        """
        self.assertEqual(
            self._query(iceberg=True),
            "CREATE DYNAMIC ICEBERG TABLE TEST_DB.TEST_SCHEMA.MY_FV$V1 "
            "(USER_ID , EVENT_TIME ) "
            "TARGET_LAG = '1 minute' "
            "COMMENT = '' "
            "TAG ( TAG_A = 'x' ) "
            "WAREHOUSE = WH_1 "
            "REFRESH_MODE = AUTO "
            "INITIALIZE = ON_CREATE "
            "CATALOG = 'SNOWFLAKE' "
            "EXTERNAL_VOLUME = MY_VOLUME "
            "BASE_LOCATION = 'base/location' "
            "AS SELECT * FROM source",
        )

    def test_native_ddl_leaves_timestamp_inferred(self) -> None:
        """A native Dynamic Table keeps inferring its columns.

        Declaring the Iceberg scale here would silently truncate sub-microsecond event
        times for every existing non-Iceberg feature view.
        """
        self.assertEqual(
            self._query(iceberg=False),
            "CREATE DYNAMIC TABLE TEST_DB.TEST_SCHEMA.MY_FV$V1 "
            "(USER_ID , EVENT_TIME ) "
            "TARGET_LAG = '1 minute' "
            "COMMENT = '' "
            "TAG ( TAG_A = 'x' ) "
            "WAREHOUSE = WH_1 "
            "REFRESH_MODE = AUTO "
            "INITIALIZE = ON_CREATE "
            "AS SELECT * FROM source",
        )

    def test_tiled_iceberg_ddl_has_no_column_clause(self) -> None:
        """A tiled feature view keeps its cast-in-query narrowing and gets no column list."""
        self.assertEqual(
            self._query(iceberg=True, is_tiled=True),
            "CREATE DYNAMIC ICEBERG TABLE TEST_DB.TEST_SCHEMA.MY_FV$V1 "
            "TARGET_LAG = '1 minute' "
            "COMMENT = '' "
            "TAG ( TAG_A = 'x' ) "
            "WAREHOUSE = WH_1 "
            "REFRESH_MODE = AUTO "
            "INITIALIZE = ON_CREATE "
            "CATALOG = 'SNOWFLAKE' "
            "EXTERNAL_VOLUME = MY_VOLUME "
            "BASE_LOCATION = 'base/location' "
            "AS SELECT * FROM tiles",
        )


if __name__ == "__main__":
    absltest.main()
