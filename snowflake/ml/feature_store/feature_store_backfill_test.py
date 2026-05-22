"""Unit tests for registration-time historical backfill and snapshot refresh."""

from __future__ import annotations

import logging
from typing import Any, Optional
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
)
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.types import (
    DataType,
    DecimalType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logger = logging.getLogger(__name__)


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
    fs._asof_join_enabled = None
    return fs


def _make_fv(*, backup_source: Optional[str] = None) -> FeatureView:
    """Build an append_only FeatureView with version/db/schema set."""
    mock_df = MagicMock()
    mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "N_RSRVS_30_DAY", "N_FUTURE_STAYS"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}

    ts_field = MagicMock()
    ts_field.datatype = TimestampType()
    mock_df.schema.__getitem__ = lambda self, key: ts_field

    entity = Entity(name="guest", join_keys=["GUEST_ID"])
    fv = FeatureView(
        name="GUEST_SNAPSHOT_FEATURES",
        entities=[entity],
        feature_df=mock_df,
        timestamp_col="SNAPSHOT_TS",
        refresh_freq="0 0 * * * UTC",
        refresh_mode="FULL",
        append_only=True,
        backup_source=backup_source,
    )
    fv._version = FeatureViewVersion("V1")
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("TEST_DB")
    fv._schema = SqlIdentifier("TEST_SCHEMA")
    return fv


_FV_PHYSICAL_NAME = SqlIdentifier("GUEST_SNAPSHOT_FEATURES$V1", case_sensitive=True)


def _describe_to_schema(describe_rows: list[dict[str, str]]) -> StructType:
    """Convert DESCRIBE TABLE-style dicts to a Snowpark StructType for mocking session.table().schema."""
    fields: list[StructField] = []
    for row in describe_rows:
        type_str = row["type"]
        snowpark_type: DataType
        if type_str.startswith("VARCHAR"):
            snowpark_type = StringType(int(type_str[len("VARCHAR(") : -1]))
        elif type_str.startswith("TIMESTAMP"):
            snowpark_type = TimestampType()
        elif type_str.startswith("NUMBER"):
            parts = type_str[len("NUMBER(") : -1].split(",")
            snowpark_type = DecimalType(int(parts[0].strip()), int(parts[1].strip()))
        else:
            snowpark_type = StringType()
        fields.append(StructField(row["name"], snowpark_type, nullable=True))
    return StructType(fields)


def _setup_table_schemas(fs: Any, table_schemas: dict[str, list[dict[str, str]]]) -> None:
    """Mock session.table() to return mocks with proper StructType schemas.

    Keys in table_schemas are matched as substrings against the table name
    argument to session.table().
    """

    def table_side_effect(table_name: str) -> MagicMock:
        mock_table = MagicMock()
        for pattern, describe_rows in table_schemas.items():
            if pattern in str(table_name):
                mock_table.schema = _describe_to_schema(describe_rows)
                return mock_table
        mock_table.schema = StructType([])
        return mock_table

    fs._session.table.side_effect = table_side_effect


class CreateSnapshotTableWithBackfillTest(absltest.TestCase):
    """Tests for _create_snapshot_table with backup_source."""

    def _get_sql_calls(self, fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_without_backfill_uses_create_like(self) -> None:
        """No backup_source -> CREATE TABLE ... LIKE ..."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source=None)

        fs._session.sql.return_value.collect.return_value = []
        fs._create_snapshot_table(fv, _FV_PHYSICAL_NAME, "TEST_DB.TEST_SCHEMA.DT_FQN")

        sql_calls = self._get_sql_calls(fs)
        like_calls = [s for s in sql_calls if "CREATE" in s and "LIKE" in s]
        clone_calls = [s for s in sql_calls if "CLONE" in s]
        self.assertLen(like_calls, 1)
        self.assertEmpty(clone_calls)
        self.assertIn("$SNAPSHOTS", like_calls[0])
        self.assertIn("DT_FQN", like_calls[0])

    def test_fresh_creation_applies_clustering(self) -> None:
        """Clustering from feature_view.cluster_by is applied on fresh snapshot creation."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source=None)

        fs._session.sql.return_value.collect.return_value = []
        fs._create_snapshot_table(fv, _FV_PHYSICAL_NAME, "TEST_DB.TEST_SCHEMA.DT_FQN")

        sql_calls = self._get_sql_calls(fs)
        cluster_calls = [s for s in sql_calls if "CLUSTER BY" in s]
        self.assertLen(cluster_calls, 1)
        self.assertIn("GUEST_ID", cluster_calls[0])
        self.assertIn("SNAPSHOT_TS", cluster_calls[0])

    def test_with_backfill_uses_clone(self) -> None:
        """backup_source -> CREATE TABLE ... CLONE.

        Schemas are kept identical here so reconciliation is a no-op: this test verifies
        CLONE is invoked, not the reconciliation behaviour. ADD-COLUMN reconciliation is
        covered separately by test_backfill_adds_missing_columns. A backfill column not
        present in the DT, or one whose position would shift relative to the DT, is
        rejected by the extend-only helper, so the schemas chosen here intentionally
        avoid both situations.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "N_RSRVS_30_DAY", "type": "NUMBER(10,0)"},
            {"name": "N_FUTURE_STAYS", "type": "NUMBER(10,0)"},
        ]
        dt_describe = list(backfill_describe)
        snapshot_describe = list(backfill_describe)

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS" in query:
                result.collect.return_value = backfill_describe
            elif "DESCRIBE TABLE TEST_DB.TEST_SCHEMA.DT_FQN" in query:
                result.collect.return_value = dt_describe
            elif "DESCRIBE TABLE" in query and "$SNAPSHOTS" in query:
                result.collect.return_value = snapshot_describe
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "HISTORICAL_SNAPSHOTS": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        fs._create_snapshot_table(
            fv,
            _FV_PHYSICAL_NAME,
            "TEST_DB.TEST_SCHEMA.DT_FQN",
            backfill_table="MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS",
        )

        sql_calls = self._get_sql_calls(fs)
        clone_calls = [s for s in sql_calls if "CLONE" in s]
        like_calls = [s for s in sql_calls if "CREATE" in s and "LIKE" in s]
        self.assertLen(clone_calls, 1)
        self.assertEmpty(like_calls)
        self.assertIn("MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS", clone_calls[0])
        self.assertIn("$SNAPSHOTS", clone_calls[0])

    def test_backfill_adds_missing_columns(self) -> None:
        """Schema reconciliation: DT columns missing from backfill are added."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "N_RSRVS_30_DAY", "type": "NUMBER(10,0)"},
            {"name": "N_FUTURE_STAYS", "type": "NUMBER(10,0)"},
        ]
        snapshot_describe = list(backfill_describe)

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS" in query:
                result.collect.return_value = backfill_describe
            elif "DESCRIBE TABLE TEST_DB.TEST_SCHEMA.DT_FQN" in query:
                result.collect.return_value = dt_describe
            elif "DESCRIBE TABLE" in query and "$SNAPSHOTS" in query:
                result.collect.return_value = snapshot_describe
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "HISTORICAL_SNAPSHOTS": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        fs._create_snapshot_table(
            fv,
            _FV_PHYSICAL_NAME,
            "TEST_DB.TEST_SCHEMA.DT_FQN",
            backfill_table="MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS",
        )

        sql_calls = self._get_sql_calls(fs)
        add_col_calls = [s for s in sql_calls if "ADD COLUMN" in s]
        self.assertLen(add_col_calls, 2)
        reconcile_text = " ".join(add_col_calls)
        self.assertIn("N_RSRVS_30_DAY", reconcile_text)
        self.assertIn("N_FUTURE_STAYS", reconcile_text)

    def test_backfill_applies_clustering(self) -> None:
        """Clustering from feature_view.cluster_by is applied to the snapshot table."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "N_RSRVS_30_DAY", "type": "NUMBER(10,0)"},
        ]
        dt_describe = list(backfill_describe)
        snapshot_describe = list(backfill_describe)

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS" in query:
                result.collect.return_value = backfill_describe
            elif "DESCRIBE TABLE TEST_DB.TEST_SCHEMA.DT_FQN" in query:
                result.collect.return_value = dt_describe
            elif "DESCRIBE TABLE" in query and "$SNAPSHOTS" in query:
                result.collect.return_value = snapshot_describe
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "HISTORICAL_SNAPSHOTS": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        fs._create_snapshot_table(
            fv,
            _FV_PHYSICAL_NAME,
            "TEST_DB.TEST_SCHEMA.DT_FQN",
            backfill_table="MY_DB.MY_SCHEMA.HISTORICAL_SNAPSHOTS",
        )

        sql_calls = self._get_sql_calls(fs)
        cluster_calls = [s for s in sql_calls if "CLUSTER BY" in s]
        self.assertLen(cluster_calls, 1)
        self.assertIn("GUEST_ID", cluster_calls[0])
        self.assertIn("SNAPSHOT_TS", cluster_calls[0])

    def test_backfill_rejects_missing_timestamp_col(self) -> None:
        """Validation rejects backfill source missing the timestamp column.

        The cloned snapshot mirrors the backfill schema, so the FV's
        ``timestamp_col`` shows up as a missing required column on the snapshot
        before the structural prefix comparison runs.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.BAD_TABLE")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "N_RSRVS_30_DAY", "type": "NUMBER(10,0)"},
            {"name": "N_FUTURE_STAYS", "type": "NUMBER(10,0)"},
        ]
        snapshot_describe = list(backfill_describe)
        fs._session.sql.return_value.collect.return_value = []
        _setup_table_schemas(
            fs,
            {
                "BAD_TABLE": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        with self.assertRaises(Exception) as cm:
            fs._create_snapshot_table(
                fv,
                _FV_PHYSICAL_NAME,
                "TEST_DB.TEST_SCHEMA.DT_FQN",
                backfill_table="MY_DB.MY_SCHEMA.BAD_TABLE",
            )
        self.assertIn("missing required columns", str(cm.exception))
        self.assertIn("SNAPSHOT_TS", str(cm.exception))

    def test_backfill_rejects_missing_entity_keys(self) -> None:
        """Validation rejects backfill source missing entity join keys."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.BAD_TABLE")

        backfill_describe = [
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "N_RSRVS_30_DAY", "type": "NUMBER(10,0)"},
            {"name": "N_FUTURE_STAYS", "type": "NUMBER(10,0)"},
        ]
        snapshot_describe = list(backfill_describe)
        fs._session.sql.return_value.collect.return_value = []
        _setup_table_schemas(
            fs,
            {
                "BAD_TABLE": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        with self.assertRaises(Exception) as cm:
            fs._create_snapshot_table(
                fv,
                _FV_PHYSICAL_NAME,
                "TEST_DB.TEST_SCHEMA.DT_FQN",
                backfill_table="MY_DB.MY_SCHEMA.BAD_TABLE",
            )
        self.assertIn("missing required columns", str(cm.exception))
        self.assertIn("GUEST_ID", str(cm.exception))

    def test_backfill_validates_case_sensitive_keys(self) -> None:
        """Validation is case-insensitive for column name matching."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.CASE_TABLE")

        backfill_describe = [
            {"name": "guest_id", "type": "NUMBER(38,0)"},
            {"name": "snapshot_ts", "type": "TIMESTAMP_NTZ(9)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        snapshot_describe = list(backfill_describe)

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE MY_DB.MY_SCHEMA.CASE_TABLE" in query:
                result.collect.return_value = backfill_describe
            elif "DESCRIBE TABLE TEST_DB.TEST_SCHEMA.DT_FQN" in query:
                result.collect.return_value = dt_describe
            elif "DESCRIBE TABLE" in query and "$SNAPSHOTS" in query:
                result.collect.return_value = snapshot_describe
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "CASE_TABLE": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        fs._create_snapshot_table(
            fv,
            _FV_PHYSICAL_NAME,
            "TEST_DB.TEST_SCHEMA.DT_FQN",
            backfill_table="MY_DB.MY_SCHEMA.CASE_TABLE",
        )

    def test_backfill_rejects_incompatible_types(self) -> None:
        """Extend-only reconciliation rejects incompatible column types (backfill vs DT).

        After ``CREATE TABLE … CLONE``, ``_create_snapshot_table`` compares Snowpark
        schemas from ``session.table`` (mocked here via ``_setup_table_schemas``) and
        calls ``get_table_schema_evolution_extend_only_commands``, which rejects a type
        change on an existing column (e.g. VARCHAR vs NUMBER on ``GUEST_ID``).
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.BAD_TYPES")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "VARCHAR(100)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]

        sql_result = MagicMock()
        sql_result.collect.return_value = []
        fs._session.sql.return_value = sql_result
        _setup_table_schemas(
            fs,
            {
                "BAD_TYPES": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": backfill_describe,
            },
        )

        with self.assertRaises(Exception) as cm:
            fs._create_snapshot_table(
                fv,
                _FV_PHYSICAL_NAME,
                "TEST_DB.TEST_SCHEMA.DT_FQN",
                backfill_table="MY_DB.MY_SCHEMA.BAD_TYPES",
            )
        self.assertIn("type changed", str(cm.exception).lower())

    def test_backfill_rejects_type_widening(self) -> None:
        """Extend-only evolution rejects type widening (e.g. NUMBER(10,0) -> NUMBER(38,0))."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.GOOD_TYPES")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(10,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        snapshot_describe = list(backfill_describe)

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE MY_DB.MY_SCHEMA.GOOD_TYPES" in query:
                result.collect.return_value = backfill_describe
            elif "DESCRIBE TABLE TEST_DB.TEST_SCHEMA.DT_FQN" in query:
                result.collect.return_value = dt_describe
            elif "DESCRIBE TABLE" in query and "$SNAPSHOTS" in query:
                result.collect.return_value = snapshot_describe
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "GOOD_TYPES": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        with self.assertRaises(Exception) as cm:
            fs._create_snapshot_table(
                fv,
                _FV_PHYSICAL_NAME,
                "TEST_DB.TEST_SCHEMA.DT_FQN",
                backfill_table="MY_DB.MY_SCHEMA.GOOD_TYPES",
            )
        self.assertIn("type changed", str(cm.exception).lower())

    def test_backfill_rejects_extra_backfill_columns(self) -> None:
        """Backfill columns absent from the DT are rejected (extend-only cannot drop)."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.EXTRA_COLS")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "LEGACY_COL", "type": "VARCHAR(100)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        snapshot_describe = list(backfill_describe)

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE MY_DB.MY_SCHEMA.EXTRA_COLS" in query:
                result.collect.return_value = backfill_describe
            elif "DESCRIBE TABLE TEST_DB.TEST_SCHEMA.DT_FQN" in query:
                result.collect.return_value = dt_describe
            elif "DESCRIBE TABLE" in query and "$SNAPSHOTS" in query:
                result.collect.return_value = snapshot_describe
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "EXTRA_COLS": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        with self.assertRaises(Exception) as cm:
            fs._create_snapshot_table(
                fv,
                _FV_PHYSICAL_NAME,
                "TEST_DB.TEST_SCHEMA.DT_FQN",
                backfill_table="MY_DB.MY_SCHEMA.EXTRA_COLS",
            )
        self.assertIn("cannot drop columns", str(cm.exception).lower())

    def test_create_drops_snapshot_table_on_post_create_failure(self) -> None:
        """Failures after the initial CREATE TABLE drop the half-built snapshot.

        Exercises the create-time compensating action on the LIKE path: the snapshot
        table is brand new, so any failure between ``CREATE TABLE`` and the final
        ``SET TAG`` is recovered by ``DROP TABLE IF EXISTS`` rather than by
        per-column rollback. The LIKE path is used here because it isolates the
        post-CREATE failure to the ``SET TAG`` step (no schema reconciliation runs);
        the same compensating action covers CLONE-path failures.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source=None)

        snapshot_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]

        executed_sql: list[str] = []

        def sql_side_effect(query: str) -> MagicMock:
            executed_sql.append(query)
            result = MagicMock()
            if "SET TAG" in query:
                # Simulate a failure of the final tagging step. The snapshot table
                # is fully built by this point and would be left orphaned if the
                # caller did not clean up.
                result.collect.side_effect = RuntimeError("simulated SET TAG failure")
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(fs, {"$SNAPSHOTS": snapshot_describe})

        with self.assertRaises(RuntimeError):
            fs._create_snapshot_table(fv, _FV_PHYSICAL_NAME, "TEST_DB.TEST_SCHEMA.DT_FQN")

        drop_calls = [s for s in executed_sql if s.startswith("DROP TABLE IF EXISTS")]
        self.assertLen(drop_calls, 1)
        self.assertIn("$SNAPSHOTS", drop_calls[0])

    def test_create_drops_snapshot_table_on_clone_path_failure(self) -> None:
        """CLONE-path schema-reconciliation failures also drop the snapshot table.

        On the CLONE path, the helper's per-column rollback is suppressed
        (``rollback_cmds=[]``) because the outer compensating action drops the whole
        table. This test forces a backfill schema that the extend-only helper
        rejects (extra column not in the DT) and asserts the resulting cleanup is
        a single ``DROP TABLE`` rather than any ``ALTER TABLE … DROP COLUMN``.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv(backup_source="MY_DB.MY_SCHEMA.EXTRA_COLS")

        backfill_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
            {"name": "LEGACY_COL", "type": "VARCHAR(100)"},
        ]
        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        snapshot_describe = list(backfill_describe)

        executed_sql: list[str] = []

        def sql_side_effect(query: str) -> MagicMock:
            executed_sql.append(query)
            result = MagicMock()
            result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect
        _setup_table_schemas(
            fs,
            {
                "EXTRA_COLS": backfill_describe,
                "DT_FQN": dt_describe,
                "$SNAPSHOTS": snapshot_describe,
            },
        )

        with self.assertRaisesRegex(Exception, "cannot drop columns"):
            fs._create_snapshot_table(
                fv,
                _FV_PHYSICAL_NAME,
                "TEST_DB.TEST_SCHEMA.DT_FQN",
                backfill_table="MY_DB.MY_SCHEMA.EXTRA_COLS",
            )

        drop_table_calls = [s for s in executed_sql if s.startswith("DROP TABLE IF EXISTS")]
        drop_column_calls = [s for s in executed_sql if "DROP COLUMN" in s]
        self.assertLen(drop_table_calls, 1)
        self.assertIn("$SNAPSHOTS", drop_table_calls[0])
        self.assertEmpty(drop_column_calls)


class BuildSnapshotTaskBodyTest(absltest.TestCase):
    """Tests for _build_snapshot_task_body (SQL scripting block with status table)."""

    def test_snapshot_task_body(self) -> None:
        fs = _create_feature_store_with_mocks()
        dt_fqn = "TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1"

        body = fs._build_snapshot_task_body(_FV_PHYSICAL_NAME, dt_fqn)

        # fmt: off
        expected = (
            """\
DECLARE
    snapshot_start_time TIMESTAMP_NTZ;
    refresh_state VARCHAR;
BEGIN
    -- Check status table for a pending snapshot (crash recovery)
    SELECT SNAPSHOT_START_TIME INTO :snapshot_start_time
        FROM TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS
        WHERE FV_FQN = 'TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1'
        LIMIT 1;
    LET skip_dynamic_table_refresh BOOLEAN := false;
    IF (snapshot_start_time IS NOT NULL) THEN
        -- Branch 1: Status row exists \u2014 previous run did not complete.
        -- Check if a DT refresh already happened after snapshot_start_time.
        SELECT STATE INTO :refresh_state FROM (SELECT STATE
        FROM TABLE(TEST_DB.INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY(
            RESULT_LIMIT => 100
        ))
        WHERE NAME = 'GUEST_SNAPSHOT_FEATURES$V1'
        AND SCHEMA_NAME = 'TEST_SCHEMA'
        AND STATE IN ('SUCCEEDED')
        AND REFRESH_END_TIME >= :snapshot_start_time
        ORDER BY REFRESH_END_TIME DESC
        LIMIT 1);

        IF (refresh_state IS NOT NULL) THEN
            skip_dynamic_table_refresh := true;
        END IF;
    END IF;

    IF (:skip_dynamic_table_refresh = false) THEN
        -- Branch 2: Fresh run \u2014 record start time and refresh DT.
        MERGE INTO TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS AS tgt
"""
            "            USING (SELECT 'TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1' AS FV_FQN,"
            " CURRENT_TIMESTAMP() AS SNAPSHOT_START_TIME) AS src\n"
            """\
            ON tgt.FV_FQN = src.FV_FQN
            WHEN MATCHED THEN UPDATE SET tgt.SNAPSHOT_START_TIME = src.SNAPSHOT_START_TIME
            WHEN NOT MATCHED THEN INSERT (FV_FQN, SNAPSHOT_START_TIME) VALUES (src.FV_FQN, src.SNAPSHOT_START_TIME);

        ALTER DYNAMIC TABLE TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1 REFRESH;
    END IF;

    -- Common: Atomic snapshot INSERT + status cleanup
    BEGIN TRANSACTION;
        INSERT INTO TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1$SNAPSHOTS
            SELECT * FROM TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1;
"""
            "        DELETE FROM TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS"
            " WHERE FV_FQN = 'TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1';\n"
            """\
    COMMIT;
END;"""
        )
        # fmt: on
        self.assertEqual(body, expected)

    def test_snapshot_task_body_escapes_embedded_single_quotes(self) -> None:
        """Embedded ``'`` in a resolved DT/FV/schema name must be escaped before being
        baked into the task body's SQL string literals — otherwise the rendered task body
        would carry a stray closing quote and Snowflake would reject the ``CREATE TASK``.

        Snowflake's quoted-identifier syntax legally allows an embedded ``'`` (e.g.
        ``"my_fv'foo$V1"``), and ``SqlIdentifier.resolved()`` preserves it.  This case
        exercises the fully-qualified name (``WHERE FV_FQN = ...`` / ``MERGE USING ...`` /
        ``DELETE ... WHERE FV_FQN = ...``) plus the unqualified DT name and schema name
        used in the ``DYNAMIC_TABLE_REFRESH_HISTORY`` lookup.
        """
        from snowflake.ml.feature_store.feature_store import _FeatureStoreConfig

        fs = _create_feature_store_with_mocks()
        fs._config = _FeatureStoreConfig(
            database=SqlIdentifier("TEST_DB"),
            schema=SqlIdentifier('"sch\'m"', case_sensitive=False),
        )
        fv_physical = SqlIdentifier('"my_fv\'foo$V1"', case_sensitive=False)
        dt_fqn = 'TEST_DB."sch\'m"."my_fv\'foo$V1"'

        body = fs._build_snapshot_task_body(fv_physical, dt_fqn)

        # fmt: off
        expected = (
            """\
DECLARE
    snapshot_start_time TIMESTAMP_NTZ;
    refresh_state VARCHAR;
BEGIN
    -- Check status table for a pending snapshot (crash recovery)
    SELECT SNAPSHOT_START_TIME INTO :snapshot_start_time
        FROM TEST_DB."sch'm".SNOWML_SNAPSHOT_STATUS
        WHERE FV_FQN = 'TEST_DB."sch\\'m"."my_fv\\'foo$V1"'
        LIMIT 1;
    LET skip_dynamic_table_refresh BOOLEAN := false;
    IF (snapshot_start_time IS NOT NULL) THEN
        -- Branch 1: Status row exists \u2014 previous run did not complete.
        -- Check if a DT refresh already happened after snapshot_start_time.
        SELECT STATE INTO :refresh_state FROM (SELECT STATE
        FROM TABLE(TEST_DB.INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY(
            RESULT_LIMIT => 100
        ))
        WHERE NAME = 'my_fv\\'foo$V1'
        AND SCHEMA_NAME = 'sch\\'m'
        AND STATE IN ('SUCCEEDED')
        AND REFRESH_END_TIME >= :snapshot_start_time
        ORDER BY REFRESH_END_TIME DESC
        LIMIT 1);

        IF (refresh_state IS NOT NULL) THEN
            skip_dynamic_table_refresh := true;
        END IF;
    END IF;

    IF (:skip_dynamic_table_refresh = false) THEN
        -- Branch 2: Fresh run \u2014 record start time and refresh DT.
        MERGE INTO TEST_DB."sch'm".SNOWML_SNAPSHOT_STATUS AS tgt
"""
            """            USING (SELECT 'TEST_DB."sch\\'m"."my_fv\\'foo$V1"' AS FV_FQN,"""
            """ CURRENT_TIMESTAMP() AS SNAPSHOT_START_TIME) AS src\n"""
            """\
            ON tgt.FV_FQN = src.FV_FQN
            WHEN MATCHED THEN UPDATE SET tgt.SNAPSHOT_START_TIME = src.SNAPSHOT_START_TIME
            WHEN NOT MATCHED THEN INSERT (FV_FQN, SNAPSHOT_START_TIME) VALUES (src.FV_FQN, src.SNAPSHOT_START_TIME);

        ALTER DYNAMIC TABLE TEST_DB."sch'm"."my_fv'foo$V1" REFRESH;
    END IF;

    -- Common: Atomic snapshot INSERT + status cleanup
    BEGIN TRANSACTION;
        INSERT INTO TEST_DB."sch'm"."my_fv'foo$V1$SNAPSHOTS"
            SELECT * FROM TEST_DB."sch'm"."my_fv'foo$V1";
        DELETE FROM TEST_DB."sch'm".SNOWML_SNAPSHOT_STATUS WHERE FV_FQN = 'TEST_DB."sch\\'m"."my_fv\\'foo$V1"';
    COMMIT;
END;"""
        )
        # fmt: on
        self.assertEqual(body, expected)


def _make_non_append_only_fv() -> FeatureView:
    """Build a managed DT FeatureView that is NOT append_only."""
    mock_df = MagicMock()
    mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "SCORE"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}

    ts_field = MagicMock()
    ts_field.datatype = TimestampType()
    mock_df.schema.__getitem__ = lambda self, key: ts_field

    entity = Entity(name="guest", join_keys=["GUEST_ID"])
    fv = FeatureView(
        name="GUEST_SNAPSHOT_FEATURES",
        entities=[entity],
        feature_df=mock_df,
        timestamp_col="SNAPSHOT_TS",
        refresh_freq="1 day",
    )
    fv._version = FeatureViewVersion("V1")
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("TEST_DB")
    fv._schema = SqlIdentifier("TEST_SCHEMA")
    return fv


def _make_view_fv() -> FeatureView:
    """Build a non-managed (View) FeatureView — refresh_freq=None, not append_only."""
    mock_df = MagicMock()
    mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "SCORE"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}

    ts_field = MagicMock()
    ts_field.datatype = TimestampType()
    mock_df.schema.__getitem__ = lambda self, key: ts_field

    entity = Entity(name="guest", join_keys=["GUEST_ID"])
    fv = FeatureView(
        name="GUEST_SNAPSHOT_FEATURES",
        entities=[entity],
        feature_df=mock_df,
        timestamp_col="SNAPSHOT_TS",
    )
    fv._version = FeatureViewVersion("V1")
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("TEST_DB")
    fv._schema = SqlIdentifier("TEST_SCHEMA")
    return fv


class CreateOfflineFeatureViewIntegrationTest(absltest.TestCase):
    """Tests for _create_offline_feature_view snapshot integration."""

    def _get_sql_calls(self, fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_append_only_creates_snapshot_then_task(self) -> None:
        """For append_only FVs, _create_offline_feature_view creates the DT,
        then the snapshot table, then the task (in that order)."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        call_order: list[str] = []

        def _snapshot_side_effect(*a: Any, **kw: Any) -> str:
            call_order.append("snapshot")
            return "TEST_DB.TEST_SCHEMA.FV$V1$SNAPSHOTS"

        fs._create_dynamic_table = MagicMock(side_effect=lambda *a, **kw: call_order.append("dt"))
        fs._create_snapshot_table = MagicMock(side_effect=_snapshot_side_effect)
        fs._create_scheduled_refresh_task = MagicMock(side_effect=lambda *a, **kw: call_order.append("task"))

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=False,
            created_resources=[],
        )

        self.assertEqual(call_order, ["dt", "snapshot", "task"])
        fs._create_dynamic_table.assert_called_once()
        fs._create_snapshot_table.assert_called_once()
        fs._create_scheduled_refresh_task.assert_called_once()
        _, task_kwargs = fs._create_scheduled_refresh_task.call_args
        self.assertEqual(task_kwargs["feature_view_name"], _FV_PHYSICAL_NAME)

    def test_non_append_only_does_not_create_snapshot(self) -> None:
        """Non-append_only FVs should not create a snapshot table."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._create_dynamic_table = MagicMock()
        fs._create_snapshot_table = MagicMock()

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=False,
            created_resources=[],
        )

        fs._create_snapshot_table.assert_not_called()

    def test_duration_based_non_append_only_does_not_create_task(self) -> None:
        """Duration-based non-append_only FVs use DT internal scheduler, no Task."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._create_dynamic_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=False,
            created_resources=[],
        )

        fs._create_scheduled_refresh_task.assert_not_called()

    def test_view_overwrite_drops_orphaned_snapshot_table(self) -> None:
        """Overwriting a DT with a View drops any orphaned snapshot table.

        When overwrite=True the new FV is guaranteed non-append_only (the
        append_only + overwrite combination is rejected upstream), so the snapshot
        table from any prior append_only registration is stale and must be cleaned up.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_view_fv()
        fs._session.sql.return_value.collect.return_value = []

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=True,
            created_resources=[],
        )

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        self.assertLen(snapshot_drops, 1, "Snapshot table should be dropped on overwrite to View")
        task_drops = [s for s in sql_calls if "DROP TASK IF EXISTS" in s]
        self.assertLen(task_drops, 1)

    def test_dt_overwrite_drops_orphaned_snapshot_table(self) -> None:
        """Overwriting an append_only DT with a non-append_only DT drops the orphaned snapshot table.

        This is the append_only -> non-append_only transition: any snapshot
        history accumulated by the prior registration becomes orphaned and is
        cleaned up so it doesn't leak storage or get re-attached on a later
        re-registration with the same name.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._create_dynamic_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        fs._session.sql.return_value.collect.return_value = []

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=True,
            created_resources=[],
        )

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        self.assertLen(snapshot_drops, 1, "Snapshot table should be dropped on DT overwrite")

    def test_overwrite_cleanup_deletes_snapshot_status_row(self) -> None:
        """Overwrite cleanup removes any stale snapshot-status row for the old FV."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._create_dynamic_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        fs._session.sql.return_value.collect.return_value = []

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=True,
            created_resources=[],
        )

        sql_calls = self._get_sql_calls(fs)
        status_deletes = [s for s in sql_calls if "DELETE FROM TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS" in s]
        self.assertLen(status_deletes, 1, "Snapshot status row should be deleted on overwrite cleanup")
        self.assertIn("FV_FQN = 'TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1'", status_deletes[0])

    def test_dt_no_overwrite_does_not_drop_snapshot_table(self) -> None:
        """A first-time (non-overwrite) registration must not emit a snapshot DROP."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._create_dynamic_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        fs._session.sql.return_value.collect.return_value = []

        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=False,
            created_resources=[],
        )

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        self.assertEmpty(snapshot_drops, "No snapshot drop should occur on a first-time registration")

    def test_append_only_returns_snapshot_in_created_resources(self) -> None:
        """Created resources list includes SNAPSHOT_TABLE for append_only FVs."""
        from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fs._create_dynamic_table = MagicMock()
        fs._create_snapshot_table = MagicMock(return_value="TEST_DB.TEST_SCHEMA.FV$V1$SNAPSHOTS")
        fs._create_scheduled_refresh_task = MagicMock()

        created: list[Any] = []
        fs._create_offline_feature_view(
            feature_view=fv,
            feature_view_name=_FV_PHYSICAL_NAME,
            fully_qualified_name="TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            column_descs="",
            tagging_clause_str="",
            block=True,
            overwrite=False,
            created_resources=created,
        )

        resource_types = [t for t, _ in created]
        self.assertIn(_FeatureStoreObjTypes.SNAPSHOT_TABLE, resource_types)
        self.assertIn(_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, resource_types)
        self.assertIn(_FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK, resource_types)


class ScheduledRefreshTaskTest(absltest.TestCase):
    """Tests for _create_scheduled_refresh_task snapshot-specific behavior."""

    def _get_sql_calls(self, fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_append_only_uses_snapshot_task_body(self) -> None:
        """For append_only FVs, the task body should contain snapshot logic, not just ALTER REFRESH."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()

        dt_describe = [
            {"name": "GUEST_ID", "type": "NUMBER(38,0)"},
            {"name": "SNAPSHOT_TS", "type": "TIMESTAMP_NTZ(9)"},
        ]
        fs._session.sql.return_value.collect.return_value = dt_describe

        fs._create_scheduled_refresh_task(
            "",
            fv,
            "TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            SqlIdentifier("WH_1"),
            feature_view_name=_FV_PHYSICAL_NAME,
        )

        sql_calls = self._get_sql_calls(fs)
        create_task_calls = [s for s in sql_calls if "CREATE" in s and "TASK" in s]
        self.assertLen(create_task_calls, 1)
        self.assertIn("DECLARE", create_task_calls[0])
        self.assertIn("$SNAPSHOTS", create_task_calls[0])
        self.assertNotIn("AS ALTER DYNAMIC TABLE", create_task_calls[0])

    def test_append_only_requires_versioned_physical_name(self) -> None:
        """append_only task scheduling must receive the versioned name (not an assert stripped by -O)."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fs._session.sql.return_value.collect.return_value = []

        # Use OR REPLACE path so a precondition failure is not mistaken for post-CREATE failure
        # (fresh-create except block would otherwise issue DROP DT / DROP TASK).
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            fs._create_scheduled_refresh_task(
                " OR REPLACE",
                fv,
                "TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
                SqlIdentifier("WH_1"),
            )

        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_SNOWML_ERROR)
        self.assertIsInstance(ctx.exception.original_exception, RuntimeError)
        self.assertEqual(
            str(ctx.exception.original_exception),
            f"({error_codes.INTERNAL_SNOWML_ERROR}) Feature Store: append-only feature views require"
            " the versioned physical name to build the refresh task.",
        )
        sql_calls = self._get_sql_calls(fs)
        create_task_calls = [s for s in sql_calls if "CREATE" in s and "TASK" in s]
        self.assertEmpty(create_task_calls)

    def test_non_append_only_uses_simple_refresh(self) -> None:
        """Non-append_only FVs should use simple ALTER DYNAMIC TABLE ... REFRESH."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._session.sql.return_value.collect.return_value = []

        fs._create_scheduled_refresh_task(
            "",
            fv,
            "TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
            SqlIdentifier("WH_1"),
        )

        sql_calls = self._get_sql_calls(fs)
        create_task_calls = [s for s in sql_calls if "CREATE" in s and "TASK" in s]
        self.assertLen(create_task_calls, 1)
        self.assertIn("ALTER DYNAMIC TABLE", create_task_calls[0])
        self.assertNotIn("DECLARE", create_task_calls[0])

    def test_error_cleanup_drops_snapshot_for_append_only(self) -> None:
        """On task creation failure for a fresh append_only FV, the snapshot table is also dropped."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()

        dt_describe = [{"name": "GUEST_ID", "type": "NUMBER(38,0)"}]

        call_count = 0

        def sql_side_effect(query: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if "DESCRIBE TABLE" in query:
                result.collect.return_value = dt_describe
            elif "CREATE" in query and "TASK" in query:
                result.collect.side_effect = Exception("Task creation failed")
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect

        with self.assertRaisesRegex(Exception, "Task creation failed"):
            fs._create_scheduled_refresh_task(
                "",
                fv,
                "TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
                SqlIdentifier("WH_1"),
                feature_view_name=_FV_PHYSICAL_NAME,
            )

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        self.assertLen(snapshot_drops, 1)

    def test_error_cleanup_skips_all_drops_on_overwrite(self) -> None:
        """On task creation failure during the OR REPLACE path, the helper must NOT drop
        the DT, task, or snapshot table itself.

        On the OR REPLACE path the caller owns rollback:
          - register_feature_view(overwrite=True) intentionally skips created_resources
            rollback so the user can retry without losing object identity.
          - update_feature_view(updated_feature_df=...) drives the recreate via
            _recreate_append_only_feature_view_atomically, which registers compensating
            actions per step. Dropping the DT here would defeat those compensating
            actions (an online-table rollback recreating ONLINE FEATURE TABLE FROM <dt>
            would fail because we just dropped the DT).
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()

        dt_describe = [{"name": "GUEST_ID", "type": "NUMBER(38,0)"}]

        def sql_side_effect(query: str) -> MagicMock:
            result = MagicMock()
            if "DESCRIBE TABLE" in query:
                result.collect.return_value = dt_describe
            elif "CREATE" in query and "TASK" in query:
                result.collect.side_effect = Exception("Task creation failed")
            else:
                result.collect.return_value = []
            return result

        fs._session.sql.side_effect = sql_side_effect

        with self.assertRaisesRegex(Exception, "Task creation failed"):
            fs._create_scheduled_refresh_task(
                " OR REPLACE",
                fv,
                "TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1",
                SqlIdentifier("WH_1"),
                feature_view_name=_FV_PHYSICAL_NAME,
            )

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        dt_drops = [s for s in sql_calls if "DROP DYNAMIC TABLE IF EXISTS" in s]
        task_drops = [s for s in sql_calls if "DROP TASK IF EXISTS" in s]
        self.assertEmpty(snapshot_drops, "Snapshot table should be preserved on overwrite error cleanup")
        self.assertEmpty(dt_drops, "DT should be preserved on overwrite — caller owns rollback")
        self.assertEmpty(task_drops, "Task drop should be skipped on overwrite — caller owns rollback")


class DeleteFeatureViewSnapshotTest(absltest.TestCase):
    """Tests for delete_feature_view snapshot table cleanup."""

    def _get_sql_calls(self, fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_delete_always_drops_snapshot_table(self) -> None:
        """delete_feature_view always attempts to drop the snapshot table,
        even for non-append_only FVs."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_append_only_fv()
        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._session.sql.return_value.collect.return_value = []

        fs.delete_feature_view(fv)

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        self.assertLen(snapshot_drops, 1, "Snapshot table should always be dropped on delete")

    def test_delete_append_only_drops_snapshot_table(self) -> None:
        """delete_feature_view drops snapshot table for append_only FVs."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._session.sql.return_value.collect.return_value = []

        fs.delete_feature_view(fv)

        sql_calls = self._get_sql_calls(fs)
        snapshot_drops = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        self.assertLen(snapshot_drops, 1)
        self.assertIn("GUEST_SNAPSHOT_FEATURES", snapshot_drops[0])

    def test_delete_also_deletes_snapshot_status_row(self) -> None:
        """delete_feature_view removes stale snapshot-status state for the deleted FV."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._session.sql.return_value.collect.return_value = []

        fs.delete_feature_view(fv)

        sql_calls = self._get_sql_calls(fs)
        status_deletes = [s for s in sql_calls if "DELETE FROM TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS" in s]
        self.assertLen(status_deletes, 1, "Snapshot status row should be deleted on feature-view delete")
        self.assertIn("FV_FQN = 'TEST_DB.TEST_SCHEMA.GUEST_SNAPSHOT_FEATURES$V1'", status_deletes[0])

    def test_status_delete_swallows_object_does_not_exist(self) -> None:
        """The status DELETE is a no-op when the status table doesn't exist.

        ``SNOWML_SNAPSHOT_STATUS`` is created lazily on append_only registrations.
        If no append_only FV has ever been registered in this schema, the DELETE
        will fail with Snowflake error 2003 ("object does not exist").  The
        cleanup path must swallow that specific error so non-append_only deletes
        don't fail in schemas that never used append_only at all.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        def _sql_side_effect(query: str) -> MagicMock:
            if "DELETE FROM TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS" in query:
                raise SnowparkSQLException(
                    "002003 (02000): SQL compilation error:\n"
                    "Object 'TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS' does not exist or not authorized.",
                    error_code="1304",
                    sql_error_code=2003,
                )
            result = MagicMock()
            result.collect.return_value = []
            return result

        fs._session.sql.side_effect = _sql_side_effect

        # Should not raise.
        fs.delete_feature_view(fv)

    def test_status_delete_reraises_other_sql_errors(self) -> None:
        """SnowparkSQLExceptions other than 2003 must propagate from cleanup.

        The swallow is narrowly scoped to "object does not exist" (2003).  Any
        other SQL error — permissions, syntax, transient backend issues —
        indicates a real problem and must surface to the caller rather than
        silently masking data-cleanup failures.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        def _sql_side_effect(query: str) -> MagicMock:
            if "DELETE FROM TEST_DB.TEST_SCHEMA.SNOWML_SNAPSHOT_STATUS" in query:
                raise SnowparkSQLException(
                    "003001 (42501): Insufficient privileges to operate on table 'SNOWML_SNAPSHOT_STATUS'.",
                    error_code="1304",
                    sql_error_code=3001,
                )
            result = MagicMock()
            result.collect.return_value = []
            return result

        fs._session.sql.side_effect = _sql_side_effect

        with self.assertRaises(SnowparkSQLException) as cm:
            fs.delete_feature_view(fv)
        self.assertEqual(cm.exception.sql_error_code, 3001)


class ApplySnapshotClusteringTest(absltest.TestCase):
    """Tests for _apply_snapshot_clustering edge cases."""

    def _get_sql_calls(self, fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_no_cluster_by_is_noop(self) -> None:
        """FV without cluster_by should not issue any ALTER TABLE."""
        fs = _create_feature_store_with_mocks()
        fv = _make_fv()
        fv._cluster_by = []

        fs._apply_snapshot_clustering(fv, "TEST_DB.TEST_SCHEMA.SNAP$V1$SNAPSHOTS")

        fs._session.sql.assert_not_called()


class RollbackSnapshotTableTest(absltest.TestCase):
    """Tests for _rollback_created_resources with SNAPSHOT_TABLE."""

    def _get_sql_calls(self, fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_rollback_drops_snapshot_table(self) -> None:
        """Rollback should issue DROP TABLE IF EXISTS for SNAPSHOT_TABLE resources."""
        from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

        fs = _create_feature_store_with_mocks()
        fs._session.sql.return_value.collect.return_value = []

        created = [
            (_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, "TEST_DB.TEST_SCHEMA.FV$V1"),
            (_FeatureStoreObjTypes.SNAPSHOT_TABLE, "TEST_DB.TEST_SCHEMA.FV$V1$SNAPSHOTS"),
            (_FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK, "TEST_DB.TEST_SCHEMA.FV$V1"),
        ]

        fs._rollback_created_resources(created)

        sql_calls = self._get_sql_calls(fs)
        drop_dt = [s for s in sql_calls if "DROP DYNAMIC TABLE IF EXISTS" in s]
        drop_snapshot = [s for s in sql_calls if "DROP TABLE IF EXISTS" in s and "$SNAPSHOTS" in s]
        drop_task = [s for s in sql_calls if "DROP TASK IF EXISTS" in s]
        self.assertLen(drop_dt, 1)
        self.assertLen(drop_snapshot, 1)
        self.assertLen(drop_task, 1)

    def test_rollback_reverses_order(self) -> None:
        """Rollback processes resources in reverse order."""
        from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

        fs = _create_feature_store_with_mocks()
        fs._session.sql.return_value.collect.return_value = []

        created = [
            (_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, "TEST_DB.TEST_SCHEMA.FV$V1"),
            (_FeatureStoreObjTypes.SNAPSHOT_TABLE, "TEST_DB.TEST_SCHEMA.FV$V1$SNAPSHOTS"),
        ]

        fs._rollback_created_resources(created)

        sql_calls = self._get_sql_calls(fs)
        snapshot_idx = next(i for i, s in enumerate(sql_calls) if "$SNAPSHOTS" in s)
        dt_idx = next(i for i, s in enumerate(sql_calls) if "DROP DYNAMIC TABLE" in s)
        self.assertLess(snapshot_idx, dt_idx, "Snapshot should be rolled back before DT (reverse order)")


if __name__ == "__main__":
    absltest.main()
