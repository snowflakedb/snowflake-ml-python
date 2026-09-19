"""Unit tests for OFT SHOW row ``store_type`` → ``OnlineConfig`` reconstruction."""

import datetime
import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import (
    feature_store as fs_mod,
    realtime_registration as rtfv_mod,
)
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.metadata_manager import RealtimeConfigMetadata
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod
from snowflake.snowpark import Row
from snowflake.snowpark.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

_RTFV_OUTPUT_SCHEMA = StructType([StructField("risk_score", DoubleType())])


def _rtfv_compute_fn(txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": txn["avg_amount"]})


def _build_rtfv_upstream_fv() -> FeatureView:
    """Build a registered-looking FV to use as an RTFV source."""
    schema = StructType([StructField("USER_ID", StringType()), StructField("avg_amount", DoubleType())])
    mock_df = MagicMock()
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": ["SELECT * FROM TXN_FV"]}
    fv = FeatureView(
        name="TXN_FV",
        entities=[Entity(name="USER", join_keys=["USER_ID"])],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
    )
    fv._version = FeatureViewVersion("v1")
    fv._infer_schema_df = mock_df
    fv._status = FeatureViewStatus.ACTIVE
    return fv


class StoreTypeFromOftShowRowTest(parameterized.TestCase):
    """Tests for ``_store_type_from_oft_show_row``."""

    @parameterized.named_parameters(  # type: ignore[misc]
        ("postgres_upper", "POSTGRES", OnlineStoreType.POSTGRES),
        ("postgres_lower", "postgres", OnlineStoreType.POSTGRES),
        ("hybrid_table", "hybrid_table", OnlineStoreType.HYBRID_TABLE),
        ("hybrid_table_upper", "HYBRID_TABLE", OnlineStoreType.HYBRID_TABLE),
        ("missing_column", None, OnlineStoreType.HYBRID_TABLE),
    )
    def test_store_type_mapping(self, store_type_value: str | None, expected: OnlineStoreType) -> None:
        if store_type_value is None:
            row = Row(TARGET_LAG="10s")
        else:
            row = Row(TARGET_LAG="10s", STORE_TYPE=store_type_value)
        self.assertEqual(fs_mod._store_type_from_oft_show_row(row), expected)

    def test_lowercase_store_type_key(self) -> None:
        row = Row(target_lag="10s", store_type="postgres")
        self.assertEqual(fs_mod._store_type_from_oft_show_row(row), OnlineStoreType.POSTGRES)

    def test_empty_or_none_store_type_defaults_hybrid(self) -> None:
        self.assertEqual(
            fs_mod._store_type_from_oft_show_row(Row(TARGET_LAG="10s", STORE_TYPE=None)),
            OnlineStoreType.HYBRID_TABLE,
        )
        self.assertEqual(
            fs_mod._store_type_from_oft_show_row(Row(TARGET_LAG="10s", STORE_TYPE="")),
            OnlineStoreType.HYBRID_TABLE,
        )

    def test_unknown_store_type_warns_and_defaults_hybrid(self) -> None:
        row = Row(TARGET_LAG="10s", STORE_TYPE="unknown_backend")
        with self.assertLogs(fs_mod.logger.name, level=logging.WARNING) as log_ctx:
            got = fs_mod._store_type_from_oft_show_row(row)
        self.assertEqual(got, OnlineStoreType.HYBRID_TABLE)
        self.assertTrue(any("Unknown SHOW ONLINE FEATURE TABLES store_type" in m for m in log_ctx.output))


class DetermineOnlineConfigFromOftTest(absltest.TestCase):
    """Tests for ``FeatureStore._determine_online_config_from_oft`` with mocked SHOW rows."""

    def _make_fs_with_oft_row(self, oft_row: Row) -> FeatureStore:
        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[oft_row]))
        return fs

    def test_postgres_store_type_in_online_config_json(self) -> None:
        oft_row = Row(
            TARGET_LAG="15 seconds",
            REFRESH_MODE="AUTO",
            SCHEDULING_STATE="STARTED",
            STORE_TYPE="POSTGRES",
        )
        fs = self._make_fs_with_oft_row(oft_row)
        json_str = fs._determine_online_config_from_oft("my_fv", "v1")
        cfg = OnlineConfig.from_json(json_str)
        self.assertTrue(cfg.enable)
        self.assertEqual(cfg.store_type, OnlineStoreType.POSTGRES)

    def test_missing_store_type_defaults_hybrid_in_json(self) -> None:
        oft_row = Row(
            TARGET_LAG="1 minute",
            REFRESH_MODE="AUTO",
            SCHEDULING_STATE="STARTED",
        )
        fs = self._make_fs_with_oft_row(oft_row)
        json_str = fs._determine_online_config_from_oft("my_fv", "v1")
        cfg = OnlineConfig.from_json(json_str)
        self.assertEqual(cfg.store_type, OnlineStoreType.HYBRID_TABLE)

    def test_include_online_service_metadata_preserves_store_type(self) -> None:
        oft_row = Row(
            TARGET_LAG="20 seconds",
            REFRESH_MODE="FULL",
            SCHEDULING_STATE="SUSPENDED",
            STORE_TYPE="postgres",
        )
        fs = self._make_fs_with_oft_row(oft_row)
        json_str = fs._determine_online_config_from_oft("my_fv", "v1", include_online_service_metadata=True)
        data = json.loads(json_str)
        self.assertEqual(data["store_type"], "postgres")
        self.assertEqual(data["refresh_mode"], "FULL")
        self.assertEqual(data["scheduling_state"], "SUSPENDED")


class CaseSensitiveNameOftLookupTest(parameterized.TestCase):
    """Tests for case-sensitive feature view names in OFT lookup."""

    @parameterized.named_parameters(  # type: ignore[misc]
        ("uppercase_name", "MY_FV", "MY_FV$V1$ONLINE"),
        ("mixed_case_name", '"UseR_BEHAVIOR_PROFILE"', "UseR_BEHAVIOR_PROFILE$V1$ONLINE"),
        ("name_with_space", '"UseR_ BEHAVIOR_PROFILE"', "UseR_ BEHAVIOR_PROFILE$V1$ONLINE"),
    )
    def test_oft_lookup_succeeds_for_case_sensitive_names(self, name_input: str, real_stored_oft_name: str) -> None:
        oft_row = Row(
            TARGET_LAG="10 seconds",
            REFRESH_MODE="INCREMENTAL",
            SCHEDULING_STATE="RUNNING",
            STORE_TYPE="postgres",
        )

        # Mimic Snowflake exact-match semantics: only return the row for the real stored name.
        def fake_find_object(*, object_type: str, object_name: object) -> list[Row]:
            self.assertEqual(object_type, "ONLINE FEATURE TABLES")
            return [oft_row] if object_name.resolved() == real_stored_oft_name else []  # type: ignore[attr-defined]

        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_find_object", fake_find_object)

        cfg = OnlineConfig.from_json(fs._determine_online_config_from_oft(name_input, "V1"))

        self.assertTrue(cfg.enable)
        self.assertEqual(cfg.store_type, OnlineStoreType.POSTGRES)


class OftSetupStatusFieldsTest(parameterized.TestCase):
    """Tests for the defensive readers behind the optional setup-readiness columns.

    ``setup_status`` / ``setup_error_msg`` / ``setup_time`` are absent from the SHOW row
    entirely -- rather than present-and-NULL -- when no setup-readiness information is
    available for the table. Absence is an ordinary case and must never raise.
    """

    def test_optional_field_absent_reports_not_present(self) -> None:
        present, value = fs_mod._oft_show_row_optional_field(Row(TARGET_LAG="10s"), "setup_status")
        self.assertFalse(present)
        self.assertIsNone(value)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("uppercase_key", "SETUP_STATUS"),
        ("lowercase_key", "setup_status"),
    )
    def test_optional_field_both_casings(self, key: str) -> None:
        row = Row(**{key: "SETUP_READY"})
        present, value = fs_mod._oft_show_row_optional_field(row, "setup_status")
        self.assertTrue(present)
        self.assertEqual(value, "SETUP_READY")

    def test_optional_field_present_but_null_is_distinguished_from_absent(self) -> None:
        """Present-and-NULL must not be reported the same way as absent: they mean different things."""
        present, value = fs_mod._oft_show_row_optional_field(Row(SETUP_STATUS=None), "setup_status")
        self.assertTrue(present)
        self.assertIsNone(value)

    def test_absent_columns_yield_no_fields(self) -> None:
        row = Row(TARGET_LAG="10s", REFRESH_MODE="AUTO", SCHEDULING_STATE="STARTED")
        self.assertEqual(fs_mod._oft_setup_status_display_fields(row), {})

    def test_present_but_null_columns_yield_explicit_nulls(self) -> None:
        """A non-Postgres-backed OFT reports all three as NULL; that is neither ready nor failed."""
        row = Row(TARGET_LAG="10s", SETUP_STATUS=None, SETUP_ERROR_MSG=None, SETUP_TIME=None)
        self.assertEqual(
            fs_mod._oft_setup_status_display_fields(row),
            {"setup_status": None, "setup_error_msg": None, "setup_time": None},
        )

    def test_setup_time_datetime_is_json_encodable(self) -> None:
        setup_time = datetime.datetime(2026, 8, 10, 12, 34, 56)
        row = Row(SETUP_STATUS="SETUP_READY", SETUP_ERROR_MSG=None, SETUP_TIME=setup_time)
        fields = fs_mod._oft_setup_status_display_fields(row)
        self.assertEqual(fields["setup_time"], setup_time.isoformat())
        # Must survive the json.dumps the listing path performs.
        self.assertEqual(json.loads(json.dumps(fields))["setup_time"], setup_time.isoformat())

    @parameterized.named_parameters(  # type: ignore[misc]
        ("ready", "SETUP_READY", None),
        ("notready", "SETUP_NOTREADY", None),
        ("failed", "SETUP_FAILED", "postgres cluster provisioning timed out"),
    )
    def test_statuses_are_passed_through_verbatim(self, status: str, error_msg: str | None) -> None:
        row = Row(SETUP_STATUS=status, SETUP_ERROR_MSG=error_msg)
        fields = fs_mod._oft_setup_status_display_fields(row)
        self.assertEqual(fields["setup_status"], status)
        self.assertEqual(fields["setup_error_msg"], error_msg)
        self.assertNotIn("setup_time", fields)


class DetermineOnlineConfigSetupStatusTest(parameterized.TestCase):
    """Setup-readiness fields on the batch/streaming listing path."""

    def _display_data(self, oft_row: Row) -> dict[str, Any]:
        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[oft_row]))
        json_str = fs._determine_online_config_from_oft("my_fv", "v1", include_online_service_metadata=True)
        data: dict[str, Any] = json.loads(json_str)
        return data

    def _oft_row(self, **extra: Any) -> Row:
        return Row(
            TARGET_LAG="15 seconds",
            REFRESH_MODE="AUTO",
            SCHEDULING_STATE="STARTED",
            STORE_TYPE="POSTGRES",
            **extra,
        )

    def test_absent_columns_do_not_raise_and_omit_keys(self) -> None:
        """A row without the setup-readiness columns must omit the keys, not raise."""
        data = self._display_data(self._oft_row())
        for key in ("setup_status", "setup_error_msg", "setup_time"):
            self.assertNotIn(key, data)
        # The pre-existing display fields are untouched.
        self.assertEqual(data["refresh_mode"], "AUTO")
        self.assertEqual(data["store_type"], "postgres")

    def test_present_but_null_columns_not_misreported(self) -> None:
        data = self._display_data(
            Row(
                TARGET_LAG="15 seconds",
                REFRESH_MODE="AUTO",
                SCHEDULING_STATE="STARTED",
                STORE_TYPE="hybrid_table",
                SETUP_STATUS=None,
                SETUP_ERROR_MSG=None,
                SETUP_TIME=None,
            )
        )
        # Keys present (so callers can tell "readiness does not apply to this table" from
        # "no readiness information available at all") but null: not ready, not failed.
        self.assertIn("setup_status", data)
        self.assertIsNone(data["setup_status"])
        self.assertIsNone(data["setup_error_msg"])
        self.assertIsNone(data["setup_time"])

    @parameterized.named_parameters(  # type: ignore[misc]
        ("ready", "SETUP_READY", None),
        ("notready", "SETUP_NOTREADY", None),
        ("failed", "SETUP_FAILED", "online store setup failed: quota exceeded"),
    )
    def test_each_status_surfaced(self, status: str, error_msg: str | None) -> None:
        setup_time = datetime.datetime(2026, 8, 10, 1, 2, 3)
        data = self._display_data(self._oft_row(SETUP_STATUS=status, SETUP_ERROR_MSG=error_msg, SETUP_TIME=setup_time))
        self.assertEqual(data["setup_status"], status)
        self.assertEqual(data["setup_error_msg"], error_msg)
        self.assertEqual(data["setup_time"], setup_time.isoformat())

    def test_lowercase_setup_columns_handled(self) -> None:
        data = self._display_data(
            Row(
                target_lag="15 seconds",
                refresh_mode="AUTO",
                scheduling_state="STARTED",
                store_type="postgres",
                setup_status="SETUP_FAILED",
                setup_error_msg="boom",
            )
        )
        self.assertEqual(data["setup_status"], "SETUP_FAILED")
        self.assertEqual(data["setup_error_msg"], "boom")

    def test_non_display_json_stays_online_config_parseable(self) -> None:
        """``_compose_feature_view`` calls ``OnlineConfig.from_json`` on the non-display JSON.

        ``from_json`` is ``cls(**data)`` and so rejects unknown keys; the setup fields must stay out
        of the ``include_online_service_metadata=False`` result.
        """
        fs = object.__new__(FeatureStore)
        object.__setattr__(
            fs,
            "_find_object",
            MagicMock(return_value=[self._oft_row(SETUP_STATUS="SETUP_NOTREADY", SETUP_ERROR_MSG=None)]),
        )
        json_str = fs._determine_online_config_from_oft("my_fv", "v1")
        self.assertNotIn("setup_status", json.loads(json_str))
        cfg = OnlineConfig.from_json(json_str)
        self.assertEqual(cfg.store_type, OnlineStoreType.POSTGRES)


class RealtimeListingSetupStatusTest(parameterized.TestCase):
    """Setup-readiness fields on the RTFV listing path."""

    # Snowpark normalizes StructField names to uppercase, hence the case-insensitive match.
    _ONLINE_CONFIG_INDEX = [f.name.strip('"').upper() for f in fs_mod._LIST_FEATURE_VIEW_BASE_FIELDS].index(
        "ONLINE_CONFIG"
    )

    def _rtfv_metadata(self) -> RealtimeConfigMetadata:
        return RealtimeConfigMetadata(
            name="RTFV",
            version="v1",
            desc="a realtime fv",
            compute_fn_name="_rtfv_compute_fn",
            compute_fn_source="def _rtfv_compute_fn(txn): ...",
            sources=[],
            request_schema_json=None,
            output_schema_json="{}",
            output_columns=["risk_score"],
            entity_names=["USER"],
        )

    def _online_config_for(self, oft_show_row: Row | None) -> dict[str, Any]:
        fs = MagicMock()
        output_values: list[list[Any]] = []
        output_values_extra: list[list[Any]] = []
        rtfv_mod.append_realtime_listing_row(
            feature_store=fs,
            rtfv_metadata=self._rtfv_metadata(),
            oft_show_row=oft_show_row,
            output_values=output_values,
            output_values_extra=output_values_extra,
            fv_kind_realtime="REALTIME",
            default_storage_config_json='{"format": "snowflake"}',
        )
        self.assertLen(output_values, 1)
        data: dict[str, Any] = json.loads(output_values[0][self._ONLINE_CONFIG_INDEX])
        return data

    def test_absent_columns_leave_online_config_unchanged(self) -> None:
        """Without the setup-readiness columns, the RTFV blob must stay byte-identical."""
        data = self._online_config_for(Row(name="RTFV$v1$ONLINE", created_on=None, owner="ROLE"))
        self.assertNotIn("setup_status", data)
        # Still round-trips through the strict OnlineConfig parser.
        self.assertEqual(
            OnlineConfig.from_json(json.dumps(data)).store_type,
            OnlineStoreType.POSTGRES,
        )

    def test_missing_oft_row_does_not_raise(self) -> None:
        data = self._online_config_for(None)
        self.assertNotIn("setup_status", data)
        self.assertTrue(data["enable"])

    @parameterized.named_parameters(  # type: ignore[misc]
        ("ready", "SETUP_READY", None),
        ("notready", "SETUP_NOTREADY", None),
        ("failed", "SETUP_FAILED", "postgres schema creation failed"),
    )
    def test_each_status_surfaced(self, status: str, error_msg: str | None) -> None:
        setup_time = datetime.datetime(2026, 8, 10, 9, 8, 7)
        data = self._online_config_for(
            Row(
                name="RTFV$v1$ONLINE",
                created_on=None,
                owner="ROLE",
                SETUP_STATUS=status,
                SETUP_ERROR_MSG=error_msg,
                SETUP_TIME=setup_time,
            )
        )
        self.assertEqual(data["setup_status"], status)
        self.assertEqual(data["setup_error_msg"], error_msg)
        self.assertEqual(data["setup_time"], setup_time.isoformat())
        # RTFVs are always Postgres-backed, so setup readiness always applies to them.
        self.assertEqual(data["store_type"], OnlineStoreType.POSTGRES.value)

    def test_lowercase_setup_columns_handled(self) -> None:
        data = self._online_config_for(
            Row(
                name="RTFV$v1$ONLINE",
                created_on=None,
                owner="ROLE",
                setup_status="SETUP_NOTREADY",
                setup_error_msg=None,
            )
        )
        self.assertEqual(data["setup_status"], "SETUP_NOTREADY")
        self.assertIsNone(data["setup_error_msg"])


class UpdateFeatureViewPreservesTiledIdentityTest(absltest.TestCase):
    """``_create_updated_feature_view`` must preserve the tiled/streaming aggregation identity.

    ``update_feature_view`` (online enable/disable) rebuilds the FV via
    ``_create_updated_feature_view`` and uses the result to create the Online Feature Table.
    If the rebuild drops the aggregation config, the OFT is built from a non-tiled view of a
    tiled FV.
    """

    def _make_mock_df(self) -> MagicMock:
        df = MagicMock()
        df.queries = {"queries": ["SELECT * FROM TBL"]}
        df.columns = ["USER_ID", "EVENT_TIME", "AMOUNT"]
        ts_field = MagicMock()
        ts_field.datatype = TimestampType()
        df.schema.__getitem__ = lambda _self, key: ts_field
        return df

    def _make_tiled_streaming_base_fv(self) -> FeatureView:
        """Build a reconstructed tiled streaming CONTINUOUS FV, as get_feature_view would."""
        return FeatureView._construct_feature_view(
            name="TILED_FV",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            feature_df=self._make_mock_df(),
            timestamp_col="EVENT_TIME",
            desc="",
            version="v1",
            status=FeatureViewStatus.ACTIVE,
            feature_descs={},
            refresh_freq="1 minute",
            database="DB",
            schema="SCH",
            warehouse="WH",
            refresh_mode="FULL",
            refresh_mode_reason=None,
            initialize="ON_CREATE",
            owner=None,
            infer_schema_df=None,
            session=MagicMock(),
            feature_granularity="1m",
            aggregation_specs=[Feature.sum("AMOUNT", "30s").to_spec()],
            feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
            is_streaming=True,
        )

    def test_create_updated_feature_view_preserves_tiled_streaming_identity(self) -> None:
        base_fv = self._make_tiled_streaming_base_fv()
        # Sanity: the base FV is tiled/streaming CONTINUOUS to begin with.
        self.assertTrue(base_fv.is_tiled)
        self.assertTrue(base_fv.is_streaming)

        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_session", MagicMock())

        updated_fv = fs._create_updated_feature_view(base_fv, OnlineConfig(enable=True))

        self.assertTrue(updated_fv.is_tiled, "updated FV lost tiled identity")
        self.assertTrue(updated_fv.is_streaming, "updated FV lost streaming identity")
        self.assertEqual(updated_fv.feature_granularity, "1m")
        self.assertEqual(updated_fv.feature_aggregation_method, FeatureAggregationMethod.CONTINUOUS)

    def test_create_updated_feature_view_preserves_append_only(self) -> None:
        base_fv = FeatureView._construct_feature_view(
            name="APPEND_FV",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            feature_df=self._make_mock_df(),
            timestamp_col="EVENT_TIME",
            desc="",
            version="v1",
            status=FeatureViewStatus.ACTIVE,
            feature_descs={},
            refresh_freq="0 0 * * * UTC",
            database="DB",
            schema="SCH",
            warehouse="WH",
            refresh_mode="FULL",
            refresh_mode_reason=None,
            initialize="ON_SCHEDULE",
            owner=None,
            infer_schema_df=None,
            session=MagicMock(),
            append_only=True,
            backup_source="DB.SCH.HISTORY",
        )
        self.assertTrue(base_fv.append_only)

        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_session", MagicMock())

        updated_fv = fs._create_updated_feature_view(base_fv, OnlineConfig(enable=True))

        self.assertTrue(updated_fv.append_only, "updated FV lost append_only identity")
        self.assertEqual(updated_fv.backup_source, "DB.SCH.HISTORY")

    def test_create_updated_feature_view_preserves_realtime(self) -> None:
        rtc = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[_build_rtfv_upstream_fv()],
            output_schema=_RTFV_OUTPUT_SCHEMA,
        )
        base_fv = FeatureView._construct_feature_view(
            name="RTFV",
            entities=[Entity(name="user", join_keys=["USER_ID"])],
            feature_df=None,
            timestamp_col=None,
            desc="",
            version="v1",
            status=FeatureViewStatus.ACTIVE,
            feature_descs={},
            refresh_freq=None,
            database="DB",
            schema="SCH",
            warehouse=None,
            refresh_mode=None,
            refresh_mode_reason=None,
            initialize="ON_CREATE",
            owner=None,
            infer_schema_df=None,
            session=MagicMock(),
            is_realtime=True,
            realtime_config=rtc,
        )
        self.assertTrue(base_fv.is_realtime_feature_view)

        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_session", MagicMock())

        updated_fv = fs._create_updated_feature_view(base_fv, OnlineConfig(enable=True))

        self.assertTrue(updated_fv.is_realtime_feature_view, "updated FV lost realtime identity")
        self.assertIsNotNone(updated_fv.realtime_config)

    def test_create_online_feature_table_rejects_hybrid_for_tiled(self) -> None:
        """A HYBRID_TABLE OFT is not supported for a tiled FV and must be rejected clearly."""
        base_fv = self._make_tiled_streaming_base_fv()
        base_fv._online_config = OnlineConfig(enable=True, store_type=OnlineStoreType.HYBRID_TABLE)
        self.assertTrue(base_fv.is_tiled)

        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_session", MagicMock())

        with self.assertRaisesRegex(Exception, "not supported for aggregation"):
            fs._create_online_feature_table(base_fv, SqlIdentifier("TILED_FV"), "v1")

    def test_construct_tiled_fv_with_hybrid_online_rejected(self) -> None:
        """The HYBRID-on-tiled rule is a feature-view invariant: rejected at construction time."""
        with self.assertRaisesRegex(ValueError, "not supported for aggregation"):
            FeatureView(
                name="TILED_FV",
                entities=[Entity(name="user", join_keys=["USER_ID"])],
                feature_df=self._make_mock_df(),
                timestamp_col="EVENT_TIME",
                refresh_freq="1h",
                feature_granularity="1h",
                features=[Feature.sum("AMOUNT", "2h")],
                online_config=OnlineConfig(enable=True),  # default HYBRID_TABLE
            )

    def test_plan_online_enable_forwards_store_type_for_tiled(self) -> None:
        """_plan_online_enable must forward the requested store_type and rebuild a still-tiled FV.

        Regression: it previously dropped store_type (defaulting to HYBRID_TABLE), which combined
        with the tiled guard left no way to enable POSTGRES online on a tiled FV.
        """
        base_fv = self._make_tiled_streaming_base_fv()
        self.assertTrue(base_fv.is_tiled)

        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_session", MagicMock())

        strategy = fs._plan_online_enable(base_fv, OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES))

        assert strategy.final_config is not None
        self.assertEqual(strategy.final_config.store_type, OnlineStoreType.POSTGRES)
        op_type, temp_fv = strategy.operations[0]
        self.assertEqual(op_type, "CREATE_ONLINE")
        assert isinstance(temp_fv, FeatureView)
        self.assertTrue(temp_fv.is_tiled, "rebuilt FV must remain tiled")
        assert temp_fv.online_config is not None
        self.assertEqual(temp_fv.online_config.store_type, OnlineStoreType.POSTGRES)


if __name__ == "__main__":
    absltest.main()
