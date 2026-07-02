"""Unit tests for ``FeatureStore.update_feature_view`` on streaming and realtime feature views."""

from __future__ import annotations

import inspect
from typing import Optional
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import FeatureStore, _FeatureStoreConfig
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
)
from snowflake.ml.feature_store.metadata_manager import RealtimeConfigMetadata
from snowflake.snowpark.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def _make_streaming_fv(
    *,
    refresh_freq: str = "1 minute",
    desc: str = "old desc",
    warehouse: str = "WH_OLD",
    online: bool = False,
    online_target_lag: str = "30s",
    name: str = "STREAM_FV",
    version: str = "v1",
) -> FeatureView:
    """Build a registered streaming FeatureView for update path tests."""
    mock_df = MagicMock()
    mock_df.queries = {"queries": ["SELECT * FROM src"]}
    schema = StructType(
        [
            StructField("USER_ID", StringType()),
            StructField("AMOUNT", DoubleType()),
            StructField("EVENT_TIME", TimestampType()),
        ]
    )
    mock_df.schema = schema
    mock_df.columns = [f.name for f in schema.fields]

    online_config = OnlineConfig(enable=True, target_lag=online_target_lag) if online else None

    entity = Entity(name="user_entity", join_keys=["USER_ID"])
    fv = FeatureView(
        name=name,
        entities=[entity],
        feature_df=mock_df,
        timestamp_col="EVENT_TIME",
        refresh_freq=refresh_freq,
        desc=desc,
        online_config=online_config,
        _is_streaming_marker=True,
    )
    fv._version = FeatureViewVersion(version)
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._warehouse = SqlIdentifier(warehouse)
    return fv


def _new_fs_with_mocks(
    *,
    session: Optional[MagicMock] = None,
    metadata_manager: Optional[MagicMock] = None,
    default_warehouse: str = "WH_DEFAULT",
) -> FeatureStore:
    """Construct a bare-bones ``FeatureStore`` with all I/O mocked."""
    fs = object.__new__(FeatureStore)
    sess = session or MagicMock()
    # ``use_warehouse`` is a no-op so ``session.sql.call_args_list`` captures only update DDL.
    sess.get_current_warehouse.return_value = default_warehouse
    sess.use_warehouse.return_value = None
    md = metadata_manager or MagicMock()
    object.__setattr__(fs, "_session", sess)
    object.__setattr__(
        fs,
        "_config",
        _FeatureStoreConfig(database=SqlIdentifier("DB"), schema=SqlIdentifier("SCH")),
    )
    object.__setattr__(fs, "_telemetry_stmp", {})
    object.__setattr__(fs, "_metadata_manager", md)
    object.__setattr__(fs, "_default_warehouse", SqlIdentifier(default_warehouse))
    object.__setattr__(fs, "_asof_join_enabled", None)
    return fs


def _captured_sql(session: MagicMock) -> list[str]:
    """Return all SQL strings issued via ``session.sql(...).collect(...)``."""
    return [c.args[0] for c in session.sql.call_args_list if c.args]


def _attach_get_feature_view(fs: FeatureStore, fv: FeatureView) -> MagicMock:
    """Make ``get_feature_view`` return ``fv`` so the post-update read works."""
    mock = MagicMock(return_value=fv)
    fs.get_feature_view = mock  # type: ignore[method-assign]
    return mock


class StreamingUpdateFeatureViewSignatureTest(absltest.TestCase):
    """Pin the recreate-only contract: ``update_feature_view`` must not accept
    ``stream_source``, ``transformation_fn``, ``feature_granularity``,
    ``aggregation_secondary_keys``, ``backfill_table``, or ``source_refs``.
    Those fields require delete + re-register.
    """

    _RECREATE_ONLY_KWARGS = (
        "stream_source",
        "transformation_fn",
        "feature_granularity",
        "aggregation_secondary_keys",
        "backfill_table",
        "source_refs",
    )

    def test_stream_source_change_rejected_on_streaming_fv(self) -> None:
        sig = inspect.signature(FeatureStore.update_feature_view)
        self.assertNotIn("stream_source", sig.parameters)

    def test_transformation_fn_change_rejected_on_streaming_fv(self) -> None:
        sig = inspect.signature(FeatureStore.update_feature_view)
        self.assertNotIn("transformation_fn", sig.parameters)

    def test_feature_granularity_change_rejected_on_streaming_fv(self) -> None:
        sig = inspect.signature(FeatureStore.update_feature_view)
        self.assertNotIn("feature_granularity", sig.parameters)

    def test_all_recreate_only_kwargs_absent_from_signature(self) -> None:
        sig = inspect.signature(FeatureStore.update_feature_view)
        for kw in self._RECREATE_ONLY_KWARGS:
            self.assertNotIn(
                kw,
                sig.parameters,
                f"update_feature_view must NOT accept {kw!r}: recreate-only fields require delete + re-register.",
            )

    def test_streaming_compatible_kwargs_present(self) -> None:
        sig = inspect.signature(FeatureStore.update_feature_view)
        for kw in ("refresh_freq", "warehouse", "desc", "online_config"):
            self.assertIn(kw, sig.parameters, f"update_feature_view must accept {kw!r} for streaming FVs.")


class StreamingUpdateFeatureViewDescTest(absltest.TestCase):
    def test_desc_only_change_succeeds_on_streaming_fv(self) -> None:
        session = MagicMock()
        session.sql.return_value = MagicMock()

        fv = _make_streaming_fv(desc="old desc")
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        fs.update_feature_view(fv, desc="updated streaming desc")

        sqls = _captured_sql(session)
        self.assertTrue(sqls, "update_feature_view should issue at least one ALTER DDL")
        joined = " ".join(sqls)
        self.assertIn("ALTER DYNAMIC TABLE", joined)
        self.assertIn("'updated streaming desc'", joined)


class StreamingUpdateFeatureViewWarehouseTest(absltest.TestCase):
    def test_warehouse_only_change_succeeds_on_streaming_fv(self) -> None:
        session = MagicMock()
        session.sql.return_value = MagicMock()

        fv = _make_streaming_fv(warehouse="WH_OLD")
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        fs.update_feature_view(fv, warehouse="WH_NEW")

        sqls = _captured_sql(session)
        self.assertTrue(sqls, "update_feature_view should issue at least one ALTER DDL")
        joined = " ".join(sqls)
        self.assertIn("ALTER DYNAMIC TABLE", joined)
        # Assert the resolved identifier form, not the quoted variant, for stability.
        self.assertIn("WAREHOUSE = WH_NEW", joined)


class StreamingUpdateFeatureViewRefreshFreqTest(absltest.TestCase):
    def test_refresh_freq_only_change_succeeds_on_streaming_fv(self) -> None:
        session = MagicMock()
        session.sql.return_value = MagicMock()

        fv = _make_streaming_fv(refresh_freq="1 minute")
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        fs.update_feature_view(fv, refresh_freq="5 minutes")

        sqls = _captured_sql(session)
        self.assertTrue(sqls, "update_feature_view should issue at least one ALTER DDL")
        joined = " ".join(sqls)
        self.assertIn("ALTER DYNAMIC TABLE", joined)
        self.assertIn("TARGET_LAG = '5 minutes'", joined)


class StreamingUpdateFeatureViewOnlineConfigTest(absltest.TestCase):
    def test_online_config_change_succeeds_on_streaming_fv(self) -> None:
        session = MagicMock()
        session.sql.return_value = MagicMock()

        fv = _make_streaming_fv(online=True, online_target_lag="30s")
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        new_config = OnlineConfig(enable=True, target_lag="10s")
        fs.update_feature_view(fv, online_config=new_config)

        sqls = _captured_sql(session)
        self.assertTrue(sqls, "update_feature_view should issue at least one DDL")
        joined = " ".join(sqls)
        self.assertIn("ALTER ONLINE FEATURE TABLE", joined)
        self.assertIn("'10s'", joined)


class StreamingUpdateFeatureViewDocstringTest(absltest.TestCase):
    """Pin: ``update_feature_view`` docstring must mention streaming support."""

    def test_docstring_mentions_streaming(self) -> None:
        doc = (FeatureStore.update_feature_view.__doc__ or "").lower()
        self.assertIn("streaming", doc)


class StreamingUpdateFeatureViewUpdatedFeatureDfTest(absltest.TestCase):
    """Pin: ``updated_feature_df`` is rejected for streaming feature views."""

    def test_updated_feature_df_rejected_on_streaming_fv(self) -> None:
        session = MagicMock()
        session.sql.return_value = MagicMock()

        fv = _make_streaming_fv()
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        new_df = MagicMock()
        new_df.session = session

        with self.assertRaises(Exception) as ctx:
            fs.update_feature_view(fv, updated_feature_df=new_df)
        self.assertIn("append_only", str(ctx.exception).lower())


def _make_realtime_fv(
    *,
    desc: str = "old rt desc",
    online: bool = True,
    online_target_lag: str = "0 seconds",
    name: str = "RT_FV",
    version: str = "v1",
) -> FeatureView:
    """Build a minimally-realistic RealtimeFeatureView for update path tests."""
    fv = object.__new__(FeatureView)
    fv._name = SqlIdentifier(name)
    fv._version = FeatureViewVersion(version)
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._warehouse = None
    fv._desc = desc
    fv._refresh_freq = None
    fv._refresh_mode = None
    fv._refresh_mode_reason = None
    fv._initialize = "ON_CREATE"
    fv._owner = None
    fv._entities = [Entity(name="user", join_keys=["USER_ID"])]
    fv._feature_df = None
    fv._timestamp_col = None
    fv._cluster_by = []
    fv._stream_config = None
    fv._is_streaming_marker = False
    fv._realtime_config = None
    fv._is_realtime_marker = True
    fv._online_config = OnlineConfig(enable=True, target_lag=online_target_lag) if online else None
    fv._storage_config = None
    fv._feature_desc = None
    fv._feature_granularity = None
    fv._aggregation_specs = None
    fv._aggregation_secondary_keys = None
    fv._feature_aggregation_method = None
    fv._rollup_metadata = None
    fv._infer_schema_df = None
    fv._append_only = False
    fv._backup_source = None
    fv._compute_fn_source = None
    return fv


class RealtimeUpdateFeatureViewTest(absltest.TestCase):
    """``update_feature_view`` must reject ``refresh_freq`` and ``warehouse`` for realtime feature views."""

    def test_refresh_freq_rejected_on_realtime_fv(self) -> None:
        session = MagicMock()
        fv = _make_realtime_fv()
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        with self.assertRaises(Exception) as ctx:
            fs.update_feature_view(fv, refresh_freq="5 minutes")
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "realtime" in msg or "real-time" in msg or "refresh_freq" in msg,
            f"Error message should explain why refresh_freq fails on RTFV; got: {ctx.exception!r}",
        )

    def test_warehouse_rejected_on_realtime_fv(self) -> None:
        session = MagicMock()
        fv = _make_realtime_fv()
        fs = _new_fs_with_mocks(session=session)
        _attach_get_feature_view(fs, fv)

        with self.assertRaises(Exception) as ctx:
            fs.update_feature_view(fv, warehouse="OTHER_WH")
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "realtime" in msg or "real-time" in msg or "warehouse" in msg,
            f"Error message should explain why warehouse fails on RTFV; got: {ctx.exception!r}",
        )


class RealtimeUpdateFeatureViewDescTest(absltest.TestCase):
    def test_desc_only_change_succeeds_on_realtime_fv(self) -> None:
        session = MagicMock()
        session.sql.return_value = MagicMock()
        md = MagicMock()

        fv = _make_realtime_fv(desc="old rt desc")
        fs = _new_fs_with_mocks(session=session, metadata_manager=md)
        _attach_get_feature_view(fs, fv)

        rt_meta = RealtimeConfigMetadata(
            name="RT_FV",
            version="v1",
            desc="old rt desc",
            compute_fn_name="fn",
            compute_fn_source="def fn(): pass",
            sources=[],
            request_schema_json=None,
            output_schema_json='{"fields": []}',
            output_columns=[],
            entity_names=["user"],
        )
        md.get_realtime_config.return_value = rt_meta

        fs.update_feature_view(fv, desc="updated RT desc")

        save_calls = md.save_realtime_config.call_args_list
        self.assertTrue(save_calls, "save_realtime_config should be called to persist RTFV desc update")
        saved_meta = save_calls[0][0][0]
        self.assertEqual(saved_meta.desc, "updated RT desc")


if __name__ == "__main__":
    absltest.main()
