"""End-to-end unit tests for FeatureStore RealtimeFeatureView CRUD.

Covers the lifecycle paths added in the RTFV milestone:

- ``register_feature_view`` rejects DRAFT upstream, offline upstream, non-Postgres
  upstream, and upstreams whose join keys escape the RTFV's declared entity
  superset, all before any side effect.
- Happy-path registration issues ``CREATE ONLINE FEATURE TABLE ... FROM
  SPECIFICATION`` + tag + per-entity tag and writes :class:`RealtimeConfigMetadata`.
- Failure during registration triggers best-effort OFT drop + metadata delete.
- ``delete_feature_view`` issues a ``DROP ONLINE FEATURE TABLE IF EXISTS`` and
  removes only the RTFV metadata (no DT/View drop).
- ``list_feature_views`` surfaces RTFV rows with ``kind=REALTIME`` next to
  batch/streaming rows.

Snowpark interactions are mocked via :class:`MagicMock`. The FeatureStore is
constructed via ``object.__new__`` so we don't hit the real ``__init__``'s
warehouse / metadata-table side effects.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store as fs_mod, online_service
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import _FV_KIND_REALTIME, FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.metadata_manager import (
    FvSourceRef,
    RealtimeConfigMetadata,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType

_RTFV_OUTPUT_SCHEMA = StructType(
    [
        StructField("risk_score", DoubleType()),
        StructField("risk_bucket", StringType()),
    ]
)


def rtfv_compute_fn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_score": req["amount"],
            "risk_bucket": ["x"] * len(req),
        }
    )


def _build_request_source() -> RequestSource:
    return RequestSource(schema=StructType([StructField("amount", DoubleType())]))


def _make_registered_upstream_fv(
    *,
    name: str = "TXN_FV",
    version: str = "v1",
    online: bool = True,
    store_type: OnlineStoreType = OnlineStoreType.POSTGRES,
    entity_name: str = "USER",
    entity_join_keys: Optional[list[str]] = None,
) -> FeatureView:
    """Build an upstream FV that looks registered for RTFV tests."""
    join_keys = entity_join_keys or ["USER_ID"]
    schema = StructType(
        [
            StructField(join_keys[0], StringType()),
            StructField("avg_amount", DoubleType()),
        ]
    )
    mock_df = MagicMock()
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": [f"SELECT * FROM {name}"]}
    fv = FeatureView(
        name=name,
        entities=[Entity(name=entity_name, join_keys=join_keys)],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=online, store_type=store_type),
    )
    fv._version = FeatureViewVersion(version)
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._infer_schema_df = mock_df
    fv._status = FeatureViewStatus.ACTIVE
    return fv


def _make_rtfv(
    *,
    name: str = "MY_RTFV",
    entity_name: str = "USER",
    entity_join_keys: Optional[list[str]] = None,
    upstream_fv: Optional[FeatureView] = None,
) -> FeatureView:
    join_keys = entity_join_keys or ["USER_ID"]
    upstream = upstream_fv or _make_registered_upstream_fv(entity_join_keys=join_keys)
    rtc = RealtimeConfig(
        compute_fn=rtfv_compute_fn,
        sources=[_build_request_source(), upstream],
        output_schema=_RTFV_OUTPUT_SCHEMA,
    )
    return FeatureView(
        name=name,
        entities=[Entity(name=entity_name, join_keys=join_keys)],
        realtime_config=rtc,
    )


def _new_fs_with_mocks(
    *,
    metadata_manager: Optional[MagicMock] = None,
    session: Optional[MagicMock] = None,
) -> FeatureStore:
    """Bare-bones FeatureStore with all I/O mocked.

    Note: ``register_feature_view`` / ``delete_feature_view`` /
    ``list_feature_views`` flow through ``switch_warehouse``, which calls
    ``session.get_current_warehouse()`` and feeds the result through
    :class:`SqlIdentifier`. Returning a real string keeps that decorator
    a no-op (the comparison ``original == default`` is False so the path
    enters but no-ops on the MagicMock ``use_warehouse``).
    """
    fs = object.__new__(FeatureStore)
    sess = session or MagicMock()
    sess.get_current_warehouse.return_value = "WH"
    md = metadata_manager or MagicMock()
    object.__setattr__(fs, "_session", sess)
    object.__setattr__(
        fs,
        "_config",
        fs_mod._FeatureStoreConfig(database=SqlIdentifier("DB"), schema=SqlIdentifier("SCH")),
    )
    object.__setattr__(fs, "_telemetry_stmp", {})
    object.__setattr__(fs, "_metadata_manager", md)
    object.__setattr__(fs, "_default_warehouse", SqlIdentifier("WH"))
    return fs


class RegisterRealtimePreconditionTest(absltest.TestCase):
    """Validation paths in ``register_feature_view`` (RTFV branch) before any side effect."""

    def test_rejects_draft_upstream(self) -> None:
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        upstream = _make_registered_upstream_fv()
        upstream._status = FeatureViewStatus.DRAFT
        upstream._version = None
        rtfv = _make_rtfv(upstream_fv=upstream)
        with self.assertRaisesRegex(ValueError, "not registered"):
            fs.register_feature_view(rtfv, "v1")

    def test_rejects_offline_upstream(self) -> None:
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        upstream = _make_registered_upstream_fv(online=False)
        rtfv = _make_rtfv(upstream_fv=upstream)
        with self.assertRaisesRegex(ValueError, "online=True"):
            fs.register_feature_view(rtfv, "v1")

    def test_rejects_hybrid_table_upstream(self) -> None:
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        upstream = _make_registered_upstream_fv(store_type=OnlineStoreType.HYBRID_TABLE)
        rtfv = _make_rtfv(upstream_fv=upstream)
        with self.assertRaisesRegex(ValueError, "OnlineStoreType.POSTGRES"):
            fs.register_feature_view(rtfv, "v1")

    def test_rejects_upstream_key_not_in_declared_entities(self) -> None:
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        # Upstream uses USER_ID; RTFV declares a different superset that
        # does NOT include USER_ID -> upstream key escapes the declared set.
        upstream = _make_registered_upstream_fv()
        rtfv = _make_rtfv(upstream_fv=upstream, entity_name="OTHER", entity_join_keys=["OTHER_ID"])
        with self.assertRaisesRegex(ValueError, "must be a subset"):
            fs.register_feature_view(rtfv, "v1")


class RegisterRealtimeHappyPathTest(absltest.TestCase):
    """End-to-end register flow with all I/O mocked."""

    def test_register_issues_create_tag_set_tag_and_metadata_save(self) -> None:
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        rtfv = _make_rtfv(name="MY_RTFV")
        object.__setattr__(fs, "get_feature_view", MagicMock(return_value=rtfv))

        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            fs.register_feature_view(rtfv, "v1")

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        create_sqls = [s for s in sql_texts if "CREATE ONLINE FEATURE TABLE" in s and "MY_RTFV$v1$ONLINE" in s]
        self.assertEqual(len(create_sqls), 1, sql_texts)
        create_sql = create_sqls[0]
        self.assertIn('PRIMARY KEY ("USER_ID")', create_sql)
        self.assertIn("TARGET_LAG='0 seconds'", create_sql)
        self.assertIn("FROM SPECIFICATION", create_sql)

        # Per-entity tag emitted alongside the FS object-type tag.
        set_tag_sqls = [s for s in sql_texts if "ALTER ONLINE FEATURE TABLE" in s and "SET TAG" in s]
        self.assertTrue(any("USER_ID" in s for s in set_tag_sqls), set_tag_sqls)

        # RealtimeConfigMetadata persisted with the right shape.
        self.assertEqual(md.save_realtime_config.call_count, 1)
        saved_meta: RealtimeConfigMetadata = md.save_realtime_config.call_args.args[0]
        self.assertEqual(saved_meta.name, "MY_RTFV")
        self.assertEqual(saved_meta.version, "v1")
        self.assertEqual(saved_meta.compute_fn_name, "rtfv_compute_fn")
        self.assertIn("def rtfv_compute_fn", saved_meta.compute_fn_source)
        self.assertEqual(len(saved_meta.sources), 1)
        self.assertEqual(saved_meta.sources[0].fv_name, "TXN_FV")
        self.assertEqual(saved_meta.sources[0].fv_version, "v1")
        self.assertIsNone(saved_meta.sources[0].slice_columns)
        # output_columns mirrors output_schema field names (case-preserved).
        self.assertEqual(
            [c.upper() for c in saved_meta.output_columns],
            ["RISK_SCORE", "RISK_BUCKET"],
        )

    def test_save_metadata_failure_drops_oft_but_skips_metadata_delete(self) -> None:
        """Save raising before the metadata row lands → OFT dropped, metadata delete NOT called.

        Calling ``delete_realtime_config`` here would tear down a concurrent
        registrar's row, which is the TOCTOU bug the rollback rewrite fixes.
        """
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        md.save_realtime_config.side_effect = RuntimeError("boom")

        rtfv = _make_rtfv(name="MY_RTFV")
        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            with self.assertRaisesRegex(RuntimeError, "Failed to register"):
                fs.register_feature_view(rtfv, "v1")

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        self.assertTrue(any("DROP ONLINE FEATURE TABLE IF EXISTS" in s for s in sql_texts), sql_texts)
        md.delete_realtime_config.assert_not_called()

    def test_post_save_failure_drops_oft_and_deletes_metadata(self) -> None:
        """Failure after the metadata row landed → OFT dropped AND metadata delete fires."""
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        # Trip the per-entity SET TAG step (runs strictly after save) by
        # matching the per-entity tag name; leave the FS object-type tag
        # set inside create_realtime_online_feature_table untouched.
        original_sql = sess.sql

        def sql_side_effect(stmt: str) -> Any:
            if "ALTER ONLINE FEATURE TABLE" in stmt and "SET TAG" in stmt and "SNOWML_FEATURE_STORE_OBJECT" not in stmt:
                raise RuntimeError("tag failed")
            return original_sql.return_value

        sess.sql = MagicMock(side_effect=sql_side_effect)
        sess.sql.return_value = original_sql.return_value

        rtfv = _make_rtfv(name="MY_RTFV")
        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            with self.assertRaisesRegex(RuntimeError, "Failed to register"):
                fs.register_feature_view(rtfv, "v1")

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        self.assertTrue(any("DROP ONLINE FEATURE TABLE IF EXISTS" in s for s in sql_texts), sql_texts)
        md.delete_realtime_config.assert_called_once()

    def test_pre_create_failure_leaves_no_resources_to_roll_back(self) -> None:
        """CREATE itself failing → no DROP issued, no metadata delete issued."""
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        original_sql = sess.sql

        def sql_side_effect(stmt: str) -> Any:
            if "CREATE ONLINE FEATURE TABLE" in stmt:
                raise RuntimeError("create failed")
            return original_sql.return_value

        sess.sql = MagicMock(side_effect=sql_side_effect)
        sess.sql.return_value = original_sql.return_value

        rtfv = _make_rtfv(name="MY_RTFV")
        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            # create_realtime_online_feature_table wraps the CREATE failure into
            # its own "Create realtime online feature table ..." SnowflakeMLException,
            # which the outer register_realtime_feature_view re-raises as-is.
            with self.assertRaisesRegex(RuntimeError, "Create realtime online feature table"):
                fs.register_feature_view(rtfv, "v1")

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        self.assertFalse(any("DROP ONLINE FEATURE TABLE IF EXISTS" in s for s in sql_texts), sql_texts)
        md.delete_realtime_config.assert_not_called()

    def test_register_persists_entity_names_in_metadata(self) -> None:
        """Resolved entity names are captured at register time for list-time use."""
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        rtfv = _make_rtfv(name="MY_RTFV")
        object.__setattr__(fs, "get_feature_view", MagicMock(return_value=rtfv))

        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            fs.register_feature_view(rtfv, "v1")

        saved_meta: RealtimeConfigMetadata = md.save_realtime_config.call_args.args[0]
        self.assertEqual(saved_meta.entity_names, ["USER"])


class DeleteRealtimeFeatureViewTest(absltest.TestCase):
    def test_delete_drops_oft_and_metadata_only(self) -> None:
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)

        rtfv = _make_rtfv(name="MY_RTFV")
        rtfv._version = FeatureViewVersion("v1")
        rtfv._status = FeatureViewStatus.ACTIVE
        rtfv._database = SqlIdentifier("DB")
        rtfv._schema = SqlIdentifier("SCH")

        fs.delete_feature_view(rtfv)
        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        # OFT drop fires; DT/View drops should NOT.
        self.assertTrue(any("DROP ONLINE FEATURE TABLE IF EXISTS" in s for s in sql_texts), sql_texts)
        self.assertFalse(any("DROP DYNAMIC TABLE" in s for s in sql_texts), sql_texts)
        self.assertFalse(any("DROP VIEW IF EXISTS" in s for s in sql_texts), sql_texts)
        md.delete_realtime_config.assert_called_once_with("MY_RTFV", "v1")
        # Streaming / aggregation metadata cleanup must not fire.
        md.delete_feature_view_metadata.assert_not_called()


class ListRealtimeFeatureViewsTest(absltest.TestCase):
    def test_list_surfaces_realtime_rows_with_kind_column(self) -> None:
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))
        # No backing OFT row needed for this test; we just verify the listing
        # row shape and ``kind`` column for the RTFV metadata entries.
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        # No actual session.create_dataframe call needed — we intercept it.
        captured: dict[str, Any] = {}

        def _create_df(values: list[list[Any]], schema: Any) -> MagicMock:
            captured["values"] = values
            captured["schema"] = schema
            return MagicMock()

        sess.create_dataframe.side_effect = _create_df

        meta = RealtimeConfigMetadata(
            name="MY_RTFV",
            version="v1",
            desc="hi",
            compute_fn_name="rtfv_compute_fn",
            compute_fn_source="def rtfv_compute_fn(req, txn): return None\n",
            sources=[FvSourceRef(fv_name="TXN_FV", fv_version="v1")],
            request_schema_json="[]",
            output_schema_json="[]",
            output_columns=["risk_score", "risk_bucket"],
            entity_names=["USER"],
        )
        md.list_realtime_config_metadata.return_value = [meta]

        # Listing must NOT rehydrate upstream FVs to derive entity names.
        get_fv_spy = MagicMock(side_effect=AssertionError("get_feature_view called during list"))
        object.__setattr__(fs, "get_feature_view", get_fv_spy)

        fs.list_feature_views()
        rows: list[list[Any]] = captured["values"]
        self.assertEqual(len(rows), 1)
        # verbose=False (default): only base fields are passed to
        # create_dataframe; verbose-only fields live in output_values_extra.
        # BASE has 18 columns: ..., kind (16), append_only (17).
        # source_refs is verbose-only and is not present in base rows.
        self.assertEqual(len(rows[0]), 18)
        self.assertEqual(rows[0][16], _FV_KIND_REALTIME)
        self.assertFalse(rows[0][17])  # append_only — always False for RTFV
        self.assertEqual(rows[0][0], "MY_RTFV")
        self.assertEqual(rows[0][7], ["USER"])
        get_fv_spy.assert_not_called()
        self.assertEqual(rows[0][1], "v1")

        captured.clear()
        fs.list_feature_views(verbose=True)
        verbose_rows: list[list[Any]] = captured["values"]
        self.assertEqual(len(verbose_rows), 1)
        # verbose=True: base + extra fields are merged (21 total).
        # Extra fields: initialization_warehouse (18), source_refs (19), backup_source (20).
        self.assertEqual(len(verbose_rows[0]), 21)
        self.assertIsNone(verbose_rows[0][18])  # initialization_warehouse — always None for RTFV
        self.assertIsNone(verbose_rows[0][19])  # source_refs — always None for RTFV
        self.assertIsNone(verbose_rows[0][20])  # backup_source — always None for RTFV


if __name__ == "__main__":
    absltest.main()
