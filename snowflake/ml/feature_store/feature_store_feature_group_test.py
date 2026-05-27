"""Unit tests for FeatureStore FeatureGroup CRUD with a mocked Snowpark session.

Covers:

- ``register_feature_group`` precondition errors (unregistered FV, missing entity,
  online+POSTGRES validation, name collision).
- Round-trip equivalence: ``register -> get`` returns a structurally equivalent
  ``FeatureGroup`` (name, desc, auto_prefix, source refs incl. slice + alias).
- ``list_feature_groups`` filters tagged objects by ``type=FEATURE_GROUP``.
- ``delete_feature_group`` issues ``DROP ONLINE FEATURE TABLE IF EXISTS`` and
  removes the metadata row.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import (
    feature_group as fg_mod,
    feature_store as fs_mod,
    online_service,
)
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_group import FeatureGroup, FeatureGroupVersion
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.metadata_manager import (
    FeatureGroupMetadata,
    FeatureGroupSourceRef,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def _make_registered_fv(
    *,
    name: str,
    version: str,
    feature_columns: list[str],
    online: bool = True,
    store_type: OnlineStoreType = OnlineStoreType.POSTGRES,
    entity_name: str = "USER",
    entity_join_keys: Optional[list[str]] = None,
    schema_key_col: Optional[str] = None,
) -> FeatureView:
    """Build a FeatureView that looks registered for FeatureGroup tests."""
    join_keys = entity_join_keys or ["USER_ID"]
    # Schema must contain every join key so FeatureView._validate accepts the FV;
    # ``schema_key_col`` is an explicit override kept for single-key callers.
    key_cols = [schema_key_col] if schema_key_col is not None else list(join_keys)
    schema = StructType(
        [StructField(c, StringType()) for c in key_cols] + [StructField(c, DoubleType()) for c in feature_columns]
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


def _rtfv_compute_fn(req: Any, upstream: Any) -> Any:
    """Placeholder ``compute_fn`` for RTFV fixtures; tests never invoke it."""
    return req


def _rtfv_compute_fn_no_request(upstream: Any) -> Any:
    """Placeholder ``compute_fn`` for no-RequestSource RTFV fixtures."""
    return upstream


def _make_registered_rtfv(
    *,
    name: str,
    version: str,
    output_fields: list[StructField],
    request_fields: Optional[list[StructField]] = None,
    upstream: Optional[FeatureView] = None,
    join_keys: Optional[list[str]] = None,
    with_request_source: bool = True,
) -> FeatureView:
    """Build a FeatureView that looks like a registered RTFV for FG tests.

    ``output_fields`` is the canonical ``compute_fn`` return shape (may
    include the join key). ``upstream`` defaults to a single-key BFV
    matching ``join_keys``; pass a custom one to exercise upstream PK
    datatype variants. Set ``with_request_source=False`` to build an RTFV
    whose ``RealtimeConfig.request_source`` is ``None`` (legal since the
    RequestSource-optional change); ``request_fields`` is ignored in that
    case.
    """
    join_keys = join_keys or ["USER_ID"]
    upstream = upstream or _make_registered_fv(
        name=f"{name}_UPSTREAM",
        version=version,
        feature_columns=["UPSTREAM_F"],
        entity_join_keys=join_keys,
    )
    if with_request_source:
        request = RequestSource(schema=StructType(request_fields or []))
        rtc = RealtimeConfig(
            compute_fn=_rtfv_compute_fn,
            sources=[request, upstream],
            output_schema=StructType(output_fields),
        )
    else:
        rtc = RealtimeConfig(
            compute_fn=_rtfv_compute_fn_no_request,
            sources=[upstream],
            output_schema=StructType(output_fields),
        )
    fv = FeatureView(
        name=name,
        entities=[Entity(name="USER", join_keys=join_keys)],
        realtime_config=rtc,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
    )
    fv._version = FeatureViewVersion(version)
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._status = FeatureViewStatus.ACTIVE
    return fv


def _new_fs_with_mocks(
    *,
    metadata_manager: Optional[MagicMock] = None,
    session: Optional[MagicMock] = None,
) -> FeatureStore:
    """Construct a bare-bones FeatureStore with all I/O mocked."""
    fs = object.__new__(FeatureStore)
    sess = session or MagicMock()
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
    object.__setattr__(fs, "_online_service_access", None)
    return fs


class RegisterFeatureGroupPreconditionTest(absltest.TestCase):
    """Validation paths in ``register_feature_group`` before any side effects."""

    def test_rejects_non_feature_group(self) -> None:
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "expects a FeatureGroup"):
            fs.register_feature_group("not a fg", "v1")  # type: ignore[arg-type]

    def test_rejects_invalid_version(self) -> None:
        fs = _new_fs_with_mocks()
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name="FG", features=[fv])
        with self.assertRaisesRegex(ValueError, "valid feature group version"):
            fs.register_feature_group(fg, "has space")

    def test_rejects_unregistered_source(self) -> None:
        fs = _new_fs_with_mocks()
        fv_draft = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv_draft._status = FeatureViewStatus.DRAFT
        fg = FeatureGroup(name="FG", features=[fv_draft])
        with self.assertRaisesRegex(ValueError, "not registered"):
            fs.register_feature_group(fg, "v1")

    def test_rejects_offline_source(self) -> None:
        fs = _new_fs_with_mocks()
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"], online=False)
        fg = FeatureGroup(name="FG", features=[fv])
        with self.assertRaisesRegex(ValueError, "online=True"):
            fs.register_feature_group(fg, "v1")

    def test_rejects_hybrid_table_source(self) -> None:
        fs = _new_fs_with_mocks()
        fv = _make_registered_fv(
            name="USER_FV",
            version="v1",
            feature_columns=["F1"],
            store_type=OnlineStoreType.HYBRID_TABLE,
        )
        fg = FeatureGroup(name="FG", features=[fv])
        with self.assertRaisesRegex(ValueError, "OnlineStoreType.POSTGRES"):
            fs.register_feature_group(fg, "v1")

    def test_rejects_when_entity_missing(self) -> None:
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=False))
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name="FG", features=[fv])
        with self.assertRaisesRegex(ValueError, "is not registered"):
            fs.register_feature_group(fg, "v1")


class RegisterFeatureGroupHappyPathTest(absltest.TestCase):
    """End-to-end register flow with all I/O mocked, asserting SQL + metadata calls."""

    def test_register_issues_create_tag_and_metadata_save(self) -> None:
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name="FG", features=[fv], desc="hello", auto_prefix=True)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            fs.register_feature_group(fg, "v1")

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        # Lowercase ``v1`` survives as a SQL-quoted identifier so casing
        # round-trips through the spec JSON and Query API.
        create_sqls = [s for s in sql_texts if "CREATE ONLINE FEATURE TABLE" in s and '"FG$v1$ONLINE"' in s]
        self.assertEqual(len(create_sqls), 1, sql_texts)
        create_sql = create_sqls[0]
        # ``PRIMARY KEY (...)`` must precede ``TARGET_LAG=...`` or the SQL parser fails.
        self.assertIn('PRIMARY KEY ("USER_ID")', create_sql)
        self.assertIn("WAREHOUSE=WH", create_sql)
        self.assertIn("TARGET_LAG='0 seconds'", create_sql)
        self.assertIn("FROM SPECIFICATION", create_sql)
        # CREATE ONLINE FEATURE TABLE has no COMMENT clause; the desc lives on metadata.
        self.assertNotIn("COMMENT=", create_sql)
        self.assertTrue(any("SET TAG" in s and '"FG$v1$ONLINE"' in s for s in sql_texts), sql_texts)
        self.assertEqual(md.save_feature_group_metadata.call_count, 1)
        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        self.assertEqual(saved_meta.name, "FG")
        self.assertEqual(saved_meta.version, "v1")
        self.assertEqual(saved_meta.desc, "hello")
        self.assertTrue(saved_meta.auto_prefix)
        self.assertEqual(len(saved_meta.sources), 1)
        self.assertEqual(saved_meta.sources[0].fv_name, "USER_FV")
        self.assertEqual(saved_meta.sources[0].fv_version, "v1")
        self.assertIsNone(saved_meta.sources[0].slice_columns)
        self.assertIsNone(saved_meta.sources[0].alias)
        self.assertEqual(saved_meta.output_columns, fg.output_columns)

    def test_register_persists_output_columns_for_multiple_sources(self) -> None:
        """Persisted ``output_columns`` matches ``FeatureGroup.output_columns`` resolved at register time."""
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        fv_a = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1", "F2"])
        fv_b = _make_registered_fv(name="ACCT_FV", version="v1", feature_columns=["F3"])
        fg = FeatureGroup(name="FG", features=[fv_a, fv_b], desc="hi", auto_prefix=True)
        expected_output_columns = list(fg.output_columns)
        # Guard against an empty equality check silently passing.
        self.assertGreater(len(expected_output_columns), 0)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        with patch.object(online_service, "assert_online_service_running_with_query_endpoint"):
            fs.register_feature_group(fg, "v1")

        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        self.assertEqual(saved_meta.output_columns, expected_output_columns)

    def test_pk_is_union_when_sources_have_different_join_keys(self) -> None:
        """OFT PK is the first-seen ordered union across heterogeneous-key sources."""
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))

        original = online_service.assert_online_service_running_with_query_endpoint
        online_service.assert_online_service_running_with_query_endpoint = MagicMock()

        # FV_A keyed on (USER_ID); FV_B keyed on (USER_ID, ITEM_ID) — wider grain.
        fv_user = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv_user_item = _make_registered_fv(
            name="USER_ITEM_FV",
            version="v1",
            feature_columns=["F2"],
            entity_name="USER_ITEM",
            entity_join_keys=["USER_ID", "ITEM_ID"],
        )
        fg = FeatureGroup(name="FG", features=[fv_user, fv_user_item])
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        try:
            fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        create_sqls = [s for s in sql_texts if "CREATE ONLINE FEATURE TABLE" in s]
        self.assertEqual(len(create_sqls), 1, sql_texts)
        create_sql = create_sqls[0]
        # PK is the first-seen union: USER_ID (from FV_A) then ITEM_ID (new in FV_B).
        self.assertIn('PRIMARY KEY ("USER_ID", "ITEM_ID")', create_sql)
        # Spec rendered into the FROM SPECIFICATION blob carries the same union.
        spec_dict = fg._to_spec(database="DB", schema="SCH", version="v1").to_dict()
        self.assertEqual(spec_dict["spec"]["ordered_entity_column_names"], ["USER_ID", "ITEM_ID"])


class RegisterFeatureGroupWithRtfvSourceTest(absltest.TestCase):
    """Register-time validations + persistence for mixed-kind / multi-RTFV FGs."""

    def _new_register_setup(self) -> tuple[FeatureStore, MagicMock, MagicMock]:
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))
        return fs, sess, md

    def _stub_online_service(self) -> Any:
        """Stash + replace the running-service assertion so tests don't hit the network."""
        original = online_service.assert_online_service_running_with_query_endpoint
        online_service.assert_online_service_running_with_query_endpoint = MagicMock()
        return original

    def test_mixed_bfv_and_rtfv_registers(self) -> None:
        """Mixed-kind FG (BFV + RTFV) passes every register-time validation."""
        fs, sess, md = self._new_register_setup()
        bfv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["BALANCE"])
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
            request_fields=[StructField("WEIGHT", DoubleType())],
        )
        fg = FeatureGroup(name="FG", features=[bfv, rtfv], auto_prefix=False)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        # Persisted output_columns inherits the deduped property; the RTFV's PK
        # column is dropped, the BFV's BALANCE column stays.
        self.assertEqual(saved_meta.output_columns, ["BALANCE", "WEIGHTED_BALANCE"])

    def test_cross_source_join_key_datatype_conflict_rejected(self) -> None:
        """BFV says VARCHAR, RTFV's upstream says INTEGER for the same key -> reject with both sources named."""
        fs, _sess, _md = self._new_register_setup()
        bfv_string_key = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["BALANCE"])
        # RTFV's upstream uses an INTEGER USER_ID, so resolve_realtime_join_key_fields
        # picks up INTEGER for the RTFV side of the FG-level type table.
        rtfv_upstream_int = _make_registered_fv(
            name="RT_UPSTREAM",
            version="v1",
            feature_columns=["AVG_BALANCE"],
            schema_key_col="USER_ID",
        )
        # Override the upstream's USER_ID column type to IntegerType after construction.
        assert rtfv_upstream_int.feature_df is not None
        rtfv_upstream_int.feature_df.schema = StructType(
            [StructField("USER_ID", IntegerType()), StructField("AVG_BALANCE", DoubleType())]
        )
        rtfv_upstream_int._infer_schema_df = rtfv_upstream_int.feature_df
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[StructField("WEIGHTED_BALANCE", DoubleType())],
            upstream=rtfv_upstream_int,
        )
        fg = FeatureGroup(name="FG", features=[bfv_string_key, rtfv])
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            with self.assertRaisesRegex(ValueError, "join key 'USER_ID'") as cm:
                fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original
        msg = str(cm.exception)
        self.assertIn("USER_FV@v1", msg)
        self.assertIn("RT_FV@v1", msg)

    def test_request_source_overlaps_fg_pk_rejected(self) -> None:
        """An RTFV's RequestSource column that collides with the FG's superset PK is rejected."""
        fs, _sess, _md = self._new_register_setup()
        bfv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["BALANCE"])
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[StructField("WEIGHTED_BALANCE", DoubleType())],
            # RequestSource column ``USER_ID`` overlaps the FG superset PK.
            request_fields=[StructField("USER_ID", StringType())],
        )
        fg = FeatureGroup(name="FG", features=[bfv, rtfv])
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            with self.assertRaisesRegex(ValueError, "overlap the FeatureGroup's superset primary key") as cm:
                fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original
        self.assertIn("RT_FV@v1", str(cm.exception))

    def test_cross_rtfv_request_source_datatype_conflict_rejected(self) -> None:
        """Two RTFVs declare the same canonical RequestSource column with conflicting types -> reject."""
        fs, _sess, _md = self._new_register_setup()
        rtfv_a = _make_registered_rtfv(
            name="RT_A",
            version="v1",
            output_fields=[StructField("FEAT_A", DoubleType())],
            request_fields=[StructField("AMOUNT", DoubleType())],
        )
        rtfv_b = _make_registered_rtfv(
            name="RT_B",
            version="v1",
            output_fields=[StructField("FEAT_B", DoubleType())],
            request_fields=[StructField("AMOUNT", IntegerType())],
        )
        fg = FeatureGroup(name="FG", features=[rtfv_a, rtfv_b])
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            with self.assertRaisesRegex(ValueError, "request column 'AMOUNT'") as cm:
                fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original
        msg = str(cm.exception)
        self.assertIn("RT_A@v1", msg)
        self.assertIn("RT_B@v1", msg)

    def test_cross_rtfv_request_source_datatype_conflict_case_insensitive(self) -> None:
        """``AMOUNT`` vs ``amount`` canonicalize to the same key and still raise on type drift."""
        fs, _sess, _md = self._new_register_setup()
        rtfv_a = _make_registered_rtfv(
            name="RT_A",
            version="v1",
            output_fields=[StructField("FEAT_A", DoubleType())],
            request_fields=[StructField("AMOUNT", DoubleType())],
        )
        rtfv_b = _make_registered_rtfv(
            name="RT_B",
            version="v1",
            output_fields=[StructField("FEAT_B", DoubleType())],
            request_fields=[StructField("amount", IntegerType())],
        )
        fg = FeatureGroup(name="FG", features=[rtfv_a, rtfv_b])
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            with self.assertRaisesRegex(ValueError, "realtime feature view sources disagree"):
                fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

    def test_cross_rtfv_request_source_same_name_same_datatype_passes(self) -> None:
        """De-facto shared column: same canonical name + same datatype across RTFVs registers."""
        fs, _sess, md = self._new_register_setup()
        rtfv_a = _make_registered_rtfv(
            name="RT_A",
            version="v1",
            output_fields=[StructField("FEAT_A", DoubleType())],
            request_fields=[StructField("AMOUNT", DoubleType())],
        )
        rtfv_b = _make_registered_rtfv(
            name="RT_B",
            version="v1",
            output_fields=[StructField("FEAT_B", DoubleType())],
            request_fields=[StructField("AMOUNT", DoubleType())],
        )
        # ``auto_prefix=False`` keeps the assertion focused on the validator
        # outcome rather than the prefixing rules covered in feature_group_test.py.
        fg = FeatureGroup(name="FG", features=[rtfv_a, rtfv_b], auto_prefix=False)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        self.assertEqual(saved_meta.output_columns, ["FEAT_A", "FEAT_B"])

    def test_register_fg_with_no_request_source_rtfv_passes(self) -> None:
        """An RTFV source with ``request_source=None`` is skipped by the RequestSource validators."""
        fs, _sess, md = self._new_register_setup()
        bfv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["BALANCE"])
        # RTFV without a RequestSource: the PK-overlap and cross-RTFV datatype
        # validators must skip it instead of dereferencing ``request_source``.
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("DOUBLED_BALANCE", DoubleType()),
            ],
            with_request_source=False,
        )
        fg = FeatureGroup(name="FG", features=[bfv, rtfv], auto_prefix=False)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        self.assertEqual(saved_meta.output_columns, ["BALANCE", "DOUBLED_BALANCE"])

    def test_cross_rtfv_request_source_datatype_conflict_skips_no_rs_rtfv(self) -> None:
        """Cross-RTFV datatype check ignores RTFVs that do not declare a ``RequestSource``."""
        fs, _sess, md = self._new_register_setup()
        rtfv_a = _make_registered_rtfv(
            name="RT_A",
            version="v1",
            output_fields=[StructField("FEAT_A", DoubleType())],
            request_fields=[StructField("AMOUNT", DoubleType())],
        )
        # No RequestSource: contributes no canonical request columns and so
        # cannot conflict with ``RT_A``'s ``AMOUNT``.
        rtfv_b = _make_registered_rtfv(
            name="RT_B",
            version="v1",
            output_fields=[StructField("FEAT_B", DoubleType())],
            with_request_source=False,
        )
        fg = FeatureGroup(name="FG", features=[rtfv_a, rtfv_b], auto_prefix=False)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        self.assertEqual(saved_meta.output_columns, ["FEAT_A", "FEAT_B"])

    def test_persisted_output_columns_dedupes_pk_for_rtfv_source(self) -> None:
        """End-to-end through register: persisted ``output_columns`` drops the FG PK column."""
        fs, _sess, md = self._new_register_setup()
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        fg = FeatureGroup(name="FG", features=[rtfv], auto_prefix=False)
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))

        original = self._stub_online_service()
        try:
            fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        saved_meta: FeatureGroupMetadata = md.save_feature_group_metadata.call_args.args[0]
        self.assertEqual(saved_meta.output_columns, ["WEIGHTED_BALANCE"])


class FeatureGroupOftNameTest(absltest.TestCase):
    """Helper-level invariants for ``feature_group_oft_name``."""

    def test_feature_group_oft_name_canonicalizes_lowercase_name(self) -> None:
        """Regression: any input casing yields the same OFT identifier as the canonical upper-case form.

        Without canonicalization the helper would emit a quoted, case-preserving
        identifier while ``register_feature_group`` stores metadata under the
        upper-cased canonical name, breaking list/delete/read lookups.
        """
        lower = fg_mod.feature_group_oft_name("fraud_features", "v1")
        upper = fg_mod.feature_group_oft_name("FRAUD_FEATURES", "v1")
        mixed = fg_mod.feature_group_oft_name("Fraud_Features", "v1")
        quoted = fg_mod.feature_group_oft_name('"FRAUD_FEATURES"', "v1")
        self.assertEqual(lower.resolved(), upper.resolved())
        self.assertEqual(lower.resolved(), mixed.resolved())
        self.assertEqual(lower.resolved(), quoted.resolved())
        self.assertIn("FRAUD_FEATURES", lower.resolved())
        self.assertTrue(lower.resolved().endswith("$ONLINE"))

    def test_feature_group_oft_name_accepts_typed_inputs(self) -> None:
        """Pre-validated ``SqlIdentifier`` / ``FeatureGroupVersion`` inputs round-trip identically."""
        raw = fg_mod.feature_group_oft_name("fraud_features", "v1")
        typed = fg_mod.feature_group_oft_name(SqlIdentifier("fraud_features"), FeatureGroupVersion("v1"))
        self.assertEqual(raw.resolved(), typed.resolved())


class RegisterFeatureGroupRollbackTest(absltest.TestCase):
    """Failure paths must only undo what THIS call created.

    Regression for a TOCTOU race: ``reject_name_collision`` is a check, not a
    lock, so two concurrent registers can both pass it. If our ``CREATE`` then
    fails because the *other* caller's OFT already exists, the rollback must
    not drop their OFT or delete their metadata row.
    """

    def _new_register_setup(self) -> tuple[FeatureStore, MagicMock, MagicMock, FeatureGroup]:
        md = MagicMock()
        sess = MagicMock()
        fs = _new_fs_with_mocks(metadata_manager=md, session=sess)
        object.__setattr__(fs, "_validate_entity_exists", MagicMock(return_value=True))
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "_get_fv_backend_representations", MagicMock(return_value=[]))
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name="FG", features=[fv])
        object.__setattr__(fs, "get_feature_group", MagicMock(return_value=fg))
        return fs, sess, md, fg

    def test_create_oft_failure_does_not_drop_other_callers_oft(self) -> None:
        """Concurrent CREATE failure must not issue a DROP — we never created it."""
        fs, sess, md, fg = self._new_register_setup()

        def sql_side_effect(query: str, *_: Any, **__: Any) -> MagicMock:
            if "CREATE ONLINE FEATURE TABLE" in query:
                raise RuntimeError("002002 (42710): Object 'FG$v1$ONLINE' already exists.")
            return MagicMock()

        sess.sql.side_effect = sql_side_effect
        original = online_service.assert_online_service_running_with_query_endpoint
        online_service.assert_online_service_running_with_query_endpoint = MagicMock()

        try:
            with self.assertRaises(Exception):  # noqa: B017
                fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        drops = [s for s in sql_texts if "DROP ONLINE FEATURE TABLE" in s]
        self.assertEqual(drops, [], f"must not drop an OFT this call never created; saw {drops}")
        md.delete_feature_group_metadata.assert_not_called()

    def test_metadata_save_failure_drops_only_oft_this_call_created(self) -> None:
        """If CREATE succeeds and metadata save raises, DROP fires; metadata delete does not."""
        fs, sess, md, fg = self._new_register_setup()
        md.save_feature_group_metadata.side_effect = RuntimeError("metadata save failed")
        object.__setattr__(fs, "_tag_oft", MagicMock())

        original = online_service.assert_online_service_running_with_query_endpoint
        online_service.assert_online_service_running_with_query_endpoint = MagicMock()

        try:
            with self.assertRaises(Exception):  # noqa: B017
                fs.register_feature_group(fg, "v1")
        finally:
            online_service.assert_online_service_running_with_query_endpoint = original

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        drops = [s for s in sql_texts if "DROP ONLINE FEATURE TABLE" in s]
        self.assertEqual(len(drops), 1, f"expected exactly one DROP for the OFT this call created; saw {drops}")
        # metadata_saved is set only after the save returns; a raising save means
        # we never confirmed the row landed, so we don't issue a delete that
        # could clobber a concurrent caller's row keyed by the same (name, version).
        md.delete_feature_group_metadata.assert_not_called()


class RoundTripTest(absltest.TestCase):
    """Verify ``register -> get`` reconstructs an equivalent FeatureGroup."""

    def test_get_from_metadata_round_trips_with_slice_and_alias(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1", "F2"])
        # Metadata as it would have been persisted by register (slice + alias).
        meta = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="hello",
            auto_prefix=True,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1", slice_columns=["F1"], alias="sender")],
        )
        md = MagicMock()
        md.get_feature_group_metadata = MagicMock(return_value=meta)
        fs = _new_fs_with_mocks(metadata_manager=md)
        object.__setattr__(fs, "get_feature_view", MagicMock(return_value=fv))
        # Stub the live-status hydrator so this test only exercises the metadata round-trip.
        object.__setattr__(fs, "_hydrate_fg_postgres_online_service", MagicMock())

        fg = fs.get_feature_group("FG", "v1")

        self.assertEqual(fg.name, "FG")
        self.assertIsNotNone(fg.version)
        self.assertEqual(str(fg.version), "v1")
        self.assertEqual(fg.desc, "hello")
        self.assertTrue(fg.auto_prefix)
        self.assertEqual(len(fg.features), 1)
        item = fg.features[0]
        from snowflake.ml.feature_store.feature_view import FeatureViewSlice

        self.assertIsInstance(item, FeatureViewSlice)
        self.assertEqual([n.resolved() for n in item.names], ["F1"])
        self.assertEqual(item.column_alias, "sender")
        self.assertEqual(fg.output_columns, ['"sender_F1"'])
        md.get_feature_group_metadata.assert_called_once_with("FG", "v1")

    def test_get_unknown_name_raises_not_found(self) -> None:
        md = MagicMock()
        md.get_feature_group_metadata = MagicMock(return_value=None)
        fs = _new_fs_with_mocks(metadata_manager=md)
        with self.assertRaisesRegex(ValueError, "not registered"):
            fs.get_feature_group("MISSING", "v1")

    def test_get_invalid_version_raises_value_error(self) -> None:
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "valid feature group version"):
            fs.get_feature_group("FG", "has space")

    def test_get_feature_group_hydrates_query_url(self) -> None:
        """``get_feature_group`` populates ``_postgres_online_query_url`` for read-time use."""
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        meta = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="hello",
            auto_prefix=True,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
        )
        md = MagicMock()
        md.get_feature_group_metadata = MagicMock(return_value=meta)
        fs = _new_fs_with_mocks(metadata_manager=md)
        object.__setattr__(fs, "get_feature_view", MagicMock(return_value=fv))

        stub_status = MagicMock()
        stub_status.status = "RUNNING"
        with patch.object(online_service, "fetch_online_service_status", return_value=stub_status), patch.object(
            online_service, "endpoint_url", return_value="https://q.example/svc"
        ):
            fg = fs.get_feature_group("FG", "v1")

        self.assertEqual(fg._postgres_online_query_url, "https://q.example/svc")


class ListFeatureGroupsTest(absltest.TestCase):
    """``list_feature_groups`` joins persisted FG metadata with the tag scan."""

    def test_filter_predicates_select_feature_group_only(self) -> None:
        captured: dict[str, Any] = {}

        def _lookup(tag_name: str, filter_fns: list[Any]) -> list[dict[str, Any]]:
            captured["tag_name"] = tag_name
            captured["filters"] = filter_fns
            return []

        sess = MagicMock()
        # Bypass real DataFrame construction; just round-trip the inputs.
        sess.create_dataframe = MagicMock(side_effect=lambda data, schema: (data, schema))
        md = MagicMock()
        md.list_feature_group_metadata = MagicMock(return_value=[])
        fs = _new_fs_with_mocks(session=sess, metadata_manager=md)
        object.__setattr__(fs, "_lookup_tagged_objects", _lookup)

        result = fs.list_feature_groups()

        self.assertEqual(captured["tag_name"], fs_mod._FEATURE_STORE_OBJECT_TAG)
        self.assertEqual(len(captured["filters"]), 2)
        self.assertTrue(captured["filters"][0]({"domain": "TABLE"}))
        self.assertFalse(captured["filters"][0]({"domain": "TASK"}))
        fg_tag = json.dumps({"type": "FEATURE_GROUP", "pkg_version": "1.2.3"})
        oft_tag = json.dumps({"type": "ONLINE_FEATURE_TABLE", "pkg_version": "1.2.3"})
        self.assertTrue(captured["filters"][1]({"tagValue": fg_tag}))
        self.assertFalse(captured["filters"][1]({"tagValue": oft_tag}))
        data, schema = result  # type: ignore[attr-defined]
        self.assertEqual(data, [])
        self.assertEqual(
            [f.name for f in schema.fields],
            ["NAME", "VERSION", "DESC", "OWNER", "AUTO_PREFIX", "SOURCES", "OUTPUT_COLUMNS"],
        )

    def test_rows_from_metadata_with_owner_join(self) -> None:
        """Rows come from list_feature_group_metadata, owner from the tag scan."""
        meta_v1 = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="first",
            auto_prefix=True,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
            output_columns=['"USER_FV_v1_F1"'],
        )
        meta_v2 = FeatureGroupMetadata(
            name="FG",
            version="v2",
            desc="second",
            auto_prefix=False,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
            output_columns=["F1"],
        )
        md = MagicMock()
        md.list_feature_group_metadata = MagicMock(return_value=[meta_v1, meta_v2])

        sess = MagicMock()
        sess.create_dataframe = MagicMock(side_effect=lambda data, schema: (data, schema))

        fs = _new_fs_with_mocks(session=sess, metadata_manager=md)
        # Only v1 has an owner row (simulates the v2 OFT dropped out of band).
        object.__setattr__(
            fs,
            "_lookup_tagged_objects",
            lambda tag, filters: [{"entityName": "FG$v1$ONLINE", "entityOwner": "ALICE"}],
        )

        result = fs.list_feature_groups()
        data, _schema = result  # type: ignore[attr-defined]
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0][0:5], ["FG", "v1", "first", "ALICE", True])
        self.assertEqual(data[1][0:5], ["FG", "v2", "second", None, False])
        self.assertEqual(data[0][6], json.dumps(['"USER_FV_v1_F1"']))
        self.assertEqual(data[1][6], json.dumps(["F1"]))

    def test_list_uses_persisted_output_columns_without_rehydrating_sources(self) -> None:
        """Regression contract: ``list_feature_groups`` must NOT call ``get_feature_view``.

        Older code rehydrated every source FV per list call (O(N*M) catalog
        hits, fragile when a source FV was unavailable). The hard
        ``assert_not_called`` here keeps a future refactor from silently
        reintroducing it.
        """
        meta = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="hi",
            auto_prefix=False,
            sources=[
                FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1"),
                FeatureGroupSourceRef(fv_name="ACCT_FV", fv_version="v1"),
            ],
            output_columns=["a", "b"],
        )
        md = MagicMock()
        md.list_feature_group_metadata = MagicMock(return_value=[meta])

        sess = MagicMock()
        sess.create_dataframe = MagicMock(side_effect=lambda data, schema: (data, schema))

        fs = _new_fs_with_mocks(session=sess, metadata_manager=md)
        object.__setattr__(
            fs,
            "_lookup_tagged_objects",
            lambda tag, filters: [{"entityName": "FG$v1$ONLINE", "entityOwner": "BOB"}],
        )
        get_fv_mock = MagicMock(spec=FeatureStore.get_feature_view)
        object.__setattr__(fs, "get_feature_view", get_fv_mock)

        data, _schema = fs.list_feature_groups()  # type: ignore[attr-defined]

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][6], json.dumps(["a", "b"]))
        get_fv_mock.assert_not_called()

    def test_list_handles_legacy_metadata_without_output_columns(self) -> None:
        """Pre-output_columns rows decode as ``None`` and render as ``[]`` (no rehydration fallback)."""
        meta = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="legacy",
            auto_prefix=False,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
            output_columns=None,
        )
        md = MagicMock()
        md.list_feature_group_metadata = MagicMock(return_value=[meta])

        sess = MagicMock()
        sess.create_dataframe = MagicMock(side_effect=lambda data, schema: (data, schema))
        fs = _new_fs_with_mocks(session=sess, metadata_manager=md)
        object.__setattr__(fs, "_lookup_tagged_objects", lambda tag, filters: [])
        get_fv_mock = MagicMock(spec=FeatureStore.get_feature_view)
        object.__setattr__(fs, "get_feature_view", get_fv_mock)

        data, _schema = fs.list_feature_groups()  # type: ignore[attr-defined]

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][6], json.dumps([]))
        get_fv_mock.assert_not_called()

    def test_owner_join_normalizes_mixed_case_version(self) -> None:
        """Mixed-case versions on either side of the join must still produce an owner match.

        Both sides route through :class:`SqlIdentifier` case-sensitive
        resolution; otherwise a mixed-case version's owner silently goes
        ``None``.
        """
        meta_lower = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="lower",
            auto_prefix=True,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
            output_columns=['"USER_FV_v1_F1"'],
        )
        meta_upper = FeatureGroupMetadata(
            name="FG",
            version="V2",
            desc="upper",
            auto_prefix=False,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
            output_columns=["F1"],
        )
        md = MagicMock()
        md.list_feature_group_metadata = MagicMock(return_value=[meta_lower, meta_upper])

        sess = MagicMock()
        sess.create_dataframe = MagicMock(side_effect=lambda data, schema: (data, schema))

        fs = _new_fs_with_mocks(session=sess, metadata_manager=md)
        object.__setattr__(
            fs,
            "_lookup_tagged_objects",
            lambda tag, filters: [
                {"entityName": "FG$v1$ONLINE", "entityOwner": "MIXED_OWNER"},
                {"entityName": "FG$V2$ONLINE", "entityOwner": "UPPER_OWNER"},
            ],
        )

        data, _schema = fs.list_feature_groups()  # type: ignore[attr-defined]

        owners_by_version = {row[1]: row[3] for row in data}
        self.assertEqual(owners_by_version, {"v1": "MIXED_OWNER", "V2": "UPPER_OWNER"})


class DeleteFeatureGroupTest(absltest.TestCase):
    """``delete_feature_group`` runs DROP and removes the metadata row."""

    def test_delete_runs_drop_and_metadata_delete(self) -> None:
        sess = MagicMock()
        md = MagicMock()
        fs = _new_fs_with_mocks(session=sess, metadata_manager=md)

        fs.delete_feature_group("FG", "v1")

        sql_texts = [c.args[0] for c in sess.sql.call_args_list]
        self.assertTrue(
            any("DROP ONLINE FEATURE TABLE IF EXISTS" in s and '"FG$v1$ONLINE"' in s for s in sql_texts),
            sql_texts,
        )
        md.delete_feature_group_metadata.assert_called_once_with("FG", "v1")

    def test_delete_invalid_version_raises_value_error(self) -> None:
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "valid feature group version"):
            fs.delete_feature_group("FG", "has space")


if __name__ == "__main__":
    absltest.main()
