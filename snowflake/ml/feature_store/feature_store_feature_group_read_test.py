"""Unit tests for ``FeatureStore.read_feature_group`` against a mocked Online Service."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store as fs_mod, online_service
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_group import FeatureGroup, FeatureGroupVersion
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
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

# Helpers duplicated from feature_store_feature_group_test.py — py_test targets can't share srcs.


def _make_registered_fv(
    *,
    name: str,
    version: str,
    feature_columns: list[str],
    online: bool = True,
    store_type: OnlineStoreType = OnlineStoreType.POSTGRES,
    entity_name: str = "USER",
    entity_join_keys: Optional[list[str]] = None,
    schema_key_col: str = "USER_ID",
) -> FeatureView:
    join_keys = entity_join_keys or ["USER_ID"]
    schema = StructType(
        [StructField(schema_key_col, StringType())] + [StructField(c, DoubleType()) for c in feature_columns]
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


def _new_fs_with_mocks(
    *,
    metadata_manager: Optional[MagicMock] = None,
    session: Optional[MagicMock] = None,
) -> FeatureStore:
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


def _make_hydrated_fg(
    *,
    name: str = "FG",
    version: str = "v1",
    fv_name: str = "USER_FV",
    fv_version: str = "v1",
    feature_columns: tuple[str, ...] = ("F1",),
    query_url: str = "https://q.example/svc",
) -> FeatureGroup:
    """Construct a FeatureGroup that mimics the one returned by ``get_feature_group``."""
    fv = _make_registered_fv(name=fv_name, version=fv_version, feature_columns=list(feature_columns))
    fg = FeatureGroup(name=name, features=[fv], auto_prefix=True)
    fg._version = FeatureGroupVersion(version)
    fg._postgres_online_query_url = query_url
    return fg


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
    """Build a FeatureView that looks like a registered RTFV for FG read tests.

    Set ``with_request_source=False`` to build an RTFV whose
    ``RealtimeConfig.request_source`` is ``None``; ``request_fields`` is
    ignored in that case.
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


def _make_hydrated_mixed_fg(
    *,
    name: str = "FG",
    version: str = "v1",
    request_fields: Optional[list[StructField]] = None,
    rtfv_output_fields: Optional[list[StructField]] = None,
    include_bfv: bool = True,
    query_url: str = "https://q.example/svc",
) -> FeatureGroup:
    """Construct a registered-looking FG with one BFV + one RTFV source."""
    sources: list[Any] = []
    if include_bfv:
        sources.append(_make_registered_fv(name="USER_FV", version="v1", feature_columns=["BALANCE"]))
    sources.append(
        _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=rtfv_output_fields or [StructField("WEIGHTED_BALANCE", DoubleType())],
            request_fields=request_fields or [StructField("WEIGHT", DoubleType())],
        )
    )
    fg = FeatureGroup(name=name, features=sources, auto_prefix=False)
    fg._version = FeatureGroupVersion(version)
    fg._postgres_online_query_url = query_url
    return fg


def _make_hydrated_no_rs_rtfv_fg(
    *,
    name: str = "FG",
    version: str = "v1",
    query_url: str = "https://q.example/svc",
) -> FeatureGroup:
    """Construct a registered-looking FG whose only RTFV source has no ``RequestSource``."""
    rtfv = _make_registered_rtfv(
        name="RT_FV",
        version="v1",
        output_fields=[
            StructField("USER_ID", StringType()),
            StructField("DOUBLED_BALANCE", DoubleType()),
        ],
        with_request_source=False,
    )
    fg = FeatureGroup(name=name, features=[rtfv], auto_prefix=False)
    fg._version = FeatureGroupVersion(version)
    fg._postgres_online_query_url = query_url
    return fg


def _make_hydrated_dual_rtfv_fg(
    *,
    name: str = "FG",
    version: str = "v1",
    query_url: str = "https://q.example/svc",
) -> FeatureGroup:
    """FG with two RTFVs: one declares a ``RequestSource``, one does not."""
    rtfv_with_rs = _make_registered_rtfv(
        name="RT_WITH",
        version="v1",
        output_fields=[StructField("WEIGHTED_BALANCE", DoubleType())],
        request_fields=[StructField("WEIGHT", DoubleType())],
    )
    rtfv_no_rs = _make_registered_rtfv(
        name="RT_NORS",
        version="v1",
        output_fields=[StructField("DOUBLED_BALANCE", DoubleType())],
        with_request_source=False,
    )
    fg = FeatureGroup(name=name, features=[rtfv_with_rs, rtfv_no_rs], auto_prefix=False)
    fg._version = FeatureGroupVersion(version)
    fg._postgres_online_query_url = query_url
    return fg


class ReadFeatureGroupHappyPathTest(absltest.TestCase):
    """``read_feature_group`` forwards the right Query API payload and returns pandas."""

    def test_forwards_object_type_name_and_version(self) -> None:
        fg = _make_hydrated_fg(name="FG", version="v1")
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_get_or_create_online_http_client", MagicMock(return_value=MagicMock()))

        captured: dict[str, object] = {}

        def _fake_read(**kwargs: object) -> tuple[list[dict[str, object]], object]:
            captured.update(kwargs)
            schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("F1", DoubleType()),
                ]
            )
            return [{"USER_ID": "u1", "F1": 10.5}], schema

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            df = fs.read_feature_group(fg, keys=[["u1"], ["u2"]])

        # Assert via captured kwargs so an arg-reorder doesn't silently break this test.
        self.assertEqual(captured["object_type"], "feature_group")
        self.assertEqual(captured["feature_view_name"], "FG")
        self.assertEqual(captured["feature_view_version"], "v1")
        self.assertEqual(captured["join_key_names"], ["USER_ID"])
        self.assertEqual(captured["keys"], [["u1"], ["u2"]])
        self.assertIsNone(captured["feature_names"])
        join_types = captured["join_key_field_types"]
        assert isinstance(join_types, dict)
        self.assertEqual(set(join_types), {"USER_ID"})
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["USER_ID", "F1"])
        self.assertEqual(df["F1"].tolist(), [10.5])

    def test_string_name_path_routes_through_get_feature_group(self) -> None:
        """``read_feature_group("FG", "v1", keys=...)`` resolves via get_feature_group."""
        fg = _make_hydrated_fg(name="FG", version="v1")
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_get_or_create_online_http_client", MagicMock(return_value=MagicMock()))
        get_fg_mock = MagicMock(return_value=fg)
        object.__setattr__(fs, "get_feature_group", get_fg_mock)

        with patch.object(
            online_service,
            "read_postgres_online_features",
            return_value=([], StructType([StructField("USER_ID", StringType())])),
        ):
            fs.read_feature_group("FG", "v1", keys=[["u1"]])

        get_fg_mock.assert_called_once_with("FG", "v1")

    def test_register_get_read_round_trip_through_get_feature_group(self) -> None:
        """Drive get -> read so a regression that drops ``_postgres_online_query_url`` would fail here."""
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        meta = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="hello",
            auto_prefix=False,
            sources=[FeatureGroupSourceRef(fv_name="USER_FV", fv_version="v1")],
        )
        md = MagicMock()
        md.get_feature_group_metadata = MagicMock(return_value=meta)
        fs = _new_fs_with_mocks(metadata_manager=md)
        object.__setattr__(fs, "get_feature_view", MagicMock(return_value=fv))
        object.__setattr__(fs, "_get_or_create_online_http_client", MagicMock(return_value=MagicMock()))

        stub_status = MagicMock()
        stub_status.status = "RUNNING"
        with patch.object(online_service, "fetch_online_service_status", return_value=stub_status), patch.object(
            online_service, "endpoint_url", return_value="https://q.example/svc"
        ):
            fg = fs.get_feature_group("FG", "v1")

        self.assertEqual(fg._postgres_online_query_url, "https://q.example/svc")

        captured: dict[str, object] = {}

        def _fake_read(**kwargs: object) -> tuple[list[dict[str, object]], object]:
            captured.update(kwargs)
            schema = StructType([StructField("USER_ID", StringType()), StructField("F1", DoubleType())])
            return [{"USER_ID": "u1", "F1": 1.0}], schema

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            df = fs.read_feature_group(fg, keys=[["u1"]])

        self.assertEqual(captured["query_url"], "https://q.example/svc")
        self.assertEqual(captured["feature_view_name"], "FG")
        self.assertIsInstance(df, pd.DataFrame)


class ReadFeatureGroupValidationTest(absltest.TestCase):
    """``read_feature_group`` rejects bad inputs before any HTTP call."""

    def test_string_name_without_version_rejected(self) -> None:
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "requires `version` when `feature_group` is a string"):
            fs.read_feature_group("FG", keys=[["u1"]])

    def test_version_disagreement_rejected(self) -> None:
        fg = _make_hydrated_fg(name="FG", version="v1")
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "disagrees with the passed FeatureGroup"):
            fs.read_feature_group(fg, "v2", keys=[["u1"]])

    def test_unregistered_draft_rejected(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        draft = FeatureGroup(name="FG", features=[fv])
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "requires a registered FeatureGroup"):
            fs.read_feature_group(draft, keys=[["u1"]])

    def test_empty_keys_rejected(self) -> None:
        fg = _make_hydrated_fg()
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "at least one row in `keys`"):
            fs.read_feature_group(fg, keys=[])

    def test_missing_query_url_rejected(self) -> None:
        fg = _make_hydrated_fg()
        fg._postgres_online_query_url = None  # simulate Online Service not RUNNING at hydrate time
        fs = _new_fs_with_mocks()
        with self.assertRaisesRegex(ValueError, "Online Service is RUNNING"):
            fs.read_feature_group(fg, keys=[["u1"]])

    def test_online_skips_warehouse_and_telemetry(self) -> None:
        """``read_feature_group(store_type=ONLINE)`` skips warehouse switch and telemetry."""
        self.assertTrue(fs_mod._predicate_read_feature_group_skip_wh_switch())
        self.assertTrue(fs_mod._predicate_read_feature_group_skip_telemetry())
        self.assertTrue(fs_mod._predicate_read_feature_group_skip_wh_switch(store_type=StoreType.ONLINE))
        self.assertTrue(fs_mod._predicate_read_feature_group_skip_telemetry(store_type=StoreType.ONLINE))

        # ``get_function_usage_statement_params`` is the first thing the telemetry
        # decorator touches; a non-call proves the skip predicate fired first.
        fg = _make_hydrated_fg(name="FG", version="v1")
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_get_or_create_online_http_client", MagicMock(return_value=MagicMock()))

        def _fake_read(**_kwargs: object) -> tuple[list[dict[str, object]], object]:
            schema = StructType([StructField("USER_ID", StringType()), StructField("F1", DoubleType())])
            return [{"USER_ID": "u1", "F1": 1.0}], schema

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read), patch(
            "snowflake.ml._internal.telemetry.get_function_usage_statement_params"
        ) as mock_telemetry_params:
            df = fs.read_feature_group(fg, keys=[["u1"]])

        mock_telemetry_params.assert_not_called()
        fs._session.use_warehouse.assert_not_called()  # type: ignore[attr-defined]
        self.assertIsInstance(df, pd.DataFrame)

    def test_offline_store_type_rejected(self) -> None:
        """``store_type=StoreType.OFFLINE`` raises ``NotImplementedError`` (with the typed FS exception as cause)."""
        fg = _make_hydrated_fg()
        fs = _new_fs_with_mocks()
        with self.assertRaises(NotImplementedError) as ctx:
            fs.read_feature_group(fg, keys=[["u1"]], store_type=StoreType.OFFLINE)
        self.assertIn("not yet supported", str(ctx.exception))
        cause = ctx.exception.__cause__
        self.assertIsInstance(cause, snowml_exceptions.SnowflakeMLException)
        self.assertEqual(cause.error_code, error_codes.INVALID_ARGUMENT)


class ReadFeatureGroupRequestContextTest(absltest.TestCase):
    """``request_context`` dispatch for FGs with one or more RTFV sources."""

    def _new_fs_for_mixed_fg(self) -> FeatureStore:
        fs = _new_fs_with_mocks()
        object.__setattr__(fs, "_get_or_create_online_http_client", MagicMock(return_value=MagicMock()))
        return fs

    @staticmethod
    def _stub_read(captured: dict[str, object]) -> Any:
        def _fake_read(**kwargs: object) -> tuple[list[dict[str, object]], object]:
            captured.update(kwargs)
            schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("BALANCE", DoubleType()),
                    StructField("WEIGHTED_BALANCE", DoubleType()),
                ]
            )
            return [{"USER_ID": "u1", "BALANCE": 1.0, "WEIGHTED_BALANCE": 0.5}], schema

        return _fake_read

    def test_rtfv_source_requires_request_context(self) -> None:
        fg = _make_hydrated_mixed_fg()
        fs = self._new_fs_for_mixed_fg()
        with self.assertRaisesRegex(ValueError, "`request_context` is required"):
            fs.read_feature_group(fg, keys=[["u1"]])

    def test_no_rtfv_source_rejects_request_context(self) -> None:
        """``request_context`` on an FG without an RTFV source is rejected before the HTTP call."""
        fg = _make_hydrated_fg()
        fs = self._new_fs_for_mixed_fg()
        with self.assertRaisesRegex(
            ValueError,
            "only supported when at least one RealtimeFeatureView source declares a RequestSource",
        ):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=pd.DataFrame({"WEIGHT": [1.0]}))

    def test_request_context_must_be_dataframe(self) -> None:
        fg = _make_hydrated_mixed_fg()
        fs = self._new_fs_for_mixed_fg()
        with self.assertRaisesRegex(ValueError, "must be a pandas.DataFrame"):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=[{"WEIGHT": 1.0}])

    def test_request_context_missing_required_column_rejected(self) -> None:
        fg = _make_hydrated_mixed_fg(request_fields=[StructField("WEIGHT", DoubleType())])
        fs = self._new_fs_for_mixed_fg()
        with self.assertRaisesRegex(ValueError, "missing required columns \\['WEIGHT'\\]"):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=pd.DataFrame({"OTHER": [1.0]}))

    def test_request_context_length_mismatch_rejected(self) -> None:
        fg = _make_hydrated_mixed_fg()
        fs = self._new_fs_for_mixed_fg()
        with self.assertRaisesRegex(ValueError, "but `keys` has 2 row"):
            fs.read_feature_group(fg, keys=[["u1"], ["u2"]], request_context=pd.DataFrame({"WEIGHT": [1.0]}))

    def test_request_context_extras_dropped_with_warning(self) -> None:
        """Extras are dropped with a ``UserWarning``; the forwarded payload omits them."""
        fg = _make_hydrated_mixed_fg()
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}
        with patch.object(online_service, "read_postgres_online_features", side_effect=self._stub_read(captured)):
            with self.assertWarns(UserWarning) as warn_ctx:
                fs.read_feature_group(
                    fg,
                    keys=[["u1"]],
                    request_context=pd.DataFrame({"WEIGHT": [1.5], "STRAY": ["x"]}),
                )
        self.assertIn("STRAY", str(warn_ctx.warning))
        payload = captured["request_context"]
        assert isinstance(payload, list)
        # Extras are dropped; only the canonical required column survives.
        self.assertEqual(payload, [{"WEIGHT": 1.5}])

    def test_case_insensitive_column_matching(self) -> None:
        """Caller-supplied lower-case column matches a canonical upper-case RequestSource field."""
        fg = _make_hydrated_mixed_fg(request_fields=[StructField("AMOUNT", DoubleType())])
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}
        with patch.object(online_service, "read_postgres_online_features", side_effect=self._stub_read(captured)):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=pd.DataFrame({"amount": [42.0]}))
        payload = captured["request_context"]
        assert isinstance(payload, list)
        # Canonical name (server-expected) — not the caller's casing.
        self.assertEqual(payload, [{"AMOUNT": 42.0}])

    def test_payload_forwarded_in_keys_order_as_list_of_dicts(self) -> None:
        """Row alignment is positional: ``request_context[i]`` rides with ``keys[i]``."""
        fg = _make_hydrated_mixed_fg(request_fields=[StructField("WEIGHT", DoubleType())])
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}
        with patch.object(online_service, "read_postgres_online_features", side_effect=self._stub_read(captured)):
            fs.read_feature_group(
                fg,
                keys=[["u1"], ["u2"], ["u3"]],
                request_context=pd.DataFrame({"WEIGHT": [1.0, 2.0, 3.0]}),
            )
        payload = captured["request_context"]
        self.assertEqual(payload, [{"WEIGHT": 1.0}, {"WEIGHT": 2.0}, {"WEIGHT": 3.0}])
        self.assertEqual(captured["object_type"], "feature_group")

    def test_rtfv_only_fg_join_key_types_use_upstream_fallback(self) -> None:
        """RTFV-only FG (no BFV anchor) emits join-key datatypes via ``resolve_realtime_join_key_fields``."""
        # No BFV in this FG -> the read path can only derive USER_ID's datatype
        # by walking the RTFV's upstream BFV (the resolve_realtime_join_key_fields
        # fallback path).
        fg = _make_hydrated_mixed_fg(include_bfv=False)
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}
        with patch.object(online_service, "read_postgres_online_features", side_effect=self._stub_read(captured)):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=pd.DataFrame({"WEIGHT": [1.0]}))
        join_types = captured["join_key_field_types"]
        assert isinstance(join_types, dict)
        self.assertEqual(set(join_types), {"USER_ID"})
        # The RTFV's upstream BFV (built by _make_registered_rtfv) keys on USER_ID
        # with StringType, so the fallback should resolve to that exact type.
        self.assertEqual(join_types["USER_ID"], StringType())

    def test_union_of_request_columns_across_multiple_rtfvs(self) -> None:
        """Two RTFVs each declare one RequestSource column -> required columns = union of both."""
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
            request_fields=[StructField("WEIGHT", IntegerType())],
        )
        fg = FeatureGroup(name="FG", features=[rtfv_a, rtfv_b], auto_prefix=False)
        fg._version = FeatureGroupVersion("v1")
        fg._postgres_online_query_url = "https://q.example/svc"
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}
        with patch.object(online_service, "read_postgres_online_features", side_effect=self._stub_read(captured)):
            fs.read_feature_group(
                fg,
                keys=[["u1"]],
                request_context=pd.DataFrame({"AMOUNT": [42.0], "WEIGHT": [7]}),
            )
        payload = captured["request_context"]
        self.assertEqual(payload, [{"AMOUNT": 42.0, "WEIGHT": 7}])

    def test_no_request_source_rtfv_omits_request_context(self) -> None:
        """FG whose only RTFV source has no ``RequestSource``: read with ``request_context=None`` succeeds."""
        fg = _make_hydrated_no_rs_rtfv_fg()
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}

        def _fake_read(**kwargs: object) -> tuple[list[dict[str, object]], object]:
            captured.update(kwargs)
            schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("DOUBLED_BALANCE", DoubleType()),
                ]
            )
            return [{"USER_ID": "u1", "DOUBLED_BALANCE": 2.0}], schema

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            fs.read_feature_group(fg, keys=[["u1"]])

        self.assertIsNone(captured["request_context"])
        self.assertEqual(captured["object_type"], "feature_group")

    def test_no_request_source_rtfv_rejects_request_context(self) -> None:
        """FG whose only RTFV source has no ``RequestSource``: any ``request_context`` is rejected."""
        fg = _make_hydrated_no_rs_rtfv_fg()
        fs = self._new_fs_for_mixed_fg()
        with self.assertRaisesRegex(
            ValueError,
            "only supported when at least one RealtimeFeatureView source declares a RequestSource",
        ):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=pd.DataFrame({"WEIGHT": [1.0]}))

    def test_dual_rtfv_with_and_without_request_source_uses_only_with_rs_columns(self) -> None:
        """Mixed RTFV-only FG: required payload columns come from the with-RS RTFV alone."""
        fg = _make_hydrated_dual_rtfv_fg()
        fs = self._new_fs_for_mixed_fg()

        captured: dict[str, object] = {}
        with patch.object(online_service, "read_postgres_online_features", side_effect=self._stub_read(captured)):
            fs.read_feature_group(fg, keys=[["u1"]], request_context=pd.DataFrame({"WEIGHT": [1.5]}))
        payload = captured["request_context"]
        self.assertEqual(payload, [{"WEIGHT": 1.5}])


if __name__ == "__main__":
    absltest.main()
