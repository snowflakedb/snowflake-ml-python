"""Unit tests for ``FeatureStore.read_feature_view`` on a RealtimeFeatureView.

Drives the ``is_realtime_feature_view`` branch of ``read_feature_view`` end-to-end against a
mocked ``online_service.read_postgres_online_features`` so we exercise the full
client-side validation order, the ``pd.DataFrame -> list[dict]`` conversion of
``request_context``, and the OFT-level schema synthesis used by the empty- and
non-empty-result paths.

All tests stay client-side; the Online Service Query API is mocked.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store as fs_mod, online_service
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.realtime_registration import (
    resolve_realtime_join_key_fields,
)
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Module-level compute_fn (avoids `inspect.getsource` issues on closures).
# ---------------------------------------------------------------------------


def _rtfv_compute_fn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    """Compute risk_score = amount / (avg_amount + 1)."""
    return pd.DataFrame(
        {
            "risk_score": req["amount"] / (txn["avg_amount"] + 1.0),
        }
    )


def _rtfv_compute_fn_no_req(txn: pd.DataFrame) -> pd.DataFrame:
    """Single-upstream variant for RTFVs without a RequestSource."""
    return pd.DataFrame(
        {
            "risk_score": txn["avg_amount"] + 1.0,
        }
    )


def _rtfv_compute_fn_two_upstreams(req: pd.DataFrame, txn_a: pd.DataFrame, txn_b: pd.DataFrame) -> pd.DataFrame:
    """Three-source variant used by the defense-in-depth tests."""
    return pd.DataFrame(
        {
            "risk_score": req["amount"] / (txn_a["avg_amount"] + txn_b["avg_amount"] + 1.0),
        }
    )


_RTFV_OUTPUT_SCHEMA = StructType(
    [
        StructField("risk_score", DoubleType()),
    ]
)


def _request_source() -> RequestSource:
    return RequestSource(
        schema=StructType([StructField("amount", DoubleType())]),
    )


def _build_upstream_fv(
    *,
    name: str = "TXN_FV",
    feature_columns: tuple[str, ...] = ("avg_amount",),
    entity_name: str = "USER",
    join_keys: tuple[str, ...] = ("USER_ID",),
    join_key_type: Any = None,
) -> FeatureView:
    """Construct a registered Postgres-backed BFV used as an RTFV upstream."""
    key_type = join_key_type if join_key_type is not None else StringType()
    schema = StructType(
        [StructField(k, key_type) for k in join_keys] + [StructField(c, DoubleType()) for c in feature_columns]
    )
    mock_df = MagicMock()
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": [f"SELECT * FROM {name}"]}

    fv = FeatureView(
        name=name,
        entities=[Entity(name=entity_name, join_keys=list(join_keys))],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
    )
    fv._version = FeatureViewVersion("v1")
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._infer_schema_df = mock_df
    fv._status = FeatureViewStatus.ACTIVE
    return fv


def _build_rtfv(
    *,
    name: str = "MY_RTFV",
    upstreams: Optional[list[FeatureView]] = None,
    declared_entities: Optional[list[Entity]] = None,
    query_url: str = "https://q.example/svc",
    output_schema: Optional[StructType] = None,
) -> FeatureView:
    """Construct a registered RTFV pre-hydrated with a Postgres query URL."""
    sources = [_request_source(), *(upstreams or [_build_upstream_fv()])]
    rt_cfg = RealtimeConfig(
        compute_fn=_rtfv_compute_fn,
        sources=sources,
        output_schema=output_schema or _RTFV_OUTPUT_SCHEMA,
    )
    entities = declared_entities or [Entity(name="USER", join_keys=["USER_ID"])]
    rtfv = FeatureView(name=name, entities=entities, realtime_config=rt_cfg)
    rtfv._version = FeatureViewVersion("v1")
    rtfv._database = SqlIdentifier("DB")
    rtfv._schema = SqlIdentifier("SCH")
    rtfv._status = FeatureViewStatus.ACTIVE
    rtfv._postgres_online_query_url = query_url
    return rtfv


def _build_rtfv_no_request_source(
    *,
    name: str = "MY_RTFV",
    upstreams: Optional[list[FeatureView]] = None,
    declared_entities: Optional[list[Entity]] = None,
    query_url: str = "https://q.example/svc",
) -> FeatureView:
    """Construct a registered RTFV with no RequestSource (single FV upstream only)."""
    rt_cfg = RealtimeConfig(
        compute_fn=_rtfv_compute_fn_no_req,
        sources=list(upstreams or [_build_upstream_fv()]),
        output_schema=_RTFV_OUTPUT_SCHEMA,
    )
    entities = declared_entities or [Entity(name="USER", join_keys=["USER_ID"])]
    rtfv = FeatureView(name=name, entities=entities, realtime_config=rt_cfg)
    rtfv._version = FeatureViewVersion("v1")
    rtfv._database = SqlIdentifier("DB")
    rtfv._schema = SqlIdentifier("SCH")
    rtfv._status = FeatureViewStatus.ACTIVE
    rtfv._postgres_online_query_url = query_url
    return rtfv


def _new_fs() -> FeatureStore:
    fs = object.__new__(FeatureStore)
    sess = MagicMock()
    # ``switch_warehouse`` calls SqlIdentifier(session.get_current_warehouse()); the
    # decorator runs on every read_feature_view call so the mock must return a
    # string (not a MagicMock attribute, which fails QUOTED_IDENTIFIER_RE.match).
    sess.get_current_warehouse.return_value = "WH"
    object.__setattr__(fs, "_session", sess)
    object.__setattr__(
        fs,
        "_config",
        fs_mod._FeatureStoreConfig(database=SqlIdentifier("DB"), schema=SqlIdentifier("SCH")),
    )
    object.__setattr__(fs, "_telemetry_stmp", {})
    object.__setattr__(fs, "_metadata_manager", MagicMock())
    object.__setattr__(fs, "_default_warehouse", SqlIdentifier("WH"))
    object.__setattr__(fs, "_get_or_create_online_http_client", MagicMock(return_value=MagicMock()))
    return fs


# ---------------------------------------------------------------------------
# Happy paths.
# ---------------------------------------------------------------------------


class ReadRealtimeFeatureViewHappyPathTest(absltest.TestCase):
    """The dispatcher converts the request_context DataFrame to per-row dicts."""

    def test_multi_row_request_context_forwarded_per_row(self) -> None:
        rtfv = _build_rtfv()
        fs = _new_fs()

        captured: dict[str, Any] = {}

        def _fake_read(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
            captured.update(kwargs)
            schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("risk_score", DoubleType()),
                ]
            )
            return (
                [
                    {"USER_ID": "u1", "risk_score": 0.5},
                    {"USER_ID": "u2", "risk_score": 1.0},
                ],
                schema,
            )

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            df = fs.read_feature_view(
                rtfv,
                keys=[["u1"], ["u2"]],
                request_context=pd.DataFrame({"amount": [10.0, 20.0]}),
            )

        self.assertIsInstance(df, pd.DataFrame)
        # request_context is forwarded as per-row dicts in the same order as keys,
        # with column names canonicalized to the Snowflake-resolved (uppercase) form
        # the server expects.
        self.assertEqual(
            captured["request_context"],
            [{"AMOUNT": 10.0}, {"AMOUNT": 20.0}],
        )
        # join_key_field_types is derived from the OFT-level schema (i.e. upstream PK).
        self.assertEqual(set(captured["join_key_field_types"].keys()), {"USER_ID"})
        self.assertIsInstance(captured["join_key_field_types"]["USER_ID"], StringType)
        # No feature_names is forwarded -- the server returns the full UDF output.
        self.assertIsNone(captured["feature_names"])

    def test_single_row_request_context(self) -> None:
        rtfv = _build_rtfv()
        fs = _new_fs()

        def _fake_read(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
            schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("risk_score", DoubleType()),
                ]
            )
            return [{"USER_ID": "u1", "risk_score": 3.14}], schema

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            df = fs.read_feature_view(
                rtfv,
                keys=[["u1"]],
                request_context=pd.DataFrame({"amount": [42.0]}),
            )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    def test_extra_columns_warn_and_drop_before_send(self) -> None:
        """Extra columns in request_context are dropped client-side with a UserWarning."""
        rtfv = _build_rtfv()
        fs = _new_fs()

        captured: dict[str, Any] = {}

        def _fake_read(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
            captured.update(kwargs)
            schema = StructType([StructField("USER_ID", StringType()), StructField("risk_score", DoubleType())])
            return [{"USER_ID": "u1", "risk_score": 1.0}], schema

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            with self.assertWarns(UserWarning) as cm:
                fs.read_feature_view(
                    rtfv,
                    keys=[["u1"]],
                    request_context=pd.DataFrame({"amount": [10.0], "extra_col": ["ignored"]}),
                )
        self.assertIn("extra_col", str(cm.warning))
        self.assertEqual(captured["request_context"], [{"AMOUNT": 10.0}])

    def test_no_request_source_read_omits_request_context(self) -> None:
        """RTFVs without a RequestSource read with request_context=None and forward None to the server."""
        rtfv = _build_rtfv_no_request_source()
        fs = _new_fs()

        captured: dict[str, Any] = {}

        def _fake_read(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
            captured.update(kwargs)
            schema = StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("risk_score", DoubleType()),
                ]
            )
            return ([{"USER_ID": "u1", "risk_score": 2.0}], schema)

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            df = fs.read_feature_view(rtfv, keys=[["u1"]])

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsNone(captured["request_context"])
        self.assertIsNone(captured["feature_names"])


# ---------------------------------------------------------------------------
# Validation-order rejections.
# ---------------------------------------------------------------------------


class ReadRealtimeFeatureViewValidationTest(absltest.TestCase):
    """The RTFV branch runs validation in a documented order; each step has its own assertion.

    NB: ``send_api_usage_telemetry`` unwraps :class:`SnowflakeMLException` and
    re-raises ``original_exception`` (here ``ValueError``), so tests assert on
    ``ValueError`` rather than the wrapped exception type.
    """

    def _expect_value_error(self, *, regex: str, **kwargs: Any) -> str:
        rtfv = kwargs.pop("rtfv", None) or _build_rtfv()
        fs = kwargs.pop("fs", None) or _new_fs()
        with patch.object(online_service, "read_postgres_online_features") as patched:
            with self.assertRaisesRegex(ValueError, regex) as ctx:
                fs.read_feature_view(rtfv, **kwargs)
            patched.assert_not_called()
        return str(ctx.exception)

    def test_non_postgres_online_backing_rejected(self) -> None:
        """An RTFV whose online_config is not POSTGRES is rejected defensively.

        RealtimeFeatureView constructors set ``online_config`` to POSTGRES
        automatically; this test simulates registry corruption by mutating
        the field post-construction to confirm the read-side guard fires.
        """
        rtfv = _build_rtfv()
        rtfv._online_config = OnlineConfig(enable=True, store_type=OnlineStoreType.HYBRID_TABLE)

        msg = self._expect_value_error(
            regex="POSTGRES",
            rtfv=rtfv,
            keys=[["u1"]],
            request_context=pd.DataFrame({"amount": [10.0]}),
        )
        self.assertIn("MY_RTFV", msg)

    def test_keys_none_raises_typed_invalid_argument(self) -> None:
        """``keys=None`` must raise INVALID_ARGUMENT, not bare TypeError on len(None)."""
        msg = self._expect_value_error(
            regex="non-empty",
            keys=None,
            request_context=pd.DataFrame({"amount": [10.0]}),
        )
        self.assertIn("`keys`", msg)
        self.assertIn("MY_RTFV", msg)

    def test_keys_empty_raises_typed_invalid_argument(self) -> None:
        msg = self._expect_value_error(
            regex="non-empty",
            keys=[],
            request_context=pd.DataFrame({"amount": []}),
        )
        self.assertIn("`keys`", msg)

    def test_request_context_missing_rejected(self) -> None:
        msg = self._expect_value_error(
            regex=r"`request_context` is required",
            keys=[["u1"]],
            request_context=None,
        )
        self.assertIn("MY_RTFV", msg)

    def test_request_context_wrong_type_rejected(self) -> None:
        msg = self._expect_value_error(
            regex=r"`request_context` must be a pandas\.DataFrame",
            keys=[["u1"]],
            request_context=[{"amount": 10.0}],
        )
        self.assertIn("list", msg)

    def test_request_context_missing_required_column_rejected(self) -> None:
        msg = self._expect_value_error(
            regex="missing required columns",
            keys=[["u1"]],
            request_context=pd.DataFrame({"other": [10.0]}),
        )
        # The canonical (Snowflake-resolved) form of "amount" is "AMOUNT"; we use
        # the canonical name in the error so the user can copy it verbatim.
        self.assertIn("AMOUNT", msg)

    def test_request_context_length_mismatch_rejected(self) -> None:
        msg = self._expect_value_error(
            regex="1 row.+2 row|2 row.+1 row",
            keys=[["u1"], ["u2"]],
            request_context=pd.DataFrame({"amount": [10.0]}),
        )
        self.assertIn("MY_RTFV", msg)

    def test_feature_names_rejected(self) -> None:
        self._expect_value_error(
            regex="feature_names filtering is not supported",
            keys=[["u1"]],
            request_context=pd.DataFrame({"amount": [10.0]}),
            feature_names=["risk_score"],
        )

    def test_request_context_rejected_when_no_request_source(self) -> None:
        """RTFVs without a RequestSource reject any provided request_context."""
        rtfv = _build_rtfv_no_request_source()
        msg = self._expect_value_error(
            regex="without a RequestSource",
            rtfv=rtfv,
            keys=[["u1"]],
            request_context=pd.DataFrame({"amount": [10.0]}),
        )
        self.assertIn("MY_RTFV", msg)


# ---------------------------------------------------------------------------
# Non-RTFV reads reject request_context.
# ---------------------------------------------------------------------------


class NonRealtimeRequestContextRejectionTest(absltest.TestCase):
    def test_request_context_on_bfv_rejected(self) -> None:
        bfv = _build_upstream_fv(name="BFV")
        fs = _new_fs()
        with self.assertRaisesRegex(ValueError, "RealtimeFeatureView"):
            fs.read_feature_view(
                bfv,
                keys=[["u1"]],
                request_context=pd.DataFrame({"amount": [10.0]}),
                store_type=StoreType.ONLINE,
            )


# ---------------------------------------------------------------------------
# Empty-result schema synthesis.
# ---------------------------------------------------------------------------


class EmptyResultSchemaTest(absltest.TestCase):
    """An empty Online Service response must still surface the join-key columns."""

    def test_empty_result_returns_dataframe_with_join_keys_and_features(self) -> None:
        rtfv = _build_rtfv()
        fs = _new_fs()

        def _fake_read(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
            return (
                [],
                StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("risk_score", DoubleType()),
                    ]
                ),
            )

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            df = fs.read_feature_view(
                rtfv,
                keys=[["u1"]],
                request_context=pd.DataFrame({"amount": [10.0]}),
            )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    def test_oft_full_schema_dedupes_join_key_redeclared_in_output_schema(self) -> None:
        """An RTFV that re-declares the join key in ``output_schema`` must not produce duplicate fields.

        The upstream-derived PK field carries the authoritative datatype; the
        duplicate from ``output_schema`` is dropped so empty-result frames are
        well-formed for both pandas and Snowpark consumers.
        """
        rtfv = _build_rtfv(
            output_schema=StructType(
                [
                    StructField("USER_ID", StringType()),
                    StructField("risk_score", DoubleType()),
                ]
            ),
        )
        fs = _new_fs()

        schema = fs._oft_full_schema(rtfv)
        canonical = [SqlIdentifier(f.name).resolved() for f in schema.fields]
        self.assertEqual(canonical.count("USER_ID"), 1)
        self.assertEqual(canonical, ["USER_ID", SqlIdentifier("risk_score").resolved()])
        # The empty-result path consumes the same schema; assert the struct it
        # would hand to ``create_dataframe`` is also duplicate-free.
        struct = fs._postgres_online_read_struct_type(rtfv, feature_names=None)
        self.assertEqual(
            [SqlIdentifier(f.name).resolved() for f in struct.fields],
            canonical,
        )


# ---------------------------------------------------------------------------
# OFT-level schema helper + join_key_field_types propagation.
# ---------------------------------------------------------------------------


class JoinKeyTypePropagationTest(parameterized.TestCase):
    """The OFT-level schema picks up the upstream's declared join-key datatype."""

    @parameterized.parameters(  # type: ignore[misc]
        (IntegerType, [{"USER_ID": 1}, {"USER_ID": 2}]),
        (StringType, [{"USER_ID": "u1"}, {"USER_ID": "u2"}]),
    )
    def test_join_key_type_propagated_to_query_api(
        self, key_type_cls: type, expected_entities: list[dict[str, Any]]
    ) -> None:
        upstream = _build_upstream_fv(name="TXN", join_key_type=key_type_cls())
        rtfv = _build_rtfv(upstreams=[upstream])
        fs = _new_fs()

        captured: dict[str, Any] = {}

        def _fake_read(**kwargs: Any) -> tuple[list[dict[str, Any]], Any]:
            captured.update(kwargs)
            return [], StructType([StructField("USER_ID", key_type_cls()), StructField("risk_score", DoubleType())])

        with patch.object(online_service, "read_postgres_online_features", side_effect=_fake_read):
            fs.read_feature_view(
                rtfv,
                keys=[[expected_entities[0]["USER_ID"]], [expected_entities[1]["USER_ID"]]],
                request_context=pd.DataFrame({"amount": [10.0, 20.0]}),
            )

        self.assertIsInstance(captured["join_key_field_types"]["USER_ID"], key_type_cls)


# ---------------------------------------------------------------------------
# Defense-in-depth on the read helper.
# ---------------------------------------------------------------------------


class ResolveRealtimeJoinKeyFieldsDefenseTest(absltest.TestCase):
    """Direct invocation of the helper rejects hand-crafted RTFVs whose sources disagree."""

    def test_conflicting_upstream_datatypes_raise_internal_python_error(self) -> None:
        u_int = _build_upstream_fv(name="A", join_key_type=IntegerType())
        u_str = _build_upstream_fv(name="B", join_key_type=StringType())

        # Bypass register-time validation by constructing the RTFV after-the-fact.
        rtfv = FeatureView(
            name="RTFV",
            entities=[Entity(name="USER", join_keys=["USER_ID"])],
            realtime_config=RealtimeConfig(
                compute_fn=_rtfv_compute_fn_two_upstreams,
                sources=[_request_source(), u_int, u_str],
                output_schema=_RTFV_OUTPUT_SCHEMA,
            ),
        )

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            resolve_realtime_join_key_fields(rtfv)
        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_PYTHON_ERROR)
        self.assertIn("Registry corruption", str(ctx.exception.original_exception))
        self.assertIn("A@v1", str(ctx.exception.original_exception))
        self.assertIn("B@v1", str(ctx.exception.original_exception))

    def test_missing_join_key_raises_internal_python_error(self) -> None:
        # Upstream declares only AVG_AMOUNT (no USER_ID-bearing fields in output_schema).
        upstream = _build_upstream_fv(name="NOKEY", feature_columns=("avg_amount",))
        # Override the upstream's output_schema to drop the join-key column entirely.
        assert upstream._infer_schema_df is not None
        upstream._infer_schema_df.schema = StructType([StructField("avg_amount", DoubleType())])

        rtfv = FeatureView(
            name="RTFV",
            entities=[Entity(name="USER", join_keys=["USER_ID"])],
            realtime_config=RealtimeConfig(
                compute_fn=_rtfv_compute_fn,
                sources=[_request_source(), upstream],
                output_schema=_RTFV_OUTPUT_SCHEMA,
            ),
        )

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            resolve_realtime_join_key_fields(rtfv)
        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_PYTHON_ERROR)
        self.assertIn("USER_ID", str(ctx.exception.original_exception))


# ---------------------------------------------------------------------------
# Dispatch-predicate skip behavior.
# ---------------------------------------------------------------------------


class ReadFeatureViewSkipPredicateTest(absltest.TestCase):
    """Warehouse switch + telemetry must skip for RTFV reads via either input form."""

    def test_request_context_signals_rtfv_for_string_form(self) -> None:
        """``read_feature_view("name", "v1", request_context=df)`` is detected without metadata lookup."""
        self.assertTrue(
            fs_mod._read_feature_view_is_realtime(
                None, "MY_RTFV", version="v1", request_context=pd.DataFrame({"amount": [1.0]})
            )
        )
        self.assertTrue(
            fs_mod._predicate_read_feature_view_skip_wh_switch(
                None, "MY_RTFV", version="v1", request_context=pd.DataFrame({"amount": [1.0]})
            )
        )
        self.assertTrue(
            fs_mod._predicate_read_feature_view_skip_telemetry(
                None, "MY_RTFV", version="v1", request_context=pd.DataFrame({"amount": [1.0]})
            )
        )

    def test_rtfv_object_skips_regardless_of_store_type(self) -> None:
        rtfv = _build_rtfv()
        self.assertTrue(fs_mod._predicate_read_feature_view_skip_wh_switch(None, rtfv, store_type=StoreType.OFFLINE))
        self.assertTrue(fs_mod._predicate_read_feature_view_skip_telemetry(None, rtfv, store_type=StoreType.OFFLINE))

    def test_non_rtfv_string_offline_read_does_not_skip(self) -> None:
        """A plain ``read_feature_view("bfv", "v1")`` must still pay warehouse + telemetry."""
        self.assertFalse(fs_mod._predicate_read_feature_view_skip_wh_switch(None, "BFV", version="v1"))
        self.assertFalse(fs_mod._predicate_read_feature_view_skip_telemetry(None, "BFV", version="v1"))


if __name__ == "__main__":
    absltest.main()
