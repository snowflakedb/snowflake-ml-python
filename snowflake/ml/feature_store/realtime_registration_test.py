"""Unit tests for :mod:`realtime_registration`."""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.realtime_registration import (
    _build_realtime_feature_view_spec,
    _resolve_realtime_unwrapped_upstream_fvs,
    _resolve_realtime_upstream_fvs,
    build_rtfv_source_refs,
    request_schema_from_json,
    request_schema_to_json,
    validate_rtfv_entity_contract,
    validate_sources_online_postgres_for_rtfv,
)
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.spec.enums import FeatureViewKind, SourceType
from snowflake.snowpark.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def _build_request_source() -> RequestSource:
    return RequestSource(schema=StructType([StructField("amount", DoubleType())]))


def _build_upstream_fv(
    name: str = "TXN_FV",
    *,
    entity_name: str = "USER",
    join_keys: tuple[str, ...] = ("USER_ID",),
    feature_columns: tuple[str, ...] = ("avg_amount",),
    join_key_type: Any = None,
) -> FeatureView:
    key_type = join_key_type if join_key_type is not None else StringType()
    fields = [StructField(k, key_type) for k in join_keys] + [StructField(c, DoubleType()) for c in feature_columns]
    schema = StructType(fields)
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
    fv._infer_schema_df = mock_df
    fv._status = FeatureViewStatus.ACTIVE
    return fv


def rtfv_compute_fn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_score": req["amount"],
            "risk_bucket": ["x"] * len(req),
        }
    )


_RTFV_OUTPUT_SCHEMA = StructType(
    [
        StructField("risk_score", DoubleType()),
        StructField("risk_bucket", StringType()),
    ]
)


def _make_rtfv() -> FeatureView:
    rtc = RealtimeConfig(
        compute_fn=rtfv_compute_fn,
        sources=[_build_request_source(), _build_upstream_fv("TXN_FV")],
        output_schema=_RTFV_OUTPUT_SCHEMA,
    )
    return FeatureView(
        name="MY_RTFV",
        entities=[Entity(name="USER", join_keys=["USER_ID"])],
        realtime_config=rtc,
    )


class BuildRealtimeFeatureViewSpecTest(absltest.TestCase):
    """Cross-validation that the spec round-trips RealtimeConfig contents faithfully."""

    def test_happy_path_spec_shape(self) -> None:
        rtfv = _make_rtfv()
        spec = _build_realtime_feature_view_spec(
            feature_view=rtfv,
            feature_view_name=SqlIdentifier("MY_RTFV$v1", case_sensitive=True),
            version="v1",
            target_lag="0 seconds",
            database="DB",
            schema="SCH",
        )
        # Kind + identity
        self.assertEqual(spec.kind, FeatureViewKind.RealtimeFeatureView)
        self.assertEqual(spec.metadata.name, "MY_RTFV")
        self.assertEqual(spec.metadata.version, "v1")
        self.assertEqual(spec.metadata.database, "DB")
        self.assertEqual(spec.metadata.schema_, "SCH")

        # No offline configs (RTFV is request-time-evaluated)
        self.assertEqual(spec.offline_configs, [])

        # Sources: RequestSource at position 0, then FEATURES upstream.
        self.assertEqual(len(spec.spec.sources), 2)
        self.assertEqual(spec.spec.sources[0].source_type, SourceType.REQUEST)
        self.assertEqual(spec.spec.sources[1].source_type, SourceType.FEATURES)
        self.assertEqual(spec.spec.sources[1].name, "TXN_FV")
        self.assertEqual(spec.spec.sources[1].source_version, "v1")

        udf = spec.spec.udf
        assert udf is not None
        self.assertEqual(udf.name, "rtfv_compute_fn")
        self.assertEqual(udf.engine, "pandas")
        output_col_names = [c.name for c in udf.output_columns]
        # Names are upper-cased by the spec builder to match Snowflake's
        # default identifier resolution.
        self.assertEqual(output_col_names, ["RISK_SCORE", "RISK_BUCKET"])

        # Features: derived from UDF output_columns minus entity columns.
        feature_names = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(sorted(feature_names), ["RISK_BUCKET", "RISK_SCORE"])

        # Entity columns derived from upstream FVs.
        self.assertEqual(spec.spec.ordered_entity_column_names, ["USER_ID"])

        # No timestamp / no granularity / no agg method.
        self.assertIsNone(spec.spec.timestamp_field)
        self.assertIsNone(spec.spec.feature_granularity_sec)
        self.assertIsNone(spec.spec.feature_aggregation_method)

    def test_rejects_non_realtime_fv(self) -> None:
        schema = StructType([StructField("USER_ID", StringType()), StructField("F", DoubleType())])
        mock_df = MagicMock()
        mock_df.columns = [f.name for f in schema.fields]
        mock_df.schema = schema
        mock_df.queries = {"queries": ["SELECT * FROM X"]}
        fv = FeatureView(
            name="X",
            entities=[Entity(name="U", join_keys=["USER_ID"])],
            feature_df=mock_df,
        )
        fv._version = FeatureViewVersion("v1")
        fv._infer_schema_df = mock_df
        fv._status = FeatureViewStatus.ACTIVE
        with self.assertRaisesRegex(ValueError, "no realtime_config"):
            _build_realtime_feature_view_spec(
                feature_view=fv,
                feature_view_name=SqlIdentifier("X$v1", case_sensitive=True),
                version="v1",
                target_lag="0 seconds",
                database="DB",
                schema="SCH",
            )


class UpstreamFvHelpersTest(absltest.TestCase):
    def test_resolve_realtime_upstream_fvs(self) -> None:
        rtfv = _make_rtfv()
        upstreams = _resolve_realtime_upstream_fvs(rtfv)
        self.assertEqual(len(upstreams), 1)
        first_upstream = upstreams[0]
        assert isinstance(first_upstream, FeatureView)
        self.assertEqual(first_upstream.name.identifier(), "TXN_FV")

    def test_resolve_realtime_unwrapped_upstream_fvs(self) -> None:
        rtfv = _make_rtfv()
        bare = _resolve_realtime_unwrapped_upstream_fvs(rtfv)
        self.assertEqual(len(bare), 1)
        self.assertIsInstance(bare[0], FeatureView)
        self.assertEqual(bare[0].name.identifier(), "TXN_FV")

    def test_resolve_realtime_helpers_return_empty_for_non_realtime(self) -> None:
        schema = StructType([StructField("F", DoubleType())])
        mock_df = MagicMock()
        mock_df.columns = ["F"]
        mock_df.schema = schema
        mock_df.queries = {"queries": ["SELECT * FROM X"]}
        fv = FeatureView(
            name="X",
            entities=[Entity(name="U", join_keys=["F"])],
            feature_df=mock_df,
        )
        self.assertEqual(_resolve_realtime_upstream_fvs(fv), [])
        self.assertEqual(_resolve_realtime_unwrapped_upstream_fvs(fv), [])


def _cf_one_upstream(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0]})


def _cf_two_upstreams(req: pd.DataFrame, txn_a: pd.DataFrame, txn_b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0]})


def _cf_one_upstream_no_req(txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0] * len(txn)})


def _make_realtime_config_with_sources(
    upstream_fvs: list[FeatureView],
    *,
    include_request_source: bool = True,
) -> RealtimeConfig:
    """Build a minimal RealtimeConfig with arity matching the upstream count.

    Restricted to one or two upstream FVs — the cases the lifecycle helpers
    care about. Adding more arities means adding another module-level
    ``compute_fn`` because :class:`RealtimeConfig` requires source-extractable
    functions and rejects dynamically generated ones.

    Args:
        upstream_fvs: Upstream FeatureView sources.
        include_request_source: Whether to include a default RequestSource at
            position 0. Set to False to exercise the no-RequestSource path.

    Returns:
        A constructed :class:`RealtimeConfig`.

    Raises:
        AssertionError: If the upstream count is not supported by the fixture.
    """
    cf: Callable[..., Any]
    if include_request_source:
        sources: list[Any] = [_build_request_source(), *upstream_fvs]
        if len(upstream_fvs) == 1:
            cf = _cf_one_upstream
        elif len(upstream_fvs) == 2:
            cf = _cf_two_upstreams
        else:
            raise AssertionError(f"unsupported upstream count {len(upstream_fvs)} in test fixture")
    else:
        sources = list(upstream_fvs)
        if len(upstream_fvs) == 1:
            cf = _cf_one_upstream_no_req
        else:
            raise AssertionError(f"unsupported upstream count {len(upstream_fvs)} for include_request_source=False")
    output_schema = StructType([StructField("risk_score", DoubleType())])
    return RealtimeConfig(compute_fn=cf, sources=sources, output_schema=output_schema)


class ValidateSourcesOnlinePostgresTest(absltest.TestCase):
    def test_rejects_offline_upstream(self) -> None:
        upstream = _build_upstream_fv("TXN_FV")
        upstream._online_config = OnlineConfig(enable=False)
        rtc = _make_realtime_config_with_sources([upstream])
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_sources_online_postgres_for_rtfv(rtc)
        self.assertIn("online=False", str(cm.exception))

    def test_rejects_non_postgres_upstream(self) -> None:
        upstream = _build_upstream_fv("TXN_FV")
        upstream._online_config = OnlineConfig(enable=True, store_type=OnlineStoreType.HYBRID_TABLE)
        rtc = _make_realtime_config_with_sources([upstream])
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_sources_online_postgres_for_rtfv(rtc)
        self.assertIn("store_type=", str(cm.exception))

    def test_accepts_postgres_online_upstream(self) -> None:
        upstream = _build_upstream_fv("TXN_FV")
        rtc = _make_realtime_config_with_sources([upstream])
        validate_sources_online_postgres_for_rtfv(rtc)


class EntityContractTest(absltest.TestCase):
    """Each upstream FV's keys must be a subset of the RTFV's declared entity keys."""

    def test_exact_match_passes(self) -> None:
        rtc = _make_realtime_config_with_sources([_build_upstream_fv("A")])
        validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)

    def test_upstream_subset_of_declared_passes(self) -> None:
        """RTFV declares (USER_ID, COUNTRY); upstream keyed by USER_ID only -> ok."""
        upstream = _build_upstream_fv("A", entity_name="USER", join_keys=("USER_ID",))
        rtc = _make_realtime_config_with_sources([upstream])
        validate_rtfv_entity_contract(
            [
                Entity(name="USER", join_keys=["USER_ID"]),
                Entity(name="COUNTRY", join_keys=["COUNTRY"]),
            ],
            rtc,
        )

    def test_disjoint_upstreams_pass_under_declared_superset(self) -> None:
        """One upstream keyed by USER_ID, another by COUNTRY; declared covers both."""
        u_user = _build_upstream_fv("U", entity_name="USER", join_keys=("USER_ID",))
        u_country = _build_upstream_fv("C", entity_name="COUNTRY", join_keys=("COUNTRY",))
        rtc = _make_realtime_config_with_sources([u_user, u_country])
        validate_rtfv_entity_contract(
            [
                Entity(name="USER", join_keys=["USER_ID"]),
                Entity(name="COUNTRY", join_keys=["COUNTRY"]),
            ],
            rtc,
        )

    def test_upstream_with_key_not_in_declared_rejected(self) -> None:
        """Upstream keyed by SESSION_ID; declared has USER_ID only -> rejected."""
        upstream = _build_upstream_fv("S", entity_name="SESSION", join_keys=("SESSION_ID",))
        rtc = _make_realtime_config_with_sources([upstream])
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)
        msg = str(cm.exception)
        self.assertIn("subset", msg)
        self.assertIn("SESSION_ID", msg)
        self.assertIn("S@v1", msg)

    def test_one_upstream_subset_one_upstream_violates(self) -> None:
        """Mixed: one upstream is a valid subset, another contributes an undeclared key."""
        ok_upstream = _build_upstream_fv("OK", entity_name="USER", join_keys=("USER_ID",))
        bad_upstream = _build_upstream_fv("BAD", entity_name="OTHER", join_keys=("OTHER_ID",))
        rtc = _make_realtime_config_with_sources([ok_upstream, bad_upstream])
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_rtfv_entity_contract(
                [
                    Entity(name="USER", join_keys=["USER_ID"]),
                    Entity(name="COUNTRY", join_keys=["COUNTRY"]),
                ],
                rtc,
            )
        msg = str(cm.exception)
        self.assertIn("BAD@v1", msg)
        self.assertIn("OTHER_ID", msg)
        self.assertNotIn("OK@v1", msg)

    def test_shared_join_key_same_datatype_passes(self) -> None:
        """Two upstreams sharing USER_ID with the same datatype -> ok."""
        u1 = _build_upstream_fv("A", entity_name="USER", join_keys=("USER_ID",), join_key_type=IntegerType())
        u2 = _build_upstream_fv("B", entity_name="USER", join_keys=("USER_ID",), join_key_type=IntegerType())
        rtc = _make_realtime_config_with_sources([u1, u2])
        validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)

    def test_shared_join_key_differing_varchar_lengths_pass(self) -> None:
        """USER_ID as StringType() in one upstream and StringType(14) in another -> ok.

        VARCHAR length is irrelevant to join semantics and can legitimately
        differ across sources (e.g. a materialized dynamic table narrows the
        length to its stored data), so it must not be treated as a conflict.
        """
        u_unbounded = _build_upstream_fv("A", entity_name="USER", join_keys=("USER_ID",), join_key_type=StringType())
        u_bounded = _build_upstream_fv("B", entity_name="USER", join_keys=("USER_ID",), join_key_type=StringType(14))
        rtc = _make_realtime_config_with_sources([u_unbounded, u_bounded])
        validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)

    def test_shared_join_key_conflicting_datatypes_rejected(self) -> None:
        """Two upstreams declare USER_ID with different Snowpark datatypes -> rejected.

        Names both conflicting sources and both datatypes so the user can
        repair the source FV definitions without re-running the schema.
        """
        u_int = _build_upstream_fv("A", entity_name="USER", join_keys=("USER_ID",), join_key_type=IntegerType())
        u_str = _build_upstream_fv("B", entity_name="USER", join_keys=("USER_ID",), join_key_type=StringType())
        rtc = _make_realtime_config_with_sources([u_int, u_str])
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)
        msg = str(cm.exception)
        self.assertIn("USER_ID", msg)
        self.assertIn("A@v1", msg)
        self.assertIn("B@v1", msg)
        self.assertIn("IntegerType", msg)
        self.assertIn("StringType", msg)

    def test_disjoint_upstream_keys_are_not_compared_for_type(self) -> None:
        """If two upstreams don't share a join key, there is nothing to type-check."""
        u_user = _build_upstream_fv(
            "U",
            entity_name="USER",
            join_keys=("USER_ID",),
            join_key_type=IntegerType(),
        )
        u_country = _build_upstream_fv(
            "C",
            entity_name="COUNTRY",
            join_keys=("COUNTRY",),
            join_key_type=StringType(),
        )
        rtc = _make_realtime_config_with_sources([u_user, u_country])
        validate_rtfv_entity_contract(
            [
                Entity(name="USER", join_keys=["USER_ID"]),
                Entity(name="COUNTRY", join_keys=["COUNTRY"]),
            ],
            rtc,
        )

    def test_request_source_overlapping_entity_key_rejected(self) -> None:
        """RequestSource declaring an entity join key is rejected at register time."""
        upstream = _build_upstream_fv("A", entity_name="USER", join_keys=("USER_ID",))
        overlapping_request_source = RequestSource(
            schema=StructType([StructField("USER_ID", StringType()), StructField("WEIGHT", DoubleType())])
        )
        rtc = RealtimeConfig(
            compute_fn=_cf_one_upstream,
            sources=[overlapping_request_source, upstream],
            output_schema=StructType([StructField("risk_score", DoubleType())]),
        )
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)
        msg = str(cm.exception)
        self.assertIn("USER_ID", msg)
        self.assertIn("RequestSource.schema", msg)
        self.assertIn("entity join keys", msg)

    def test_request_source_overlapping_entity_key_case_insensitive_rejected(self) -> None:
        """Snowpark normalizes unquoted ``user_id`` to ``USER_ID``; overlap is still detected."""
        upstream = _build_upstream_fv("A", entity_name="USER", join_keys=("USER_ID",))
        lower_overlap = RequestSource(
            schema=StructType([StructField("user_id", StringType()), StructField("amount", DoubleType())])
        )
        rtc = RealtimeConfig(
            compute_fn=_cf_one_upstream,
            sources=[lower_overlap, upstream],
            output_schema=StructType([StructField("risk_score", DoubleType())]),
        )
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)
        msg = str(cm.exception)
        # Snowpark canonicalized "user_id" to "USER_ID" before we read the field.
        self.assertIn("USER_ID", msg)
        self.assertIn("RequestSource.schema", msg)
        self.assertIn("entity join keys", msg)

    def test_request_source_disjoint_from_entity_keys_passes(self) -> None:
        """The default ``RequestSource(amount: DOUBLE)`` is disjoint from ``USER_ID`` -> ok."""
        rtc = _make_realtime_config_with_sources([_build_upstream_fv("A")])
        validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)

    def test_no_request_source_skips_overlap_validator(self) -> None:
        """RTFVs without a RequestSource still pass entity-contract validation."""
        rtc = _make_realtime_config_with_sources(
            [_build_upstream_fv("A")],
            include_request_source=False,
        )
        validate_rtfv_entity_contract([Entity(name="USER", join_keys=["USER_ID"])], rtc)


class BuildRtfvSourceRefsTest(absltest.TestCase):
    def test_unsliced_unaliased_source(self) -> None:
        rtc = _make_realtime_config_with_sources([_build_upstream_fv("TXN_FV")])
        refs = build_rtfv_source_refs(rtc)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].fv_name, "TXN_FV")
        self.assertEqual(refs[0].fv_version, "v1")
        self.assertIsNone(refs[0].slice_columns)
        self.assertIsNone(refs[0].alias)


class RealtimeConfigMetadataRoundTripTest(absltest.TestCase):
    def test_entity_names_round_trip(self) -> None:
        """entity_names survives to_dict/from_dict and defaults empty for legacy rows."""
        from snowflake.ml.feature_store.metadata_manager import (
            FvSourceRef as FvSourceRefImported,
            RealtimeConfigMetadata,
        )

        original = RealtimeConfigMetadata(
            name="MY_RTFV",
            version="v1",
            desc="hi",
            compute_fn_name="fn",
            compute_fn_source="def fn(): pass\n",
            sources=[FvSourceRefImported(fv_name="TXN_FV", fv_version="v1")],
            request_schema_json="[]",
            output_schema_json="[]",
            output_columns=["risk_score"],
            entity_names=["USER", "MERCHANT"],
        )
        restored = RealtimeConfigMetadata.from_dict(original.to_dict())
        self.assertEqual(restored.entity_names, ["USER", "MERCHANT"])

        legacy_payload = original.to_dict()
        del legacy_payload["entity_names"]
        legacy_restored = RealtimeConfigMetadata.from_dict(legacy_payload)
        self.assertEqual(legacy_restored.entity_names, [])

    def test_request_schema_json_optional_round_trip(self) -> None:
        """RTFVs registered without a RequestSource persist request_schema_json=None."""
        from snowflake.ml.feature_store.metadata_manager import (
            FvSourceRef as FvSourceRefImported,
            RealtimeConfigMetadata,
        )

        original = RealtimeConfigMetadata(
            name="MY_RTFV",
            version="v1",
            desc="",
            compute_fn_name="fn",
            compute_fn_source="def fn(): pass\n",
            sources=[FvSourceRefImported(fv_name="TXN_FV", fv_version="v1")],
            request_schema_json=None,
            output_schema_json="[]",
            output_columns=["risk_score"],
        )
        payload = original.to_dict()
        self.assertIsNone(payload["request_schema_json"])
        restored = RealtimeConfigMetadata.from_dict(payload)
        self.assertIsNone(restored.request_schema_json)

        # Forward-compat: payloads written by older code without the key
        # decode the same way (request_schema_json absent -> None).
        absent_payload = dict(payload)
        del absent_payload["request_schema_json"]
        absent_restored = RealtimeConfigMetadata.from_dict(absent_payload)
        self.assertIsNone(absent_restored.request_schema_json)


class RequestSchemaRoundTripTest(absltest.TestCase):
    def test_round_trip_preserves_schema(self) -> None:
        original = StructType(
            [
                StructField("amount", DoubleType()),
                StructField("merchant_id", StringType()),
            ]
        )
        payload = request_schema_to_json(original)
        reconstructed = request_schema_from_json(payload)
        # Snowpark normalizes bare identifiers to upper case; compare on a
        # case-insensitive basis.
        self.assertEqual(
            [f.name.upper() for f in reconstructed.fields],
            ["AMOUNT", "MERCHANT_ID"],
        )
        self.assertEqual(
            [type(f.datatype).__name__ for f in reconstructed.fields],
            ["DoubleType", "StringType"],
        )


if __name__ == "__main__":
    absltest.main()
