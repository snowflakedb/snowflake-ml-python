"""Unit tests for :class:`snowflake.ml.feature_store.feature_group.FeatureGroup`.

Covers:

- Constructor validation (name, type, emptiness, duplicate-by-(name,version)).
- ``output_columns`` property: ``auto_prefix`` on/off, ``with_name`` precedence,
  slice ordering, and never-raise semantics on collisions.
- ``_to_spec`` end-to-end against the spec builder, exercising the version-aware
  prefix map and surfacing the builder's duplicate-output validation.
"""
from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_group import FeatureGroup, FeatureGroupVersion
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.spec.enums import FeatureViewKind, SourceType
from snowflake.snowpark import DataFrame
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType


def _make_registered_fv(
    *,
    name: str,
    version: str,
    feature_columns: list[str],
    join_keys: Optional[list[str]] = None,
    online: bool = True,
    store_type: OnlineStoreType = OnlineStoreType.POSTGRES,
) -> FeatureView:
    """Build a :class:`FeatureView` that looks registered for spec-builder use.

    Mocks the ``feature_df`` so ``output_schema`` returns a real
    :class:`StructType`. Sets ``_version`` and ``_database`` / ``_schema``
    fields directly to mimic the post-registration state without round-tripping
    through Snowflake.
    """
    join_keys = join_keys or ["USER_ID"]
    schema = StructType(
        [StructField(jk, StringType()) for jk in join_keys] + [StructField(c, DoubleType()) for c in feature_columns]
    )

    mock_df = MagicMock(spec=DataFrame)
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": [f"SELECT * FROM {name}"]}

    entity = Entity(name="USER", join_keys=join_keys)

    fv = FeatureView(
        name=name,
        entities=[entity],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=online, store_type=store_type),
    )
    fv._version = FeatureViewVersion(version)
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    # Schema inference uses _infer_schema_df.schema; share the same mock.
    fv._infer_schema_df = mock_df
    return fv


def _rtfv_compute_fn(req: Any, upstream: Any) -> Any:
    """Placeholder ``compute_fn`` for RTFV fixtures; tests never invoke it."""
    return req


def _make_registered_rtfv(
    *,
    name: str,
    version: str,
    output_fields: list[StructField],
    request_fields: Optional[list[StructField]] = None,
    upstream: Optional[FeatureView] = None,
    join_keys: Optional[list[str]] = None,
) -> FeatureView:
    """Build a :class:`FeatureView` that looks like a registered RTFV.

    The compute_fn is a placeholder; the FG-side helpers under test never
    invoke it. ``output_fields`` are the canonical ``compute_fn`` return
    shape; pass ``StructField('<jk>', ...)`` entries to exercise the PK
    re-declaration case.
    """
    join_keys = join_keys or ["USER_ID"]
    upstream = upstream or _make_registered_fv(
        name=f"{name}_UPSTREAM",
        version=version,
        feature_columns=["UPSTREAM_F"],
        join_keys=join_keys,
    )
    request = RequestSource(schema=StructType(request_fields or []))
    rtc = RealtimeConfig(
        compute_fn=_rtfv_compute_fn,
        sources=[request, upstream],
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


class FeatureGroupConstructorTest(parameterized.TestCase):
    """Constructor: name validation, emptiness, type, duplicates."""

    def test_valid_construction(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1", "F2"])
        fg = FeatureGroup(name="MY_FG", features=[fv])
        self.assertEqual(fg.name, "MY_FG")
        self.assertEqual(len(fg.features), 1)
        self.assertEqual(fg.desc, "")
        self.assertTrue(fg.auto_prefix)

    @parameterized.parameters(  # type: ignore[misc]
        ("",),  # empty
        ("has$dollar",),  # version delimiter is the only alphabet-level reject
        ("a" * 256,),  # exceeds 255-char cap
    )
    def test_invalid_names_rejected(self, bad_name: str) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        with self.assertRaises(ValueError):
            FeatureGroup(name=bad_name, features=[fv])

    @parameterized.parameters(  # type: ignore[misc]
        ("my.fg",),  # dotted
        ("my-fg",),  # hyphenated
        ("My_FG",),  # mixed case
        ("1starts_with_digit",),  # FV permits leading digits
        ("has space",),  # FV permits spaces
        ("a" * 255,),  # at the length cap
    )
    def test_relaxed_names_accepted(self, name: str) -> None:
        """FG name validation mirrors FeatureView: any string except `$`."""
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name=name, features=[fv])
        self.assertEqual(fg.name, name)

    def test_empty_features_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one feature view"):
            FeatureGroup(name="FG", features=[])

    def test_non_fv_item_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "FeatureView or FeatureViewSlice"):
            FeatureGroup(name="FG", features=["not_a_fv"])  # type: ignore[list-item]

    def test_duplicate_name_version_rejected(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv2 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F2"])
        with self.assertRaisesRegex(ValueError, "must be unique by"):
            FeatureGroup(name="FG", features=[fv1, fv2])

    def test_same_name_different_version_allowed(self) -> None:
        fv_v1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv_v2 = _make_registered_fv(name="USER_FV", version="v2", feature_columns=["G1"])
        fg = FeatureGroup(name="FG", features=[fv_v1, fv_v2])
        self.assertEqual(len(fg.features), 2)

    def test_slice_and_full_fv_treated_as_same_ref(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1", "F2"])
        sliced = fv.slice(["F1"])
        with self.assertRaisesRegex(ValueError, "must be unique by"):
            FeatureGroup(name="FG", features=[fv, sliced])


class FeatureGroupOutputColumnsTest(absltest.TestCase):
    """``output_columns`` property semantics."""

    def test_auto_prefix_on(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1", "F2"])
        fv2 = _make_registered_fv(name="TXN_FV", version="v1", feature_columns=["T1"])
        fg = FeatureGroup(name="FG", features=[fv1, fv2], auto_prefix=True)
        # Mixed-case ``v1`` forces SQL-quoting (matches what read/generate emit).
        self.assertEqual(
            fg.output_columns,
            ['"USER_FV_v1_F1"', '"USER_FV_v1_F2"', '"TXN_FV_v1_T1"'],
        )

    def test_auto_prefix_off_keeps_raw_names(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv2 = _make_registered_fv(name="TXN_FV", version="v1", feature_columns=["T1"])
        fg = FeatureGroup(name="FG", features=[fv1, fv2], auto_prefix=False)
        self.assertEqual(fg.output_columns, ["F1", "T1"])

    def test_with_name_overrides_auto_prefix(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv2 = _make_registered_fv(name="TXN_FV", version="v1", feature_columns=["T1"])
        fg = FeatureGroup(
            name="FG",
            features=[fv1.with_name("U"), fv2],
            auto_prefix=True,
        )
        # Upper-case-only prefixes stay bare; mixed-case (``TXN_FV_v1_``) gets SQL-quoted.
        self.assertEqual(fg.output_columns, ["U_F1", '"TXN_FV_v1_T1"'])

    def test_with_name_empty_disables_prefix(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name="FG", features=[fv.with_name("")], auto_prefix=True)
        self.assertEqual(fg.output_columns, ["F1"])

    def test_slice_preserves_caller_order(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1", "F2", "F3"])
        # Slice in non-source order.
        sliced = fv.slice(["F3", "F1"])
        fg = FeatureGroup(name="FG", features=[sliced], auto_prefix=False)
        self.assertEqual(fg.output_columns, ["F3", "F1"])

    def test_output_columns_does_not_raise_on_collision(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv2 = _make_registered_fv(name="TXN_FV", version="v1", feature_columns=["F1"])
        # Without prefixing both contribute "F1" — output_columns must still return.
        fg = FeatureGroup(name="FG", features=[fv1, fv2], auto_prefix=False)
        self.assertEqual(fg.output_columns, ["F1", "F1"])

    def test_rtfv_source_contributes_feature_columns_minus_pk(self) -> None:
        """RTFV ``output_schema`` re-declaring the FG superset PK must not double-emit it."""
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            # Canonical compute_fn return shape: PK + computed feature.
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        fg = FeatureGroup(name="FG", features=[rtfv], auto_prefix=False)
        self.assertEqual(fg.output_columns, ["WEIGHTED_BALANCE"])

    def test_rtfv_source_pk_dedupe_under_auto_prefix(self) -> None:
        """Auto-prefix is applied to feature columns only; PK never gets prefixed in."""
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        fg = FeatureGroup(name="FG", features=[rtfv], auto_prefix=True)
        # Mixed-case ``v1`` forces SQL-quoting on the prefixed name.
        self.assertEqual(fg.output_columns, ['"RT_FV_v1_WEIGHTED_BALANCE"'])

    def test_rtfv_source_pk_dedupe_with_explicit_alias(self) -> None:
        """``with_name`` overrides the auto prefix without changing the PK-dedupe contract."""
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        fg = FeatureGroup(name="FG", features=[rtfv.with_name("rt")], auto_prefix=True)
        # Mixed-case alias ``rt_`` triggers SQL-quoting.
        self.assertEqual(fg.output_columns, ['"rt_WEIGHTED_BALANCE"'])

    def test_mixed_bfv_and_rtfv_dedupe_against_superset_pk(self) -> None:
        """FG superset PK is the union across sources; both sources drop any column matching it."""
        bfv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["BALANCE"])
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        fg = FeatureGroup(name="FG", features=[bfv, rtfv], auto_prefix=False)
        self.assertEqual(fg.output_columns, ["BALANCE", "WEIGHTED_BALANCE"])

    def test_rtfv_source_without_pk_in_output_schema_unchanged(self) -> None:
        """If the RTFV's ``output_schema`` doesn't re-declare the PK, the property is unchanged."""
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[StructField("WEIGHTED_BALANCE", DoubleType())],
        )
        fg = FeatureGroup(name="FG", features=[rtfv], auto_prefix=False)
        self.assertEqual(fg.output_columns, ["WEIGHTED_BALANCE"])


class FeatureGroupComposeFromMetadataTest(absltest.TestCase):
    """Rehydration: ``compose_from_metadata`` -> :attr:`FeatureGroup.output_columns` stays deduped."""

    def test_rehydrated_fg_with_rtfv_source_drops_pk_in_output_columns(self) -> None:
        from snowflake.ml.feature_store.feature_group import compose_from_metadata
        from snowflake.ml.feature_store.metadata_manager import (
            FeatureGroupMetadata,
            FeatureGroupSourceRef,
        )

        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        fake_fs = MagicMock()
        # ``compose_from_metadata`` delegates to ``fs.get_feature_view`` to
        # rehydrate each source ref into a live FV.
        fake_fs.get_feature_view.return_value = rtfv

        meta = FeatureGroupMetadata(
            name="FG",
            version="v1",
            desc="rehydrate test",
            auto_prefix=False,
            sources=[FeatureGroupSourceRef(fv_name="RT_FV", fv_version="v1")],
        )
        rehydrated = compose_from_metadata(fake_fs, meta)
        self.assertEqual(rehydrated.output_columns, ["WEIGHTED_BALANCE"])


class FeatureGroupToSpecTest(absltest.TestCase):
    """``_to_spec`` produces a validated FeatureViewSpec."""

    def test_spec_kind_and_metadata(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        spec = FeatureGroup(name="FG", features=[fv])._to_spec(database="DB", schema="SCH", version="v1")
        self.assertEqual(spec.kind, FeatureViewKind.FeatureGroup)
        self.assertEqual(spec.metadata.database, "DB")
        self.assertEqual(spec.metadata.schema_, "SCH")
        self.assertEqual(spec.metadata.name, "FG")
        self.assertEqual(spec.metadata.version, "v1")
        self.assertEqual(len(spec.offline_configs), 0)
        self.assertIsNone(spec.spec.timestamp_field)
        self.assertIsNone(spec.spec.feature_granularity_sec)
        self.assertIsNone(spec.spec.udf)

    def test_spec_threads_user_version(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        spec = FeatureGroup(name="FG", features=[fv])._to_spec(database="DB", schema="SCH", version="2024_q1")
        self.assertEqual(spec.metadata.version, "2024_q1")

    def test_auto_prefix_drives_spec_features(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv2 = _make_registered_fv(name="TXN_FV", version="v1", feature_columns=["T1"])
        spec = FeatureGroup(name="FG", features=[fv1, fv2], auto_prefix=True)._to_spec(
            database="DB", schema="SCH", version="v1"
        )
        outputs = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(outputs, ["USER_FV_v1_F1", "TXN_FV_v1_T1"])

    def test_with_name_drives_spec_features(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        spec = FeatureGroup(name="FG", features=[fv1.with_name("u")], auto_prefix=False)._to_spec(
            database="DB", schema="SCH", version="v1"
        )
        outputs = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(outputs, ["u_F1"])

    def test_same_fv_different_versions_distinct_prefixes(self) -> None:
        fv_v1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv_v2 = _make_registered_fv(name="USER_FV", version="v2", feature_columns=["F1"])
        spec = FeatureGroup(name="FG", features=[fv_v1, fv_v2], auto_prefix=True)._to_spec(
            database="DB", schema="SCH", version="v1"
        )
        outputs = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(outputs, ["USER_FV_v1_F1", "USER_FV_v2_F1"])

    def test_unprefixed_collision_surfaces_builder_error(self) -> None:
        fv1 = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fv2 = _make_registered_fv(name="TXN_FV", version="v1", feature_columns=["F1"])
        fg = FeatureGroup(name="FG", features=[fv1, fv2], auto_prefix=False)
        with self.assertRaisesRegex(ValueError, "Duplicate output column"):
            fg._to_spec(database="DB", schema="SCH", version="v1")

    def test_features_sources_only(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        spec = FeatureGroup(name="FG", features=[fv])._to_spec(database="DB", schema="SCH", version="v1")
        self.assertTrue(all(s.source_type == SourceType.FEATURES for s in spec.spec.sources))

    def test_rtfv_source_emits_computed_features_minus_pk(self) -> None:
        """Spec features for an RTFV source come from ``realtime_config.output_schema`` minus the PK."""
        rtfv = _make_registered_rtfv(
            name="RT_FV",
            version="v1",
            output_fields=[
                StructField("USER_ID", StringType()),
                StructField("WEIGHTED_BALANCE", DoubleType()),
            ],
        )
        spec = FeatureGroup(name="FG", features=[rtfv], auto_prefix=False)._to_spec(
            database="DB", schema="SCH", version="v1"
        )
        outputs = [f.output_column.name for f in spec.spec.features]
        self.assertEqual(outputs, ["WEIGHTED_BALANCE"])
        # The OFT's PK comes from the derived entity columns, not the features list.
        self.assertEqual(spec.spec.ordered_entity_column_names, ["USER_ID"])


class FeatureGroupVersionTest(absltest.TestCase):
    """``FeatureGroupVersion`` validates against the same alphabet as FV."""

    def test_valid_versions_accepted(self) -> None:
        for v in ("v1", "v1.0", "2024_q1", "alpha-1", "9"):
            self.assertEqual(str(FeatureGroupVersion(v)), v)

    def test_invalid_versions_rejected(self) -> None:
        for bad in ("", "_starts_with_underscore", "has space", "has$dollar", "has/slash"):
            with self.assertRaises(ValueError):
                FeatureGroupVersion(bad)

    def test_overlong_version_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FeatureGroupVersion("v" * 200)


if __name__ == "__main__":
    absltest.main()
