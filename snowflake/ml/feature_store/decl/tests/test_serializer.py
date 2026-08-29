from snowflake.ml.test_utils import pytest_driver

"""Tests for decl/serializer.py — spec_to_dict, spec_to_yaml, spec_to_json, callable_to_source."""

import json
from typing import Any

import yaml

from snowflake.ml.feature_store.decl.serializer import (
    callable_to_source,
    spec_to_dict,
    spec_to_json,
    spec_to_yaml,
)
from snowflake.ml.feature_store.decl.spec_models import (
    UDF,
    BatchSource,
    Entity,
    Feature,
    FeatureGroup,
    FeatureView,
    FeatureViewRef,
    FSColumn,
    SourceRef,
    StreamingSource,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _customer_entity() -> Entity:
    return Entity(
        kind="Entity",
        name="customer",
        version="v1",
        join_keys=[FSColumn(name="customer_id", type="StringType")],
    )


def _streaming_source() -> StreamingSource:
    return StreamingSource(
        name="events",
        version="v1",
        columns=[
            FSColumn(name="customer_id", type="StringType"),
            FSColumn(name="amount", type="FloatType"),
        ],
    )


def _batch_source() -> BatchSource:
    return BatchSource(
        name="order_history",
        source_database="PROD",
        source_schema="PUBLIC",
        table="ORDERS",
    )


def _simple_fv() -> FeatureView:
    return FeatureView(
        name="click_stats",
        version="v1",
        kind="StreamingFeatureView",
        online=True,
        entities=["customer_id"],
        sources=[SourceRef(name="events", source_type="Stream")],
        features=[
            Feature(
                source_column=FSColumn(name="amount", type="FloatType"),
                output_column=FSColumn(name="total_spend_7d", type="FloatType"),
                function="sum",
                window="7d",
            )
        ],
    )


# ---------------------------------------------------------------------------
# spec_to_dict
# ---------------------------------------------------------------------------


class TestSpecToDict:
    def test_entity_basic(self) -> None:
        e = _customer_entity()
        d = spec_to_dict(e)
        assert d["kind"] == "Entity"
        assert d["name"] == "customer"
        assert d["version"] == "v1"

    def test_entity_join_keys_serialized(self) -> None:
        e = _customer_entity()
        d = spec_to_dict(e)
        assert isinstance(d["join_keys"], list)
        assert d["join_keys"][0]["name"] == "customer_id"
        assert d["join_keys"][0]["type"] == "StringType"

    def test_none_values_excluded(self) -> None:
        e = Entity(name="customer")
        d = spec_to_dict(e)
        assert "version" not in d
        assert "database" not in d
        assert "description" not in d

    def test_empty_list_excluded(self) -> None:
        e = Entity(name="customer")
        d = spec_to_dict(e)
        assert "join_keys" not in d

    def test_feature_view_sources_as_sourcerefs(self) -> None:
        fv = _simple_fv()
        d = spec_to_dict(fv)
        assert isinstance(d["sources"], list)
        src = d["sources"][0]
        assert src["name"] == "events"
        assert src["source_type"] == "Stream"

    def test_streaming_source_in_sources_collapsed_to_sourceref(self) -> None:
        """StreamingSource instances in sources list become SourceRef dicts."""
        src = _streaming_source()
        fv = FeatureView(
            name="fv",
            sources=[src],
        )
        d = spec_to_dict(fv)
        assert d["sources"][0]["name"] == "events"
        assert d["sources"][0]["source_type"] == "Stream"
        # Should NOT be the full source dict
        assert "columns" not in d["sources"][0]

    def test_batch_source_in_sources_collapsed_to_sourceref(self) -> None:
        """BatchSource instances in sources list become SourceRef dicts.

        Post-consolidation, the canonical ``source_type`` value is
        ``"Batch"`` (mirroring :data:`spec.enums.SourceType.BATCH`),
        not the legacy ``"BatchSource"`` string.
        """
        src = _batch_source()
        fv = FeatureView(
            name="fv",
            sources=[src],
        )
        d = spec_to_dict(fv)
        assert d["sources"][0]["name"] == "order_history"
        assert d["sources"][0]["source_type"] == "Batch"

    def test_entity_in_entities_expanded(self) -> None:
        """Entity objects in ``entities`` become join-key column name strings.

        The output dict carries the authoring key name (``entities``) —
        translation to the wire-form ``ordered_entity_column_names`` only
        happens inside :func:`spec_compiler.compile_to_spec`.
        """
        entity = _customer_entity()
        fv = FeatureView(
            name="fv",
            entities=[entity],
        )
        d = spec_to_dict(fv)
        assert d["entities"] == ["customer_id"]
        assert "ordered_entity_column_names" not in d

    def test_string_entities_passthrough(self) -> None:
        fv = _simple_fv()
        d = spec_to_dict(fv)
        assert d["entities"] == ["customer_id"]
        assert "ordered_entity_column_names" not in d

    def test_feature_group_feature_view_refs_passthrough(self) -> None:
        ref = FeatureViewRef(name="click_stats", version="V1")
        fg = FeatureGroup(name="my_group", feature_views=[ref])
        d = spec_to_dict(fg)
        assert d["feature_views"] == [{"name": "click_stats", "version": "V1"}]

    def test_feature_group_feature_view_refs_with_slice_and_alias(self) -> None:
        ref = FeatureViewRef(
            name="click_stats",
            version="V1",
            slice_columns=["A", "B"],
            alias="cs",
        )
        fg = FeatureGroup(name="my_group", feature_views=[ref])
        d = spec_to_dict(fg)
        assert d["feature_views"] == [
            {
                "name": "click_stats",
                "version": "V1",
                "slice_columns": ["A", "B"],
                "alias": "cs",
            }
        ]

    def test_udf_callable_converted_to_source(self) -> None:
        def my_transform(df: Any) -> Any:
            return df * 2

        udf = UDF(name="my_transform", function_definition=my_transform)
        fv = FeatureView(name="fv", udf=udf)
        d = spec_to_dict(fv)
        assert isinstance(d["udf"]["function_definition"], str)
        assert "def my_transform" in d["udf"]["function_definition"]

    def test_udf_string_source_passthrough(self) -> None:
        udf = UDF(name="fn", function_definition="def fn(df):\n    return df\n")
        fv = FeatureView(name="fv", udf=udf)
        d = spec_to_dict(fv)
        assert d["udf"]["function_definition"] == "def fn(df):\n    return df\n"

    def test_features_serialized(self) -> None:
        fv = _simple_fv()
        d = spec_to_dict(fv)
        feat = d["features"][0]
        assert feat["function"] == "sum"
        assert feat["window"] == "7d"
        assert feat["output_column"]["name"] == "total_spend_7d"

    def test_boolean_true_included(self) -> None:
        fv = FeatureView(name="fv", online=True)
        d = spec_to_dict(fv)
        assert d["online"] is True

    def test_boolean_false_excluded(self) -> None:
        # The ``False`` exclusion rule is documented as a serializer
        # contract; exercise it on ``BatchFeatureView`` because
        # ``StreamingFeatureView`` rejects ``online=False`` via
        # ``_enforce_always_online_for_stream_or_realtime`` (streaming
        # FVs are always online by design).
        fv = FeatureView(kind="BatchFeatureView", name="fv", online=False)
        d = spec_to_dict(fv)
        assert "online" not in d


class TestSpecToDictFeatureAggregationMethod:
    def test_feature_aggregation_method_included_when_set(self) -> None:
        fv = FeatureView(name="fv", feature_aggregation_method="continuous")
        d = spec_to_dict(fv)
        assert d["feature_aggregation_method"] == "continuous"

    def test_feature_aggregation_method_tiles_included(self) -> None:
        fv = FeatureView(name="fv", feature_aggregation_method="tiles")
        d = spec_to_dict(fv)
        assert d["feature_aggregation_method"] == "tiles"

    def test_feature_aggregation_method_excluded_when_none(self) -> None:
        fv = FeatureView(name="fv")
        d = spec_to_dict(fv)
        assert "feature_aggregation_method" not in d

    def test_feature_aggregation_method_survives_json_round_trip(self) -> None:
        import json as _json

        fv = FeatureView(name="fv", feature_aggregation_method="continuous")
        json_str = spec_to_json(fv)
        parsed = _json.loads(json_str)
        assert parsed["feature_aggregation_method"] == "continuous"


# ---------------------------------------------------------------------------
# spec_to_yaml
# ---------------------------------------------------------------------------


class TestSpecToYaml:
    def test_returns_string(self) -> None:
        e = _customer_entity()
        result = spec_to_yaml(e)
        assert isinstance(result, str)

    def test_valid_yaml(self) -> None:
        e = _customer_entity()
        result = spec_to_yaml(e)
        parsed = yaml.safe_load(result)
        assert parsed["kind"] == "Entity"
        assert parsed["name"] == "customer"

    def test_join_keys_in_yaml(self) -> None:
        e = _customer_entity()
        result = spec_to_yaml(e)
        parsed = yaml.safe_load(result)
        assert parsed["join_keys"][0]["name"] == "customer_id"

    def test_round_trip(self) -> None:
        fv = _simple_fv()
        yaml_str = spec_to_yaml(fv)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["kind"] == "StreamingFeatureView"
        assert parsed["sources"][0]["name"] == "events"


# ---------------------------------------------------------------------------
# spec_to_json
# ---------------------------------------------------------------------------


class TestSpecToJson:
    def test_returns_string(self) -> None:
        e = _customer_entity()
        result = spec_to_json(e)
        assert isinstance(result, str)

    def test_valid_json(self) -> None:
        e = _customer_entity()
        result = spec_to_json(e)
        parsed = json.loads(result)
        assert parsed["kind"] == "Entity"

    def test_round_trip(self) -> None:
        fv = _simple_fv()
        json_str = spec_to_json(fv)
        parsed = json.loads(json_str)
        assert parsed["kind"] == "StreamingFeatureView"
        assert parsed["features"][0]["function"] == "sum"


# ---------------------------------------------------------------------------
# callable_to_source
# ---------------------------------------------------------------------------


class TestCallableToSource:
    def test_returns_string(self) -> None:
        def my_fn(df: Any) -> Any:
            return df

        result = callable_to_source(my_fn)
        assert isinstance(result, str)

    def test_contains_def(self) -> None:
        def transform(df: Any) -> Any:
            return df * 2

        result = callable_to_source(transform)
        assert result.startswith("def transform")

    def test_strips_decorator(self) -> None:
        import functools

        def simple_deco(fn: Any) -> Any:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            return wrapper

        @simple_deco
        def my_fn(df: Any) -> Any:
            return df + 1

        result = callable_to_source(my_fn)
        assert "def my_fn" in result
        assert not result.startswith("@")

    def test_dedented(self) -> None:
        def outer() -> Any:
            def inner(df: Any) -> Any:
                return df

            return inner

        fn = outer()
        result = callable_to_source(fn)
        # Should not start with whitespace
        assert result.startswith("def inner")


# ---------------------------------------------------------------------------
# target_name round-trip in PlanFile envelope (Phase 3+4 D4-ext)
#
# The plan envelope carries a target_name field so apply --target X can
# reject a plan generated for a different target with status="target_mismatch"
# (D4-ext / Plan-file target match). Old plan JSONs without the field
# deserialize with target_name == "" so legacy plans still apply when no
# --target was requested.
# ---------------------------------------------------------------------------


class TestPlanFileTargetName:
    """Cover the new ``target_name`` field on the serialized plan envelope."""

    def _empty_plan(self) -> Any:
        from snowflake.ml.feature_store.decl.types import Plan

        return Plan(ops=[], warnings=[])

    def test_serialize_plan_writes_target_name_field(self) -> None:
        """``serialize_plan`` MUST include ``target_name`` in the JSON envelope."""
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = self._empty_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", ["a.yaml"], target_name="DEV")
        parsed = json.loads(out)
        assert parsed["target_name"] == "DEV"

    def test_serialize_plan_default_target_name_is_empty(self) -> None:
        """When ``target_name`` is omitted, the field round-trips as ``""``."""
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = self._empty_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", [])
        parsed = json.loads(out)
        assert parsed.get("target_name", "") == ""

    def test_deserialize_plan_round_trips_target_name(self) -> None:
        """``deserialize_plan`` populates ``target_name`` from the JSON."""
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = self._empty_plan()
        json_str = decl_api.serialize_plan(plan, "DB", "SCH", [], target_name="PROD")
        pf = decl_api.deserialize_plan(json_str)
        assert pf.target_name == "PROD"

    def test_deserialize_plan_legacy_json_defaults_to_empty(self) -> None:
        """A pre-Phase-3+4 plan (no ``target_name``) deserialises with ``""``."""
        from snowflake.ml.feature_store.decl import api as decl_api

        legacy_json = json.dumps(
            {
                "version": "1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "target_database": "DB",
                "target_schema": "SCH",
                "source_files": [],
                "plan": {"ops": [], "warnings": []},
                "summary": {},
            }
        )
        pf = decl_api.deserialize_plan(legacy_json)
        assert pf.target_name == ""

    def test_planfile_target_name_field_default_empty(self) -> None:
        """``PlanFile.target_name`` defaults to ``""`` when constructed bare."""
        from snowflake.ml.feature_store.decl.types import PlanFile

        pf = PlanFile()
        assert pf.target_name == ""

    def test_serialize_plan_target_name_preserves_case(self) -> None:
        """Case is preserved through serialize/deserialize (caller normalises)."""
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = self._empty_plan()
        json_str = decl_api.serialize_plan(plan, "DB", "SCH", [], target_name="MixedCase")
        pf = decl_api.deserialize_plan(json_str)
        assert pf.target_name == "MixedCase"


if __name__ == "__main__":
    pytest_driver.main()
