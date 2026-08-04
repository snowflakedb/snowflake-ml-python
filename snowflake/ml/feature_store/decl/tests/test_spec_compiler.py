from typing import Any

"""Tests for decl/spec_compiler.py — compile_to_spec and sanitize_json_for_dollar_quoting."""

import json

import pytest

from snowflake.ml.feature_store.decl.spec_compiler import (
    compile_to_spec,
    sanitize_json_for_dollar_quoting,
)

# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------

_SAMPLE_SPEC: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "name": "user_event_features",
    "version": "v1",
    "entities": ["user_id"],
    "sources": [
        {
            "name": "user_events",
            "source_type": "Stream",
            "columns": [
                {"name": "user_id", "type": "StringType"},
                {"name": "timestamp", "type": "TimestampType"},
                {"name": "event_count", "type": "IntegerType"},
            ],
        }
    ],
    "udf": {
        "name": "compute_event_features",
        "engine": "pandas",
        "function_definition": "def compute_event_features(df):\n    return df\n",
        "output_columns": [
            {"name": "user_id", "type": "StringType"},
            {"name": "timestamp", "type": "TimestampType"},
            {"name": "event_count", "type": "IntegerType"},
            {"name": "total_value", "type": "FloatType"},
        ],
    },
    "features": [
        {
            "source_column": {"name": "event_count", "type": "IntegerType"},
            "output_column": {"name": "event_count_1h", "type": "IntegerType"},
            "function": "count",
            "window": "1h",
        },
        {
            "source_column": {"name": "event_count", "type": "IntegerType"},
            "output_column": {"name": "event_count_5m", "type": "IntegerType"},
            "function": "count",
            "window": "5m",
        },
    ],
    "timestamp_col": "timestamp",
    "feature_granularity": "5m",
}


# ---------------------------------------------------------------------------
# Basic top-level structure
# ---------------------------------------------------------------------------


class TestCompileToSpec:
    def test_basic_compilation_returns_dict(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert isinstance(result, dict)

    def test_kind_preserved(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["kind"] == "StreamingFeatureView"

    def test_online_store_type_is_postgres(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["online_store_type"] == "postgres"

    def test_result_has_all_top_level_keys(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert "kind" in result
        assert "metadata" in result
        assert "offline_configs" in result
        assert "spec" in result
        assert "online_store_type" in result


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_database(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["metadata"]["database"] == "DEMO_DB"

    def test_metadata_schema(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["metadata"]["schema"] == "PUBLIC"

    def test_metadata_name(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["metadata"]["name"] == "user_event_features"

    def test_metadata_version(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["metadata"]["version"] == "v1"

    def test_metadata_spec_format_version(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["metadata"]["spec_format_version"] == "1"

    def test_metadata_internal_data_version(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["metadata"]["internal_data_version"] == "1"

    def test_metadata_client_version_present(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert "client_version" in result["metadata"]
        assert isinstance(result["metadata"]["client_version"], str)
        assert len(result["metadata"]["client_version"]) > 0

    def test_metadata_reflects_caller_database_and_schema(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "MY_DB", "MY_SCHEMA")
        assert result["metadata"]["database"] == "MY_DB"
        assert result["metadata"]["schema"] == "MY_SCHEMA"


# ---------------------------------------------------------------------------
# offline_configs
# ---------------------------------------------------------------------------


class TestOfflineConfigs:
    def test_offline_configs_has_one_entry_with_udf(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert len(result["offline_configs"]) == 1

    def test_offline_config_store_type(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["offline_configs"][0]["store_type"] == "snowflake"

    def test_offline_config_table_type(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["offline_configs"][0]["table_type"] == "UDFTransformed"

    def test_offline_config_database(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["offline_configs"][0]["database"] == "DEMO_DB"

    def test_offline_config_schema(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["offline_configs"][0]["schema"] == "PUBLIC"

    def test_offline_config_table_name_uppercased_name(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["offline_configs"][0]["table"] == "USER_EVENT_FEATURES$V1$UDF_TRANSFORMED"

    def test_offline_config_table_name_preserves_version_case(self) -> None:
        spec = dict(_SAMPLE_SPEC)
        spec["version"] = "V2"
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        assert result["offline_configs"][0]["table"] == "USER_EVENT_FEATURES$V2$UDF_TRANSFORMED"

    def test_offline_config_columns_from_udf_output_columns(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        cols = result["offline_configs"][0]["columns"]
        assert len(cols) == 4

    def test_offline_config_columns_content(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        cols = result["offline_configs"][0]["columns"]
        assert cols[0] == {"name": "user_id", "type": "StringType"}
        assert cols[1] == {"name": "timestamp", "type": "TimestampType"}
        assert cols[2] == {"name": "event_count", "type": "IntegerType"}
        assert cols[3] == {"name": "total_value", "type": "FloatType"}


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


class TestSpec:
    def test_ordered_entity_column_names(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["spec"]["ordered_entity_column_names"] == ["user_id"]

    def test_sources_count(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert len(result["spec"]["sources"]) == 1

    def test_sources_name_preserved(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["spec"]["sources"][0]["name"] == "user_events"

    def test_timestamp_field(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["spec"]["timestamp_field"] == "timestamp"

    def test_feature_granularity_sec_converted_from_string(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["spec"]["feature_granularity_sec"] == 300

    def test_feature_granularity_key_absent(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert "feature_granularity" not in result["spec"]

    def test_feature_granularity_sec_passthrough_when_already_int(self) -> None:
        spec = {k: v for k, v in _SAMPLE_SPEC.items()}
        spec = dict(_SAMPLE_SPEC)
        del spec["feature_granularity"]
        spec["feature_granularity_sec"] = 600
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        assert result["spec"]["feature_granularity_sec"] == 600

    def test_features_window_sec_1h_to_3600(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        feat = result["spec"]["features"][0]
        assert feat["window_sec"] == 3600

    def test_features_window_sec_5m_to_300(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        feat = result["spec"]["features"][1]
        assert feat["window_sec"] == 300

    def test_features_window_key_absent(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        for feat in result["spec"]["features"]:
            assert "window" not in feat

    def test_features_window_sec_30s_to_30(self) -> None:
        spec = dict(_SAMPLE_SPEC)
        spec["features"] = [
            {
                "source_column": {"name": "val", "type": "DoubleType"},
                "output_column": {"name": "val_30s", "type": "DoubleType"},
                "function": "sum",
                "window": "30s",
            }
        ]
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        assert result["spec"]["features"][0]["window_sec"] == 30

    def test_feature_aggregation_method_tiles_when_windows_present(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["spec"]["feature_aggregation_method"] == "tiles"

    def test_udf_function_definition_present(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert "function_definition" in result["spec"]["udf"]

    def test_udf_function_definition_content(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert "compute_event_features" in result["spec"]["udf"]["function_definition"]

    def test_udf_file_key_removed(self) -> None:
        spec = dict(_SAMPLE_SPEC)
        spec["udf"] = dict(_SAMPLE_SPEC["udf"])
        spec["udf"]["file"] = "./example_udf.py"
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        assert "file" not in result["spec"]["udf"]

    def test_udf_name_and_engine_passthrough(self) -> None:
        """compile_to_spec keeps authoring-shape ``name`` / ``engine`` keys.

        Live ``DESCRIBE ONLINE FEATURE TABLE ... TYPE = SPECIFICATION``
        output uses ``name`` and ``engine`` directly (verified in
        ``tests/golden_specs/USER_CLICK_STATS.json``), so the compiler
        must NOT rename them to ``function_name`` / ``language`` — doing
        so would invert the round-trip invariant and force every UDF FV
        to ``RECREATE`` on a clean export → plan cycle.  Only ``source``
        is renamed to ``function_definition`` (the legacy alias the live
        payload uses for the inlined Python source body).
        """
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert result["spec"]["udf"]["name"] == "compute_event_features"
        assert result["spec"]["udf"]["engine"] == "pandas"
        assert "function_name" not in result["spec"]["udf"]
        assert "language" not in result["spec"]["udf"]

    def test_udf_output_columns_in_spec_udf(self) -> None:
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        assert len(result["spec"]["udf"]["output_columns"]) == 4


# ---------------------------------------------------------------------------
# Features-only FV (no UDF)
# ---------------------------------------------------------------------------


class TestNoUdf:
    _NO_UDF_SPEC: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "simple_fv",
        "version": "v1",
        "entities": ["user_id"],
        "sources": [{"name": "src", "source_type": "Stream", "columns": []}],
        "features": [
            {
                "source_column": {"name": "val", "type": "DoubleType"},
                "output_column": {"name": "val_sum", "type": "DoubleType"},
                "function": "sum",
                "window": "30s",
            }
        ],
        "timestamp_col": "ts",
        "feature_granularity": "30s",
    }

    def test_offline_configs_empty_without_udf(self) -> None:
        result = compile_to_spec(self._NO_UDF_SPEC, "DB", "SCH")
        assert result["offline_configs"] == []

    def test_udf_absent_from_spec_without_udf(self) -> None:
        result = compile_to_spec(self._NO_UDF_SPEC, "DB", "SCH")
        assert "udf" not in result["spec"]

    def test_online_store_type_still_postgres_without_udf(self) -> None:
        result = compile_to_spec(self._NO_UDF_SPEC, "DB", "SCH")
        assert result["online_store_type"] == "postgres"

    def test_features_still_converted_without_udf(self) -> None:
        result = compile_to_spec(self._NO_UDF_SPEC, "DB", "SCH")
        assert result["spec"]["features"][0]["window_sec"] == 30


# ---------------------------------------------------------------------------
# Dollar-quoting sanitization
# ---------------------------------------------------------------------------


class TestSanitizeJsonForDollarQuoting:
    def test_double_dollar_replaced(self) -> None:
        result = sanitize_json_for_dollar_quoting('{"code": "$$python$$"}')
        assert "$$" not in result
        assert "$\\u0024" in result

    def test_single_dollar_unchanged(self) -> None:
        payload = '{"code": "$python"}'
        assert sanitize_json_for_dollar_quoting(payload) == payload

    def test_empty_string(self) -> None:
        assert sanitize_json_for_dollar_quoting("") == ""

    def test_no_dollars(self) -> None:
        payload = '{"key": "value"}'
        assert sanitize_json_for_dollar_quoting(payload) == payload

    def test_sanitized_json_round_trips(self) -> None:
        """Parsing the sanitized JSON restores the original $$ value."""
        spec = dict(_SAMPLE_SPEC)
        spec["udf"] = dict(_SAMPLE_SPEC["udf"])
        spec["udf"]["function_definition"] = "def fn():\n    x = '$$tag$$'\n"
        result = compile_to_spec(spec, "DB", "SCH")
        raw_json = json.dumps(result)
        sanitized = sanitize_json_for_dollar_quoting(raw_json)
        assert "$$" not in sanitized
        # Standard JSON parser decodes \u0024 back to $
        parsed = json.loads(sanitized)
        assert "$$tag$$" in parsed["spec"]["udf"]["function_definition"]


# ---------------------------------------------------------------------------
# Source column enrichment
# ---------------------------------------------------------------------------


class TestSourceColumnEnrichment:
    """Verify that sources without columns get enriched by the compiler."""

    _UDF_SPEC_NO_SOURCE_COLS: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "user_event_features",
        "version": "v1",
        "entities": ["user_id"],
        "sources": [
            {
                "name": "user_events",
                "source_type": "Stream",
                # No "columns" key — this is the bug scenario
            }
        ],
        "udf": {
            "name": "compute_event_features",
            "engine": "pandas",
            "function_definition": "def compute_event_features(df):\n    return df\n",
            "output_columns": [
                {"name": "user_id", "type": "StringType"},
                {"name": "timestamp", "type": "TimestampType"},
                {"name": "event_count", "type": "LongType"},
                {"name": "total_value", "type": "DoubleType"},
            ],
        },
        "features": [
            {
                "source_column": {"name": "event_count", "type": "LongType"},
                "output_column": {"name": "event_count_1h", "type": "LongType"},
                "function": "count",
                "window": "1h",
            },
        ],
        "timestamp_col": "timestamp",
        "feature_granularity": "5m",
    }

    def test_stream_source_without_columns_not_auto_enriched(self) -> None:
        """Stream source without columns should NOT be auto-enriched by compiler.

        Source columns must come from the datasource YAML, resolved by the
        apply pipeline before compilation — not inferred from UDF output.
        """
        result = compile_to_spec(self._UDF_SPEC_NO_SOURCE_COLS, "DEMO_DB", "PUBLIC")
        sources = result["spec"]["sources"]
        assert len(sources) == 1
        assert "columns" not in sources[0], "Source should NOT have columns auto-enriched by compiler"

    def test_stream_source_with_pre_resolved_columns_preserved(self) -> None:
        """If source already has columns (pre-resolved), they are preserved."""
        spec = dict(self._UDF_SPEC_NO_SOURCE_COLS)
        spec["sources"] = [
            {
                "name": "user_events",
                "source_type": "Stream",
                "columns": [
                    {"name": "user_id", "type": "StringType"},
                    {"name": "event_type", "type": "StringType"},
                    {"name": "event_value", "type": "DoubleType"},
                    {"name": "timestamp", "type": "TimestampType"},
                ],
            }
        ]
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        cols = result["spec"]["sources"][0]["columns"]
        assert len(cols) == 4
        assert cols[1]["name"] == "event_type"
        assert cols[2]["name"] == "event_value"

    def test_explicit_source_columns_preserved(self) -> None:
        """If source already has columns, they should NOT be overwritten."""
        result = compile_to_spec(_SAMPLE_SPEC, "DEMO_DB", "PUBLIC")
        cols = result["spec"]["sources"][0]["columns"]
        # _SAMPLE_SPEC has 3 explicit source columns
        assert len(cols) == 3
        assert cols[0]["name"] == "user_id"
        assert cols[1]["name"] == "timestamp"
        assert cols[2]["name"] == "event_count"

    def test_non_stream_source_not_enriched(self) -> None:
        """Non-Stream sources (e.g. BatchSource) should not get auto-enriched."""
        spec: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "batch_fv",
            "version": "v1",
            "entities": ["user_id"],
            "sources": [{"name": "batch_src", "source_type": "Batch"}],
            "features": [
                {
                    "source_column": {"name": "val", "type": "DoubleType"},
                    "output_column": {"name": "val", "type": "DoubleType"},
                },
            ],
            "timestamp_col": "ts",
        }
        result = compile_to_spec(spec, "DB", "SCH")
        # BatchSource should not be enriched
        assert "columns" not in result["spec"]["sources"][0]

    def test_no_udf_stream_without_columns_not_enriched(self) -> None:
        """Stream source without UDF and without columns should not be enriched."""
        spec: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "name": "simple_stream",
            "version": "v1",
            "entities": ["user_id"],
            "sources": [{"name": "events", "source_type": "Stream"}],
            "features": [
                {
                    "source_column": {"name": "amount", "type": "DoubleType"},
                    "output_column": {"name": "amount_sum", "type": "DoubleType"},
                    "function": "sum",
                    "window": "1h",
                },
            ],
            "timestamp_col": "ts",
            "feature_granularity": "5m",
        }
        result = compile_to_spec(spec, "DB", "SCH")
        assert "columns" not in result["spec"]["sources"][0]


# ---------------------------------------------------------------------------
# BUG_BASH step 7 audit — pin compiler aggregation preservation
# ---------------------------------------------------------------------------


# Authoring-format spec dict mirroring docs/BUG_BASH.md §5 (the canned
# USER_CLICK_STATS_DECL YAML).  Carries two windowed aggregation features
# (TOTAL_ENGAGEMENT_1H/sum, HAS_CONVERSION_24H/max), feature_granularity_sec
# 300, and feature_aggregation_method "tiles".  Used by the audit tests
# below to pin that the compiler preserves every aggregation field
# end-to-end so a regression that drops one of them surfaces here before
# the executor gets a chance to misbehave on a live apply.
_BUG_BASH_AUTHORING_SPEC: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "name": "USER_CLICK_STATS_DECL",
    "version": "V1",
    "online": True,
    "entities": ["USER_ID"],
    "timestamp_col": "TIMESTAMP",
    "feature_granularity_sec": 300,
    "feature_aggregation_method": "tiles",
    "sources": [
        {
            "name": "CLICKSTREAM_EVENTS",
            "source_type": "Stream",
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TYPE", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
                {"name": "TIME_ON_PAGE_SECONDS", "type": "DoubleType"},
            ],
        }
    ],
    "features": [
        {
            "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
            "window_sec": 3600,
            "function": "sum",
            "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
        },
        {
            "output_column": {"name": "HAS_CONVERSION_24H", "type": "BooleanType"},
            "window_sec": 86400,
            "function": "max",
            "source_column": {"name": "IS_CONVERSION", "type": "BooleanType"},
        },
    ],
    "udf": {
        "name": "compute_engagement_metrics",
        "engine": "pandas",
        "function_definition": "def compute_engagement_metrics(df):\n    return df\n",
        "output_columns": [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "TIMESTAMP", "type": "TimestampType"},
            {"name": "IS_CONVERSION", "type": "BooleanType"},
            {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
        ],
    },
}


class TestStreamingFvAggregationRoundTrip:
    """Pin BUG_BASH §5 aggregation triple survives :func:`compile_to_spec`.

    The bug surfaced as ``snow feature query`` returning the raw UDF
    output columns (``IS_CONVERSION``, ``ENGAGEMENT_SCORE``) instead of
    the windowed-aggregation outputs (``TOTAL_ENGAGEMENT_1H``,
    ``HAS_CONVERSION_24H``).  These tests pin that the compiler — the
    layer just below the planner / executor — preserves every field
    the streaming branch of ``imperative_executor._build_feature_view``
    needs to construct a tiled ``FeatureView``.  A regression in the
    compiler that drops any of these fields would surface here before
    the live ``snow feature apply`` round-trip can fail.
    """

    def test_bug_bash_yaml_compiles_to_payload_with_two_aggregations(self) -> None:
        """Both BUG_BASH features survive compilation with their aggregation
        triple intact (window_sec, function, source_column, output_column).
        """
        result = compile_to_spec(_BUG_BASH_AUTHORING_SPEC, "JKEW_DB", "JKEW_SCHEMA")
        features = result["spec"]["features"]
        assert len(features) == 2

        first, second = features
        assert first["window_sec"] == 3600
        assert first["function"] == "sum"
        assert first["source_column"]["name"] == "ENGAGEMENT_SCORE"
        assert first["output_column"]["name"] == "TOTAL_ENGAGEMENT_1H"

        assert second["window_sec"] == 86400
        assert second["function"] == "max"
        assert second["source_column"]["name"] == "IS_CONVERSION"
        assert second["output_column"]["name"] == "HAS_CONVERSION_24H"

    def test_compile_preserves_feature_aggregation_method_tiles(self) -> None:
        """feature_aggregation_method must round-trip as ``tiles`` so the
        executor passes ``FeatureAggregationMethod.TILES`` to the
        imperative ``FeatureView`` constructor and ``is_tiled`` triggers.
        """
        result = compile_to_spec(_BUG_BASH_AUTHORING_SPEC, "JKEW_DB", "JKEW_SCHEMA")
        assert result["spec"]["feature_aggregation_method"] == "tiles"

    def test_compile_preserves_feature_granularity_sec_300(self) -> None:
        """feature_granularity_sec must round-trip as the integer 300 so
        the executor passes ``feature_granularity="300s"`` to the
        imperative constructor.
        """
        result = compile_to_spec(_BUG_BASH_AUTHORING_SPEC, "JKEW_DB", "JKEW_SCHEMA")
        assert result["spec"]["feature_granularity_sec"] == 300


# ---------------------------------------------------------------------------
# target_lag stripping for streaming / realtime kinds
# ---------------------------------------------------------------------------


class TestCompileToSpecStripsTargetLagForStreamingAndRealtime:
    """Pin :func:`compile_to_spec`'s wire-format symmetry with the runtime.

    Streaming and realtime feature views always run at 0 seconds target
    lag — the Snowflake runtime stamps ``target_lag_sec: 0`` onto the
    deployed ``DESCRIBE … TYPE = SPECIFICATION`` payload regardless of
    the authored value.  The Pydantic-layer validator
    (``FeatureView._reject_target_lag_on_stream_or_realtime``) already
    rejects authored values on the Python / loader path; this pins the
    compiler's defence-in-depth: any payload that still carries
    ``target_lag_sec`` (a hand-built dict that bypasses the validator,
    or a re-applied SPECIFICATION already carrying the runtime stamp)
    must NOT propagate that key into the compiled wire ``spec`` for
    streaming / realtime kinds.

    The compiler stripping is what keeps :func:`_full_spec_hash`
    symmetric with the runtime side (``_RUNTIME_STAMPED_SPEC_KEYS``
    strips the same key from the deployed-side payload before hashing).
    Batch FVs are unaffected — they continue to carry
    ``spec.target_lag_sec`` per the existing ``compile_to_spec``
    contract.
    """

    def _streaming_spec_with_target_lag(self, target_lag_sec: int) -> dict[str, Any]:
        spec = dict(_BUG_BASH_AUTHORING_SPEC)
        spec["target_lag_sec"] = target_lag_sec
        return spec

    def test_compile_streaming_fv_strips_target_lag_sec_nonzero(self) -> None:
        """An authored ``target_lag_sec`` on a streaming kind must not
        leak into the compiled ``spec`` (would create a phantom drift
        against the runtime-stamped ``0``).
        """
        result = compile_to_spec(
            self._streaming_spec_with_target_lag(3600),
            "JKEW_DB",
            "JKEW_SCHEMA",
        )
        assert "target_lag_sec" not in result["spec"]

    def test_compile_streaming_fv_strips_target_lag_sec_zero(self) -> None:
        """Even the runtime-stamped ``0`` must be stripped so a
        re-compile of a previously-described spec does not carry the
        key forward (the runtime adds it back; keeping it would
        double-count and defeat the ``_RUNTIME_STAMPED_SPEC_KEYS``
        hash symmetry).
        """
        result = compile_to_spec(
            self._streaming_spec_with_target_lag(0),
            "JKEW_DB",
            "JKEW_SCHEMA",
        )
        assert "target_lag_sec" not in result["spec"]

    def test_compile_streaming_fv_strips_target_lag_string(self) -> None:
        """The compiler must also strip the human-friendly ``target_lag``
        authoring string before :func:`_resolve_duration` would convert
        it into ``target_lag_sec`` for streaming kinds.
        """
        spec = dict(_BUG_BASH_AUTHORING_SPEC)
        spec["target_lag"] = "1h"
        result = compile_to_spec(spec, "JKEW_DB", "JKEW_SCHEMA")
        assert "target_lag_sec" not in result["spec"]
        assert "target_lag" not in result["spec"]

    def test_compile_realtime_fv_strips_target_lag_sec(self) -> None:
        """Same contract for ``RealtimeFeatureView`` — the runtime treats
        realtime FVs identically to streaming FVs for ``target_lag``.
        """
        realtime_spec: dict[str, Any] = {
            "kind": "RealtimeFeatureView",
            "name": "REALTIME_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "SRC",
                    "source_type": "Stream",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
            "target_lag_sec": 60,
        }
        result = compile_to_spec(realtime_spec, "JKEW_DB", "JKEW_SCHEMA")
        assert "target_lag_sec" not in result["spec"]

    def test_compile_batch_fv_still_emits_target_lag_sec(self) -> None:
        """Regression pin: ``BatchFeatureView`` continues to carry
        ``spec.target_lag_sec`` (the stripping is scoped to streaming /
        realtime kinds only).

        After the ``feature_granularity`` / ``refresh_freq`` /
        ``target_lag`` decoupling, the wire-form ``spec.target_lag_sec``
        is sourced from the authoring ``refresh_freq`` (the canonical
        DT-refresh field).  The authoring ``target_lag`` field is
        OFT-only and does NOT populate this wire field.
        """
        batch_spec: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "BATCH_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "SRC",
                    "source_type": "Batch",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
            "refresh_freq": "1 hour",
        }
        result = compile_to_spec(batch_spec, "JKEW_DB", "JKEW_SCHEMA")
        assert result["spec"]["target_lag_sec"] == 3600


class TestNormalizeDurationsStripsTargetLagForStreamingAndRealtime:
    """Pin :func:`compiler.normalize_durations`'s kind-gated strip.

    ``normalize_durations`` runs over the *raw* authoring dict before
    Pydantic validation — it is the earliest opportunity to coerce
    human-friendly duration strings on the wire path.  Stripping
    ``target_lag`` / ``target_lag_sec`` here for streaming and realtime
    kinds is defence-in-depth: a YAML that carries the key (e.g. a
    legacy exported spec from before the exporter fix) won't even reach
    the Pydantic validator with the offending key still set, so the
    failure mode is graceful (key dropped) rather than hard-failing
    early on a re-load of a pre-fix export.
    """

    def test_streaming_fv_normalize_durations_strips_target_lag_sec(self) -> None:
        from snowflake.ml.feature_store.decl.compiler import normalize_durations

        data: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "name": "x",
            "target_lag_sec": 3600,
        }
        out = normalize_durations(data)
        assert "target_lag_sec" not in out
        assert "target_lag" not in out

    def test_streaming_fv_normalize_durations_strips_target_lag_string(self) -> None:
        from snowflake.ml.feature_store.decl.compiler import normalize_durations

        data: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "name": "x",
            "target_lag": "1h",
        }
        out = normalize_durations(data)
        assert "target_lag_sec" not in out
        assert "target_lag" not in out

    def test_realtime_fv_normalize_durations_strips_target_lag(self) -> None:
        from snowflake.ml.feature_store.decl.compiler import normalize_durations

        data: dict[str, Any] = {
            "kind": "RealtimeFeatureView",
            "name": "x",
            "target_lag_sec": 0,
        }
        out = normalize_durations(data)
        assert "target_lag_sec" not in out

    def test_batch_fv_normalize_durations_keeps_target_lag(self) -> None:
        from snowflake.ml.feature_store.decl.compiler import normalize_durations

        data: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "x",
            "target_lag": "1h",
        }
        out = normalize_durations(data)
        assert out.get("target_lag_sec") == 3600


# ---------------------------------------------------------------------------
# Streaming + feature_aggregation_method=continuous: compile-time
# feature_granularity_sec default = 60 (mirrors imperative-side
# ``_DEFAULT_CONTINUOUS_FEATURE_GRANULARITY`` = "1m").
# ---------------------------------------------------------------------------


class TestStreamingContinuousGranularityDefault:
    """A ``StreamingFeatureView`` with ``feature_aggregation_method:
    continuous`` and windowed features but no authored
    ``feature_granularity`` must compile to a wire spec that carries
    ``feature_granularity_sec: 60``.

    This mirrors snowml-core's
    ``feature_view._DEFAULT_CONTINUOUS_FEATURE_GRANULARITY = "1m"``
    (which kicks in when the imperative ``FeatureView`` constructor
    sees ``CONTINUOUS`` + ``features`` but no granularity), so the
    deployed SPECIFICATION JSON (which is post-defaulted by the
    runtime) and the local-compile spec stay byte-for-byte hash-equal
    on a clean export → re-plan cycle.

    Negative cases pin the carve-outs:

    * Explicit ``feature_granularity`` still wins (no clobbering).
    * ``feature_aggregation_method: tiles`` does NOT trigger the
      default — snowml-core rejects ``tiles`` without granularity,
      and surfacing the imperative error is the desired UX.
    * ``BatchFeatureView`` does NOT trigger the default —
      snowml-core's BFV constructor rejects
      ``feature_aggregation_method`` entirely, and the
      decl-side ``BATCH_FV_TILING_GRANULARITY`` invariant is the
      surfacing layer.
    """

    def _continuous_streaming_spec(self, **overrides: Any) -> Any:
        spec: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "name": "user_clicks_continuous",
            "version": "v1",
            "entities": ["user_id"],
            "sources": [
                {
                    "name": "user_events",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "user_id", "type": "StringType"},
                        {"name": "event_count", "type": "IntegerType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "event_count", "type": "IntegerType"},
                    "output_column": {"name": "event_count_1h", "type": "IntegerType"},
                    "function": "sum",
                    "window": "1h",
                }
            ],
            "timestamp_col": "timestamp",
            "feature_aggregation_method": "continuous",
            "refresh_freq": "5 minutes",
        }
        spec.update(overrides)
        return spec

    def test_streaming_continuous_defaults_granularity_to_60s(self) -> None:
        """Authoring SFV + ``continuous`` + windowed features without an
        explicit ``feature_granularity`` must produce a wire spec
        carrying ``feature_granularity_sec == 60``."""
        result = compile_to_spec(self._continuous_streaming_spec(), "DEMO_DB", "PUBLIC")
        assert result["spec"]["feature_granularity_sec"] == 60, (
            "CONTINUOUS streaming FV without authored feature_granularity must "
            "default to 60 seconds at compile time (mirrors imperative-side "
            "_DEFAULT_CONTINUOUS_FEATURE_GRANULARITY = '1m'); got "
            f"{result['spec'].get('feature_granularity_sec')!r}."
        )
        assert result["spec"]["feature_aggregation_method"] == "continuous"

    def test_streaming_continuous_explicit_granularity_wins(self) -> None:
        """An explicit ``feature_granularity`` (string or ``_sec``) must
        always win over the default — the defaulter only fills the
        ``None`` case."""
        result = compile_to_spec(
            self._continuous_streaming_spec(feature_granularity="5m"),
            "DEMO_DB",
            "PUBLIC",
        )
        assert (
            result["spec"]["feature_granularity_sec"] == 300
        ), "Explicit feature_granularity must override the CONTINUOUS default."

    def test_streaming_tiles_omits_granularity_when_unset(self) -> None:
        """``feature_aggregation_method: tiles`` (the default) without an
        explicit granularity must NOT trigger the CONTINUOUS default —
        snowml-core's ``FeatureView`` constructor raises
        ``features requires feature_granularity to be specified.`` for
        the tiles case, and surfacing that imperative error is the
        intended UX (the decl-side BFV validator already blocks the
        ``BatchFeatureView`` half of the same shape)."""
        spec = self._continuous_streaming_spec()
        spec["feature_aggregation_method"] = "tiles"
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        assert "feature_granularity_sec" not in result["spec"], (
            "tiles + no granularity must not be defaulted; got " f"{result['spec'].get('feature_granularity_sec')!r}."
        )

    def test_batch_continuous_does_not_default(self) -> None:
        """A ``BatchFeatureView`` carrying
        ``feature_aggregation_method: continuous`` must NOT receive the
        defaulted granularity at compile time.  snowml-core's
        ``FeatureView`` constructor explicitly rejects
        ``feature_aggregation_method`` on non-streaming FVs
        (``feature_view.py:974-977``), and the decl-side
        ``BATCH_FV_TILING_GRANULARITY`` invariant is the layer that
        surfaces a friendly error for the operator — silently
        defaulting here would mask both."""
        spec: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "bfv_continuous_misuse",
            "version": "v1",
            "online": False,
            "entities": ["user_id"],
            "sources": [
                {
                    "name": "user_events",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS",
                    "columns": [
                        {"name": "user_id", "type": "StringType"},
                        {"name": "event_count", "type": "IntegerType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "event_count", "type": "IntegerType"},
                    "output_column": {"name": "event_count_1h", "type": "IntegerType"},
                    "function": "sum",
                    "window": "1h",
                }
            ],
            "timestamp_col": "event_time",
            "feature_aggregation_method": "continuous",
            "refresh_freq": "5 minutes",
        }
        result = compile_to_spec(spec, "DEMO_DB", "PUBLIC")
        assert "feature_granularity_sec" not in result["spec"], (
            "BatchFeatureView + continuous must not be defaulted; the BFV "
            "invariant chain surfaces BATCH_FV_TILING_GRANULARITY instead."
        )


class TestRefreshFreqDrivesTargetLagSec:
    """``compile_to_spec`` reads the renamed ``refresh_freq`` authoring
    field and emits the imperative wire-form ``spec.target_lag_sec``.

    After the ``refresh_freq -> refresh_freq`` rename, the compiler's
    DT-cadence source must be ``refresh_freq`` (the canonical name),
    not the legacy ``refresh_freq``.  ``parse_duration_to_seconds``
    handles the human-friendly cadence string verbatim.

    Realtime / streaming kinds are out of scope here — the rejection
    validator on those kinds prevents ``refresh_freq`` from reaching
    the compiler at all.  These tests therefore cover BatchFeatureView
    only.
    """

    def _bfv_with_refresh_freq(self, refresh_freq: str) -> dict[str, Any]:
        return {
            "kind": "BatchFeatureView",
            "name": "BATCH_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "SRC",
                    "source_type": "Batch",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
            "refresh_freq": refresh_freq,
        }

    def test_compile_batch_fv_refresh_freq_string_drives_target_lag_sec(self) -> None:
        """``refresh_freq: "5 minutes"`` → ``spec.target_lag_sec: 300``."""
        result = compile_to_spec(self._bfv_with_refresh_freq("5 minutes"), "JKEW_DB", "JKEW_SCHEMA")
        assert result["spec"]["target_lag_sec"] == 300, (
            "compile_to_spec must source spec.target_lag_sec from the "
            "renamed authoring key 'refresh_freq' (not the legacy "
            "'refresh_freq'). Got "
            f"target_lag_sec={result['spec'].get('target_lag_sec')!r}."
        )

    def test_compile_batch_fv_refresh_freq_hours_drives_target_lag_sec(self) -> None:
        """``refresh_freq: "1 hour"`` → ``spec.target_lag_sec: 3600``."""
        result = compile_to_spec(self._bfv_with_refresh_freq("1 hour"), "JKEW_DB", "JKEW_SCHEMA")
        assert result["spec"]["target_lag_sec"] == 3600

    def test_compile_batch_fv_without_refresh_freq_omits_target_lag_sec(self) -> None:
        """When the BFV omits ``refresh_freq`` entirely, the compiler must
        not synthesise a ``target_lag_sec`` from any other field — the
        decoupled cadence contract requires the wire field to come from
        ``refresh_freq`` exclusively (the authoring ``target_lag`` is OFT
        staleness only).
        """
        bfv_no_cadence: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "name": "BATCH_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "SRC",
                    "source_type": "Batch",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
        }
        result = compile_to_spec(bfv_no_cadence, "JKEW_DB", "JKEW_SCHEMA")
        assert "target_lag_sec" not in result["spec"], (
            "A BFV without an authored cadence must not emit a "
            "spec.target_lag_sec value. Got "
            f"target_lag_sec={result['spec'].get('target_lag_sec')!r}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
