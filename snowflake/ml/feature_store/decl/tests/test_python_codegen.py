"""Tests for decl/python_codegen.py — Python source generation from spec dicts.

Covers spec_dict_to_python_source() for all six spec kinds:
Entity, BatchSource, StreamingSource, BatchFeatureView,
StreamingFeatureView, RealtimeFeatureView, FeatureGroup.

Each test class verifies:
1. The generated source is syntactically valid Python.
2. The generated file is loadable by loader.load_python_file().
3. The loaded Pydantic object has the expected kind/name/version.
4. Round-trip fields (join_keys, columns, features, udf, feature_views) match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.loader import load_python_file
from snowflake.ml.feature_store.decl.python_codegen import spec_dict_to_python_source
from snowflake.ml.feature_store.decl.spec_models import (
    BatchFeatureView,
    BatchSource,
    Entity,
    FeatureGroup,
    StreamingFeatureView,
    StreamingSource,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_and_load(tmp_path: Path, filename: str, source: str) -> list[tuple[str, Any]]:
    """Write *source* to *tmp_path/<filename>* and load via load_python_file."""
    py_file = tmp_path / filename
    py_file.write_text(source)
    return load_python_file(str(py_file))


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class TestEntityCodegen:
    def test_single_join_key(self, tmp_path: Path) -> None:
        spec = {
            "kind": "Entity",
            "name": "USER_ID",
            "join_keys": [{"name": "USER_ID", "type": "StringType"}],
        }
        source = spec_dict_to_python_source("Entity", "user_id", spec)
        results = _write_and_load(tmp_path, "user_id.py", source)

        assert len(results) == 1
        _, obj = results[0]
        assert isinstance(obj, Entity)
        assert obj.name == "USER_ID"
        assert len(obj.join_keys) == 1
        assert obj.join_keys[0].name == "USER_ID"
        assert obj.join_keys[0].type == "StringType"

    def test_multiple_join_keys(self, tmp_path: Path) -> None:
        spec = {
            "kind": "Entity",
            "name": "USER_ITEM",
            "join_keys": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "ITEM_ID", "type": "IntegerType"},
            ],
        }
        source = spec_dict_to_python_source("Entity", "user_item", spec)
        results = _write_and_load(tmp_path, "user_item.py", source)

        _, obj = results[0]
        assert isinstance(obj, Entity)
        assert len(obj.join_keys) == 2
        assert obj.join_keys[1].name == "ITEM_ID"
        assert obj.join_keys[1].type == "IntegerType"

    def test_with_description(self, tmp_path: Path) -> None:
        spec = {
            "kind": "Entity",
            "name": "CUSTOMER",
            "join_keys": [{"name": "CUSTOMER_ID", "type": "StringType"}],
            "description": "Primary customer entity.",
        }
        source = spec_dict_to_python_source("Entity", "customer", spec)
        results = _write_and_load(tmp_path, "customer.py", source)

        _, obj = results[0]
        assert obj.description == "Primary customer entity."

    def test_no_description_not_emitted(self, tmp_path: Path) -> None:
        spec = {
            "kind": "Entity",
            "name": "ANON",
            "join_keys": [{"name": "ID", "type": "StringType"}],
        }
        source = spec_dict_to_python_source("Entity", "anon", spec)
        assert "description" not in source


# ---------------------------------------------------------------------------
# BatchSource
# ---------------------------------------------------------------------------


class TestBatchSourceCodegen:
    def test_table_backed(self, tmp_path: Path) -> None:
        spec = {
            "kind": "BatchSource",
            "name": "EVENTS_BATCH",
            "table": "RAW_EVENTS",
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
            ],
        }
        source = spec_dict_to_python_source("BatchSource", "events_batch", spec)
        results = _write_and_load(tmp_path, "events_batch.py", source)

        _, obj = results[0]
        assert isinstance(obj, BatchSource)
        assert obj.name == "EVENTS_BATCH"
        assert obj.table == "RAW_EVENTS"
        assert len(obj.columns) == 2
        assert obj.columns[0].name == "USER_ID"

    def test_query_backed(self, tmp_path: Path) -> None:
        spec = {
            "kind": "BatchSource",
            "name": "EVENTS_SQL",
            "query": "SELECT * FROM RAW WHERE dt > '2024-01-01'",
            "columns": [{"name": "USER_ID", "type": "StringType"}],
        }
        source = spec_dict_to_python_source("BatchSource", "events_sql", spec)
        results = _write_and_load(tmp_path, "events_sql.py", source)

        _, obj = results[0]
        assert isinstance(obj, BatchSource)
        assert "SELECT" in (obj.query or "")

    def test_no_columns(self, tmp_path: Path) -> None:
        spec = {
            "kind": "BatchSource",
            "name": "SIMPLE",
            "table": "MY_TABLE",
        }
        source = spec_dict_to_python_source("BatchSource", "simple", spec)
        results = _write_and_load(tmp_path, "simple.py", source)

        _, obj = results[0]
        assert isinstance(obj, BatchSource)
        assert obj.name == "SIMPLE"

    def test_with_source_database_schema(self, tmp_path: Path) -> None:
        spec = {
            "kind": "BatchSource",
            "name": "CROSS_DB",
            "table": "MY_TABLE",
            "source_database": "OTHER_DB",
            "source_schema": "OTHER_SCHEMA",
            "columns": [],
        }
        source = spec_dict_to_python_source("BatchSource", "cross_db", spec)
        results = _write_and_load(tmp_path, "cross_db.py", source)

        _, obj = results[0]
        assert obj.source_database == "OTHER_DB"
        assert obj.source_schema == "OTHER_SCHEMA"


# ---------------------------------------------------------------------------
# StreamingSource
# ---------------------------------------------------------------------------


class TestStreamingSourceCodegen:
    def test_rest_type_with_columns(self, tmp_path: Path) -> None:
        spec = {
            "kind": "StreamingSource",
            "name": "CLICK_STREAM",
            "type": "REST",
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
            ],
        }
        source = spec_dict_to_python_source("StreamingSource", "click_stream", spec)
        results = _write_and_load(tmp_path, "click_stream.py", source)

        _, obj = results[0]
        assert isinstance(obj, StreamingSource)
        assert obj.name == "CLICK_STREAM"
        assert obj.type == "REST"
        assert len(obj.columns) == 2

    def test_no_columns(self, tmp_path: Path) -> None:
        spec = {
            "kind": "StreamingSource",
            "name": "STREAM_BARE",
            "type": "REST",
        }
        source = spec_dict_to_python_source("StreamingSource", "stream_bare", spec)
        results = _write_and_load(tmp_path, "stream_bare.py", source)

        _, obj = results[0]
        assert isinstance(obj, StreamingSource)


# ---------------------------------------------------------------------------
# BatchFeatureView
# ---------------------------------------------------------------------------


_BATCH_SPEC: dict[str, Any] = {
    "kind": "BatchFeatureView",
    "name": "MY_BATCH_FV",
    "version": "V1",
    "online": True,
    "entities": ["USER_ID"],
    "sources": [{"name": "EVENTS_BATCH", "source_type": "Batch"}],
    "timestamp_col": "EVENT_TS",
    "refresh_freq": "5 minutes",
    "features": [
        {
            "source_column": {"name": "METRIC_VAL", "type": "FloatType"},
            "output_column": {"name": "SUM_METRIC", "type": "FloatType"},
            "function": "sum",
            "window_sec": 3600,
        }
    ],
}


class TestBatchFVCodegen:
    def test_minimal_bfv_loadable(self, tmp_path: Path) -> None:
        spec = {
            "kind": "BatchFeatureView",
            "name": "MINIMAL_BFV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [{"name": "DS", "source_type": "Batch"}],
            "offline": True,
        }
        source = spec_dict_to_python_source("BatchFeatureView", "minimal_bfv", spec)
        results = _write_and_load(tmp_path, "minimal_bfv.py", source)

        _, obj = results[0]
        assert isinstance(obj, BatchFeatureView)
        assert obj.name == "MINIMAL_BFV"
        assert obj.version == "V1"

    def test_full_bfv_loadable(self, tmp_path: Path) -> None:
        source = spec_dict_to_python_source("BatchFeatureView", "my_batch_fv", _BATCH_SPEC)
        results = _write_and_load(tmp_path, "my_batch_fv.py", source)

        _, obj = results[0]
        assert isinstance(obj, BatchFeatureView)
        assert obj.name == "MY_BATCH_FV"
        assert obj.version == "V1"
        assert obj.online is True
        assert "USER_ID" in obj.entities
        assert obj.timestamp_col == "EVENT_TS"
        assert obj.refresh_freq == "5 minutes"

    def test_bfv_features_preserved(self, tmp_path: Path) -> None:
        source = spec_dict_to_python_source("BatchFeatureView", "my_batch_fv", _BATCH_SPEC)
        results = _write_and_load(tmp_path, "my_batch_fv.py", source)

        _, obj = results[0]
        assert len(obj.features) == 1
        feat = obj.features[0]
        assert feat.function == "sum"
        assert feat.window_sec == 3600

    def test_bfv_advanced_fields(self, tmp_path: Path) -> None:
        spec = dict(_BATCH_SPEC)
        spec["cluster_by"] = ["USER_ID"]
        spec["refresh_mode"] = "INCREMENTAL"
        spec["initialize"] = "ON_SCHEDULE"
        source = spec_dict_to_python_source("BatchFeatureView", "adv_bfv", spec)
        results = _write_and_load(tmp_path, "adv_bfv.py", source)

        _, obj = results[0]
        assert obj.cluster_by == ["USER_ID"]
        assert obj.refresh_mode == "INCREMENTAL"
        assert obj.initialize == "ON_SCHEDULE"

    def test_bfv_append_only(self, tmp_path: Path) -> None:
        """An append-only passthrough BFV round-trips through Python codegen."""
        spec = {
            "kind": "BatchFeatureView",
            "name": "APPEND_BFV",
            "version": "V1",
            "online": False,
            "entities": ["USER_ID"],
            "sources": [{"name": "DS", "source_type": "Batch"}],
            "timestamp_col": "EVENT_TS",
            "refresh_mode": "FULL",
            "refresh_freq": "0 0 * * * UTC",
            "append_only": True,
        }
        source = spec_dict_to_python_source("BatchFeatureView", "append_bfv", spec)
        assert "append_only=True" in source
        results = _write_and_load(tmp_path, "append_bfv.py", source)
        _, obj = results[0]
        assert obj.append_only is True
        assert obj.refresh_mode == "FULL"

    def test_bfv_append_only_omitted_when_absent(self, tmp_path: Path) -> None:
        """A plain BFV never emits ``append_only`` in the generated Python."""
        source = spec_dict_to_python_source("BatchFeatureView", "my_batch_fv", _BATCH_SPEC)
        assert "append_only" not in source

    def test_bfv_with_udf_emits_def_block(self, tmp_path: Path) -> None:
        spec = dict(_BATCH_SPEC)
        spec["udf"] = {
            "name": "my_transform",
            "engine": "pandas",
            "function_definition": "def my_transform(df):\n    return df\n",
            "output_columns": [{"name": "OUT", "type": "FloatType"}],
        }
        source = spec_dict_to_python_source("BatchFeatureView", "udf_bfv", spec)

        # The def block must appear before the variable assignment
        def_pos = source.find("def my_transform")
        assign_pos = source.find("udf_bfv =")
        assert def_pos != -1, "def block not found in generated source"
        assert assign_pos != -1, "variable assignment not found in generated source"
        assert def_pos < assign_pos, "def block must precede the variable assignment"

        # The UDF constructor must reference the callable by name, not inline source
        assert "function_definition=my_transform" in source.replace(" ", "").replace("\n", "")

    def test_bfv_with_udf_loadable(self, tmp_path: Path) -> None:
        spec = dict(_BATCH_SPEC)
        spec["udf"] = {
            "name": "my_transform",
            "engine": "pandas",
            "function_definition": "def my_transform(df):\n    return df\n",
            "output_columns": [{"name": "OUT", "type": "FloatType"}],
        }
        source = spec_dict_to_python_source("BatchFeatureView", "udf_bfv", spec)
        results = _write_and_load(tmp_path, "udf_bfv.py", source)

        _, obj = results[0]
        assert isinstance(obj, BatchFeatureView)
        assert obj.udf is not None
        assert obj.udf.name == "my_transform"
        # function_definition is a callable after load
        assert callable(obj.udf.function_definition)

    def test_bfv_online_false_not_emitted_when_false(self, tmp_path: Path) -> None:
        spec = {
            "kind": "BatchFeatureView",
            "name": "OFFLINE_BFV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [{"name": "DS", "source_type": "Batch"}],
        }
        source = spec_dict_to_python_source("BatchFeatureView", "offline_bfv", spec)
        # online=False should not appear (matches spec_to_dict convention)
        assert "online=False" not in source


# ---------------------------------------------------------------------------
# StreamingFeatureView
# ---------------------------------------------------------------------------


class TestStreamingFVCodegen:
    def test_streaming_fv_loadable(self, tmp_path: Path) -> None:
        spec = {
            "kind": "StreamingFeatureView",
            "name": "USER_CLICKS",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [{"name": "CLICK_STREAM", "source_type": "Stream"}],
            "timestamp_col": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "tiles",
            "features": [
                {
                    "source_column": {"name": "EVENT", "type": "StringType"},
                    "output_column": {"name": "EVENT_COUNT", "type": "IntegerType"},
                    "function": "count",
                    "window_sec": 3600,
                }
            ],
        }
        source = spec_dict_to_python_source("StreamingFeatureView", "user_clicks", spec)
        results = _write_and_load(tmp_path, "user_clicks.py", source)

        _, obj = results[0]
        assert isinstance(obj, StreamingFeatureView)
        assert obj.name == "USER_CLICKS"
        assert obj.feature_granularity_sec == 300

    def test_streaming_fv_udf_inline_def(self, tmp_path: Path) -> None:
        udf_source = "def compute_metrics(df):\n" "    df['SCORE'] = df['EVENT'].apply(len)\n" "    return df\n"
        spec = {
            "kind": "StreamingFeatureView",
            "name": "SCORED",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [{"name": "EVENTS", "source_type": "Stream"}],
            "udf": {
                "name": "compute_metrics",
                "engine": "pandas",
                "function_definition": udf_source,
                "output_columns": [{"name": "SCORE", "type": "FloatType"}],
            },
        }
        source = spec_dict_to_python_source("StreamingFeatureView", "scored", spec)

        # def block inline
        assert "def compute_metrics" in source
        # no .py sidecar reference
        assert "file=" not in source
        # callable reference in UDF constructor
        assert "compute_metrics" in source

        results = _write_and_load(tmp_path, "scored.py", source)
        _, obj = results[0]
        assert isinstance(obj, StreamingFeatureView)
        assert obj.udf is not None
        assert callable(obj.udf.function_definition)


# ---------------------------------------------------------------------------
# FeatureGroup
# ---------------------------------------------------------------------------


class TestFeatureGroupCodegen:
    def test_fg_basic_loadable(self, tmp_path: Path) -> None:
        spec = {
            "kind": "FeatureGroup",
            "name": "USER_FG",
            "version": "V1",
            "desc": "User feature group.",
            "auto_prefix": True,
            "feature_views": [
                {"name": "MY_BATCH_FV", "version": "V1"},
            ],
        }
        source = spec_dict_to_python_source("FeatureGroup", "user_fg", spec)
        results = _write_and_load(tmp_path, "user_fg.py", source)

        _, obj = results[0]
        assert isinstance(obj, FeatureGroup)
        assert obj.name == "USER_FG"
        assert obj.version == "V1"
        assert obj.desc == "User feature group."
        assert len(obj.feature_views) == 1
        assert obj.feature_views[0].name == "MY_BATCH_FV"
        assert obj.feature_views[0].version == "V1"

    def test_fg_slice_columns_preserved(self, tmp_path: Path) -> None:
        spec = {
            "kind": "FeatureGroup",
            "name": "SLICED_FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [
                {
                    "name": "MY_BFV",
                    "version": "V1",
                    "slice_columns": ["COL_A", "COL_B"],
                }
            ],
        }
        source = spec_dict_to_python_source("FeatureGroup", "sliced_fg", spec)
        results = _write_and_load(tmp_path, "sliced_fg.py", source)

        _, obj = results[0]
        ref = obj.feature_views[0]
        assert ref.slice_columns == ["COL_A", "COL_B"]

    def test_fg_alias_preserved(self, tmp_path: Path) -> None:
        spec = {
            "kind": "FeatureGroup",
            "name": "ALIAS_FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [
                {
                    "name": "MY_BFV",
                    "version": "V1",
                    "alias": "clicks",
                }
            ],
        }
        source = spec_dict_to_python_source("FeatureGroup", "alias_fg", spec)
        results = _write_and_load(tmp_path, "alias_fg.py", source)

        _, obj = results[0]
        assert obj.feature_views[0].alias == "clicks"

    def test_fg_empty_alias_preserved(self, tmp_path: Path) -> None:
        """alias="" is semantically distinct from alias=None — must survive round-trip."""
        spec = {
            "kind": "FeatureGroup",
            "name": "EMPTY_ALIAS_FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [
                {
                    "name": "MY_BFV",
                    "version": "V1",
                    "alias": "",
                }
            ],
        }
        source = spec_dict_to_python_source("FeatureGroup", "empty_alias_fg", spec)
        results = _write_and_load(tmp_path, "empty_alias_fg.py", source)

        _, obj = results[0]
        assert obj.feature_views[0].alias == ""

    def test_fg_multiple_views(self, tmp_path: Path) -> None:
        spec = {
            "kind": "FeatureGroup",
            "name": "MULTI_FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": False,
            "feature_views": [
                {"name": "FV_A", "version": "V1"},
                {"name": "FV_B", "version": "V2"},
            ],
        }
        source = spec_dict_to_python_source("FeatureGroup", "multi_fg", spec)
        results = _write_and_load(tmp_path, "multi_fg.py", source)

        _, obj = results[0]
        assert len(obj.feature_views) == 2
        assert obj.auto_prefix is False


# ---------------------------------------------------------------------------
# Unsupported kind
# ---------------------------------------------------------------------------


class TestUnsupportedKind:
    def test_raises_on_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            spec_dict_to_python_source("WeirdThing", "x", {"kind": "WeirdThing", "name": "X"})


# ---------------------------------------------------------------------------
# Generated source smoke tests
# ---------------------------------------------------------------------------


class TestGeneratedSourceSyntax:
    """Verify generated source is syntactically valid Python via compile()."""

    def _assert_compilable(self, source: str) -> None:
        try:
            compile(source, "<generated>", "exec")
        except SyntaxError as exc:
            pytest.fail(f"Generated source has syntax error: {exc}\n\nSource:\n{source}")

    def test_entity_compilable(self) -> None:
        spec = {"kind": "Entity", "name": "E", "join_keys": [{"name": "E_ID", "type": "StringType"}]}
        self._assert_compilable(spec_dict_to_python_source("Entity", "e", spec))

    def test_batch_source_compilable(self) -> None:
        spec = {"kind": "BatchSource", "name": "BS", "table": "T", "columns": []}
        self._assert_compilable(spec_dict_to_python_source("BatchSource", "bs", spec))

    def test_streaming_source_compilable(self) -> None:
        spec = {"kind": "StreamingSource", "name": "SS", "type": "REST", "columns": []}
        self._assert_compilable(spec_dict_to_python_source("StreamingSource", "ss", spec))

    def test_batch_fv_compilable(self) -> None:
        self._assert_compilable(spec_dict_to_python_source("BatchFeatureView", "bfv", _BATCH_SPEC))

    def test_feature_group_compilable(self) -> None:
        spec = {
            "kind": "FeatureGroup",
            "name": "FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [{"name": "FV", "version": "V1"}],
        }
        self._assert_compilable(spec_dict_to_python_source("FeatureGroup", "fg", spec))


class TestPythonCodegenCleanups:
    """P12: python_codegen cleanups — UDF fallback, conditional imports, indentation."""

    def test_udf_source_without_def_raises(self) -> None:
        # A UDF body with no ``def`` is unrecoverable; emitting a module that
        # references an undefined name (``function_definition=_udf_func``) is the
        # wrong response — fail loudly instead.
        with pytest.raises(ValueError):
            spec_dict_to_python_source(
                "StreamingFeatureView",
                "FV",
                {"name": "FV", "udf": {"name": "u", "function_definition": "x = 1"}},
            )

    @pytest.mark.parametrize(
        "udf",
        [
            {"name": "u"},
            {"name": "u", "function_definition": ""},
        ],
        ids=["missing_function_definition", "empty_function_definition"],
    )
    def test_udf_without_function_definition_raises(self, udf: dict[str, str]) -> None:
        # YAML's ``_extract_udf_to_py_file`` degrades when the body is missing;
        # the Python renderer must not write ``function_definition=,``.
        with pytest.raises(ValueError, match="no function_definition"):
            spec_dict_to_python_source(
                "StreamingFeatureView",
                "FV",
                {"name": "FV", "udf": udf},
            )

    def test_entity_without_join_keys_omits_fscolumn_import(self) -> None:
        src = spec_dict_to_python_source("Entity", "e", {"name": "E"})
        import_line = next(line for line in src.splitlines() if line.startswith("from snowflake"))
        assert "FSColumn" not in import_line, f"unused FSColumn import in {import_line!r}"

    def test_udf_output_columns_block_is_correctly_indented(self) -> None:
        spec = {
            "name": "FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "timestamp_col": "TS",
            "udf": {
                "name": "compute",
                "engine": "pandas",
                "function_definition": "def compute(x):\n    return x",
                "output_columns": [{"name": "USER_ID", "type": "StringType"}],
            },
        }
        src = spec_dict_to_python_source("StreamingFeatureView", "FV", spec)
        # ``output_columns=[`` sits at column 8 (a UDF(...) constructor arg), so
        # the nested FSColumn items must be at 12 and the closing ``]`` at 8.
        assert "\n            FSColumn(name='USER_ID', type='StringType'),\n        ]" in src
        compile(src, "<generated>", "exec")


if __name__ == "__main__":
    pytest_driver.main()
