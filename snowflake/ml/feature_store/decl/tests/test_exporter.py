"""Tests for decl/exporter.py — strict full-fidelity export from spec JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers — shared test data
# ---------------------------------------------------------------------------

_SHOW_ROW_1: dict[str, Any] = {
    "name": "USER_CLICKS$V1$ONLINE",
    "database_name": "MYDB",
    "schema_name": "PUBLIC",
    "scheduling_state": "ACTIVE",
}


_FULL_SPEC: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "MYDB",
        "schema": "PUBLIC",
        "name": "user_clicks",
        "version": "v1",
        "spec_format_version": "1",
        "internal_data_version": "1",
        "client_version": "0.1.0",
    },
    "offline_configs": [
        {
            "store_type": "snowflake",
            "table_type": "UDFTransformed",
            "database": "MYDB",
            "schema": "PUBLIC",
            "table": "USER_CLICKS$V1$UDF_TRANSFORMED",
            "columns": [{"name": "event_count", "type": "IntegerType"}],
        }
    ],
    "spec": {
        "ordered_entity_column_names": ["user_id"],
        "sources": [
            {
                "name": "user_events",
                "source_type": "Stream",
                "columns": [
                    {"name": "user_id", "type": "StringType"},
                    {"name": "event", "type": "StringType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "event", "type": "StringType"},
                "output_column": {"name": "event_count", "type": "IntegerType"},
                "function": "count",
                "window_sec": 3600,
                "offset_sec": 0,
            }
        ],
        "udf": {
            "name": "transform",
            "function_definition": "def transform(x):\n    return len(x)",
            "engine": "python",
            "output_columns": [{"name": "event_count", "type": "IntegerType"}],
        },
        "timestamp_field": "event_time",
        "feature_granularity_sec": 60,
        "feature_aggregation_method": "tiles",
    },
    "online_store_type": "postgres",
}


_REALTIME_SPEC: dict[str, Any] = {
    "kind": "RealtimeFeatureView",
    "metadata": {
        "database": "MYDB",
        "schema": "PUBLIC",
        "name": "user_realtime",
        "version": "v2",
    },
    "offline_configs": [],
    "spec": {
        "ordered_entity_column_names": ["session_id"],
        "sources": [
            {
                "name": "session_events",
                "source_type": "Stream",
                "columns": [{"name": "session_id", "type": "StringType"}],
            }
        ],
        "features": [
            {
                "source_column": {"name": "session_id", "type": "StringType"},
                "output_column": {"name": "session_count", "type": "IntegerType"},
                "function": "count",
            }
        ],
    },
    "online_store_type": "postgres",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportSpecsEmptyInput:
    def test_empty_show_rows_returns_early(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
        )
        assert result["status"] == "exported"
        assert result["files"] == []
        assert result["directory"] == ""


class TestExportSpecsFullFidelity:
    def test_full_spec_yaml_udf_uses_file_reference_not_inline_source(self, tmp_path: Path) -> None:
        """UDF source must be written to a sibling .py file; YAML carries ``file:`` ref only."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        assert fv_path.exists()
        data = yaml.safe_load(fv_path.read_text())

        # UDF metadata is preserved, but inline source is replaced by a file reference.
        # The live SPECIFICATION JSON already uses the authoring-shape keys
        # ``name`` / ``engine`` (verified against captured fixtures in
        # ``tests/golden_specs/``), so the exporter passes them through as-is
        # — no rename is performed.  ``function_definition`` is extracted into
        # a sibling ``.py`` file and replaced with ``file:`` for the YAML.
        assert "udf" in data
        assert data["udf"]["name"] == "transform"
        assert data["udf"]["engine"] == "python"
        assert "function_name" not in data["udf"]
        assert "language" not in data["udf"]
        assert data["udf"]["output_columns"] == [{"name": "event_count", "type": "IntegerType"}]
        assert "function_definition" not in data["udf"]
        assert data["udf"]["file"] == "user_clicks_v1.py"

    def test_full_spec_writes_udf_py_file_with_original_source(self, tmp_path: Path) -> None:
        """The sibling .py file must contain the original ``function_definition`` source."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        py_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.py"
        assert py_path.exists()
        assert py_path.read_text() == "def transform(x):\n    return len(x)"

    def test_full_spec_udf_py_file_appears_in_returned_files_list(self, tmp_path: Path) -> None:
        """``files`` must include the sibling .py file alongside the YAML."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        py_path = str(tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.py")
        assert py_path in result["files"]

    def test_full_spec_udf_file_path_is_relative_to_yaml(self, tmp_path: Path) -> None:
        """``udf.file`` must be a bare filename (relative to the YAML), not absolute."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        ref = data["udf"]["file"]
        # Must be a bare filename, not absolute and not navigating up.
        assert "/" not in ref
        assert "\\" not in ref
        assert not ref.startswith("/")
        assert ref == "user_clicks_v1.py"

    def test_no_udf_py_file_when_function_definition_missing(self, tmp_path: Path) -> None:
        """When the spec carries no UDF source, no .py file is written and ``file:`` is omitted."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec_no_source = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": "MYDB",
                "schema": "PUBLIC",
                "name": "user_clicks",
                "version": "v1",
            },
            "spec": {
                "ordered_entity_column_names": ["user_id"],
                "udf": {
                    "name": "transform",
                    "engine": "python",
                    "output_columns": [{"name": "event_count", "type": "IntegerType"}],
                },
            },
        }
        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": spec_no_source},
        )
        fv_dir = tmp_path / "MYDB.PUBLIC" / "feature_views"
        assert (fv_dir / "user_clicks_v1.yaml").exists()
        assert not (fv_dir / "user_clicks_v1.py").exists()

        data = yaml.safe_load((fv_dir / "user_clicks_v1.yaml").read_text())
        assert "udf" in data
        assert "file" not in data["udf"]
        assert "function_definition" not in data["udf"]

    def test_no_udf_py_file_when_function_definition_empty(self, tmp_path: Path) -> None:
        """An empty ``function_definition`` is treated as missing; no .py is written."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec_empty_source = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": "MYDB",
                "schema": "PUBLIC",
                "name": "user_clicks",
                "version": "v1",
            },
            "spec": {
                "udf": {
                    "name": "transform",
                    "function_definition": "",
                    "engine": "python",
                    "output_columns": [{"name": "x", "type": "IntegerType"}],
                },
            },
        }
        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": spec_empty_source},
        )
        fv_dir = tmp_path / "MYDB.PUBLIC" / "feature_views"
        assert not (fv_dir / "user_clicks_v1.py").exists()

    def test_round_trip_inline_udf_source_resolves_exported_file(self, tmp_path: Path) -> None:
        """``compiler.inline_udf_source`` must resolve the exporter's ``file:`` reference."""
        from snowflake.ml.feature_store.decl.compiler import inline_udf_source
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_dir = tmp_path / "MYDB.PUBLIC" / "feature_views"
        fv_path = fv_dir / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())

        # Loader resolves ``udf.file`` against the YAML's directory.
        compiled = inline_udf_source(data, str(fv_dir))
        assert compiled["udf"]["function_definition"] == "def transform(x):\n    return len(x)"
        assert "file" not in compiled["udf"]

    def test_full_spec_yaml_has_sources_with_columns(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "sources" in data
        assert len(data["sources"]) == 1
        src = data["sources"][0]
        assert src["name"] == "user_events"
        assert src["source_type"] == "Stream"
        assert {"name": "user_id", "type": "StringType"} in src["columns"]
        assert {"name": "event", "type": "StringType"} in src["columns"]

    def test_full_spec_yaml_has_features_with_aggregation(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "features" in data
        assert len(data["features"]) == 1
        feat = data["features"][0]
        assert feat["function"] == "count"
        assert feat["window_sec"] == 3600
        assert feat["offset_sec"] == 0
        assert feat["source_column"] == {"name": "event", "type": "StringType"}
        assert feat["output_column"] == {"name": "event_count", "type": "IntegerType"}

    def test_full_spec_yaml_has_metadata_fields(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data["kind"] == "StreamingFeatureView"
        assert data["name"] == "user_clicks"
        assert data["version"] == "v1"
        assert data["database"] == "MYDB"
        assert data["schema"] == "PUBLIC"
        # ``online`` is intentionally omitted for streaming and
        # realtime kinds — the implicit-online contract on
        # ``spec_models.FeatureView`` defaults it to ``True`` and
        # rejects an explicit ``False``, so writing it here would be
        # redundant noise.  See
        # ``TestExportSpecsOmitsOnlineForStreamingAndRealtime`` below.
        assert "online" not in data
        # ``scheduling_state`` is server-stamped runtime metadata
        # surfaced via ``decl_api.enrich_list_results`` (``details.``
        # block on the ``snow feature list`` row) — it is NOT part of
        # the authoring surface, so the exporter must not emit it
        # even though the SHOW row carries a value.  See
        # ``TestExportSpecsOmitsSchedulingState`` below for the
        # per-kind pins.
        assert "scheduling_state" not in data
        assert data["entities"] == ["user_id"]
        assert data["timestamp_col"] == "event_time"
        assert data["feature_granularity_sec"] == 60
        assert data["feature_aggregation_method"] == "tiles"

    def test_full_spec_writes_entity_files_from_join_keys(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        # ``entity_rows`` is the authoritative source for entity YAML
        # emission; the test passes an entry matching the FV's join key
        # to pin the post-emission shape (Entity ``kind`` + name).
        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
        )
        from pathlib import Path

        entity_files = [f for f in result["files"] if Path(f).parent.name == "entities"]
        assert len(entity_files) == 1
        entity_path = tmp_path / "MYDB.PUBLIC" / "entities" / "USER_ID.yaml"
        assert entity_path.exists()
        ent_data = yaml.safe_load(entity_path.read_text())
        assert ent_data["kind"] == "Entity"
        assert ent_data["name"] == "USER_ID"

    def test_kind_preserved_for_realtime_feature_view(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        show_row = {
            "name": "USER_REALTIME$V2$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_REALTIME$V2$ONLINE": _REALTIME_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_realtime_v2.yaml"
        assert fv_path.exists()
        data = yaml.safe_load(fv_path.read_text())
        assert data["kind"] == "RealtimeFeatureView"
        assert data["name"] == "user_realtime"
        assert data["version"] == "v2"

    def test_dedupes_entities_across_two_full_specs(self, tmp_path: Path) -> None:
        """Two FVs sharing a join key must yield exactly one Entity YAML."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        other_spec = dict(_FULL_SPEC)
        other_spec["metadata"] = dict(_FULL_SPEC["metadata"])
        other_spec["metadata"]["name"] = "user_other"
        other_spec["metadata"]["version"] = "v1"

        show_rows = [
            _SHOW_ROW_1,
            {
                "name": "USER_OTHER$V1$ONLINE",
                "database_name": "MYDB",
                "schema_name": "PUBLIC",
                "scheduling_state": "ACTIVE",
            },
        ]
        result = export_specs(
            show_rows=show_rows,
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={
                "USER_CLICKS$V1$ONLINE": _FULL_SPEC,
                "USER_OTHER$V1$ONLINE": other_spec,
            },
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
        )
        from pathlib import Path

        entity_files = [f for f in result["files"] if Path(f).parent.name == "entities"]
        assert len(entity_files) == 1


class TestExportSpecsOmitsTargetLagForStreamingAndRealtime:
    """Pin :func:`export_specs` omits ``target_lag`` / ``target_lag_sec``
    from exported streaming / realtime FV YAML.

    The Snowflake runtime stamps ``target_lag_sec: 0`` onto the deployed
    SPECIFICATION for streaming and realtime FVs regardless of the
    authored value (the
    :data:`snowflake.ml.feature_store.decl.invariants._RUNTIME_STAMPED_SPEC_KEYS`
    set strips it from hashes for the same reason).  When ``snow feature
    init`` runs ``export_specs`` against a live deployment, the
    described ``spec.target_lag_sec`` carries the stamped ``0`` — and
    if the exporter wrote that ``0`` into the freshly-emitted YAML, a
    subsequent ``snow feature plan`` would fail to load it because the
    Pydantic-layer validator
    (``FeatureView._reject_target_lag_on_stream_or_realtime``) rejects
    any presence of the key on streaming / realtime kinds.

    The exporter's defence is to omit both keys from the YAML for
    streaming and realtime kinds — keeping the ``snow feature init``
    → ``snow feature plan`` round-trip clean.  Batch FVs are
    unaffected: the exporter must continue to write
    ``target_lag_sec`` for ``BatchFeatureView`` payloads (the
    declarative validator accepts the field there, and the compiled
    spec carries it).
    """

    def _streaming_spec_with_target_lag(self, target_lag_sec: int) -> dict[str, Any]:
        import copy

        spec = copy.deepcopy(_FULL_SPEC)
        spec["spec"]["target_lag_sec"] = target_lag_sec
        return spec

    def test_exported_streaming_fv_yaml_omits_target_lag_sec_zero(self, tmp_path: Path) -> None:
        # Runtime-stamped ``target_lag_sec: 0`` on a streaming FV is
        # the common case (``snow feature init`` over a live
        # deployment); the exporter must drop it so the YAML re-loads
        # cleanly through the Pydantic validator.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": self._streaming_spec_with_target_lag(0)},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "target_lag_sec" not in data
        assert "target_lag" not in data

    def test_exported_streaming_fv_yaml_omits_target_lag_sec_nonzero(self, tmp_path: Path) -> None:
        # A hand-stamped non-zero ``target_lag_sec`` on a streaming FV
        # (shouldn't happen in production, but defence-in-depth) is
        # also stripped from the exported YAML.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": self._streaming_spec_with_target_lag(3600)},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "target_lag_sec" not in data
        assert "target_lag" not in data

    def test_exported_realtime_fv_yaml_omits_target_lag_sec(self, tmp_path: Path) -> None:
        # Same contract for ``RealtimeFeatureView`` — the runtime
        # treats realtime FVs identically to streaming FVs for
        # ``target_lag_sec``.
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        realtime_spec = copy.deepcopy(_REALTIME_SPEC)
        realtime_spec["spec"]["target_lag_sec"] = 0
        show_row = {
            "name": "USER_REALTIME$V2$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_REALTIME$V2$ONLINE": realtime_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_realtime_v2.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "target_lag_sec" not in data
        assert "target_lag" not in data

    def test_exported_batch_fv_yaml_emits_refresh_freq_for_dt_refresh(self, tmp_path: Path) -> None:
        # Regression pin: ``BatchFeatureView`` carries the offline DT
        # refresh under the authoring-form key ``refresh_freq`` in
        # the exported YAML.  The wire-form ``target_lag_sec`` (DT
        # refresh in seconds) is converted to ``"<n> seconds"`` and
        # written under ``refresh_freq`` after the
        # ``feature_granularity`` / ``refresh_freq`` / ``target_lag``
        # decoupling — that's the canonical authoring shape.  The
        # strip of streaming / realtime continues to apply.
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        batch_spec = copy.deepcopy(_FULL_SPEC)
        batch_spec["kind"] = "BatchFeatureView"
        # Batch FVs require a Batch-typed source — flip the source
        # type so the exporter doesn't surface a kind/source mismatch
        # (it doesn't, but keep the fixture coherent for the validator
        # round-trip).
        batch_spec["spec"]["sources"][0]["source_type"] = "Batch"
        batch_spec["spec"]["target_lag_sec"] = 3600
        show_row = {
            "name": "USER_CLICKS$V1$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": batch_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data.get("refresh_freq") == "3600 seconds"
        assert "target_lag_sec" not in data

    def _tiled_streaming_spec_with_refresh_freq(self, refresh_freq: str) -> dict[str, Any]:
        import copy

        spec = copy.deepcopy(_FULL_SPEC)
        # Runtime always stamps the OFT lag to 0; the offline tile DT
        # cadence rides on refresh_freq (recovered by state.py).
        spec["spec"]["target_lag_sec"] = 0
        spec["spec"]["refresh_freq"] = refresh_freq
        return spec

    def test_exported_tiled_streaming_fv_yaml_emits_refresh_freq(self, tmp_path: Path) -> None:
        # A tiled streaming FV whose applied spec carries a recovered
        # refresh_freq (the offline tile DT cadence) must round-trip that
        # cadence into the exported YAML so `snow feature init` -> re-apply
        # preserves it.  target_lag_sec (OFT ingest lag, stamped 0) stays
        # stripped.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": self._tiled_streaming_spec_with_refresh_freq("1 minute")},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data.get("refresh_freq") == "1 minute", (
            "Tiled streaming FV export must emit the recovered refresh_freq; " f"got {data.get('refresh_freq')!r}."
        )
        assert "target_lag_sec" not in data
        assert "target_lag" not in data

    def test_exported_streaming_fv_yaml_roundtrips_through_loader(self, tmp_path: Path) -> None:
        # End-to-end: an exported streaming FV YAML (with the
        # runtime-stamped ``target_lag_sec: 0`` dropped) must load
        # cleanly through ``loader.load_specs`` and produce a
        # ``StreamingFeatureView`` instance — confirms the exporter's
        # omission keeps the ``snow feature init`` →
        # ``snow feature plan`` round-trip green.
        from snowflake.ml.feature_store.decl.exporter import export_specs
        from snowflake.ml.feature_store.decl.loader import load_specs
        from snowflake.ml.feature_store.decl.spec_models import StreamingFeatureView

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": self._streaming_spec_with_target_lag(0)},
        )
        export_root = tmp_path / "MYDB.PUBLIC"
        batch = load_specs([f"{export_root}/..."])
        streaming = [s for s in batch.specs if isinstance(s, StreamingFeatureView)]
        assert len(streaming) == 1
        assert streaming[0].target_lag is None
        assert streaming[0].target_lag_sec is None


class TestExportSpecsOmitsOnlineForStreamingAndRealtime:
    """Pin :func:`export_specs` omits ``online`` from exported streaming /
    realtime FV YAML.

    ``StreamingFeatureView`` and ``RealtimeFeatureView`` are always
    online by design (the Snowflake runtime materialises an Online
    Feature Table for every deployed instance), so the authoring
    YAML's ``online: true`` is redundant.  The base
    ``FeatureView._enforce_always_online_for_stream_or_realtime``
    validator in ``spec_models`` defaults ``online=True`` on these
    kinds when the field is omitted and rejects an explicit
    ``online=False``.

    The exporter's job is to emit the thin authoring shape that
    matches what ``snow feature init`` scaffolds — for streaming /
    realtime FVs that means dropping the redundant ``online:`` key
    entirely.  ``BatchFeatureView`` keeps the field first-class
    because batch FVs can legitimately be offline-only (the exporter
    derives ``online: true | false`` from the presence of
    ``online_store_type`` on the recovered SPECIFICATION).
    """

    def test_exported_streaming_fv_yaml_omits_online_key(self, tmp_path: Path) -> None:
        # The most common case: ``_FULL_SPEC`` is a streaming FV with
        # ``online_store_type: postgres`` recovered from the live
        # SPECIFICATION; the exporter must NOT round-trip that into
        # an ``online: true`` authoring key — the kind alone implies it.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "online" not in data

    def test_exported_realtime_fv_yaml_omits_online_key(self, tmp_path: Path) -> None:
        # Same contract for ``RealtimeFeatureView``.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        show_row = {
            "name": "USER_REALTIME$V2$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_REALTIME$V2$ONLINE": _REALTIME_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_realtime_v2.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "online" not in data

    def test_exported_batch_fv_yaml_still_emits_online_true(self, tmp_path: Path) -> None:
        # Regression pin: ``BatchFeatureView`` with a Postgres-backed
        # online store continues to emit ``online: true`` in the
        # exported YAML — the strip is scoped to streaming / realtime
        # kinds only so batch FVs preserve their authoring contract.
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        batch_spec = copy.deepcopy(_FULL_SPEC)
        batch_spec["kind"] = "BatchFeatureView"
        batch_spec["spec"]["sources"][0]["source_type"] = "Batch"
        # ``online_store_type`` is preserved from ``_FULL_SPEC`` and
        # drives the exporter's ``online: true`` emission for batch.
        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": batch_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data["online"] is True

    def test_exported_batch_fv_offline_emits_online_false(self, tmp_path: Path) -> None:
        # Regression pin: an offline-only ``BatchFeatureView``
        # (no ``online_store_type`` on the recovered spec) keeps
        # emitting ``online: false`` so the round-trip preserves the
        # offline intent.
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        batch_spec = copy.deepcopy(_FULL_SPEC)
        batch_spec["kind"] = "BatchFeatureView"
        batch_spec["spec"]["sources"][0]["source_type"] = "Batch"
        batch_spec.pop("online_store_type", None)
        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": batch_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data["online"] is False

    def test_exported_streaming_fv_yaml_roundtrips_to_online_true(self, tmp_path: Path) -> None:
        # End-to-end: exporter drops ``online:`` for streaming, loader
        # picks up the kind discriminator and the
        # ``_enforce_always_online_for_stream_or_realtime`` validator
        # defaults ``online=True`` — the round-trip is invisible.
        from snowflake.ml.feature_store.decl.exporter import export_specs
        from snowflake.ml.feature_store.decl.loader import load_specs
        from snowflake.ml.feature_store.decl.spec_models import StreamingFeatureView

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        export_root = tmp_path / "MYDB.PUBLIC"
        batch = load_specs([f"{export_root}/..."])
        streaming = [s for s in batch.specs if isinstance(s, StreamingFeatureView)]
        assert len(streaming) == 1
        assert streaming[0].online is True


class TestExportSpecsOmitsSchedulingState:
    """Pin :func:`export_specs` omits ``scheduling_state`` from every
    exported FV YAML kind.

    ``scheduling_state`` is server-stamped runtime metadata returned
    by ``SHOW ONLINE FEATURE TABLES`` — it describes whether the
    deployed Online Feature Table is currently RUNNING / SUSPENDED /
    etc.  The authoring spec models in
    :mod:`snowflake.ml.feature_store.decl.spec_models` do NOT declare
    the field; the spec compiler and imperative executor never read
    it; and the only previous writer was the exporter (this module),
    which copied the SHOW-row value into the round-tripped YAML.

    That round-trip created two operator-confusing properties:

    1. The exported authoring YAML carried a key the loader silently
       ignored (Pydantic's default ``extra="ignore"`` policy), so an
       operator editing the field thought they were controlling
       runtime state when they were actually no-op'ing.
    2. The same authoring tree exported from two different runtime
       states (RUNNING vs SUSPENDED) would byte-differ on the
       ``scheduling_state`` line even though every authoring concern
       was identical.

    The runtime surface stays available via
    ``decl_api.enrich_list_results`` (``details.scheduling_state`` on
    the ``snow feature list`` row) so scripts that genuinely depend
    on the runtime state are unaffected.

    The pin covers all three FV kinds (Streaming / Realtime / Batch)
    plus the offline-only BatchFV path that goes through the
    ``_overlay_applied_state_on_export_inputs`` synthetic-show-row
    helper.
    """

    def test_exported_streaming_fv_yaml_omits_scheduling_state_key(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        # ``_SHOW_ROW_1`` carries ``scheduling_state: ACTIVE`` — the
        # exporter must NOT round-trip that into the YAML.
        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "scheduling_state" not in data

    def test_exported_realtime_fv_yaml_omits_scheduling_state_key(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        show_row = {
            "name": "USER_REALTIME$V2$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_REALTIME$V2$ONLINE": _REALTIME_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_realtime_v2.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "scheduling_state" not in data

    def test_exported_batch_fv_yaml_omits_scheduling_state_key(self, tmp_path: Path) -> None:
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        batch_spec = copy.deepcopy(_FULL_SPEC)
        batch_spec["kind"] = "BatchFeatureView"
        batch_spec["spec"]["sources"][0]["source_type"] = "Batch"
        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": batch_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "scheduling_state" not in data

    def test_exported_yaml_omits_scheduling_state_when_show_row_value_is_running(self, tmp_path: Path) -> None:
        # Defence-in-depth: even when the SHOW row carries a non-empty
        # non-default ``scheduling_state`` value (e.g. RUNNING), the
        # exporter must still omit the key.  Pins the behaviour
        # against every real-world value, not just ``ACTIVE``.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        show_row = {
            "name": "USER_CLICKS$V1$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "RUNNING",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "scheduling_state" not in data


class TestExportSpecsStrictMode:
    def test_export_specs_raises_when_specification_missing_for_oft(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={},
            )
        msg = str(exc_info.value)
        assert "USER_CLICKS$V1$ONLINE" in msg
        assert "DESCRIBE" in msg or "SPECIFICATION" in msg
        fv_dir = tmp_path / "MYDB.PUBLIC" / "feature_views"
        if fv_dir.exists():
            assert list(fv_dir.glob("*.yaml")) == []

    def test_export_specs_raises_when_specification_map_omitted(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
            )
        assert "USER_CLICKS$V1$ONLINE" in str(exc_info.value)

    def test_export_specs_raises_when_spec_value_is_none(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": None},  # type: ignore[dict-item]
            )
        assert "USER_CLICKS$V1$ONLINE" in str(exc_info.value)

    def test_export_specs_raises_when_spec_value_is_empty_dict(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": {}},
            )
        assert "USER_CLICKS$V1$ONLINE" in str(exc_info.value)

    def test_export_specs_raises_on_first_missing_oft_in_mixed_input(self, tmp_path: Path) -> None:
        """When only some OFTs have specs, the export must abort on the first missing one."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        show_rows = [
            _SHOW_ROW_1,
            {
                "name": "OTHER_FV$V1$ONLINE",
                "database_name": "MYDB",
                "schema_name": "PUBLIC",
                "scheduling_state": "ACTIVE",
            },
        ]
        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=show_rows,
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            )
        assert "OTHER_FV$V1$ONLINE" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Datasource emission helpers (Phase A — sibling ``datasources/`` directory)
# ---------------------------------------------------------------------------


def _show_row(oft_name: str) -> dict[str, Any]:
    return {
        "name": oft_name,
        "database_name": "MYDB",
        "schema_name": "PUBLIC",
        "scheduling_state": "ACTIVE",
    }


def _spec_with_sources(name: str, version: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal full spec JSON dict with the given top-level sources list."""
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": "MYDB",
            "schema": "PUBLIC",
            "name": name,
            "version": version,
        },
        "offline_configs": [],
        "spec": {
            "ordered_entity_column_names": ["user_id"],
            "sources": sources,
            "features": [],
        },
    }


class TestExportSpecsDatasourceEmission:
    """Phase A: every FV's ``spec.sources[]`` is emitted as a deduped, column-superset
    ``datasources/<name>.yaml`` alongside the existing FV/entity outputs."""

    def test_export_writes_datasource_yaml_for_each_source(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        ds_path = tmp_path / "MYDB.PUBLIC" / "datasources" / "user_events.yaml"
        assert ds_path.exists()
        data = yaml.safe_load(ds_path.read_text())
        assert data["kind"] == "StreamingSource"
        assert data["name"] == "user_events"
        col_names = [c["name"] for c in data["columns"]]
        assert "user_id" in col_names
        assert "event" in col_names

    def test_export_dedupes_datasource_across_two_fvs_same_columns(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        shared_source = {
            "name": "user_events",
            "source_type": "Stream",
            "columns": [
                {"name": "user_id", "type": "StringType"},
                {"name": "event", "type": "StringType"},
            ],
        }
        spec_a = _spec_with_sources("fv_a", "v1", [dict(shared_source, columns=list(shared_source["columns"]))])
        spec_b = _spec_with_sources("fv_b", "v1", [dict(shared_source, columns=list(shared_source["columns"]))])

        export_specs(
            show_rows=[_show_row("FV_A$V1$ONLINE"), _show_row("FV_B$V1$ONLINE")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={
                "FV_A$V1$ONLINE": spec_a,
                "FV_B$V1$ONLINE": spec_b,
            },
        )
        ds_dir = tmp_path / "MYDB.PUBLIC" / "datasources"
        assert ds_dir.exists()
        assert len(list(ds_dir.glob("*.yaml"))) == 1
        assert (ds_dir / "user_events.yaml").exists()

    def test_export_datasource_columns_are_superset_across_two_fvs(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec_a = _spec_with_sources(
            "fv_a",
            "v1",
            [
                {
                    "name": "user_events",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "A", "type": "StringType"},
                        {"name": "B", "type": "StringType"},
                    ],
                }
            ],
        )
        spec_b = _spec_with_sources(
            "fv_b",
            "v1",
            [
                {
                    "name": "user_events",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "B", "type": "StringType"},
                        {"name": "C", "type": "StringType"},
                    ],
                }
            ],
        )
        export_specs(
            show_rows=[_show_row("FV_A$V1$ONLINE"), _show_row("FV_B$V1$ONLINE")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={
                "FV_A$V1$ONLINE": spec_a,
                "FV_B$V1$ONLINE": spec_b,
            },
        )
        ds_path = tmp_path / "MYDB.PUBLIC" / "datasources" / "user_events.yaml"
        data = yaml.safe_load(ds_path.read_text())
        col_names = [c["name"] for c in data["columns"]]
        assert col_names == ["A", "B", "C"]

    def test_export_raises_when_two_fvs_disagree_on_column_type(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec_a = _spec_with_sources(
            "fv_a",
            "v1",
            [
                {
                    "name": "clickstream",
                    "source_type": "Stream",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
        )
        spec_b = _spec_with_sources(
            "fv_b",
            "v1",
            [
                {
                    "name": "clickstream",
                    "source_type": "Stream",
                    "columns": [{"name": "USER_ID", "type": "VarcharType"}],
                }
            ],
        )
        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_show_row("FV_A$V1$ONLINE"), _show_row("FV_B$V1$ONLINE")],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={
                    "FV_A$V1$ONLINE": spec_a,
                    "FV_B$V1$ONLINE": spec_b,
                },
            )
        msg = str(exc_info.value)
        assert "clickstream" in msg
        assert "USER_ID" in msg
        assert "StringType" in msg
        assert "VarcharType" in msg
        assert "fv_a" in msg
        assert "fv_b" in msg

        ds_dir = tmp_path / "MYDB.PUBLIC" / "datasources"
        assert (not ds_dir.exists()) or list(ds_dir.glob("*.yaml")) == []

    def test_export_skips_realtime_synthetic_sources(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec_request = _spec_with_sources(
            "fv_request",
            "v1",
            [
                {
                    "name": "user_request",
                    "source_type": "Request",
                    "columns": [{"name": "x", "type": "StringType"}],
                }
            ],
        )
        spec_features = _spec_with_sources(
            "fv_features",
            "v1",
            [
                {
                    "name": "feature_lookup",
                    "source_type": "Features",
                    "columns": [{"name": "y", "type": "StringType"}],
                }
            ],
        )
        spec_normal = _spec_with_sources(
            "fv_normal",
            "v1",
            [
                {
                    "name": "user_events",
                    "source_type": "Stream",
                    "columns": [{"name": "user_id", "type": "StringType"}],
                }
            ],
        )
        export_specs(
            show_rows=[
                _show_row("FV_REQUEST$V1$ONLINE"),
                _show_row("FV_FEATURES$V1$ONLINE"),
                _show_row("FV_NORMAL$V1$ONLINE"),
            ],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={
                "FV_REQUEST$V1$ONLINE": spec_request,
                "FV_FEATURES$V1$ONLINE": spec_features,
                "FV_NORMAL$V1$ONLINE": spec_normal,
            },
        )
        ds_dir = tmp_path / "MYDB.PUBLIC" / "datasources"
        assert (ds_dir / "user_events.yaml").exists()
        assert not (ds_dir / "user_request.yaml").exists()
        assert not (ds_dir / "feature_lookup.yaml").exists()
        assert not (ds_dir / "Request.yaml").exists()
        assert not (ds_dir / "Features.yaml").exists()

    def test_export_unknown_source_type_raises(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = _spec_with_sources(
            "fv_weird",
            "v1",
            [
                {
                    "name": "weird",
                    "source_type": "Frobnicate",
                    "columns": [{"name": "x", "type": "StringType"}],
                }
            ],
        )
        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_show_row("FV_WEIRD$V1$ONLINE")],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"FV_WEIRD$V1$ONLINE": spec},
            )
        assert "Frobnicate" in str(exc_info.value)
        ds_dir = tmp_path / "MYDB.PUBLIC" / "datasources"
        assert (not ds_dir.exists()) or list(ds_dir.glob("*.yaml")) == []

    def test_export_batch_source_passes_through_table_identifiers(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = _spec_with_sources(
            "fv_batch",
            "v1",
            [
                {
                    "name": "user_orders",
                    # ``Batch`` is the canonical source_type post-consolidation;
                    # the legacy ``BatchSource`` string is still accepted on
                    # read for plan-file backward compatibility — see the
                    # next test.
                    "source_type": "Batch",
                    "source_database": "DB1",
                    "source_schema": "SCH1",
                    "table": "T1",
                    "columns": [{"name": "user_id", "type": "StringType"}],
                }
            ],
        )
        export_specs(
            show_rows=[_show_row("FV_BATCH$V1$ONLINE")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"FV_BATCH$V1$ONLINE": spec},
        )
        ds_path = tmp_path / "MYDB.PUBLIC" / "datasources" / "user_orders.yaml"
        assert ds_path.exists()
        text = ds_path.read_text()
        data = yaml.safe_load(text)
        assert data["kind"] == "BatchSource"
        assert data["source_database"] == "DB1"
        assert data["source_schema"] == "SCH1"
        assert data["table"] == "T1"
        # Confirm key order: source_database, source_schema, table appear before columns.
        keys = list(data.keys())
        assert keys.index("source_database") < keys.index("source_schema")
        assert keys.index("source_schema") < keys.index("table")
        assert keys.index("table") < keys.index("columns")

    def test_export_legacy_batchsource_source_type_back_compat(self, tmp_path: Path) -> None:
        # Old plan files emitted ``source_type: "BatchSource"`` before the
        # decl/spec enum consolidation collapsed that value to ``"Batch"``.
        # The exporter still resolves the legacy string so existing
        # ``out/plan/*.json`` files keep loading.
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = _spec_with_sources(
            "fv_legacy",
            "v1",
            [
                {
                    "name": "legacy_orders",
                    "source_type": "BatchSource",
                    "table": "T1",
                    "columns": [{"name": "user_id", "type": "StringType"}],
                }
            ],
        )
        export_specs(
            show_rows=[_show_row("FV_LEGACY$V1$ONLINE")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"FV_LEGACY$V1$ONLINE": spec},
        )
        ds_path = tmp_path / "MYDB.PUBLIC" / "datasources" / "legacy_orders.yaml"
        assert ds_path.exists()
        data = yaml.safe_load(ds_path.read_text())
        assert data["kind"] == "BatchSource"

    def test_export_returns_datasource_paths_in_files_list(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        ds_path = str(tmp_path / "MYDB.PUBLIC" / "datasources" / "user_events.yaml")
        assert ds_path in result["files"]

    def test_round_trip_loader_can_parse_emitted_datasource(self, tmp_path: Path) -> None:
        """The emitted datasource YAML must round-trip through ``load_specs``."""
        from snowflake.ml.feature_store.decl.exporter import export_specs
        from snowflake.ml.feature_store.decl.loader import load_specs
        from snowflake.ml.feature_store.decl.spec_models import StreamingSource

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        ds_path = tmp_path / "MYDB.PUBLIC" / "datasources" / "user_events.yaml"
        batch = load_specs([str(ds_path)])
        assert len(batch.specs) == 1
        spec = batch.specs[0]
        assert isinstance(spec, StreamingSource)
        assert spec.name == "user_events"
        col_names = [c.name for c in spec.columns]
        assert "user_id" in col_names
        assert "event" in col_names

    def test_export_no_datasources_dir_when_no_sources(self, tmp_path: Path) -> None:
        """When no FV carries any source, the ``datasources/`` directory is not created."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec_no_sources = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": "MYDB",
                "schema": "PUBLIC",
                "name": "fv_empty",
                "version": "v1",
            },
            "spec": {
                "ordered_entity_column_names": ["user_id"],
            },
        }
        export_specs(
            show_rows=[_show_row("FV_EMPTY$V1$ONLINE")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"FV_EMPTY$V1$ONLINE": spec_no_sources},
        )
        ds_dir = tmp_path / "MYDB.PUBLIC" / "datasources"
        assert not ds_dir.exists()


# ---------------------------------------------------------------------------
# Entity emission from list_entities() rows (export ↔ plan parity)
# ---------------------------------------------------------------------------
#
# Pins the fix for the export ↔ plan asymmetry: the exporter must drive
# entity YAML emission from the authoritative ``entity_rows`` list (the
# legacy ``SHOW TAGS`` shape returned by
# :func:`decl_api.fetch_entity_rows`), not from FV ``ordered_entity_column_names``
# alone.  Orphan entity tags (registered via ``FeatureStore.register_entity()``
# without ever being attached to an FV) must round-trip cleanly through
# export → plan instead of producing false-positive ``DROP_ENTITY`` ops in
# full-directory mode.
#
# Hypotheses pinned by these tests:
#   - H1: ``entity_rows`` shape is ``{name, allowed_values, comment, ...}``
#   - H2: Entity ``_structural_fingerprint`` is name-only (``state.py``
#     L242), so a YAML stub keyed only on ``name`` round-trips identically
#     to the applied side
#   - H3: Tag rows do not carry per-join-key types; both
#     ``state._build_entity_object`` and the exporter normalize to
#     ``StringType``


def _entity_row(
    name: str,
    *,
    join_keys: Optional[list[str]] = None,
    comment: Optional[str] = None,
    database: str = "MYDB",
    schema: str = "PUBLIC",
) -> dict[str, Any]:
    """Build a SHOW TAGS-shape entity row for the tests.

    Mirrors the shape produced by
    :func:`decl_api.imperative_executor.fetch_entity_rows`: ``name`` carries
    the full ``SNOWML_FEATURE_STORE_ENTITY_<NAME>`` tag prefix; join keys
    live in ``allowed_values`` as a JSON-encoded list; ``comment`` carries
    the user-authored description.

    Args:
        name: Entity name (without the ``SNOWML_FEATURE_STORE_ENTITY_`` prefix).
        join_keys: Optional list of join-key column names.  Encoded as
            ``allowed_values`` JSON.
        comment: Optional description, written to the ``comment`` field.
        database: Snowflake database name.
        schema: Snowflake schema name.

    Returns:
        Row dict in the legacy SHOW TAGS shape consumed by
        :func:`decl.state._build_entity_object`.
    """
    row: dict[str, Any] = {
        "name": f"SNOWML_FEATURE_STORE_ENTITY_{name.upper()}",
        "database_name": database,
        "schema_name": schema,
    }
    if join_keys is not None:
        row["allowed_values"] = json.dumps([jk.upper() for jk in join_keys])
    if comment is not None:
        row["comment"] = comment
    return row


class TestExportSpecsEntityRows:
    """Pin entity emission against the authoritative ``entity_rows`` cross-check."""

    def test_orphan_entity_in_entity_rows_emits_yaml(self, tmp_path: Path) -> None:
        """Entity registered as a tag but not referenced by any FV must still emit YAML."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[
                _entity_row("USER_ID", join_keys=["USER_ID"]),
                _entity_row("ORPHAN_KEY", join_keys=["ORPHAN_KEY"], comment="not yet used"),
            ],
        )
        from pathlib import Path

        entity_files = sorted(Path(f).name for f in result["files"] if Path(f).parent.name == "entities")
        assert (
            "ORPHAN_KEY.yaml" in entity_files
        ), "orphan entity tag must produce a YAML stub even when no FV references it"
        orphan_path = tmp_path / "MYDB.PUBLIC" / "entities" / "ORPHAN_KEY.yaml"
        assert orphan_path.exists()
        ent = yaml.safe_load(orphan_path.read_text())
        assert ent["kind"] == "Entity"
        assert ent["name"].upper() == "ORPHAN_KEY"

    def test_referenced_entity_full_fidelity_from_entity_rows(self, tmp_path: Path) -> None:
        """When entity_rows is provided, emitted YAML carries description + real join_keys."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[
                _entity_row(
                    "USER_ID",
                    join_keys=["USER_ID"],
                    comment="Anonymized clickstream user identifier",
                )
            ],
        )
        entity_path = tmp_path / "MYDB.PUBLIC" / "entities" / "USER_ID.yaml"
        assert entity_path.exists()
        ent = yaml.safe_load(entity_path.read_text())
        assert ent["kind"] == "Entity"
        assert ent["name"].upper() == "USER_ID"
        assert (
            ent.get("description") == "Anonymized clickstream user identifier"
        ), f"description from tag comment must round-trip into entity YAML; got {ent!r}"
        join_keys = ent.get("join_keys") or []
        assert len(join_keys) == 1
        jk0 = join_keys[0]
        assert jk0["name"].upper() == "USER_ID"
        assert jk0["type"] == "StringType"

    def test_no_double_emission_when_entity_in_rows_and_referenced(self, tmp_path: Path) -> None:
        """An entity present both in entity_rows and ordered_entity_column_names emits once."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
        )
        from pathlib import Path

        entity_files = [f for f in result["files"] if Path(f).parent.name == "entities"]
        assert len(entity_files) == 1, f"expected one entity YAML, got {entity_files!r}"

    def test_entity_rows_none_emits_no_warning_and_no_entity_yamls(self, tmp_path: Path) -> None:
        """``entity_rows=None`` is treated as "no entities registered" — silent, no fallback."""
        import warnings

        from snowflake.ml.feature_store.decl.exporter import export_specs

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
                entity_rows=None,
            )

        deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert not deprecation_warnings, (
            "passing entity_rows=None must NOT emit a DeprecationWarning — the legacy "
            f"FV-derived fallback was removed; got {deprecation_warnings!r}"
        )

        entity_path = tmp_path / "MYDB.PUBLIC" / "entities" / "user_id.yaml"
        assert not entity_path.exists(), (
            "with the legacy fallback removed, entity_rows=None must produce no entity YAML "
            "(callers must forward decl_api.fetch_entity_rows output to get entity emission)"
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        assert fv_path.exists(), "FV YAML must still be written when entity_rows is omitted"
        from pathlib import Path

        ds_files = [f for f in result["files"] if Path(f).parent.name == "datasources"]
        assert ds_files, "datasource YAML must still be emitted regardless of entity_rows"

    def test_entity_rows_empty_list_emits_no_warning_and_no_entity_yamls(self, tmp_path: Path) -> None:
        """``entity_rows=[]`` matches the None case exactly — silent, no entity YAMLs."""
        import warnings

        from snowflake.ml.feature_store.decl.exporter import export_specs

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
                entity_rows=[],
            )

        deprecation_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert not deprecation_warnings, (
            "passing entity_rows=[] must NOT emit a DeprecationWarning — the legacy "
            f"FV-derived fallback was removed; got {deprecation_warnings!r}"
        )

        from pathlib import Path

        entity_files = [f for f in result["files"] if Path(f).parent.name == "entities"]
        assert entity_files == [], "with the legacy fallback removed, entity_rows=[] must produce no entity YAMLs"

    def test_entity_rows_none_skips_strict_subset_check(self, tmp_path: Path) -> None:
        """When ``entity_rows`` is empty the FV ⊆ entity_rows check is vacuous and must be skipped."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        # Today (with the legacy fallback in place) this exits via the legacy
        # branch before reaching the strict check.  Tomorrow the legacy branch
        # is gone and the strict check is gated on ``entity_rows`` being
        # non-empty, so this call must still succeed without raising.
        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=None,
        )
        assert result["status"] == "exported"

    def test_entity_rows_none_still_returns_noop_for_empty_schema(self, tmp_path: Path) -> None:
        """``show_rows=[]`` + ``entity_rows=None`` is the genuine empty-schema case."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            entity_rows=None,
        )
        assert result == {"status": "exported", "directory": "", "files": []}

    def test_fv_references_unknown_entity_strict_raises(self, tmp_path: Path) -> None:
        """An FV referencing an entity column absent from entity_rows is a hard error."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        with pytest.raises(ValueError) as exc_info:
            export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
                entity_rows=[_entity_row("OTHER_KEY", join_keys=["OTHER_KEY"])],
            )
        msg = str(exc_info.value)
        assert "USER_ID" in msg.upper()
        assert "user_clicks".upper() in msg.upper() or "USER_CLICKS$V1$ONLINE" in msg
        # No partial state: entities/ dir either missing or empty.
        ent_dir = tmp_path / "MYDB.PUBLIC" / "entities"
        if ent_dir.exists():
            assert list(ent_dir.glob("*.yaml")) == []

    def test_emitted_orphan_entity_yaml_round_trips_to_no_change(self, tmp_path: Path) -> None:
        """The fingerprint of the loaded orphan-stub matches the applied entity hash."""
        from snowflake.ml.feature_store.decl.exporter import export_specs
        from snowflake.ml.feature_store.decl.invariants import (
            structural_fingerprint_hash,
        )
        from snowflake.ml.feature_store.decl.state import _build_entity_object

        row = _entity_row("ORPHAN_KEY", join_keys=["ORPHAN_KEY"], comment="orphan tag")

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"]), row],
        )
        orphan_yaml = tmp_path / "MYDB.PUBLIC" / "entities" / "ORPHAN_KEY.yaml"
        loaded = yaml.safe_load(orphan_yaml.read_text())
        # Mirror every field the Entity branch of
        # :func:`_structural_fingerprint` reads — name, join_keys, and
        # description.  Without ``description`` the round-trip would be
        # asymmetric: applied side carries the tag's COMMENT under
        # ``spec_payload['description']`` (see :func:`state._build_entity_object`),
        # so the canonical authoring dict must surface the YAML's
        # ``description:`` (the exporter writes it from the same comment)
        # to keep the fingerprint symmetric.
        canonical = {
            "kind": "Entity",
            "name": loaded["name"].upper(),
            "description": loaded.get("description", ""),
            "join_keys": [{"name": jk["name"].upper(), "type": "StringType"} for jk in loaded.get("join_keys", [])],
        }
        emitted_hash = structural_fingerprint_hash(canonical)

        applied = _build_entity_object(row, "MYDB", "PUBLIC")
        assert applied is not None
        assert emitted_hash == applied.content_hash, (
            "exported orphan YAML must hash identically to the applied entity tag; "
            f"emitted={emitted_hash} applied={applied.content_hash}"
        )

    def test_entity_rows_only_no_fvs_still_emits_entities(self, tmp_path: Path) -> None:
        """A schema with registered entities but no FVs must still export entity YAMLs."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            entity_rows=[_entity_row("LONELY_KEY", join_keys=["LONELY_KEY"])],
        )
        from pathlib import Path

        entity_files = [f for f in result["files"] if Path(f).parent.name == "entities"]
        assert len(entity_files) == 1
        entity_path = tmp_path / "MYDB.PUBLIC" / "entities" / "LONELY_KEY.yaml"
        assert entity_path.exists()


# ===========================================================================
# Sources/ layout — drives the new init-subsumes-export flow.
#
# When ``layout="sources"`` the exporter writes directly into
# ``<output_dir>/sources/{entities,datasources,feature_views}/``, mirroring
# the manifest project tree scaffolded by ``snow feature init``.  The legacy
# ``<DB>.<SCHEMA>/`` subdir is bypassed entirely so the same files re-apply
# without any manual move.
# ===========================================================================


class TestExportSpecsSourcesLayout:
    """``export_specs(..., layout="sources")`` writes into the manifest layout.

    Pinned by the init-subsumes-export plan: when init pulls deployed
    artifacts onto disk it must drop them straight into
    ``<project_root>/sources/{entities,datasources,feature_views}/`` so
    a follow-up ``snow feature plan`` is byte-for-byte NO_CHANGE.
    """

    def test_sources_layout_writes_feature_view_into_sources_dir(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
            layout="sources",
        )
        # No <DB>.<SCHEMA>/ subdir under the new layout.
        assert not (
            tmp_path / "MYDB.PUBLIC"
        ).exists(), "layout='sources' must NOT create the legacy <DB>.<SCHEMA>/ tree"
        fv_path = tmp_path / "sources" / "feature_views" / "user_clicks_v1.yaml"
        assert fv_path.exists()
        data = yaml.safe_load(fv_path.read_text())
        assert data["kind"] == "StreamingFeatureView"
        assert data["name"] == "user_clicks"

    def test_sources_layout_writes_udf_py_alongside_yaml(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
            layout="sources",
        )
        py_path = tmp_path / "sources" / "feature_views" / "user_clicks_v1.py"
        assert py_path.exists()
        assert py_path.read_text() == "def transform(x):\n    return len(x)"

    def test_sources_layout_writes_entities_into_sources_entities_dir(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
            layout="sources",
        )
        entity_path = tmp_path / "sources" / "entities" / "USER_ID.yaml"
        assert entity_path.exists()
        data = yaml.safe_load(entity_path.read_text())
        assert data["kind"] == "Entity"
        assert data["name"] == "USER_ID"

    def test_sources_layout_writes_datasources_into_sources_datasources_dir(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
            layout="sources",
        )
        ds_path = tmp_path / "sources" / "datasources" / "user_events.yaml"
        assert ds_path.exists()
        data = yaml.safe_load(ds_path.read_text())
        assert data["name"] == "user_events"

    def test_sources_layout_returned_directory_points_at_sources_root(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
            layout="sources",
        )
        assert result["status"] == "exported"
        assert result["directory"] == str(tmp_path / "sources")
        # All emitted files must live under <output_dir>/sources/, not under
        # the legacy <DB>.<SCHEMA>/ root.
        from pathlib import Path

        for f in result["files"]:
            rel = Path(f).relative_to(tmp_path)
            assert rel.parts[0] == "sources", f"file {f} escaped sources/ root"

    def test_sources_layout_empty_input_short_circuits_with_no_dir(self, tmp_path: Path) -> None:
        """An empty schema (no FVs, no entities) must not create sources/."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            entity_rows=[],
            layout="sources",
        )
        assert result["status"] == "exported"
        assert result["files"] == []
        assert result["directory"] == ""
        assert not (tmp_path / "sources").exists()

    def test_sources_layout_entity_only_schema_writes_entity_yaml(self, tmp_path: Path) -> None:
        """A schema with entity tags but no FVs still writes entity YAML under sources/."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            entity_rows=[_entity_row("LONELY_KEY", join_keys=["LONELY_KEY"])],
            layout="sources",
        )
        entity_path = tmp_path / "sources" / "entities" / "LONELY_KEY.yaml"
        assert entity_path.exists()
        assert str(entity_path) in result["files"]

    def test_default_layout_is_db_schema_for_back_compat(self, tmp_path: Path) -> None:
        """Omitting ``layout=`` must keep the legacy <DB>.<SCHEMA>/ behaviour."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
            entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
        )
        # Legacy layout still wins when layout is unspecified.
        assert (tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml").exists()
        assert not (tmp_path / "sources").exists()

    def test_sources_layout_unknown_value_raises(self, tmp_path: Path) -> None:
        """An unrecognised layout token must raise a clear error."""
        from snowflake.ml.feature_store.decl.exporter import export_specs

        with pytest.raises(ValueError, match="layout"):
            export_specs(
                show_rows=[_SHOW_ROW_1],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
                entity_rows=[_entity_row("USER_ID", join_keys=["USER_ID"])],
                layout="bogus_value",
            )


class TestBackfillRoundTrip:
    """``FeatureView.backfill`` must round-trip cleanly through the loader's
    Pydantic layer.

    The exporter itself never produces a ``backfill:`` block because the
    deployed ``DESCRIBE … TYPE = SPECIFICATION`` payload doesn't carry one
    (backfill is operational, not structural).  So the round-trip we pin
    here is the **author's** YAML → loaded model → ``model_dump`` →
    re-loaded model — the path the planner exercises on every
    ``snow feature plan`` invocation.
    """

    def test_feature_view_backfill_round_trips_through_model_dump(self) -> None:
        from snowflake.ml.feature_store.decl.spec_models import Backfill, FeatureView

        fv = FeatureView.model_validate(
            {
                "kind": "StreamingFeatureView",
                "name": "USER_CLICK_BACKFILL_DECL",
                "version": "V1",
                "entities": ["USER_ID"],
                "sources": [{"name": "CLICKSTREAM_EVENTS", "source_type": "Stream"}],
                "backfill": {
                    "table": "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL",
                    "start_time": "2026-05-18T22:00:00",
                },
            }
        )

        dumped = fv.model_dump(exclude_none=True)
        assert "backfill" in dumped, (
            "FeatureView.model_dump must preserve the FV-level backfill block "
            "so the loader → planner → executor pipeline can read it back."
        )
        assert dumped["backfill"]["table"] == "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL"

        rebuilt = FeatureView.model_validate(dumped)
        assert isinstance(rebuilt.backfill, Backfill)
        assert rebuilt.backfill.table == "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL"

    def test_batch_feature_view_backfill_overwrite_round_trips(self) -> None:
        from snowflake.ml.feature_store.decl.spec_models import Backfill, FeatureView

        fv = FeatureView.model_validate(
            {
                "kind": "BatchFeatureView",
                "name": "ORDERS_TOTAL_DECL",
                "version": "V1",
                "entities": ["USER_ID"],
                "sources": [{"name": "ORDERS_DECL", "source_type": "Batch", "table": "RAW_ORDERS"}],
                "backfill": {"overwrite": True, "initialize": "ON_SCHEDULE"},
            }
        )

        dumped = fv.model_dump(exclude_none=True)
        assert dumped["backfill"]["overwrite"] is True
        assert dumped["backfill"]["initialize"] == "ON_SCHEDULE"

        rebuilt = FeatureView.model_validate(dumped)
        assert isinstance(rebuilt.backfill, Backfill)
        assert rebuilt.backfill.overwrite is True
        assert rebuilt.backfill.initialize == "ON_SCHEDULE"

    def test_exporter_top_level_order_lists_backfill(self) -> None:
        """Sanity guard: ``backfill`` is a recognised FV top-level key in the
        exporter's ordering tuple, so any future code path that wires
        backfill into the export dict produces stable YAML output."""
        from snowflake.ml.feature_store.decl.exporter import _FV_TOP_LEVEL_ORDER

        assert "backfill" in _FV_TOP_LEVEL_ORDER

    # ------------------------------------------------------------------
    # B8 — backfill: block re-emission from applied spec_payload.
    #
    # Phase A's metadata-roundtrip extensions persist the operator's
    # authored backfill table (StreamingMetadata.backfill_table) and
    # carry the existing backfill_start_time (STREAM_CONFIG) through to
    # the recovered ``spec_payload``.  The exporter must surface those
    # back into the YAML so ``snow feature init`` produces a re-applicable
    # spec.  ``backfill.overwrite`` is intentionally excluded (one-shot
    # operational flag — round-tripping it would force re-overwrites on
    # every replan; LIMITATIONS L2 documents the authoring-only
    # contract).
    # ------------------------------------------------------------------

    @staticmethod
    def _streaming_backfill_show_row() -> dict[str, Any]:
        return {
            "name": "USER_CLICKS_BF$V1$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }

    @staticmethod
    def _streaming_backfill_spec(
        *,
        backfill_table: Optional[str] = None,
        backfill_start_time: Optional[str] = None,
        backfill_overwrite: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Build a StreamingFeatureView spec_payload optionally carrying
        the recovered backfill metadata.

        Mirrors the post-recovery shape ``state.fetch_applied_state``
        produces after Phase A's metadata extensions have surfaced
        ``backfill_table`` / ``backfill_start_time`` into the inner
        spec.

        Args:
            backfill_table: Optional ``StreamingMetadata.backfill_table``
                value to inject under ``inner["backfill_table"]``.
            backfill_start_time: Optional ``StreamConfig.backfill_start_time``
                value to inject under ``inner["backfill_start_time"]``.
            backfill_overwrite: Optional sentinel value for the
                "exporter never re-emits overwrite" test — placed
                under ``inner["backfill_overwrite"]`` so the
                exporter sees a key it MUST ignore.

        Returns:
            A dict in the same shape ``AppliedObject.spec_payload``
            carries for a recovered StreamingFeatureView, ready to be
            handed to :func:`export_specs` via ``specification_map``.
        """
        inner: dict[str, Any] = {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "USER_EVENTS",
                    "source_type": "Stream",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
            "udf": {
                "name": "noop",
                "function_definition": "def noop(x):\n    return x",
                "engine": "python",
                "output_columns": [{"name": "USER_ID", "type": "StringType"}],
            },
            "timestamp_field": "EVENT_TS",
            "feature_granularity_sec": 60,
            "feature_aggregation_method": "tiles",
        }
        if backfill_table is not None:
            inner["backfill_table"] = backfill_table
        if backfill_start_time is not None:
            inner["backfill_start_time"] = backfill_start_time
        # Synthesise the authoring-only ``backfill_overwrite`` key so the
        # "no overwrite emission" test can pin that the exporter never
        # propagates this field even when the upstream payload happens
        # to surface it (defensive — A3 explicitly does NOT persist
        # ``backfill_overwrite``).
        if backfill_overwrite is not None:
            inner["backfill_overwrite"] = backfill_overwrite
        return {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": "MYDB",
                "schema": "PUBLIC",
                "name": "USER_CLICKS_BF",
                "version": "V1",
            },
            "offline_configs": [],
            "spec": inner,
            "online_store_type": "postgres",
        }

    def test_export_re_emits_backfill_block_when_backfill_table_present(self, tmp_path: Path) -> None:
        """When the recovered ``spec_payload`` carries ``backfill_table``,
        the exported YAML must include a ``backfill: { table: ... }``
        block so ``snow feature apply`` can re-bind the streaming
        backfill source on the next round-trip.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = self._streaming_backfill_spec(
            backfill_table="MYDB.PUBLIC.USER_CLICK_HISTORY",
        )

        export_specs(
            show_rows=[self._streaming_backfill_show_row()],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS_BF$V1$ONLINE": spec},
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "USER_CLICKS_BF_V1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        backfill = data.get("backfill")
        assert isinstance(backfill, dict), (
            "exporter must re-emit a backfill: block when the applied "
            f"spec_payload carries backfill_table; got data={data!r}"
        )
        assert backfill.get("table") == "MYDB.PUBLIC.USER_CLICK_HISTORY"

    def test_export_re_emits_no_backfill_block_when_metadata_absent(self, tmp_path: Path) -> None:
        """A streaming FV whose spec_payload carries no backfill metadata
        must NOT grow a phantom ``backfill:`` block on export.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = self._streaming_backfill_spec()

        export_specs(
            show_rows=[self._streaming_backfill_show_row()],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS_BF$V1$ONLINE": spec},
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "USER_CLICKS_BF_V1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "backfill" not in data, (
            "exporter must NOT emit a backfill: block when the applied "
            f"spec_payload carries no backfill metadata; got data={data!r}"
        )

    def test_export_does_not_re_emit_overwrite_field(self, tmp_path: Path) -> None:
        """``backfill.overwrite`` is the authoring-only contract per the
        Phase B plan: A3 explicitly does NOT persist it, and the exporter
        must NOT re-emit it even if some upstream payload surfaces an
        ``backfill_overwrite`` key.

        Round-tripping ``overwrite`` would silently force re-overwrites
        on every replan — LIMITATIONS L2 documents this contract; this
        test pins it on the exporter side.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = self._streaming_backfill_spec(
            backfill_table="MYDB.PUBLIC.USER_CLICK_HISTORY",
            backfill_overwrite=True,
        )

        export_specs(
            show_rows=[self._streaming_backfill_show_row()],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS_BF$V1$ONLINE": spec},
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "USER_CLICKS_BF_V1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        backfill = data.get("backfill") or {}
        assert "overwrite" not in backfill, (
            "exporter must NEVER re-emit backfill.overwrite (authoring-only "
            f"contract per LIMITATIONS L2); got backfill={backfill!r}"
        )

    def test_export_re_emits_backfill_start_time_when_present(self, tmp_path: Path) -> None:
        """``backfill_start_time`` (already in STREAM_CONFIG) must round-trip
        through the exporter as ``backfill.start_time``.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        spec = self._streaming_backfill_spec(
            backfill_start_time="2026-05-18T22:00:00",
        )

        export_specs(
            show_rows=[self._streaming_backfill_show_row()],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS_BF$V1$ONLINE": spec},
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "USER_CLICKS_BF_V1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        backfill = data.get("backfill")
        assert isinstance(backfill, dict)
        assert backfill.get("start_time") == "2026-05-18T22:00:00"


# ---------------------------------------------------------------------------
# B8 — applied_state kwarg forwarding from decl.api -> decl.exporter.
#
# AS11 added ``applied_state: Optional[Any] = None`` to
# ``decl.api.export_specs`` as a no-op stub; B8 is where the actual
# wiring lands.  The CLI manager (``snowflake-cli``'s
# ``manager._export_into_sources``) passes the reconstructed
# ``AppliedState`` down so the exporter can prefer the metadata-driven
# BatchFV source bindings (from B-state's ``_inject_batch_fv_source_
# from_metadata`` rewrite of ``_inject_batch_fv_source_from_dt_text``)
# over the lossy raw DESCRIBE payload.  When an FV is missing from the
# applied state (legacy / fresh-init case), the exporter must soft-
# fall-back to the raw DESCRIBE entry rather than dropping the FV.
# ---------------------------------------------------------------------------


class TestAppliedStateForward:
    """Pin the api.export_specs -> exporter.export_specs forwarding contract
    for the new ``applied_state`` kwarg, plus the per-FV soft fallback
    when the applied state is partial.
    """

    def test_export_specs_forwards_applied_state_kwarg(self) -> None:
        """``decl.api.export_specs`` must forward ``applied_state`` to the
        exporter so B-state's metadata-recovered FV source bindings reach
        the YAML emission path.

        Without this wiring the kwarg added during the AS11 fix stays
        a no-op and the exporter never sees the recovered state.
        """
        from unittest.mock import patch

        from snowflake.ml.feature_store.decl import api as decl_api

        sentinel_state = object()  # opaque — only identity matters here.

        with patch(
            "snowflake.ml.feature_store.decl.exporter.export_specs",
            return_value={"status": "exported", "directory": "", "files": []},
        ) as mock_exporter:
            decl_api.export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir="/tmp/never-written",
                database="DB",
                schema="SCH",
                specification_map={},
                applied_state=sentinel_state,
            )

        mock_exporter.assert_called_once()
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("applied_state") is sentinel_state, (
            "decl.api.export_specs must forward applied_state by-reference " f"into the exporter; got kwargs={kwargs!r}"
        )

    def test_export_falls_back_to_raw_describe_when_applied_state_missing_fv(self, tmp_path: Path) -> None:
        """When ``applied_state`` is provided but does NOT carry a particular
        FV (legacy / fresh-init), the exporter must soft-fall-back to the
        raw ``specification_map`` entry for that FV instead of dropping it.

        Pins the partial-recovery contract: ``applied_state`` wins for
        FVs it knows about; the raw DESCRIBE survives for FVs it does
        not.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        # FV-A is in applied_state — recovered shape carries an authored
        # source name.  FV-B is NOT in applied_state — only the raw
        # specification_map carries it.
        recovered_fv_a = _applied_object_dict(
            name="FV_A",
            version="V1",
            database="MYDB",
            schema="PUBLIC",
            kind="BatchFeatureView",
            sources=[
                {
                    "name": "AUTHORED_SOURCE_A",
                    "source_type": "Batch",
                    "table": "RAW_A",
                }
            ],
        )
        applied_state = _applied_state_with(recovered_fv_a)

        # FV-B's raw DESCRIBE — a plain BatchFV spec with its own source.
        # The exporter should still emit FV-B's YAML using this payload.
        raw_fv_b = {
            "kind": "BatchFeatureView",
            "metadata": {
                "database": "MYDB",
                "schema": "PUBLIC",
                "name": "FV_B",
                "version": "V1",
            },
            "offline_configs": [],
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [
                    {
                        "name": "RAW_DESCRIBE_SOURCE_B",
                        "source_type": "Batch",
                        "table": "RAW_B",
                    }
                ],
                "features": [],
            },
            "online_store_type": "postgres",
        }

        result = export_specs(
            show_rows=[
                {
                    "name": "FV_A$V1$ONLINE",
                    "database_name": "MYDB",
                    "schema_name": "PUBLIC",
                    "scheduling_state": "ACTIVE",
                },
                {
                    "name": "FV_B$V1$ONLINE",
                    "database_name": "MYDB",
                    "schema_name": "PUBLIC",
                    "scheduling_state": "ACTIVE",
                },
            ],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={
                "FV_A$V1$ONLINE": {
                    # Lossy raw DESCRIBE for FV-A: empty sources.  The
                    # applied-state recovered payload should win.
                    "kind": "BatchFeatureView",
                    "metadata": {
                        "database": "MYDB",
                        "schema": "PUBLIC",
                        "name": "FV_A",
                        "version": "V1",
                    },
                    "offline_configs": [],
                    "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
                    "online_store_type": "postgres",
                },
                "FV_B$V1$ONLINE": raw_fv_b,
            },
            applied_state=applied_state,
        )

        # Both FVs must produce a YAML on disk.
        fv_a_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "FV_A_V1.yaml"
        fv_b_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "FV_B_V1.yaml"
        assert fv_a_path.exists(), "FV_A (in applied_state) must be exported; " f"got files={result['files']!r}"
        assert fv_b_path.exists(), (
            "FV_B (NOT in applied_state) must still be exported via "
            f"the raw DESCRIBE fallback; got files={result['files']!r}"
        )

        # FV-A picks up the recovered authored source name from
        # applied_state, NOT the empty list from the raw DESCRIBE.
        fv_a_data = yaml.safe_load(fv_a_path.read_text())
        sources_a = fv_a_data.get("sources") or []
        assert sources_a and sources_a[0].get("name") == "AUTHORED_SOURCE_A", (
            "applied_state must win for FVs it knows about; got " f"sources={sources_a!r}"
        )

        # FV-B picks up the raw DESCRIBE source (the soft-fallback).
        fv_b_data = yaml.safe_load(fv_b_path.read_text())
        sources_b = fv_b_data.get("sources") or []
        assert sources_b and sources_b[0].get("name") == "RAW_DESCRIBE_SOURCE_B", (
            "raw DESCRIBE must survive for FVs not in applied_state; " f"got sources={sources_b!r}"
        )


# ---------------------------------------------------------------------------
# export_specs(applied_state=...) — init-export applied-state unification.
#
# Background.  ``snow feature init`` previously fed the raw
# ``DESCRIBE … TYPE = SPECIFICATION`` JSON (``specification_map``) straight
# through to ``export_specs``.  For BatchFeatureView the SPECIFICATION JSON
# always returns ``spec.sources: []`` (snowml-core's FROM SPECIFICATION
# serializer encodes the source binding into the offline Dynamic Table's
# ``SELECT … FROM …`` body, not in the spec payload).  The plan path
# already runs the BatchFV source-recovery injectors
# (``state._inject_batch_fv_source_from_dt_text`` +
# ``_inject_advanced_bfv_fields_from_dt_text``) and supports offline-only
# BFVs via ``state._build_offline_fv_object`` — both surfaces inside
# :func:`fetch_applied_state`.  Init export was never wired to the same
# recovery, so the exported YAML drifted from the deployed runtime, and a
# follow-up ``snow feature plan`` against the exported tree spuriously
# emitted ``RECREATE_FV``.  The applied-state unification routes init
# through the same ``fetch_applied_state`` path and hands the recovered
# state to ``export_specs`` via the new ``applied_state=`` kwarg.
# ---------------------------------------------------------------------------


def _applied_object_dict(
    *,
    name: str,
    version: str,
    database: str,
    schema: str,
    kind: str,
    sources: list[dict[str, Any]],
    extra_inner: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the ``spec_payload`` shape an :class:`AppliedObject` carries
    for a BatchFV after :func:`fetch_applied_state` has run.

    Mirrors the structure produced when ``dt_text_map`` recovery has
    already injected ``spec.sources`` and the advanced BFV fields —
    i.e. what the exporter is supposed to read when an
    ``applied_state=`` kwarg is provided.

    Args:
        name: BatchFV logical name (the ``metadata.name`` slot).
        version: BatchFV version (the ``metadata.version`` slot).
        database: Containing database for both ``metadata`` and the
            offline DT's ``offline_configs[0]``.
        schema: Containing schema for both ``metadata`` and the
            offline DT's ``offline_configs[0]``.
        kind: Spec ``kind`` field (e.g. ``"BatchFeatureView"``).
        sources: Already-recovered ``spec.sources`` list — what
            ``state._inject_batch_fv_source_from_dt_text`` would have
            written for a real applied state.
        extra_inner: Optional extra keys merged into the ``spec``
            inner dict (e.g. recovered advanced BFV knobs like
            ``cluster_by``, ``refresh_mode``, ``initialize``).

    Returns:
        A plain dict in the same shape ``AppliedObject.spec_payload``
        carries for a recovered BatchFV — the exporter consumes this
        through the ``applied_state=`` kwarg.
    """
    inner: dict[str, Any] = {
        "ordered_entity_column_names": ["USER_ID"],
        "sources": sources,
        "features": [],
        "timestamp_field": "EVENT_TS",
        "feature_granularity_sec": 3600,
        "feature_aggregation_method": "tiles",
    }
    if extra_inner:
        inner.update(extra_inner)
    return {
        "kind": kind,
        "metadata": {
            "database": database,
            "schema": schema,
            "name": name,
            "version": version,
        },
        "offline_configs": [
            {
                "database": database,
                "schema": schema,
                "table": f"{name}${version}",
                "table_type": "BatchSource",
                "store_type": "snowflake",
            }
        ],
        "spec": inner,
        "online_store_type": "postgres",
    }


def _applied_state_with(
    *objects: dict[str, Any], default_database: str = "MYDB", default_schema: str = "PUBLIC"
) -> Any:
    """Build a minimal :class:`AppliedState` carrying the given FV
    spec_payloads (already in the post-recovery shape).

    Returns the live ``AppliedState`` pydantic model — the exporter
    consumes ``applied_state.objects[key].spec_payload`` so a real
    model instance pins the contract end-to-end.

    Args:
        *objects: BatchFV ``spec_payload`` dicts (typically built by
            :func:`_applied_object_dict`) to wrap as
            :class:`AppliedObject` entries on the returned state.
        default_database: Database used when a *spec_payload* lacks an
            explicit ``metadata.database`` (drives the spec key only).
        default_schema: Schema used when a *spec_payload* lacks an
            explicit ``metadata.schema`` (drives the spec key only).

    Returns:
        An ``AppliedState`` instance whose ``objects`` map is
        keyed by the same spec key the exporter computes via
        ``_build_spec_key``, ready to be passed to
        ``export_specs(applied_state=...)``.
    """
    from snowflake.ml.feature_store.decl.state import _build_spec_key
    from snowflake.ml.feature_store.decl.types import AppliedObject, AppliedState

    fv_objects: dict[str, AppliedObject] = {}
    for spec_payload in objects:
        metadata = spec_payload.get("metadata", {})
        kind = spec_payload.get("kind", "BatchFeatureView")
        key = _build_spec_key(
            kind,
            {
                "database": metadata.get("database", default_database),
                "schema": metadata.get("schema", default_schema),
                "name": metadata.get("name", ""),
            },
        )
        fv_objects[key] = AppliedObject(
            key=key,
            kind=kind,
            name=metadata.get("name", ""),
            version=metadata.get("version", ""),
            content_hash="deadbeef",
            spec_payload=spec_payload,
            columns=[],
            from_specification=True,
        )
    return AppliedState(objects=fv_objects)


class TestExportSpecsWithAppliedState:
    """``export_specs(applied_state=...)`` is the new init-export contract.

    When an ``AppliedState`` carrying recovered BatchFV ``spec.sources``,
    advanced BFV fields, and offline-only FV entries is provided, the
    exporter MUST prefer those recovered payloads over the raw
    ``specification_map`` (which is lossy for BatchFV).  The kwarg is
    optional — callers that don't pass it still fall through to the
    legacy ``specification_map`` path so existing tests / callers keep
    working byte-stably.
    """

    _BATCH_SHOW_ROW = {
        "name": "MY_BATCH_FV$V1$ONLINE",
        "database_name": "MYDB",
        "schema_name": "PUBLIC",
        "scheduling_state": "ACTIVE",
    }

    def _broken_spec(self) -> dict[str, Any]:
        """Mirror the lossy SPECIFICATION JSON: BatchFV with empty sources."""
        return _applied_object_dict(
            name="MY_BATCH_FV",
            version="V1",
            database="MYDB",
            schema="PUBLIC",
            kind="BatchFeatureView",
            sources=[],
        )

    def _recovered_spec(self) -> dict[str, Any]:
        """Post-recovery shape: sources injected from DT text."""
        return _applied_object_dict(
            name="MY_BATCH_FV",
            version="V1",
            database="MYDB",
            schema="PUBLIC",
            kind="BatchFeatureView",
            sources=[
                {
                    "name": "RAW_EVENTS",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS",
                }
            ],
        )

    def test_applied_state_recovers_batch_fv_sources(self, tmp_path: Path) -> None:
        """Given an ``applied_state`` whose BatchFV ``spec.sources`` was
        recovered from DT text, the exported YAML must surface the
        recovered ``table`` reference — not the lossy empty list.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        applied_state = _applied_state_with(self._recovered_spec())

        export_specs(
            show_rows=[self._BATCH_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"MY_BATCH_FV$V1$ONLINE": self._broken_spec()},
            applied_state=applied_state,
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "MY_BATCH_FV_V1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data["kind"] == "BatchFeatureView"
        sources = data.get("sources")
        assert isinstance(sources, list) and len(sources) == 1, (
            "init export must surface the recovered BatchFV source; got " f"sources={sources!r}"
        )
        assert sources[0]["name"] == "RAW_EVENTS"
        assert sources[0]["source_type"] == "Batch"
        assert sources[0]["table"] == "RAW_EVENTS"

    def test_applied_state_takes_precedence_over_specification_map(self, tmp_path: Path) -> None:
        """When both kwargs are present, ``applied_state`` wins.

        Pins the precedence rule: the legacy ``specification_map`` is
        the lossy DESCRIBE JSON, ``applied_state`` is the recovered
        version.  Reading from the lossy map would re-emit the bug
        even after the wiring is in place.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        applied_state = _applied_state_with(self._recovered_spec())

        export_specs(
            show_rows=[self._BATCH_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"MY_BATCH_FV$V1$ONLINE": self._broken_spec()},
            applied_state=applied_state,
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "MY_BATCH_FV_V1.yaml"
        sources = yaml.safe_load(fv_path.read_text()).get("sources") or []
        assert sources and sources[0].get("table") == "RAW_EVENTS"

    def test_applied_state_surfaces_offline_only_bfv_not_in_show_rows(self, tmp_path: Path) -> None:
        """An offline-only BFV reachable only via
        :func:`fetch_applied_state` (``feature_view_rows`` →
        ``_build_offline_fv_object``) must still get a YAML.

        Pre-fix init only iterated ``show_rows`` (``SHOW ONLINE FEATURE
        TABLES``), so any FV deployed with ``online: false`` was
        silently dropped from the exported tree.  After the
        applied-state unification, ``export_specs`` must walk
        ``applied_state.objects`` for FV-kind entries not represented
        in ``show_rows``.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        # Offline-only BFV — no matching show_row, only an applied-state entry.
        offline_only = _applied_object_dict(
            name="OFFLINE_BFV",
            version="V1",
            database="MYDB",
            schema="PUBLIC",
            kind="BatchFeatureView",
            sources=[
                {
                    "name": "OFFLINE_SRC",
                    "source_type": "Batch",
                    "table": "OFFLINE_SRC",
                }
            ],
        )
        applied_state = _applied_state_with(offline_only)

        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            applied_state=applied_state,
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "OFFLINE_BFV_V1.yaml"
        assert fv_path.exists(), (
            "Offline-only BFV (surfaced via applied_state only) must be " f"exported; got files={result['files']!r}"
        )
        data = yaml.safe_load(fv_path.read_text())
        assert data["kind"] == "BatchFeatureView"
        assert data["sources"][0]["table"] == "OFFLINE_SRC"

    def test_applied_state_recovers_advanced_bfv_fields(self, tmp_path: Path) -> None:
        """Advanced BFV authoring knobs (``cluster_by``, ``refresh_mode``,
        ``initialize``) recovered into the spec payload by
        :func:`state._inject_advanced_bfv_fields_from_dt_text` must
        appear in the exported YAML.

        These travel through the same applied-state surface as
        ``sources`` and are part of the same recovery contract.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        recovered = _applied_object_dict(
            name="MY_ADV_BFV",
            version="V1",
            database="MYDB",
            schema="PUBLIC",
            kind="BatchFeatureView",
            sources=[{"name": "RAW_EVENTS", "source_type": "Batch", "table": "RAW_EVENTS"}],
            extra_inner={
                "cluster_by": ["USER_ID", "EVENT_TS"],
                "refresh_mode": "INCREMENTAL",
                "initialize": "ON_CREATE",
            },
        )
        applied_state = _applied_state_with(recovered)

        export_specs(
            show_rows=[
                {
                    "name": "MY_ADV_BFV$V1$ONLINE",
                    "database_name": "MYDB",
                    "schema_name": "PUBLIC",
                    "scheduling_state": "ACTIVE",
                }
            ],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            applied_state=applied_state,
        )

        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "MY_ADV_BFV_V1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data["cluster_by"] == ["USER_ID", "EVENT_TS"]
        assert data["refresh_mode"] == "INCREMENTAL"
        assert data["initialize"] == "ON_CREATE"

    def test_export_without_applied_state_kwarg_still_works(self, tmp_path: Path) -> None:
        """Legacy callers that don't pass ``applied_state`` still get the
        ``specification_map`` codepath unchanged.

        Back-compat sentinel: keeps the door open for callers outside
        this repo (tests, downstream tools) that haven't migrated to
        the applied-state surface yet.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        result = export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        assert result["status"] == "exported"
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        assert fv_path.exists()


# ---------------------------------------------------------------------------
# Phase 3 RED — refresh_freq is the YAML emission key for BFV; omitted on
# streaming / realtime
# ---------------------------------------------------------------------------


class TestRefreshFreqEmitted:
    """The exporter writes the renamed authoring key ``refresh_freq`` for
    BatchFeatureView YAML.

    Streaming and realtime kinds carry no cadence knob in the
    declarative surface (the new validator rejects ``refresh_freq`` on
    those kinds), so the exporter must omit the key on those YAMLs even
    when the deployed SPECIFICATION carries a non-zero
    ``target_lag_sec``.

    Closes the BFV cadence-roundtrip path: an exported BFV YAML loads
    back through ``loader.load_specs`` cleanly with the cadence
    preserved, and a streaming export carries no cadence key that the
    validator would later reject.
    """

    def _streaming_spec_with_target_lag(self, target_lag_sec: int) -> dict[str, Any]:
        import copy

        spec = copy.deepcopy(_FULL_SPEC)
        spec["spec"]["target_lag_sec"] = target_lag_sec
        return spec

    def test_exported_batch_fv_yaml_emits_refresh_freq(self, tmp_path: Path) -> None:
        """BFV YAML carries ``refresh_freq:`` (the renamed authoring
        key), built from the deployed ``spec.target_lag_sec``.

        Args:
            tmp_path: pytest tmp directory fixture for the export root.
        """
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        batch_spec = copy.deepcopy(_FULL_SPEC)
        batch_spec["kind"] = "BatchFeatureView"
        batch_spec["spec"]["sources"][0]["source_type"] = "Batch"
        batch_spec["spec"]["target_lag_sec"] = 3600
        show_row = {
            "name": "USER_CLICKS$V1$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": batch_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert data.get("refresh_freq") == "3600 seconds", (
            "Exported BFV YAML must carry the renamed authoring key "
            "'refresh_freq' (not 'refresh_freq'). Got "
            f"refresh_freq={data.get('refresh_freq')!r}, "
            f"refresh_freq={data.get('refresh_freq')!r}."
        )
        assert "batch_schedule" not in data, (
            "Legacy 'batch_schedule' key must not appear in exported "
            "YAML — the rename is hard. Got "
            f"batch_schedule={data.get('batch_schedule')!r}."
        )

    def test_exported_streaming_fv_yaml_omits_refresh_freq(self, tmp_path: Path) -> None:
        """Streaming FV YAML must omit ``refresh_freq`` (and the legacy
        ``refresh_freq``) — the new validator rejects the field on
        streaming kinds, so an exported file that carried it would fail
        to load back.

        Args:
            tmp_path: pytest tmp directory fixture for the export root.
        """
        from snowflake.ml.feature_store.decl.exporter import export_specs

        export_specs(
            show_rows=[_SHOW_ROW_1],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_CLICKS$V1$ONLINE": self._streaming_spec_with_target_lag(0)},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_clicks_v1.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "refresh_freq" not in data, (
            "Streaming FV exports must omit refresh_freq (the validator "
            "rejects it on streaming kinds). Got "
            f"refresh_freq={data.get('refresh_freq')!r}."
        )
        assert "batch_schedule" not in data, (
            "Streaming FV exports must omit the legacy batch_schedule "
            "key. Got "
            f"batch_schedule={data.get('batch_schedule')!r}."
        )

    def test_exported_realtime_fv_yaml_omits_refresh_freq(self, tmp_path: Path) -> None:
        """Realtime FV YAML must also omit the cadence keys."""
        import copy

        from snowflake.ml.feature_store.decl.exporter import export_specs

        realtime_spec = copy.deepcopy(_FULL_SPEC)
        realtime_spec["kind"] = "RealtimeFeatureView"
        realtime_spec["metadata"]["name"] = "user_realtime"
        realtime_spec["metadata"]["version"] = "v2"
        realtime_spec["spec"]["target_lag_sec"] = 0
        show_row = {
            "name": "USER_REALTIME$V2$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "scheduling_state": "ACTIVE",
        }
        export_specs(
            show_rows=[show_row],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={"USER_REALTIME$V2$ONLINE": realtime_spec},
        )
        fv_path = tmp_path / "MYDB.PUBLIC" / "feature_views" / "user_realtime_v2.yaml"
        data = yaml.safe_load(fv_path.read_text())
        assert "refresh_freq" not in data
        assert "batch_schedule" not in data


if __name__ == "__main__":
    pytest_driver.main()
