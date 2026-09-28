"""Tests for the ``decl.api.export_specs`` facade.

Pins the contract that the public facade in :mod:`decl.api` forwards
its ``entity_rows`` kwarg verbatim to :func:`decl.exporter.export_specs`.
The CLI manager goes through this facade only — never directly into
``decl.exporter`` — so the facade is the single seam where the
``entity_rows`` plumbing must be respected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.test_utils import pytest_driver


class TestApiExportSpecsForwardsEntityRows:
    """Pin the api -> exporter forwarding contract for entity_rows."""

    def test_forwards_entity_rows_kwarg_when_provided(self) -> None:
        """A non-empty entity_rows must be forwarded verbatim to the exporter."""
        rows = [
            {
                "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
                "database_name": "DB",
                "schema_name": "SCH",
                "allowed_values": '["USER_ID"]',
                "comment": "user identifier",
            }
        ]
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
                entity_rows=rows,
            )

        mock_exporter.assert_called_once()
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("entity_rows") is rows, (
            "api.export_specs must forward entity_rows by-reference into the exporter; " f"got kwargs={kwargs!r}"
        )

    def test_default_entity_rows_is_none_when_caller_omits_it(self) -> None:
        """Omitting the kwarg must surface inside the exporter as entity_rows=None."""
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
            )

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("entity_rows", "<unset>") is None, (
            "Omitting entity_rows must surface as entity_rows=None inside "
            f"the exporter (it normalises None to [] internally); got {kwargs!r}"
        )

    def test_forwards_empty_entity_rows_list(self) -> None:
        """An explicit empty list is forwarded verbatim (caller signalled 'no entities')."""
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
                entity_rows=[],
            )

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("entity_rows") == [], (
            "api.export_specs must distinguish entity_rows=[] from entity_rows=None; " f"got kwargs={kwargs!r}"
        )


# ---------------------------------------------------------------------------
# Round-trip integration tests for export_specs_as_python
# ---------------------------------------------------------------------------

_STREAMING_SHOW_ROW: dict[str, Any] = {
    "name": "CLICKS$V1$ONLINE",
    "database_name": "TESTDB",
    "schema_name": "TESTSCH",
    "scheduling_state": "ACTIVE",
}

_STREAMING_SPEC: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "TESTDB",
        "schema": "TESTSCH",
        "name": "clicks",
        "version": "v1",
    },
    "offline_configs": [],
    "spec": {
        "ordered_entity_column_names": ["user_id"],
        "sources": [
            {
                "name": "click_stream",
                "source_type": "Stream",
                "columns": [
                    {"name": "user_id", "type": "StringType"},
                    {"name": "page", "type": "StringType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "page", "type": "StringType"},
                "output_column": {"name": "page_count", "type": "IntegerType"},
                "function": "count",
                "window_sec": 3600,
            }
        ],
        "udf": {
            "name": "transform",
            "function_definition": "def transform(x):\n    return len(x)",
            "engine": "python",
            "output_columns": [{"name": "page_count", "type": "IntegerType"}],
        },
        "timestamp_field": "ts",
        "feature_granularity_sec": 60,
        "feature_aggregation_method": "tiles",
    },
    "online_store_type": "postgres",
}


# A rich BatchFeatureView spec that populates every branch of
# ``_build_full_fidelity_fv`` that a BFV can legally carry: a non-default
# schema, desc, entities (with a secondary key), timestamp, aggregated
# features, refresh_freq, cluster_by, refresh_mode, initialize, iceberg
# storage_config, aggregation_secondary_keys, and a batch source.  Used by the
# full-fidelity round-trip test so a future renderer omission fails loudly.
_RICH_BFV_SHOW_ROW: dict[str, Any] = {
    "name": "ORDERS$V1$ONLINE",
    "database_name": "DB",
    "schema_name": "SCH",
    "scheduling_state": "ACTIVE",
}

_RICH_BFV_SPEC: dict[str, Any] = {
    "kind": "BatchFeatureView",
    "metadata": {"database": "DB", "schema": "SCH", "name": "orders", "version": "V1"},
    "offline_configs": [],
    "online_store_type": "postgres",
    "desc": "rich bfv",
    "spec": {
        "ordered_entity_column_names": ["USER_ID", "SK1"],
        "ordered_secondary_key_column_names": ["SK1"],
        "aggregation_secondary_keys": ["SK1"],
        "timestamp_field": "TS",
        "refresh_freq": "1 hour",
        "sources": [
            {
                "name": "orders_src",
                "source_type": "Batch",
                "table": "T1",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "features": [
            {
                "source_column": {"name": "amt", "type": "IntType"},
                "output_column": {"name": "amt_sum", "type": "IntType"},
                "function": "sum",
            }
        ],
        "cluster_by": ["USER_ID"],
        "refresh_mode": "FULL",
        "initialize": "on_create",
        "storage_config": {"format": "iceberg", "external_volume": "MY_VOL", "base_location": "loc/x"},
    },
}


class TestApiExportSpecsAsPythonFacade:
    """Pin the api -> exporter forwarding contract for export_specs_as_python."""

    def test_forwards_kwargs_to_exporter(self) -> None:
        """All kwargs must be forwarded verbatim to the underlying exporter."""
        with patch(
            "snowflake.ml.feature_store.decl.exporter.export_specs_as_python",
            return_value={"status": "exported", "directory": "", "files": []},
        ) as mock_exporter:
            decl_api.export_specs_as_python(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir="/tmp/never-written",
                database="DB",
                schema="SCH",
                entity_rows=[],
                feature_group_rows=[],
                layout="sources",
            )
        mock_exporter.assert_called_once()
        kw = mock_exporter.call_args.kwargs
        assert kw.get("entity_rows") == []
        assert kw.get("layout") == "sources"

    def test_default_entity_rows_is_none_when_omitted(self) -> None:
        """Omitting entity_rows must surface as None in the exporter."""
        with patch(
            "snowflake.ml.feature_store.decl.exporter.export_specs_as_python",
            return_value={"status": "exported", "directory": "", "files": []},
        ) as mock_exporter:
            decl_api.export_specs_as_python(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir="/tmp/never-written",
                database="DB",
                schema="SCH",
            )
        kw = mock_exporter.call_args.kwargs
        assert kw.get("entity_rows", "<unset>") is None


class TestExportSpecsAsPythonRoundTrip:
    """Export → ``load_python_file`` → ``spec_to_dict`` round-trip fidelity.

    ``test_py_export_round_trips_every_emitted_field`` is the authoritative
    guard: it compares the exporter's own ``_build_full_fidelity_fv`` output
    against the reloaded spec key by key, so a future renderer omission (like
    the ``description`` / ``schema`` losses fixed in P2 / P3) fails loudly
    instead of slipping past a three-field spot check.
    """

    def test_streaming_fv_round_trip(self, tmp_path: Path) -> None:
        """Exported .py must load back to a spec whose spec_to_dict matches the export dict."""
        from snowflake.ml.feature_store.decl.loader import load_python_file
        from snowflake.ml.feature_store.decl.serializer import spec_to_dict

        result = decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": _STREAMING_SPEC},
        )
        assert result["status"] == "exported"
        assert len(result["files"]) >= 1

        fv_py = tmp_path / "TESTDB.TESTSCH" / "feature_views" / "clicks_v1.py"
        assert fv_py.exists(), "clicks_v1.py must be written"

        specs = load_python_file(str(fv_py))
        assert len(specs) >= 1
        name, obj = specs[0]
        assert name == "clicks_v1"
        assert obj.__class__.__name__ == "StreamingFeatureView"

        d = spec_to_dict(obj)
        assert d["name"] == "clicks"
        assert d["version"] == "v1"
        assert "udf" in d
        assert d["udf"]["name"] == "transform"

    def test_py_export_emits_description(self, tmp_path: Path) -> None:
        """A described FV exported as .py must reload with its description set.

        Mirrors the YAML path (which already emits ``description``) so a described
        FeatureView does not drift to a perpetual ``UPDATE_FV`` on re-plan.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.loader import load_python_file

        spec = {**_STREAMING_SPEC, "desc": "my important comment"}
        decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": spec},
        )
        fv_py = tmp_path / "TESTDB.TESTSCH" / "feature_views" / "clicks_v1.py"
        (specs) = load_python_file(str(fv_py))
        assert len(specs) == 1
        _, obj = specs[0]
        assert obj.description == "my important comment"

    def test_py_export_omits_empty_description(self, tmp_path: Path) -> None:
        """A whitespace-only / absent ``desc`` must not emit a ``description=`` arg."""
        from snowflake.ml.feature_store.decl.loader import load_python_file

        spec = {**_STREAMING_SPEC, "desc": "   "}
        decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": spec},
        )
        fv_py = tmp_path / "TESTDB.TESTSCH" / "feature_views" / "clicks_v1.py"
        assert "description=" not in fv_py.read_text()
        _, obj = load_python_file(str(fv_py))[0]
        assert obj.description is None

    def test_py_export_binds_schema(self, tmp_path: Path) -> None:
        """An exported FV .py must reload with ``schema_`` set to the deployed schema.

        The Pydantic field is ``schema_`` (``schema`` collides with
        ``BaseModel.schema``) and there is no ``schema`` alias, so a generated
        ``schema=`` kwarg is silently discarded — the codegen must emit
        ``schema_=`` instead.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.loader import load_python_file

        decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": _STREAMING_SPEC},
        )
        fv_py = next((tmp_path / "TESTDB.TESTSCH" / "feature_views").glob("*.py"))
        _, obj = load_python_file(str(fv_py))[0]
        assert obj.schema_ == "TESTSCH"
        assert obj.database == "TESTDB"

    def test_py_codegen_feature_group_binds_schema(self, tmp_path: Path) -> None:
        """The FeatureGroup codegen must emit a binding ``schema_=`` kwarg too."""
        from snowflake.ml.feature_store.decl import python_codegen
        from snowflake.ml.feature_store.decl.loader import load_python_file

        fg_spec = {
            "kind": "FeatureGroup",
            "name": "MY_FG",
            "version": "v1",
            "schema": "TESTSCH",
            "database": "TESTDB",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [{"name": "clicks", "version": "v1"}],
        }
        src = python_codegen.spec_dict_to_python_source("FeatureGroup", "MY_FG", fg_spec)
        assert "schema_='TESTSCH'" in src
        fg_py = tmp_path / "MY_FG.py"
        fg_py.write_text(src)
        _, obj = load_python_file(str(fg_py))[0]
        assert obj.schema_ == "TESTSCH"

    def test_py_codegen_falls_back_when_schema_key_is_none(self) -> None:
        """``dict.get`` only defaults on absent keys; ``schema: None`` still takes ``schema_``."""
        from snowflake.ml.feature_store.decl import python_codegen

        fv_src = python_codegen.spec_dict_to_python_source(
            "StreamingFeatureView",
            "clicks",
            {"name": "clicks", "schema": None, "schema_": "SC1"},
        )
        assert "schema_='SC1'" in fv_src

        fg_src = python_codegen.spec_dict_to_python_source(
            "FeatureGroup",
            "MY_FG",
            {
                "kind": "FeatureGroup",
                "name": "MY_FG",
                "schema": None,
                "schema_": "SC1",
                "desc": "",
                "feature_views": [],
            },
        )
        assert "schema_='SC1'" in fg_src

    def test_entity_round_trip(self, tmp_path: Path) -> None:
        """Exported entity .py must load back as an Entity with correct join keys."""
        from snowflake.ml.feature_store.decl.loader import load_python_file
        from snowflake.ml.feature_store.decl.serializer import spec_to_dict

        decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": _STREAMING_SPEC},
            entity_rows=[
                {
                    "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
                    "database_name": "TESTDB",
                    "schema_name": "TESTSCH",
                    "allowed_values": '["USER_ID"]',
                }
            ],
        )
        entity_py = tmp_path / "TESTDB.TESTSCH" / "entities" / "USER_ID.py"
        assert entity_py.exists()

        specs = load_python_file(str(entity_py))
        assert len(specs) >= 1
        _, obj = specs[0]
        assert obj.__class__.__name__ == "Entity"
        d = spec_to_dict(obj)
        assert d["name"] == "USER_ID"
        assert len(d["join_keys"]) >= 1

    def test_all_exported_py_files_are_loadable(self, tmp_path: Path) -> None:
        """Every exported .py file must load via load_python_file without error.

        This is a loadability smoke check only — it does not assert a
        ``NO_CHANGE`` re-plan; the field-level fidelity contract is pinned by
        ``test_py_export_round_trips_every_emitted_field``.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.loader import load_python_file

        result = decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": _STREAMING_SPEC},
            layout="sources",
        )
        for fpath in result["files"]:
            if fpath.endswith(".py"):
                specs = load_python_file(fpath)
                assert len(specs) >= 1, f"load_python_file returned empty list for {fpath}"

    def test_py_export_round_trips_every_emitted_field(self, tmp_path: Path) -> None:
        """The reloaded python spec must reproduce every field the exporter emits.

        Compares ``_build_full_fidelity_fv``'s authoring dict against the
        reloaded ``spec_to_dict`` key by key over a rich BatchFeatureView, so
        any future renderer omission (P2 ``description`` / P3 ``schema`` were
        exactly such omissions) fails here instead of passing a spot check.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl import exporter
        from snowflake.ml.feature_store.decl.loader import load_python_file
        from snowflake.ml.feature_store.decl.serializer import spec_to_dict

        fv_doc = exporter._build_full_fidelity_fv(
            _RICH_BFV_SPEC,
            fallback_name="ORDERS",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SCH",
        )
        decl_api.export_specs_as_python(
            show_rows=[_RICH_BFV_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"ORDERS$V1$ONLINE": _RICH_BFV_SPEC},
        )
        fv_py = next((tmp_path / "DB.SCH" / "feature_views").glob("*.py"))
        _, obj = load_python_file(str(fv_py))[0]
        reloaded = spec_to_dict(obj)

        for key, expected in fv_doc.items():
            model_key = "schema_" if key == "schema" else key
            got = reloaded.get(model_key)
            if key == "sources":
                # ``sources`` on an FV legitimately collapses to a SourceRef
                # ({name, source_type}) on reload — the full source body lives
                # in datasources/.  Compare the referenced names only.
                assert isinstance(got, list) and isinstance(expected, list)
                assert [s.get("name") for s in got] == [
                    s.get("name") for s in expected
                ], f"source refs lost in python round-trip: {got!r}"
                continue
            assert got == expected, f"{key} lost in python round-trip: expected {expected!r}, got {got!r}"

    def test_py_export_streaming_udf_round_trips(self, tmp_path: Path) -> None:
        """A streaming FV with a UDF reloads with its udf, description and schema."""
        from snowflake.ml.feature_store.decl.loader import load_python_file
        from snowflake.ml.feature_store.decl.serializer import spec_to_dict

        spec = {**_STREAMING_SPEC, "desc": "streaming with udf"}
        decl_api.export_specs_as_python(
            show_rows=[_STREAMING_SHOW_ROW],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={"CLICKS$V1$ONLINE": spec},
        )
        fv_py = next((tmp_path / "TESTDB.TESTSCH" / "feature_views").glob("*.py"))
        _, obj = load_python_file(str(fv_py))[0]
        d = spec_to_dict(obj)
        assert d["schema_"] == "TESTSCH"
        assert d["description"] == "streaming with udf"
        assert d["udf"]["name"] == "transform"
        assert "def transform" in d["udf"]["function_definition"]
        assert d["udf"]["output_columns"] == [{"name": "page_count", "type": "IntegerType"}]

    def test_py_multi_version_fv_reload_reports_own_version(self, tmp_path: Path) -> None:
        """Two versions of one FV exported as .py each reload with their own version."""
        from snowflake.ml.feature_store.decl.loader import load_python_file

        def _spec(version: str) -> dict[str, Any]:
            s = {**_STREAMING_SPEC}
            s["metadata"] = {**_STREAMING_SPEC["metadata"], "version": version}
            return s

        decl_api.export_specs_as_python(
            show_rows=[
                {"name": "CLICKS$V1$ONLINE", "database_name": "TESTDB", "schema_name": "TESTSCH"},
                {"name": "CLICKS$V2$ONLINE", "database_name": "TESTDB", "schema_name": "TESTSCH"},
            ],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="TESTDB",
            schema="TESTSCH",
            specification_map={
                "CLICKS$V1$ONLINE": _spec("v1"),
                "CLICKS$V2$ONLINE": _spec("v2"),
            },
        )
        fv_dir = tmp_path / "TESTDB.TESTSCH" / "feature_views"
        versions = {load_python_file(str(p))[0][1].version for p in sorted(fv_dir.glob("*.py"))}
        assert versions == {"v1", "v2"}

    def test_yaml_and_python_forms_agree_structurally(self, tmp_path: Path) -> None:
        """Exporting the same spec both ways yields structurally-equal loaded specs.

        The cheapest guard against the YAML and Python export paths drifting
        apart (it would have caught P1/P2/P3 together).  Two differences are
        excluded by design:

        - ``udf.function_definition`` trailing-newline: the inline-callable
          python form and the sidecar-file YAML form differ only in a trailing
          ``\\n``; compared after ``rstrip()``.
        - ``schema_``: the batch ``load_specs`` path does not inject/rename the
          YAML ``schema`` key (that happens later with an explicit target), so a
          YAML-loaded spec reports ``schema_ = None`` here while the python form
          binds it directly.  This is a loader-path detail outside the exporter
          slice; excluded from this structural comparison.

        Args:
            tmp_path: pytest tmpdir fixture — receives the exported tree.
        """
        from snowflake.ml.feature_store.decl.loader import load_python_file, load_specs
        from snowflake.ml.feature_store.decl.serializer import spec_to_dict

        smap = {"CLICKS$V1$ONLINE": _STREAMING_SPEC}
        row = [{"name": "CLICKS$V1$ONLINE", "database_name": "TESTDB", "schema_name": "TESTSCH"}]
        py_dir = tmp_path / "py"
        yaml_dir = tmp_path / "yaml"
        decl_api.export_specs_as_python(row, {}, str(py_dir), "TESTDB", "TESTSCH", specification_map=smap)
        decl_api.export_specs(row, {}, str(yaml_dir), "TESTDB", "TESTSCH", specification_map=smap)

        py_file = next((py_dir / "TESTDB.TESTSCH" / "feature_views").glob("*.py"))
        _, py_obj = load_python_file(str(py_file))[0]
        py_d = spec_to_dict(py_obj)

        yaml_batch = load_specs([str(yaml_dir / "TESTDB.TESTSCH" / "...")])
        yaml_obj = next(s for s in yaml_batch.specs if s.__class__.__name__ == "StreamingFeatureView")
        yaml_d = spec_to_dict(yaml_obj)

        def _normalize(d: dict[str, Any]) -> dict[str, Any]:
            out = {k: v for k, v in d.items() if k != "schema_"}
            if isinstance(out.get("udf"), dict) and "function_definition" in out["udf"]:
                out["udf"] = {
                    **out["udf"],
                    "function_definition": out["udf"]["function_definition"].rstrip(),
                }
            return out

        assert _normalize(py_d) == _normalize(yaml_d)


class TestPythonCodegenCleanups:
    """P12: python_codegen cleanups — UDF fallback, conditional imports, indentation."""

    def test_udf_source_without_def_raises(self) -> None:
        # A UDF body with no ``def`` is unrecoverable; emitting a module that
        # references an undefined name (``function_definition=_udf_func``) is the
        # wrong response — fail loudly instead.
        from snowflake.ml.feature_store.decl import python_codegen

        with pytest.raises(ValueError):
            python_codegen.spec_dict_to_python_source(
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
        from snowflake.ml.feature_store.decl import python_codegen

        with pytest.raises(ValueError, match="no function_definition"):
            python_codegen.spec_dict_to_python_source(
                "StreamingFeatureView",
                "FV",
                {"name": "FV", "udf": udf},
            )

    def test_entity_without_join_keys_omits_fscolumn_import(self) -> None:
        from snowflake.ml.feature_store.decl import python_codegen

        src = python_codegen.spec_dict_to_python_source("Entity", "e", {"name": "E"})
        import_line = next(line for line in src.splitlines() if line.startswith("from snowflake"))
        assert "FSColumn" not in import_line, f"unused FSColumn import in {import_line!r}"

    def test_udf_output_columns_block_is_correctly_indented(self) -> None:
        from snowflake.ml.feature_store.decl import python_codegen

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
        src = python_codegen.spec_dict_to_python_source("StreamingFeatureView", "FV", spec)
        # ``output_columns=[`` sits at column 8 (a UDF(...) constructor arg), so
        # the nested FSColumn items must be at 12 and the closing ``]`` at 8.
        assert "\n            FSColumn(name='USER_ID', type='StringType'),\n        ]" in src
        # And the emitted source must remain syntactically valid.
        import ast

        ast.parse(src)


class TestApiFormatOpDisplayRow:
    """Pin the ``decl_api.format_op_display_row`` facade the CLI plan path uses."""

    def test_type_precedes_name_from_payload_kind(self) -> None:
        from snowflake.ml.feature_store.decl.enums import OpKind
        from snowflake.ml.feature_store.decl.types import PlanOp

        op = PlanOp(
            kind=OpKind.CREATE_FV,
            name="MY_BFV",
            reason="new",
            payload={"kind": "BatchFeatureView", "name": "MY_BFV", "version": "V1"},
        )
        row = decl_api.format_op_display_row(op)
        assert list(row.keys()) == ["type", "name", "version", "operation", "reason", "destructive"]
        assert row["type"] == "BatchFeatureView"
        assert row["name"] == "MY_BFV"
        assert row["version"] == "V1"
        assert row["operation"] == "CREATE_FV"

    def test_status_included_when_supplied(self) -> None:
        from snowflake.ml.feature_store.decl.enums import OpKind
        from snowflake.ml.feature_store.decl.types import PlanOp

        op = PlanOp(kind=OpKind.CREATE_ENTITY, name="USER_ID", payload={"kind": "Entity"})
        row = decl_api.format_op_display_row(op, status="success")
        assert row["type"] == "Entity"
        assert row["status"] == "success"


if __name__ == "__main__":
    pytest_driver.main()
