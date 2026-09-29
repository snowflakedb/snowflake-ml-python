"""Tests for the ``name_filter`` parameter in exporter.export_specs and
export_specs_as_python.

These tests drive the TDD cycle for the ``snow feature sync --name`` flag:
all tests initially fail because the ``name_filter`` kwarg does not exist.
They pass once Phase 1 (snowml) is implemented.

name_filter behaviour:
  - None (default) → all object kinds written (unchanged from before)
  - exact match, case-insensitive → every deployed object with that name is
    written, across all kinds and all versions.  File stems are distinct per
    (kind, version) and land in per-kind subdirectories, so a name shared by
    (say) a feature view and an entity, or by two versions of one feature
    view, exports all of them and clobbers nothing.
  - no matching object → empty files list (not an error at the exporter level)
  - selecting a name never auto-pulls its dependencies, and an unrelated,
    unrecoverable object must not break a filtered export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.exporter import (
    export_specs,
    export_specs_as_python,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_ENTITY_ROW_USER: dict[str, Any] = {
    "name": "SNOWML_FEATURE_STORE_ENTITY_MY_ENTITY",
    "allowed_values": '["USER_ID"]',
    "comment": "user entity",
}

_ENTITY_ROW_DRIVER: dict[str, Any] = {
    "name": "SNOWML_FEATURE_STORE_ENTITY_DRIVER_ENTITY",
    "allowed_values": '["DRIVER_ID"]',
    "comment": "",
}

_SHOW_ROW_FV: dict[str, Any] = {
    "name": "MY_FV$V1$ONLINE",
    "database_name": "DB",
    "schema_name": "SCH",
}

_SHOW_ROW_FV2: dict[str, Any] = {
    "name": "OTHER_FV$V1$ONLINE",
    "database_name": "DB",
    "schema_name": "SCH",
}

_SPEC_FV: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "DB",
        "schema": "SCH",
        "name": "MY_FV",
        "version": "V1",
    },
    "spec": {
        "ordered_entity_column_names": [],
        "sources": [
            {
                "name": "MY_DS",
                "source_type": "Stream",
                "columns": [{"name": "id", "type": "StringType"}],
            }
        ],
        "features": [],
    },
}

_SPEC_FV2: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "DB",
        "schema": "SCH",
        "name": "OTHER_FV",
        "version": "V1",
    },
    "spec": {
        "ordered_entity_column_names": [],
        "sources": [
            {
                "name": "OTHER_DS",
                "source_type": "Stream",
                "columns": [{"name": "id", "type": "StringType"}],
            }
        ],
        "features": [],
    },
}

_FG_ROW: dict[str, Any] = {
    "name": "MY_FG",
    "version": "V1",
    "desc": "",
    "auto_prefix": True,
    "sources": [{"fv_name": "MY_FV", "fv_version": "V1"}],
}


def _spec_map() -> dict[str, Any]:
    return {
        "MY_FV$V1$ONLINE": _SPEC_FV,
        "OTHER_FV$V1$ONLINE": _SPEC_FV2,
    }


# ---------------------------------------------------------------------------
# export_specs — YAML form
# ---------------------------------------------------------------------------


class TestExportSpecsNameFilterNone:
    """Without name_filter, all object kinds are written."""

    def test_name_filter_none_exports_all_fvs(self, tmp_path: Path) -> None:
        """Both FVs are written when name_filter is omitted."""
        result = export_specs(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_FV2],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map=_spec_map(),
            layout="sources",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.yaml").exists()
        assert (fv_dir / "OTHER_FV_V1.yaml").exists()
        fv_yamls = [f for f in result["files"] if "feature_views" in f and f.endswith(".yaml")]
        assert len(fv_yamls) == 2

    def test_name_filter_none_exports_entities(self, tmp_path: Path) -> None:
        """Entity YAMLs are written for all entity_rows when filter is absent."""
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            entity_rows=[_ENTITY_ROW_USER, _ENTITY_ROW_DRIVER],
            layout="sources",
        )
        ent_dir = tmp_path / "sources" / "entities"
        assert (ent_dir / "MY_ENTITY.yaml").exists()
        assert (ent_dir / "DRIVER_ENTITY.yaml").exists()


class TestExportSpecsNameFilterMatchesFv:
    """name_filter='MY_FV' writes only the MY_FV feature view."""

    def test_only_matching_fv_written(self, tmp_path: Path) -> None:
        export_specs(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_FV2],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map=_spec_map(),
            layout="sources",
            name_filter="MY_FV",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.yaml").exists()
        assert not (fv_dir / "OTHER_FV_V1.yaml").exists()

    def test_entities_not_written_when_fv_filter_set(self, tmp_path: Path) -> None:
        """Filtering by FV name does NOT emit entities (no auto-pull of deps)."""
        result = export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            entity_rows=[_ENTITY_ROW_USER],
            layout="sources",
            name_filter="MY_FV",
        )
        entity_dir = tmp_path / "sources" / "entities"
        assert not entity_dir.exists() or not any(entity_dir.iterdir()), (
            "Filtering by FV name must NOT auto-emit entity files; " f"got files={result['files']!r}"
        )

    def test_datasources_not_written_when_fv_filter_set(self, tmp_path: Path) -> None:
        """Filtering by FV name does NOT emit datasource YAMLs (no auto-pull)."""
        result = export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            layout="sources",
            name_filter="MY_FV",
        )
        ds_dir = tmp_path / "sources" / "datasources"
        assert not ds_dir.exists() or not any(ds_dir.iterdir()), (
            "Filtering by FV name must NOT auto-emit datasource files; " f"got files={result['files']!r}"
        )


class TestExportSpecsNameFilterMatchesEntity:
    """name_filter='MY_ENTITY' writes only the matching entity, skips FVs."""

    def test_only_matching_entity_written(self, tmp_path: Path) -> None:
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            entity_rows=[_ENTITY_ROW_USER, _ENTITY_ROW_DRIVER],
            layout="sources",
            name_filter="MY_ENTITY",
        )
        ent_dir = tmp_path / "sources" / "entities"
        assert (ent_dir / "MY_ENTITY.yaml").exists()
        assert not (ent_dir / "DRIVER_ENTITY.yaml").exists()

    def test_fvs_not_written_when_entity_filter_set(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            entity_rows=[_ENTITY_ROW_USER],
            layout="sources",
            name_filter="MY_ENTITY",
        )
        fv_files = [f for f in result["files"] if "feature_views" in f]
        assert fv_files == [], "Filtering by entity name must NOT emit FV files; " f"got {fv_files!r}"


class TestExportSpecsNameFilterCaseInsensitive:
    """name_filter matching is case-insensitive."""

    def test_lowercase_filter_matches_uppercase_fv_name(self, tmp_path: Path) -> None:
        export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            layout="sources",
            name_filter="my_fv",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (
            fv_dir / "MY_FV_V1.yaml"
        ).exists(), "name_filter='my_fv' must match object named 'MY_FV' (case-insensitive)"

    def test_mixed_case_filter_matches_entity(self, tmp_path: Path) -> None:
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            entity_rows=[_ENTITY_ROW_USER],
            layout="sources",
            name_filter="My_Entity",
        )
        ent_dir = tmp_path / "sources" / "entities"
        assert (ent_dir / "MY_ENTITY.yaml").exists(), "name_filter='My_Entity' must match entity named 'MY_ENTITY'"


class TestExportSpecsNameFilterNoMatch:
    """When no deployed object has the given name, files list is empty."""

    def test_no_match_fv_writes_nothing(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            layout="sources",
            name_filter="NONEXISTENT_FV",
        )
        assert result["files"] == [], "No-match filter must produce empty files list; " f"got {result['files']!r}"
        # A no-match filter is a noop envelope, identical to an empty schema:
        # no ``directory`` and no scaffolded ``sources/`` tree on disk.
        assert result["directory"] == "", f"No-match filter must return an empty directory; got {result['directory']!r}"
        assert not (tmp_path / "sources").exists(), "No-match filter must not scaffold any sources/ directory"

    def test_no_match_entity_writes_nothing(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            entity_rows=[_ENTITY_ROW_USER],
            layout="sources",
            name_filter="NO_SUCH_ENTITY",
        )
        assert result["files"] == [], (
            "No-match entity filter must produce empty files list; " f"got {result['files']!r}"
        )
        assert result["directory"] == "", f"No-match filter must return an empty directory; got {result['directory']!r}"
        assert not (tmp_path / "sources").exists(), "No-match filter must not scaffold any sources/ directory"


class TestExportSpecsNameFilterDoesNotScaffoldUnusedKinds:
    """A partial match writes only its own kind's directory, not siblings."""

    def test_entity_filter_does_not_scaffold_feature_views_dir(self, tmp_path: Path) -> None:
        """``--name MY_ENTITY`` with FVs present writes the entity, not feature_views/."""
        result = export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            entity_rows=[_ENTITY_ROW_USER],
            layout="sources",
            name_filter="MY_ENTITY",
        )
        assert (tmp_path / "sources" / "entities" / "MY_ENTITY.yaml").exists()
        assert str(tmp_path / "sources" / "entities" / "MY_ENTITY.yaml") in result["files"]
        assert not (
            tmp_path / "sources" / "feature_views"
        ).exists(), "An entity-only filter must not scaffold an empty feature_views/ directory"

    def test_fv_filter_does_not_scaffold_feature_groups_dir(self, tmp_path: Path) -> None:
        """``--name MY_FV`` with unrelated FG rows writes the FV, not feature_groups/."""
        export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            feature_group_rows=[_FG_ROW],
            layout="sources",
            name_filter="MY_FV",
        )
        assert (tmp_path / "sources" / "feature_views" / "MY_FV_V1.yaml").exists()
        assert not (
            tmp_path / "sources" / "feature_groups"
        ).exists(), "An FV-only filter must not scaffold an empty feature_groups/ directory"


class TestExportSpecsNameFilterMatchesFeatureGroup:
    """name_filter='MY_FG' writes only the matching feature group."""

    def test_only_matching_fg_written(self, tmp_path: Path) -> None:
        other_fg_row: dict[str, Any] = {
            "name": "OTHER_FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "sources": [{"fv_name": "OTHER_FV", "fv_version": "V1"}],
        }
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            feature_group_rows=[_FG_ROW, other_fg_row],
            layout="sources",
            name_filter="MY_FG",
        )
        fg_dir = tmp_path / "sources" / "feature_groups"
        assert (fg_dir / "MY_FG_V1.yaml").exists()
        assert not (fg_dir / "OTHER_FG_V1.yaml").exists()

    def test_fv_not_written_when_fg_filter_set(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            feature_group_rows=[_FG_ROW],
            layout="sources",
            name_filter="MY_FG",
        )
        fv_files = [f for f in result["files"] if "feature_views" in f]
        assert fv_files == []


class TestExportSpecsNameFilterMatchesDatasource:
    """name_filter='MY_DS' writes only the matching datasource."""

    def test_only_matching_datasource_written(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_FV2],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map=_spec_map(),
            layout="sources",
            name_filter="MY_DS",
        )
        ds_dir = tmp_path / "sources" / "datasources"
        assert (ds_dir / "MY_DS.yaml").exists()
        assert not (ds_dir / "OTHER_DS.yaml").exists()
        fv_files = [f for f in result["files"] if "feature_views" in f]
        assert fv_files == []


# ---------------------------------------------------------------------------
# export_specs_as_python — Python form (parallel to YAML tests)
# ---------------------------------------------------------------------------


class TestExportSpecsAsPythonNameFilter:
    """Parallel name_filter tests for export_specs_as_python."""

    def test_name_filter_none_exports_all_fvs(self, tmp_path: Path) -> None:
        export_specs_as_python(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_FV2],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map=_spec_map(),
            layout="sources",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.py").exists()
        assert (fv_dir / "OTHER_FV_V1.py").exists()

    def test_name_filter_matches_fv(self, tmp_path: Path) -> None:
        export_specs_as_python(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_FV2],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map=_spec_map(),
            layout="sources",
            name_filter="MY_FV",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.py").exists()
        assert not (fv_dir / "OTHER_FV_V1.py").exists()

    def test_name_filter_case_insensitive(self, tmp_path: Path) -> None:
        export_specs_as_python(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            layout="sources",
            name_filter="my_fv",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.py").exists()

    def test_name_filter_no_match_writes_nothing(self, tmp_path: Path) -> None:
        result = export_specs_as_python(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            layout="sources",
            name_filter="DOES_NOT_EXIST",
        )
        assert result["files"] == []
        assert result["directory"] == "", f"No-match filter must return an empty directory; got {result['directory']!r}"
        assert not (tmp_path / "sources").exists(), "No-match filter must not scaffold any sources/ directory"

    def test_entity_filter_does_not_scaffold_feature_views_dir(self, tmp_path: Path) -> None:
        """Python form: ``--name MY_ENTITY`` with FVs present writes no feature_views/."""
        export_specs_as_python(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            entity_rows=[_ENTITY_ROW_USER],
            layout="sources",
            name_filter="MY_ENTITY",
        )
        assert (tmp_path / "sources" / "entities" / "MY_ENTITY.py").exists()
        assert not (
            tmp_path / "sources" / "feature_views"
        ).exists(), "An entity-only filter must not scaffold an empty feature_views/ directory"

    def test_fv_filter_does_not_scaffold_feature_groups_dir(self, tmp_path: Path) -> None:
        """Python form: ``--name MY_FV`` with unrelated FG rows writes no feature_groups/."""
        export_specs_as_python(
            show_rows=[_SHOW_ROW_FV],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            feature_group_rows=[_FG_ROW],
            layout="sources",
            name_filter="MY_FV",
        )
        assert (tmp_path / "sources" / "feature_views" / "MY_FV_V1.py").exists()
        assert not (
            tmp_path / "sources" / "feature_groups"
        ).exists(), "An FV-only filter must not scaffold an empty feature_groups/ directory"

    def test_name_filter_matches_entity(self, tmp_path: Path) -> None:
        export_specs_as_python(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            entity_rows=[_ENTITY_ROW_USER, _ENTITY_ROW_DRIVER],
            layout="sources",
            name_filter="MY_ENTITY",
        )
        ent_dir = tmp_path / "sources" / "entities"
        assert (ent_dir / "MY_ENTITY.py").exists()
        assert not (ent_dir / "DRIVER_ENTITY.py").exists()

    def test_name_filter_matches_feature_group(self, tmp_path: Path) -> None:
        other_fg_row: dict[str, Any] = {
            "name": "OTHER_FG",
            "version": "V1",
            "desc": "",
            "auto_prefix": True,
            "sources": [{"fv_name": "OTHER_FV", "fv_version": "V1"}],
        }
        export_specs_as_python(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={},
            feature_group_rows=[_FG_ROW, other_fg_row],
            layout="sources",
            name_filter="MY_FG",
        )
        fg_dir = tmp_path / "sources" / "feature_groups"
        assert (fg_dir / "MY_FG_V1.py").exists()
        assert not (fg_dir / "OTHER_FG_V1.py").exists()


# ---------------------------------------------------------------------------
# name_filter syncs every deployed object with that name (all kinds, versions)
# ---------------------------------------------------------------------------


class TestExportSpecsNameFilterSyncsAllMatches:
    """A ``name_filter`` resolves to *every* deployed object with that name —
    across all kinds and all versions — and exports them all.  File stems are
    version-qualified and per-kind subdirectories keep the paths distinct, so
    ``snow feature sync --name`` never clobbers a file and never refuses an
    ambiguous name."""

    def test_multiple_versions_same_kind_exports_all_versions(self, tmp_path: Path) -> None:
        """Two deployed versions of one feature view name both export."""
        v1_show = {"name": "MY_FV$V1$ONLINE", "database_name": "DB", "schema_name": "SCH"}
        v2_show = {"name": "MY_FV$V2$ONLINE", "database_name": "DB", "schema_name": "SCH"}
        v1_spec = {
            "kind": "StreamingFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "MY_FV", "version": "V1"},
            "spec": {"ordered_entity_column_names": [], "sources": [], "features": []},
        }
        v2_spec = {
            "kind": "StreamingFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "MY_FV", "version": "V2"},
            "spec": {"ordered_entity_column_names": [], "sources": [], "features": []},
        }
        result = export_specs(
            show_rows=[v1_show, v2_show],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"MY_FV$V1$ONLINE": v1_spec, "MY_FV$V2$ONLINE": v2_spec},
            layout="sources",
            name_filter="MY_FV",
        )
        assert result["status"] == "exported"
        fv_dir = tmp_path / "sources" / "feature_views"
        written = sorted(p.name for p in fv_dir.glob("*.yaml"))
        assert written == ["MY_FV_V1.yaml", "MY_FV_V2.yaml"]

    def test_cross_kind_collision_exports_all(self, tmp_path: Path) -> None:
        """A name shared by a feature view and an entity exports both."""
        collide_fv_show = {
            "name": "SHARED$V1$ONLINE",
            "database_name": "DB",
            "schema_name": "SCH",
        }
        collide_fv_spec = {
            "kind": "StreamingFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "SHARED", "version": "V1"},
            "spec": {"ordered_entity_column_names": [], "sources": [], "features": []},
        }
        collide_entity_row = {
            "name": "SNOWML_FEATURE_STORE_ENTITY_SHARED",
            "allowed_values": '["SHARED_ID"]',
            "comment": "",
        }
        result = export_specs(
            show_rows=[collide_fv_show],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"SHARED$V1$ONLINE": collide_fv_spec},
            entity_rows=[collide_entity_row],
            layout="sources",
            name_filter="SHARED",
        )
        assert result["status"] == "exported"
        assert (tmp_path / "sources" / "feature_views" / "SHARED_V1.yaml").exists()
        assert (tmp_path / "sources" / "entities" / "SHARED.yaml").exists()

    def test_datasource_featureview_name_clash_exports_all(self, tmp_path: Path) -> None:
        """A datasource sharing a feature view's name exports both."""
        fv_show = {"name": "SHARED$V1$ONLINE", "database_name": "DB", "schema_name": "SCH"}
        fv_spec = {
            "kind": "StreamingFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "SHARED", "version": "V1"},
            "spec": {
                "ordered_entity_column_names": [],
                "sources": [
                    {
                        "name": "SHARED",
                        "source_type": "Stream",
                        "columns": [{"name": "id", "type": "StringType"}],
                    }
                ],
                "features": [],
            },
        }
        result = export_specs(
            show_rows=[fv_show],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"SHARED$V1$ONLINE": fv_spec},
            layout="sources",
            name_filter="SHARED",
        )
        assert result["status"] == "exported"
        assert (tmp_path / "sources" / "feature_views" / "SHARED_V1.yaml").exists()
        assert (tmp_path / "sources" / "datasources" / "SHARED.yaml").exists()

    def test_python_cross_kind_collision_exports_all(self, tmp_path: Path) -> None:
        """The Python export path also syncs every matching object."""
        collide_fv_show = {
            "name": "SHARED$V1$ONLINE",
            "database_name": "DB",
            "schema_name": "SCH",
        }
        collide_fv_spec = {
            "kind": "StreamingFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "SHARED", "version": "V1"},
            "spec": {"ordered_entity_column_names": [], "sources": [], "features": []},
        }
        collide_entity_row = {
            "name": "SNOWML_FEATURE_STORE_ENTITY_SHARED",
            "allowed_values": '["SHARED_ID"]',
            "comment": "",
        }
        result = export_specs_as_python(
            show_rows=[collide_fv_show],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"SHARED$V1$ONLINE": collide_fv_spec},
            entity_rows=[collide_entity_row],
            layout="sources",
            name_filter="SHARED",
        )
        assert result["status"] == "exported"
        assert (tmp_path / "sources" / "feature_views" / "SHARED_V1.py").exists()
        assert (tmp_path / "sources" / "entities" / "SHARED.py").exists()


# ---------------------------------------------------------------------------
# Broken-spec fixtures: objects that *have* a SPECIFICATION which is invalid as
# a datasource.  Unlike the specless OFT above, these exercise the datasource
# collection path, which must be scoped to the filter so an unrelated FV's
# malformed source cannot abort a filtered export.
# ---------------------------------------------------------------------------

_SHOW_ROW_BAD_SRC: dict[str, Any] = {
    "name": "BAD_FV$V1$ONLINE",
    "database_name": "DB",
    "schema_name": "SCH",
}

_SPEC_BAD_SOURCE_TYPE: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {"database": "DB", "schema": "SCH", "name": "BAD_FV", "version": "V1"},
    "spec": {
        "ordered_entity_column_names": [],
        "sources": [
            {
                "name": "BAD_DS",
                "source_type": "Bogus",
                "columns": [{"name": "id", "type": "StringType"}],
            }
        ],
        "features": [],
    },
}

_SHOW_ROW_CONFLICT_A: dict[str, Any] = {
    "name": "CONFLICT_A$V1$ONLINE",
    "database_name": "DB",
    "schema_name": "SCH",
}

_SHOW_ROW_CONFLICT_B: dict[str, Any] = {
    "name": "CONFLICT_B$V1$ONLINE",
    "database_name": "DB",
    "schema_name": "SCH",
}

_SPEC_CONFLICT_A: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {"database": "DB", "schema": "SCH", "name": "CONFLICT_A", "version": "V1"},
    "spec": {
        "ordered_entity_column_names": [],
        "sources": [
            {
                "name": "CONFLICT_DS",
                "source_type": "Stream",
                "columns": [{"name": "id", "type": "StringType"}],
            }
        ],
        "features": [],
    },
}

# Same source name as CONFLICT_A but a conflicting column ``type`` — collecting
# both together raises a datasource column-type conflict.
_SPEC_CONFLICT_B: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {"database": "DB", "schema": "SCH", "name": "CONFLICT_B", "version": "V1"},
    "spec": {
        "ordered_entity_column_names": [],
        "sources": [
            {
                "name": "CONFLICT_DS",
                "source_type": "Stream",
                "columns": [{"name": "id", "type": "IntType"}],
            }
        ],
        "features": [],
    },
}


class TestExportSpecsNameFilterIsolatesInvalid:
    """A filtered export validates only the object it selects; an unrelated
    object that cannot be recovered (no SPECIFICATION) must not break it, but
    a *matching* object that cannot be recovered still fails."""

    def test_unrelated_specless_oft_does_not_break_match(self, tmp_path: Path) -> None:
        """An unrelated specless OFT is skipped when the filter targets another object."""
        broken_show = {"name": "BROKEN_FV$V1$ONLINE", "database_name": "DB", "schema_name": "SCH"}
        result = export_specs(
            show_rows=[_SHOW_ROW_FV, broken_show],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            # Note: BROKEN_FV$V1$ONLINE deliberately has no spec entry.
            specification_map={"MY_FV$V1$ONLINE": _SPEC_FV},
            layout="sources",
            name_filter="MY_FV",
        )
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.yaml").exists()
        assert not (fv_dir / "BROKEN_FV_V1.yaml").exists()
        fv_yamls = [f for f in result["files"] if "feature_views" in f and f.endswith(".yaml")]
        assert len(fv_yamls) == 1

    def test_matching_specless_oft_still_raises(self, tmp_path: Path) -> None:
        """A filter that selects an OFT with no SPECIFICATION still raises."""
        broken_show = {"name": "MY_FV$V1$ONLINE", "database_name": "DB", "schema_name": "SCH"}
        with pytest.raises(ValueError, match="SPECIFICATION"):
            export_specs(
                show_rows=[broken_show],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="DB",
                schema="SCH",
                specification_map={},
                layout="sources",
                name_filter="MY_FV",
            )

    def test_unrelated_bad_source_type_does_not_break_fv_filter(self, tmp_path: Path) -> None:
        """An unrelated FV's unrecognised ``source_type`` must not abort ``--name MY_FV``."""
        result = export_specs(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_BAD_SRC],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "MY_FV$V1$ONLINE": _SPEC_FV,
                "BAD_FV$V1$ONLINE": _SPEC_BAD_SOURCE_TYPE,
            },
            layout="sources",
            name_filter="MY_FV",
        )
        assert result["status"] == "exported"
        fv_dir = tmp_path / "sources" / "feature_views"
        assert (fv_dir / "MY_FV_V1.yaml").exists()
        assert not (fv_dir / "BAD_FV_V1.yaml").exists()

    def test_unrelated_bad_source_type_does_not_break_datasource_filter(self, tmp_path: Path) -> None:
        """An unrelated FV's bad ``source_type`` must not abort ``--name MY_DS``."""
        # The datasource is discovered across FVs (``MY_DS`` lives in
        # ``MY_FV``'s sources), so it is still written while the unrelated bad
        # source is skipped before validation.
        result = export_specs(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_BAD_SRC],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "MY_FV$V1$ONLINE": _SPEC_FV,
                "BAD_FV$V1$ONLINE": _SPEC_BAD_SOURCE_TYPE,
            },
            layout="sources",
            name_filter="MY_DS",
        )
        assert result["status"] == "exported"
        ds_dir = tmp_path / "sources" / "datasources"
        assert (ds_dir / "MY_DS.yaml").exists()
        assert not (ds_dir / "BAD_DS.yaml").exists()

    def test_unrelated_column_conflict_does_not_break_fv_filter(self, tmp_path: Path) -> None:
        """A cross-FV column-type conflict on a different source must not abort ``--name MY_FV``."""
        result = export_specs(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_CONFLICT_A, _SHOW_ROW_CONFLICT_B],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "MY_FV$V1$ONLINE": _SPEC_FV,
                "CONFLICT_A$V1$ONLINE": _SPEC_CONFLICT_A,
                "CONFLICT_B$V1$ONLINE": _SPEC_CONFLICT_B,
            },
            layout="sources",
            name_filter="MY_FV",
        )
        assert result["status"] == "exported"
        assert (tmp_path / "sources" / "feature_views" / "MY_FV_V1.yaml").exists()

    def test_matching_datasource_conflict_still_raises(self, tmp_path: Path) -> None:
        """A genuine conflict on the *filtered* datasource still raises."""
        with pytest.raises(ValueError, match="conflict"):
            export_specs(
                show_rows=[_SHOW_ROW_CONFLICT_A, _SHOW_ROW_CONFLICT_B],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="DB",
                schema="SCH",
                specification_map={
                    "CONFLICT_A$V1$ONLINE": _SPEC_CONFLICT_A,
                    "CONFLICT_B$V1$ONLINE": _SPEC_CONFLICT_B,
                },
                layout="sources",
                name_filter="CONFLICT_DS",
            )

    def test_unfiltered_export_still_raises_on_bad_source_type(self, tmp_path: Path) -> None:
        """Without a filter, schema-wide datasource validation is unchanged."""
        with pytest.raises(ValueError, match="source_type"):
            export_specs(
                show_rows=[_SHOW_ROW_FV, _SHOW_ROW_BAD_SRC],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="DB",
                schema="SCH",
                specification_map={
                    "MY_FV$V1$ONLINE": _SPEC_FV,
                    "BAD_FV$V1$ONLINE": _SPEC_BAD_SOURCE_TYPE,
                },
                layout="sources",
            )

    def test_python_unrelated_bad_source_type_does_not_break_fv_filter(self, tmp_path: Path) -> None:
        """The Python export path isolates an unrelated bad source too."""
        result = export_specs_as_python(
            show_rows=[_SHOW_ROW_FV, _SHOW_ROW_BAD_SRC],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "MY_FV$V1$ONLINE": _SPEC_FV,
                "BAD_FV$V1$ONLINE": _SPEC_BAD_SOURCE_TYPE,
            },
            layout="sources",
            name_filter="MY_FV",
        )
        assert result["status"] == "exported"
        assert (tmp_path / "sources" / "feature_views" / "MY_FV_V1.py").exists()


# ---------------------------------------------------------------------------
# Orphan-OFT diagnostic scope: the ``orphaned_oft_warnings`` diagnostic keys on
# the deployed OFT identity just like the FV write loop, so a filtered export
# must only warn about OFTs it may actually emit.  The known-FV set stays
# schema-wide (an OFT is judged orphaned against the full registry, never a
# one-row slice of it).
# ---------------------------------------------------------------------------


def _scope_show_row(name: str) -> dict[str, Any]:
    return {
        "name": f"{name}$V1$ONLINE",
        "database_name": "DB",
        "schema_name": "SCH",
    }


def _scope_spec(name: str) -> dict[str, Any]:
    # A recoverable per-FV source so the FV is exportable on every branch,
    # including 6f1's source-less-export guard.  Distinct source names avoid a
    # cross-FV datasource column conflict; the orphan diagnostic (keyed on the
    # OFT identity, not the source) is unaffected by the source shape.
    return {
        "kind": "StreamingFeatureView",
        "metadata": {"database": "DB", "schema": "SCH", "name": name, "version": "V1"},
        "spec": {
            "ordered_entity_column_names": [],
            "sources": [
                {
                    "name": f"{name}_SRC",
                    "source_type": "Stream",
                    "columns": [{"name": "id", "type": "StringType"}],
                }
            ],
            "features": [],
        },
    }


def _scope_fv_row(name: str) -> dict[str, Any]:
    return {"name": name, "version": "V1", "database_name": "DB", "schema_name": "SCH"}


class TestExportSpecsNameFilterScopesWarnings:
    """A filtered export warns only about OFTs it may write.

    ``GHOST_FV`` deliberately has a recoverable SPECIFICATION but no matching
    ``feature_view_rows`` entry, so the orphan-OFT diagnostic fires for it
    unless the filter scopes the diagnostic away.
    """

    def test_unrelated_orphan_not_warned_under_fv_filter(self, tmp_path: Path) -> None:
        """``--name GOOD_FV`` must not warn about an unrelated orphan ``GHOST_FV``."""
        result = export_specs(
            show_rows=[_scope_show_row("GOOD_FV"), _scope_show_row("GHOST_FV")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "GOOD_FV$V1$ONLINE": _scope_spec("GOOD_FV"),
                "GHOST_FV$V1$ONLINE": _scope_spec("GHOST_FV"),
            },
            feature_view_rows=[_scope_fv_row("GOOD_FV")],
            layout="sources",
            name_filter="GOOD_FV",
        )
        assert (tmp_path / "sources" / "feature_views" / "GOOD_FV_V1.yaml").exists()
        assert not any(
            "GHOST_FV" in w for w in result["warnings"]
        ), f"filtered export must not warn about the unrelated orphan; got {result['warnings']!r}"

    def test_matching_orphan_still_warned(self, tmp_path: Path) -> None:
        """``--name GHOST_FV`` still warns when the selected object is the orphan."""
        result = export_specs(
            show_rows=[_scope_show_row("GOOD_FV"), _scope_show_row("GHOST_FV")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "GOOD_FV$V1$ONLINE": _scope_spec("GOOD_FV"),
                "GHOST_FV$V1$ONLINE": _scope_spec("GHOST_FV"),
            },
            feature_view_rows=[_scope_fv_row("GOOD_FV")],
            layout="sources",
            name_filter="GHOST_FV",
        )
        assert any(
            "GHOST_FV" in w for w in result["warnings"]
        ), f"the selected orphan must still be surfaced; got {result['warnings']!r}"

    def test_known_set_is_not_filtered(self, tmp_path: Path) -> None:
        """A healthy filtered OFT is judged against the full registry, not a slice."""
        result = export_specs(
            show_rows=[_scope_show_row("GOOD_FV")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={"GOOD_FV$V1$ONLINE": _scope_spec("GOOD_FV")},
            # Multi-row registry: if the diagnostic filtered the known set it
            # would judge GOOD_FV against a one-row slice and false-flag it.
            feature_view_rows=[_scope_fv_row("GOOD_FV"), _scope_fv_row("OTHER_FV")],
            layout="sources",
            name_filter="GOOD_FV",
        )
        assert result["warnings"] == [], f"a healthy filtered OFT must not warn; got {result['warnings']!r}"

    def test_lowercase_filter_suppresses_unrelated_orphan(self, tmp_path: Path) -> None:
        """The scope check is case-insensitive through the shared helper."""
        result = export_specs(
            show_rows=[_scope_show_row("GOOD_FV"), _scope_show_row("GHOST_FV")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "GOOD_FV$V1$ONLINE": _scope_spec("GOOD_FV"),
                "GHOST_FV$V1$ONLINE": _scope_spec("GHOST_FV"),
            },
            feature_view_rows=[_scope_fv_row("GOOD_FV")],
            layout="sources",
            name_filter="good_fv",
        )
        assert not any("GHOST_FV" in w for w in result["warnings"])

    def test_unfiltered_export_still_warns(self, tmp_path: Path) -> None:
        """Without a filter, the schema-wide orphan diagnostic is unchanged."""
        result = export_specs(
            show_rows=[_scope_show_row("GOOD_FV"), _scope_show_row("GHOST_FV")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "GOOD_FV$V1$ONLINE": _scope_spec("GOOD_FV"),
                "GHOST_FV$V1$ONLINE": _scope_spec("GHOST_FV"),
            },
            feature_view_rows=[_scope_fv_row("GOOD_FV")],
            layout="sources",
        )
        assert any(
            "GHOST_FV" in w for w in result["warnings"]
        ), f"unfiltered export must still warn about the orphan; got {result['warnings']!r}"

    def test_python_unrelated_orphan_not_warned_under_fv_filter(self, tmp_path: Path) -> None:
        """The Python export path scopes the orphan diagnostic too."""
        result = export_specs_as_python(
            show_rows=[_scope_show_row("GOOD_FV"), _scope_show_row("GHOST_FV")],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="DB",
            schema="SCH",
            specification_map={
                "GOOD_FV$V1$ONLINE": _scope_spec("GOOD_FV"),
                "GHOST_FV$V1$ONLINE": _scope_spec("GHOST_FV"),
            },
            feature_view_rows=[_scope_fv_row("GOOD_FV")],
            layout="sources",
            name_filter="GOOD_FV",
        )
        assert (tmp_path / "sources" / "feature_views" / "GOOD_FV_V1.py").exists()
        assert not any("GHOST_FV" in w for w in result["warnings"])


if __name__ == "__main__":
    pytest_driver.main()
