"""Tests — exporter writes ``feature_groups/<NAME>.yaml``.

Pin the contract:

* ``export_specs(..., feature_group_rows=...)`` writes one
  ``<base>/feature_groups/<NAME>.yaml`` per row, in the canonical authoring
  shape (``feature_views: [{name, version, slice_columns?, alias?}]``).
* Slice columns and aliases (including the literal empty string) survive
  the round-trip.
* When ``feature_group_rows`` is ``None`` or ``[]``, no FG YAMLs are
  written and the early-exit envelope is preserved.
* The base directory under ``layout="sources"`` is ``<output_dir>/sources``;
  FG YAMLs land in its ``feature_groups/`` subdirectory.

In addition, the ``TestBuildFgYamlDocVersion`` /
``TestFgSourceToAuthoringDictVersion`` / ``TestExportSpecsFgYamlVersion``
suites at the bottom of this file pin the FG ``version`` round-trip
contract — see the docstring on ``TestBuildFgYamlDocVersion`` for the
"why" (the existing hash-only round-trip test masked a regression where
the FG exporter could emit ``version: null``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from snowflake.ml.feature_store.decl.exporter import (
    _build_fg_yaml_doc,
    _fg_source_to_authoring_dict,
    export_specs,
)
from snowflake.ml.test_utils import pytest_driver


def _fg_row(
    *,
    name: str = "USER_FRAUD_FG",
    version: str = "V1",
    desc: str = "",
    auto_prefix: bool = True,
    sources: list[dict[str, Any]] | None = None,
    output_columns: list[str] | None = None,
    database_name: str = "MYDB",
    schema_name: str = "PUBLIC",
) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "desc": desc,
        "owner": "ROLE",
        "auto_prefix": auto_prefix,
        "sources": sources if sources is not None else [{"fv_name": "USER_CLICKS", "fv_version": "V1"}],
        "output_columns": output_columns,
        "database_name": database_name,
        "schema_name": schema_name,
    }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestExportSpecsFGEmpty:
    def test_no_show_no_entity_no_fg_yields_noop(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=[],
            layout="sources",
        )
        assert result["status"] == "exported"
        assert result["directory"] == ""
        assert result["files"] == []


# ---------------------------------------------------------------------------
# FG YAML emission
# ---------------------------------------------------------------------------


class TestExportSpecsFGEmission:
    def test_writes_single_fg_yaml(self, tmp_path: Path) -> None:
        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=[_fg_row(name="USER_FRAUD_FG", desc="hi")],
            layout="sources",
        )

        # Base under sources layout is <output_dir>/sources.
        base = tmp_path / "sources"
        fg_dir = base / "feature_groups"
        fg_path = fg_dir / "USER_FRAUD_FG_V1.yaml"
        assert fg_path.exists(), f"expected {fg_path}, got {result['files']}"
        # The result envelope lists the file.
        assert str(fg_path) in result["files"]

        # Verify YAML round-trips to the canonical authoring shape.
        parsed = yaml.safe_load(fg_path.read_text())
        assert parsed["kind"] == "FeatureGroup"
        assert parsed["name"] == "USER_FRAUD_FG"
        assert parsed["version"] == "V1"
        assert parsed["desc"] == "hi"
        assert parsed["auto_prefix"] is True
        assert parsed["feature_views"] == [{"name": "USER_CLICKS", "version": "V1"}]

    def test_preserves_slice_columns_and_alias(self, tmp_path: Path) -> None:
        sources: list[dict[str, Any]] = [
            {"fv_name": "FV_A", "fv_version": "V1"},
            {
                "fv_name": "FV_B",
                "fv_version": "V1",
                "slice_columns": ["X", "Y"],
                "alias": "b",
            },
        ]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=[_fg_row(name="FG_X", sources=sources)],
            layout="sources",
        )
        fg_path = tmp_path / "sources" / "feature_groups" / "FG_X_V1.yaml"
        parsed = yaml.safe_load(fg_path.read_text())
        # Translation from imperative ``fv_name`` / ``fv_version`` to
        # declarative ``name`` / ``version`` must happen during emission.
        assert parsed["feature_views"] == [
            {"name": "FV_A", "version": "V1"},
            {
                "name": "FV_B",
                "version": "V1",
                "slice_columns": ["X", "Y"],
                "alias": "b",
            },
        ]

    def test_alias_empty_string_preserved(self, tmp_path: Path) -> None:
        sources = [
            {"fv_name": "FV_A", "fv_version": "V1", "alias": ""},
        ]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=[_fg_row(name="FG_E", sources=sources)],
            layout="sources",
        )
        fg_path = tmp_path / "sources" / "feature_groups" / "FG_E_V1.yaml"
        parsed = yaml.safe_load(fg_path.read_text())
        # alias="" is preserved as an explicit YAML key (semantically
        # "no prefix"); distinct from the absent-key default which means
        # "use auto_prefix".
        assert parsed["feature_views"] == [{"name": "FV_A", "version": "V1", "alias": ""}]

    def test_multiple_fgs_each_get_own_yaml(self, tmp_path: Path) -> None:
        rows = [
            _fg_row(name="FG_A"),
            _fg_row(name="FG_B"),
        ]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )
        fg_dir = tmp_path / "sources" / "feature_groups"
        assert (fg_dir / "FG_A_V1.yaml").exists()
        assert (fg_dir / "FG_B_V1.yaml").exists()


# ---------------------------------------------------------------------------
# FG version round-trip — strict-fail contract
# ---------------------------------------------------------------------------
#
# Pinned by the FG export ``version`` round-trip plan
# (``plans/fg-export-version-fix_*.plan.md``).  The existing
# ``test_export_plan_round_trip_feature_group.py`` checks hash-equality,
# but a missing/empty ``version`` on both sides hashes the same — masking
# the bug that the FG exporter could silently emit ``version: null`` into
# the YAML, which then re-validates as ``MISSING_VERSION`` on the next
# ``snow feature plan``.
#
# These suites close that gap by asserting:
#   1. ``_build_fg_yaml_doc`` always emits a literal ``"version"`` key.
#   2. ``export_specs`` writes a YAML on disk whose ``version:`` key
#      parses back to the input value (not ``null`` / missing).
#   3. Both code paths *strict-fail* (raise :class:`ValueError`) when the
#      input row lacks a non-empty ``version`` — refusing to emit a YAML
#      that would re-validate as ``MISSING_VERSION`` is the contract that
#      protects the export round-trip from upstream regressions.


def _versioned_fg_row(
    *,
    name: str = "USER_FRAUD_FG_DECL",
    version: str = "V1",
    desc: str = "Round-trip test FG.",
    auto_prefix: bool = True,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Synthetic FG row in the shape produced by
    # imperative_executor.fetch_feature_group_rows, defaulting to a
    # well-formed two-source ref list.  Distinct from `_fg_row` above
    # so each suite gets a stable default it controls.
    return {
        "name": name,
        "version": version,
        "desc": desc,
        "owner": "ROLE_X",
        "auto_prefix": auto_prefix,
        "sources": (
            sources
            if sources is not None
            else [
                {"fv_name": "USER_CLICKS_FG_DECL", "fv_version": "V1"},
                {"fv_name": "USER_AMOUNTS_FG_DECL", "fv_version": "V1"},
            ]
        ),
        "output_columns": None,
        "database_name": "MYDB",
        "schema_name": "PUBLIC",
    }


class TestBuildFgYamlDocVersion:
    """Pin the FG ``version`` round-trip contract on ``_build_fg_yaml_doc``.

    The existing hash-only round-trip masks regressions where the
    exporter could emit ``version: null`` (or omit the key entirely),
    because empty/empty hashes the same on both sides.  These tests
    assert the literal key and a strict-fail when the upstream row
    lacks a non-empty ``version``.
    """

    def test_build_fg_yaml_doc_emits_version_key(self) -> None:
        doc = _build_fg_yaml_doc(_versioned_fg_row(version="V1"))

        assert "version" in doc
        assert doc["version"] == "V1"

    def test_build_fg_yaml_doc_preserves_non_default_version(self) -> None:
        doc = _build_fg_yaml_doc(_versioned_fg_row(version="2025_05_28_001"))

        assert doc["version"] == "2025_05_28_001"

    def test_build_fg_yaml_doc_raises_on_missing_version(self) -> None:
        """Refusing to emit a YAML that would re-validate as
        ``MISSING_VERSION`` is the contract that protects the export
        round-trip from upstream metadata regressions.
        """
        row = _versioned_fg_row()
        del row["version"]

        with pytest.raises(ValueError, match="USER_FRAUD_FG_DECL"):
            _build_fg_yaml_doc(row)

    def test_build_fg_yaml_doc_raises_on_empty_version(self) -> None:
        with pytest.raises(ValueError, match="USER_FRAUD_FG_DECL"):
            _build_fg_yaml_doc(_versioned_fg_row(version=""))

    def test_build_fg_yaml_doc_raises_on_none_version(self) -> None:
        """``None`` version (e.g. from a JSON-decoded NULL cell) is a
        strict-fail — without this guard, ``yaml.dump`` would emit
        ``version: null`` and the next load would re-validate as
        ``MISSING_VERSION``.
        """
        with pytest.raises(ValueError, match="USER_FRAUD_FG_DECL"):
            _build_fg_yaml_doc(_versioned_fg_row(version=None))  # type: ignore[arg-type]


class TestFgSourceToAuthoringDictVersion:
    """Pin the strict-fail contract on per-source FV refs.  A ref without
    ``fv_name`` / ``fv_version`` would re-load as
    ``FeatureViewRef(version="")``, which the Pydantic validator rejects
    at load time — we want the failure surfaced at export time so the
    operator sees the actual upstream metadata problem instead of a
    downstream loader error.
    """

    def test_translates_well_formed_source(self) -> None:
        out = _fg_source_to_authoring_dict({"fv_name": "USER_CLICKS_FG_DECL", "fv_version": "V1"})

        assert out == {"name": "USER_CLICKS_FG_DECL", "version": "V1"}

    def test_raises_on_missing_fv_version(self) -> None:
        with pytest.raises(ValueError, match="USER_CLICKS_FG_DECL"):
            _fg_source_to_authoring_dict({"fv_name": "USER_CLICKS_FG_DECL"})

    def test_raises_on_empty_fv_version(self) -> None:
        with pytest.raises(ValueError, match="USER_CLICKS_FG_DECL"):
            _fg_source_to_authoring_dict({"fv_name": "USER_CLICKS_FG_DECL", "fv_version": ""})

    def test_raises_on_missing_fv_name(self) -> None:
        with pytest.raises(ValueError, match="fv_name"):
            _fg_source_to_authoring_dict({"fv_version": "V1"})


class TestExportSpecsFgYamlVersion:
    """End-to-end: drive ``export_specs`` and assert the literal
    ``version:`` key in the on-disk YAML, plus that strict-fail
    propagates through the public entry point without writing a
    partial-state YAML.
    """

    def test_export_specs_writes_version_to_yaml(self, tmp_path: Path) -> None:
        rows = [_versioned_fg_row(name="USER_FRAUD_FG_DECL", version="V1")]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )

        fg_yaml = tmp_path / "sources" / "feature_groups" / "USER_FRAUD_FG_DECL_V1.yaml"
        assert fg_yaml.exists()
        doc = yaml.safe_load(fg_yaml.read_text())

        assert "version" in doc, (
            "exported FG YAML is missing the 'version:' key — it would "
            "re-validate as MISSING_VERSION on the next snow feature plan"
        )
        assert doc["version"] == "V1"

    def test_export_specs_yaml_text_contains_version_line(self, tmp_path: Path) -> None:
        # Belt-and-suspenders: assert the literal substring
        # ``version: V1`` appears in the YAML text.  Catches a
        # regression where the key is present but written as
        # ``version: null`` (which would pass the ``"version" in doc``
        # check above but break the loader).
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database="MYDB",
            schema="PUBLIC",
            specification_map={},
            entity_rows=[],
            feature_group_rows=[_versioned_fg_row()],
            layout="sources",
        )

        fg_yaml = tmp_path / "sources" / "feature_groups" / "USER_FRAUD_FG_DECL_V1.yaml"
        text = fg_yaml.read_text()

        assert "version: V1" in text, f"expected 'version: V1' in exported FG YAML, got:\n{text}"
        assert "version: null" not in text
        assert "version: ''" not in text

    def test_export_specs_raises_on_fg_row_without_version(self, tmp_path: Path) -> None:
        # Strict-fail propagates through ``export_specs``: a malformed
        # FG row aborts the export with a ``ValueError`` naming the FG.
        row = _versioned_fg_row()
        del row["version"]

        with pytest.raises(ValueError, match="USER_FRAUD_FG_DECL"):
            export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={},
                entity_rows=[],
                feature_group_rows=[row],
                layout="sources",
            )

        fg_yaml = tmp_path / "sources" / "feature_groups" / "USER_FRAUD_FG_DECL_V1.yaml"
        assert not fg_yaml.exists()

    def test_export_specs_raises_on_source_ref_without_version(self, tmp_path: Path) -> None:
        # Strict-fail also catches a malformed source FV ref — the
        # loader would reject ``FeatureViewRef(version="")`` at load
        # time, but raising at export time tells the operator the
        # upstream metadata is broken instead of producing a YAML that
        # won't load.
        row = _versioned_fg_row(
            sources=[
                {"fv_name": "USER_CLICKS_FG_DECL", "fv_version": "V1"},
                {"fv_name": "USER_AMOUNTS_FG_DECL"},  # missing fv_version
            ]
        )

        with pytest.raises(ValueError, match="USER_AMOUNTS_FG_DECL"):
            export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="MYDB",
                schema="PUBLIC",
                specification_map={},
                entity_rows=[],
                feature_group_rows=[row],
                layout="sources",
            )


if __name__ == "__main__":
    pytest_driver.main()
