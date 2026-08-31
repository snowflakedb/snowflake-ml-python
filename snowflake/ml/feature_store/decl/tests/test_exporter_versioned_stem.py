"""RED — exporter must not lose a version to a filename-stem collision.

Applied state can hold several versions per name (the 6b1 identity work).
But :func:`exporter.export_specs` builds FV / FG YAML (and UDF sidecar) file
stems from the ``name`` alone, so two versions of one name map to the same
path: the second FV ``write_text`` overwrites the first, and the FG loop
skips the duplicate via ``seen_fg_files``.  Only one version's YAML lands on
disk, so a following ``snow feature plan ./...`` (``full_directory_mode=True``)
sees the missing version as an orphan and proposes a destructive ``DROP``.

These tests pin:

1. Two versions of one FV name export to two distinct files.
2. A freshly-exported two-version FV tree plans to zero ``DROP_FV`` under
   ``full_directory_mode=True`` (the reviewer's end-to-end scenario).
3. + 4. The FG analogs (distinct files, zero ``DROP_FG``).
5. A residual stem collision raises ``ValueError`` instead of silently
   overwriting / skipping a version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.exporter import export_specs
from snowflake.ml.feature_store.decl.loader import load_from_project, load_specs
from snowflake.ml.feature_store.decl.types import PlanOptions
from snowflake.ml.test_utils import pytest_driver

_DB = "MYDB"
_SCHEMA = "PUBLIC"


def _show_row(name: str, version: str) -> dict[str, Any]:
    return {
        "name": f"{name.upper()}${version.upper()}$ONLINE",
        "database_name": _DB,
        "schema_name": _SCHEMA,
        "scheduling_state": "ACTIVE",
    }


def _oft_name(name: str, version: str) -> str:
    return f"{name.upper()}${version.upper()}$ONLINE"


def _batch_spec(name: str, version: str) -> dict[str, Any]:
    # SPECIFICATION-shape offline BatchFeatureView payload (mirrors the
    # proven-clean ``_BATCH_SPEC`` fixture in test_export_plan_round_trip.py)
    # so the exported YAML round-trips to NO_CHANGE and the only signal under
    # test is the two-version stem behaviour.
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": name,
            "version": version,
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "0.1.0",
        },
        "offline_configs": [
            {
                "store_type": "snowflake",
                "table_type": "BatchSource",
                "database": _DB,
                "schema": _SCHEMA,
                "table": f"{name.upper()}${version.upper()}",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "UPDATED_AT", "type": "TimestampType"},
                    {"name": "CITY", "type": "StringType"},
                ],
            }
        ],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [
                {
                    "source_column": {"name": "CITY", "type": "StringType"},
                    "output_column": {"name": "CITY", "type": "StringType"},
                }
            ],
            "timestamp_field": "UPDATED_AT",
            "target_lag_sec": 10,
        },
        "online_store_type": "postgres",
    }


def _user_id_entity_row() -> dict[str, Any]:
    return {
        "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
        "database_name": _DB,
        "schema_name": _SCHEMA,
        "allowed_values": '["USER_ID"]',
    }


def _fv_applied_two_versions(name: str, versions: list[str]) -> Any:
    show_rows = [_show_row(name, v) for v in versions]
    spec_map = {_oft_name(name, v): _batch_spec(name, v) for v in versions}
    return decl_api.fetch_applied_state(
        show_rows,
        None,
        specification_map=spec_map,
        entity_rows=[_user_id_entity_row()],
        default_database=_DB,
        default_schema=_SCHEMA,
    )


def _fg_row(name: str, version: str) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "desc": "",
        "owner": "ROLE",
        "auto_prefix": True,
        "sources": [{"fv_name": "USER_CLICKS", "fv_version": "V1"}],
        "output_columns": None,
        "database_name": _DB,
        "schema_name": _SCHEMA,
    }


# ---------------------------------------------------------------------------
# FeatureView — two versions must not collide
# ---------------------------------------------------------------------------


class TestVersionedFeatureViewStemCollision:
    def test_two_fv_versions_write_distinct_files(self, tmp_path: Path) -> None:
        versions = ["V1", "V2"]
        export_specs(
            show_rows=[_show_row("MY_FV", v) for v in versions],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database=_DB,
            schema=_SCHEMA,
            specification_map={_oft_name("MY_FV", v): _batch_spec("MY_FV", v) for v in versions},
            entity_rows=[_user_id_entity_row()],
        )

        fv_dir = tmp_path / f"{_DB}.{_SCHEMA}" / "feature_views"
        assert (fv_dir / "MY_FV_V1.yaml").exists(), (
            "V1 FV YAML missing — the name-only stem let V2 overwrite it. "
            f"present: {sorted(p.name for p in fv_dir.glob('*.yaml'))}"
        )
        assert (fv_dir / "MY_FV_V2.yaml").exists(), (
            "V2 FV YAML missing. " f"present: {sorted(p.name for p in fv_dir.glob('*.yaml'))}"
        )

    def test_two_fv_versions_round_trip_no_drop(self, tmp_path: Path) -> None:
        versions = ["V1", "V2"]
        applied_state = _fv_applied_two_versions("MY_FV", versions)

        export_specs(
            show_rows=[_show_row("MY_FV", v) for v in versions],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database=_DB,
            schema=_SCHEMA,
            specification_map={_oft_name("MY_FV", v): _batch_spec("MY_FV", v) for v in versions},
            entity_rows=[_user_id_entity_row()],
        )

        export_root = tmp_path / f"{_DB}.{_SCHEMA}"
        batch = load_specs([f"{export_root}/..."])
        decl_api.resolve_datasource_columns(batch)
        plan = decl_api.generate_plan(
            batch,
            applied_state,
            PlanOptions(full_directory_mode=True),
            database=_DB,
            schema=_SCHEMA,
        )

        drop_fv = [op for op in plan.ops if op.kind.value == "DROP_FV"]
        assert drop_fv == [], (
            "full-directory plan must emit ZERO DROP_FV after an unmodified "
            "two-version export; a dropped version means the exporter lost a "
            f"YAML to a stem collision. got: {[(o.name, o.payload) for o in drop_fv]!r}"
        )


# ---------------------------------------------------------------------------
# FeatureGroup — two versions must not collide
# ---------------------------------------------------------------------------


class TestVersionedFeatureGroupStemCollision:
    def test_two_fg_versions_write_distinct_files(self, tmp_path: Path) -> None:
        rows = [_fg_row("MY_FG", "V1"), _fg_row("MY_FG", "V2")]
        result = export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database=_DB,
            schema=_SCHEMA,
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )

        fg_dir = Path(result["directory"]) / "feature_groups"
        assert (fg_dir / "MY_FG_V1.yaml").exists(), (
            "V1 FG YAML missing — the name-only stem + seen_fg_files skip "
            f"dropped a version. present: {sorted(p.name for p in fg_dir.glob('*.yaml'))}"
        )
        assert (fg_dir / "MY_FG_V2.yaml").exists(), (
            "V2 FG YAML missing. " f"present: {sorted(p.name for p in fg_dir.glob('*.yaml'))}"
        )

    def test_two_fg_versions_round_trip_no_drop(self, tmp_path: Path) -> None:
        rows = [_fg_row("MY_FG", "V1"), _fg_row("MY_FG", "V2")]
        export_specs(
            show_rows=[],
            describe_rows_by_oft={},
            output_dir=str(tmp_path),
            database=_DB,
            schema=_SCHEMA,
            specification_map={},
            entity_rows=[],
            feature_group_rows=rows,
            layout="sources",
        )

        applied_state = decl_api.fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database=_DB,
            default_schema=_SCHEMA,
        )
        batch = load_from_project(tmp_path, database=_DB, schema=_SCHEMA)
        plan = decl_api.generate_plan(
            batch,
            applied_state,
            PlanOptions(full_directory_mode=True),
            database=_DB,
            schema=_SCHEMA,
        )

        drop_fg = [op for op in plan.ops if op.kind.value == "DROP_FG"]
        assert drop_fg == [], (
            "full-directory plan must emit ZERO DROP_FG after an unmodified "
            "two-version FG export; a dropped version means the FG loop skipped "
            f"a duplicate stem. got: {[(o.name, o.payload) for o in drop_fg]!r}"
        )


# ---------------------------------------------------------------------------
# Residual stem collision must raise, not silently overwrite / skip
# ---------------------------------------------------------------------------


class TestStemCollisionRaises:
    def test_fv_stem_collision_raises(self, tmp_path: Path) -> None:
        # Two distinct OFT rows whose SPECIFICATION metadata resolves to the
        # same (name, version) -> same stem.  An error is recoverable; a
        # silently-overwritten version is not.
        show_rows = [_show_row("MY_FV", "V1"), _show_row("MY_FV_DUP", "V1")]
        spec_map = {
            _oft_name("MY_FV", "V1"): _batch_spec("MY_FV", "V1"),
            _oft_name("MY_FV_DUP", "V1"): _batch_spec("MY_FV", "V1"),
        }
        with pytest.raises(ValueError, match="MY_FV"):
            export_specs(
                show_rows=show_rows,
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database=_DB,
                schema=_SCHEMA,
                specification_map=spec_map,
                entity_rows=[_user_id_entity_row()],
            )

    def test_fg_stem_collision_raises(self, tmp_path: Path) -> None:
        rows = [_fg_row("MY_FG", "V1"), _fg_row("MY_FG", "V1")]
        with pytest.raises(ValueError, match="MY_FG"):
            export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database=_DB,
                schema=_SCHEMA,
                specification_map={},
                entity_rows=[],
                feature_group_rows=rows,
                layout="sources",
            )


if __name__ == "__main__":
    pytest_driver.main()
