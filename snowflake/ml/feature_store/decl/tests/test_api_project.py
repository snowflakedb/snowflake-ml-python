"""Tests for the manifest.yml-based project facades on ``decl.api`` (Phase 2).

Covers the public surface added in Phase 2 — ``load_manifest``,
``discover_project``, ``resolve_target``, ``load_project`` — and the
private ``_merge_template_vars`` helper that powers the templating
precedence rules (D5).

Locked decisions enforced here:

* **D2 (with-role).** ``resolve_target`` returns a target whose
  ``database`` / ``schema`` are non-empty; ``warehouse`` is never part
  of the resolved target.
* **D5 (drop-config).** Template variable precedence is
  ``manifest.templating.defaults <
  manifest.templating.configurations[target.templating_config] <
  runtime_vars``. Dictionary-typed defaults cannot be overridden at
  runtime — a ``ValueError`` naming the key is raised.

The integration test writes a minimal project on disk (manifest.yml +
``sources/entities`` + ``sources/feature_views``) and round-trips it
through ``load_project`` for both DEV and PROD targets, asserting that
the entity's Jinja-templated description reflects the right config or
runtime override.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from snowflake.ml.feature_store.decl import (
    api as decl_api,
    manifest as manifest_mod,
    project_paths as project_paths_mod,
)
from snowflake.ml.feature_store.decl.manifest import (
    FSManifest,
    FSTarget,
    InvalidManifestError,
    ManifestConfigurationError,
    ManifestNotFoundError,
)
from snowflake.ml.feature_store.decl.project_paths import FSProjectPaths
from snowflake.ml.feature_store.decl.types import SpecBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ENTITY_YAML = "kind: Entity\nname: customer\njoin_keys:\n  - name: customer_id\n    type: str\n"

_FV_YAML = (
    "kind: StreamingFeatureView\n"
    "name: user_click_stats\n"
    "online: true\n"
    "entities:\n  - customer_id\n"
    "sources:\n  - name: clickstream\n    source_type: Stream\n"
)


def _write_manifest(project_root: pathlib.Path, body: str) -> pathlib.Path:
    project_root.mkdir(parents=True, exist_ok=True)
    manifest_path = project_root / "manifest.yml"
    manifest_path.write_text(textwrap.dedent(body))
    return manifest_path


def _make_sources_tree(
    project_root: pathlib.Path,
    *,
    entities: dict[str, str] | None = None,
    feature_views: dict[str, str] | None = None,
) -> None:
    sources = project_root / "sources"
    for sub in ("entities", "datasources", "feature_views"):
        (sources / sub).mkdir(parents=True, exist_ok=True)
    for fname, body in (entities or {}).items():
        (sources / "entities" / fname).write_text(body)
    for fname, body in (feature_views or {}).items():
        (sources / "feature_views" / fname).write_text(body)


_DEFAULT_MANIFEST_BODY = """\
manifest_version: 1
type: feature_store
default_target: DEV
targets:
  DEV:
    account_identifier: ORG-ACCT
    database: MY_DB
    schema: MY_SCHEMA
    role: ENG_ROLE
"""


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_happy_path_returns_fsmanifest(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(tmp_path, _DEFAULT_MANIFEST_BODY)

        manifest = decl_api.load_manifest(tmp_path)

        assert isinstance(manifest, FSManifest)
        assert manifest.manifest_version == 1
        assert manifest.project_type == "feature_store"
        assert "DEV" in manifest.targets
        assert manifest.targets["DEV"].database == "MY_DB"
        assert manifest.targets["DEV"].schema == "MY_SCHEMA"

    def test_missing_manifest_raises_manifest_not_found(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ManifestNotFoundError):
            decl_api.load_manifest(tmp_path)

    def test_invalid_manifest_raises_invalid_manifest(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "manifest.yml").write_text("")
        with pytest.raises(InvalidManifestError):
            decl_api.load_manifest(tmp_path)

    def test_unsupported_version_raises_invalid_manifest(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(
            tmp_path,
            """\
            manifest_version: 99
            type: feature_store
            targets:
              DEV:
                account_identifier: ORG-ACCT
                database: MY_DB
                schema: MY_SCHEMA
            """,
        )
        with pytest.raises(InvalidManifestError):
            decl_api.load_manifest(tmp_path)

    def test_target_with_warehouse_raises_configuration_error(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(
            tmp_path,
            """\
            manifest_version: 1
            type: feature_store
            targets:
              DEV:
                account_identifier: ORG-ACCT
                database: MY_DB
                schema: MY_SCHEMA
                warehouse: COMPUTE_WH
            """,
        )
        with pytest.raises(ManifestConfigurationError):
            decl_api.load_manifest(tmp_path)

    def test_reexported_fsmanifest_is_same_class(self) -> None:
        # The class re-exported through decl.api must be the same object as
        # the canonical class in decl.manifest — re-exports should not wrap
        # the type behind a façade subclass.
        assert decl_api.FSManifest is manifest_mod.FSManifest


# ---------------------------------------------------------------------------
# discover_project
# ---------------------------------------------------------------------------


class TestDiscoverProject:
    def test_manifest_in_start_directory(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(tmp_path, _DEFAULT_MANIFEST_BODY)
        paths = decl_api.discover_project(tmp_path)
        assert isinstance(paths, FSProjectPaths)
        assert paths.project_root == tmp_path.resolve()

    def test_manifest_one_level_up(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(tmp_path, _DEFAULT_MANIFEST_BODY)
        sub = tmp_path / "sources"
        sub.mkdir()
        paths = decl_api.discover_project(sub)
        assert paths.project_root == tmp_path.resolve()

    def test_missing_manifest_raises_manifest_not_found(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ManifestNotFoundError):
            decl_api.discover_project(tmp_path)

    def test_default_start_is_cwd(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_manifest(tmp_path, _DEFAULT_MANIFEST_BODY)
        monkeypatch.chdir(tmp_path)
        paths = decl_api.discover_project()
        assert paths.project_root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def _build_manifest_two_targets() -> FSManifest:
    return FSManifest.from_dict(
        {
            "manifest_version": 1,
            "type": "feature_store",
            "default_target": "DEV",
            "targets": {
                "DEV": {
                    "account_identifier": "ORG-ACCT",
                    "database": "DEV_DB",
                    "schema": "DEV_SCHEMA",
                    "role": "DEV_ROLE",
                    "templating_config": "DEV",
                },
                "PROD": {
                    "account_identifier": "ORG-ACCT",
                    "database": "PROD_DB",
                    "schema": "PROD_SCHEMA",
                    "role": "PROD_ROLE",
                    "templating_config": "PROD",
                },
            },
            "templating": {
                "defaults": {"env_suffix": "_BASE"},
                "configurations": {
                    "DEV": {"env_suffix": "_DEV"},
                    "PROD": {"env_suffix": "_PROD"},
                },
            },
        }
    )


def _build_manifest_single_target() -> FSManifest:
    return FSManifest.from_dict(
        {
            "manifest_version": 1,
            "type": "feature_store",
            "targets": {
                "ONLY": {
                    "account_identifier": "ORG-ACCT",
                    "database": "ONLY_DB",
                    "schema": "ONLY_SCHEMA",
                },
            },
        }
    )


class TestResolveTarget:
    def test_by_explicit_name(self) -> None:
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest, "PROD")
        assert isinstance(target, FSTarget)
        assert target.name == "PROD"
        assert target.database == "PROD_DB"
        assert target.schema == "PROD_SCHEMA"

    def test_by_name_is_case_insensitive(self) -> None:
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest, "prod")
        assert target.name == "PROD"

    def test_by_default(self) -> None:
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest)
        assert target.name == "DEV"

    def test_single_target_auto_default(self) -> None:
        manifest = _build_manifest_single_target()
        target = decl_api.resolve_target(manifest)
        assert target.name == "ONLY"

    def test_missing_target_raises_configuration_error(self) -> None:
        manifest = _build_manifest_two_targets()
        with pytest.raises(ManifestConfigurationError):
            decl_api.resolve_target(manifest, "STAGE")

    def test_resolved_target_never_carries_warehouse(self) -> None:
        # D2 invariant: warehouse is not a field on FSTarget at all.
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest, "DEV")
        assert not hasattr(target, "warehouse")


# ---------------------------------------------------------------------------
# _merge_template_vars (private helper)
# ---------------------------------------------------------------------------


class TestMergeTemplateVars:
    def test_defaults_only(self) -> None:
        manifest = FSManifest.from_dict(
            {
                "manifest_version": 1,
                "type": "feature_store",
                "targets": {
                    "ONLY": {
                        "account_identifier": "A",
                        "database": "D",
                        "schema": "S",
                    },
                },
                "templating": {"defaults": {"env_suffix": "_BASE"}},
            }
        )
        target = decl_api.resolve_target(manifest)
        merged = decl_api._merge_template_vars(manifest, target, None)
        assert merged == {"env_suffix": "_BASE"}

    def test_configuration_overrides_default(self) -> None:
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest, "DEV")
        merged = decl_api._merge_template_vars(manifest, target, None)
        assert merged["env_suffix"] == "_DEV"

    def test_runtime_overrides_configuration(self) -> None:
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest, "PROD")
        merged = decl_api._merge_template_vars(
            manifest,
            target,
            {"env_suffix": "_PRODX"},
        )
        assert merged["env_suffix"] == "_PRODX"

    def test_runtime_vars_cannot_override_dict_default(self) -> None:
        manifest = FSManifest.from_dict(
            {
                "manifest_version": 1,
                "type": "feature_store",
                "targets": {
                    "ONLY": {
                        "account_identifier": "A",
                        "database": "D",
                        "schema": "S",
                    },
                },
                "templating": {"defaults": {"tags": {"team": "ml"}}},
            }
        )
        target = decl_api.resolve_target(manifest)
        with pytest.raises(ValueError) as exc_info:
            decl_api._merge_template_vars(manifest, target, {"tags": "overwrite"})
        assert "tags" in str(exc_info.value)

    def test_no_runtime_vars_is_safe(self) -> None:
        manifest = _build_manifest_two_targets()
        target = decl_api.resolve_target(manifest, "DEV")
        merged = decl_api._merge_template_vars(manifest, target, {})
        assert merged["env_suffix"] == "_DEV"


# ---------------------------------------------------------------------------
# load_project
# ---------------------------------------------------------------------------


class TestLoadProject:
    def test_empty_sources_returns_empty_batch(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(tmp_path, _DEFAULT_MANIFEST_BODY)
        _make_sources_tree(tmp_path)  # all three subdirs but no files

        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest)
        batch = decl_api.load_project(tmp_path, target=target)
        assert isinstance(batch, SpecBatch)
        assert batch.specs == []

    def test_entity_and_fv_carry_target_db_and_schema(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(tmp_path, _DEFAULT_MANIFEST_BODY)
        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest)
        batch = decl_api.load_project(tmp_path, target=target)

        assert len(batch.specs) == 2
        for spec in batch.specs:
            assert spec.database == target.database
            assert spec.schema_ == target.schema

    def test_runtime_vars_override_scalar(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(
            tmp_path,
            """\
            manifest_version: 1
            type: feature_store
            default_target: DEV
            targets:
              DEV:
                account_identifier: ORG-ACCT
                database: DEV_DB
                schema: DEV_SCHEMA
                templating_config: DEV
            templating:
              defaults:
                env_suffix: _BASE
              configurations:
                DEV:
                  env_suffix: _DEV
            """,
        )
        entity_yaml_template = (
            "kind: Entity\n"
            "name: customer\n"
            "description: 'env={{ env_suffix }}'\n"
            "join_keys:\n  - name: customer_id\n    type: str\n"
        )
        _make_sources_tree(tmp_path, entities={"customer.yaml": entity_yaml_template})

        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest)

        batch_default = decl_api.load_project(tmp_path, target=target)
        assert len(batch_default.specs) == 1
        assert batch_default.specs[0].description == "env=_DEV"

        batch_override = decl_api.load_project(
            tmp_path,
            target=target,
            runtime_vars={"env_suffix": "_OVERRIDE"},
        )
        assert batch_override.specs[0].description == "env=_OVERRIDE"

    def test_runtime_vars_cannot_override_dict_default(self, tmp_path: pathlib.Path) -> None:
        _write_manifest(
            tmp_path,
            """\
            manifest_version: 1
            type: feature_store
            default_target: DEV
            targets:
              DEV:
                account_identifier: ORG-ACCT
                database: DEV_DB
                schema: DEV_SCHEMA
            templating:
              defaults:
                tags:
                  team: ml
            """,
        )
        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})

        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest)

        with pytest.raises(ValueError) as exc_info:
            decl_api.load_project(
                tmp_path,
                target=target,
                runtime_vars={"tags": "nope"},
            )
        assert "tags" in str(exc_info.value)

    def test_target_db_and_schema_are_injected_not_connection(self, tmp_path: pathlib.Path) -> None:
        # Two targets in the same project; load_project must thread the
        # target's database/schema through to every spec — not anything
        # else, and certainly not a warehouse field.
        _write_manifest(
            tmp_path,
            """\
            manifest_version: 1
            type: feature_store
            default_target: PROD
            targets:
              DEV:
                account_identifier: ORG-ACCT
                database: DEV_DB
                schema: DEV_SCHEMA
              PROD:
                account_identifier: ORG-ACCT
                database: PROD_DB
                schema: PROD_SCHEMA
            """,
        )
        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})

        manifest = decl_api.load_manifest(tmp_path)

        dev_target = decl_api.resolve_target(manifest, "DEV")
        dev_batch = decl_api.load_project(tmp_path, target=dev_target)
        assert dev_batch.specs[0].database == "DEV_DB"
        assert dev_batch.specs[0].schema_ == "DEV_SCHEMA"

        prod_target = decl_api.resolve_target(manifest, "PROD")
        prod_batch = decl_api.load_project(tmp_path, target=prod_target)
        assert prod_batch.specs[0].database == "PROD_DB"
        assert prod_batch.specs[0].schema_ == "PROD_SCHEMA"

    def test_integration_two_configs_and_runtime_override(self, tmp_path: pathlib.Path) -> None:
        # End-to-end project on disk:
        #   manifest.yml with two configs (DEV / PROD)
        #   one templated entity yaml
        #   one feature view yaml
        # Verify that the entity description follows the templating
        # precedence rules for both targets and both override modes.
        _write_manifest(
            tmp_path,
            """\
            manifest_version: 1
            type: feature_store
            targets:
              DEV:
                account_identifier: ORG-ACCT
                database: DEV_DB
                schema: DEV_SCHEMA
                templating_config: DEV
              PROD:
                account_identifier: ORG-ACCT
                database: PROD_DB
                schema: PROD_SCHEMA
                templating_config: PROD
            templating:
              defaults:
                env_suffix: _BASE
              configurations:
                DEV:
                  env_suffix: _DEV
                PROD:
                  env_suffix: _PROD
            """,
        )
        entity_yaml_template = (
            "kind: Entity\n"
            "name: customer\n"
            "description: 'env={{ env_suffix }}'\n"
            "join_keys:\n  - name: customer_id\n    type: str\n"
        )
        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": entity_yaml_template},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        manifest = decl_api.load_manifest(tmp_path)

        dev_target = decl_api.resolve_target(manifest, "DEV")
        dev_batch = decl_api.load_project(tmp_path, target=dev_target)
        dev_entity = next(spec for spec in dev_batch.specs if spec.kind == "Entity")
        assert "_DEV" in (dev_entity.description or "")

        prod_target = decl_api.resolve_target(manifest, "PROD")
        prod_batch = decl_api.load_project(
            tmp_path,
            target=prod_target,
            runtime_vars={"env_suffix": "_PRODX"},
        )
        prod_entity = next(spec for spec in prod_batch.specs if spec.kind == "Entity")
        assert "_PRODX" in (prod_entity.description or "")


# ---------------------------------------------------------------------------
# Public surface (re-exports)
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_decl_api_exposes_project_facades(self) -> None:
        for name in (
            "load_manifest",
            "discover_project",
            "resolve_target",
            "load_project",
        ):
            assert hasattr(decl_api, name), f"decl.api must expose {name}"
            assert callable(getattr(decl_api, name))

    def test_decl_api_reexports_manifest_types(self) -> None:
        assert decl_api.FSManifest is manifest_mod.FSManifest
        assert decl_api.FSTarget is manifest_mod.FSTarget
        assert decl_api.FSTemplating is manifest_mod.FSTemplating
        assert decl_api.TargetContext is manifest_mod.TargetContext

    def test_decl_api_reexports_project_paths(self) -> None:
        assert decl_api.FSProjectPaths is project_paths_mod.FSProjectPaths

    def test_decl_api_reexports_manifest_errors(self) -> None:
        assert decl_api.ManifestNotFoundError is manifest_mod.ManifestNotFoundError
        assert decl_api.InvalidManifestError is manifest_mod.InvalidManifestError
        assert decl_api.ManifestConfigurationError is manifest_mod.ManifestConfigurationError

    def test_decl_package_reexports_project_facades(self) -> None:
        from snowflake.ml.feature_store import decl as decl_pkg

        for name in (
            "load_manifest",
            "discover_project",
            "resolve_target",
            "load_project",
            "FSManifest",
            "FSTarget",
            "FSTemplating",
            "TargetContext",
            "FSProjectPaths",
            "ManifestNotFoundError",
            "InvalidManifestError",
            "ManifestConfigurationError",
        ):
            assert hasattr(decl_pkg, name), f"decl package must expose {name}"
            assert name in decl_pkg.__all__, f"decl.__all__ must list {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
