"""Tests for decl/manifest.py — FSManifest model + parser (Phase 1A).

Covers the manifest.yml-based project layout per
`plans/MANIFEST_YML_LAYOUT_DECISIONS.md` decisions D1-D8 and the
worker requirement file `plans/manifest_layout/phase1a_manifest_module.md`.

The parser/model mirrors the DCM reference
(`snowflake-cli/src/snowflake/cli/_plugins/dcm/models.py`) with
feature-store-specific differences:
  - MANIFEST_TYPE = "feature_store"
  - SUPPORTED_MANIFEST_VERSION = 1
  - per-target fields: account_identifier, database, schema (required);
    role, templating_config (optional)
  - `warehouse:` is REJECTED per D2 with ``ManifestConfigurationError``.
"""

from __future__ import annotations

import pathlib
import tempfile
import textwrap
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import manifest as manifest_mod
from snowflake.ml.feature_store.decl.manifest import (
    MANIFEST_FILE_NAME,
    MANIFEST_TYPE,
    PLANS_SUBPATH,
    SOURCES_FOLDER,
    SUPPORTED_MANIFEST_VERSION,
    FSManifest,
    FSTarget,
    FSTemplating,
    InvalidManifestError,
    ManifestConfigurationError,
    ManifestNotFoundError,
    TargetContext,
)
from snowflake.ml.test_utils import pytest_driver


def _valid_target_data(**overrides: Any) -> Any:
    base = {
        "account_identifier": "ORG-ACCT",
        "database": "MY_DB",
        "schema": "MY_SCHEMA",
        "role": "ENG_ROLE",
    }
    base.update(overrides)
    return base


def _valid_manifest_data(**overrides: Any) -> Any:
    base = {
        "manifest_version": 1,
        "type": "feature_store",
        "default_target": "DEV",
        "targets": {
            "DEV": _valid_target_data(),
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_manifest_file_name(self) -> None:
        assert MANIFEST_FILE_NAME == "manifest.yml"

    def test_manifest_type(self) -> None:
        assert MANIFEST_TYPE == "feature_store"

    def test_supported_manifest_version(self) -> None:
        assert SUPPORTED_MANIFEST_VERSION == 1

    def test_sources_folder(self) -> None:
        assert SOURCES_FOLDER == "sources"

    def test_plans_subpath(self) -> None:
        assert PLANS_SUBPATH == ("out", "plan")


# ---------------------------------------------------------------------------
# FSTemplating.from_dict
# ---------------------------------------------------------------------------


class TestFSTemplatingFromDict:
    def test_none_returns_empty_templating(self) -> None:
        t = FSTemplating.from_dict(None)
        assert t.defaults == {}
        assert t.configurations == {}

    def test_empty_dict_returns_empty_templating(self) -> None:
        t = FSTemplating.from_dict({})
        assert t.defaults == {}
        assert t.configurations == {}

    def test_defaults_passed_through(self) -> None:
        t = FSTemplating.from_dict({"defaults": {"region": "us-west-2"}})
        assert t.defaults == {"region": "us-west-2"}
        assert t.configurations == {}

    def test_configurations_are_uppercased_on_load(self) -> None:
        t = FSTemplating.from_dict(
            {
                "configurations": {
                    "dev": {"region": "us-west-2"},
                    "Prod": {"region": "us-east-1"},
                }
            }
        )
        assert "DEV" in t.configurations
        assert "PROD" in t.configurations
        assert t.configurations["DEV"] == {"region": "us-west-2"}
        assert t.configurations["PROD"] == {"region": "us-east-1"}


# ---------------------------------------------------------------------------
# FSTarget.from_dict
# ---------------------------------------------------------------------------


class TestFSTargetFromDict:
    def test_happy_path_all_fields(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
            "role": "eng_role",
            "templating_config": "dev",
        }
        target = FSTarget.from_dict(data)
        assert target.name == "DEV"
        assert target.account_identifier == "ORG-ACCT"
        assert target.database == "MY_DB"
        assert target.schema == "MY_SCHEMA"
        assert target.role == "ENG_ROLE"
        assert target.templating_config == "DEV"

    def test_missing_account_identifier_raises(self) -> None:
        data = {
            "name": "DEV",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
        }
        with pytest.raises(ManifestConfigurationError) as ei:
            FSTarget.from_dict(data)
        assert "account_identifier" in str(ei.value)
        assert "DEV" in str(ei.value)

    def test_missing_database_raises(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "schema": "MY_SCHEMA",
        }
        with pytest.raises(ManifestConfigurationError) as ei:
            FSTarget.from_dict(data)
        assert "database" in str(ei.value)
        assert "DEV" in str(ei.value)

    def test_missing_schema_raises(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
        }
        with pytest.raises(ManifestConfigurationError) as ei:
            FSTarget.from_dict(data)
        assert "schema" in str(ei.value)
        assert "DEV" in str(ei.value)

    def test_warehouse_field_rejected_d2(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
            "warehouse": "COMPUTE_WH",
        }
        with pytest.raises(ManifestConfigurationError) as ei:
            FSTarget.from_dict(data)
        msg = str(ei.value)
        assert (
            msg == "Target 'DEV' has unsupported field 'warehouse'. "
            "Warehouse comes from the active connection (per D2)."
        )

    def test_role_optional_absent(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
        }
        target = FSTarget.from_dict(data)
        assert target.role in ("", None)

    def test_role_identifier_normalized_uppercase(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
            "role": "data_eng",
        }
        target = FSTarget.from_dict(data)
        assert target.role == "DATA_ENG"

    def test_templating_config_optional_absent(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
        }
        target = FSTarget.from_dict(data)
        assert target.templating_config is None

    def test_templating_config_uppercased_on_load(self) -> None:
        data = {
            "name": "DEV",
            "account_identifier": "ORG-ACCT",
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
            "templating_config": "dev",
        }
        target = FSTarget.from_dict(data)
        assert target.templating_config == "DEV"


# ---------------------------------------------------------------------------
# FSManifest.from_dict
# ---------------------------------------------------------------------------


class TestFSManifestFromDict:
    def test_happy_path_three_targets_with_templating(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "default_target": "DEV",
            "targets": {
                "DEV": _valid_target_data(templating_config="dev"),
                "STAGE": _valid_target_data(database="STAGE_DB"),
                "PROD": _valid_target_data(database="PROD_DB"),
            },
            "templating": {
                "defaults": {"region": "us-west-2"},
                "configurations": {
                    "dev": {"region": "us-west-2"},
                    "prod": {"region": "us-east-1"},
                },
            },
        }
        m = FSManifest.from_dict(data)
        assert m.manifest_version == 1
        assert m.project_type == "feature_store"
        assert m.default_target == "DEV"
        assert set(m.targets.keys()) == {"DEV", "STAGE", "PROD"}
        assert m.templating.defaults == {"region": "us-west-2"}
        assert set(m.templating.configurations.keys()) == {"DEV", "PROD"}

    def test_manifest_version_two_raises_invalid(self) -> None:
        data = _valid_manifest_data(manifest_version=2)
        with pytest.raises(InvalidManifestError) as ei:
            FSManifest.from_dict(data)
        msg = str(ei.value)
        assert "2" in msg
        assert str(SUPPORTED_MANIFEST_VERSION) in msg

    def test_manifest_version_missing_raises_invalid(self) -> None:
        data = _valid_manifest_data()
        data.pop("manifest_version")
        with pytest.raises(InvalidManifestError):
            FSManifest.from_dict(data)

    def test_manifest_version_non_integer_raises_invalid(self) -> None:
        data = _valid_manifest_data(manifest_version="abc")
        with pytest.raises(InvalidManifestError) as ei:
            FSManifest.from_dict(data)
        assert "abc" in str(ei.value)

    def test_type_non_feature_store_raises_invalid(self) -> None:
        data = _valid_manifest_data()
        data["type"] = "dcm_project"
        with pytest.raises(InvalidManifestError) as ei:
            FSManifest.from_dict(data)
        msg = str(ei.value)
        assert "feature_store" in msg

    def test_missing_type_raises_invalid(self) -> None:
        data = _valid_manifest_data()
        data.pop("type")
        with pytest.raises(InvalidManifestError) as ei:
            FSManifest.from_dict(data)
        assert "feature_store" in str(ei.value)

    def test_type_case_insensitive_canonicalized_to_lowercase(self) -> None:
        data = _valid_manifest_data()
        data["type"] = "Feature_Store"
        m = FSManifest.from_dict(data)
        assert m.project_type == "feature_store"

    def test_single_target_auto_default(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "targets": {
                "ONLY": _valid_target_data(),
            },
        }
        m = FSManifest.from_dict(data)
        assert m.default_target == "ONLY"

    def test_default_target_references_missing_target_raises(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "default_target": "GHOST",
            "targets": {
                "DEV": _valid_target_data(),
            },
        }
        with pytest.raises(ManifestConfigurationError) as ei:
            FSManifest.from_dict(data)
        msg = str(ei.value)
        assert "GHOST" in msg

    def test_target_names_are_uppercased_on_load(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "default_target": "dev",
            "targets": {
                "dev": _valid_target_data(),
                "Prod": _valid_target_data(database="PROD_DB"),
            },
        }
        m = FSManifest.from_dict(data)
        assert set(m.targets.keys()) == {"DEV", "PROD"}
        assert m.default_target == "DEV"

    def test_warehouse_in_target_rejected_via_from_dict(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "default_target": "DEV",
            "targets": {
                "DEV": _valid_target_data(warehouse="COMPUTE_WH"),
            },
        }
        with pytest.raises(ManifestConfigurationError) as ei:
            FSManifest.from_dict(data)
        assert (
            str(ei.value) == "Target 'DEV' has unsupported field 'warehouse'. "
            "Warehouse comes from the active connection (per D2)."
        )


# ---------------------------------------------------------------------------
# FSManifest.load
# ---------------------------------------------------------------------------


_VALID_YAML = textwrap.dedent(
    """\
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
)


class TestFSManifestLoad:
    def test_load_valid_manifest_returns_fsmanifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / MANIFEST_FILE_NAME).write_text(_VALID_YAML)
            m = FSManifest.load(root)
        assert isinstance(m, FSManifest)
        assert m.manifest_version == 1
        assert m.project_type == "feature_store"
        assert m.default_target == "DEV"
        assert "DEV" in m.targets

    def test_load_missing_file_raises_not_found_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            with pytest.raises(ManifestNotFoundError) as ei:
                FSManifest.load(root)
            msg = str(ei.value)
            assert MANIFEST_FILE_NAME in msg
            assert str(root) in msg

    def test_load_empty_file_raises_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / MANIFEST_FILE_NAME).write_text("")
            with pytest.raises(InvalidManifestError):
                FSManifest.load(root)

    def test_load_yaml_parse_error_wrapped_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / MANIFEST_FILE_NAME).write_text("manifest_version: 1\ntype: feature_store\ntargets: [unclosed")
            with pytest.raises(InvalidManifestError):
                FSManifest.load(root)


# ---------------------------------------------------------------------------
# FSManifest.get_target / get_effective_target
# ---------------------------------------------------------------------------


def _build_manifest_with_targets() -> Any:
    data = {
        "manifest_version": 1,
        "type": "feature_store",
        "default_target": "DEV",
        "targets": {
            "DEV": _valid_target_data(templating_config="dev"),
            "PROD": _valid_target_data(database="PROD_DB"),
        },
        "templating": {
            "configurations": {
                "dev": {"region": "us-west-2"},
            },
        },
    }
    return FSManifest.from_dict(data)


class TestFSManifestGetTarget:
    def test_get_target_is_case_insensitive(self) -> None:
        m = _build_manifest_with_targets()
        target = m.get_target("dev")
        assert target.name == "DEV"

    def test_get_target_missing_raises_configuration_error(self) -> None:
        m = _build_manifest_with_targets()
        with pytest.raises(ManifestConfigurationError) as ei:
            m.get_target("MISSING")
        assert "MISSING" in str(ei.value)

    def test_get_target_with_unknown_templating_config_raises(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "default_target": "DEV",
            "targets": {
                "DEV": _valid_target_data(templating_config="ghost"),
            },
            "templating": {
                "configurations": {
                    "dev": {"region": "us-west-2"},
                },
            },
        }
        m = FSManifest.from_dict(data)
        with pytest.raises(ManifestConfigurationError) as ei:
            m.get_target("DEV")
        msg = str(ei.value)
        assert "DEV" in msg
        assert "GHOST" in msg


class TestFSManifestGetEffectiveTarget:
    def test_get_effective_target_returns_default(self) -> None:
        m = _build_manifest_with_targets()
        target = m.get_effective_target(None)
        assert target.name == "DEV"

    def test_get_effective_target_explicit_name(self) -> None:
        m = _build_manifest_with_targets()
        target = m.get_effective_target("PROD")
        assert target.name == "PROD"

    def test_get_effective_target_no_default_raises(self) -> None:
        data = {
            "manifest_version": 1,
            "type": "feature_store",
            "targets": {
                "DEV": _valid_target_data(),
                "PROD": _valid_target_data(database="PROD_DB"),
            },
        }
        m = FSManifest.from_dict(data)
        assert m.default_target is None
        with pytest.raises(ManifestConfigurationError):
            m.get_effective_target(None)


# ---------------------------------------------------------------------------
# Public surface sanity
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_target_context_dataclass_present(self) -> None:
        ctx = TargetContext(
            target_name="DEV",
            account_identifier="ORG-ACCT",
            database="MY_DB",
            schema="MY_SCHEMA",
            role="ENG_ROLE",
            template_vars={"region": "us-west-2"},
        )
        assert ctx.target_name == "DEV"
        assert ctx.template_vars == {"region": "us-west-2"}

    def test_exception_hierarchy(self) -> None:
        assert issubclass(ManifestNotFoundError, Exception)
        assert issubclass(InvalidManifestError, Exception)
        assert issubclass(ManifestConfigurationError, Exception)

    def test_module_does_not_import_forbidden_snowflake_packages(self) -> None:
        import inspect

        src = inspect.getsource(manifest_mod)
        for forbidden in (
            "snowflake.snowpark",
            "snowflake.connector",
            "snowflake.ml.feature_store.",
            "snowflake.cli.",
        ):
            assert forbidden not in src, f"manifest.py must not reference {forbidden}"


if __name__ == "__main__":
    pytest_driver.main()
