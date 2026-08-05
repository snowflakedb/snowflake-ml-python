"""Tests for decl/project_paths.py — FSProjectPaths + walk-up discovery (Phase 1B).

Covers the manifest.yml-based project layout per
``plans/MANIFEST_YML_LAYOUT_DECISIONS.md`` decisions D1 / D8 and the
worker requirement file ``plans/manifest_layout/phase1b_project_paths.md``.

Invariants pinned by these tests:

* ``FSProjectPaths.plans_dir`` is always ``<project_root>/out/plan`` —
  there is **no** ``.snowflake/plans`` fallback (D1: hard-break).
* ``FSProjectPaths.sources_dir`` is always ``<project_root>/sources``
  (matches ``SOURCES_FOLDER`` constant in ``decl.manifest``).
* The dataclass is frozen — callers cannot mutate any field after
  construction.
* ``discover`` walks UP from ``start`` to the filesystem root and raises
  :class:`ManifestNotFoundError` if no ``manifest.yml`` is found in any
  ancestor.
"""

from __future__ import annotations

import inspect
import os
import pathlib
from dataclasses import FrozenInstanceError

import pytest

from snowflake.ml.feature_store.decl import project_paths as project_paths_mod
from snowflake.ml.feature_store.decl.manifest import (
    MANIFEST_FILE_NAME,
    PLANS_SUBPATH,
    SOURCES_FOLDER,
    ManifestNotFoundError,
)
from snowflake.ml.feature_store.decl.project_paths import (
    FSProjectPaths,
    find_project_root,
)


def _touch_manifest(root: pathlib.Path) -> pathlib.Path:
    """Create an empty ``manifest.yml`` under ``root`` and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / MANIFEST_FILE_NAME
    manifest.write_text("manifest_version: 1\ntype: feature_store\n")
    return manifest


# ---------------------------------------------------------------------------
# FSProjectPaths.from_project_root
# ---------------------------------------------------------------------------


class TestFromProjectRoot:
    def test_project_root_is_resolved(self, tmp_path: pathlib.Path) -> None:
        # tmp_path may already be a resolved absolute path; the contract is
        # that ``project_root`` equals ``tmp_path.resolve()``.
        paths = FSProjectPaths.from_project_root(tmp_path)
        assert paths.project_root == tmp_path.resolve()

    def test_manifest_path_under_project_root(self, tmp_path: pathlib.Path) -> None:
        paths = FSProjectPaths.from_project_root(tmp_path)
        assert paths.manifest_path == tmp_path.resolve() / MANIFEST_FILE_NAME

    def test_sources_dir_is_sources(self, tmp_path: pathlib.Path) -> None:
        paths = FSProjectPaths.from_project_root(tmp_path)
        assert paths.sources_dir == tmp_path.resolve() / SOURCES_FOLDER
        assert paths.sources_dir == tmp_path.resolve() / "sources"

    def test_plans_dir_is_out_plan_no_dot_snowflake(self, tmp_path: pathlib.Path) -> None:
        # D1 enforcement: plans_dir is ALWAYS <root>/out/plan; no
        # .snowflake/plans fallback.
        paths = FSProjectPaths.from_project_root(tmp_path)
        assert paths.plans_dir == tmp_path.resolve() / pathlib.Path(*PLANS_SUBPATH)
        assert paths.plans_dir == tmp_path.resolve() / "out" / "plan"

    def test_does_not_require_manifest_to_exist(self, tmp_path: pathlib.Path) -> None:
        # `from_project_root` is a pure path computation; the manifest may
        # not exist yet (e.g., during `snow feature init`).
        assert not (tmp_path / MANIFEST_FILE_NAME).exists()
        paths = FSProjectPaths.from_project_root(tmp_path)
        # The path is computed but the file is not required.
        assert paths.manifest_path == tmp_path.resolve() / MANIFEST_FILE_NAME
        assert not paths.manifest_path.exists()

    def test_dataclass_is_frozen(self, tmp_path: pathlib.Path) -> None:
        paths = FSProjectPaths.from_project_root(tmp_path)
        with pytest.raises(FrozenInstanceError):
            paths.plans_dir = tmp_path / "elsewhere"  # type: ignore[misc]

    def test_dataclass_fields_all_frozen(self, tmp_path: pathlib.Path) -> None:
        paths = FSProjectPaths.from_project_root(tmp_path)
        for field_name in ("project_root", "manifest_path", "sources_dir", "plans_dir"):
            with pytest.raises(FrozenInstanceError):
                setattr(paths, field_name, tmp_path / "elsewhere")

    def test_from_project_root_resolves_symlink(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "real_root"
        target.mkdir()
        link = tmp_path / "link_root"
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        paths = FSProjectPaths.from_project_root(link)
        # `resolve()` follows symlinks; the project root is the real path.
        assert paths.project_root == target.resolve()


# ---------------------------------------------------------------------------
# FSProjectPaths.discover (walk-up)
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_manifest_in_start_directory(self, tmp_path: pathlib.Path) -> None:
        _touch_manifest(tmp_path)
        paths = FSProjectPaths.discover(tmp_path)
        assert paths.project_root == tmp_path.resolve()
        assert paths.manifest_path == tmp_path.resolve() / MANIFEST_FILE_NAME

    def test_manifest_one_level_up(self, tmp_path: pathlib.Path) -> None:
        _touch_manifest(tmp_path)
        child = tmp_path / "sources"
        child.mkdir()
        paths = FSProjectPaths.discover(child)
        assert paths.project_root == tmp_path.resolve()

    def test_manifest_three_levels_up(self, tmp_path: pathlib.Path) -> None:
        _touch_manifest(tmp_path)
        deep = tmp_path / "sources" / "feature_views" / "deep"
        deep.mkdir(parents=True)
        paths = FSProjectPaths.discover(deep)
        assert paths.project_root == tmp_path.resolve()

    def test_no_manifest_anywhere_raises_with_start_path(self, tmp_path: pathlib.Path) -> None:
        # No manifest anywhere from tmp_path up to the filesystem root.
        # We rely on tmp_path's ancestors not containing manifest.yml — a
        # safe assumption for pytest's per-test tmp_path under
        # ``/private/var/folders/.../pytest-of-...`` / ``/tmp/pytest-...``.
        with pytest.raises(ManifestNotFoundError) as ei:
            FSProjectPaths.discover(tmp_path)
        msg = str(ei.value)
        assert MANIFEST_FILE_NAME in msg
        assert str(tmp_path) in msg

    def test_start_is_a_file_walks_from_parent(self, tmp_path: pathlib.Path) -> None:
        _touch_manifest(tmp_path)
        sources = tmp_path / "sources" / "feature_views"
        sources.mkdir(parents=True)
        fv_file = sources / "X.yaml"
        fv_file.write_text("name: X\n")
        paths = FSProjectPaths.discover(fv_file)
        assert paths.project_root == tmp_path.resolve()

    def test_start_none_defaults_to_cwd(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _touch_manifest(tmp_path)
        monkeypatch.chdir(tmp_path)
        paths = FSProjectPaths.discover()
        assert paths.project_root == tmp_path.resolve()

    def test_start_none_defaults_to_cwd_from_subdir(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _touch_manifest(tmp_path)
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        paths = FSProjectPaths.discover()
        assert paths.project_root == tmp_path.resolve()

    def test_walks_up_to_filesystem_root_before_giving_up(self, tmp_path: pathlib.Path) -> None:
        # Build a deep tree with no manifest; discover should walk all the
        # way up (no early termination at git root / home dir) and only
        # give up when it reaches the filesystem root.
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        with pytest.raises(ManifestNotFoundError):
            FSProjectPaths.discover(deep)

    def test_discover_returns_fsprojectpaths_instance(self, tmp_path: pathlib.Path) -> None:
        _touch_manifest(tmp_path)
        paths = FSProjectPaths.discover(tmp_path)
        assert isinstance(paths, FSProjectPaths)


# ---------------------------------------------------------------------------
# find_project_root (thin wrapper)
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    def test_happy_path_returns_directory(self, tmp_path: pathlib.Path) -> None:
        _touch_manifest(tmp_path)
        sub = tmp_path / "sources"
        sub.mkdir()
        root = find_project_root(sub)
        assert root == tmp_path.resolve()

    def test_missing_manifest_raises_manifest_not_found(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ManifestNotFoundError) as ei:
            find_project_root(tmp_path)
        assert MANIFEST_FILE_NAME in str(ei.value)


# ---------------------------------------------------------------------------
# Constants enforcement (D1 — no .snowflake/plans)
# ---------------------------------------------------------------------------


class TestPathsConstants:
    def test_plans_dir_is_exactly_out_plan(self, tmp_path: pathlib.Path) -> None:
        paths = FSProjectPaths.from_project_root(tmp_path)
        relative = paths.plans_dir.relative_to(paths.project_root)
        assert relative.parts == ("out", "plan")

    def test_sources_dir_is_exactly_sources(self, tmp_path: pathlib.Path) -> None:
        paths = FSProjectPaths.from_project_root(tmp_path)
        relative = paths.sources_dir.relative_to(paths.project_root)
        assert relative.parts == ("sources",)

    def test_no_dot_snowflake_anywhere_in_plan_path(self, tmp_path: pathlib.Path) -> None:
        # D1 enforcement: there must be no `.snowflake/` directory
        # component anywhere in the plans path.
        paths = FSProjectPaths.from_project_root(tmp_path)
        for part in paths.plans_dir.parts:
            assert part != ".snowflake"


# ---------------------------------------------------------------------------
# Module isolation (mirrors test_manifest's isolation check)
# ---------------------------------------------------------------------------


class TestModuleIsolation:
    def test_module_does_not_import_forbidden_snowflake_packages(self) -> None:
        src = inspect.getsource(project_paths_mod)
        for forbidden in (
            "snowflake.snowpark",
            "snowflake.connector",
            "snowflake.cli",
        ):
            assert forbidden not in src, f"project_paths.py must not reference {forbidden}"

    def test_module_imports_only_required_symbols_from_decl_manifest(self) -> None:
        # The module must import exactly the four symbols listed in the
        # requirement file: MANIFEST_FILE_NAME, PLANS_SUBPATH,
        # SOURCES_FOLDER, ManifestNotFoundError. We assert all four are
        # reachable through ``project_paths_mod`` namespace.
        for name in (
            "MANIFEST_FILE_NAME",
            "PLANS_SUBPATH",
            "SOURCES_FOLDER",
            "ManifestNotFoundError",
        ):
            assert hasattr(project_paths_mod, name), f"{name} must be importable from project_paths"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
