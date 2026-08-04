"""Feature-store project paths resolver (Phase 1B).

Resolves the on-disk layout for a feature-store project: the project
root (the directory containing ``manifest.yml``), the manifest file
itself, the ``sources/`` tree, and the ``out/plan/`` directory used by
the plan-file lifecycle. The resolver is the feature-store analog of
DCM's ``ProjectPaths`` helper but lives inside ``decl/`` so it remains
free of CLI / connector / Snowpark dependencies (see
``docs/ARCHITECTURE.md`` boundary rules and
``plans/MANIFEST_YML_LAYOUT_DECISIONS.md`` decisions D1 / D8).

Locked invariants enforced here:

* **D1 (hard-break).** The plan directory is unconditionally
  ``<project_root>/out/plan``. There is no legacy fallback path
  anywhere in the resolved layout.
* **D8 (preserve).** Plan filenames keep the
  ``feature_plan_<UTC ts>.json{,.applied,.discarded}`` shape — only
  the parent directory moves. Lifecycle code is owned by Phase 4; this
  module only exposes the directory.

The resolver intentionally walks all the way to the filesystem root
when discovering the project — it does not stop at the git root or the
user's home directory. This mirrors how DCM-style tools locate their
project files and avoids surprising "we stopped one level too early"
behaviour for nested checkouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from snowflake.ml.feature_store.decl.manifest import (
    MANIFEST_FILE_NAME,
    PLANS_SUBPATH,
    SOURCES_FOLDER,
    ManifestNotFoundError,
)


@dataclass(frozen=True)
class FSProjectPaths:
    """Resolved on-disk paths for a feature-store project.

    Attributes:
        project_root: Absolute, resolved path to the directory that
            contains ``manifest.yml``. All other paths are derived from
            this root.
        manifest_path: ``project_root / "manifest.yml"``. The file may
            or may not exist; ``FSProjectPaths`` itself does not assert
            existence — callers typically follow up with
            :meth:`FSManifest.load`.
        sources_dir: ``project_root / "sources"`` — the root of the
            authoring tree (``entities/``, ``datasources/``,
            ``feature_views/``).
        plans_dir: ``project_root / "out" / "plan"`` (per D1). The
            plan-file lifecycle (L1–L7) stores
            ``feature_plan_<UTC ts>.json{,.applied,.discarded}`` here.
    """

    project_root: Path
    manifest_path: Path
    sources_dir: Path
    plans_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> FSProjectPaths:
        """Build paths from an explicit project-root directory.

        Does NOT verify that ``manifest.yml`` exists; that is the
        caller's job (typically via :meth:`FSManifest.load`). All
        sub-paths are computed deterministically and the project root
        is resolved (symlinks followed) so downstream comparisons are
        canonical.

        Args:
            project_root: Directory to treat as the project root. May
                be relative or contain symlinks; it is resolved to an
                absolute path on construction.

        Returns:
            FSProjectPaths with all four fields populated.
        """
        resolved = Path(project_root).resolve()
        return cls(
            project_root=resolved,
            manifest_path=resolved / MANIFEST_FILE_NAME,
            sources_dir=resolved / SOURCES_FOLDER,
            plans_dir=resolved / Path(*PLANS_SUBPATH),
        )

    @classmethod
    def discover(cls, start: Optional[Path] = None) -> FSProjectPaths:
        """Walk up from ``start`` until a directory containing
        ``manifest.yml`` is found.

        Args:
            start: Directory (or file) to start the walk from. Defaults
                to :func:`pathlib.Path.cwd`. If ``start`` is a file,
                discovery begins at its parent directory.

        Returns:
            FSProjectPaths rooted at the first ancestor containing
            ``manifest.yml``.

        Raises:
            ManifestNotFoundError: If no ``manifest.yml`` is found in
                ``start`` or any ancestor up to the filesystem root.
        """
        origin = Path(start) if start is not None else Path.cwd()
        try:
            project_root = find_project_root(origin)
        except ManifestNotFoundError:
            raise
        return cls.from_project_root(project_root)


def find_project_root(start: Path) -> Path:
    """Walk up from ``start`` until ``manifest.yml`` is found.

    The walk terminates only when the filesystem root is reached — it
    does not stop at the git root or the user's home directory. This
    is deliberate: feature-store projects are often nested inside a
    larger checkout and the discovery contract is "find the nearest
    enclosing manifest.yml".

    Args:
        start: Directory or file to start the walk from. Files are
            resolved to their parent directory before the walk begins.

    Returns:
        The directory path containing ``manifest.yml``.

    Raises:
        ManifestNotFoundError: If no ``manifest.yml`` is found at
            ``start`` or in any ancestor up to the filesystem root. The
            error message names the starting path so operators can see
            where the walk began.
    """
    origin = Path(start)
    current = origin if origin.is_dir() else origin.parent
    current = current.resolve()

    while True:
        candidate = current / MANIFEST_FILE_NAME
        if candidate.is_file():
            return current
        parent = current.parent
        if parent == current:
            raise ManifestNotFoundError(f"No '{MANIFEST_FILE_NAME}' found in '{origin}' or any ancestor.")
        current = parent
