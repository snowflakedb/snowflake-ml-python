"""Import isolation tests.

Verifies that no source file in ``decl/`` (excluding ``tests/``) imports from
forbidden packages (``snowflake.ml.feature_store.spec.models`` /
``snowflake.ml.feature_store.spec.builder``, ``snowflake.connector``).
``snowflake.snowpark`` is permitted (see DEVELOPMENT_STANDARDS.md).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DECL_ROOT = pathlib.Path(__file__).parent.parent  # …/decl/

# NOTE: ``snowflake.ml.feature_store.spec.enums`` and
# ``snowflake.ml.feature_store.interval_utils`` are intentionally NOT
# forbidden — both are stdlib-only (no Snowpark / SnowML-core deps) and
# are the canonical homes for vocabulary shared between the imperative
# and declarative paths.  ``spec.enums`` owns the FS enum vocabulary
# (``FSBaseType``, ``FeatureViewKind``, ``SourceType``, etc.) and
# ``interval_utils`` owns duration parsing
# (``parse_interval`` / ``interval_to_seconds`` / the ``"lifetime"``
# sentinel).  ``spec.models`` and ``spec.builder`` remain forbidden
# because they pull in heavy SnowML-core spec/builder machinery that must
# stay out of the lightweight planning path.  ``snowflake.snowpark`` is
# permitted per DEVELOPMENT_STANDARDS.md (Snowpark is an accepted
# dependency of the ``decl/`` package).
_FORBIDDEN_PREFIXES = (
    "snowflake.ml.feature_store.spec.builder",
    "snowflake.ml.feature_store.spec.models",
    "snowflake.connector",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_source_files() -> list[pathlib.Path]:
    """Return all .py files under decl/ that are NOT inside tests/."""
    return [
        p
        for p in _DECL_ROOT.rglob("*.py")
        if "tests" not in p.relative_to(_DECL_ROOT).parts and "__pycache__" not in p.parts
    ]


def _collect_imports(source: str) -> list[str]:
    """Parse *source* and return all top-level import module names.

    Skips imports inside try/except blocks (optional/guarded imports).

    Args:
        source: Python source code to parse.

    Returns:
        List of dotted module names found in import statements.
    """
    tree = ast.parse(source)
    names: list[str] = []
    # Collect nodes that are inside Try blocks
    guarded_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                guarded_nodes.add(id(child))

    for node in ast.walk(tree):
        if id(node) in guarded_nodes:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


# ---------------------------------------------------------------------------
# Test 1 — no forbidden imports in production source files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_file", _collect_source_files(), ids=lambda p: p.name)
def test_no_forbidden_imports(source_file: pathlib.Path) -> None:
    """Each production module in decl/ must not import forbidden packages."""
    source = source_file.read_text(encoding="utf-8")
    imports = _collect_imports(source)
    for imp in imports:
        for prefix in _FORBIDDEN_PREFIXES:
            assert not imp.startswith(prefix), (
                f"{source_file.name} imports '{imp}' which starts with " f"forbidden prefix '{prefix}'"
            )


# ---------------------------------------------------------------------------
# Test 1b — narrowed isolation contract: spec.enums allowed, models/builder forbidden
# ---------------------------------------------------------------------------


class TestNarrowedSpecIsolation:
    """The wheel-isolation rule permits ``spec.enums`` (stdlib-only) and
    ``snowflake.snowpark`` while continuing to block ``spec.models`` and
    ``spec.builder`` (which pull in heavy SnowML-core spec/builder
    machinery) and ``snowflake.connector``.
    """

    def test_spec_enums_is_permitted(self) -> None:
        # ``spec.enums`` is stdlib-only and is the canonical home for the
        # FS enums shared between the imperative and declarative paths.
        imp = "snowflake.ml.feature_store.spec.enums"
        for prefix in _FORBIDDEN_PREFIXES:
            assert not imp.startswith(prefix), f"'{imp}' should be permitted but matched forbidden prefix '{prefix}'"

    def test_interval_utils_is_permitted(self) -> None:
        # ``interval_utils`` is stdlib-only (``import re`` only) and is the
        # canonical home for duration parsing
        # (``parse_interval`` / ``interval_to_seconds`` / ``"lifetime"``
        # sentinel) shared between the imperative aggregation layer and
        # ``decl/compiler.py:parse_duration_to_seconds``.
        imp = "snowflake.ml.feature_store.interval_utils"
        for prefix in _FORBIDDEN_PREFIXES:
            assert not imp.startswith(prefix), f"'{imp}' should be permitted but matched forbidden prefix '{prefix}'"

    def test_spec_models_remains_forbidden(self) -> None:
        imp = "snowflake.ml.feature_store.spec.models"
        assert any(
            imp.startswith(p) for p in _FORBIDDEN_PREFIXES
        ), f"'{imp}' must remain forbidden — it pulls heavy SnowML-core spec machinery"

    def test_spec_builder_remains_forbidden(self) -> None:
        imp = "snowflake.ml.feature_store.spec.builder"
        assert any(
            imp.startswith(p) for p in _FORBIDDEN_PREFIXES
        ), f"'{imp}' must remain forbidden — it pulls heavy SnowML-core builder machinery"

    def test_snowpark_is_permitted(self) -> None:
        # Snowpark is now an accepted dependency of the decl/ package
        # (DEVELOPMENT_STANDARDS.md); importing it must not be flagged.
        for imp in ("snowflake.snowpark", "snowflake.snowpark.types"):
            for prefix in _FORBIDDEN_PREFIXES:
                assert not imp.startswith(
                    prefix
                ), f"'{imp}' should be permitted but matched forbidden prefix '{prefix}'"

    def test_connector_remains_forbidden(self) -> None:
        for imp in ("snowflake.connector", "snowflake.connector.cursor"):
            assert any(imp.startswith(p) for p in _FORBIDDEN_PREFIXES), f"'{imp}' must remain forbidden"


# ---------------------------------------------------------------------------
# Syntax / compile checks — catch IndentationError, SyntaxError early
# ---------------------------------------------------------------------------


class TestSourceFilesCompile:
    """Every .py file under decl/ must parse and compile without errors."""

    @pytest.mark.parametrize("src", _collect_source_files(), ids=lambda p: p.name)
    def test_source_file_compiles(self, src: pathlib.Path) -> None:
        """py_compile each source file — catches SyntaxError, IndentationError."""
        import py_compile

        py_compile.compile(str(src), doraise=True)

    @pytest.mark.parametrize("src", _collect_source_files(), ids=lambda p: p.name)
    def test_source_file_parses_as_ast(self, src: pathlib.Path) -> None:
        """ast.parse each source file — catches subtle AST-level issues."""
        source = src.read_text()
        ast.parse(source, filename=str(src))


class TestModulesImportable:
    """All public decl modules must be importable without errors."""

    _MODULES = [
        "snowflake.ml.feature_store.decl.api",
        "snowflake.ml.feature_store.decl.compiler",
        "snowflake.ml.feature_store.decl.dependencies",
        "snowflake.ml.feature_store.decl.enums",
        "snowflake.ml.feature_store.decl.errors",
        "snowflake.ml.feature_store.decl.exporter",
        "snowflake.ml.feature_store.decl.invariants",
        "snowflake.ml.feature_store.decl.loader",
        "snowflake.ml.feature_store.decl.planner",
        "snowflake.ml.feature_store.decl.serializer",
        "snowflake.ml.feature_store.decl.service",
        "snowflake.ml.feature_store.decl.spec_models",
        "snowflake.ml.feature_store.decl.state",
        "snowflake.ml.feature_store.decl.templating",
        "snowflake.ml.feature_store.decl.types",
    ]

    @pytest.mark.parametrize("module_name", _MODULES)
    def test_module_imports_cleanly(self, module_name: str) -> None:
        """importlib.import_module should succeed for every public module."""
        import importlib

        importlib.import_module(module_name)


if __name__ == "__main__":
    pytest_driver.main()
