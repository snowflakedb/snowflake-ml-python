"""Locking tests for the "no regex / no SQL-text parsing in decl/state.py" invariant.

Phase B1 of the metadata-roundtrip plan: pin the post-refactor
architecture so future "quick fix" PRs cannot reintroduce raw DT-text
parsing into the declarative state-recovery layer.

After Phase A landed authoritative ``FV_SOURCE_REFS`` metadata, every
regex helper in :mod:`snowflake.ml.feature_store.decl.state` became
obsolete (the operator-authored source bindings, cluster_by, refresh
mode, and initialize values now ride in
``list_feature_views()`` rows + ``FvSourceRefsMetadata`` instead of
being recovered by string-matching the ``CREATE DYNAMIC TABLE`` body).

These tests lock that invariant by AST scan:

- **NR1 — No ``import re`` in state.py:** Walks the AST of
  ``decl/state.py`` and asserts no top-level ``import re`` /
  ``from re import …`` statement is present.
- **NR2 — No regex literal call sites:** Asserts no ``re.compile``,
  ``re.match``, ``re.search``, or ``re.split`` call appears anywhere
  in the module body.

Sibling to ``test_no_entity_tag_sql_in_decl.py``; the pattern is the
same (AST scan over a single decl module).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _state_module_path() -> Path:
    """Return the resolved path to ``decl/state.py``.

    Returns:
        Filesystem ``Path`` to the production state module.  Resolved
        from this test file's location so the test is robust to
        ``cwd`` changes.
    """
    decl_dir = Path(__file__).resolve().parent.parent
    assert decl_dir.name == "decl", f"Expected this file to live in decl/tests/; computed parent={decl_dir!r}"
    return decl_dir / "state.py"


class TestNoReModuleInState:
    """The ``re`` module must not be imported by ``decl/state.py``.

    Post-Phase-B every recovery path consumes metadata rather than
    string-parsing DT text, so the ``import re`` line is dead code
    and must not be re-added by future PRs without rejustifying it.
    """

    def test_no_import_re_in_state(self) -> None:
        path = _state_module_path()
        tree = ast.parse(path.read_text(), filename=str(path))

        offenders: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "re" or alias.name.startswith("re."):
                        offenders.append((node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                if node.module == "re":
                    names = ", ".join(a.name for a in node.names)
                    offenders.append((node.lineno, f"from re import {names}"))

        assert not offenders, (
            f"{path} must not import the ``re`` module — every SQL/DT-text "
            "regex parser was removed in Phase B1 because authoritative "
            "metadata (FV_SOURCE_REFS, list_feature_views() columns, "
            "FvSourceRefsMetadata) replaces every recovery path that "
            f"previously needed string matching.  Offenders: {offenders!r}"
        )


class TestNoRegexCallSitesInState:
    """No ``re.compile`` / ``re.match`` / ``re.search`` / ``re.split``
    call may appear anywhere in ``decl/state.py``.

    Covers the case where a future refactor inlines a regex call
    without an explicit ``import re`` (e.g. via attribute access on
    an already-imported namespace), which the import-only check
    above would miss.
    """

    _FORBIDDEN_ATTRS = ("compile", "match", "search", "split", "sub", "findall", "fullmatch", "finditer")

    def test_no_re_calls_in_state(self) -> None:
        path = _state_module_path()
        tree = ast.parse(path.read_text(), filename=str(path))

        offenders: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            value = func.value
            if isinstance(value, ast.Name) and value.id == "re" and func.attr in self._FORBIDDEN_ATTRS:
                offenders.append((node.lineno, f"re.{func.attr}(...)"))

        assert not offenders, (
            f"{path} must not call any ``re.*`` helper — Phase B1 "
            "removed every regex-driven recovery path.  "
            f"Offenders: {offenders!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
