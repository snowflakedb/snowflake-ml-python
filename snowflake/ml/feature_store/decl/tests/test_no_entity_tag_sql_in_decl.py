"""Locking tests for the "no raw entity-tag SQL in decl/" invariant.

Phase 7 of the
``remove_duplicate_entity_tag_prefix`` plan: pin three orthogonal
contracts so future refactors cannot reintroduce raw entity-tag DDL
into the declarative module.

- **NT1 — AST scan over ``decl/*.py``:** Walks every non-test ``.py``
  file under ``decl/`` and asserts no string literal contains
  ``"CREATE TAG"``, ``"DROP TAG"``, ``"ALTER TAG"``, or
  ``"SHOW TAGS"``.  Docstrings are excluded — they may legitimately
  describe the historical DDL while explaining why the current code
  no longer emits it.
- **NT2 — Behavioural negative test (write path):** Patches
  ``session.sql`` to raise ``AssertionError`` on any invocation;
  runs ``execute_plan`` against an entity-only plan with a mocked
  ``FeatureStore``; asserts no exception (proves entity ops route
  exclusively through the imperative API).
- **NT3 — Behavioural negative test (read path):** Same pattern for
  :func:`fetch_entity_rows` — confirms the read path makes zero
  ``session.sql`` calls and goes through ``fs.list_entities()``
  instead.

These are *locking* tests: they don't drive new behaviour, they pin
the post-refactor architecture so a future "quick fix" cannot
silently regress to raw SQL emission.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.imperative_executor import (
    execute_plan,
    fetch_entity_rows,
)
from snowflake.ml.feature_store.decl.types import Plan, PlanOp, PlanOptions

_FORBIDDEN_LITERALS = (
    "CREATE TAG",
    "DROP TAG",
    "ALTER TAG",
    "SHOW TAGS",
)


def _decl_module_paths() -> Iterator[Path]:
    """Yield every non-test ``.py`` file under the decl package.

    Excludes:

    * ``__pycache__`` directories.
    * ``tests/`` subpackage (test fixtures may legitimately contain
      the forbidden DDL — that's the whole point of asserting the
      production modules don't).

    Yields:
        Each ``Path`` of a production decl module, sorted for stable
        parametrise ordering.
    """
    decl_dir = Path(__file__).resolve().parent.parent
    assert decl_dir.name == "decl", f"Expected this file to live in decl/tests/; computed parent={decl_dir!r}"

    for path in sorted(decl_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if "tests" in path.parts:
            continue
        yield path


def _collect_docstring_node_ids(tree: ast.AST) -> set[int]:
    """Return the set of ``id(node)`` values for docstring constants.

    The AST scan must allow docstrings to legitimately mention the
    historical SQL (e.g. a comment in the ``fetch_entity_rows``
    docstring saying "previously this issued ``SHOW TAGS``").  Only
    *executable* string literals are forbidden.

    Args:
        tree: A parsed Python AST for the module under inspection.

    Returns:
        ``set`` of ``id(node)`` values; each ID identifies an
        ``ast.Constant`` string node that serves as the docstring of
        a module, class, function, or async function.
    """
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            docstring_ids.add(id(first.value))
    return docstring_ids


class TestNT1NoEntityTagDdlInDeclSource:
    """AST scan: no executable string literal in ``decl/*.py`` may
    contain raw entity-tag DDL fragments.
    """

    @pytest.mark.parametrize(
        "module_path",
        list(_decl_module_paths()),
        ids=lambda p: p.relative_to(Path(__file__).resolve().parent.parent).as_posix(),
    )
    def test_no_forbidden_literals_in_module(self, module_path: Path) -> None:
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        docstring_ids = _collect_docstring_node_ids(tree)

        offenders: list[tuple[int, str, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            if id(node) in docstring_ids:
                continue
            for forbidden in _FORBIDDEN_LITERALS:
                if forbidden in node.value:
                    offenders.append(
                        (
                            getattr(node, "lineno", -1),
                            forbidden,
                            node.value,
                        )
                    )

        assert not offenders, (
            f"{module_path} contains forbidden entity-tag SQL literals "
            "outside docstrings — the declarative module must route "
            "all entity DDL through the imperative ``FeatureStore`` API. "
            f"Offenders: {offenders!r}"
        )


def _entity_only_plan() -> Plan:
    """Build a minimal plan containing only entity ops.

    All three OpKinds (CREATE / UPDATE / DROP) are exercised so the
    behavioural negative test covers every branch in the
    ``_execute_op`` dispatcher.

    Returns:
        Plan with one CREATE_ENTITY, one UPDATE_ENTITY, and one
        DROP_ENTITY op (the latter destructive so the
        ``allow_recreate`` gate must be set to true at apply time).
    """
    return Plan(
        ops=[
            PlanOp(
                kind=OpKind.CREATE_ENTITY,
                name="USER",
                payload={
                    "kind": "Entity",
                    "name": "USER",
                    "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                    "description": "u",
                },
            ),
            PlanOp(
                kind=OpKind.UPDATE_ENTITY,
                name="OTHER",
                payload={
                    "kind": "Entity",
                    "name": "OTHER",
                    "join_keys": [{"name": "OTHER_ID", "type": "StringType"}],
                    "description": "o-new",
                },
            ),
            PlanOp(
                kind=OpKind.DROP_ENTITY,
                name="GONE",
                destructive=True,
                payload={"kind": "Entity", "name": "GONE"},
            ),
        ],
        warnings=[],
    )


class TestNT2NoSessionSqlForEntityWritePath:
    """``execute_plan`` for an entity-only plan must NEVER touch
    ``session.sql`` — every op must route through ``fs.register_entity``
    / ``fs.delete_entity`` / ``fs.update_entity``.

    Tripwire mechanism: ``session.sql.side_effect`` raises
    ``AssertionError`` on every call.  If anything in the entity
    branch slips back to raw SQL, the test fails loudly with the
    exact SQL string at the failure site.
    """

    def test_entity_ops_emit_zero_session_sql_calls(self) -> None:
        plan = _entity_only_plan()
        session = MagicMock(name="session")

        def _refuse(sql_text: str) -> None:
            raise AssertionError(
                "session.sql called for entity op — raw entity-tag SQL "
                "must not be emitted by the declarative executor. "
                f"Forbidden SQL: {sql_text!r}"
            )

        session.sql.side_effect = _refuse

        fs = MagicMock(name="FeatureStore")
        fs.register_entity.return_value = MagicMock()
        fs.update_entity.return_value = MagicMock()

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(
                plan,
                session,
                "DB",
                "SCH",
                "WH",
                PlanOptions(allow_recreate=True),
            )

        assert result.status == "applied", (
            f"All three entity ops must succeed via the imperative API.  "
            f"Got status={result.status!r}, ops={result.ops!r}."
        )
        fs.register_entity.assert_called_once()
        fs.update_entity.assert_called_once()
        fs.delete_entity.assert_called_once()
        session.sql.assert_not_called()


class TestNT3NoSessionSqlForEntityReadPath:
    """``fetch_entity_rows`` must NEVER touch ``session.sql`` — the
    read path goes through ``fs.list_entities()`` exclusively.

    Same tripwire pattern as NT2: ``session.sql.side_effect`` raises
    ``AssertionError`` on every call.  If the read path slips back
    to raw ``SHOW TAGS``, the test fails with the SQL string.
    """

    def test_fetch_entity_rows_emits_zero_session_sql_calls(self) -> None:
        session = MagicMock(name="session")

        def _refuse(sql_text: str) -> None:
            raise AssertionError(
                "session.sql called for fetch_entity_rows — the read "
                "path must go through fs.list_entities() instead. "
                f"Forbidden SQL: {sql_text!r}"
            )

        session.sql.side_effect = _refuse

        listed_df = MagicMock(name="listed_df")
        listed_df.collect.return_value = []

        fs = MagicMock(name="FeatureStore")
        fs.list_entities.return_value = listed_df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_entity_rows(session, "DB", "SCH", "WH")

        assert rows == [], f"Empty list_entities() output must yield an empty SHOW TAGS " f"row list; got {rows!r}."
        fs.list_entities.assert_called_once()
        session.sql.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
