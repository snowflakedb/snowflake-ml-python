"""Hypothesis tests for the init-first declarative-client policy.

This file enumerates the testable predictions ("hypotheses") behind the
``remove_duplicate_entity_tag_prefix`` plan. Each test is a red
trip-wire that turns green when the corresponding implementation phase
ships. If a hypothesis fails to flip green during its phase, STOP and
ask the operator before continuing — the falsification surfaces a
design decision that must be approved.

Hypothesis ↔ implementation phase ↔ falsification → design impact:

- **H1** — ``FeatureStore.register_entity`` is idempotent on duplicates.
  Implementation phase: Phase 4 (route ``CREATE_ENTITY`` through the
  imperative API). Falsification: declarative ``CREATE_ENTITY`` op
  semantics shift from silent-success to loud warning — operator must
  approve.

- **H2** — ``FeatureStore.delete_entity`` is NOT idempotent (raises
  ``NOT_FOUND`` when entity is missing). Implementation phase: Phase 4
  (route ``DROP_ENTITY`` through the imperative API with a
  soft-NOT_FOUND wrap matching today's silent ``DROP TAG IF EXISTS``
  behavior). Falsification: either soft-wrap NOT_FOUND (recommended
  path) or let ``DROP_ENTITY`` fail loudly on stale plan files.

- **H3** — ``FeatureStore.delete_entity`` enforces FV-reference checks
  natively (raises ``SNOWML_DELETE_FAILED``), so the executor only
  needs to rewrap the error code (not string-match the message).
  Implementation phase: Phase 4. Falsification: the planner must
  enforce the dependency check itself (more work).

- **H4** — ``FeatureStore.update_entity`` remains ``desc=``-only
  (join keys are not an update argument). **GREEN-as-pin**: locks the
  public signature so declarative apply does not invent unsupported
  kwargs.

- **H5** — ``FeatureStore.__init__(FAIL_IF_NOT_EXIST)`` raises a
  deterministic exception when the schema lacks internal tags, and
  ``decl.assert_feature_store_initialized`` rewraps that exception as
  ``FeatureStoreNotInitializedError`` carrying actionable guidance.
  Implementation phase: Phase 2. Falsification: rewrap is unreliable;
  must enumerate every snowml-core error variant.

- **H6** — All BUG_BASH steps already operate on an initialized
  feature store. **Integration-level** — see
  ``scripts/verify_bug_bash.sh`` (Phase 8). Stub here for
  documentation only.

- **H7** — ``decl/state.py``, ``decl/api.py``, ``decl/exporter.py``
  only need ``ENTITY_TAG_PREFIX`` for string parsing, never for SQL
  execution. **GREEN-as-pin**: locks the invariant that future
  refactors don't introduce raw entity-tag SQL into these three
  parsing modules.

- **H8** — Removing the ``_fs_box`` / ``_get_fs`` lazy-construction
  dance does not regress entity-only plans against initialized
  schemas. Implementation phase: Phase 4. Falsification:
  ``FeatureStore.__init__`` has unexpected side effects (e.g.
  warehouse validation); design an opt-out.

- **H9** — ``snow feature list`` against an uninitialized schema is
  acceptable as a hard error. **Integration-level** — see
  ``scripts/verify_uninitialized_schema.sh`` (Phase 8). Stub here for
  documentation only.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.feature_store import feature_store as fs_module
from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.errors import DependencyError
from snowflake.ml.feature_store.decl.imperative_executor import execute_plan
from snowflake.ml.feature_store.decl.types import Plan, PlanOp, PlanOptions
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.test_utils import pytest_driver


def _entity_payload(
    name: str,
    join_keys: list[str],
    description: str = "",
) -> dict[str, Any]:
    """Build an entity op payload matching the planner's compiled shape."""
    return {
        "kind": "Entity",
        "name": name,
        "join_keys": [{"name": jk, "type": "StringType"} for jk in join_keys],
        "description": description,
    }


def _collect_docstring_node_ids(tree: ast.AST) -> set[int]:
    """Identify ``ast.Constant`` nodes that serve as module / class /
    function docstrings so an AST string-literal scan can skip them.

    Args:
        tree: The parsed AST to walk.

    Returns:
        Set of ``id(node)`` values for every ``Constant`` string that
        appears as the docstring of a module, class, function, or async
        function.
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


# ---------------------------------------------------------------------------
# H1 — register_entity idempotency under the declarative CREATE_ENTITY path
# ---------------------------------------------------------------------------


class TestH1RegisterEntityIdempotent:
    """The declarative CREATE_ENTITY op must route through
    ``FeatureStore.register_entity`` (Phase 4), and ``register_entity``
    must be idempotent — re-registering an existing entity warns and
    returns without raising.

    RED today (decl emits raw ``CREATE TAG`` SQL; ``register_entity``
    is never called). GREEN after Phase 4.
    """

    def test_create_entity_op_routes_through_register_entity(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER",
                    payload=_entity_payload("USER", ["USER_ID"], "u"),
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor

        fs = MagicMock(name="FeatureStore")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.register_entity.assert_called_once()
        call_arg = fs.register_entity.call_args.args[0]
        assert isinstance(call_arg, Entity)
        assert str(call_arg.name).upper() == "USER"

    def test_register_entity_idempotent_warn_branch_present_in_source(self) -> None:
        """Source-level pin for ``register_entity``'s idempotency branch.

        The imperative method's behaviour is well-covered by
        ``feature_store_test.py``; this test only guards against a
        future refactor that removes the silent-warn branch (which
        Phase 4 relies on for ``CREATE_ENTITY`` idempotency).
        """
        source = inspect.getsource(fs_module.FeatureStore.register_entity)
        assert "already exists" in source
        assert "UserWarning" in source


# ---------------------------------------------------------------------------
# H2 — delete_entity is NOT idempotent; declarative path soft-wraps NOT_FOUND
# ---------------------------------------------------------------------------


class TestH2DeleteEntityNotIdempotent:
    """``FeatureStore.delete_entity`` raises ``NOT_FOUND`` when the
    entity does not exist, while today's raw ``DROP TAG IF EXISTS`` is
    silent.  The Phase-4 declarative ``DROP_ENTITY`` op must soft-wrap
    ``NOT_FOUND`` as an info-level skip so re-applying a plan that
    contains a stale DROP is forgiving.

    RED today (decl emits raw ``DROP TAG IF EXISTS``; ``delete_entity``
    is never called, so the soft-wrap branch is never reached). GREEN
    after Phase 4.
    """

    def test_decl_drop_entity_soft_wraps_not_found_as_skip(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_ENTITY,
                    name="USER",
                    payload={"name": "USER", "kind": "Entity"},
                    destructive=True,
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor

        not_found = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Entity USER does not exist."),
        )

        fs = MagicMock(name="FeatureStore")
        fs.delete_entity.side_effect = not_found

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        fs.delete_entity.assert_called_once()

    def test_delete_entity_not_found_branch_present_in_source(self) -> None:
        """Source-level pin for ``delete_entity``'s ``NOT_FOUND`` raise
        branch.  ``feature_store_test.py`` already covers the runtime
        behaviour; this guards against a refactor that silences the
        ``NOT_FOUND`` raise that Phase 4 depends on for soft-wrapping.
        """
        source = inspect.getsource(fs_module.FeatureStore.delete_entity)
        assert "NOT_FOUND" in source
        assert "does not exist" in source


# ---------------------------------------------------------------------------
# H3 — delete_entity enforces FV-reference check natively
# ---------------------------------------------------------------------------


class TestH3DeleteEntityFVDependency:
    """``FeatureStore.delete_entity`` raises
    ``SNOWML_DELETE_FAILED`` when an active feature view references
    the entity.  The Phase-4 declarative ``DROP_ENTITY`` op must
    rewrap this error code as ``DependencyError`` so the CLI surfaces
    a clear "drop the FV first" message — no string-matching on the
    backend error text.

    RED today (decl emits raw ``DROP TAG`` and string-matches the
    backend error; it never sees the ``SNOWML_DELETE_FAILED`` code).
    GREEN after Phase 4.
    """

    def test_decl_drop_entity_rewraps_snowml_delete_failed_as_dependency_error(
        self,
    ) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_ENTITY,
                    name="USER",
                    payload={"name": "USER", "kind": "Entity"},
                    destructive=True,
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor

        dep_error = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.SNOWML_DELETE_FAILED,
            original_exception=ValueError("Cannot delete Entity USER due to active FeatureViews: ['MY_FV']."),
        )

        fs = MagicMock(name="FeatureStore")
        fs.delete_entity.side_effect = dep_error

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ), pytest.raises(DependencyError):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        fs.delete_entity.assert_called_once()


# ---------------------------------------------------------------------------
# H4 — update_entity is desc-only (no join_keys=)
# ---------------------------------------------------------------------------


class TestH4UpdateEntityDescOnly:
    """``FeatureStore.update_entity`` accepts ``desc=`` only.

    Join keys are set at ``register_entity`` time and are not an
    ``update_entity`` argument. Pin the public signature so the
    declarative executor does not pass unsupported kwargs.
    """

    def test_update_entity_signature_preserves_desc_keyword(self) -> None:
        sig = inspect.signature(fs_module.FeatureStore.update_entity)
        assert (
            "desc" in sig.parameters
        ), "FeatureStore.update_entity must preserve the existing `desc=` keyword (backward-compat pin for H4)."
        assert sig.parameters["desc"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["desc"].default is None
        assert "join_keys" not in sig.parameters


# ---------------------------------------------------------------------------
# H5 — FeatureStore.__init__(FAIL_IF_NOT_EXIST) errors are rewrapped
# ---------------------------------------------------------------------------


class TestH5AssertFeatureStoreInitializedRewrap:
    """``decl.imperative_executor.assert_feature_store_initialized`` must
    construct a ``FeatureStore(FAIL_IF_NOT_EXIST)`` and rewrap the
    snowml-core ``NOT_FOUND`` exception as the operator-facing
    ``FeatureStoreNotInitializedError`` carrying ``"snow feature init"``
    in its message.

    RED today (neither the helper nor the error class exist). GREEN
    after Phase 2.
    """

    def test_helper_exists_in_imperative_executor(self) -> None:
        mod = importlib.import_module("snowflake.ml.feature_store.decl.imperative_executor")
        assert hasattr(mod, "assert_feature_store_initialized"), (
            "decl.imperative_executor.assert_feature_store_initialized " "must exist after Phase 2 (H5)."
        )

    def test_error_class_exists_in_errors_module(self) -> None:
        mod = importlib.import_module("snowflake.ml.feature_store.decl.errors")
        assert hasattr(mod, "FeatureStoreNotInitializedError"), (
            "decl.errors.FeatureStoreNotInitializedError must exist " "after Phase 2 (H5)."
        )

    def test_error_class_is_re_exported_from_decl_api(self) -> None:
        mod = importlib.import_module("snowflake.ml.feature_store.decl.api")
        assert hasattr(mod, "FeatureStoreNotInitializedError"), (
            "decl.api must re-export FeatureStoreNotInitializedError " "for CLI consumers (H5)."
        )
        assert hasattr(mod, "assert_feature_store_initialized"), (
            "decl.api must re-export assert_feature_store_initialized " "for CLI consumers (H5)."
        )

    def test_assert_feature_store_initialized_rewraps_missing_tags(self) -> None:
        from snowflake.ml.feature_store.decl import (
            errors as decl_errors,
            imperative_executor,
        )

        missing_tag = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError(
                "Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist. "
                "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
            ),
        )

        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_tag,
        ), pytest.raises(decl_errors.FeatureStoreNotInitializedError) as exc_info:
            imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

        assert "DB" in str(exc_info.value)
        assert "SCH" in str(exc_info.value)
        assert "snow feature init" in str(exc_info.value)

    def test_assert_feature_store_initialized_rewraps_missing_schema(self) -> None:
        from snowflake.ml.feature_store.decl import (
            errors as decl_errors,
            imperative_executor,
        )

        missing_schema = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError(
                "Feature store schema SCH does not exist. "
                "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
            ),
        )

        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_schema,
        ), pytest.raises(decl_errors.FeatureStoreNotInitializedError) as exc_info:
            imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

        assert "snow feature init" in str(exc_info.value)

    def test_assert_feature_store_initialized_returns_fs_on_success(self) -> None:
        from snowflake.ml.feature_store.decl import imperative_executor

        fs_instance = MagicMock(name="FeatureStore")
        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs_instance,
        ):
            result = imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

        assert result is fs_instance


# ---------------------------------------------------------------------------
# H6 — BUG_BASH steps already operate on initialized feature store
# (integration-level; lives in Phase 8 verify_bug_bash.sh)
# ---------------------------------------------------------------------------


class TestH6BugBashAlwaysInitialized:
    """**Integration-level placeholder.**

    H6 is verified by running ``scripts/verify_bug_bash.sh`` after the
    refactor lands (Phase 8). A zero-finding run confirms every BUG_BASH
    step operates on an already-initialized feature store.
    """

    @pytest.mark.skip(reason="Integration test — see scripts/verify_bug_bash.sh (Phase 8).")
    def test_bug_bash_runs_clean(self) -> None:
        pass


# ---------------------------------------------------------------------------
# H7 — decl/state.py, decl/api.py, decl/exporter.py contain no entity-tag SQL
# (green-as-pin: locks the invariant for future refactors)
# ---------------------------------------------------------------------------


class TestH7DeclParsingModulesEmitNoEntityTagSql:
    """The three parsing modules only consume ``ENTITY_TAG_PREFIX`` as a
    string for output parsing — they never emit ``CREATE TAG``,
    ``DROP TAG``, ``ALTER TAG``, or ``SHOW TAGS`` SQL.

    GREEN today (these modules don't emit entity-tag SQL). Stays GREEN
    as a refactor trip-wire.
    """

    PARSING_MODULE_PATHS = (
        Path(__file__).resolve().parents[1] / "state.py",
        Path(__file__).resolve().parents[1] / "api.py",
        Path(__file__).resolve().parents[1] / "exporter.py",
    )

    FORBIDDEN_SQL_FRAGMENTS = (
        "CREATE TAG",
        "DROP TAG",
        "ALTER TAG",
        "SHOW TAGS",
    )

    @pytest.mark.parametrize("module_path", PARSING_MODULE_PATHS)
    def test_no_entity_tag_ddl_in_string_literals(self, module_path: Path) -> None:
        source = module_path.read_text()
        tree = ast.parse(source)

        docstring_ids = _collect_docstring_node_ids(tree)

        offending: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if id(node) in docstring_ids:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                upper = node.value.upper()
                for forbidden in self.FORBIDDEN_SQL_FRAGMENTS:
                    if forbidden in upper:
                        offending.append((node.lineno, node.value[:120]))
                        break

        assert not offending, (
            f"{module_path.name} contains forbidden entity-tag SQL " f"literals (H7 trip-wire): {offending}"
        )


# ---------------------------------------------------------------------------
# H8 — eager FeatureStore construction for entity-only plans
# ---------------------------------------------------------------------------


class TestH8EagerFeatureStoreConstruction:
    """After Phase 4 ``execute_plan`` constructs ``FeatureStore`` eagerly
    (no ``_fs_box`` / ``_get_fs`` lazy dance) so the init-first invariant
    fires uniformly across FV-only, entity-only, and mixed plans.

    RED today (entity-only plans deliberately skip ``FeatureStore``
    construction). GREEN after Phase 4.
    """

    def test_execute_plan_constructs_feature_store_eagerly_for_entity_only_plans(
        self,
    ) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER",
                    payload=_entity_payload("USER", ["USER_ID"], "u"),
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor
        fs = MagicMock(name="FeatureStore")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ) as mock_fs_cls:
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        mock_fs_cls.assert_called_once()


# ---------------------------------------------------------------------------
# H9 — snow feature list against uninitialized schema is a hard error
# (integration-level; lives in Phase 8 verify_uninitialized_schema.sh)
# ---------------------------------------------------------------------------


class TestH9UninitializedSchemaIsHardError:
    """**Integration-level placeholder.**

    H9 is verified by ``scripts/verify_uninitialized_schema.sh`` —
    every ``snow feature`` command except ``init`` must exit non-zero
    with the "Run ``snow feature init``" guidance when the target
    schema lacks internal tags.
    """

    @pytest.mark.skip(reason="Integration test — see scripts/verify_uninitialized_schema.sh (Phase 8).")
    def test_feature_list_against_uninitialized_schema_exits_nonzero(self) -> None:
        pass


if __name__ == "__main__":
    pytest_driver.main()
