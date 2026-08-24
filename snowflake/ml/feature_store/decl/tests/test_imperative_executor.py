"""Tests for decl/imperative_executor.py — session priming hook + entity resolution."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.errors import DependencyError
from snowflake.ml.feature_store.decl.imperative_executor import (
    _build_feature_df,
    _build_feature_view,
    _build_features,
    execute_plan,
    fetch_entity_rows,
)
from snowflake.ml.feature_store.decl.types import Plan, PlanOp, PlanOptions
from snowflake.ml.feature_store.entity import Entity


class TestExecutePlanFeatureStoreConstruction:
    def test_execute_plan_constructs_feature_store_eagerly_for_entity_only_plans(self) -> None:
        """Entity ops route through the imperative ``FeatureStore``
        API, so ``FeatureStore`` MUST be constructed eagerly — even
        when the plan contains only entity ops — so the init-first
        invariant fires uniformly across FV-only, entity-only, and
        mixed plans (hypothesis H8).
        """
        plan = Plan(
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
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
        ) as mock_fs:
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        mock_fs.assert_called_once()

    def test_execute_plan_raises_feature_store_not_initialized_when_tags_missing(self) -> None:
        """When the target schema lacks the bootstrap tags, the
        eager ``FeatureStore`` construction raises ``NOT_FOUND`` and
        ``execute_plan`` rewraps it via ``assert_feature_store_initialized``
        so the CLI surfaces the actionable "run snow feature init"
        error instead of a snowml-core stack trace (NT4).
        """
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
        from snowflake.ml.feature_store.decl.errors import (
            FeatureStoreNotInitializedError,
        )

        plan = Plan(
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
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor

        missing_tag = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist."),
        )

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_tag,
        ), pytest.raises(FeatureStoreNotInitializedError):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())


class TestFetchEntityRows:
    """Verify ``fetch_entity_rows`` translates ``FeatureStore.list_entities()``
    output back to the SHOW TAGS row shape that
    :func:`state._build_entity_object` consumes.

    The imperative ``list_entities()`` helper:
      - strips the ``SNOWML_FEATURE_STORE_ENTITY_`` prefix from ``NAME``;
      - returns ``JOIN_KEYS`` as a JSON-decodable string;
      - renames ``comment`` to ``DESC`` and ``owner`` to ``OWNER``.

    The translation must re-add the prefix, copy the JSON join_keys to
    ``allowed_values``, and copy ``DESC`` to ``comment`` so existing
    entity-row consumers (``state.py``, ``api.enrich_list_results``)
    keep working unchanged.
    """

    def test_translates_imperative_rows_to_show_tags_shape(self) -> None:
        session = MagicMock(name="session")

        def make_row(name: Any, join_keys: Any, desc: Any, owner: Any) -> Any:
            return {"NAME": name, "JOIN_KEYS": join_keys, "DESC": desc, "OWNER": owner}

        listed_rows = [
            make_row("USER", '["USER_ID"]', "User entity.", "ROLE_X"),
            make_row("DEVICE", '["DEVICE_ID"]', "", "ROLE_X"),
        ]
        df = MagicMock(name="DataFrame")
        df.collect.return_value = listed_rows
        fs = MagicMock(name="FeatureStore")
        fs.list_entities.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_entity_rows(session, "DB", "SCH", "WH")

        assert len(rows) == 2

        first = rows[0]
        assert first["name"] == "SNOWML_FEATURE_STORE_ENTITY_USER"
        assert first["database_name"] == "DB"
        assert first["schema_name"] == "SCH"
        assert first["allowed_values"] == '["USER_ID"]'
        assert first["comment"] == "User entity."
        assert first.get("owner") == "ROLE_X"

        second = rows[1]
        assert second["name"] == "SNOWML_FEATURE_STORE_ENTITY_DEVICE"
        assert second["allowed_values"] == '["DEVICE_ID"]'
        assert second.get("comment", "") == ""

    def test_fetch_entity_rows_raises_feature_store_not_initialized_when_tags_missing(self) -> None:
        """``fetch_entity_rows`` no longer falls back to a raw
        ``SHOW TAGS`` query when ``FeatureStore`` construction fails —
        the init-first policy makes that scenario a hard error.  The
        snowml-core ``NOT_FOUND`` is rewrapped as
        ``FeatureStoreNotInitializedError`` so the CLI can surface the
        "run ``snow feature init``" guidance (NT5).
        """
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
        from snowflake.ml.feature_store.decl.errors import (
            FeatureStoreNotInitializedError,
        )

        session = MagicMock(name="session")
        missing_tag = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist."),
        )
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_tag,
        ), pytest.raises(FeatureStoreNotInitializedError):
            fetch_entity_rows(session, "DB", "SCH", "WH")

        # Negative pin: the read path must NOT have attempted any raw
        # ``SHOW TAGS`` fallback against the session.
        session.sql.assert_not_called()

    def test_translation_round_trips_through_build_entity_object(self) -> None:
        from snowflake.ml.feature_store.decl.state import _build_entity_object

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            {
                "NAME": "USER",
                "JOIN_KEYS": '["USER_ID"]',
                "DESC": "User entity.",
                "OWNER": "ROLE_X",
            }
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_entities.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            translated = fetch_entity_rows(session, "DB", "SCH")

        applied = _build_entity_object(translated[0], "DB", "SCH")
        assert applied is not None
        assert applied.kind == "Entity"
        assert applied.name == "USER"
        assert applied.details["join_keys"] == ["USER_ID"]


class TestExecuteEntityOps:
    """Verify entity-DDL ops route through imperative ``FeatureStore`` calls.

    After the init-first refactor (Phase 4), ``CREATE_ENTITY``,
    ``DROP_ENTITY``, and ``UPDATE_ENTITY`` call ``register_entity``,
    ``delete_entity``, and ``update_entity`` respectively.  No raw
    ``CREATE TAG`` / ``DROP TAG`` / ``ALTER TAG`` SQL is issued from
    ``decl/`` — the imperative API is the single execution path for
    every entity lifecycle operation.

    The ``delete_entity`` paths additionally rewrap snowml-core error
    codes: ``NOT_FOUND`` becomes a soft-skip (H2 design call —
    forgiving re-apply of stale plan files), and
    ``SNOWML_DELETE_FAILED`` becomes a :class:`DependencyError` with
    operator-actionable guidance (H3).
    """

    def _entity_payload(self, name: Any, join_keys: Any, description: Any = "") -> Any:
        return {
            "kind": "Entity",
            "name": name,
            "join_keys": [{"name": jk, "type": "StringType"} for jk in join_keys],
            "description": description,
        }

    def test_create_entity_calls_fs_register_entity(self) -> None:
        """``CREATE_ENTITY`` dispatches to ``FeatureStore.register_entity``
        with a real ``Entity`` carrying the spec's name, join keys, and
        description.  No raw SQL is emitted by the executor.
        """
        session = MagicMock(name="session")
        fs = MagicMock(name="FeatureStore")

        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER",
                    payload=self._entity_payload("USER", ["USER_ID"], "u"),
                )
            ],
            warnings=[],
        )
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.register_entity.assert_called_once()
        passed = fs.register_entity.call_args.args[0]
        assert isinstance(passed, Entity)
        assert str(passed.name).upper() == "USER"
        # The executor must NOT have issued any entity-tag SQL.
        for call in session.sql.call_args_list:
            sql_text = call.args[0]
            assert "CREATE TAG" not in sql_text.upper()
            assert "ALTER TAG" not in sql_text.upper()
            assert "DROP TAG" not in sql_text.upper()

    def test_drop_entity_calls_fs_delete_entity(self) -> None:
        session = MagicMock(name="session")
        fs = MagicMock(name="FeatureStore")

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
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        fs.delete_entity.assert_called_once_with("USER")
        for call in session.sql.call_args_list:
            assert "DROP TAG" not in call.args[0].upper()

    def test_drop_entity_soft_wraps_not_found_as_skip(self) -> None:
        """H2 design call: re-applying a plan with a stale
        ``DROP_ENTITY`` for an entity that's already absent must be
        forgiving.  ``delete_entity``'s ``NOT_FOUND`` is logged at
        info level and the op completes successfully.
        """
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )

        not_found = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Entity USER does not exist."),
        )

        session = MagicMock(name="session")
        fs = MagicMock(name="FeatureStore")
        fs.delete_entity.side_effect = not_found

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
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        assert result.status == "applied"
        fs.delete_entity.assert_called_once_with("USER")

    def test_drop_entity_rewraps_referenced_error_as_dependency_error(self) -> None:
        """H3 design call: ``delete_entity``'s ``SNOWML_DELETE_FAILED``
        error code (raised when an active FV still references the
        entity) is rewrapped as a :class:`DependencyError` with a
        clear "drop the FV first" message.  No string-matching on the
        backend error text — the executor branches on the structured
        error code.
        """
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )

        dep_error = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.SNOWML_DELETE_FAILED,
            original_exception=ValueError("Cannot delete Entity USER due to active FeatureViews: ['MY_FV']."),
        )

        session = MagicMock(name="session")
        fs = MagicMock(name="FeatureStore")
        fs.delete_entity.side_effect = dep_error

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
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ), pytest.raises(DependencyError) as exc_info:
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        message = str(exc_info.value)
        assert "USER" in message
        assert "drop the FV first" in message.lower() or "feature view" in message.lower()

    def test_drop_entity_propagates_unexpected_snowml_errors(self) -> None:
        """Errors with codes other than ``NOT_FOUND`` /
        ``SNOWML_DELETE_FAILED`` propagate verbatim — no silent
        swallowing of e.g. permission errors.
        """
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )

        unexpected = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError("permission denied"),
        )

        session = MagicMock(name="session")
        fs = MagicMock(name="FeatureStore")
        fs.delete_entity.side_effect = unexpected

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
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ), pytest.raises(snowml_exceptions.SnowflakeMLException):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

    def test_update_entity_calls_fs_update_entity_with_join_keys_and_desc(self) -> None:
        """``UPDATE_ENTITY`` dispatches to
        ``FeatureStore.update_entity(name, desc=..., join_keys=...)``,
        threading both fields from the payload through.  Pin the new
        Phase 3 ``join_keys=`` keyword is wired correctly from the
        declarative executor.
        """
        session = MagicMock(name="session")
        fs = MagicMock(name="FeatureStore")

        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.UPDATE_ENTITY,
                    name="USER",
                    payload=self._entity_payload("USER", ["USER_ID", "ORG_ID"], "Updated desc"),
                    destructive=False,
                )
            ],
            warnings=[],
        )
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.update_entity.assert_called_once()
        call = fs.update_entity.call_args
        # Positional or keyword — accept either, but the name must be present.
        if call.args:
            assert call.args[0] == "USER"
        else:
            assert call.kwargs.get("name") == "USER"
        assert call.kwargs.get("desc") == "Updated desc"
        assert call.kwargs.get("join_keys") == ["USER_ID", "ORG_ID"]
        for sql_call in session.sql.call_args_list:
            assert "ALTER TAG" not in sql_call.args[0].upper()


# ---------------------------------------------------------------------------
# TestBuildFeatureView — entity resolution contract
#
# Pin the contract that a CREATE_FV / UPDATE_FV / RECREATE_FV op MUST
# resolve every entity referenced in the payload's
# ``ordered_entity_column_names`` against the live ``FeatureStore``
# registry — never fabricate a synthetic ``f"{fv_name}_entity"``
# placeholder.  The synthetic shape was the root cause of
# ``(2101) Entity USER_CLICK_STATS_DECL_ENTITY has not been registered``
# when ``snow feature apply`` tried to deploy a freshly-planned FV
# against a schema where the spec-declared entity was already
# registered with its real name.
# ---------------------------------------------------------------------------


def _fv_payload(
    name: str = "USER_CLICK_STATS_DECL",
    version: str = "V1",
    entity_names: list[str] | None = None,
    *,
    table: str = "CLICKS_RAW",
) -> dict[str, Any]:
    """Build a minimal CREATE_FV payload matching what the planner emits.

    Args:
        name: Feature view logical name.
        version: Feature view version string.
        entity_names: Names listed in ``ordered_entity_column_names`` —
            each must correspond to an Entity that is (or will be)
            registered before the FV op executes.
        table: Source table reference; ``_build_feature_df`` will turn
            this into a ``session.table(<db>.<schema>.<table>)`` call,
            which we mock at the session level.

    Returns:
        A dict shaped like ``op.payload`` for an FV create/update op.
    """
    if entity_names is None:
        entity_names = ["USER_ID"]
    return {
        "kind": "BatchFeatureView",
        "name": name,
        "version": version,
        "entities": list(entity_names),
        "sources": [{"name": "clicks", "source_type": "Batch", "table": table}],
        "features": [],
    }


def _mock_fv_session() -> MagicMock:
    """Mock Snowpark session: ``session.table(...)`` returns a stub DataFrame.

    Returns:
        ``MagicMock`` configured so that ``_build_feature_df`` can return
        without exercising real Snowpark.
    """
    session = MagicMock()
    session.table.return_value = MagicMock(name="snowpark_dataframe")
    session.sql.return_value = MagicMock(name="snowpark_dataframe")
    return session


def _registered_entity(
    name: str,
    join_keys: list[str] | None = None,
    desc: str = "",
) -> Entity:
    """Return a real ``Entity`` instance as ``FeatureStore.get_entity`` would.

    Args:
        name: Entity name (matches ``ordered_entity_column_names`` entry).
        join_keys: Authoritative join keys recorded in Snowflake.  Defaults
            to ``[name]`` (the common single-key case).
        desc: Free-text description.

    Returns:
        A real ``Entity`` constructed with the same keyword arguments
        ``FeatureStore.get_entity`` would use.
    """
    if join_keys is None:
        join_keys = [name]
    return Entity(name=name, join_keys=join_keys, desc=desc)


class TestBuildFeatureView:
    """Unit-level coverage of ``_build_feature_view`` entity resolution.

    These tests deliberately patch ``FeatureView`` itself (the class
    constructed inside ``_build_feature_view``) so the assertions are
    on the kwargs passed to the constructor, not on the resulting
    instance.  This isolates the entity-resolution behaviour from
    snowml-core's ``FeatureView._validate`` (which would otherwise
    require a real Snowpark DataFrame).
    """

    def _build(
        self,
        payload: dict[str, Any],
        get_entity_side_effect: Any = None,
        get_entity_return_value: Any = None,
    ) -> dict[str, Any]:
        """Drive ``_build_feature_view`` with a mocked FS + FeatureView.

        Args:
            payload: CREATE_FV payload.
            get_entity_side_effect: If set, used as the ``side_effect`` of
                the mocked ``fs.get_entity``.  Useful for either raising
                or returning per-name results.
            get_entity_return_value: If set (and ``side_effect`` is not),
                used as a fixed return value.

        Returns:
            ``dict[str, Any]`` of the kwargs ``FeatureView`` was invoked
            with — including ``entities`` (the resolved list).
        """
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        if get_entity_side_effect is not None:
            fs.get_entity.side_effect = get_entity_side_effect
        elif get_entity_return_value is not None:
            fs.get_entity.return_value = get_entity_return_value

        with patch("snowflake.ml.feature_store.feature_view.FeatureView") as fv_ctor:
            fv_ctor.return_value = MagicMock(name="feature_view_instance")
            _build_feature_view(
                payload,
                session,
                "JKEW_DB",
                "JKEW_SCHEMA",
                "JKEW_WAREHOUSE",
                fs=fs,
            )
            assert fv_ctor.called, "FeatureView constructor must be invoked exactly once"
            _, kwargs = fv_ctor.call_args
            return dict(kwargs)

    def test_single_entity_resolved_from_registry(self) -> None:
        """Single entity in ``ordered_entity_column_names`` gets looked up by name."""
        registered = _registered_entity("USER_ID", join_keys=["USER_ID"], desc="Click users")
        kwargs = self._build(_fv_payload(entity_names=["USER_ID"]), get_entity_return_value=registered)

        entities = kwargs["entities"]
        assert len(entities) == 1, "Exactly one entity must be attached"
        assert entities[0] is registered, (
            "FV must receive the registry-returned Entity instance, " "not a synthetic Entity built from the FV name"
        )
        assert str(entities[0].name) == "USER_ID"

    def test_multi_entity_resolved_from_registry(self) -> None:
        """Multi-entity FV: each name resolved independently, order preserved."""
        user = _registered_entity("USER_ID")
        sess = _registered_entity("SESSION_ID")

        def _resolve(name: str) -> Entity:
            return {"USER_ID": user, "SESSION_ID": sess}[name]

        kwargs = self._build(
            _fv_payload(entity_names=["USER_ID", "SESSION_ID"]),
            get_entity_side_effect=_resolve,
        )
        entities = kwargs["entities"]
        assert len(entities) == 2
        assert entities[0] is user
        assert entities[1] is sess

    def test_missing_entity_raises_clear_error(self) -> None:
        """``fs.get_entity`` raising must propagate so apply fails loudly."""

        def _raise(_name: str) -> None:
            raise RuntimeError("Entity FOO_BAR not found.")

        with pytest.raises(RuntimeError, match="FOO_BAR"):
            self._build(_fv_payload(entity_names=["FOO_BAR"]), get_entity_side_effect=_raise)

    def test_entity_with_real_join_keys_preserved(self) -> None:
        """Real entity join_keys must round-trip through to the FV."""
        compound = _registered_entity(
            "TENANT_USER",
            join_keys=["TENANT_ID", "USER_ID"],
            desc="Multi-key entity",
        )
        kwargs = self._build(
            _fv_payload(entity_names=["TENANT_USER"]),
            get_entity_return_value=compound,
        )
        entities = kwargs["entities"]
        assert len(entities) == 1
        ent = entities[0]
        assert str(ent.name) == "TENANT_USER"
        join_key_strs = [str(jk) for jk in ent.join_keys]
        assert join_key_strs == [
            "TENANT_ID",
            "USER_ID",
        ], f"Entity join_keys must come from the registry, got {join_key_strs!r}"

    def test_no_synthetic_entity_name(self) -> None:
        """Explicit anti-regression guard: no entity name derived from the FV name."""
        registered = _registered_entity("USER_ID")
        payload = _fv_payload(name="USER_CLICK_STATS_DECL", entity_names=["USER_ID"])
        kwargs = self._build(payload, get_entity_return_value=registered)

        for ent in kwargs["entities"]:
            ent_name = str(ent.name)
            assert "_entity" not in ent_name.lower(), (
                f"Entity name {ent_name!r} looks like a synthetic placeholder "
                "(fv_name + '_entity').  Use fs.get_entity(name) to resolve "
                "the real registered entity instead."
            )
            assert ent_name != "USER_CLICK_STATS_DECL_ENTITY"

    def test_empty_entities_returns_zero_entities(self) -> None:
        """An FV with no declared entities must NOT touch the registry."""
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView") as fv_ctor:
            fv_ctor.return_value = MagicMock(name="feature_view_instance")
            _build_feature_view(
                _fv_payload(entity_names=[]),
                session,
                "JKEW_DB",
                "JKEW_SCHEMA",
                "JKEW_WAREHOUSE",
                fs=fs,
            )
            _, kwargs = fv_ctor.call_args
            assert kwargs["entities"] == [], "Empty entity list must produce empty FV.entities"
        fs.get_entity.assert_not_called()


_BUG_BASH_UDF_SOURCE = """
import pandas as pd

def compute_engagement_metrics(clickstream: pd.DataFrame) -> pd.DataFrame:
    df = clickstream.copy()
    weights = {"page_view": 1.0, "click": 2.0, "purchase": 10.0}
    df["ENGAGEMENT_SCORE"] = df["EVENT_TYPE"].map(weights).fillna(1.0)
    df["IS_CONVERSION"] = df["EVENT_TYPE"].isin(["purchase", "signup"])
    return df[["USER_ID", "TIMESTAMP", "IS_CONVERSION", "ENGAGEMENT_SCORE"]]
"""


_BUG_BASH_FEATURES: list[dict[str, Any]] = [
    {
        "function": "sum",
        "window_sec": 3600,
        "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
        "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
    },
    {
        "function": "max",
        "window_sec": 86400,
        "source_column": {"name": "IS_CONVERSION", "type": "BooleanType"},
        "output_column": {"name": "HAS_CONVERSION_24H", "type": "BooleanType"},
    },
]


def _streaming_fv_payload(
    name: str = "USER_CLICK_STATS_DECL",
    *,
    backfill_table: str | None = None,
    backfill_start_time: Any = None,
    udf_source: str | None = None,
    features: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a CREATE_FV payload for a streaming FV mirroring BUG_BASH §5.

    The default features list carries both BUG_BASH aggregations
    (sum + max) so the executor's tiled-FV branch is exercised
    end-to-end.  Pass features=[] to assert anti-regression for
    non-aggregated streaming FVs.

    Args:
        name: FV name; defaults to the BUG_BASH-canonical name.
        backfill_table: Optional FQN that the helper places under the
            FV-level ``backfill: { table: ... }`` block (drives the
            ``session.table`` resolution branch in
            ``_build_streaming_backfill_df``).
        backfill_start_time: Optional value for ``backfill.start_time``
            (string or ``datetime``).  Forwarded as
            ``StreamConfig.backfill_start_time`` by the executor.
        udf_source: Override the UDF Python source string.  Defaults to
            _BUG_BASH_UDF_SOURCE.
        features: Override the features list.  Defaults to the two
            BUG_BASH aggregations (sum + max).  Pass [] to assert
            anti-regression for non-aggregated streaming FVs.

    Returns:
        A dict-shaped CREATE_FV payload suitable for handing to
        _build_feature_view.
    """
    if udf_source is None:
        udf_source = _BUG_BASH_UDF_SOURCE
    if features is None:
        features = [dict(f) for f in _BUG_BASH_FEATURES]
    source: dict[str, Any] = {
        "name": "CLICKSTREAM_EVENTS",
        "source_type": "Stream",
        "columns": [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TYPE", "type": "StringType"},
            {"name": "TIMESTAMP", "type": "TimestampType"},
        ],
    }
    payload: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": name,
        "version": "V1",
        "entities": ["USER_ID"],
        "timestamp_col": "TIMESTAMP",
        "feature_granularity_sec": 300,
        "feature_aggregation_method": "tiles",
        "sources": [source],
        "udf": {
            "name": "compute_engagement_metrics",
            "engine": "pandas",
            "function_definition": udf_source,
            "output_columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
                {"name": "IS_CONVERSION", "type": "BooleanType"},
                {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
            ],
        },
        "features": features,
    }
    if backfill_table is not None or backfill_start_time is not None:
        backfill_block: dict[str, Any] = {}
        if backfill_table is not None:
            backfill_block["table"] = backfill_table
        if backfill_start_time is not None:
            backfill_block["start_time"] = backfill_start_time
        payload["backfill"] = backfill_block
    return payload


class TestBuildFeatureViewStreaming:
    """``_build_feature_view`` must construct a real ``StreamConfig`` for
    ``kind: StreamingFeatureView`` payloads instead of passing a stub
    ``feature_df`` through.  This is the surface that fixed BUG_BASH step 6's
    ``CREATE VIEW name () ...`` syntax error: when ``feature_df`` is
    unresolvable, snowml-core's schema introspection silently sets
    ``feature_descs = None`` and emits an empty column list.
    """

    def _patch_constructors(self) -> tuple[Any, Any]:
        """Capture ``FeatureView`` and ``StreamConfig`` kwargs for assertion."""
        fv_calls: list[dict[str, Any]] = []
        sc_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            fv_calls.append(kwargs)
            return type("_FakeFV", (), {**kwargs})()

        def _fake_sc(**kwargs: Any) -> Any:
            sc_calls.append(kwargs)
            return type("_FakeSC", (), {**kwargs})()

        return (fv_calls, _fake_fv), (sc_calls, _fake_sc)

    def test_streaming_fv_builds_stream_config(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (fv_calls, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()

        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(_streaming_fv_payload(), session, "DB", "SCH", "WH", fs=fs)

        assert len(sc_calls) == 1, "StreamConfig must be constructed exactly once for a streaming FV"
        assert sc_calls[0]["stream_source"] == "CLICKSTREAM_EVENTS"
        assert callable(sc_calls[0]["transformation_fn"])
        assert sc_calls[0]["backfill_df"] is not None

    def test_streaming_fv_does_not_pass_feature_df(self) -> None:
        """The streaming branch passes ``stream_config=`` to ``FeatureView``,
        not ``feature_df=``.  snowml-core treats the two as mutually exclusive
        for streaming registration.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (fv_calls, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(_streaming_fv_payload(), session, "DB", "SCH", "WH", fs=fs)

        assert len(fv_calls) == 1
        assert "stream_config" in fv_calls[0], "Streaming FV must be constructed with stream_config=…, not feature_df=…"
        assert "feature_df" not in fv_calls[0], (
            "feature_df must not appear alongside stream_config — snowml-core " "treats them as mutually exclusive."
        )

    def test_backfill_table_resolves_to_session_table(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        backfill_df_sentinel = MagicMock(name="backfill_df")
        session.table.return_value = backfill_df_sentinel

        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(backfill_table="JKEW_DB.JKEW_SCHEMA.HIST"),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        session.table.assert_any_call("JKEW_DB.JKEW_SCHEMA.HIST")
        assert sc_calls[0]["backfill_df"] is backfill_df_sentinel
        # Phase E E3 (A-bis) — the authored FQN must also flow into
        # ``StreamConfig.backfill_table`` so ``save_streaming_metadata``
        # persists it as ``StreamingMetadata.backfill_table`` and the
        # exporter can re-emit the operator's ``backfill.table:`` block
        # on the second plan.  Without this assertion the previous
        # ``backfill_df``-only check left A3's write side silently broken
        # in production (live-verify symptom: streaming-FV roundtrip
        # cannot reach NO_CHANGE on a clean replan).
        assert sc_calls[0].get("backfill_table") == "JKEW_DB.JKEW_SCHEMA.HIST"

    def test_no_backfill_table_synthesizes_typed_one_row_df(self) -> None:
        """Without ``backfill_table``, the executor falls back to
        ``session.create_dataframe([row], schema=StructType(...))`` so the
        streaming preamble's ``.limit(10).to_pandas()`` probe sees one row
        instead of failing on a missing PLACEHOLDER table.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        synthetic_df_sentinel = MagicMock(name="synthetic_backfill_df")
        session.create_dataframe.return_value = synthetic_df_sentinel

        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(backfill_table=None),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        session.create_dataframe.assert_called_once()
        rows_arg, _ = session.create_dataframe.call_args.args[:2]
        assert isinstance(rows_arg, list) and len(rows_arg) == 1, (
            "Synthesized backfill DF must carry exactly one row so the "
            "streaming preamble's .limit(10).to_pandas() probe is non-empty."
        )
        assert sc_calls[0]["backfill_df"] is synthetic_df_sentinel
        session.table.assert_not_called()
        # Phase E E3 — symmetric negative assertion: when the operator
        # does NOT author ``backfill.table:``, ``StreamConfig.backfill_table``
        # must stay absent (or ``None``) so ``StreamingMetadata.backfill_table``
        # round-trips as ``None`` — otherwise the exporter would synthesise
        # a phantom ``backfill.table:`` block on the second plan.
        assert sc_calls[0].get("backfill_table") in (None,)

    def test_backfill_start_time_iso_string_forwarded_as_datetime(self) -> None:
        """``backfill.start_time`` (ISO 8601 string in the authoring YAML) is
        coerced to a real ``datetime`` and forwarded as
        ``StreamConfig.backfill_start_time``."""
        import datetime as _dt

        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(
                    backfill_table="JKEW_DB.JKEW_SCHEMA.HIST",
                    backfill_start_time="2026-05-18T22:00:00",
                ),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        sc_kwargs = sc_calls[0]
        assert "backfill_start_time" in sc_kwargs, (
            "Executor must forward backfill.start_time to StreamConfig as "
            "backfill_start_time so the imperative streaming preamble can "
            "filter the historical table."
        )
        forwarded = sc_kwargs["backfill_start_time"]
        assert isinstance(forwarded, _dt.datetime), (
            "Authoring strings must be coerced to a datetime before reaching "
            f"StreamConfig (which types it as Optional[datetime]); got: {forwarded!r}"
        )
        assert forwarded == _dt.datetime(2026, 5, 18, 22, 0, 0)

    def test_backfill_start_time_datetime_passed_through(self) -> None:
        import datetime as _dt

        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        ts = _dt.datetime(2026, 5, 18, 22, 0, 0)
        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(
                    backfill_table="JKEW_DB.JKEW_SCHEMA.HIST",
                    backfill_start_time=ts,
                ),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        assert sc_calls[0]["backfill_start_time"] == ts

    def test_no_backfill_block_omits_start_time(self) -> None:
        """When the FV has no ``backfill:`` block, the executor must NOT
        synthesize a ``backfill_start_time`` (snowml-core ``StreamConfig``
        defaults it to ``None``)."""
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        sc_kwargs = sc_calls[0]
        # Either omitted or explicitly None — both are equivalent at the
        # imperative boundary.
        assert sc_kwargs.get("backfill_start_time") is None

    def test_udf_callable_compiled_from_function_definition(self) -> None:
        """The compiled ``transformation_fn`` is the bug-bash UDF — its
        ``__name__`` must match the spec's ``udf.name`` and
        ``inspect.getsource`` must succeed (precondition for snowml-core's
        AST guard).
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        fn = sc_calls[0]["transformation_fn"]
        assert fn.__name__ == "compute_engagement_metrics"
        import inspect

        source = inspect.getsource(fn)
        assert "def compute_engagement_metrics" in source, (
            "transformation_fn source must be inspect.getsource-able so "
            "StreamConfig.__post_init__'s AST import-guard can run."
        )

    def test_streaming_fv_does_not_defensively_register_stream_source(self) -> None:
        """Wave 1B contract update (``plans/stream_source_contract.md`` §4):
        the FV-side defensive ``register_stream_source`` call inside
        ``_build_stream_config`` is REMOVED.  With applied state populated
        by :func:`fetch_stream_source_rows`, the planner emits
        ``CREATE_SOURCE`` exactly once per genuinely-new source before any
        ``CREATE_FV`` op runs (topological-sort contract in
        :mod:`dependencies`).  Keeping the FV-side defensive register
        produced the second source of ``UserWarning: StreamSource <name>
        already exists. Skip registration.`` noise — that's the secondary
        fix the user requested.

        The previous version of this test asserted
        ``fs.register_stream_source.assert_called_once()``; it now asserts
        the opposite to lock in the removal.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        fs.register_stream_source.assert_not_called()

    def test_batch_fv_path_unchanged_anti_regression(self) -> None:
        """Non-streaming kinds (``BatchFeatureView``, ``RealtimeFeatureView``)
        keep the old ``feature_df=…`` path so this fix does not regress
        established codepaths.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (fv_calls, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        batch_payload = _fv_payload(name="BATCH_FV", entity_names=["USER_ID"])
        batch_payload["kind"] = "BatchFeatureView"

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(batch_payload, session, "DB", "SCH", "WH", fs=fs)

        assert len(fv_calls) == 1
        assert "feature_df" in fv_calls[0]
        assert "stream_config" not in fv_calls[0]
        assert sc_calls == [], "StreamConfig must NOT be constructed for non-streaming FVs"


class TestPhaseESourceRefsWritePath:
    """Phase E (A-bis) — ``_build_feature_view`` MUST stamp ``_source_refs``
    onto the ``FeatureView`` kwargs so ``register_feature_view``'s metadata
    write-gate (``if feature_view.source_refs:`` in feature_store.py) fires
    on declarative apply.

    Before Phase E the gate was forever false in production because no
    code in ``decl/`` stamped ``_source_refs`` — the comment claimed the
    "declarative authoring layer stamps source_refs from compile output"
    but no production path actually did.  Live-verify symptom: tiled
    offline BFVs like ``MY_ADV_BFV_DECL`` emitted ``RECREATE_FV`` on the
    second plan because applied-state recovery could not find the
    ``FV_SOURCE_REFS`` metadata row and fell back to the legacy DT-text
    shim, which produced ``sources: []``.

    These tests assert the contract end-to-end — they patch
    ``FeatureView`` to capture the kwargs ``_build_feature_view`` would
    pass into snowml-core's constructor, then verify ``_source_refs`` is
    present with the expected shape.  Without these tests the gap is
    silently masked by every other test in the suite that mocks
    ``feature_view_rows`` or sets ``source_refs`` directly on a
    fixture-built FV instance.
    """

    def _patch_constructors(self) -> tuple[Any, Any]:
        """Mirror :class:`TestBuildFeatureViewStreaming._patch_constructors`."""
        fv_calls: list[dict[str, Any]] = []
        sc_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            fv_calls.append(kwargs)
            # ``MagicMock(name=...)`` reserves ``name`` for the mock label;
            # the payload's ``name`` kwarg would collide.  Wrap a plain
            # MagicMock and attach the FV name as a post-construction
            # attribute so tests can introspect it without the TypeError.
            stub = MagicMock(name="FeatureView")
            stub.fv_name = kwargs.get("name", "")
            return stub

        def _fake_sc(**kwargs: Any) -> Any:
            sc_calls.append(kwargs)
            return MagicMock(name="StreamConfig")

        return (fv_calls, _fake_fv), (sc_calls, _fake_sc)

    def test_batch_fv_source_refs_stamped_from_payload_sources(self) -> None:
        """Plain ``BatchFeatureView`` with table-backed source: the authored
        ``sources[0]`` flows verbatim into ``kwargs["_source_refs"]``.
        """
        (fv_calls, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _fv_payload(name="EVENTS_BFV_DECL", table="RAW_EVENTS"),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        assert len(fv_calls) == 1
        assert "_source_refs" in fv_calls[0], (
            "Phase E E1 regression: _build_feature_view must stamp "
            "_source_refs onto FeatureView kwargs so register_feature_view "
            "persists MetadataType.FV_SOURCE_REFS"
        )
        source_refs = fv_calls[0]["_source_refs"]
        assert isinstance(source_refs, list) and len(source_refs) == 1
        assert source_refs[0]["name"] == "clicks"
        assert source_refs[0]["source_type"] == "Batch"
        assert source_refs[0]["table"] == "RAW_EVENTS"

    def test_advanced_tiled_offline_bfv_source_refs_stamped(self) -> None:
        """The exact shape that triggered the live AS2 invalidation on
        ``MY_ADV_BFV_DECL`` — tiled offline BFV with all six advanced
        fields.  The full payload is the live reproduction; the
        assertion confirms ``_source_refs`` flows even when every other
        kwarg (``cluster_by``, ``refresh_mode``, ``initialize``,
        ``aggregation_secondary_keys``) is present.
        """
        (fv_calls, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        payload = {
            "kind": "BatchFeatureView",
            "name": "MY_ADV_BFV_DECL",
            "version": "V1",
            "online": False,
            "entities": ["USER_ID"],
            "timestamp_col": "EVENT_TS",
            "feature_granularity_sec": 3600,
            "feature_aggregation_method": "tiles",
            "refresh_freq": "5 minutes",
            "warehouse": "JKEW_WH",
            "cluster_by": ["USER_ID"],
            "refresh_mode": "AUTO",
            "initialize": "ON_CREATE",
            "aggregation_secondary_keys": ["SESSION_ID"],
            "sources": [
                {
                    "name": "EVENTS_ADV_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_ADV_DECL",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "SESSION_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "AMOUNT", "type": "FloatType"},
                    ],
                },
            ],
            "features": [
                {
                    "source_column": {"name": "AMOUNT", "type": "FloatType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "FloatType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        }

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(payload, session, "JKEW_DB", "JKEW_SCHEMA", "JKEW_WH", fs=fs)

        assert len(fv_calls) == 1
        ctor = fv_calls[0]
        assert "_source_refs" in ctor
        source_refs = ctor["_source_refs"]
        assert isinstance(source_refs, list) and len(source_refs) == 1
        assert source_refs[0]["name"] == "EVENTS_ADV_DECL"
        assert source_refs[0]["table"] == "RAW_EVENTS_ADV_DECL"
        assert len(source_refs[0]["columns"]) == 4
        assert ctor["cluster_by"] == ["USER_ID"]
        assert ctor["refresh_mode"] == "AUTO"
        assert ctor["aggregation_secondary_keys"] == ["SESSION_ID"]

    def test_streaming_fv_source_refs_stamped(self) -> None:
        """Streaming FVs also flow through the same kwarg-collection block
        before the streaming-branch ``FeatureView(stream_config=...)``
        constructor — so ``_source_refs`` must be present on
        streaming-FV kwargs too.
        """
        (fv_calls, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(name="USER_CLICK_STATS_DECL"),
                session,
                "JKEW_DB",
                "JKEW_SCHEMA",
                "JKEW_WH",
                fs=fs,
            )

        assert len(fv_calls) == 1
        assert "_source_refs" in fv_calls[0]
        source_refs = fv_calls[0]["_source_refs"]
        assert isinstance(source_refs, list) and len(source_refs) == 1
        assert source_refs[0]["name"] == "CLICKSTREAM_EVENTS"
        assert source_refs[0]["source_type"] == "Stream"

    def test_empty_sources_does_not_stamp_source_refs(self) -> None:
        """When ``payload["sources"]`` is empty (defensive: the planner
        normally rejects this at validation time), ``_source_refs`` must
        NOT be stamped so the metadata write-gate stays false and
        ``register_feature_view`` skips the write.  Stamping an empty
        list would persist a row with ``sources=[]`` and break the
        legacy-fallback contract for FVs that pre-date the metadata
        row (Phase B4).
        """
        (fv_calls, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        payload = _fv_payload(name="NO_SOURCES_FV", table="X")
        payload["sources"] = []

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            try:
                _build_feature_view(payload, session, "DB", "SCH", "WH", fs=fs)
            except (ValueError, KeyError):
                # ``_build_feature_df`` raises ``ValueError`` when sources
                # is empty (no resolvable table/query); we only care that
                # the kwarg-collection block did not insert ``_source_refs``
                # before the raise.
                pass

        if fv_calls:
            assert "_source_refs" not in fv_calls[0]


class TestBuildFeatureViewBatchBackfill:
    """Batch FV ``backfill: { initialize: ON_CREATE | ON_SCHEDULE }`` maps to
    ``FeatureView(initialize=...)`` on the imperative constructor.

    ``backfill.overwrite`` does NOT flow through the FeatureView
    constructor — it lives at the ``FeatureStore.register_feature_view``
    boundary and is exercised in :class:`TestExecutePlanBatchBackfillOverwrite`.
    """

    def _patch_constructors(self) -> tuple[Any, Any]:
        fv_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            fv_calls.append(kwargs)
            return type("_FakeFV", (), {**kwargs})()

        return fv_calls, _fake_fv

    def test_batch_initialize_on_schedule_forwarded_to_feature_view_constructor(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        fv_calls, fake_fv = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        batch_payload = _fv_payload(name="BATCH_FV", entity_names=["USER_ID"])
        batch_payload["kind"] = "BatchFeatureView"
        batch_payload["backfill"] = {"initialize": "ON_SCHEDULE"}
        # Batch FVs need a refresh cadence for ON_SCHEDULE to be meaningful.
        batch_payload["refresh_freq"] = "1 hour"

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv):
            _build_feature_view(batch_payload, session, "DB", "SCH", "WH", fs=fs)

        assert len(fv_calls) == 1
        assert fv_calls[0].get("initialize") == "ON_SCHEDULE", (
            "Batch backfill.initialize must reach FeatureView(initialize=...) "
            "so the imperative library defers the first materialisation to "
            f"the next scheduled refresh; got: {fv_calls[0].get('initialize')!r}"
        )

    def test_batch_initialize_on_create_forwarded(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        fv_calls, fake_fv = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        batch_payload = _fv_payload(name="BATCH_FV", entity_names=["USER_ID"])
        batch_payload["kind"] = "BatchFeatureView"
        batch_payload["backfill"] = {"initialize": "ON_CREATE"}

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv):
            _build_feature_view(batch_payload, session, "DB", "SCH", "WH", fs=fs)

        assert fv_calls[0].get("initialize") == "ON_CREATE"

    def test_no_backfill_block_does_not_set_initialize(self) -> None:
        """When backfill is absent, the executor must not pass ``initialize``
        — the imperative library defaults to ``ON_CREATE`` and we want that
        default to apply unchanged for backward compatibility."""
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        fv_calls, fake_fv = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        batch_payload = _fv_payload(name="BATCH_FV", entity_names=["USER_ID"])
        batch_payload["kind"] = "BatchFeatureView"

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv):
            _build_feature_view(batch_payload, session, "DB", "SCH", "WH", fs=fs)

        assert "initialize" not in fv_calls[0], (
            "Without an explicit FV-level backfill.initialize, the executor "
            "must NOT pass ``initialize=`` so the imperative default applies."
        )


class TestExecutePlanBatchBackfillOverwrite:
    """``backfill.overwrite=True`` on a Batch FV maps to
    ``FeatureStore.register_feature_view(..., overwrite=True)``.

    This is the operator opt-in for the imperative "backfill cost" path —
    it forces a full re-materialization of the offline DT and online OFT.
    The destructive nature is gated by ``--allow-recreate`` at the planner
    level (covered in invariants/planner tests, Agent C).
    """

    def test_overwrite_true_forwarded_to_register_feature_view(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER_ID",
                    payload={
                        "kind": "Entity",
                        "name": "USER_ID",
                        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                        "description": "",
                    },
                ),
                PlanOp(
                    kind=OpKind.CREATE_FV,
                    name="ORDERS_TOTAL_DECL",
                    payload={
                        **_fv_payload(name="ORDERS_TOTAL_DECL", entity_names=["USER_ID"]),
                        "backfill": {"overwrite": True},
                    },
                ),
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        registered_names: set[str] = set()

        def _register_entity(entity: Any) -> Any:
            registered_names.add(str(entity.name))
            return entity

        def _get_entity(name: str) -> Entity:
            if name not in registered_names:
                raise RuntimeError(f"Entity {name} has not been registered.")
            return _registered_entity(name)

        register_calls: list[dict[str, Any]] = []

        def _register_fv(feature_view: Any, version: str, **kwargs: Any) -> Any:
            register_calls.append({"version": version, **kwargs})
            return feature_view

        fs = MagicMock(name="FeatureStore")
        fs.register_entity.side_effect = _register_entity
        fs.get_entity.side_effect = _get_entity
        fs.register_feature_view.side_effect = _register_fv

        with patch("snowflake.ml.feature_store.feature_store.FeatureStore", return_value=fs,), patch(
            "snowflake.ml.feature_store.feature_view.FeatureView",
            side_effect=lambda **kwargs: type("_FakeFV", (), {**kwargs})(),
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        assert len(register_calls) == 1
        assert register_calls[0].get("overwrite") is True, (
            "FV-level backfill.overwrite=True must reach "
            "FeatureStore.register_feature_view(overwrite=True). "
            f"Captured register_feature_view kwargs: {register_calls[0]!r}"
        )

    def test_overwrite_false_or_missing_does_not_force_overwrite(self) -> None:
        """Without an explicit ``backfill.overwrite=True``, the executor must
        not pass ``overwrite=True`` to ``register_feature_view`` — that
        boundary remains driven by ``PlanOptions.overwrite`` for the legacy
        (non-backfill) opt-in path."""
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER_ID",
                    payload={
                        "kind": "Entity",
                        "name": "USER_ID",
                        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                        "description": "",
                    },
                ),
                PlanOp(
                    kind=OpKind.CREATE_FV,
                    name="ORDERS_TOTAL_DECL",
                    payload=_fv_payload(name="ORDERS_TOTAL_DECL", entity_names=["USER_ID"]),
                ),
            ],
            warnings=[],
        )
        session = _mock_fv_session()
        registered_names: set[str] = set()

        def _register_entity(entity: Any) -> Any:
            registered_names.add(str(entity.name))
            return entity

        def _get_entity(name: str) -> Entity:
            if name not in registered_names:
                raise RuntimeError(f"Entity {name} has not been registered.")
            return _registered_entity(name)

        register_calls: list[dict[str, Any]] = []

        def _register_fv(feature_view: Any, version: str, **kwargs: Any) -> Any:
            register_calls.append({"version": version, **kwargs})
            return feature_view

        fs = MagicMock(name="FeatureStore")
        fs.register_entity.side_effect = _register_entity
        fs.get_entity.side_effect = _get_entity
        fs.register_feature_view.side_effect = _register_fv

        with patch("snowflake.ml.feature_store.feature_store.FeatureStore", return_value=fs,), patch(
            "snowflake.ml.feature_store.feature_view.FeatureView",
            side_effect=lambda **kwargs: type("_FakeFV", (), {**kwargs})(),
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        assert len(register_calls) == 1
        # PlanOptions(overwrite=False) is the default, so the boundary call
        # should reflect that — backfill.overwrite is None, so nothing forces
        # overwrite=True.
        assert register_calls[0].get("overwrite") is False


class TestBuildFeatureViewStreamingAggregation:
    """Pin the streaming-FV aggregation pass-through (BUG_BASH step 7).

    Root cause: the streaming branch of
    :func:`imperative_executor._build_feature_view` constructs the
    imperative ``FeatureView`` without forwarding the payload's
    ``features`` / ``feature_granularity_sec`` /
    ``feature_aggregation_method``, so snowml-core's
    ``_aggregation_specs`` stays ``None`` and ``is_tiled`` never fires
    and the FV registers with the UDF output columns instead of the
    aggregation outputs.

    These tests pin the contract on the executor's input -> constructor
    kwargs map.  They patch ``FeatureView`` / ``StreamConfig`` with
    fakes that capture kwargs verbatim so we can assert without
    invoking snowml-core's heavy registration code.
    """

    def _patch_constructors(self) -> tuple[Any, Any]:
        """Capture FeatureView and StreamConfig kwargs for assertion."""
        fv_calls: list[dict[str, Any]] = []
        sc_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            fv_calls.append(kwargs)
            return type("_FakeFV", (), {**kwargs})()

        def _fake_sc(**kwargs: Any) -> Any:
            sc_calls.append(kwargs)
            return type("_FakeSC", (), {**kwargs})()

        return (fv_calls, _fake_fv), (sc_calls, _fake_sc)

    def _run_build(self, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Drive _build_feature_view against payload and capture kwargs."""
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (fv_calls, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(payload, session, "DB", "SCH", "WH", fs=fs)

        return fv_calls, sc_calls

    def test_streaming_fv_passes_features_to_constructor(self) -> None:
        """The streaming branch must forward features=[Feature, Feature]
        to FeatureView so snowml-core builds _aggregation_specs and the
        tiled-FV branch fires.  Without this, _feature_desc falls back
        to the raw UDF output columns.
        """
        from snowflake.ml.feature_store.feature import Feature

        fv_calls, _ = self._run_build(_streaming_fv_payload())
        assert len(fv_calls) == 1
        features = fv_calls[0].get("features")
        assert isinstance(features, list)
        assert len(features) == 2
        assert all(isinstance(f, Feature) for f in features)

    def test_streaming_fv_passes_feature_granularity_to_constructor(self) -> None:
        """feature_granularity_sec=300 must surface as
        feature_granularity="300s" on the FeatureView constructor — that
        is the trigger for snowml-core's is_tiled branch.
        """
        fv_calls, _ = self._run_build(_streaming_fv_payload())
        assert fv_calls[0].get("feature_granularity") == "300s"

    def test_streaming_fv_passes_feature_aggregation_method_tiles(self) -> None:
        """feature_aggregation_method="tiles" must reach the constructor
        as the imperative FeatureAggregationMethod.TILES enum (not the
        raw string).  Without the enum coercion, snowml-core's
        isinstance check fails and aggregation specs are dropped.
        """
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        fv_calls, _ = self._run_build(_streaming_fv_payload())
        assert fv_calls[0].get("feature_aggregation_method") is FeatureAggregationMethod.TILES

    def test_streaming_fv_window_sec_3600_becomes_3600s(self) -> None:
        """window_sec=3600 integer must convert to the "3600s" string the
        imperative Feature constructor accepts via interval_to_seconds.
        Pinning the seconds-suffix convention guards against silent unit
        drift between the planner payload and the imperative builder.
        """
        fv_calls, _ = self._run_build(_streaming_fv_payload())
        features = fv_calls[0].get("features", [])
        assert len(features) == 2
        assert features[0]._window == "3600s"
        assert features[1]._window == "86400s"

    def test_streaming_fv_function_string_to_aggregation_type(self) -> None:
        """function="sum" / "max" must become AggregationType.SUM /
        AggregationType.MAX (case-insensitive lookup via the enum value).
        The imperative Feature requires the enum, not the bare string.
        """
        from snowflake.ml.feature_store.aggregation import AggregationType

        fv_calls, _ = self._run_build(_streaming_fv_payload())
        features = fv_calls[0].get("features", [])
        assert features[0]._function is AggregationType.SUM
        assert features[1]._function is AggregationType.MAX

    def test_streaming_fv_alias_set_from_output_column(self) -> None:
        """The payload's output_column.name must surface as the imperative
        Feature._alias so the deployed FV's _feature_desc is keyed off
        the windowed-aggregation outputs (TOTAL_ENGAGEMENT_1H /
        HAS_CONVERSION_24H) instead of the UDF's raw outputs.
        Feature.alias() uppercases by default, and the BUG_BASH names
        are already upper-case so we expect verbatim equality.
        """
        fv_calls, _ = self._run_build(_streaming_fv_payload())
        features = fv_calls[0].get("features", [])
        assert features[0]._alias == "TOTAL_ENGAGEMENT_1H"
        assert features[1]._alias == "HAS_CONVERSION_24H"

    def test_streaming_fv_no_features_when_payload_empty(self) -> None:
        """Anti-regression: a streaming FV without aggregation features
        (features=[]) must NOT pass features= to FeatureView, so
        non-aggregated streaming FVs stay on the existing
        stream_config-only path.  Likewise, feature_granularity and
        feature_aggregation_method should be omitted unless the payload
        truly carries aggregations — the imperative API treats them as
        a triple.
        """
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        payload = _streaming_fv_payload(features=[])
        payload.pop("feature_granularity_sec", None)
        payload.pop("feature_aggregation_method", None)

        fv_calls, _ = self._run_build(payload)
        assert "features" not in fv_calls[0], (
            "Empty features list must not become an empty kwarg — "
            "snowml-core treats features=None and features=[] as "
            "non-aggregated, but passing the kwarg explicitly couples "
            "non-aggregated streaming FVs to a regression in the "
            "is_tiled branch."
        )
        assert "feature_granularity" not in fv_calls[0]
        assert fv_calls[0].get("feature_aggregation_method") is not FeatureAggregationMethod.TILES

    def test_streaming_fv_offset_sec_threaded_when_present(self) -> None:
        """offset_sec=60 on a feature dict must thread through to
        Feature._offset = "60s".  Mirrors the window_sec -> "<n>s"
        convention.
        """
        offset_features = [
            {
                "function": "sum",
                "window_sec": 3600,
                "offset_sec": 60,
                "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
            },
        ]
        fv_calls, _ = self._run_build(_streaming_fv_payload(features=offset_features))
        features = fv_calls[0].get("features", [])
        assert len(features) == 1
        assert features[0]._offset == "60s"

    def test_streaming_fv_function_params_threaded_for_last_n(self) -> None:
        """function_params (e.g. {"n": 10} for last_n) must thread through
        to the imperative Feature.__init__ as **params, so snowml-core's
        aggregation builder receives the parameters it needs to
        materialize the SQL — without this, last_n / approx_percentile
        features silently drop their parameters.
        """
        from snowflake.ml.feature_store.aggregation import AggregationType

        last_n_features = [
            {
                "function": "last_n",
                "window_sec": 3600,
                "function_params": {"n": 10},
                "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                "output_column": {"name": "RECENT_SCORES_1H", "type": "ArrayType"},
            },
        ]
        fv_calls, _ = self._run_build(_streaming_fv_payload(features=last_n_features))
        features = fv_calls[0].get("features", [])
        assert len(features) == 1
        assert features[0]._function is AggregationType.LAST_N
        assert features[0]._params == {"n": 10}

    def test_tiled_streaming_fv_no_longer_defaults_refresh_freq_to_granularity(self) -> None:
        """After the ``feature_granularity`` / ``refresh_freq`` /
        ``target_lag`` decoupling, tiled streaming FVs no longer get a
        synthetic ``refresh_freq`` from the granularity.  ``refresh_freq``
        comes exclusively from the authored ``refresh_freq`` — the
        validator side surfaces a friendlier error when it is missing.

        snowml-core's ``FeatureView`` constructor still raises
        ``refresh_freq is required for tile-based aggregations`` when
        the payload omits ``refresh_freq`` (and the test's payload
        does).  The executor must NOT inject a synthetic refresh_freq
        from the granularity to mask that error.
        """
        fv_calls, _ = self._run_build(_streaming_fv_payload())
        # ``refresh_freq`` is absent from the kwargs the executor
        # passed to ``FeatureView(...)`` — the imperative constructor
        # surfaces the error on its own.
        assert "refresh_freq" not in (fv_calls[0] if fv_calls else {})

    def test_tiled_streaming_fv_ignores_target_lag_sec_in_payload(self) -> None:
        """Streaming FVs always run at 0 seconds target lag — the runtime
        enforces this and stamps ``target_lag_sec: 0`` onto the deployed
        SPECIFICATION regardless of the authored value.  The Pydantic
        validator rejects ``target_lag_sec`` on the Python / loader path
        (:meth:`FeatureView._reject_target_lag_on_stream_or_realtime`),
        and the spec compiler strips the key from streaming compiled
        specs, but a hand-built payload that bypasses both layers must
        still be tolerated — the executor's streaming branch ignores
        any injected ``target_lag_sec``.  After the decoupling, no
        granularity-based fallback fires either; ``refresh_freq``
        remains absent.
        """
        payload = _streaming_fv_payload()
        payload["target_lag_sec"] = 600
        fv_calls, _ = self._run_build(payload)
        # No refresh_freq leaks through — neither from the injected
        # ``target_lag_sec`` (OFT-only contract) nor from the
        # granularity (no synthetic default after decoupling).
        assert "refresh_freq" not in (fv_calls[0] if fv_calls else {})

    def test_tiled_streaming_fv_drops_smuggled_refresh_freq(self) -> None:
        """Even when a hand-built payload smuggles ``refresh_freq`` past
        the spec validator (which now rejects authoring ``refresh_freq``
        on streaming kinds), the executor's streaming branch must not
        forward the kwarg to ``FeatureView(refresh_freq=...)``.  The
        Snowflake runtime stamps ``target_lag_sec=0`` regardless of any
        cadence string, so an authored value would have no effect.

        Defence-in-depth pin for the rename's kind-aware forwarding gate
        (``payload["kind"] == "BatchFeatureView"`` in
        ``_build_feature_view``).
        """
        payload = _streaming_fv_payload()
        payload["refresh_freq"] = "USING CRON 0 * * * * UTC"
        fv_calls, _ = self._run_build(payload)
        assert "refresh_freq" not in fv_calls[0], (
            "Streaming FV branch must drop refresh_freq; got " f"refresh_freq={fv_calls[0].get('refresh_freq')!r}."
        )

    def test_tiled_streaming_fv_drops_target_lag_sec_and_refresh_freq(self) -> None:
        """Both ``target_lag_sec`` and ``refresh_freq`` are dropped on the
        streaming branch — the runtime always stamps target_lag_sec=0
        and the validator rejects refresh_freq.
        """
        payload = _streaming_fv_payload()
        payload["target_lag_sec"] = 600
        payload["refresh_freq"] = "USING CRON 0 * * * * UTC"
        fv_calls, _ = self._run_build(payload)
        assert "refresh_freq" not in fv_calls[0]

    def test_non_tiled_streaming_fv_keeps_no_refresh_freq_default(self) -> None:
        """Non-aggregated streaming FVs (no features) must NOT receive the
        granularity-based refresh_freq default — that default exists only
        to satisfy snowml-core's tile-based-aggregation validation, and
        forcing it onto streaming FVs that opted out of tiles would couple
        the stream_config-only path to unrelated Dynamic-Table semantics.
        """
        payload = _streaming_fv_payload(features=[])
        payload.pop("feature_granularity_sec", None)
        payload.pop("feature_aggregation_method", None)
        fv_calls, _ = self._run_build(payload)
        assert "refresh_freq" not in fv_calls[0]

    def test_continuous_streaming_fv_compiler_default_threads_granularity(self) -> None:
        """End-to-end of the CONTINUOUS-granularity default: when the
        compiler stamps ``feature_granularity_sec=60`` for a streaming
        FV authored as ``feature_aggregation_method=continuous`` without
        an explicit granularity, the imperative builder must:

        * forward ``feature_granularity="60s"`` to ``FeatureView(...)``
          so snowml-core's ``_resolve_tiled_config`` accepts the
          continuous aggregation without re-deriving the default
          (matching the deployed SPECIFICATION JSON that the runtime
          post-defaults to ``60``);
        * forward ``feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS``
          (not the raw string) so the isinstance check on the core side
          fires;
        * still leave ``refresh_freq`` **absent** unless the payload
          also carries ``refresh_freq`` — the decoupled-refresh
          contract is unchanged and ``STREAM_FV_TILING_REFRESH`` /
          the imperative ``refresh_freq is required for tile-based
          aggregations.`` error remain the surfacing layers when the
          author omits ``refresh_freq``.
        """
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        payload = _streaming_fv_payload()
        # Simulate what spec_compiler emits post-default for an
        # authoring spec that had no explicit ``feature_granularity``
        # but did declare ``feature_aggregation_method: continuous``.
        payload["feature_granularity_sec"] = 60
        payload["feature_aggregation_method"] = "continuous"
        # Author deliberately did NOT supply a refresh_freq — the
        # decoupled-refresh contract says the executor must NOT
        # synthesise refresh_freq from feature_granularity.
        payload.pop("refresh_freq", None)

        fv_calls, _ = self._run_build(payload)
        assert fv_calls[0].get("feature_granularity") == "60s", (
            "Compiler-defaulted granularity must surface as "
            f'feature_granularity="60s"; got {fv_calls[0].get("feature_granularity")!r}.'
        )
        assert (
            fv_calls[0].get("feature_aggregation_method") is FeatureAggregationMethod.CONTINUOUS
        ), "feature_aggregation_method=continuous must coerce to the CONTINUOUS enum."
        assert "refresh_freq" not in fv_calls[0], (
            "Decoupled-refresh contract: a continuous SFV without refresh_freq "
            "must not pick up a synthetic refresh_freq from the granularity default."
        )


class TestBuildFeatureDfFailsLoudlyWithoutSource:
    """Anti-regression: the placeholder-fallback in ``_build_feature_df``
    that emitted ``SELECT cols FROM db.schema.PLACEHOLDER WHERE 1=0`` is
    gone.  A non-streaming FV with no resolvable ``table:`` or ``query:``
    must now raise a clear error instead of silently corrupting the
    downstream ``CREATE VIEW`` DDL.
    """

    def test_non_streaming_fv_with_no_source_raises(self) -> None:
        session = _mock_fv_session()
        payload = {
            "kind": "BatchFeatureView",
            "name": "MISCONFIGURED",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [{"name": "X", "source_type": "Batch"}],  # no table, no query
            "features": [],
        }

        with pytest.raises(ValueError, match="MISCONFIGURED"):
            _build_feature_df(payload, session, "DB", "SCH")


class TestBuildFeatureDfQueryPath:
    """``BatchSource.query`` is a first-class authoring shape: a payload with
    ``sources: [{name, source_type: Batch, query: <SQL>}]`` (no table)
    must route through ``session.sql(query)`` so the resulting DataFrame
    feeds the imperative ``CREATE DYNAMIC TABLE … AS <query>`` body. The
    behaviour itself was already wired (Phase 1's ``SourceRef.query``
    plumbing); these tests pin the contract so a future refactor cannot
    silently drop the path.
    """

    def test_query_path_calls_session_sql(self) -> None:
        session = _mock_fv_session()
        sentinel = MagicMock(name="snowpark_dataframe_query")
        session.sql.return_value = sentinel
        payload = {
            "kind": "BatchFeatureView",
            "name": "FV_QUERY",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_QRY",
                    "source_type": "Batch",
                    "query": "SELECT user_id, event_ts FROM RAW.EVENTS WHERE ts > '2024-01-01'",
                }
            ],
            "features": [],
        }
        df = _build_feature_df(payload, session, "DB", "SCH")
        session.sql.assert_called_once_with("SELECT user_id, event_ts FROM RAW.EVENTS WHERE ts > '2024-01-01'")
        session.table.assert_not_called()
        assert df is sentinel

    def test_query_path_with_join_passes_through_unchanged(self) -> None:
        # The executor must not parse / rewrite the SQL — whitespace
        # normalization is a compile-time concern (Phase 3); the
        # executor passes whatever it receives to session.sql verbatim.
        session = _mock_fv_session()
        body = "SELECT a.user_id, a.event_ts, b.kind " "FROM RAW.EVENTS a JOIN RAW.EVENT_KINDS b ON a.kind_id = b.id"
        payload = {
            "kind": "BatchFeatureView",
            "name": "FV_JOIN",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [{"name": "JOINED_EVENTS", "source_type": "Batch", "query": body}],
            "features": [],
        }
        _build_feature_df(payload, session, "DB", "SCH")
        session.sql.assert_called_once_with(body)
        session.table.assert_not_called()

    def test_table_takes_precedence_over_query_when_both_present(self) -> None:
        # Defensive: if a payload accidentally carries BOTH table and
        # query (e.g. a misbehaving compiler upstream), table wins.
        # This pins the existing iteration order in _build_feature_df
        # so a future refactor doesn't silently flip semantics.
        session = _mock_fv_session()
        payload = {
            "kind": "BatchFeatureView",
            "name": "FV_BOTH",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS",
                    "query": "SELECT * FROM RAW.OTHER",
                }
            ],
            "features": [],
        }
        _build_feature_df(payload, session, "DB", "SCH")
        session.table.assert_called_once_with("DB.SCH.RAW_EVENTS")
        session.sql.assert_not_called()

    def test_query_with_source_database_and_schema_does_not_qualify(self) -> None:
        # source_database / source_schema are TABLE qualifiers; for
        # a query-backed source they MUST be ignored — any qualification
        # the user wants belongs in the SQL itself.
        session = _mock_fv_session()
        payload = {
            "kind": "BatchFeatureView",
            "name": "FV_QRY_QUALIFIED",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_QRY",
                    "source_type": "Batch",
                    "query": "SELECT * FROM RAW.EVENTS",
                    "source_database": "OTHER_DB",
                    "source_schema": "OTHER_SCH",
                }
            ],
            "features": [],
        }
        _build_feature_df(payload, session, "DB", "SCH")
        # The executor passes the SQL verbatim; the source_database /
        # source_schema fields are ignored for query-backed sources.
        session.sql.assert_called_once_with("SELECT * FROM RAW.EVENTS")


class TestRegisterStreamSource:
    """``CREATE_SOURCE`` for ``StreamingSource`` must call
    ``FeatureStore.register_stream_source`` so streaming feature views
    can be registered downstream — they require the source to exist in
    the SnowML metadata table before ``register_feature_view`` runs the
    streaming preamble.  ``BatchSource`` remains a virtual no-op (the
    imperative API has no equivalent registration).
    """

    def _streaming_source_payload(
        self,
        name: str = "CLICKSTREAM_EVENTS",
        columns: list[dict[str, Any]] | None = None,
        desc: str = "",
    ) -> dict[str, Any]:
        if columns is None:
            columns = [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TYPE", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
            ]
        return {
            "kind": "StreamingSource",
            "name": name,
            "type": "REST",
            "columns": columns,
            "description": desc,
        }

    def test_create_source_streaming_calls_register_stream_source(self) -> None:
        """Spec-declared ``StreamingSource`` lands as a real
        ``fs.register_stream_source(StreamSource(...))`` call so the
        downstream FV's ``StreamConfig.stream_source`` lookup succeeds.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_SOURCE,
                    name="CLICKSTREAM_EVENTS",
                    payload=self._streaming_source_payload(desc="Real-time click events"),
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.register_stream_source.return_value = MagicMock(name="StreamSource")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.register_stream_source.assert_called_once()
        registered_arg = fs.register_stream_source.call_args.args[0]
        assert str(registered_arg.name) == "CLICKSTREAM_EVENTS"
        assert registered_arg.desc == "Real-time click events"
        # Schema mirrors the spec's columns (3 fields)
        assert len(registered_arg.schema.fields) == 3
        # Op status is success — no longer "skipped"
        assert result.ops[0]["operation"] == "CREATE_SOURCE"
        assert result.ops[0]["status"] == "success", (
            "Streaming CREATE_SOURCE must execute, not be silently skipped — "
            "register_feature_view's streaming preamble depends on the source "
            "being registered first."
        )

    def test_create_source_batch_remains_no_op(self) -> None:
        """``BatchSource`` must remain virtual — the imperative API has
        no ``register_batch_source`` analogue, so that op still reports
        ``status: skipped`` for backward compatibility with the v0
        contract.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_SOURCE,
                    name="ORDER_HISTORY",
                    payload={
                        "kind": "BatchSource",
                        "name": "ORDER_HISTORY",
                        "table": "ORDERS",
                    },
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.register_stream_source.assert_not_called()
        assert result.ops[0]["status"] == "skipped"

    def test_create_source_streaming_idempotent_on_already_registered(self) -> None:
        """``register_stream_source`` is idempotent in snowml-core —
        already-registered sources just emit a UserWarning and return
        the existing object.  The executor must let that pass through
        as a successful op (no exception).
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_SOURCE,
                    name="CLICKSTREAM_EVENTS",
                    payload=self._streaming_source_payload(),
                )
            ],
            warnings=[],
        )

        import warnings as _warnings

        session = _mock_fv_session()

        def _idempotent_register(stream_source: Any) -> Any:
            _warnings.warn(
                f"StreamSource {stream_source.name} already exists. Skip registration.",
                stacklevel=2,
                category=UserWarning,
            )
            return stream_source

        fs = MagicMock(name="FeatureStore")
        fs.register_stream_source.side_effect = _idempotent_register

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        assert result.ops[0]["status"] == "success"


class TestExecutePlanResolvesEntities:
    """End-to-end ``execute_plan`` integration: CREATE_ENTITY + CREATE_FV.

    Pins that ``register_feature_view`` receives an ``FV`` whose
    ``entities`` carry the spec-declared name (``USER_ID``), not a
    synthetic ``f"{fv_name}_entity"`` placeholder.  This is the
    regression that surfaced in ``scripts/verify_bug_bash.sh`` step 6
    as ``(2101) Entity USER_CLICK_STATS_DECL_ENTITY has not been registered``.

    Mirrors the planner's topological order (Entity before FV).
    CREATE_ENTITY in the executor now delegates to
    ``fs.register_entity`` (no raw entity-tag SQL remains in this
    module), so the fake registry tracks ``register_entity`` calls
    via ``fs.register_entity.side_effect`` to populate
    ``registered_names``; by the time the FV op invokes
    ``fs.get_entity("USER_ID")`` the name is already in the set,
    mirroring the live wire-up.
    """

    def test_create_fv_uses_spec_declared_entity_name(self) -> None:
        """Tracked registry: register_entity populates names; FV ends up with 'USER_ID'."""
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER_ID",
                    payload={
                        "kind": "Entity",
                        "name": "USER_ID",
                        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                        "description": "Click users",
                    },
                ),
                PlanOp(
                    kind=OpKind.CREATE_FV,
                    name="USER_CLICK_STATS_DECL",
                    payload=_fv_payload(name="USER_CLICK_STATS_DECL", entity_names=["USER_ID"]),
                ),
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        registered_names: set[str] = set()

        def _register_entity(entity: Any) -> Any:
            registered_names.add(str(entity.name))
            return entity

        def _get_entity(name: str) -> Entity:
            if name not in registered_names:
                raise RuntimeError(f"Entity {name} has not been registered.")
            return _registered_entity(name, join_keys=[name], desc="Click users")

        register_fv_calls: list[Any] = []

        def _register_fv(feature_view: Any, _version: str, **_kwargs: Any) -> Any:
            # Mirror the real ``register_feature_view`` validation: every
            # entity name on the FV must be in the registered set.
            for ent in feature_view.entities:
                if str(ent.name) not in registered_names:
                    raise RuntimeError(f"Entity {ent.name} has not been registered.")
            register_fv_calls.append(feature_view)
            return feature_view

        fs = MagicMock(name="FeatureStore")
        fs.register_entity.side_effect = _register_entity
        fs.get_entity.side_effect = _get_entity
        fs.register_feature_view.side_effect = _register_fv

        with patch("snowflake.ml.feature_store.feature_store.FeatureStore", return_value=fs,), patch(
            "snowflake.ml.feature_store.feature_view.FeatureView",
            # Capture the constructor kwargs as instance attrs so the
            # fake ``register_feature_view`` can read ``fv.entities``.
            side_effect=lambda **kwargs: type("_FakeFV", (), {**kwargs})(),
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        assert "USER_ID" in registered_names, (
            "CREATE_ENTITY op must execute and add USER_ID to the registry "
            "before CREATE_FV runs (planner topological-sort contract)."
        )
        assert len(register_fv_calls) == 1, "register_feature_view must be called exactly once"
        fv = register_fv_calls[0]
        entity_names_on_fv = [str(e.name) for e in fv.entities]
        assert entity_names_on_fv == ["USER_ID"], (
            f"FV.entities must carry the spec-declared name 'USER_ID', got {entity_names_on_fv!r}.  "
            "Regression: imperative_executor._build_feature_view fabricated a synthetic "
            "'_entity'-suffixed name that register_feature_view's _validate_entity_exists "
            "could not match against the registered tag."
        )
        assert all(
            "_entity" not in n.lower() for n in entity_names_on_fv
        ), "No synthetic '_entity'-suffixed names allowed (anti-regression guard)."

    def test_streaming_fv_apply_propagates_aggregations_end_to_end(self) -> None:
        """End-to-end: CREATE_ENTITY + CREATE_FV with the BUG_BASH §5
        streaming aggregation payload must register a FeatureView whose
        constructor kwargs carry the full aggregation triple.

        This is the integration counterpart to
        ``TestBuildFeatureViewStreamingAggregation`` — it drives
        ``execute_plan`` (not just ``_build_feature_view``) so the
        topological order CREATE_ENTITY -> CREATE_FV runs through the
        same dispatcher used in production, including
        :func:`_execute_create_entity` and :func:`_build_stream_config`.
        Failure here means the bug surfaced in BUG_BASH step 7 (raw UDF
        outputs in OFT spec) would re-emerge on any future refactor of
        execute_plan even if the unit-level tests still pass.
        """
        from snowflake.ml.feature_store.aggregation import AggregationType
        from snowflake.ml.feature_store.feature import Feature
        from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod

        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER_ID",
                    payload={
                        "kind": "Entity",
                        "name": "USER_ID",
                        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                        "description": "Click users",
                    },
                ),
                PlanOp(
                    kind=OpKind.CREATE_FV,
                    name="USER_CLICK_STATS_DECL",
                    payload=_streaming_fv_payload(),
                ),
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        session.create_dataframe.return_value = MagicMock(name="synthetic_backfill_df")
        registered_names: set[str] = set()

        def _register_entity(entity: Any) -> Any:
            registered_names.add(str(entity.name))
            return entity

        def _get_entity(name: str) -> Entity:
            if name not in registered_names:
                raise RuntimeError(f"Entity {name} has not been registered.")
            return _registered_entity(name, join_keys=[name], desc="Click users")

        register_fv_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            # Capture kwargs verbatim so the assertions below can inspect
            # the aggregation triple — same pattern as the unit-level
            # TestBuildFeatureViewStreamingAggregation tests but going
            # through execute_plan's full dispatcher.
            register_fv_calls.append(kwargs)
            return type("_FakeFV", (), {**kwargs})()

        def _register_fv(feature_view: Any, _version: str, **_kwargs: Any) -> Any:
            for ent in feature_view.entities:
                if str(ent.name) not in registered_names:
                    raise RuntimeError(f"Entity {ent.name} has not been registered.")
            return feature_view

        fs = MagicMock(name="FeatureStore")
        fs.register_entity.side_effect = _register_entity
        fs.get_entity.side_effect = _get_entity
        fs.register_feature_view.side_effect = _register_fv

        with patch("snowflake.ml.feature_store.feature_store.FeatureStore", return_value=fs,), patch(
            "snowflake.ml.feature_store.feature_view.FeatureView",
            side_effect=_fake_fv,
        ), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig",
            side_effect=lambda **kwargs: type("_FakeSC", (), {**kwargs})(),
        ):
            execute_plan(plan, session, "JKEW_DB", "JKEW_SCHEMA", "WH", PlanOptions())

        assert "USER_ID" in registered_names, "CREATE_ENTITY must run before CREATE_FV"
        assert len(register_fv_calls) == 1, "register_feature_view must be called exactly once"
        fv_kwargs = register_fv_calls[0]

        assert fv_kwargs.get("feature_aggregation_method") is FeatureAggregationMethod.TILES, (
            "Streaming FV apply must forward feature_aggregation_method=TILES "
            "so snowml-core's is_tiled branch fires and _feature_desc is keyed "
            "off the windowed-aggregation outputs (BUG_BASH step 7)."
        )
        assert fv_kwargs.get("feature_granularity") == "300s", (
            "Streaming FV apply must forward feature_granularity='300s' "
            "(seconds-suffix convention; matches DESCRIBE TYPE = SPECIFICATION output)."
        )

        features = fv_kwargs.get("features")
        assert isinstance(features, list)
        assert len(features) == 2, (
            "Both BUG_BASH §5 aggregations (TOTAL_ENGAGEMENT_1H/sum, "
            "HAS_CONVERSION_24H/max) must reach the FeatureView constructor."
        )
        assert all(isinstance(f, Feature) for f in features)
        assert features[0]._function is AggregationType.SUM
        assert features[0]._window == "3600s"
        assert features[0]._alias == "TOTAL_ENGAGEMENT_1H"
        assert features[1]._function is AggregationType.MAX
        assert features[1]._window == "86400s"
        assert features[1]._alias == "HAS_CONVERSION_24H"


class TestExecutePlanAllowRecreateGate:
    """Apply-time enforcement of ``PlanOptions.allow_recreate``.

    The planner is policy-free: it always emits ``RECREATE_FV`` /
    ``DROP_FV`` / ``DROP_ENTITY`` for destructive diffs so operators
    can see them in the ``snow feature plan`` output.  The
    ``allow_recreate`` flag is the operator opt-in that gates *apply*,
    not plan generation.

    These tests pin the gate's location at
    :func:`execute_plan` (decl-library boundary), so the standalone
    declarative wheel enforces the policy regardless of which CLI
    consumes it.  Refusing here (before any FV side-effect) keeps
    the plan file unrenamed under the L5 invariant
    (Mark-Failed-Stays-Unapplied) so a follow-up
    ``snow feature apply --allow-recreate`` consumes the same plan.

    Bug context: BUG_BASH §14 plain ``apply`` was silently executing
    the destructive ``RECREATE_FV`` and renaming the plan file to
    ``.applied``; the subsequent ``apply --allow-recreate`` then
    discovered no unapplied plan and reported ``Status: no_plan``.
    """

    def test_execute_plan_refuses_when_destructive_op_and_not_allow_recreate(self) -> None:
        """``RECREATE_FV`` (destructive) + ``allow_recreate=False`` → refused.

        Asserts:

        - ``result.status == "refused"``.
        - ``FeatureStore`` is NEVER constructed (so the destructive
          ``delete_feature_view`` / ``register_feature_view`` calls
          cannot happen).
        - ``errors`` mentions ``--allow-recreate`` so the operator
          knows the remediation.
        - The op renders with ``status="refused"`` for downstream
          display.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.RECREATE_FV,
                    name="USER_CLICK_STATS_DECL",
                    destructive=True,
                    reason="Full-spec change detected (UDF source).",
                    payload={
                        "name": "USER_CLICK_STATS_DECL",
                        "version": "V1",
                        "kind": "StreamingFeatureView",
                    },
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        options = PlanOptions(allow_recreate=False)

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
        ) as mock_fs:
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "refused", (
            f"Plain apply with destructive op must return status='refused'.  "
            f"Got status={result.status!r}.  See "
            f"plans/apply_allow_recreate_destructive_gate_*.plan.md."
        )
        mock_fs.assert_not_called()
        assert any("--allow-recreate" in e for e in result.errors), (
            f"Refused error must direct operator at --allow-recreate.  " f"Got errors={result.errors!r}."
        )
        assert len(result.ops) == 1
        assert result.ops[0]["operation"] == "RECREATE_FV"
        assert result.ops[0]["name"] == "USER_CLICK_STATS_DECL"
        assert result.ops[0]["status"] == "refused"
        assert result.ops[0]["destructive"] is True

    def test_execute_plan_executes_when_destructive_op_and_allow_recreate_true(self) -> None:
        """``RECREATE_FV`` + ``allow_recreate=True`` → executes normally.

        Pins that the gate only fires when the operator omits the flag.
        With the flag set, the destructive ops flow through to
        ``FeatureStore.delete_feature_view`` + ``register_feature_view``
        unchanged.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.RECREATE_FV,
                    name="USER_CLICK_STATS_DECL",
                    destructive=True,
                    reason="Full-spec change detected (UDF source).",
                    payload={
                        "name": "USER_CLICK_STATS_DECL",
                        "version": "V1",
                        "kind": "StreamingFeatureView",
                    },
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        options = PlanOptions(allow_recreate=True)

        fs_instance = MagicMock(name="FeatureStore")

        with patch("snowflake.ml.feature_store.feature_store.FeatureStore", return_value=fs_instance,), patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_feature_view",
            return_value=(MagicMock(name="fv"), "V1"),
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "applied", (
            f"Apply with --allow-recreate must execute the destructive op.  " f"Got status={result.status!r}."
        )
        fs_instance.delete_feature_view.assert_called_once_with("USER_CLICK_STATS_DECL", "V1")
        fs_instance.register_feature_view.assert_called_once()
        assert result.ops[0]["status"] == "success"

    def test_execute_plan_refuses_drop_fv_when_not_allow_recreate(self) -> None:
        """``DROP_FV`` (destructive) + ``allow_recreate=False`` → refused.

        Full-sync orphan FV deletion (planner emits ``DROP_FV`` with
        ``destructive=True``) must also require the operator opt-in,
        so a missing YAML on a shared schema cannot silently nuke a
        deployed FV.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_FV,
                    name="OLD_FV",
                    destructive=True,
                    reason="Not present in local spec files: will be dropped.",
                    payload={
                        "name": "OLD_FV",
                        "version": "V1",
                        "kind": "StreamingFeatureView",
                    },
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        options = PlanOptions(allow_recreate=False)

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
        ) as mock_fs:
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "refused"
        mock_fs.assert_not_called()
        assert result.ops[0]["operation"] == "DROP_FV"
        assert result.ops[0]["status"] == "refused"

    def test_execute_plan_refuses_drop_entity_when_not_allow_recreate(self) -> None:
        """``DROP_ENTITY`` (destructive) + ``allow_recreate=False`` → refused.

        Same gating rule as ``DROP_FV``.  The ``DROP_ENTITY`` path
        normally issues a raw ``DROP TAG`` on ``session.sql``; the
        gate must short-circuit before any such SQL is issued.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_ENTITY,
                    name="ORPHAN_ENTITY",
                    destructive=True,
                    reason="Not present in local spec files: will be dropped.",
                    payload={
                        "name": "ORPHAN_ENTITY",
                        "kind": "Entity",
                    },
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        options = PlanOptions(allow_recreate=False)

        result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "refused"
        session.sql.assert_not_called()
        assert result.ops[0]["operation"] == "DROP_ENTITY"
        assert result.ops[0]["status"] == "refused"

    def test_execute_plan_executes_non_destructive_ops_unchanged(self) -> None:
        """Non-destructive plans (``CREATE_ENTITY`` + ``UPDATE_ENTITY``)
        bypass the gate entirely regardless of ``allow_recreate``.

        Pins that the BUG_BASH §11 entity-description edit
        (``UPDATE_ENTITY``, non-destructive) keeps working under
        plain ``apply``.  ``allow_recreate=False`` must NOT block
        non-destructive ops.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER_ID",
                    destructive=False,
                    payload={
                        "kind": "Entity",
                        "name": "USER_ID",
                        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                        "description": "u",
                    },
                ),
                PlanOp(
                    kind=OpKind.UPDATE_ENTITY,
                    name="OTHER_ID",
                    destructive=False,
                    payload={
                        "kind": "Entity",
                        "name": "OTHER_ID",
                        "join_keys": [{"name": "OTHER_ID", "type": "StringType"}],
                        "description": "o",
                    },
                ),
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        cursor = MagicMock(name="cursor")
        cursor.collect.return_value = []
        session.sql.return_value = cursor
        options = PlanOptions(allow_recreate=False)

        fs_instance = MagicMock(name="FeatureStore")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs_instance,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "applied", (
            f"Non-destructive plans must NOT trip the allow_recreate gate.  " f"Got status={result.status!r}."
        )
        assert all(
            op["status"] == "success" for op in result.ops
        ), f"Both ops must execute successfully; got {result.ops!r}."
        fs_instance.register_entity.assert_called_once()
        fs_instance.update_entity.assert_called_once()

    def test_execute_plan_refused_status_carries_skipped_for_non_destructive(self) -> None:
        """Mixed plan (one destructive + one non-destructive) +
        ``allow_recreate=False`` → whole-plan refuse.

        The refuse is whole-plan (atomic): no op is executed, even the
        non-destructive ones.  The ``ops`` list reports
        ``status="refused"`` for destructive entries and
        ``status="skipped"`` for non-destructive entries so operators
        can see which op tripped the gate.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.CREATE_ENTITY,
                    name="USER_ID",
                    destructive=False,
                    payload={
                        "kind": "Entity",
                        "name": "USER_ID",
                        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
                        "description": "u",
                    },
                ),
                PlanOp(
                    kind=OpKind.RECREATE_FV,
                    name="USER_CLICK_STATS_DECL",
                    destructive=True,
                    payload={
                        "name": "USER_CLICK_STATS_DECL",
                        "version": "V1",
                        "kind": "StreamingFeatureView",
                    },
                ),
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        options = PlanOptions(allow_recreate=False)

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
        ) as mock_fs:
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "refused"
        mock_fs.assert_not_called()
        # Non-destructive ops must NOT have been executed (no DDL
        # issued via ``session.sql``).
        session.sql.assert_not_called()

        ops_by_kind = {op["operation"]: op for op in result.ops}
        assert "RECREATE_FV" in ops_by_kind
        assert "CREATE_ENTITY" in ops_by_kind
        assert ops_by_kind["RECREATE_FV"]["status"] == "refused"
        assert ops_by_kind["RECREATE_FV"]["destructive"] is True
        assert ops_by_kind["CREATE_ENTITY"]["status"] == "skipped"
        assert ops_by_kind["CREATE_ENTITY"]["destructive"] is False

    def test_recreate_fv_passes_overwrite_when_delete_fails(self) -> None:
        """``RECREATE_FV`` must pass ``overwrite=True`` to ``register_feature_view``
        even when ``delete_feature_view`` raises (e.g. FV referenced by FeatureGroup).

        Pins the fix for Bug F: without ``overwrite=True`` the re-registration fails
        with a version-conflict error whenever the FV could not be deleted, producing
        an infinite RECREATE loop on subsequent plans.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.RECREATE_FV,
                    name="USER_AMOUNTS_FG_DECL",
                    destructive=True,
                    reason="spec drift",
                    payload={
                        "name": "USER_AMOUNTS_FG_DECL",
                        "version": "V1",
                        "kind": "StreamingFeatureView",
                    },
                )
            ],
            warnings=[],
        )
        session = MagicMock(name="session")
        options = PlanOptions(allow_recreate=True)

        fs_instance = MagicMock(name="FeatureStore")
        fs_instance.delete_feature_view.side_effect = RuntimeError("FV referenced by FeatureGroup")

        with patch("snowflake.ml.feature_store.feature_store.FeatureStore", return_value=fs_instance,), patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_feature_view",
            return_value=(MagicMock(name="fv"), "V1"),
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        assert result.status == "applied"
        call_kwargs = fs_instance.register_feature_view.call_args
        assert call_kwargs is not None, "register_feature_view was never called"
        overwrite_passed = call_kwargs.kwargs.get("overwrite") or (len(call_kwargs.args) > 2 and call_kwargs.args[2])
        assert overwrite_passed, (
            "register_feature_view must be called with overwrite=True during RECREATE_FV "
            "so version-conflict is not raised when delete_feature_view fails"
        )


class TestAssertFeatureStoreInitialized:
    """Verify the ``assert_feature_store_initialized`` helper.

    The helper constructs a ``FeatureStore(FAIL_IF_NOT_EXIST)`` and
    rewraps any ``NOT_FOUND`` raised by the imperative API as the
    operator-facing ``FeatureStoreNotInitializedError`` so the CLI can
    surface a clear "run ``snow feature init``" message instead of
    leaking snowml-core internals.  This is the foundation of the
    init-first invariant that Phase 4 + the CLI Phase 6 build on.
    """

    def test_returns_fs_when_tags_present(self) -> None:
        from snowflake.ml.feature_store.decl import imperative_executor

        fs_instance = MagicMock(name="FeatureStore")
        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs_instance,
        ) as mock_fs_cls:
            result = imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

        assert result is fs_instance
        mock_fs_cls.assert_called_once()

    def test_raises_feature_store_not_initialized_when_internal_tags_missing(self) -> None:
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
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
        ), pytest.raises(decl_errors.FeatureStoreNotInitializedError):
            imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

    def test_raises_feature_store_not_initialized_when_schema_missing(self) -> None:
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
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
        ), pytest.raises(decl_errors.FeatureStoreNotInitializedError):
            imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

    def test_error_message_names_db_schema_and_init_command(self) -> None:
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
        from snowflake.ml.feature_store.decl import (
            errors as decl_errors,
            imperative_executor,
        )

        missing_tag = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist."),
        )

        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_tag,
        ), pytest.raises(decl_errors.FeatureStoreNotInitializedError) as exc_info:
            imperative_executor.assert_feature_store_initialized(session, "MY_DB", "MY_SCH", "WH")

        message = str(exc_info.value)
        assert "MY_DB" in message
        assert "MY_SCH" in message
        assert "snow feature init" in message
        # The wrapped exception is preserved on the instance so callers
        # debugging the original snowml-core error can find it.
        assert exc_info.value.database == "MY_DB"
        assert exc_info.value.schema == "MY_SCH"
        assert exc_info.value.wrapped is missing_tag

    def test_propagates_unexpected_errors_without_rewrap(self) -> None:
        """Non-``NOT_FOUND`` errors from ``FeatureStore.__init__`` —
        e.g. auth/permissions — must propagate unchanged so the CLI's
        outer handler sees the original error, not a misleading
        "not initialized" message.
        """
        from snowflake.ml.feature_store.decl import (
            errors as decl_errors,
            imperative_executor,
        )

        unexpected = PermissionError("network unreachable")
        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=unexpected,
        ), pytest.raises(PermissionError, match="network unreachable"):
            imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

        # Negative: the rewrap class must NOT have caught this.
        assert not isinstance(unexpected, decl_errors.FeatureStoreNotInitializedError)

    def test_rewraps_telemetry_unwrapped_value_error_with_2101_prefix(self) -> None:
        """In real execution, snowml-core's telemetry decorator catches
        the ``SnowflakeMLException`` thrown by
        ``_check_internal_objects_exist_or_throw`` and re-raises
        ``e.original_exception`` — a plain ``ValueError`` whose
        ``str()`` starts with the ``(<error_code>)`` marker.  The
        helper MUST match that shape so the init-first guard fires
        end-to-end against a live Snowflake schema (not just in the
        unit-test direct-raise path).

        Regression test for the live-verification failure where
        ``snow feature list`` against an uninit schema silently
        returned ``specs: []`` because the helper only caught the
        typed ``SnowflakeMLException`` shape.
        """
        from snowflake.ml.feature_store.decl import (
            errors as decl_errors,
            imperative_executor,
        )

        unwrapped = ValueError(
            "(2101) Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist. "
            "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
        )
        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=unwrapped,
        ), pytest.raises(decl_errors.FeatureStoreNotInitializedError) as exc_info:
            imperative_executor.assert_feature_store_initialized(session, "MY_DB", "MY_SCH", "WH")

        message = str(exc_info.value)
        assert "MY_DB" in message
        assert "MY_SCH" in message
        assert "snow feature init" in message
        assert exc_info.value.wrapped is unwrapped

    def test_propagates_value_errors_without_2101_prefix(self) -> None:
        """``ValueError`` without the ``(<NOT_FOUND>)`` marker must
        propagate unchanged — only the specific snowml-core uninit
        signal gets rewrapped.  Pins H5: a non-init ``ValueError``
        (e.g. a bad config arg) is not silently relabelled as
        "not initialised".
        """
        from snowflake.ml.feature_store.decl import (
            errors as decl_errors,
            imperative_executor,
        )

        unrelated = ValueError("bogus config: warehouse_required=True")
        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=unrelated,
        ), pytest.raises(ValueError, match="bogus config"):
            imperative_executor.assert_feature_store_initialized(session, "DB", "SCH", "WH")

        assert not isinstance(unrelated, decl_errors.FeatureStoreNotInitializedError)


class TestDeclApiAssertFeatureStoreInitialized:
    """``decl.api`` re-exports the helper + error class so CLI callers
    do not need to import from the internal ``imperative_executor``
    module.  These are import-surface tests only — the runtime
    behaviour is covered above.
    """

    def test_assert_feature_store_initialized_re_exported(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api
        from snowflake.ml.feature_store.decl.imperative_executor import (
            assert_feature_store_initialized as _helper,
        )

        assert decl_api.assert_feature_store_initialized is _helper

    def test_feature_store_not_initialized_error_re_exported(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api
        from snowflake.ml.feature_store.decl.errors import (
            FeatureStoreNotInitializedError as _exc,
        )

        assert decl_api.FeatureStoreNotInitializedError is _exc


# ---------------------------------------------------------------------------
# Wave 1B — StreamSource applied-state read path + UPDATE/RECREATE/DROP_SOURCE
#
# The shared interface contract for these tests lives in
# ``plans/stream_source_contract.md`` (sections 3 + 4).  ``fetch_stream_source_rows``
# mirrors ``fetch_entity_rows`` / ``fetch_feature_view_rows`` /
# ``fetch_feature_group_rows`` and lets the planner see registered streaming
# sources so it can emit ``NO_CHANGE`` / ``UPDATE_SOURCE`` / ``RECREATE_SOURCE``
# / ``DROP_SOURCE`` instead of always re-emitting ``CREATE_SOURCE``.
#
# ``OpKind.RECREATE_SOURCE`` is owned by sub-agent W1A and is not present in
# the enum at the time of writing (parallel work).  The dispatch tests below
# discover the value via ``getattr(OpKind, "RECREATE_SOURCE", None)`` and
# ``pytest.skip`` themselves if the enum extension has not landed yet — this
# keeps the W1B test suite green when run on its own and turns RED for the
# orchestrator's merge-review the moment W1A's commit shows up.
# ---------------------------------------------------------------------------


def _stream_source_row(
    name: str = "CLICKSTREAM_EVENTS",
    schema_cols: list[dict[str, str]] | None = None,
    desc: str = "Real-time click events",
    owner: str = "ROLE_X",
) -> dict[str, Any]:
    """Build a ``FeatureStore.list_stream_sources().collect()`` row dict.

    Mirrors the schema of ``_LIST_STREAM_SOURCE_SCHEMA`` (``NAME``,
    ``SCHEMA``, ``DESC``, ``OWNER``) where ``SCHEMA`` is the JSON-string
    serialisation of the source's column schema.

    Args:
        name: Stream-source name (uppercase, matching the registered form).
        schema_cols: List of ``{"name", "type"}`` dicts.  Defaults to a
            three-column clickstream schema.
        desc: Free-text description.
        owner: Snowflake role that owns the source.

    Returns:
        A dict shaped like a single ``Row.as_dict()`` from
        ``FeatureStore.list_stream_sources``.
    """
    import json as _json

    if schema_cols is None:
        schema_cols = [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TYPE", "type": "StringType"},
            {"name": "TIMESTAMP", "type": "TimestampType"},
        ]
    return {
        "NAME": name,
        "SCHEMA": _json.dumps(schema_cols),
        "DESC": desc,
        "OWNER": owner,
    }


class TestFetchStreamSourceRows:
    """Verify ``fetch_stream_source_rows`` translates
    ``FeatureStore.list_stream_sources()`` output into the contract §3
    row schema: ``{"name", "schema", "desc", "owner"}`` with ``schema``
    decoded from the JSON string column into a ``list[{"name", "type"}]``.

    Mirrors :class:`TestFetchEntityRows` for the entity read path.  Init-
    first semantics are inherited from ``assert_feature_store_initialized``;
    malformed ``SCHEMA`` JSON degrades that row's ``schema`` to ``[]`` so
    the downstream consumer can fall back to a ``recreate`` diff rather
    than crashing on metadata drift.
    """

    def test_translates_imperative_rows_to_contract_shape(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_stream_source_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            _stream_source_row(
                name="CLICKSTREAM_EVENTS",
                schema_cols=[
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TYPE", "type": "StringType"},
                ],
                desc="Real-time click events",
                owner="ROLE_X",
            ),
            _stream_source_row(
                name="USER_AMOUNTS",
                schema_cols=[{"name": "AMOUNT", "type": "DoubleType"}],
                desc="",
                owner="ROLE_Y",
            ),
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_stream_sources.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_stream_source_rows(session, "DB", "SCH", "WH")

        assert len(rows) == 2
        first = rows[0]
        assert first["name"] == "CLICKSTREAM_EVENTS"
        assert first["schema"] == [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TYPE", "type": "StringType"},
        ]
        assert first["desc"] == "Real-time click events"
        assert first["owner"] == "ROLE_X"

        second = rows[1]
        assert second["name"] == "USER_AMOUNTS"
        assert second["schema"] == [{"name": "AMOUNT", "type": "DoubleType"}]
        assert second["desc"] == ""
        assert second["owner"] == "ROLE_Y"

    def test_empty_result_returns_empty_list(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_stream_source_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = []
        fs = MagicMock(name="FeatureStore")
        fs.list_stream_sources.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_stream_source_rows(session, "DB", "SCH")

        assert rows == []

    def test_malformed_schema_json_degrades_to_empty_list(self) -> None:
        """Malformed ``SCHEMA`` JSON must not crash the read path —
        the row is emitted with ``schema=[]`` so the planner's diff
        helper falls through to a ``recreate`` rather than the read
        failing wholesale.  Per contract §3 the failure is logged at
        ``debug`` level only (no operator-visible warning) because the
        recovery path is automatic.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_stream_source_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            {
                "NAME": "BROKEN_SOURCE",
                "SCHEMA": "{not-valid-json",
                "DESC": "broken metadata",
                "OWNER": "ROLE_X",
            }
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_stream_sources.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_stream_source_rows(session, "DB", "SCH")

        assert len(rows) == 1
        assert rows[0]["name"] == "BROKEN_SOURCE"
        assert rows[0]["schema"] == [], (
            "Malformed SCHEMA JSON must degrade the row's schema to [] "
            "so the planner can fall through to a recreate decision."
        )
        assert rows[0]["desc"] == "broken metadata"
        assert rows[0]["owner"] == "ROLE_X"

    def test_init_first_error_propagates(self) -> None:
        """When the schema lacks the bootstrap tags,
        ``assert_feature_store_initialized`` raises
        ``FeatureStoreNotInitializedError`` and ``fetch_stream_source_rows``
        does NOT swallow it — the CLI catches and rewraps so the operator
        sees the actionable "run snow feature init" message.
        """
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
        from snowflake.ml.feature_store.decl.errors import (
            FeatureStoreNotInitializedError,
        )
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_stream_source_rows,
        )

        session = MagicMock(name="session")
        missing_tag = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist."),
        )
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_tag,
        ), pytest.raises(FeatureStoreNotInitializedError):
            fetch_stream_source_rows(session, "DB", "SCH", "WH")

        # Negative pin: no raw fallback SQL — fetch_stream_source_rows
        # is strictly imperative.
        session.sql.assert_not_called()


class TestExecuteUpdateStreamSource:
    """``OpKind.UPDATE_SOURCE`` for ``StreamingSource`` must call
    ``FeatureStore.update_stream_source(name, desc=desc)`` — the only
    field ``update_stream_source`` accepts is ``desc``.  ``BatchSource``
    has no runtime metadata to update, so an UPDATE_SOURCE for that kind
    is an informational no-op (no FS call, ``status: success``).
    """

    def _streaming_payload(self, name: str = "CLICKSTREAM_EVENTS", desc: str = "Updated description") -> Any:
        return {
            "kind": "StreamingSource",
            "name": name,
            "type": "REST",
            "description": desc,
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TYPE", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
            ],
        }

    def test_streaming_update_calls_fs_update_stream_source(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.UPDATE_SOURCE,
                    name="CLICKSTREAM_EVENTS",
                    payload=self._streaming_payload(desc="Updated click description"),
                    destructive=False,
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.update_stream_source.assert_called_once_with(
            "CLICKSTREAM_EVENTS",
            desc="Updated click description",
        )
        # The dispatch must NOT touch register_stream_source on update — that
        # path is reserved for CREATE_SOURCE / RECREATE_SOURCE.
        fs.register_stream_source.assert_not_called()
        fs.delete_stream_source.assert_not_called()
        assert result.ops[0]["status"] == "success"

    def test_streaming_update_with_legacy_desc_key(self) -> None:
        """Payloads written via the legacy ``desc`` key (not
        ``description``) must still thread through ``update_stream_source``.
        Mirrors the resolution order used by
        ``_execute_create_stream_source``.
        """
        payload = self._streaming_payload(desc="")
        payload.pop("description", None)
        payload["desc"] = "Updated via legacy key"
        plan = Plan(
            ops=[PlanOp(kind=OpKind.UPDATE_SOURCE, name="CLICKSTREAM_EVENTS", payload=payload)],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.update_stream_source.assert_called_once_with(
            "CLICKSTREAM_EVENTS",
            desc="Updated via legacy key",
        )

    def test_batch_update_is_informational_no_op(self) -> None:
        """BatchSource has no runtime metadata table — the planner still
        emits ``UPDATE_SOURCE`` for parity with the StreamingSource
        diff path, but the executor must NOT call any FS method.  The op
        completes with ``status: success`` so the operator sees the
        informational entry in the apply report.
        """
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.UPDATE_SOURCE,
                    name="ORDER_HISTORY",
                    payload={
                        "kind": "BatchSource",
                        "name": "ORDER_HISTORY",
                        "table": "ORDERS",
                        "description": "Updated batch desc",
                    },
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())

        fs.update_stream_source.assert_not_called()
        fs.register_stream_source.assert_not_called()
        fs.delete_stream_source.assert_not_called()
        assert result.ops[0]["status"] == "success"


def _recreate_source_opkind_or_skip() -> Any:
    """Return ``OpKind.RECREATE_SOURCE`` if W1A's enum extension has landed.

    The W1A sub-agent owns ``decl/enums.py`` and adds the new value in a
    sibling commit.  Until that commit lands, the value is absent and
    any test that tries to construct a ``PlanOp(kind=OpKind.RECREATE_SOURCE)``
    will fail at Pydantic-validation time.  Returning ``None`` lets the
    caller ``pytest.skip`` the test cleanly; once W1A merges the enum
    extension, every skipped test starts running automatically.

    Returns:
        The enum member, or ``None`` if the value is not yet defined.
    """
    return getattr(OpKind, "RECREATE_SOURCE", None)


class TestExecuteRecreateStreamSource:
    """``OpKind.RECREATE_SOURCE`` is the destructive form for streaming-source
    schema or binding changes.  The executor must call
    ``fs.delete_stream_source(name)`` followed by ``fs.register_stream_source(...)``
    for ``StreamingSource``; ``BatchSource`` remains a no-op.

    These tests skip until ``OpKind.RECREATE_SOURCE`` (owned by W1A) lands.
    """

    def _streaming_payload(self, name: str = "CLICKSTREAM_EVENTS") -> Any:
        return {
            "kind": "StreamingSource",
            "name": name,
            "type": "REST",
            "description": "Recreated stream",
            "columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TYPE", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
                {"name": "AMOUNT", "type": "DoubleType"},
            ],
        }

    def test_streaming_recreate_deletes_then_registers(self) -> None:
        recreate_kind = _recreate_source_opkind_or_skip()
        if recreate_kind is None:
            pytest.skip("OpKind.RECREATE_SOURCE not yet added by W1A.")

        plan = Plan(
            ops=[
                PlanOp(
                    kind=recreate_kind,
                    name="CLICKSTREAM_EVENTS",
                    payload=self._streaming_payload(),
                    destructive=True,
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.register_stream_source.return_value = MagicMock(name="StreamSource")

        call_order: list[str] = []

        def _record_delete(name: str) -> None:
            call_order.append(f"delete:{name}")

        def _record_register(stream_source: Any) -> Any:
            call_order.append(f"register:{stream_source.name}")
            return stream_source

        fs.delete_stream_source.side_effect = _record_delete
        fs.register_stream_source.side_effect = _record_register

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        # Order matters: delete must happen before register so the
        # snowml-core idempotency check on register_stream_source does
        # not warn-and-skip the new schema.
        assert call_order == [
            "delete:CLICKSTREAM_EVENTS",
            "register:CLICKSTREAM_EVENTS",
        ], f"Expected delete-then-register; got {call_order!r}"
        fs.delete_stream_source.assert_called_once_with("CLICKSTREAM_EVENTS")
        fs.register_stream_source.assert_called_once()
        assert result.ops[0]["status"] == "success"

    def test_batch_recreate_is_no_op(self) -> None:
        recreate_kind = _recreate_source_opkind_or_skip()
        if recreate_kind is None:
            pytest.skip("OpKind.RECREATE_SOURCE not yet added by W1A.")

        plan = Plan(
            ops=[
                PlanOp(
                    kind=recreate_kind,
                    name="ORDER_HISTORY",
                    payload={
                        "kind": "BatchSource",
                        "name": "ORDER_HISTORY",
                        "table": "ORDERS",
                    },
                    destructive=True,
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        fs.delete_stream_source.assert_not_called()
        fs.register_stream_source.assert_not_called()
        fs.update_stream_source.assert_not_called()
        assert result.ops[0]["status"] == "success"


class TestExecuteDropStreamSource:
    """``OpKind.DROP_SOURCE`` for ``StreamingSource`` dispatches to
    ``FeatureStore.delete_stream_source(name)``.  Any exception (e.g.
    ``ref_count > 0`` because an FV still references the source) is
    propagated so the planner's op records ``partial_failure``.
    ``BatchSource`` ops remain informational no-ops.
    """

    def test_streaming_drop_calls_fs_delete_stream_source(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_SOURCE,
                    name="CLICKSTREAM_EVENTS",
                    payload={
                        "kind": "StreamingSource",
                        "name": "CLICKSTREAM_EVENTS",
                    },
                    destructive=True,
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        fs.delete_stream_source.assert_called_once_with("CLICKSTREAM_EVENTS")
        fs.register_stream_source.assert_not_called()
        fs.update_stream_source.assert_not_called()
        assert result.ops[0]["status"] == "success"

    def test_batch_drop_is_no_op(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_SOURCE,
                    name="ORDER_HISTORY",
                    payload={
                        "kind": "BatchSource",
                        "name": "ORDER_HISTORY",
                    },
                    destructive=True,
                )
            ],
            warnings=[],
        )

        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))

        fs.delete_stream_source.assert_not_called()
        fs.register_stream_source.assert_not_called()
        fs.update_stream_source.assert_not_called()
        assert result.ops[0]["status"] == "success"


class TestStreamingFvDoesNotDefensivelyRegisterStreamSource:
    """The FV-side defensive ``_execute_create_stream_source`` call inside
    ``_execute_create_streaming_fv`` (the old re-register-just-in-case
    safety net at lines ~2070-2080) is removed.

    With ``fetch_stream_source_rows`` populating applied state, the
    planner emits ``CREATE_SOURCE`` exactly once per genuinely-new source
    before any ``CREATE_FV`` op runs.  The FV-side defensive register was
    the second source of the ``UserWarning: StreamSource <name> already
    exists. Skip registration.`` noise the user reported.

    Critically: the FV's ``StreamConfig`` must still receive the
    resolved ``stream_source`` name — only the ``register_stream_source``
    call is removed.
    """

    def _patch_constructors(self) -> tuple[Any, Any]:
        """Capture ``FeatureView`` and ``StreamConfig`` kwargs."""
        fv_calls: list[dict[str, Any]] = []
        sc_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            fv_calls.append(kwargs)
            return type("_FakeFV", (), {**kwargs})()

        def _fake_sc(**kwargs: Any) -> Any:
            sc_calls.append(kwargs)
            return type("_FakeSC", (), {**kwargs})()

        return (fv_calls, _fake_fv), (sc_calls, _fake_sc)

    def test_streaming_fv_build_does_not_call_execute_create_stream_source(self) -> None:
        """``_build_feature_view`` for a StreamingFV must NOT invoke
        ``_execute_create_stream_source`` defensively.  The planner's
        CREATE_SOURCE op (driven by ``fetch_stream_source_rows`` /
        applied state) is the single source of truth now.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (_, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ), patch("snowflake.ml.feature_store.decl.imperative_executor._execute_create_stream_source") as defensive_call:
            _build_feature_view(
                _streaming_fv_payload(),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        defensive_call.assert_not_called()
        # The previous defensive path also routed through fs.register_stream_source —
        # belt-and-suspenders: pin neither path fired during FV build.
        fs.register_stream_source.assert_not_called()

    def test_streaming_fv_stream_config_still_receives_resolved_source_name(self) -> None:
        """Removing the defensive register must NOT regress the
        ``StreamConfig.stream_source`` name: the FV-side branch still
        looks up the source spec in ``payload['sources']`` to extract
        the canonical name and pass it through to ``StreamConfig``.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        (_, fake_fv), (sc_calls, fake_sc) = self._patch_constructors()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig", side_effect=fake_sc
        ):
            _build_feature_view(
                _streaming_fv_payload(),
                session,
                "DB",
                "SCH",
                "WH",
                fs=fs,
            )

        assert len(sc_calls) == 1, "StreamConfig must be constructed exactly once for a streaming FV"
        assert sc_calls[0]["stream_source"] == "CLICKSTREAM_EVENTS", (
            "StreamConfig must still receive the resolved stream-source "
            "name extracted from payload['sources'] even after the "
            "defensive register call is removed."
        )


class TestExecutePlanSourceDispatch:
    """End-to-end dispatch matrix for the source-side OpKinds defined in
    contract §4.  Confirms ``execute_plan`` routes each
    ``(OpKind, payload['kind'])`` pair to the right helper without
    falling through to the legacy ``status: skipped`` no-op branch for
    every non-CREATE source op.
    """

    def test_dispatch_streaming_update_source(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.UPDATE_SOURCE,
                    name="CLICKSTREAM_EVENTS",
                    payload={
                        "kind": "StreamingSource",
                        "name": "CLICKSTREAM_EVENTS",
                        "description": "d",
                    },
                )
            ],
            warnings=[],
        )
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())
        fs.update_stream_source.assert_called_once()

    def test_dispatch_streaming_drop_source(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_SOURCE,
                    name="CLICKSTREAM_EVENTS",
                    payload={"kind": "StreamingSource", "name": "CLICKSTREAM_EVENTS"},
                    destructive=True,
                )
            ],
            warnings=[],
        )
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))
        fs.delete_stream_source.assert_called_once()

    def test_dispatch_streaming_recreate_source(self) -> None:
        recreate_kind = _recreate_source_opkind_or_skip()
        if recreate_kind is None:
            pytest.skip("OpKind.RECREATE_SOURCE not yet added by W1A.")

        plan = Plan(
            ops=[
                PlanOp(
                    kind=recreate_kind,
                    name="CLICKSTREAM_EVENTS",
                    payload={
                        "kind": "StreamingSource",
                        "name": "CLICKSTREAM_EVENTS",
                        "columns": [{"name": "USER_ID", "type": "StringType"}],
                    },
                    destructive=True,
                )
            ],
            warnings=[],
        )
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))
        fs.delete_stream_source.assert_called_once()
        fs.register_stream_source.assert_called_once()

    def test_dispatch_batch_update_source_no_op(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.UPDATE_SOURCE,
                    name="ORDER_HISTORY",
                    payload={"kind": "BatchSource", "name": "ORDER_HISTORY"},
                )
            ],
            warnings=[],
        )
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions())
        for method in ("update_stream_source", "delete_stream_source", "register_stream_source"):
            assert not getattr(fs, method).called, f"BatchSource UPDATE must not invoke fs.{method}"
        assert result.ops[0]["status"] == "success"

    def test_dispatch_batch_drop_source_no_op(self) -> None:
        plan = Plan(
            ops=[
                PlanOp(
                    kind=OpKind.DROP_SOURCE,
                    name="ORDER_HISTORY",
                    payload={"kind": "BatchSource", "name": "ORDER_HISTORY"},
                    destructive=True,
                )
            ],
            warnings=[],
        )
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            result = execute_plan(plan, session, "DB", "SCH", "WH", PlanOptions(allow_recreate=True))
        for method in ("update_stream_source", "delete_stream_source", "register_stream_source"):
            assert not getattr(fs, method).called, f"BatchSource DROP must not invoke fs.{method}"
        assert result.ops[0]["status"] == "success"


# ---------------------------------------------------------------------------
# B6 / B7 — UPDATE_FV imperative routing: desc + refresh_freq forwarding;
# streaming-kind extension via the A5-extended fs.update_feature_view path.
# ---------------------------------------------------------------------------


class TestUpdateFvBatch:
    """B6: ``_apply_update_fv`` forwards ``desc`` and ``refresh_freq`` from the
    plan payload to ``fs.update_feature_view(...)``.

    Closes L3 and L4 on the executor side: when the planner emits an
    UPDATE_FV op carrying ``description`` and ``refresh_freq`` in the
    payload, the executor must pass them through as ``desc=...`` and
    ``refresh_freq=...`` kwargs so the imperative ALTER path actually
    applies the change.
    """

    def test_forwards_desc_and_refresh_freq(self) -> None:
        """Payload ``description`` → ``desc`` kwarg; ``refresh_freq`` → ``refresh_freq``."""
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _execute_update_feature_view,
        )

        fs = MagicMock()
        op = PlanOp(
            kind=OpKind.UPDATE_FV,
            name="BFV",
            depends_on=[],
            destructive=False,
            reason="test",
            payload={
                "kind": "BatchFeatureView",
                "name": "BFV",
                "version": "V1",
                "description": "updated description",
                "refresh_freq": "10 minutes",
            },
        )
        _execute_update_feature_view(fs, op, "WH0")

        fs.update_feature_view.assert_called_once()
        _, kwargs = fs.update_feature_view.call_args
        assert kwargs.get("desc") == "updated description", (
            "Payload ``description`` must be forwarded to fs.update_feature_view " f"as ``desc``; got kwargs={kwargs!r}"
        )
        assert kwargs.get("refresh_freq") == "10 minutes", (
            "Payload ``refresh_freq`` must be forwarded to fs.update_feature_view "
            f"as ``refresh_freq``; got kwargs={kwargs!r}"
        )


class TestUpdateFvStreaming:
    """B7: ``_apply_update_fv`` routes StreamingFV UPDATE_FV through the
    A5-extended ``fs.update_feature_view`` path.

    Before B7 the executor rejected non-BatchFV kinds with a hard
    ``ValueError``.  A5 extended ``FeatureStore.update_feature_view``
    to accept the streaming-compatible kwargs (``refresh_freq``,
    ``warehouse``, ``desc``, ``online_config``), so the executor must
    now forward them instead of raising.
    """

    def test_routes_to_extended_fs_update(self) -> None:
        """StreamingFV UPDATE_FV must reach ``fs.update_feature_view`` (no ValueError)."""
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _execute_update_feature_view,
        )

        fs = MagicMock()
        op = PlanOp(
            kind=OpKind.UPDATE_FV,
            name="SFV",
            depends_on=[],
            destructive=False,
            reason="test",
            payload={
                "kind": "StreamingFeatureView",
                "name": "SFV",
                "version": "V1",
                "description": "streaming desc",
                "refresh_freq": "5 minutes",
                "warehouse": "WH_NEW",
            },
        )
        # Must not raise — pre-B7 this raised ValueError("UPDATE_FV is only
        # supported for BatchFeatureView ...").  Post-B7 the executor
        # gates by kind and routes streaming through the extended A5 path.
        _execute_update_feature_view(fs, op, "WH0")

        fs.update_feature_view.assert_called_once()
        args, kwargs = fs.update_feature_view.call_args
        assert args[0] == "SFV"
        assert args[1] == "V1"
        assert kwargs.get("desc") == "streaming desc", (
            "StreamingFV update must forward ``description`` as ``desc``; " f"got kwargs={kwargs!r}"
        )
        # ``refresh_freq`` is BFV-only post-rename: the spec validator
        # rejects authoring on streaming kinds and the runtime stamps
        # target_lag_sec=0 regardless.  Phase 3 GREEN scoped the
        # forwarding to BatchFeatureView, so even a hand-built payload
        # that smuggles refresh_freq past the validator must not leak
        # the kwarg into a streaming UPDATE_FV.
        assert "refresh_freq" not in kwargs, (
            "StreamingFV update must not forward refresh_freq; " f"got kwargs={kwargs!r}"
        )
        assert kwargs.get("warehouse") == "WH_NEW", (
            "StreamingFV update must forward ``warehouse``; " f"got kwargs={kwargs!r}"
        )


# ---------------------------------------------------------------------------
# Phase 3 RED — refresh_freq propagation through imperative_executor
# ---------------------------------------------------------------------------


class TestRefreshFreqForwarding:
    """``_build_feature_view`` and ``_execute_update_feature_view`` must
    read the renamed authoring key ``refresh_freq`` (not the legacy
    ``refresh_freq``) from the plan payload.

    The forwarding is also kind-gated as defence-in-depth: streaming and
    realtime FVs never receive the kwarg, mirroring the spec-validator
    rejection of ``refresh_freq`` on those kinds.  Even a hand-built
    payload that smuggles ``refresh_freq`` past the validator must not
    end up driving the imperative constructor / update on the wrong
    kind.
    """

    def _patch_fv_constructor(self) -> tuple[list[dict[str, Any]], Any]:
        fv_calls: list[dict[str, Any]] = []

        def _fake_fv(**kwargs: Any) -> Any:
            fv_calls.append(kwargs)
            return type("_FakeFV", (), {**kwargs})()

        return fv_calls, _fake_fv

    def _patch_stream_config(self) -> tuple[list[dict[str, Any]], Any]:
        sc_calls: list[dict[str, Any]] = []

        def _fake_sc(**kwargs: Any) -> Any:
            sc_calls.append(kwargs)
            return type("_FakeSC", (), {**kwargs})()

        return sc_calls, _fake_sc

    def test_build_batch_fv_forwards_refresh_freq_kwarg(self) -> None:
        """BFV payload with ``refresh_freq`` flows into ``FeatureView(refresh_freq=...)``."""
        fv_calls, fake_fv = self._patch_fv_constructor()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        payload = _fv_payload(name="BFV_REFRESH_FREQ_FORWARD", entity_names=["USER_ID"])
        payload["refresh_freq"] = "5 minutes"

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv):
            _build_feature_view(payload, session, "DB", "SCH", "WH", fs=fs)

        assert len(fv_calls) == 1
        assert fv_calls[0].get("refresh_freq") == "5 minutes", (
            "_build_feature_view must source the FeatureView constructor "
            "kwarg from the renamed payload key 'refresh_freq' (not the "
            "legacy 'refresh_freq'). Got "
            f"refresh_freq={fv_calls[0].get('refresh_freq')!r}, "
            f"all kwargs={sorted(fv_calls[0].keys())}."
        )

    def test_build_streaming_fv_does_not_forward_refresh_freq(self) -> None:
        """Streaming-kind payload must never set ``refresh_freq`` on the
        FeatureView constructor — even when the field is present in the
        payload (defence-in-depth against hand-built bypass).
        """
        fv_calls, fake_fv = self._patch_fv_constructor()
        sc_calls, fake_sc = self._patch_stream_config()
        session = _mock_fv_session()
        fs = MagicMock(name="FeatureStore")
        fs.get_entity.return_value = _registered_entity("USER_ID")

        payload = _streaming_fv_payload(name="SFV_REFRESH_FREQ_FORWARD")
        payload["refresh_freq"] = "5 minutes"

        with patch("snowflake.ml.feature_store.feature_view.FeatureView", side_effect=fake_fv,), patch(
            "snowflake.ml.feature_store.stream_config.StreamConfig",
            side_effect=fake_sc,
        ):
            _build_feature_view(payload, session, "DB", "SCH", "WH", fs=fs)

        assert len(fv_calls) == 1, fv_calls
        assert "refresh_freq" not in fv_calls[0], (
            "Streaming FV branch of _build_feature_view must NOT forward "
            "refresh_freq to the FeatureView constructor — the runtime "
            "stamps target_lag_sec=0 regardless, so the authored value "
            "has no effect.  Got "
            f"refresh_freq={fv_calls[0].get('refresh_freq')!r}."
        )

    def test_update_batch_fv_forwards_refresh_freq_kwarg(self) -> None:
        """``_execute_update_feature_view`` reads ``refresh_freq`` from the
        BFV payload and forwards it to ``fs.update_feature_view``.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _execute_update_feature_view,
        )

        fs = MagicMock()
        op = PlanOp(
            kind=OpKind.UPDATE_FV,
            name="BFV",
            depends_on=[],
            destructive=False,
            reason="test",
            payload={
                "kind": "BatchFeatureView",
                "name": "BFV",
                "version": "V1",
                "refresh_freq": "10 minutes",
            },
        )
        _execute_update_feature_view(fs, op, "WH0")

        fs.update_feature_view.assert_called_once()
        _, kwargs = fs.update_feature_view.call_args
        assert kwargs.get("refresh_freq") == "10 minutes", (
            "_execute_update_feature_view must source the kwarg from the "
            "renamed payload key 'refresh_freq' (not 'refresh_freq'); "
            f"got kwargs={kwargs!r}."
        )

    def test_update_streaming_fv_does_not_forward_refresh_freq(self) -> None:
        """Defence-in-depth: even if a hand-built UPDATE_FV payload for a
        streaming kind carries ``refresh_freq``, the executor must not
        forward it.  Mirrors the realtime gate already present.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _execute_update_feature_view,
        )

        fs = MagicMock()
        op = PlanOp(
            kind=OpKind.UPDATE_FV,
            name="SFV",
            depends_on=[],
            destructive=False,
            reason="test",
            payload={
                "kind": "StreamingFeatureView",
                "name": "SFV",
                "version": "V1",
                "refresh_freq": "5 minutes",
                "description": "x",
            },
        )
        _execute_update_feature_view(fs, op, "WH0")

        if fs.update_feature_view.called:
            _, kwargs = fs.update_feature_view.call_args
            assert "refresh_freq" not in kwargs, (
                "Streaming UPDATE_FV must not forward refresh_freq — the "
                "kind gate must drop the kwarg.  Got "
                f"kwargs={kwargs!r}."
            )


class TestBuildFeaturesIncomplete:
    """Incomplete aggregation rows must raise instead of being dropped."""

    def test_passthrough_returns_empty(self) -> None:
        features = _build_features(
            [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event", "type": "StringType"},
                }
            ]
        )
        assert features == []

    def test_complete_aggregation_builds_one(self) -> None:
        features = _build_features(
            [
                {
                    "function": "sum",
                    "window_sec": 3600,
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                }
            ]
        )
        assert len(features) == 1
        assert features[0]._window == "3600s"

    def test_bare_numeric_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            _build_features(
                [
                    {
                        "function": "sum",
                        "window": "300",
                        "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                        "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    }
                ]
            )

    def test_missing_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            _build_features(
                [
                    {
                        "function": "sum",
                        "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                        "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    }
                ]
            )

    def test_missing_function_raises(self) -> None:
        with pytest.raises(ValueError, match="function"):
            _build_features(
                [
                    {
                        "window_sec": 3600,
                        "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                        "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    }
                ]
            )

    def test_missing_source_column_raises(self) -> None:
        with pytest.raises(ValueError, match="source_column"):
            _build_features(
                [
                    {
                        "function": "sum",
                        "window_sec": 3600,
                        "source_column": {"name": "", "type": "DoubleType"},
                        "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
                    }
                ]
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
