"""Tests for decl/invariants.py — all invariant rule checks.

Each test constructs minimal SpecBatch / AppliedState fixtures, calls
validate_specs(), and asserts the expected ValidationResult severity and code.
"""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest

from snowflake.ml.feature_store.decl.invariants import (
    _check_column_evolution,
    _check_database_schema_mismatch,
    _check_dependencies,
    _check_destructive,
    _check_idempotency,
    _check_state_sync,
    _check_versions,
    _content_hash,
    _spec_hash,
    spec_key,
    structural_fingerprint_hash,
    validate_specs,
)
from snowflake.ml.feature_store.decl.types import AppliedObject, AppliedState, SpecBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity_spec(name: str = "user", join_key: str = "user_id") -> dict[str, Any]:
    return {
        "kind": "Entity",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "join_keys": [{"name": join_key, "type": "StringType"}],
    }


def _make_source_spec(name: str = "clicks", col: str = "event") -> dict[str, Any]:
    return {
        "kind": "StreamingSource",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "type": "REST",
        "columns": [{"name": "user_id", "type": "StringType"}, {"name": col, "type": "StringType"}],
    }


def _make_fv_spec(
    name: str = "click_features",
    version: str | None = "V1",
    features: list[Any] | None = None,
    sources: list[Any] | None = None,
    entity_cols: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "version": version,
        "entities": entity_cols or ["user_id"],
        "sources": sources or [{"name": "clicks", "source_type": "Stream"}],
        "features": features
        or [
            {
                "source_column": {"name": "event", "type": "StringType"},
                "output_column": {"name": "event", "type": "StringType"},
            }
        ],
    }


def _kind_to_model(spec: dict[str, Any]) -> Any:
    """Create a typed Pydantic model from a raw spec dict."""
    from snowflake.ml.feature_store.decl.spec_models import (
        Entity,
        FeatureGroup,
        FeatureView,
        SpecBase,
        StreamingSource,
    )

    kind = spec.get("kind", "")
    model_cls = cast(
        "type[SpecBase]",
        {
            "Entity": Entity,
            "StreamingFeatureView": FeatureView,
            "RealtimeFeatureView": FeatureView,
            "BatchFeatureView": FeatureView,
            "StreamingSource": StreamingSource,
            "FeatureGroup": FeatureGroup,
        }.get(kind, SpecBase),
    )
    return model_cls.model_validate(spec)


def _normalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw spec dict the same way validate_specs does."""
    from snowflake.ml.feature_store.decl.invariants import model_to_dict

    return model_to_dict(_kind_to_model(spec))


def _make_applied_object(spec: dict[str, Any]) -> AppliedObject:
    """Create AppliedObject using the same normalization as validate_specs."""
    normalized = _normalize(spec)
    key = spec_key(normalized)
    return AppliedObject(
        key=key,
        kind=normalized.get("kind", ""),
        name=normalized.get("name", ""),
        version=normalized.get("version"),
        content_hash=structural_fingerprint_hash(normalized),
        spec_payload=normalized,
        columns=[],
    )


def _empty_applied() -> AppliedState:
    return AppliedState(objects={})


def _applied_with(spec: dict[str, Any]) -> AppliedState:
    obj = _make_applied_object(spec)
    return AppliedState(objects={obj.key: obj})


def _batch(*specs: dict[str, Any]) -> SpecBatch:
    spec_objs = [_kind_to_model(s) for s in specs]
    return SpecBatch(specs=spec_objs, source_files=[])


# ---------------------------------------------------------------------------
# spec_key tests
# ---------------------------------------------------------------------------


class TestSpecKey:
    def test_basic_key(self) -> None:
        spec = {"kind": "Entity", "name": "user", "database": "DB", "schema": "SCH"}
        assert spec_key(spec) == "Entity:DB.SCH:USER"

    def test_missing_db_schema(self) -> None:
        spec = {"kind": "Entity", "name": "user"}
        # qualifier empty string
        assert spec_key(spec) == "Entity::USER"

    def test_partial_qualifier(self) -> None:
        spec = {"kind": "FeatureView", "name": "fv1", "database": "PROD"}
        key = spec_key(spec)
        assert key.startswith("FeatureView:")
        assert key.endswith(":FV1")

    def test_spec_key_falls_back_to_context_when_dict_lacks_db_schema(self) -> None:
        """Context kwargs supply database/schema only when the dict omits them.

        This pins the round-trip contract for ``snow feature plan`` /
        ``snow feature apply``: when a YAML spec is loaded from a path that
        does not encode database+schema (e.g. a single-file invocation or a
        bare entity YAML lacking ``database:``/``schema:``), the planner
        must qualify the key against the *connection* context so the lookup
        collides with the applied-state key (which is *always* fully
        qualified by ``state._build_spec_key``).

        Without this fallback, ``spec_key`` returns ``Entity::USER`` while
        the applied side returns ``Entity:JKEW_DB.JKEW_SCHEMA:USER``, the
        match fails, and the planner emits a phantom ``CREATE_ENTITY`` plus
        a phantom ``DROP_ENTITY`` for the same object on a clean
        round-trip.
        """

        # 1) Dict lacks both → context wins.
        spec = {"kind": "Entity", "name": "USER"}
        assert spec_key(spec, database="JKEW_DB", schema="JKEW_SCHEMA") == "Entity:JKEW_DB.JKEW_SCHEMA:USER"

        # 2) Dict carries explicit db/schema → dict wins; context is *fallback*
        #    only, never an override (a multi-store batch must keep its own
        #    qualifier on each spec).
        spec_explicit = {
            "kind": "Entity",
            "name": "USER",
            "database": "OTHER_DB",
            "schema_": "OTHER_SCHEMA",
        }
        assert spec_key(spec_explicit, database="JKEW_DB", schema="JKEW_SCHEMA") == "Entity:OTHER_DB.OTHER_SCHEMA:USER"

        # 3) No kwargs supplied → backwards-compat with the existing
        #    ``Entity::USER`` shape (every callsite that doesn't yet thread
        #    context must keep working unchanged).
        assert spec_key({"kind": "Entity", "name": "USER"}) == "Entity::USER"


# ---------------------------------------------------------------------------
# _spec_hash / _content_hash tests
# ---------------------------------------------------------------------------


class TestHashing:
    def test_spec_hash_deterministic(self) -> None:
        spec = _make_fv_spec()
        assert _spec_hash(spec) == _spec_hash(spec)

    def test_spec_hash_excludes_internal_keys(self) -> None:
        spec = _make_fv_spec()
        spec_with_meta = dict(spec, _hash="ignored", _content_hash="ignored")
        assert _spec_hash(spec) == _spec_hash(spec_with_meta)

    def test_content_hash_excludes_version(self) -> None:
        spec_v1 = _make_fv_spec(version="V1")
        spec_v2 = _make_fv_spec(version="V2")
        assert _content_hash(spec_v1) == _content_hash(spec_v2)

    def test_spec_hash_changes_on_content_change(self) -> None:
        spec1 = _make_fv_spec(name="fv1")
        spec2 = _make_fv_spec(name="fv2")
        assert _spec_hash(spec1) != _spec_hash(spec2)


# ---------------------------------------------------------------------------
# _check_versions tests
# ---------------------------------------------------------------------------


class TestCheckVersions:
    def test_missing_version_non_dev_mode_is_error(self) -> None:
        spec = _make_fv_spec(version=None)
        results = _check_versions(spec, None, dev_mode=False)
        assert any(r.severity == "ERROR" and "version" in r.code.lower() for r in results)

    def test_missing_version_dev_mode_is_ok(self) -> None:
        spec = _make_fv_spec(version=None)
        results = _check_versions(spec, None, dev_mode=True)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) == 0

    def test_version_not_greater_than_deployed_is_error(self) -> None:
        spec = _make_fv_spec(version="V1")
        applied_spec = _make_fv_spec(version="V2")
        applied = _make_applied_object(applied_spec)
        results = _check_versions(spec, applied, dev_mode=False)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) >= 1
        assert any("VERSION_CONFLICT" in r.code for r in errors)

    def test_version_equal_to_deployed_is_not_a_validator_error(self) -> None:
        """Equal versions never fire VERSION_CONFLICT in the validator.

        ``_check_versions`` flags only *strictly lower* local versions as
        VERSION_CONFLICT (re-plan-of-identical-spec semantics — see
        ``plans/planner_revalidate_identical_spec.plan.md``).  When
        ``version == applied_version`` and the content actually differs
        the planner emits a destructive ``RECREATE_FV`` op (gated by
        ``--allow-recreate``); the validator stays silent.  When the
        content matches, ``_check_idempotency`` short-circuits to
        ``NO_CHANGE`` upstream and ``_check_versions`` is never invoked.
        """
        spec = _make_fv_spec(version="V1")
        applied_spec = _make_fv_spec(version="V1")
        applied = _make_applied_object(applied_spec)
        applied.content_hash = "different_hash"
        results = _check_versions(spec, applied, dev_mode=False)
        errors = [r for r in results if r.severity == "ERROR"]
        assert not any(
            "VERSION_CONFLICT" in r.code for r in errors
        ), f"VERSION_CONFLICT must not fire for equal versions; got errors={errors!r}"

    def test_version_greater_than_deployed_ok(self) -> None:
        spec = _make_fv_spec(version="V2")
        applied_spec = _make_fv_spec(version="V1")
        applied = _make_applied_object(applied_spec)
        results = _check_versions(spec, applied, dev_mode=False)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) == 0

    def test_new_object_no_version_conflict(self) -> None:
        spec = _make_fv_spec(version="V1")
        results = _check_versions(spec, None, dev_mode=False)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# _check_idempotency tests
# ---------------------------------------------------------------------------


class TestCheckIdempotency:
    def test_same_spec_is_up_to_date(self) -> None:
        spec = _make_fv_spec(version="V1")
        applied = _make_applied_object(spec)
        is_up_to_date, results = _check_idempotency(_normalize(spec), applied, dev_mode=False)
        assert is_up_to_date is True
        assert any(r.code == "NO_CHANGE" for r in results)

    def test_different_spec_not_up_to_date(self) -> None:
        spec = _make_fv_spec(version="V2")
        applied_spec = _make_fv_spec(version="V1")
        applied = _make_applied_object(applied_spec)
        is_up_to_date, results = _check_idempotency(_normalize(spec), applied, dev_mode=False)
        assert is_up_to_date is False

    def test_dev_mode_version_excluded_hash_match(self) -> None:
        spec_v1 = _make_fv_spec(version="V1")
        spec_devver = _make_fv_spec(version="dev-20240101T120000Z")
        # content is the same except version — content_hash should match
        applied = AppliedObject(
            key=spec_key(spec_v1),
            kind="StreamingFeatureView",
            name="click_features",
            version="V1",
            content_hash=_content_hash(spec_v1),
            spec_payload=spec_v1,
        )
        is_up_to_date, results = _check_idempotency(spec_devver, applied, dev_mode=True)
        assert is_up_to_date is True

    def test_no_applied_object_is_not_up_to_date(self) -> None:
        spec = _make_fv_spec()
        is_up_to_date, results = _check_idempotency(spec, None, dev_mode=False)
        assert is_up_to_date is False


# ---------------------------------------------------------------------------
# _check_column_evolution tests
# ---------------------------------------------------------------------------


class TestCheckColumnEvolution:
    def _fv_with_features(self, cols: list[tuple[str, str]], version: str = "V1") -> dict[str, Any]:
        features = [{"source_column": {"name": c, "type": t}, "output_column": {"name": c, "type": t}} for c, t in cols]
        return _make_fv_spec(version=version, features=features)

    def test_new_output_col_without_default_is_warning(self) -> None:
        old_spec = self._fv_with_features([("event", "StringType")])
        new_spec = self._fv_with_features(
            [
                ("event", "StringType"),
                ("new_col", "StringType"),  # no default
            ],
            version="V2",
        )
        applied = _make_applied_object(old_spec)
        results = _check_column_evolution(new_spec, applied)
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "COLUMN_ADDED" for r in warnings)

    def test_new_output_col_with_default_is_ok(self) -> None:
        old_spec = self._fv_with_features([("event", "StringType")])
        new_spec = {**self._fv_with_features([("event", "StringType")], version="V2")}
        # Add a feature with default
        new_spec["features"].append(
            {
                "source_column": {"name": "new_col", "type": "StringType"},
                "output_column": {"name": "new_col", "type": "StringType", "default": "unknown"},
            }
        )
        applied = _make_applied_object(old_spec)
        results = _check_column_evolution(new_spec, applied)
        errors = [r for r in results if r.code == "COLUMN_MISSING_DEFAULT"]
        assert len(errors) == 0

    def test_output_col_type_changed_is_warning(self) -> None:
        old_spec = self._fv_with_features([("event", "StringType")])
        new_spec = self._fv_with_features([("event", "IntegerType")], version="V2")
        applied = _make_applied_object(old_spec)
        results = _check_column_evolution(new_spec, applied)
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "COLUMN_TYPE_CHANGED" for r in warnings)

    def test_output_col_removed_is_warning(self) -> None:
        old_spec = self._fv_with_features([("event", "StringType"), ("count", "IntegerType")])
        new_spec = self._fv_with_features([("event", "StringType")], version="V2")
        applied = _make_applied_object(old_spec)
        results = _check_column_evolution(new_spec, applied)
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "COLUMN_REMOVED" for r in warnings)

    def test_no_applied_object_no_results(self) -> None:
        spec = self._fv_with_features([("event", "StringType")])
        results = _check_column_evolution(spec, None)
        assert results == []

    def test_source_col_added_without_default_is_warning(self) -> None:
        old_src = _make_source_spec()
        new_src = {
            **old_src,
            "columns": [
                {"name": "user_id", "type": "StringType"},
                {"name": "event", "type": "StringType"},
                {"name": "new_field", "type": "StringType"},  # no default
            ],
        }
        applied = _make_applied_object(old_src)
        results = _check_column_evolution(new_src, applied)
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "COLUMN_ADDED" for r in warnings)


# ---------------------------------------------------------------------------
# _check_dependencies tests
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_missing_entity_join_key_is_error(self) -> None:
        fv_spec = _make_fv_spec(entity_cols=["nonexistent_id"])
        batch_names = {"click_features", "clicks"}
        results = _check_dependencies(fv_spec, batch_names, _empty_applied())
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "MISSING_ENTITY" for r in errors)

    def test_entity_join_key_resolved_from_batch(self) -> None:
        entity_spec = _make_entity_spec(join_key="user_id")
        fv_spec = _make_fv_spec(entity_cols=["user_id"])
        batch_names = {"user", "clicks", "click_features"}
        applied = _applied_with(entity_spec)
        results = _check_dependencies(fv_spec, batch_names, applied)
        errors = [r for r in results if r.code == "MISSING_ENTITY"]
        assert len(errors) == 0

    def test_missing_source_reference_is_error(self) -> None:
        fv_spec = _make_fv_spec(sources=[{"name": "missing_source", "source_type": "Stream"}])
        batch_names = {"click_features"}
        results = _check_dependencies(fv_spec, batch_names, _empty_applied())
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "MISSING_SOURCE" for r in errors)

    def test_fg_missing_feature_view_is_error(self) -> None:
        fg_spec = {
            "kind": "FeatureGroup",
            "name": "my_group",
            "database": "DB",
            "schema": "SCH",
            "feature_views": [{"name": "nonexistent_fv"}],
        }
        results = _check_dependencies(fg_spec, {"my_group"}, _empty_applied())
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "MISSING_FEATURE_VIEW" for r in errors)

    def test_entity_resolved_from_applied_state(self) -> None:
        entity_spec = _make_entity_spec(join_key="user_id")
        applied = _applied_with(entity_spec)
        fv_spec = _make_fv_spec(entity_cols=["user_id"])
        batch_names = {"click_features", "clicks"}
        results = _check_dependencies(fv_spec, batch_names, applied)
        errors = [r for r in results if r.code == "MISSING_ENTITY"]
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# _check_destructive tests
# ---------------------------------------------------------------------------


class TestCheckDestructive:
    def test_expression_change_is_error_without_allow_recreate(self) -> None:
        old_spec = _make_fv_spec(
            features=[
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event_upper", "type": "StringType"},
                    "function": "upper",
                }
            ]
        )
        new_spec = _make_fv_spec(
            version="V2",
            features=[
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event_upper", "type": "StringType"},
                    "function": "lower",  # changed expression
                }
            ],
        )
        applied = _make_applied_object(old_spec)
        results = _check_destructive(new_spec, applied)
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "DESTRUCTIVE_CHANGE" for r in errors)

    def test_no_change_no_destructive_error(self) -> None:
        spec = _make_fv_spec()
        applied = _make_applied_object(spec)
        results = _check_destructive(spec, applied)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) == 0

    def test_no_applied_no_destructive_error(self) -> None:
        spec = _make_fv_spec()
        results = _check_destructive(spec, None)
        assert results == []


# ---------------------------------------------------------------------------
# _check_state_sync tests
# ---------------------------------------------------------------------------


class TestCheckStateSync:
    def test_object_at_unexpected_version_is_error(self) -> None:
        # Spec says V2, but applied state has V3 (concurrent modification)
        spec = _make_fv_spec(version="V2")
        applied_spec = _make_fv_spec(version="V3")
        applied = _make_applied_object(applied_spec)
        # Simulate state drift: applied version doesn't match what we expected
        results = _check_state_sync(spec, applied)
        # STATE_DRIFT if applied has a NEWER version than spec (concurrent write)
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "STATE_DRIFT" for r in errors)

    def test_matching_version_no_drift(self) -> None:
        spec = _make_fv_spec(version="V1")
        applied = _make_applied_object(spec)
        results = _check_state_sync(spec, applied)
        errors = [r for r in results if r.code == "STATE_DRIFT"]
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# validate_specs integration tests
# ---------------------------------------------------------------------------


class TestValidateSpecs:
    def test_returns_empty_for_valid_batch(self) -> None:
        entity_spec = _make_entity_spec()
        source_spec = _make_source_spec()
        fv_spec = _make_fv_spec(
            version="V1",
            entity_cols=["user_id"],
            sources=[{"name": "clicks", "source_type": "Stream"}],
        )
        batch = _batch(entity_spec, source_spec, fv_spec)
        applied = AppliedState(objects={})
        results = validate_specs(batch, applied)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) == 0

    def test_missing_version_produces_error(self) -> None:
        fv_spec = _make_fv_spec(version=None)
        batch = _batch(fv_spec)
        applied = _empty_applied()
        results = validate_specs(batch, applied)
        errors = [r for r in results if r.severity == "ERROR"]
        assert len(errors) >= 1

    def test_second_apply_same_spec_produces_no_change(self) -> None:
        spec = _make_fv_spec(version="V1")
        batch = _batch(spec)
        applied = _applied_with(spec)
        results = validate_specs(batch, applied)
        no_change = [r for r in results if r.code == "NO_CHANGE"]
        assert len(no_change) >= 1

    def test_version_conflict_produces_error(self) -> None:
        spec = _make_fv_spec(version="V1")
        deployed_spec = _make_fv_spec(version="V2")
        applied = _applied_with(deployed_spec)
        # Ensure hash is different so idempotency doesn't trigger
        applied.objects[spec_key(_normalize(deployed_spec))].content_hash = "different"
        batch = _batch(spec)
        results = validate_specs(batch, applied)
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "VERSION_CONFLICT" for r in errors)

    def test_column_added_is_warning(self) -> None:
        old_spec = _make_fv_spec(
            version="V1",
            features=[
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event", "type": "StringType"},
                }
            ],
        )
        new_spec = _make_fv_spec(
            version="V2",
            features=[
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event", "type": "StringType"},
                },
                {
                    "source_column": {"name": "new_col", "type": "StringType"},
                    "output_column": {"name": "new_col", "type": "StringType"},
                },  # no default
            ],
        )
        applied = _applied_with(old_spec)
        batch = _batch(new_spec)
        results = validate_specs(batch, applied)
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "COLUMN_ADDED" for r in warnings)

    def test_state_drift_detection(self) -> None:
        """Object exists at unexpected (higher) version → StateDrift ERROR."""
        spec = _make_fv_spec(version="V2")
        applied_spec = _make_fv_spec(version="V3")
        applied = _applied_with(applied_spec)
        # Different content to avoid idempotency skip
        applied.objects[spec_key(_normalize(applied_spec))].content_hash = "something_else"
        batch = _batch(spec)
        results = validate_specs(batch, applied)
        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "STATE_DRIFT" for r in errors)


# ---------------------------------------------------------------------------
# _check_database_schema_mismatch tests
# ---------------------------------------------------------------------------


class TestCheckDatabaseSchemaMismatch:
    def test_spec_with_different_database_emits_warning(self) -> None:
        """Spec with database != target_database produces a DB_MISMATCH warning."""
        spec = _make_fv_spec()
        spec["database"] = "OTHER_DB"
        results = _check_database_schema_mismatch([spec], "MY_DB", "MY_SCHEMA")
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "DB_MISMATCH" for r in warnings)

    def test_spec_with_different_schema_emits_warning(self) -> None:
        """Spec with schema != target_schema produces a SCHEMA_MISMATCH warning."""
        spec = _make_fv_spec()
        spec["schema"] = "OTHER_SCHEMA"
        results = _check_database_schema_mismatch([spec], "DB", "MY_SCHEMA")
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "SCHEMA_MISMATCH" for r in warnings)

    def test_spec_schema_field_alias_checked(self) -> None:
        """schema_ (Pydantic alias) is also checked for mismatch."""
        spec = _make_fv_spec()
        spec.pop("schema", None)
        spec["schema_"] = "WRONG_SCHEMA"
        results = _check_database_schema_mismatch([spec], "DB", "MY_SCHEMA")
        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "SCHEMA_MISMATCH" for r in warnings)

    def test_spec_matching_target_no_warning(self) -> None:
        """Spec whose database/schema match the target produces no mismatch warnings."""
        spec = _make_fv_spec()
        spec["database"] = "MY_DB"
        spec["schema"] = "MY_SCHEMA"
        results = _check_database_schema_mismatch([spec], "MY_DB", "MY_SCHEMA")
        mismatch = [r for r in results if r.code in ("DB_MISMATCH", "SCHEMA_MISMATCH")]
        assert len(mismatch) == 0

    def test_spec_without_database_field_no_warning(self) -> None:
        """Spec with no database field does not produce a DB_MISMATCH warning."""
        spec = _make_fv_spec()
        spec.pop("database", None)
        results = _check_database_schema_mismatch([spec], "MY_DB", "MY_SCHEMA")
        db_warnings = [r for r in results if r.code == "DB_MISMATCH"]
        assert len(db_warnings) == 0

    def test_empty_target_no_warning(self) -> None:
        """When target_database/schema are empty, no mismatch warnings are produced."""
        spec = _make_fv_spec()
        spec["database"] = "SOME_DB"
        spec["schema"] = "SOME_SCHEMA"
        results = _check_database_schema_mismatch([spec], "", "")
        mismatch = [r for r in results if r.code in ("DB_MISMATCH", "SCHEMA_MISMATCH")]
        assert len(mismatch) == 0

    def test_multiple_specs_each_checked(self) -> None:
        """All specs in the batch are checked for mismatches."""
        spec1 = _make_fv_spec(name="fv1")
        spec2 = _make_fv_spec(name="fv2")
        spec1["database"] = "WRONG_DB"
        spec2["database"] = "WRONG_DB"
        results = _check_database_schema_mismatch([spec1, spec2], "MY_DB", "MY_SCHEMA")
        warnings = [r for r in results if r.code == "DB_MISMATCH"]
        assert len(warnings) == 2

    def test_validate_specs_with_target_passes_mismatch_warnings(self) -> None:
        """validate_specs with target_database/schema returns DB_MISMATCH warnings."""
        spec = _make_fv_spec(version="V1")
        spec["database"] = "OTHER_DB"
        batch = _batch(spec)
        results = validate_specs(
            batch,
            _empty_applied(),
            target_database="MY_DB",
            target_schema="MY_SCHEMA",
        )
        warnings = [r for r in results if r.code == "DB_MISMATCH"]
        assert len(warnings) >= 1

    def test_validate_specs_without_target_no_mismatch_warnings(self) -> None:
        """validate_specs without target args does not produce mismatch warnings."""
        spec = _make_fv_spec(version="V1")
        spec["database"] = "OTHER_DB"
        batch = _batch(spec)
        results = validate_specs(batch, _empty_applied())
        mismatch = [r for r in results if r.code in ("DB_MISMATCH", "SCHEMA_MISMATCH")]
        assert len(mismatch) == 0


# ---------------------------------------------------------------------------
# _check_source_compatibility tests
# ---------------------------------------------------------------------------


class TestCheckSourceCompatibility:
    """Tests for cross-object source-to-FV compatibility validation."""

    def setup_method(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_source_compatibility,
        )

        self._fn = _check_source_compatibility

    def _src_with_cols(self, name: str, cols: list[tuple[str, str]]) -> dict[str, Any]:
        """Build a StreamingSource spec with given (name, type) column pairs."""
        return {
            "kind": "StreamingSource",
            "name": name,
            "database": "DB",
            "schema": "SCH",
            "type": "REST",
            "columns": [{"name": n, "type": t} for n, t in cols],
        }

    def _fv_using_cols(self, src_name: str, src_cols: list[str]) -> dict[str, Any]:
        """Build a FV spec that references src_name and uses the given source columns."""
        features = [
            {
                "source_column": {"name": c, "type": "StringType"},
                "output_column": {"name": c, "type": "StringType"},
            }
            for c in src_cols
        ]
        return _make_fv_spec(
            sources=[{"name": src_name, "source_type": "Stream"}],
            features=features,
        )

    # ------------------------------------------------------------------
    # SOURCE_COLUMN_REMOVED
    # ------------------------------------------------------------------

    def test_removed_column_used_by_batch_fv_is_error(self) -> None:
        """Removing a source column that a batch FV uses → ERROR SOURCE_COLUMN_REMOVED."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("event", "StringType")])
        new_src = self._src_with_cols("clicks", [("user_id", "StringType")])  # event removed
        fv = self._fv_using_cols("clicks", ["event"])

        applied = _applied_with(old_src)
        batch_dicts = [_normalize(new_src), _normalize(fv)]
        results = self._fn(batch_dicts, applied)

        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "SOURCE_COLUMN_REMOVED" for r in errors)

    def test_removed_column_not_used_by_any_fv_no_source_compat_error(self) -> None:
        """Removing a source column no FV references → no SOURCE_COLUMN_REMOVED error."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("unused", "StringType")])
        new_src = self._src_with_cols("clicks", [("user_id", "StringType")])  # unused removed
        fv = self._fv_using_cols("clicks", ["user_id"])  # only uses user_id

        applied = _applied_with(old_src)
        batch_dicts = [_normalize(new_src), _normalize(fv)]
        results = self._fn(batch_dicts, applied)

        errors = [r for r in results if r.code == "SOURCE_COLUMN_REMOVED"]
        assert len(errors) == 0

    # ------------------------------------------------------------------
    # SOURCE_COLUMN_ADDED
    # ------------------------------------------------------------------

    def test_column_added_to_source_is_warning(self) -> None:
        """Adding a column to a source → WARNING SOURCE_COLUMN_ADDED."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("event", "StringType")])
        new_src = self._src_with_cols(
            "clicks",
            [("user_id", "StringType"), ("event", "StringType"), ("new_col", "IntegerType")],
        )

        applied = _applied_with(old_src)
        batch_dicts = [_normalize(new_src)]
        results = self._fn(batch_dicts, applied)

        warnings = [r for r in results if r.severity == "WARNING"]
        assert any(r.code == "SOURCE_COLUMN_ADDED" for r in warnings)

    # ------------------------------------------------------------------
    # SOURCE_COLUMN_TYPE_CHANGED
    # ------------------------------------------------------------------

    def test_column_type_changed_used_by_batch_fv_is_error(self) -> None:
        """Changing the type of a source column used by a batch FV → ERROR SOURCE_COLUMN_TYPE_CHANGED."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("event", "StringType")])
        new_src = self._src_with_cols(
            "clicks", [("user_id", "StringType"), ("event", "IntegerType")]
        )  # event type changed
        fv = self._fv_using_cols("clicks", ["event"])

        applied = _applied_with(old_src)
        batch_dicts = [_normalize(new_src), _normalize(fv)]
        results = self._fn(batch_dicts, applied)

        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "SOURCE_COLUMN_TYPE_CHANGED" for r in errors)

    def test_column_type_changed_not_used_by_fv_no_error(self) -> None:
        """Changing type of a column not referenced by any FV → no SOURCE_COLUMN_TYPE_CHANGED."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("unused", "StringType")])
        new_src = self._src_with_cols(
            "clicks", [("user_id", "StringType"), ("unused", "IntegerType")]
        )  # unused type changed
        fv = self._fv_using_cols("clicks", ["user_id"])  # only uses user_id

        applied = _applied_with(old_src)
        batch_dicts = [_normalize(new_src), _normalize(fv)]
        results = self._fn(batch_dicts, applied)

        errors = [r for r in results if r.code == "SOURCE_COLUMN_TYPE_CHANGED"]
        assert len(errors) == 0

    # ------------------------------------------------------------------
    # SOURCE_DELETED_WITH_DEPENDENTS
    # ------------------------------------------------------------------

    def test_source_deleted_fv_still_in_batch_is_error(self) -> None:
        """Source absent from batch but referenced by a batch FV → ERROR SOURCE_DELETED_WITH_DEPENDENTS."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("event", "StringType")])
        fv = self._fv_using_cols("clicks", ["event"])

        applied = _applied_with(old_src)
        # Batch has the FV but NOT the source
        batch_dicts = [_normalize(fv)]
        results = self._fn(batch_dicts, applied)

        errors = [r for r in results if r.severity == "ERROR"]
        assert any(r.code == "SOURCE_DELETED_WITH_DEPENDENTS" for r in errors)

    def test_source_deleted_no_dependents_no_error(self) -> None:
        """Source absent from batch with no referencing FVs → no SOURCE_DELETED_WITH_DEPENDENTS."""
        old_src = self._src_with_cols("orphan", [("user_id", "StringType")])
        # Batch FV references a different source, not orphan
        fv = self._fv_using_cols("other_source", ["user_id"])

        applied = _applied_with(old_src)
        batch_dicts = [_normalize(fv)]
        results = self._fn(batch_dicts, applied)

        errors = [r for r in results if r.code == "SOURCE_DELETED_WITH_DEPENDENTS"]
        assert len(errors) == 0

    # ------------------------------------------------------------------
    # New source / no applied state
    # ------------------------------------------------------------------

    def test_new_source_no_applied_no_errors(self) -> None:
        """Brand-new source (not in applied state) → no errors or warnings."""
        new_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("event", "StringType")])
        batch_dicts = [_normalize(new_src)]
        results = self._fn(batch_dicts, _empty_applied())
        assert results == []

    # ------------------------------------------------------------------
    # validate_specs integration
    # ------------------------------------------------------------------

    def test_validate_specs_calls_source_compatibility(self) -> None:
        """validate_specs surfaces SOURCE_COLUMN_REMOVED when a used column is removed."""
        old_src = self._src_with_cols("clicks", [("user_id", "StringType"), ("event", "StringType")])
        new_src = self._src_with_cols("clicks", [("user_id", "StringType")])  # event removed
        fv = self._fv_using_cols("clicks", ["event"])

        applied = _applied_with(old_src)
        batch = _batch(new_src, fv)
        results = validate_specs(batch, applied)

        errors = [r for r in results if r.code == "SOURCE_COLUMN_REMOVED"]
        assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Idempotency / VERSION_CONFLICT alignment with planner
# ---------------------------------------------------------------------------
#
# Pins the contract that ``_check_idempotency`` mirrors the planner's hash
# strategy when ``applied.from_specification == True`` (i.e. the FV came
# from ``DESCRIBE ... TYPE = SPECIFICATION``).  Without this, the validator
# always disagrees with the planner on full-spec FVs and emits a spurious
# ``VERSION_CONFLICT`` for an unchanged round-trip — see plan
# ``restore_export_plan_invariant``.


def _fv_round_trip_dict(
    *,
    udf_body: str = "def transform(x):\n    return x",
    name: str = "click_fv",
    version: str = "V1",
) -> dict[str, Any]:
    """Authoring-format FV dict suitable for both model_validate and compile_to_spec."""
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema_": "SCH",
        "version": version,
        "entities": ["user_id"],
        "sources": [{"name": "clicks", "source_type": "Stream"}],
        "feature_aggregation_method": "tiles",
        "features": [
            {
                "source_column": {"name": "event", "type": "StringType"},
                "output_column": {"name": "event", "type": "StringType"},
            }
        ],
        "udf": {
            "name": "transform",
            "engine": "pandas",
            "function_definition": udf_body,
            "output_columns": [{"name": "event", "type": "StringType"}],
        },
    }


def _applied_full_spec_state(deployed_dict: dict[str, Any]) -> AppliedState:
    """Build an AppliedState whose FV came from DESCRIBE ... TYPE = SPECIFICATION."""
    from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    compile_input = dict(deployed_dict)
    if "schema_" in compile_input:
        compile_input["schema"] = compile_input.pop("schema_")
    deployed_spec = compile_to_spec(compile_input, "DB", "SCH")
    key = spec_key({**compile_input, "kind": deployed_dict["kind"]})
    applied = AppliedObject(
        key=key,
        kind=deployed_dict["kind"],
        name=deployed_dict["name"],
        version=deployed_dict["version"],
        content_hash=_full_spec_hash(deployed_spec),
        spec_payload=deployed_spec,
        from_specification=True,
    )
    return AppliedState(objects={key: applied})


class TestCheckIdempotencyFullSpec:
    """``_check_idempotency`` must agree with ``planner.generate_plan``.

    When ``applied.from_specification`` is True and the kind is a FeatureView,
    the planner uses ``compute_local_spec_hash`` (full-spec compile) for the
    desired side and ``_full_spec_hash(spec_payload)`` for the applied side.
    The validator must use the same strategy so an unchanged FV reports
    ``NO_CHANGE`` rather than a spurious ``VERSION_CONFLICT``.
    """

    def test_no_version_conflict_when_full_spec_matches(self) -> None:
        """Identical local + deployed FV under from_specification path = NO_CHANGE only."""
        from snowflake.ml.feature_store.decl.spec_models import FeatureView

        deployed_dict = _fv_round_trip_dict()
        applied_state = _applied_full_spec_state(deployed_dict)
        local_fv = FeatureView.model_validate(deployed_dict)

        results = validate_specs(
            SpecBatch(specs=[local_fv]),
            applied_state,
            target_database="DB",
            target_schema="SCH",
        )

        codes = [r.code for r in results]
        assert (
            "VERSION_CONFLICT" not in codes
        ), f"VERSION_CONFLICT must not fire when content is identical; got results={results!r}"
        assert "NO_CHANGE" in codes, f"Expected NO_CHANGE; got results={results!r}"

    def test_no_version_conflict_when_db_schema_taken_from_spec_dict(self) -> None:
        """db/schema may be absent from kwargs; spec_dict fallback must still match planner."""
        from snowflake.ml.feature_store.decl.spec_models import FeatureView

        deployed_dict = _fv_round_trip_dict()
        applied_state = _applied_full_spec_state(deployed_dict)
        local_fv = FeatureView.model_validate(deployed_dict)

        results = validate_specs(SpecBatch(specs=[local_fv]), applied_state)

        codes = [r.code for r in results]
        assert "VERSION_CONFLICT" not in codes, (
            f"VERSION_CONFLICT must not fire when db/schema fallback succeeds; " f"got results={results!r}"
        )
        assert "NO_CHANGE" in codes


class TestContentChangedAtSameVersion:
    """When content changes at the same version the validator stays
    silent on VERSION_CONFLICT and the planner emits a destructive op.

    Prior to the planner re-validation fix
    (``plans/planner_revalidate_identical_spec.plan.md``) the validator
    fired ``VERSION_CONFLICT`` whenever ``version <= applied_version``.
    That semantics produced spurious errors on a re-plan of an
    *identical* spec whose hash drifted slightly between
    ``compile_to_spec(local)`` and the live ``DESCRIBE`` payload.  The
    fix tightens the rule to ``version < applied_version`` only, so
    same-version + different-content surfaces as a destructive
    ``RECREATE_FV`` plan op (gated by ``--allow-recreate``) rather
    than a validator error.
    """

    def test_no_version_conflict_when_content_differs_at_same_version(self) -> None:
        """Different UDF body, same version => no VERSION_CONFLICT.

        With the new semantics the validator does *not* emit
        VERSION_CONFLICT for equal versions; the planner / executor
        is responsible for surfacing the destructive recreate.
        ``NO_CHANGE`` must also be absent (content actually differs).
        """
        from snowflake.ml.feature_store.decl.spec_models import FeatureView

        deployed_dict = _fv_round_trip_dict(udf_body="def transform(x):\n    return x")
        applied_state = _applied_full_spec_state(deployed_dict)

        local_dict = _fv_round_trip_dict(udf_body="def transform(x):\n    return x.upper()")
        local_fv = FeatureView.model_validate(local_dict)

        results = validate_specs(
            SpecBatch(specs=[local_fv]),
            applied_state,
            target_database="DB",
            target_schema="SCH",
        )

        codes = [r.code for r in results]
        assert "VERSION_CONFLICT" not in codes, (
            f"VERSION_CONFLICT must not fire on equal versions even when content " f"differs; got results={results!r}"
        )
        assert "NO_CHANGE" not in codes, (
            f"NO_CHANGE must not appear when content actually differs; " f"got results={results!r}"
        )

    def test_planner_emits_recreate_fv_when_content_differs_at_same_version(self) -> None:
        """Same-version + different content surfaces as destructive RECREATE_FV.

        The destructive flag forces an ``--allow-recreate`` opt-in at
        apply time and serves the friction role that ``VERSION_CONFLICT``
        used to play, but without false-positives on hash drift.
        """
        from snowflake.ml.feature_store.decl.enums import OpKind
        from snowflake.ml.feature_store.decl.planner import generate_plan
        from snowflake.ml.feature_store.decl.spec_models import FeatureView
        from snowflake.ml.feature_store.decl.types import PlanOptions

        deployed_dict = _fv_round_trip_dict(udf_body="def transform(x):\n    return x")
        applied_state = _applied_full_spec_state(deployed_dict)

        local_dict = _fv_round_trip_dict(udf_body="def transform(x):\n    return x.upper()")
        local_fv = FeatureView.model_validate(local_dict)

        plan = generate_plan(
            SpecBatch(specs=[local_fv]),
            applied_state,
            PlanOptions(),
            database="DB",
            schema="SCH",
        )

        recreate_ops = [op for op in plan.ops if op.kind == OpKind.RECREATE_FV]
        assert len(recreate_ops) == 1, (
            f"Expected exactly one RECREATE_FV op for same-version content "
            f"change; got ops={[(o.kind, o.name) for o in plan.ops]!r}"
        )
        assert recreate_ops[0].destructive is True


# ---------------------------------------------------------------------------
# BatchFV hash parity — lossy SPECIFICATION round-trip
#
# Background.  ``DESCRIBE ONLINE FEATURE TABLE … TYPE = SPECIFICATION``
# returns a BatchFV spec with ``spec.sources = []`` and ``spec.features``
# populated with 1:1 ``source_column == output_column`` pass-throughs
# auto-derived from the source columns.  The local-compile
# (:func:`spec_compiler.compile_to_spec`) emits the opposite shape:
# ``sources = [{name, table, columns}]`` and ``features = []`` for an
# authoring YAML without an explicit ``features:`` block (the BUG_BASH §5
# non-tiled BatchFV shape).  Without normalisation the two hashes always
# differ even on a clean round-trip, and the planner can't distinguish
# a schedule-only edit (UPDATE_FV) from a source-table swap (RECREATE_FV).
#
# Fix: in :func:`_full_spec_hash`, for FeatureView kinds, normalise
# ``spec.sources`` to a sorted ``[{table}]`` projection (drop the
# unstable ``name`` / ``columns`` / ``source_type`` shapes) and strip
# auto-derived 1:1 pass-through features.  Both sides go through the
# same normalisation, so any divergence remaining after normalisation
# is a real semantic change.  Detailed motivation: docs/BATCH_FV_BUG_BASH.md.
# ---------------------------------------------------------------------------


def _bugbash_local_batch_fv_dict() -> dict[str, Any]:
    """Mirror the docs/BATCH_FV_BUG_BASH.md §5 authoring shape (post-compile)."""
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_BATCH_DECL",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "METRIC_VAL", "type": "FloatType"},
                    ],
                }
            ],
            "features": [],
            "target_lag_sec": 60,
        },
        "online_store_type": "postgres",
    }


def _bugbash_applied_batch_fv_dict() -> dict[str, Any]:
    """Mirror the DESCRIBE TYPE = SPECIFICATION payload for the same FV.

    Post-:func:`_inject_batch_fv_source_from_dt_text` the deployed spec
    has the table recovered into ``sources[0]``; everything else stays
    in the lossy round-trip shape (auto-derived features).

    Returns:
        The applied-side batch FV dict matching the BUG_BASH §5 round-trip
        shape after dynamic-table source injection.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
            "client_version": "1.38.0",
            "spec_format_version": "1",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {"name": "RAW_EVENTS_BATCH_DECL", "source_type": "Batch", "table": "RAW_EVENTS_BATCH_DECL"},
            ],
            "features": [
                {
                    "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                    "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
                },
                {
                    "output_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                    "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                },
            ],
            "target_lag_sec": 60,
        },
        "online_store_type": "postgres",
    }


class TestBatchFvFullSpecHashParity:
    def test_local_and_applied_round_trip_share_full_spec_hash(self) -> None:
        """A locally-compiled BatchFV (BUG_BASH §5 shape) and its DT-injected
        applied counterpart MUST hash to the same value after normalisation —
        otherwise the planner falsely emits ``RECREATE_FV`` for unedited
        round trips and ``UPDATE_FV`` is unreachable for §7 schedule edits.
        """
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        local = _bugbash_local_batch_fv_dict()
        applied = _bugbash_applied_batch_fv_dict()
        assert _full_spec_hash(local) == _full_spec_hash(applied), (
            "BatchFV hash parity: local ``features=[] / sources=[{table=…}]`` "
            "and applied ``features=[auto-derived 1:1] / sources=[{table=…}]`` "
            "must normalise to the same hash so the planner can tell "
            "schedule-only edits apart from source-table swaps."
        )

    def test_hash_differs_when_source_table_swaps(self) -> None:
        """Swapping ``sources[0].table`` (BUG_BASH §8) MUST still bump the
        hash even after normalisation — otherwise the planner emits
        ``NO_CHANGE`` for a destructive source switch.
        """
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        local = _bugbash_local_batch_fv_dict()
        applied = _bugbash_applied_batch_fv_dict()
        local["spec"]["sources"][0]["table"] = "RAW_EVENTS_BATCH_DECL_V2"
        assert _full_spec_hash(local) != _full_spec_hash(applied), (
            "BatchFV hash parity: a ``sources[0].table`` swap must NOT be "
            "absorbed by normalisation — the planner relies on this hash "
            "diff (combined with structural-equivalent=False) to emit "
            "``RECREATE_FV`` for source-table swaps (BUG_BASH §8)."
        )

    def test_auto_derived_features_are_normalised_away(self) -> None:
        """A ``source_column == output_column`` feature is auto-derived
        on the applied side and absent on the local side; the
        normalisation must drop it from BOTH sides so the comparison
        focuses on the structural binding (sources + entities +
        aggregation) rather than on snowml-core's per-create
        spec-payload defaults.
        """
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        local = _bugbash_local_batch_fv_dict()
        local_with_passthrough = copy.deepcopy(local)
        local_with_passthrough["spec"]["features"] = [
            {
                "output_column": {"name": "EVENT_TS", "type": "TimestampType"},
                "source_column": {"name": "EVENT_TS", "type": "TimestampType"},
            },
        ]
        assert _full_spec_hash(local) == _full_spec_hash(local_with_passthrough), (
            "Auto-derived 1:1 features must not contribute to the FV's "
            "full-spec hash — they are equivalent to ``features: []`` "
            "semantically (snowml-core auto-derives them at CREATE)."
        )

    def test_explicit_aggregated_features_still_bump_hash(self) -> None:
        """An explicit aggregation feature (output_column != source_column)
        is NOT auto-derived and MUST contribute to the hash — otherwise
        the planner misses real changes to tiled BatchFV semantics.
        """
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        local = _bugbash_local_batch_fv_dict()
        local_tiled = copy.deepcopy(local)
        local_tiled["spec"]["features"] = [
            {
                "output_column": {"name": "METRIC_VAL_AVG_1H", "type": "DoubleType"},
                "source_column": {"name": "METRIC_VAL", "type": "DoubleType"},
                "feature_aggregation_method": "AVG",
            },
        ]
        assert _full_spec_hash(local) != _full_spec_hash(local_tiled), (
            "Tiled-aggregation features MUST bump the FV hash so the "
            "planner emits ``RECREATE_FV`` when the aggregation surface "
            "changes — they are NOT 1:1 pass-throughs."
        )

    def test_batch_feature_view_structural_equivalent_handles_lossy_roundtrip(self) -> None:
        """The planner's UPDATE_FV gate (``structural_equivalent``) MUST
        return True for the BUG_BASH §7 shape — schedule edits keep the
        FV's structural projection identical to the applied one even
        though the deployed SPECIFICATION lost ``sources`` and stamped
        auto-derived features.
        """
        from snowflake.ml.feature_store.decl.invariants import (
            batch_feature_view_structural_equivalent,
        )

        local_authoring = {
            "kind": "BatchFeatureView",
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
            "online": True,
            "target_lag": "1 minute",
            "refresh_freq": "2 minutes",  # the §7 edit
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_BATCH_DECL",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "METRIC_VAL", "type": "FloatType"},
                    ],
                }
            ],
        }
        applied_payload = _bugbash_applied_batch_fv_dict()
        assert batch_feature_view_structural_equivalent(local_authoring, applied_payload, "JKEW_DB", "JKEW_SCHEMA"), (
            "§7 schedule-only edit must be classified structurally "
            "equivalent so the planner emits UPDATE_FV — the lossy "
            "DESCRIBE round-trip is not a real structural change."
        )

    def test_batch_feature_view_structural_equivalent_detects_table_swap(self) -> None:
        """§8 ``BatchSource.table`` swap MUST NOT be hidden by the
        parity tolerance — local ``sources[0].table = V2`` against
        applied ``sources[0].table = V1`` must surface as a structural
        difference so the planner emits ``RECREATE_FV``.
        """
        from snowflake.ml.feature_store.decl.invariants import (
            batch_feature_view_structural_equivalent,
        )

        local_authoring = {
            "kind": "BatchFeatureView",
            "name": "MY_BATCH_FV_BATCH_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
            "online": True,
            "target_lag": "1 minute",
            "refresh_freq": "1 minute",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "EVENTS_BATCH_DECL",
                    "source_type": "Batch",
                    "table": "RAW_EVENTS_BATCH_DECL_V2",  # the §8 edit
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                        {"name": "METRIC_VAL", "type": "FloatType"},
                    ],
                }
            ],
        }
        applied_payload = _bugbash_applied_batch_fv_dict()  # still V1
        assert not batch_feature_view_structural_equivalent(
            local_authoring, applied_payload, "JKEW_DB", "JKEW_SCHEMA"
        ), (
            "§8 source-table swap must surface as a structural change "
            "so the planner emits RECREATE_FV; otherwise an unintended "
            "UPDATE_FV silently keeps the OFT bound to the old table."
        )


class TestBackfillStrippedFromHash:
    """``backfill`` is operational metadata, not structural identity.

    The FV-level ``backfill: {table, start_time, overwrite, initialize}``
    block influences how the imperative library performs registration but
    does NOT change the FV's deployed schema, source binding, or feature
    semantics.  Stripping it from both ``_full_spec_hash`` and
    ``structural_fingerprint_hash`` keeps idempotent re-applies as
    ``NO_CHANGE`` and lets the planner surface backfill intent as a
    separate destructive flag instead of a structural diff.
    """

    def _streaming_fv_authoring(self, **extras: Any) -> Any:
        base = {
            "kind": "StreamingFeatureView",
            "name": "USER_CLICK_BACKFILL_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TYPE", "type": "StringType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [],
        }
        base.update(extras)
        return base

    def _batch_fv_authoring(self, **extras: Any) -> Any:
        base = {
            "kind": "BatchFeatureView",
            "name": "ORDERS_TOTAL_DECL",
            "version": "V1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "ORDERS_DECL",
                    "source_type": "Batch",
                    "table": "RAW_ORDERS_DECL",
                }
            ],
            "features": [],
        }
        base.update(extras)
        return base

    def test_streaming_fv_backfill_block_does_not_change_full_spec_hash(self) -> None:
        """A streaming FV with ``backfill: {table: ..., start_time: ...}`` must
        hash identically to one without that block — the deployed FV's
        ``DESCRIBE … TYPE = SPECIFICATION`` payload never carries backfill
        config, so a local YAML that adds it would otherwise drift forever."""
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        without = {"kind": "StreamingFeatureView", "spec": self._streaming_fv_authoring()}
        with_block = {
            "kind": "StreamingFeatureView",
            "spec": self._streaming_fv_authoring(
                backfill={
                    "table": "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL",
                    "start_time": "2026-05-18T22:00:00",
                }
            ),
        }
        assert _full_spec_hash(without) == _full_spec_hash(with_block), (
            "FV-level backfill block is operational, not structural — adding "
            "it to a streaming FV must not change ``_full_spec_hash`` so "
            "round-trips against a backfill-less DESCRIBE payload stay NO_CHANGE."
        )

    def test_batch_fv_backfill_overwrite_does_not_change_full_spec_hash(self) -> None:
        """``backfill.overwrite=True`` is the operator opt-in for a forced
        re-materialisation — it must not bump the structural identity hash;
        the planner surfaces it through the destructive flag on the plan op
        rather than through hash-based diff detection."""
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        without = {"kind": "BatchFeatureView", "spec": self._batch_fv_authoring()}
        with_block = {
            "kind": "BatchFeatureView",
            "spec": self._batch_fv_authoring(backfill={"overwrite": True}),
        }
        assert _full_spec_hash(without) == _full_spec_hash(with_block)

    def test_batch_fv_backfill_initialize_does_not_change_full_spec_hash(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import _full_spec_hash

        without = {"kind": "BatchFeatureView", "spec": self._batch_fv_authoring()}
        with_init = {
            "kind": "BatchFeatureView",
            "spec": self._batch_fv_authoring(backfill={"initialize": "ON_SCHEDULE"}),
        }
        assert _full_spec_hash(without) == _full_spec_hash(with_init)

    def test_structural_fingerprint_strips_backfill(self) -> None:
        """Same contract for ``structural_fingerprint_hash`` — the legacy
        diff path used for non-FROM-SPECIFICATION applied state must also
        ignore backfill, otherwise older stores see false-positive diffs."""
        from snowflake.ml.feature_store.decl.invariants import (
            structural_fingerprint_hash,
        )

        without = self._batch_fv_authoring()
        with_block = self._batch_fv_authoring(backfill={"overwrite": True, "initialize": "ON_CREATE"})
        assert structural_fingerprint_hash(without) == structural_fingerprint_hash(with_block)


# ---------------------------------------------------------------------------
# B5 — Source-diff columns canonicalization (Bug 1 closure)
# ---------------------------------------------------------------------------


class TestSourceDiffColumnsCanonicalization:
    """Pin the Bug 1 hash-equivalence contract for ``compute_source_diff_kind``.

    Bug 1 in the metadata-roundtrip plan: a clean apply followed by a
    plan re-emits ``RECREATE_SOURCE`` for a ``BatchSource`` whose local
    YAML carries an authoritative ``columns:`` block but whose applied
    Datasource (derived from the FV's ``spec.sources[]`` via the legacy
    DT-text recovery path) ends up with an empty ``columns: []`` because
    that recovery path never recovers per-column schema for a
    table-backed FV.

    The fix is two-fold inside ``compute_source_diff_kind``:

    1. When the applied side has no recovered columns (empty list /
       missing key), drop ``columns`` from BOTH sides of the structural
       hash.  We can't compare what we don't know, and forcing a
       recreate on a clean apply is the regression Bug 1 named.
    2. When both sides do have columns, run them through a shared
       :func:`_canonicalize_columns` helper before hashing — sort by
       name, strip whitespace from type strings — so cosmetic
       reorderings or whitespace drift don't bump the hash.

    Each test below pins one of the three documented behaviours and
    is flagged in the plan's ``B5`` red-test list.
    """

    def _local_batch_source(self, *, columns: list[dict[str, Any]]) -> dict[str, Any]:
        """Authoring-side BatchSource payload (Pydantic-shape dict)."""
        return {
            "kind": "BatchSource",
            "name": "EVENTS_BATCH_DECL",
            "database": "DB",
            "schema": "SCH",
            "table": "RAW_EVENTS_BATCH_DECL",
            "source_type": "Batch",
            "columns": columns,
        }

    def _applied_batch_datasource(self, *, columns: list[dict[str, Any]]) -> dict[str, Any]:
        """Applied-side Datasource payload (state.py-shape dict).

        Mirrors what ``state._datasource_objects_from_specs`` produces
        for an FV-derived BatchSource: ``kind: Datasource``,
        ``source_type: Batch``, ``table: <physical>``, and possibly
        empty ``columns: []`` when the legacy DT-text recovery path
        could not enumerate per-column schema.

        Args:
            columns: ``list[dict]`` to attach to the recovered
                Datasource payload — fixtures pass the full
                authored schema for the FV_SOURCE_REFS path and
                ``[]`` for the legacy DT-text path so the canonical
                drop-empty-applied-columns rule can be exercised.

        Returns:
            A Datasource payload dict in the shape emitted by
            :func:`state._datasource_objects_from_specs`.
        """
        return {
            "kind": "Datasource",
            "name": "EVENTS_BATCH_DECL",
            "database": "DB",
            "schema": "SCH",
            "table": "RAW_EVENTS_BATCH_DECL",
            "source_type": "Batch",
            "columns": columns,
        }

    def test_empty_applied_columns_with_full_local_does_not_force_recreate(self) -> None:
        """Bug 1 regression pin.

        Local YAML carries the authored columns; applied Datasource
        carries empty ``columns: []`` (the legacy DT-text recovery
        path's typical shape for a table-backed BatchSource).  The
        diff helper MUST NOT return ``"recreate"`` — the columns
        on the applied side are unknown, not different.

        Closes the Bug 1 cascade in
        ``plans/metadata-roundtrip-limitations_fe945c22.plan.md``.
        """
        from snowflake.ml.feature_store.decl.invariants import compute_source_diff_kind

        local = self._local_batch_source(
            columns=[
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
                {"name": "METRIC_VAL", "type": "FloatType"},
            ],
        )
        applied = self._applied_batch_datasource(columns=[])
        result = compute_source_diff_kind(local, applied)
        assert result != "recreate", (
            "compute_source_diff_kind must NOT force a recreate when the "
            "applied side has empty columns (legacy DT-text recovery never "
            "recovers per-column schema for table-backed BatchSources). "
            f"Got result={result!r} but expected 'no_change' (or "
            "'update_desc_only' if a description drift is also present)."
        )
        assert result == "no_change"

    def test_columns_sorted_by_name_invariant(self) -> None:
        """Local and applied carry the same columns in different order.

        The ``_canonicalize_columns`` helper must sort by name so a
        YAML reorder (or a runtime that returns columns in CREATE-
        order vs alphabetical) does not bump the structural hash.
        """
        from snowflake.ml.feature_store.decl.invariants import compute_source_diff_kind

        local_cols = [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
            {"name": "METRIC_VAL", "type": "FloatType"},
        ]
        applied_cols = [
            {"name": "METRIC_VAL", "type": "FloatType"},
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
        ]
        local = self._local_batch_source(columns=local_cols)
        applied = self._applied_batch_datasource(columns=applied_cols)
        result = compute_source_diff_kind(local, applied)
        assert result == "no_change", (
            "Column reordering must not bump the structural hash. "
            "_canonicalize_columns must sort by name before hashing. "
            f"Got result={result!r}."
        )

    def test_column_type_strings_stripped(self) -> None:
        """Type strings differ only in surrounding whitespace.

        A runtime that returns ``"  StringType  "`` (e.g. extra
        whitespace from a verbose DESCRIBE row formatter) must hash
        identically to a local ``"StringType"`` — _canonicalize_columns
        must strip whitespace from type strings.
        """
        from snowflake.ml.feature_store.decl.invariants import compute_source_diff_kind

        local = self._local_batch_source(
            columns=[
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
            ],
        )
        applied = self._applied_batch_datasource(
            columns=[
                {"name": "USER_ID", "type": "  StringType  "},
                {"name": "EVENT_TS", "type": "TimestampType\t"},
            ],
        )
        result = compute_source_diff_kind(local, applied)
        assert result == "no_change", (
            "Whitespace differences in column type strings must not bump "
            "the structural hash. _canonicalize_columns must strip "
            f"whitespace from type strings. Got result={result!r}."
        )


# ---------------------------------------------------------------------------
# B5 — Operational-drift canonicalization (Bug 2 closure)
# ---------------------------------------------------------------------------


class TestOperationalCanonicalization:
    """Pin the Bug 2 closure: ``online_store_type`` and ``target_lag_sec``
    canonicalize to the same form on both halves of the operational-drift
    comparison surface.

    Bug 2 in the metadata-roundtrip plan: a tiled online BatchFV emits
    a phantom ``UPDATE_FV`` on the first re-plan after a clean apply
    because the local-compile and applied-state paths disagree on
    cosmetic representation of two operational fields:

    * ``online_store_type`` — Snowflake stamps mixed-case enum values
      (e.g. ``"POSTGRES"``) into the deployed SPECIFICATION while the
      local-compile emits the lowercase form (``"postgres"``).
    * ``target_lag_sec`` — the OFT sometimes returns the cadence as a
      string (``"5 minutes"``) and sometimes as ``"300 SECONDS"``;
      diffing on seconds (not strings) makes the comparison stable.

    This test class pins the contract by exercising the lazy-imported
    ``_canonicalize_operational_fields`` helper from snowml-core via
    the decl-side wrapper.  The helper is the single source of truth
    for the canonical form on both sides of the operational-drift
    diff.
    """

    def test_online_store_type_canonical_form(self) -> None:
        """``"POSTGRES"`` and ``"postgres"`` canonicalize to the same form.

        Pins the Bug 2 closure for ``online_store_type`` — the
        operational-drift comparison surface in
        :mod:`decl.invariants` must match A4's canonical form
        (lowercased + whitespace-stripped).
        """
        from snowflake.ml.feature_store.decl.invariants import (
            _canonicalize_operational_for_drift,
        )

        upper = _canonicalize_operational_for_drift({"online_store_type": "POSTGRES"})
        lower = _canonicalize_operational_for_drift({"online_store_type": "postgres"})
        spaced = _canonicalize_operational_for_drift({"online_store_type": "  POSTGRES  "})
        assert upper["online_store_type"] == lower["online_store_type"], (
            "Mixed-case `online_store_type` must canonicalize to the "
            "same form as the lowercase variant. "
            f"upper={upper!r} lower={lower!r}."
        )
        assert spaced["online_store_type"] == lower["online_store_type"], (
            "Whitespace-padded `online_store_type` must canonicalize to "
            "the trimmed lowercase form. "
            f"spaced={spaced!r} lower={lower!r}."
        )

    def test_target_lag_sec_canonical_form(self) -> None:
        """``target_lag: "5 minutes"`` and ``target_lag_sec: 300`` produce
        the same canonical seconds value.

        Pins the Bug 2 closure for ``target_lag_sec`` — the
        operational-drift comparison surface must reduce both
        representations to ``int(seconds)`` so the diff is stable.
        """
        from snowflake.ml.feature_store.decl.invariants import (
            _canonicalize_operational_for_drift,
        )

        from_string = _canonicalize_operational_for_drift({"target_lag": "5 minutes"})
        from_seconds_string = _canonicalize_operational_for_drift({"target_lag": "300 SECONDS"})

        assert from_string.get("target_lag_sec") == 300, (
            "_canonicalize_operational_for_drift must reduce "
            "`target_lag: '5 minutes'` to `target_lag_sec: 300`. "
            f"Got {from_string!r}."
        )
        assert from_seconds_string.get("target_lag_sec") == 300, (
            "_canonicalize_operational_for_drift must reduce "
            "`target_lag: '300 SECONDS'` to `target_lag_sec: 300`. "
            f"Got {from_seconds_string!r}."
        )
        assert from_string["target_lag_sec"] == from_seconds_string["target_lag_sec"], (
            "Two different string representations of the same lag must "
            "canonicalize to the same int seconds. "
            f"from_string={from_string!r} from_seconds_string={from_seconds_string!r}."
        )


# ---------------------------------------------------------------------------
# B5 — BatchFV structural inner key list
# ---------------------------------------------------------------------------


class TestBatchFvStructuralKeys:
    """Pin the structural-inner key list contract.

    The planner uses ``_BATCH_FV_STRUCTURAL_INNER_KEYS`` to decide which
    inner-spec edits should fall through to ``RECREATE_FV``.  Phase A
    (snowml-core ``947c4bf42``) introduced ``aggregation_secondary_keys``
    on ``AggregationMetadata``; per L1 in the metadata-roundtrip plan,
    edits to that field are structural and must route through
    ``RECREATE_FV`` rather than ``UPDATE_FV``.
    """

    def test_aggregation_secondary_keys_in_structural_inner_set(self) -> None:
        """``aggregation_secondary_keys`` is a structural key (L1)."""
        from snowflake.ml.feature_store.decl.invariants import (
            _BATCH_FV_STRUCTURAL_INNER_KEYS,
        )

        assert "aggregation_secondary_keys" in _BATCH_FV_STRUCTURAL_INNER_KEYS, (
            "_BATCH_FV_STRUCTURAL_INNER_KEYS must include "
            "`aggregation_secondary_keys` so secondary-key edits flow "
            "through RECREATE_FV (Phase B5 / L1 closure). Currently: "
            f"{sorted(_BATCH_FV_STRUCTURAL_INNER_KEYS)}"
        )


# ---------------------------------------------------------------------------
# Incomplete aggregation features (must fail at plan time)
# ---------------------------------------------------------------------------


def _agg_feature(**overrides: Any) -> dict[str, Any]:
    feat: dict[str, Any] = {
        "function": "sum",
        "window_sec": 3600,
        "source_column": {"name": "AMOUNT", "type": "DoubleType"},
        "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
    }
    feat.update(overrides)
    return feat


class TestFeatureIncomplete:
    """A features[] row that looks like an aggregation must carry function,
    source_column, and integer window_sec.  Silent drops at apply time
    used to deploy a shorter FV than the spec declared.
    """

    def _validate(self, features: list[dict[str, Any]]) -> list[Any]:
        entity_spec = _make_entity_spec()
        source_spec = _make_source_spec()
        fv_spec = _make_fv_spec(features=features)
        batch = _batch(entity_spec, source_spec, fv_spec)
        return validate_specs(batch, _empty_applied())

    def _incomplete_codes(self, features: list[dict[str, Any]]) -> list[str]:
        results = self._validate(features)
        return [r.code for r in results if r.code == "FEATURE_INCOMPLETE"]

    def test_bare_numeric_window_string_is_error(self) -> None:
        feat = _agg_feature()
        feat.pop("window_sec")
        feat["window"] = "300"
        assert self._incomplete_codes([feat]) == ["FEATURE_INCOMPLETE"]

    def test_fractional_window_string_is_error(self) -> None:
        feat = _agg_feature()
        feat.pop("window_sec")
        feat["window"] = "1.5m"
        assert self._incomplete_codes([feat]) == ["FEATURE_INCOMPLETE"]

    def test_missing_window_is_error(self) -> None:
        feat = _agg_feature()
        feat.pop("window_sec")
        assert self._incomplete_codes([feat]) == ["FEATURE_INCOMPLETE"]

    def test_missing_function_is_error(self) -> None:
        feat = _agg_feature()
        feat.pop("function")
        assert self._incomplete_codes([feat]) == ["FEATURE_INCOMPLETE"]

    def test_missing_source_column_is_error(self) -> None:
        feat = _agg_feature(source_column={"name": "", "type": "DoubleType"})
        assert self._incomplete_codes([feat]) == ["FEATURE_INCOMPLETE"]

    def test_passthrough_features_are_not_incomplete(self) -> None:
        results = self._validate(
            [
                {
                    "source_column": {"name": "event", "type": "StringType"},
                    "output_column": {"name": "event", "type": "StringType"},
                }
            ]
        )
        assert not any(r.code == "FEATURE_INCOMPLETE" for r in results)

    def test_complete_aggregation_is_not_incomplete(self) -> None:
        results = self._validate([_agg_feature()])
        assert not any(r.code == "FEATURE_INCOMPLETE" for r in results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
