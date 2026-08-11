"""Tests pinning the loader's ``SpecBase`` fallback preserves identity fields.

When ``_dict_to_spec`` cannot validate the input under the kind's
concrete Pydantic model (for any non-allowlisted reason), it falls back
to bare :class:`SpecBase`.  Historically the fallback dropped every
field except ``kind`` and ``name`` (loader.py:390-392), which silently
discarded the ``version`` that was present in the input.  Downstream
validation then saw ``version=None`` and fired a misleading
``MISSING_VERSION`` error instead of the real shape problem (e.g.
"FeatureGroup must reference at least one FeatureView" for a YAML
labelled ``kind: FeatureGroup`` but written in the FV-OFT-spec shape).

The defensive contract these tests pin:

* The :class:`SpecBase` fallback preserves identity fields whenever
  they are present in the input dict: ``kind``, ``name``, ``version``,
  ``database``, ``schema_`` (already renamed from YAML ``schema`` by
  the upstream helper at loader.py:669-670), and ``description``.
* Non-identity, kind-specific fields are intentionally **not** copied
  through — :class:`SpecBase` doesn't model them and Pydantic would
  reject them as extras.
* The migration-error allowlist (``"has been renamed to"`` /
  ``"has been removed"`` / ``"is not valid on"``) still propagates
  unchanged — operators see the actionable rename / removal message
  rather than a silent SpecBase degrade.
* The contract applies to every kind in the loader's ``kind_map``
  (FeatureGroup, BatchFeatureView, StreamingFeatureView,
  RealtimeFeatureView, Entity, StreamingSource, BatchSource) plus
  unknown kinds (the implicit ``cls = kind_map.get(kind, SpecBase)``
  fallback at loader.py:372).

End-to-end protection: with the identity-field preserve in place, a
degraded ``SpecBase(kind="FeatureGroup", version="V1")`` no longer
trips ``_check_versions`` MISSING_VERSION — the validator sees the
real version and can report the real shape problem on a later check.

Pinned by ``plans/fg_export_version_fix_loader_fallback_preserves_version.plan.md``.
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store.decl.loader import _dict_to_spec
from snowflake.ml.feature_store.decl.spec_models import SpecBase

# ---------------------------------------------------------------------------
# _dict_to_spec — versioned-kind fallback preserves identity fields
# ---------------------------------------------------------------------------


class TestDictToSpecFallbackPreservesVersion:
    """Pin that ``_dict_to_spec`` preserves the ``version`` (and other
    identity fields) when it falls back to bare ``SpecBase`` because the
    concrete model failed to validate the malformed input.
    """

    def test_fg_fallback_preserves_version(self) -> None:
        # The stray FG-in-FV-dir YAML that surfaces in verify_bug_bash:
        # ``kind: FeatureGroup`` with ``version: V1`` but no
        # ``feature_views[]``.  Pydantic FeatureGroup validation fails
        # (post-validator ``_validate_feature_group`` requires at least
        # one feature view); the loader must degrade gracefully WITHOUT
        # discarding the version that was on disk.
        spec = _dict_to_spec(
            {
                "kind": "FeatureGroup",
                "name": "USER_FRAUD_FG_DECL",
                "version": "V1",
                "database": "JKEW_DB",
                "schema_": "JKEW_SCHEMA",
                # NOTE: no ``feature_views[]`` — this is what fails
                # FeatureGroup Pydantic validation and triggers the
                # fallback.
            }
        )

        assert isinstance(spec, SpecBase)
        assert spec.kind == "FeatureGroup"
        assert spec.name == "USER_FRAUD_FG_DECL"
        assert spec.version == "V1", (
            "loader SpecBase fallback dropped the version — downstream "
            "validator will misfire MISSING_VERSION instead of the real "
            "shape problem"
        )
        assert spec.database == "JKEW_DB"
        assert spec.schema_ == "JKEW_SCHEMA"

    def test_fv_fallback_preserves_version_database_schema(self) -> None:
        # Same contract for FeatureView kinds: a malformed body must not
        # cost the loader its identity fields.
        spec = _dict_to_spec(
            {
                "kind": "BatchFeatureView",
                "name": "USER_CLICK_STATS_DECL",
                "version": "V1",
                "database": "JKEW_DB",
                "schema_": "JKEW_SCHEMA",
                # No sources / features / entities — BatchFeatureView
                # Pydantic validation fails.
            }
        )

        assert isinstance(spec, SpecBase)
        assert spec.kind == "BatchFeatureView"
        assert spec.name == "USER_CLICK_STATS_DECL"
        assert spec.version == "V1"
        assert spec.database == "JKEW_DB"
        assert spec.schema_ == "JKEW_SCHEMA"

    def test_streaming_fv_fallback_preserves_version(self) -> None:
        spec = _dict_to_spec(
            {
                "kind": "StreamingFeatureView",
                "name": "USER_CLICK_BACKFILL_DECL",
                "version": "V2",
            }
        )

        assert spec.kind == "StreamingFeatureView"
        assert spec.version == "V2"

    def test_realtime_fv_fallback_preserves_version(self) -> None:
        spec = _dict_to_spec(
            {
                "kind": "RealtimeFeatureView",
                "name": "USER_REALTIME",
                "version": "V3",
            }
        )

        assert spec.kind == "RealtimeFeatureView"
        assert spec.version == "V3"


# ---------------------------------------------------------------------------
# _dict_to_spec — non-versioned kinds still get identity fields preserved
# ---------------------------------------------------------------------------


class TestDictToSpecFallbackNonVersionedKinds:
    """Even kinds that don't carry a ``version`` (Entity, sources)
    benefit from the defensive preserve — the fallback should round-trip
    every identity field it has, not just ``kind`` + ``name``.
    """

    def test_entity_fallback_preserves_identity_fields(self) -> None:
        # Entity carries no version, but should still round-trip
        # database/schema/description through a fallback.
        spec = _dict_to_spec(
            {
                "kind": "Entity",
                "name": "USER_ID",
                "database": "DB",
                "schema_": "SCH",
                "description": "User identity entity",
                # Missing join_keys — Pydantic Entity validation may or
                # may not fail depending on defaults, but the identity
                # fields must round-trip either way.
            }
        )

        assert spec.kind == "Entity"
        assert spec.name == "USER_ID"
        assert spec.database == "DB"
        assert spec.schema_ == "SCH"
        assert spec.description == "User identity entity"

    def test_streaming_source_fallback_preserves_identity_fields(self) -> None:
        spec = _dict_to_spec(
            {
                "kind": "StreamingSource",
                "name": "CLICKSTREAM_EVENTS",
                "database": "DB",
                "schema_": "SCH",
                # malformed body that fails the StreamingSource model
                "garbage_field": object(),
            }
        )

        assert spec.kind == "StreamingSource"
        assert spec.name == "CLICKSTREAM_EVENTS"
        assert spec.database == "DB"
        assert spec.schema_ == "SCH"


# ---------------------------------------------------------------------------
# _dict_to_spec — unknown kinds also preserve identity fields
# ---------------------------------------------------------------------------


class TestDictToSpecFallbackUnknownKind:
    """The implicit ``cls = kind_map.get(kind, SpecBase)`` fallback at
    loader.py:372 already routes unknown kinds straight to SpecBase.
    The contract is the same — identity fields round-trip.
    """

    def test_unknown_kind_fallback_preserves_version(self) -> None:
        spec = _dict_to_spec(
            {
                "kind": "NotAKind",
                "name": "X",
                "version": "V99",
                "database": "DB",
                "schema_": "SCH",
            }
        )

        assert spec.kind == "NotAKind"
        assert spec.name == "X"
        assert spec.version == "V99"
        assert spec.database == "DB"
        assert spec.schema_ == "SCH"

    def test_empty_kind_falls_back_with_identity_fields(self) -> None:
        # ``kind`` absent or empty — the implicit SpecBase branch fires.
        spec = _dict_to_spec({"name": "Y", "version": "V1"})

        assert spec.kind == ""
        assert spec.name == "Y"
        assert spec.version == "V1"


# ---------------------------------------------------------------------------
# _dict_to_spec — extra fields are NOT smuggled into the SpecBase fallback
# ---------------------------------------------------------------------------


class TestDictToSpecFallbackDropsNonSpecBaseFields:
    """SpecBase doesn't model kind-specific fields like ``feature_views``,
    ``sources``, ``entities``, ``features``, etc.  Pydantic in strict
    mode would reject them as extras.  The fallback must only copy
    identity fields through — never the full kind-specific body.
    """

    def test_fg_fallback_drops_feature_views_field(self) -> None:
        # Input has a ``feature_views`` field that is invalid for the
        # FeatureGroup model (the field is ``list[FeatureViewRef]``).
        # The fallback fires, identity fields are preserved, and the
        # kind-specific body does NOT survive on the bare SpecBase.
        spec = _dict_to_spec(
            {
                "kind": "FeatureGroup",
                "name": "FG",
                "version": "V1",
                "feature_views": "not_a_list",  # forces validation fail
            }
        )

        assert spec.version == "V1"
        # The degraded fallback is a bare ``SpecBase`` — the FG-specific
        # ``feature_views`` attribute simply doesn't exist on it.
        assert type(spec) is SpecBase

    def test_fv_fallback_drops_kind_specific_fields(self) -> None:
        # Force a Pydantic validation failure on BatchFeatureView by
        # giving ``entities`` a non-list value (the model field is
        # ``list[str]``).  The fallback fires and we then verify that
        # kind-specific fields don't survive on the bare SpecBase.
        spec = _dict_to_spec(
            {
                "kind": "BatchFeatureView",
                "name": "FV",
                "version": "V1",
                "database": "DB",
                "schema_": "SCH",
                "entities": "not_a_list",  # forces validation fail
                "sources": [{"name": "S", "columns": []}],
                "features": [],
                "timestamp_col": "EVENT_TS",
            }
        )

        # Identity fields preserved.
        assert spec.kind == "BatchFeatureView"
        assert spec.version == "V1"
        assert spec.database == "DB"
        assert spec.schema_ == "SCH"
        # The degraded fallback is a bare ``SpecBase`` — kind-specific
        # attributes simply don't exist on it.  This pins that the
        # fallback path does NOT smuggle the kind-specific body
        # through.
        assert type(spec) is SpecBase


# ---------------------------------------------------------------------------
# Migration-error allowlist — regression guard
# ---------------------------------------------------------------------------


class TestDictToSpecMigrationErrorsStillPropagate:
    """The fallback must not swallow migration errors — operators rely
    on the actionable rename / removal message instead of a silent
    SpecBase degrade.
    """

    def test_legacy_backfill_table_on_streaming_source_propagates(self) -> None:
        # StreamingSource._reject_legacy_backfill_table raises
        # ValueError with "has been removed" in the message.  The
        # loader's allowlist must surface it.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _dict_to_spec(
                {
                    "kind": "StreamingSource",
                    "name": "SRC",
                    "backfill_table": "old_field",
                }
            )

    def test_renamed_field_error_propagates_unchanged(self) -> None:
        # Any ValidationError carrying "has been renamed to" should
        # propagate.  We simulate by handing a kind that lacks the
        # renamed key path — but this test asserts the contract by
        # patching the model_validate call.
        from unittest.mock import patch

        from pydantic import ValidationError
        from pydantic_core import PydanticCustomError

        def _raise_renamed(*_args: Any, **_kwargs: Any) -> None:
            raise ValidationError.from_exception_data(
                "FeatureGroup",
                [
                    {
                        "type": PydanticCustomError(
                            "value_error",
                            "Field 'foo' has been renamed to 'bar' — update your spec.",
                        ),
                        "loc": ("foo",),
                        "input": "x",
                    }
                ],
            )

        with patch(
            "snowflake.ml.feature_store.decl.spec_models.FeatureGroup.model_validate",
            side_effect=_raise_renamed,
        ), pytest.raises(ValidationError, match="has been renamed to"):
            _dict_to_spec({"kind": "FeatureGroup", "name": "X", "version": "V1"})

    def test_kind_targeted_rejection_propagates(self) -> None:
        # Validators that say "is not valid on" (e.g.
        # ``_reject_target_lag_on_stream_or_realtime``) propagate.
        from unittest.mock import patch

        from pydantic import ValidationError
        from pydantic_core import PydanticCustomError

        def _raise_not_valid_on(*_args: Any, **_kwargs: Any) -> None:
            raise ValidationError.from_exception_data(
                "BatchFeatureView",
                [
                    {
                        "type": PydanticCustomError(
                            "value_error",
                            "Field 'target_lag' is not valid on offline BatchFeatureView.",
                        ),
                        "loc": ("target_lag",),
                        "input": "x",
                    }
                ],
            )

        with patch(
            "snowflake.ml.feature_store.decl.spec_models.BatchFeatureView.model_validate",
            side_effect=_raise_not_valid_on,
        ), pytest.raises(ValidationError, match="is not valid on"):
            _dict_to_spec({"kind": "BatchFeatureView", "name": "X", "version": "V1"})


# ---------------------------------------------------------------------------
# End-to-end: defensive preserve prevents misleading MISSING_VERSION
# ---------------------------------------------------------------------------


class TestNoMissingVersionOnDegradedSpecWithVersion:
    """End-to-end protection: a degraded ``SpecBase`` that still carries
    its ``version`` from the input must NOT trip
    ``invariants._check_versions`` MISSING_VERSION.  This is the
    user-visible outcome of the loader fix — the operator no longer
    sees a misleading MISSING_VERSION on a YAML that literally had
    ``version: V1`` on disk.
    """

    def test_degraded_fg_with_version_does_not_fire_missing_version(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import validate_specs
        from snowflake.ml.feature_store.decl.types import AppliedState, SpecBatch

        spec = _dict_to_spec(
            {
                "kind": "FeatureGroup",
                "name": "USER_FRAUD_FG_DECL",
                "version": "V1",
                "database": "JKEW_DB",
                "schema_": "JKEW_SCHEMA",
            }
        )
        batch = SpecBatch(specs=[spec], source_files=["/fake/path.yaml"])
        applied_state = AppliedState(objects={})

        results = validate_specs(
            batch,
            applied_state,
            target_database="JKEW_DB",
            target_schema="JKEW_SCHEMA",
        )

        missing_version = [r for r in results if getattr(r, "code", "") == "MISSING_VERSION"]
        assert not missing_version, (
            f"degraded SpecBase with version='V1' incorrectly tripped " f"MISSING_VERSION; results: {results!r}"
        )

    def test_degraded_fg_without_version_still_fires_missing_version(self) -> None:
        # Negative pin: the fix preserves real behavior — if the input
        # truly has no version, MISSING_VERSION still fires.  This is
        # what catches the genuine "operator forgot to author a
        # version" case.
        from snowflake.ml.feature_store.decl.invariants import validate_specs
        from snowflake.ml.feature_store.decl.types import AppliedState, SpecBatch

        spec = _dict_to_spec(
            {
                "kind": "FeatureGroup",
                "name": "BAD_FG",
                # no ``version`` key at all
                "database": "DB",
                "schema_": "SCH",
            }
        )
        batch = SpecBatch(specs=[spec], source_files=["/fake/path.yaml"])
        applied_state = AppliedState(objects={})

        results = validate_specs(
            batch,
            applied_state,
            target_database="DB",
            target_schema="SCH",
        )

        missing_version = [r for r in results if getattr(r, "code", "") == "MISSING_VERSION"]
        assert missing_version, (
            f"FG with no version did NOT fire MISSING_VERSION — the "
            f"defensive preserve must not weaken genuine version-missing "
            f"detection; results: {results!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
