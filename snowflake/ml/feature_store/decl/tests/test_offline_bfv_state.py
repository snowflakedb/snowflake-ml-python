"""Failing tests for offline-only BatchFeatureView applied-state visibility.

Pins hypotheses H1 and H4 from
``plans/offline_bfv_state_fix_b9da0006.plan.md``.

Bug repro: an operator authors a ``BatchFeatureView`` with
``online: false`` and runs ``snow feature plan`` then ``apply``.  The
apply succeeds (offline Dynamic Table ``MY_FV$V1`` is created in
``INFORMATION_SCHEMA.TABLES``), but the next ``snow feature plan``
re-emits ``CREATE_FV MY_FV "New object: not found in applied state."``
instead of ``NO_CHANGE``.

Root cause (verified by code reading): ``state.fetch_applied_state``
ingests only ``raw_show_results`` (rows from
``SHOW ONLINE FEATURE TABLES``), and offline-only BFVs never produce
an OFT row — so they are missing from ``AppliedState`` regardless of
how many other inputs the function is given.

This test file pins the absence of an offline-only BFV in applied state
on the **current** code (H1 — RED) and the hash-equivalence target the
fix has to hit (H4 — RED until Phase 2 reconstructs the spec_payload).
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    compute_local_spec_hash,
)
from snowflake.ml.feature_store.decl.state import fetch_applied_state
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Shared fixtures — match docs/BATCH_FV_BUG_BASH.md §5 authoring
# ---------------------------------------------------------------------------


_DB = "JKEW_DB"
_SCH = "JKEW_SCHEMA"
_FV_NAME = "MY_BATCH_FV_BATCH_DECL"
_FV_VERSION = "V1"
_OFFLINE_DT_NAME = f"{_FV_NAME}${_FV_VERSION}"
_SOURCE_TABLE = "RAW_EVENTS_BATCH_DECL"


def _offline_bfv_authoring_dict() -> dict[str, Any]:
    """Return the authoring-side BFV dict matching the §5 walkthrough.

    Mirrors ``sources/feature_views/MY_BATCH_FV_BATCH_DECL.yaml`` after
    the loader's compile pass, with ``online: false`` so the apply
    creates only the offline Dynamic Table.

    Returns:
        dict: Authoring-side BFV dict with ``online: false`` and a
        single table-backed batch source.
    """
    return {
        "kind": "BatchFeatureView",
        "name": _FV_NAME,
        "version": _FV_VERSION,
        "database": _DB,
        "schema": _SCH,
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": _SOURCE_TABLE,
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "METRIC_VAL", "type": "FloatType"},
                ],
            }
        ],
        "refresh_freq": "1 minute",
        "target_lag": "1 minute",
    }


def _imperative_list_fv_row(*, online_enabled: bool = False) -> dict[str, Any]:
    """Return the row shape that ``imperative_executor.fetch_feature_view_rows``
    must emit for the offline-only BFV (Phase 1 contract — see
    Section 7 of the plan).

    Phase B3: ``source_refs`` carries the authoritative
    :class:`FvSourceRefsMetadata` payload so
    :func:`state._inject_batch_fv_source_from_metadata` can populate
    ``spec.sources[]`` directly — no DT-text parsing, no logical-name
    shim.  The shape mirrors :meth:`SourceRef.model_dump` for the
    matching ``BatchSource`` declaration in
    :func:`_offline_bfv_authoring_dict`.

    Args:
        online_enabled: Mirror of the ``online_config.enable`` flag
            on the imperative-list-FV row.

    Returns:
        Row dict in the Phase-1 contract shape.
    """
    return {
        "name": _FV_NAME,
        "version": _FV_VERSION,
        "database_name": _DB,
        "schema_name": _SCH,
        "kind": "BATCH",
        "entities": ["USER_ID"],
        "online_enabled": online_enabled,
        "target_lag": "1 minute",
        "refresh_freq": "1 minute",
        "warehouse": "TEST_WH",
        "cluster_by": "",
        "refresh_mode": "",
        "desc": "",
        "physical_dt_name": _OFFLINE_DT_NAME,
        "source_refs": [
            {
                "name": "EVENTS_BATCH_DECL",
                "source_type": "Batch",
                "table": _SOURCE_TABLE,
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "METRIC_VAL", "type": "FloatType"},
                ],
            }
        ],
    }


def _offline_dt_text() -> str:
    """Return the ``CREATE DYNAMIC TABLE`` text Snowflake records for
    a flat ``BatchSource.table`` BFV — this is what
    ``SHOW DYNAMIC TABLES`` would return in the ``text`` column for
    the offline DT created by ``register_feature_view`` with
    ``online: false``.

    Returns:
        DDL string mirroring the SHOW DYNAMIC TABLES ``text`` column.
    """
    return (
        f"CREATE DYNAMIC TABLE {_DB}.{_SCH}.{_OFFLINE_DT_NAME}\n"
        "TARGET_LAG = '1 minute'\n"
        "WAREHOUSE = TEST_WH\n"
        f"AS SELECT * FROM {_DB}.{_SCH}.{_SOURCE_TABLE}"
    )


# ===========================================================================
# H1 — Offline-only BFV is missing from AppliedState (current behaviour)
# ===========================================================================


class TestOfflineBfvMissingFromAppliedStateBeforeFix:
    """RED before fix, GREEN after Phase 2 — applied state must surface
    offline-only BFVs once ``feature_view_rows`` is threaded through.
    """

    def test_no_oft_row_no_feature_view_rows_yields_empty_state(self) -> None:
        """Sanity: with no OFT rows and no FV-list rows, applied state
        is empty.  This is the *current* observable behaviour against
        an offline-only deploy and it is exactly what makes the
        planner emit a spurious ``CREATE_FV``.  This test must keep
        passing after the fix when ``feature_view_rows`` is omitted.
        """
        state = fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={},
            default_database=_DB,
            default_schema=_SCH,
        )
        assert state.objects == {}

    def test_offline_bfv_appears_when_feature_view_rows_supplied(self) -> None:
        """The fix: when ``feature_view_rows`` is provided,
        ``fetch_applied_state`` builds an ``AppliedObject`` for the
        offline-only BFV keyed by ``BatchFeatureView:DB.SCH:NAME``.
        """
        state = fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={_OFFLINE_DT_NAME: _offline_dt_text()},
            feature_view_rows=[_imperative_list_fv_row(online_enabled=False)],
            default_database=_DB,
            default_schema=_SCH,
        )
        expected_key = f"BatchFeatureView:{_DB}.{_SCH}:{_FV_NAME}:{_FV_VERSION}"
        assert expected_key in state.objects
        obj = state.objects[expected_key]
        assert obj.kind == "BatchFeatureView"
        assert obj.name == _FV_NAME
        assert obj.version == _FV_VERSION
        assert obj.from_specification is True
        assert obj.content_hash != ""
        assert len(obj.content_hash) == 64

    def test_offline_bfv_skipped_when_oft_row_already_present(self) -> None:
        """If the SAME FV appears in both ``raw_show_results`` (online:
        true case) and ``feature_view_rows``, the OFT path wins — the
        new path must be purely additive for the offline subset.
        """
        oft_row = {
            "name": f"{_FV_NAME}${_FV_VERSION}$ONLINE",
            "database_name": _DB,
            "schema_name": _SCH,
            "created_on": "2024-01-01 00:00:00",
        }
        spec_payload = {
            "kind": "BatchFeatureView",
            "metadata": {
                "database": _DB,
                "schema": _SCH,
                "name": _FV_NAME,
                "version": _FV_VERSION,
            },
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [{"name": "EVENTS", "source_type": "Batch"}],
                "features": [],
                "target_lag_sec": 60,
            },
            "online_store_type": "postgres",
        }
        state = fetch_applied_state(
            raw_show_results=[oft_row],
            raw_table_results=[],
            specification_map={
                f"{_FV_NAME}${_FV_VERSION}$ONLINE": spec_payload,
            },
            entity_rows=[],
            dt_text_map={},
            feature_view_rows=[_imperative_list_fv_row(online_enabled=True)],
            default_database=_DB,
            default_schema=_SCH,
        )
        expected_key = f"BatchFeatureView:{_DB}.{_SCH}:{_FV_NAME}:{_FV_VERSION}"
        assert expected_key in state.objects
        # Exactly one object with this key — no duplicate from the
        # FV-list path.
        assert sum(1 for k in state.objects if k == expected_key) == 1
        # The OFT-derived path uses ``online_store_type=postgres``;
        # the FV-list path produces an offline-only payload (no
        # ``online_store_type``).  Verify the OFT shape won.
        obj = state.objects[expected_key]
        assert obj.spec_payload.get("online_store_type") == "postgres"


# ===========================================================================
# H4 — Reconstructed applied spec hash-equals local-compile hash
# ===========================================================================


class TestOfflineBfvHashRoundTrip:
    """RED until Phase 2 — once ``fetch_applied_state`` reconstructs a
    SPECIFICATION-equivalent ``spec_payload`` from the FV-list row +
    DT text, ``_full_spec_hash(applied) == compute_local_spec_hash(local)``
    on a clean round-trip.  This is the gating invariant for NO_CHANGE
    detection on the second plan.
    """

    def test_local_compile_hash_equals_reconstructed_applied_hash(self) -> None:
        """The fix: hash equality on a clean round-trip enables
        NO_CHANGE on the second plan."""
        local = _offline_bfv_authoring_dict()
        local_hash = compute_local_spec_hash(local, _DB, _SCH)

        state = fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={_OFFLINE_DT_NAME: _offline_dt_text()},
            feature_view_rows=[_imperative_list_fv_row(online_enabled=False)],
            default_database=_DB,
            default_schema=_SCH,
        )
        expected_key = f"BatchFeatureView:{_DB}.{_SCH}:{_FV_NAME}:{_FV_VERSION}"
        applied = state.objects[expected_key]
        # ``content_hash`` must be the ``_full_spec_hash`` of the
        # reconstructed ``spec_payload`` so the planner's
        # ``compute_local_spec_hash(local) == applied.content_hash``
        # branch in :func:`generate_plan` lights up NO_CHANGE.
        assert applied.content_hash == _full_spec_hash(applied.spec_payload)
        assert applied.content_hash == local_hash

    def test_reconstructed_spec_payload_has_required_planner_shape(self) -> None:
        """The reconstructed ``spec_payload`` must satisfy the
        planner's BatchFV branches:

        - ``kind == "BatchFeatureView"``
        - ``metadata.{database, schema, name, version}`` populated
        - ``spec.ordered_entity_column_names`` from row entities
        - ``spec.sources[0].table`` recovered from DT text
        - ``spec.target_lag_sec`` set so
          :func:`_batch_fv_operational_drift` can compare against
          local ``refresh_freq``
        - ``online_store_type`` absent — offline-only signal
        """
        state = fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={_OFFLINE_DT_NAME: _offline_dt_text()},
            feature_view_rows=[_imperative_list_fv_row(online_enabled=False)],
            default_database=_DB,
            default_schema=_SCH,
        )
        applied = state.objects[f"BatchFeatureView:{_DB}.{_SCH}:{_FV_NAME}:{_FV_VERSION}"]
        payload = applied.spec_payload

        assert payload["kind"] == "BatchFeatureView"

        meta = payload["metadata"]
        assert meta["database"] == _DB
        assert meta["schema"] == _SCH
        assert meta["name"] == _FV_NAME
        assert meta["version"] == _FV_VERSION

        inner = payload["spec"]
        assert inner["ordered_entity_column_names"] == ["USER_ID"]
        assert isinstance(inner["sources"], list) and inner["sources"]
        first_source = inner["sources"][0]
        assert first_source.get("table", "").upper() == _SOURCE_TABLE.upper()
        # ``target_lag_sec`` is parsed from the row's ``target_lag``
        # so ``_batch_fv_operational_drift`` can compare it against
        # ``refresh_freq`` on the local authoring side.
        assert inner.get("target_lag_sec") == 60

        # ``online_store_type`` MUST be absent for offline-only FVs
        # so the planner detects ``online: false → online: true``
        # transitions as drift.
        assert "online_store_type" not in payload or payload.get("online_store_type") in (None, "")

    def test_target_lag_falls_back_to_refresh_freq_when_target_lag_empty(self) -> None:
        """Live ``list_feature_views()`` rows populate ``REFRESH_FREQ``
        (uppercase) and leave ``target_lag`` absent — that is the column
        ``_LIST_FEATURE_VIEW_SCHEMA`` defines.  ``_build_offline_fv_object``
        must fall back to ``refresh_freq`` so the recovered payload
        carries ``spec.target_lag_sec`` and the planner does not surface
        a phantom ``UPDATE_FV`` on every replan after a clean apply.

        Pins the BATCH_FV_BUG_BASH §7b post-apply NO_CHANGE contract.
        """
        row_without_target_lag = {
            "name": _FV_NAME,
            "version": _FV_VERSION,
            "database_name": _DB,
            "schema_name": _SCH,
            "kind": "BATCH",
            "entities": ["USER_ID"],
            "online_enabled": False,
            "target_lag": "",
            "refresh_freq": "1 minute",
            "warehouse": "TEST_WH",
            "desc": "",
            "physical_dt_name": _OFFLINE_DT_NAME,
        }
        state = fetch_applied_state(
            raw_show_results=[],
            raw_table_results=[],
            specification_map={},
            entity_rows=[],
            dt_text_map={_OFFLINE_DT_NAME: _offline_dt_text()},
            feature_view_rows=[row_without_target_lag],
            default_database=_DB,
            default_schema=_SCH,
        )
        applied = state.objects[f"BatchFeatureView:{_DB}.{_SCH}:{_FV_NAME}:{_FV_VERSION}"]
        assert applied.spec_payload["spec"].get("target_lag_sec") == 60


# ===========================================================================
# Bug B — Advanced offline BFV (online:false, offline:false, cluster_by +
#          aggregation_secondary_keys) produces hash mismatch on second plan
# ===========================================================================

_ADV_FV_NAME = "MY_ADV_BFV_DECL"
_ADV_DT_NAME = f"{_ADV_FV_NAME}$V1"
_ADV_SOURCE_REF = {
    "name": "EVENTS_ADV_DECL",
    "source_type": "Batch",
    "table": "RAW_ADV_EVENTS",
}


def _adv_bfv_spec_text(*, with_secondary_keys: bool) -> dict[str, Any]:
    """Construct the spec_text dict that _serialize_batch_fv_spec produces.

    Args:
        with_secondary_keys: When True, include aggregation_secondary_keys in the
            inner spec dict.

    Returns:
        A SPECIFICATION-shaped dict matching the _serialize_batch_fv_spec contract.
    """
    inner: dict[str, Any] = {
        "ordered_entity_column_names": ["ENTITY_ID"],
        "sources": [_ADV_SOURCE_REF],
        "features": [],
        "target_lag_sec": 60,
        "cluster_by": ["ENTITY_ID"],
        "refresh_mode": "AUTO",
        "warehouse": "TEST_WH",
    }
    if with_secondary_keys:
        inner["aggregation_secondary_keys"] = ["REGION"]
    return {
        "kind": "BatchFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCH,
            "name": _ADV_FV_NAME,
            "version": "V1",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "0.1.0",
        },
        "spec": inner,
    }


def _adv_bfv_row(*, with_secondary_keys: bool) -> dict[str, Any]:
    """Build the feature_view_rows entry for MY_ADV_BFV_DECL.

    ``with_secondary_keys`` is forwarded to :func:`_adv_bfv_spec_text` to
    control whether the embedded spec_text carries aggregation_secondary_keys.

    Args:
        with_secondary_keys: Forwarded to _adv_bfv_spec_text.

    Returns:
        A row dict in the feature_view_rows contract shape.
    """
    import json as _json

    return {
        "name": _ADV_FV_NAME,
        "version": "V1",
        "database_name": _DB,
        "schema_name": _SCH,
        "kind": "BATCH",
        "entities": ["ENTITY_ID"],
        "online_enabled": False,
        "target_lag": "",
        "refresh_freq": "1 minute",
        "warehouse": "TEST_WH",
        "cluster_by": _json.dumps(["ENTITY_ID"]),
        "refresh_mode": "AUTO",
        "desc": "",
        "physical_dt_name": _ADV_DT_NAME,
        "source_refs": [_ADV_SOURCE_REF],
        "spec_text": _adv_bfv_spec_text(with_secondary_keys=with_secondary_keys),
    }


def _adv_bfv_authoring_dict() -> dict[str, Any]:
    """Return the authoring-side dict for MY_ADV_BFV_DECL.

    Matches sources/feature_views/MY_ADV_BFV_DECL.yaml in the example store.

    Returns:
        dict: Authoring-side BFV dict with online:false, offline:false, cluster_by,
        and aggregation_secondary_keys set.
    """
    return {
        "kind": "BatchFeatureView",
        "name": _ADV_FV_NAME,
        "version": "V1",
        "database": _DB,
        "schema": _SCH,
        "online": False,
        "offline": False,
        "entities": ["ENTITY_ID"],
        "sources": [_ADV_SOURCE_REF],
        "refresh_freq": "1 minute",
        "cluster_by": ["ENTITY_ID"],
        "aggregation_secondary_keys": ["REGION"],
        "refresh_mode": "AUTO",
        "initialize": "ON_CREATE",
    }


def _adv_applied_and_local_hashes(*, with_secondary_keys: bool) -> tuple[str, str]:
    """Return (applied content_hash, local compile hash) for MY_ADV_BFV_DECL.

    Args:
        with_secondary_keys: Whether reconstructed spec_text includes
            aggregation_secondary_keys.

    Returns:
        Applied hash from fetch_applied_state and the local compile hash.
    """
    local_hash = compute_local_spec_hash(_adv_bfv_authoring_dict(), _DB, _SCH)
    state = fetch_applied_state(
        raw_show_results=[],
        raw_table_results=[],
        specification_map={},
        entity_rows=[],
        dt_text_map={},
        feature_view_rows=[_adv_bfv_row(with_secondary_keys=with_secondary_keys)],
        default_database=_DB,
        default_schema=_SCH,
    )
    applied_key = f"BatchFeatureView:{_DB}.{_SCH}:{_ADV_FV_NAME}:V1"
    assert (
        applied_key in state.objects
    ), f"Advanced offline BFV must appear in applied state; keys present: {list(state.objects)}"
    return state.objects[applied_key].content_hash, local_hash


class TestAdvancedOfflineBfvHashRoundTrip:
    """aggregation_secondary_keys in applied spec_text must participate in the hash.

    Authoring includes aggregation_secondary_keys=["REGION"].  When reconstructed
    spec_text carries those keys, applied hash matches local compile.  When they
    are omitted, hashes differ and the planner would emit RECREATE_FV.
    """

    def test_hash_matches_when_spec_text_includes_secondary_keys(self) -> None:
        applied_hash, local_hash = _adv_applied_and_local_hashes(with_secondary_keys=True)
        assert applied_hash == local_hash

    def test_hash_mismatches_when_spec_text_omits_secondary_keys(self) -> None:
        applied_hash, local_hash = _adv_applied_and_local_hashes(with_secondary_keys=False)
        assert applied_hash != local_hash


if __name__ == "__main__":
    pytest_driver.main()
