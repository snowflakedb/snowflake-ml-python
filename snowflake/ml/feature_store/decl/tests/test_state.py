"""Tests for decl/state.py — applied state parsing from raw SHOW results."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.invariants import structural_fingerprint_hash
from snowflake.ml.feature_store.decl.state import (
    _datasource_objects_from_specs,
    _extract_spec_from_oft,
    _parse_cluster_by_list,
    _parse_oft_name,
    fetch_applied_state,
    orphaned_oft_warnings,
    resolve_oft_name,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC_PAYLOAD: dict[str, Any] = {
    "kind": "StreamingFeatureView",
    "metadata": {"name": "user_clicks", "version": "V1", "database": "DB", "schema": "SCH"},
    "spec": {
        "ordered_entity_column_names": ["USER_ID"],
        "sources": [],
        "features": [
            {
                "source_column": {"name": "EVENT", "type": "StringType"},
                "output_column": {"name": "EVENT", "type": "StringType"},
            }
        ],
    },
}

_SHOW_ROW: dict[str, Any] = {
    "name": "USER_CLICKS$V1$ONLINE",
    "created_on": "2024-01-01 00:00:00",
    "specification": json.dumps(_SPEC_PAYLOAD),
}


# ---------------------------------------------------------------------------
# _parse_oft_name tests
# ---------------------------------------------------------------------------


class TestParseOftName:
    def test_standard_naming_convention(self) -> None:
        base, version = _parse_oft_name("USER_CLICKS$V1$ONLINE")
        assert base == "USER_CLICKS"
        assert version == "V1"

    def test_version_with_dots(self) -> None:
        base, version = _parse_oft_name("MY_FV$1.2.3$ONLINE")
        assert base == "MY_FV"
        assert version == "1.2.3"

    def test_no_online_suffix(self) -> None:
        # Malformed name — should still not crash
        base, version = _parse_oft_name("MY_FV$V2")
        assert base == "MY_FV"
        assert version == "V2"

    def test_minimal_name(self) -> None:
        base, version = _parse_oft_name("FV$V1$ONLINE")
        assert base == "FV"
        assert version == "V1"


# ---------------------------------------------------------------------------
# resolve_oft_name tests
# ---------------------------------------------------------------------------


def _oft_row(name: str) -> dict[str, Any]:
    return {"name": name}


class TestResolveOftName:
    """``resolve_oft_name`` maps a user-supplied ``(name, version)`` onto a
    single deployed OFT name, or returns an actionable error string when the
    request is ambiguous or unresolvable.
    """

    def test_single_version_resolves_by_bare_name(self) -> None:
        rows = [_oft_row("USER_CLICKS$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS")
        assert error is None
        assert oft_name == "USER_CLICKS$V1$ONLINE"

    def test_bare_name_is_case_insensitive(self) -> None:
        rows = [_oft_row("USER_CLICKS$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "user_clicks")
        assert error is None
        assert oft_name == "USER_CLICKS$V1$ONLINE"

    def test_version_selects_matching_oft(self) -> None:
        rows = [
            _oft_row("USER_CLICKS$V1$ONLINE"),
            _oft_row("USER_CLICKS$V2$ONLINE"),
        ]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS", "V2")
        assert error is None
        assert oft_name == "USER_CLICKS$V2$ONLINE"

    def test_version_is_case_insensitive(self) -> None:
        rows = [
            _oft_row("USER_CLICKS$V1$ONLINE"),
            _oft_row("USER_CLICKS$V2$ONLINE"),
        ]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS", "v2")
        assert error is None
        assert oft_name == "USER_CLICKS$V2$ONLINE"

    def test_ambiguous_bare_name_returns_error_listing_versions(self) -> None:
        rows = [
            _oft_row("USER_CLICKS$V2$ONLINE"),
            _oft_row("USER_CLICKS$V1$ONLINE"),
        ]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS")
        assert oft_name is None
        assert error is not None
        # Versions listed in deterministic (sorted) order and the message
        # tells the operator how to disambiguate.
        assert "V1" in error and "V2" in error
        assert error.index("V1") < error.index("V2")
        assert "--version" in error

    def test_not_found_bare_name(self) -> None:
        rows = [_oft_row("OTHER$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS")
        assert oft_name is None
        assert error is not None
        assert "not found" in error

    def test_not_found_with_version(self) -> None:
        rows = [_oft_row("USER_CLICKS$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS", "V9")
        assert oft_name is None
        assert error is not None
        assert "not found" in error
        assert "V9" in error

    def test_full_oft_name_passthrough(self) -> None:
        rows = [
            _oft_row("USER_CLICKS$V1$ONLINE"),
            _oft_row("USER_CLICKS$V2$ONLINE"),
        ]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS$V2$ONLINE")
        assert error is None
        assert oft_name == "USER_CLICKS$V2$ONLINE"

    def test_full_oft_name_passthrough_case_insensitive(self) -> None:
        rows = [_oft_row("USER_CLICKS$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "user_clicks$v1$online")
        assert error is None
        assert oft_name == "USER_CLICKS$V1$ONLINE"

    def test_full_oft_name_with_matching_version_resolves(self) -> None:
        """A full OFT name paired with the matching ``--version`` still
        resolves via the fast path."""
        rows = [_oft_row("USER_CLICKS$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS$V1$ONLINE", "V1")
        assert error is None
        assert oft_name == "USER_CLICKS$V1$ONLINE"

    def test_full_oft_name_with_conflicting_version_is_not_found(self) -> None:
        """A full OFT name plus a conflicting ``--version`` must not return
        the wrong OFT.

        Before the guard, ``resolve_oft_name("USER_CLICKS$V1$ONLINE",
        version="V2")`` silently returned the V1 OFT.  Now the fast path is
        skipped and no row matches, so the caller gets a not-found error.
        """
        rows = [
            _oft_row("USER_CLICKS$V1$ONLINE"),
            _oft_row("USER_CLICKS$V2$ONLINE"),
        ]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS$V1$ONLINE", "V2")
        assert oft_name is None
        assert error is not None
        assert "not found" in error

    def test_full_oft_name_with_unavailable_version_errors(self) -> None:
        """Full name + a version that matches nothing → not-found error,
        never the mismatched fast-path row."""
        rows = [_oft_row("USER_CLICKS$V1$ONLINE")]
        oft_name, error = resolve_oft_name(rows, "USER_CLICKS$V1$ONLINE", "V9")
        assert oft_name is None
        assert error is not None
        assert "V9" in error


class TestApiResolveOftNameFacade:
    """``api.resolve_oft_name`` is a thin re-export of
    ``state.resolve_oft_name``.  Pin that the facade delegates rather than
    drifting from the implementation.
    """

    def test_facade_delegates_success(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        rows = [_oft_row("USER_CLICKS$V1$ONLINE"), _oft_row("USER_CLICKS$V2$ONLINE")]
        assert decl_api.resolve_oft_name(rows, "USER_CLICKS", "V2") == resolve_oft_name(rows, "USER_CLICKS", "V2")
        assert decl_api.resolve_oft_name(rows, "USER_CLICKS", "V2")[0] == "USER_CLICKS$V2$ONLINE"

    def test_facade_delegates_error(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        rows = [_oft_row("USER_CLICKS$V1$ONLINE"), _oft_row("USER_CLICKS$V2$ONLINE")]
        oft_name, error = decl_api.resolve_oft_name(rows, "USER_CLICKS")
        assert oft_name is None
        assert error is not None and "--version" in error


# ---------------------------------------------------------------------------
# _extract_spec_from_oft tests
# ---------------------------------------------------------------------------


class TestExtractSpecFromOft:
    def test_extracts_valid_json(self) -> None:
        row: dict[str, Any] = {"specification": json.dumps(_SPEC_PAYLOAD)}
        spec = _extract_spec_from_oft(row)
        assert spec is not None
        assert spec.get("kind") == "StreamingFeatureView"

    def test_missing_specification_returns_none(self) -> None:
        row: dict[str, Any] = {"name": "FV$V1$ONLINE"}
        spec = _extract_spec_from_oft(row)
        assert spec is None

    def test_invalid_json_returns_none(self) -> None:
        row: dict[str, Any] = {"specification": "not-valid-json"}
        spec = _extract_spec_from_oft(row)
        assert spec is None

    def test_empty_specification_returns_none(self) -> None:
        row: dict[str, Any] = {"specification": ""}
        spec = _extract_spec_from_oft(row)
        assert spec is None


# ---------------------------------------------------------------------------
# fetch_applied_state tests
# ---------------------------------------------------------------------------


class TestFetchAppliedState:
    def test_empty_results_returns_empty_state(self) -> None:
        state = fetch_applied_state([], None)
        assert state.objects == {}

    def test_single_row_parsed_correctly(self) -> None:
        state = fetch_applied_state([_SHOW_ROW], None)
        assert len(state.objects) == 1
        key = list(state.objects.keys())[0]
        obj = state.objects[key]
        assert obj.name == "USER_CLICKS"
        assert obj.version == "V1"

    def test_content_hash_is_populated(self) -> None:
        state = fetch_applied_state([_SHOW_ROW], None)
        obj = list(state.objects.values())[0]
        assert obj.content_hash != ""
        assert len(obj.content_hash) == 64  # SHA-256 hex

    def test_spec_payload_is_stored(self) -> None:
        state = fetch_applied_state([_SHOW_ROW], None)
        obj = list(state.objects.values())[0]
        assert obj.spec_payload != {}

    def test_multiple_rows_parsed(self) -> None:
        row2_spec = dict(_SPEC_PAYLOAD)
        row2_spec["metadata"] = dict(row2_spec["metadata"], name="profile_fv", version="V2")
        row2: dict[str, Any] = {
            "name": "PROFILE_FV$V2$ONLINE",
            "created_on": "2024-01-02 00:00:00",
            "specification": json.dumps(row2_spec),
        }
        state = fetch_applied_state([_SHOW_ROW, row2], None)
        assert len(state.objects) == 2

    def test_row_without_specification_column_is_skipped(self) -> None:
        row: dict[str, Any] = {"name": "FV$V1$ONLINE", "created_on": "2024-01-01"}
        state = fetch_applied_state([row], None)
        assert len(state.objects) == 0

    def test_kind_is_set_from_spec(self) -> None:
        state = fetch_applied_state([_SHOW_ROW], None)
        obj = list(state.objects.values())[0]
        assert obj.kind == "StreamingFeatureView"

    def test_key_format(self) -> None:
        state = fetch_applied_state([_SHOW_ROW], None)
        key = list(state.objects.keys())[0]
        # Key should follow kind:db.schema:name format
        assert ":" in key

    def test_same_name_two_versions_do_not_shadow(self) -> None:
        """Two deployed versions of one FV name must not collide.

        Object identity is (name, version); before the fix both versions
        keyed to ``kind:DB.SCH:USER_CLICKS`` and the second silently
        overwrote the first in ``AppliedState.objects``.
        """
        v1_spec = copy.deepcopy(_SPEC_PAYLOAD)
        v2_spec = copy.deepcopy(_SPEC_PAYLOAD)
        v2_spec["metadata"] = dict(v2_spec["metadata"], version="V2")
        v1_row: dict[str, Any] = {
            "name": "USER_CLICKS$V1$ONLINE",
            "created_on": "2024-01-01 00:00:00",
            "specification": json.dumps(v1_spec),
        }
        v2_row: dict[str, Any] = {
            "name": "USER_CLICKS$V2$ONLINE",
            "created_on": "2024-01-02 00:00:00",
            "specification": json.dumps(v2_spec),
        }
        state = fetch_applied_state([v1_row, v2_row], None)
        assert len(state.objects) == 2
        assert {obj.version for obj in state.objects.values()} == {"V1", "V2"}


# ---------------------------------------------------------------------------
# fetch_applied_state with describe_map (Phase 1 — idempotent apply)
# ---------------------------------------------------------------------------


class TestFetchAppliedStateWithDescribeMap:
    """Verify that DESCRIBE data is used to build AppliedObjects when
    the SHOW result has no ``specification`` column (the normal case)."""

    _SHOW_ROW_NO_SPEC: dict[str, Any] = {
        "name": "CLICK_FV$V1$ONLINE",
        "database_name": "DB",
        "schema_name": "SCH",
        "created_on": "2024-01-01 00:00:00",
    }

    _DESCRIBE_ROWS = [
        {"name": "USER_ID", "type": "VARCHAR", "primary key": "Y"},
        {"name": "EVENT", "type": "VARCHAR", "primary key": "N"},
    ]

    def test_describe_map_builds_applied_object(self) -> None:
        """Without a specification column, describe_map should produce an AppliedObject."""
        state = fetch_applied_state(
            [self._SHOW_ROW_NO_SPEC],
            None,
            describe_map={"CLICK_FV$V1$ONLINE": self._DESCRIBE_ROWS},
        )
        assert len(state.objects) == 1
        obj = list(state.objects.values())[0]
        assert obj.name == "CLICK_FV"
        assert obj.version == "V1"
        assert obj.content_hash != ""
        assert len(obj.content_hash) == 64  # SHA-256 hex

    def test_describe_map_key_format(self) -> None:
        """Key must be StreamingFeatureView:DB.SCH:CLICK_FV:V1 (identity = name+version)."""
        state = fetch_applied_state(
            [self._SHOW_ROW_NO_SPEC],
            None,
            describe_map={"CLICK_FV$V1$ONLINE": self._DESCRIBE_ROWS},
        )
        key = list(state.objects.keys())[0]
        assert key == "StreamingFeatureView:DB.SCH:CLICK_FV:V1"

    def test_describe_map_content_hash_matches_spec_fingerprint(self) -> None:
        """Hash from DESCRIBE must equal structural_fingerprint_hash of authoring spec."""
        state = fetch_applied_state(
            [self._SHOW_ROW_NO_SPEC],
            None,
            describe_map={"CLICK_FV$V1$ONLINE": self._DESCRIBE_ROWS},
        )
        obj = list(state.objects.values())[0]

        # Authoring spec with the same structural shape (StringType → VARCHAR)
        spec_data: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "name": "CLICK_FV",
            "version": "V1",
            "features": [
                {"output_column": {"name": "EVENT", "type": "StringType"}},
            ],
        }
        expected_hash = structural_fingerprint_hash(spec_data)
        assert obj.content_hash == expected_hash

    def test_no_describe_map_skips_row_without_spec(self) -> None:
        """Without describe_map, rows lacking specification are skipped (old behaviour)."""
        state = fetch_applied_state([self._SHOW_ROW_NO_SPEC], None, describe_map=None)
        assert len(state.objects) == 0

    def test_empty_describe_map_skips_row(self) -> None:
        """describe_map present but missing entry for this OFT → row skipped."""
        state = fetch_applied_state([self._SHOW_ROW_NO_SPEC], None, describe_map={})
        assert len(state.objects) == 0

    def test_multiple_ofts_with_describe_map(self) -> None:
        """Multiple OFTs are all parsed when describe_map has entries for each."""
        row2: dict[str, Any] = {
            "name": "PROFILE_FV$V2$ONLINE",
            "database_name": "DB",
            "schema_name": "SCH",
        }
        describe_map: dict[str, Any] = {
            "CLICK_FV$V1$ONLINE": self._DESCRIBE_ROWS,
            "PROFILE_FV$V2$ONLINE": [{"name": "SCORE", "type": "FLOAT", "primary key": "N"}],
        }
        state = fetch_applied_state([self._SHOW_ROW_NO_SPEC, row2], None, describe_map=describe_map)
        assert len(state.objects) == 2


# ---------------------------------------------------------------------------
# parse_specification_rows — DESCRIBE ... TYPE = SPECIFICATION result parsing
# ---------------------------------------------------------------------------


class TestParseSpecificationRows:
    """The new `DESCRIBE ONLINE FEATURE TABLE <name> TYPE = SPECIFICATION` SQL
    returns the original spec JSON.  ``parse_specification_rows`` extracts and
    parses it from raw cursor rows into a dict."""

    def test_returns_none_for_empty_rows(self) -> None:
        from snowflake.ml.feature_store.decl.state import parse_specification_rows

        assert parse_specification_rows([]) is None
        assert parse_specification_rows(None) is None

    def test_parses_single_row_with_specification_column(self) -> None:
        from snowflake.ml.feature_store.decl.state import parse_specification_rows

        spec: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "name": "click_fv",
                "version": "v1",
                "database": "DB",
                "schema": "SCH",
            },
            "spec": {"sources": [], "features": []},
        }
        rows = [{"specification": json.dumps(spec)}]
        parsed = parse_specification_rows(rows)
        assert parsed is not None
        assert parsed["kind"] == "StreamingFeatureView"
        assert parsed["metadata"]["name"] == "click_fv"

    def test_parses_uppercase_specification_column(self) -> None:
        from snowflake.ml.feature_store.decl.state import parse_specification_rows

        spec: dict[str, Any] = {"kind": "StreamingFeatureView", "metadata": {"name": "x"}}
        rows = [{"SPECIFICATION": json.dumps(spec)}]
        parsed = parse_specification_rows(rows)
        assert parsed is not None
        assert parsed["kind"] == "StreamingFeatureView"

    def test_invalid_json_returns_none(self) -> None:
        from snowflake.ml.feature_store.decl.state import parse_specification_rows

        rows = [{"specification": "not json {{{"}]
        assert parse_specification_rows(rows) is None

    def test_missing_specification_column_returns_none(self) -> None:
        from snowflake.ml.feature_store.decl.state import parse_specification_rows

        rows = [{"name": "FOO", "value": "bar"}]
        assert parse_specification_rows(rows) is None

    def test_falls_back_to_first_string_value(self) -> None:
        """Some cursor adapters expose unnamed first-column values; the parser
        accepts a single-key row with a JSON string value as the spec."""
        from snowflake.ml.feature_store.decl.state import parse_specification_rows

        spec: dict[str, Any] = {"kind": "StreamingFeatureView", "metadata": {"name": "y"}}
        rows = [{"value": json.dumps(spec)}]
        parsed = parse_specification_rows(rows)
        assert parsed is not None
        assert parsed["kind"] == "StreamingFeatureView"


# ---------------------------------------------------------------------------
# fetch_applied_state with specification_map (Phase 1 — full-spec state)
# ---------------------------------------------------------------------------


class TestFetchAppliedStateWithSpecificationMap:
    _SPEC_FULL: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "metadata": {
            "name": "click_fv",
            "version": "v1",
            "database": "DB",
            "schema": "SCH",
        },
        "spec": {
            "ordered_entity_column_names": ["user_id"],
            "sources": [
                {
                    "name": "user_events",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "user_id", "type": "StringType"},
                        {"name": "event", "type": "StringType"},
                    ],
                },
            ],
            "features": [{"output_column": {"name": "event", "type": "StringType"}}],
            "udf": {
                "function_name": "transform",
                "function_definition": "def transform(x): return x",
                "language": "python",
                "output_columns": [
                    {"name": "event", "type": "StringType"},
                ],
            },
        },
    }

    _SHOW_ROW: dict[str, Any] = {
        "name": "CLICK_FV$V1$ONLINE",
        "database_name": "DB",
        "schema_name": "SCH",
        "created_on": "2024-01-01 00:00:00",
    }

    def test_specification_map_populates_spec_payload(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"CLICK_FV$V1$ONLINE": self._SPEC_FULL},
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "StreamingFeatureView"]
        assert len(fv_objs) == 1
        obj = fv_objs[0]
        assert obj.spec_payload  # not empty
        # The full spec JSON should be present in the payload
        assert "spec" in obj.spec_payload or "udf" in obj.spec_payload.get("spec", {})

    def test_specification_map_uses_uppercase_canonical_key(self) -> None:
        """Applied keys must be uppercased so they collide with planner keys.

        Without uppercasing, a deployed FV named ``user_profile_info``
        produces an applied-state key
        ``StreamingFeatureView:DB.SCH:user_profile_info`` while the
        loaded YAML's ``name: user_profile_info`` produces a batch-side
        key ``StreamingFeatureView:DB.SCH:USER_PROFILE_INFO`` (because
        :func:`invariants.spec_key` uppercases the name).  In
        full-directory mode this mismatch makes every FV simultaneously
        "new" *and* "orphaned", emitting spurious
        ``CREATE_FV`` + ``DROP_FV`` ops on a clean round-trip.
        """
        mixed_case_spec = copy.deepcopy(self._SPEC_FULL)
        mixed_case_spec["metadata"]["name"] = "Click_FV"
        mixed_case_spec["metadata"]["database"] = "db"
        mixed_case_spec["metadata"]["schema"] = "sch"

        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"CLICK_FV$V1$ONLINE": mixed_case_spec},
        )
        fv_keys = [k for k in state.objects.keys() if "FeatureView" in k]
        assert any(":DB.SCH:CLICK_FV" in k for k in fv_keys), fv_keys
        assert not any("Click_FV" in k for k in fv_keys), fv_keys
        assert not any(":db.sch:" in k for k in fv_keys), fv_keys

    def test_specification_marks_from_specification_true(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"CLICK_FV$V1$ONLINE": self._SPEC_FULL},
        )
        obj = list(state.objects.values())[0]
        assert obj.from_specification is True

    def test_no_specification_falls_back_to_describe_map(self) -> None:
        describe_rows = [
            {"name": "USER_ID", "type": "VARCHAR", "primary key": "Y"},
            {"name": "EVENT", "type": "VARCHAR", "primary key": "N"},
        ]
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            describe_map={"CLICK_FV$V1$ONLINE": describe_rows},
        )
        obj = list(state.objects.values())[0]
        assert obj.from_specification is False

    def test_entity_rows_become_applied_objects(self) -> None:
        entity_rows = [
            {
                "name": "SNOWML_FEATURE_STORE_ENTITY_USER",
                "database_name": "DB",
                "schema_name": "SCH",
                "allowed_values": '["USER_ID"]',
            },
            {
                "name": "SNOWML_FEATURE_STORE_ENTITY_DEVICE",
                "database_name": "DB",
                "schema_name": "SCH",
                "allowed_values": '["DEVICE_ID"]',
            },
        ]
        state = fetch_applied_state(
            [],
            None,
            entity_rows=entity_rows,
        )
        kinds = [o.kind for o in state.objects.values()]
        assert kinds.count("Entity") == 2
        names = {o.name for o in state.objects.values() if o.kind == "Entity"}
        assert "user" in names or "USER" in names
        assert "device" in names or "DEVICE" in names

    def test_entity_join_keys_in_details(self) -> None:
        entity_rows = [
            {
                "name": "SNOWML_FEATURE_STORE_ENTITY_USER",
                "database_name": "DB",
                "schema_name": "SCH",
                "allowed_values": '["USER_ID"]',
            }
        ]
        state = fetch_applied_state([], None, entity_rows=entity_rows)
        ent = next(o for o in state.objects.values() if o.kind == "Entity")
        join_keys = ent.details.get("join_keys", [])
        assert "USER_ID" in join_keys

    def test_entity_rows_from_imperative_shape(self) -> None:
        """Pin the round-trip from ``imperative_executor.fetch_entity_rows``
        through :func:`fetch_applied_state` so any future drift in either
        the imperative ``list_entities()`` projection or the SHOW TAGS row
        consumer is caught immediately.

        The translator in :func:`imperative_executor.fetch_entity_rows`
        produces rows in the SHOW TAGS shape, which is what
        :func:`fetch_applied_state` accepts unchanged.  This test wires
        the two together with hand-built rows that match exactly what
        the translator emits, so the parser keeps producing
        ``AppliedObject(kind="Entity")`` after the read-path migration.
        """
        from unittest.mock import MagicMock, patch

        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_entity_rows,
        )

        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            {
                "NAME": "USER",
                "JOIN_KEYS": '["USER_ID"]',
                "DESC": "u",
                "OWNER": "ROLE",
            },
            {
                "NAME": "DEVICE",
                "JOIN_KEYS": '["DEVICE_ID"]',
                "DESC": "",
                "OWNER": "ROLE",
            },
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_entities.return_value = df
        session = MagicMock(name="session")

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            entity_rows = fetch_entity_rows(session, "DB", "SCH")

        state = fetch_applied_state([], None, entity_rows=entity_rows)
        entity_objs = [o for o in state.objects.values() if o.kind == "Entity"]
        assert len(entity_objs) == 2
        names = {o.name for o in entity_objs}
        assert names == {"USER", "DEVICE"}
        user = next(o for o in entity_objs if o.name == "USER")
        assert user.details["join_keys"] == ["USER_ID"]

    def test_datasources_derived_from_specification_map(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"CLICK_FV$V1$ONLINE": self._SPEC_FULL},
        )
        ds_objs = [o for o in state.objects.values() if o.kind == "Datasource"]
        assert len(ds_objs) == 1
        # Datasource names are uppercased to match :func:`invariants.spec_key`,
        # so a re-applied YAML produces the same key and emits ``NO_CHANGE``.
        assert ds_objs[0].name == "USER_EVENTS"

    def test_datasources_deduplicated_across_fvs(self) -> None:
        spec2: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "name": "other_fv",
                "version": "v1",
                "database": "DB",
                "schema": "SCH",
            },
            "spec": {
                "sources": [
                    {"name": "user_events", "source_type": "Stream"},
                ],
                "features": [],
            },
        }
        row2: dict[str, Any] = {
            "name": "OTHER_FV$V1$ONLINE",
            "database_name": "DB",
            "schema_name": "SCH",
        }
        state = fetch_applied_state(
            [self._SHOW_ROW, row2],
            None,
            specification_map={
                "CLICK_FV$V1$ONLINE": self._SPEC_FULL,
                "OTHER_FV$V1$ONLINE": spec2,
            },
        )
        ds_objs = [o for o in state.objects.values() if o.kind == "Datasource"]
        # Same source name across two FVs → only one Datasource object
        assert len(ds_objs) == 1
        # Names are uppercased — see :func:`invariants.spec_key` for rationale.
        assert ds_objs[0].name == "USER_EVENTS"


class TestRealtimeDescRecovery:
    """The deployed ``desc`` must be recovered for a ``RealtimeFeatureView``
    too, not just Batch / Streaming.

    The RTFV SPECIFICATION payload never carries ``desc`` but the list-FV
    row does, and ``desc`` is operational for realtime, so without the
    kind-agnostic hoist a described RTFV re-plans as a spurious ``UPDATE_FV``.
    """

    _RTFV_SPEC: dict[str, Any] = {
        "kind": "RealtimeFeatureView",
        "metadata": {"name": "rt_fv", "version": "v1", "database": "DB", "schema": "SCH"},
        "spec": {
            "ordered_entity_column_names": ["user_id"],
            "sources": [{"name": "req_src", "source_type": "Request"}],
            "udf": {
                "function_name": "transform",
                "function_definition": "def transform(x): return x",
                "language": "python",
                "output_columns": [{"name": "score", "type": "DoubleType"}],
            },
        },
    }

    _SHOW_ROW: dict[str, Any] = {
        "name": "RT_FV$V1$ONLINE",
        "database_name": "DB",
        "schema_name": "SCH",
        "created_on": "2024-01-01 00:00:00",
    }

    def _fv_row(self, desc: str) -> dict[str, Any]:
        return {
            "name": "RT_FV",
            "version": "V1",
            "database_name": "DB",
            "schema_name": "SCH",
            "kind": "REALTIME",
            "entities": ["USER_ID"],
            "desc": desc,
        }

    def test_realtime_desc_injected_at_top_level(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"RT_FV$V1$ONLINE": copy.deepcopy(self._RTFV_SPEC)},
            feature_view_rows=[self._fv_row("rt docs")],
        )
        rt_objs = [o for o in state.objects.values() if o.kind == "RealtimeFeatureView"]
        assert len(rt_objs) == 1
        assert rt_objs[0].spec_payload.get("desc") == "rt docs"

    def test_realtime_empty_desc_not_injected(self) -> None:
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"RT_FV$V1$ONLINE": copy.deepcopy(self._RTFV_SPEC)},
            feature_view_rows=[self._fv_row("")],
        )
        rt_objs = [o for o in state.objects.values() if o.kind == "RealtimeFeatureView"]
        assert len(rt_objs) == 1
        assert "desc" not in rt_objs[0].spec_payload

    def test_realtime_desc_recovered_when_list_row_version_case_differs(self) -> None:
        """A list-FV row whose ``version`` differs only in case from the OFT
        name's parsed version must still match the recovery index.

        Unquoted identifiers in ``SHOW ONLINE FEATURE TABLES`` come back
        upper-cased (``RT_FV$V1$ONLINE`` → parsed ``V1``), but
        ``list_feature_views`` can return the authored case (``v1``).  The
        ``fv_row_by_name_version`` index and its lookup must both
        ``.upper()`` the version — mirroring the ``(name, version)``
        identity in :func:`orphaned_oft_warnings` / :func:`_build_spec_key`
        / :func:`invariants.spec_key` — or the case-only difference silently
        skips desc / ``source_refs`` / ``refresh_freq`` recovery.
        """
        lowercase_version_row = self._fv_row("rt docs")
        lowercase_version_row["version"] = "v1"

        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"RT_FV$V1$ONLINE": copy.deepcopy(self._RTFV_SPEC)},
            feature_view_rows=[lowercase_version_row],
        )
        rt_objs = [o for o in state.objects.values() if o.kind == "RealtimeFeatureView"]
        assert len(rt_objs) == 1
        assert rt_objs[0].spec_payload.get("desc") == "rt docs"


# ---------------------------------------------------------------------------
# Phase B1/B3: ``dt_text_map`` is no longer the recovery path for BatchFV
# source bindings.  ``FV_SOURCE_REFS`` metadata (surfaced via
# ``list_feature_views().source_refs``) is now authoritative; the
# ``dt_text_map`` kwarg is preserved on the API signature for backward
# compatibility but is no longer consulted.  The minimal back-compat tests
# below pin (a) that the API still accepts the kwarg and (b) that
# streaming FVs are not perturbed by anything in the map.
# ---------------------------------------------------------------------------


class TestFetchAppliedStateWithDtTextMap:
    _BATCH_SPEC: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "my_batch_fv",
            "version": "v1",
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
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
        "offline_configs": [
            {
                "database": "JKEW_DB",
                "schema": "JKEW_SCHEMA",
                "table": "MY_BATCH_FV$V1",
                "table_type": "BatchSource",
                "store_type": "snowflake",
            }
        ],
        "online_store_type": "postgres",
    }

    _SHOW_ROW: dict[str, Any] = {
        "name": "MY_BATCH_FV$V1$ONLINE",
        "database_name": "JKEW_DB",
        "schema_name": "JKEW_SCHEMA",
    }

    def test_dt_text_map_is_accepted_for_back_compat(self) -> None:
        """The ``dt_text_map`` kwarg is preserved on the signature for
        back-compat with existing CLI manager call sites.  Phase B1
        removed the DT-text recovery path; the kwarg is now a no-op so
        passing it must not raise.
        """
        state = fetch_applied_state(
            [self._SHOW_ROW],
            None,
            specification_map={"MY_BATCH_FV$V1$ONLINE": copy.deepcopy(self._BATCH_SPEC)},
            dt_text_map={"MY_BATCH_FV$V1": "CREATE DYNAMIC TABLE x AS SELECT * FROM RAW_EVENTS"},
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "BatchFeatureView"]
        assert len(fv_objs) == 1, fv_objs

    def test_dt_text_map_optional_for_non_batch_fvs(self) -> None:
        """Streaming FVs already preserve sources on the deployed side;
        passing ``dt_text_map`` must not perturb them.
        """
        streaming_spec: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "name": "streaming_fv",
                "version": "v1",
                "database": "DB",
                "schema": "SCH",
            },
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [
                    {"name": "USER_EVENTS", "source_type": "Stream"},
                ],
                "features": [],
            },
        }
        show_row: dict[str, Any] = {
            "name": "STREAMING_FV$V1$ONLINE",
            "database_name": "DB",
            "schema_name": "SCH",
        }
        state = fetch_applied_state(
            [show_row],
            None,
            specification_map={"STREAMING_FV$V1$ONLINE": streaming_spec},
            dt_text_map={
                "STREAMING_FV$V1": (
                    "CREATE DYNAMIC TABLE DB.SCH.STREAMING_FV$V1 AS SELECT * FROM DB.SCH.SHOULD_NOT_OVERWRITE"
                )
            },
        )
        fv_objs = [o for o in state.objects.values() if o.kind == "StreamingFeatureView"]
        # Existing source binding survives untouched.
        assert fv_objs[0].spec_payload["spec"]["sources"][0]["name"] == "USER_EVENTS"


# ---------------------------------------------------------------------------
# Phase B1: regex-driven DT-text recovery has been deleted.  The unit tests
# for ``_extract_source_table_from_dt_text``, ``_extract_dt_body_from_text``,
# and ``_classify_dt_body`` were removed alongside the helpers because the
# applied-state recovery layer now consumes authoritative metadata
# (``FV_SOURCE_REFS`` + ``list_feature_views()`` columns) instead of
# string-matching DT bodies.  See the ``TestSourceRefsConsumption`` /
# ``TestLegacyShimFallback`` classes below for the replacement coverage.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase B1/B3: ``_inject_batch_fv_source_from_dt_text`` was deleted.  The
# query-shape and local-name recovery scenarios it covered are now served
# by the ``FvSourceRefsMetadata`` round-trip (operator-authored ``name``,
# ``query``, ``table`` all ride in metadata authoritatively).  See
# ``TestSourceRefsConsumption`` below for the metadata-backed coverage.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_datasources_by_table — helper that maps physical → logical names
# ---------------------------------------------------------------------------
#
# The CLI manager loads the local project first, then constructs the
# lookup dict and threads it into ``fetch_applied_state``.  The helper
# walks every ``BatchSource`` spec in the loaded batch and maps each
# ``BatchSource.table`` → ``BatchSource.name``.  Multi-match (two
# BatchSources declaring the same ``table``) is detected at build time:
# the colliding entry's value carries the list of conflicting logical
# names so the downstream recovery raises with a precise message.
# ---------------------------------------------------------------------------


class TestBuildDatasourcesByTable:
    """``_build_datasources_by_table`` walks loaded ``BatchSource`` specs
    and emits the physical-table → logical-name lookup the recovery
    helper consumes.  Query-backed and query_file-backed sources are
    skipped — only ``table``-backed sources participate in the table
    lookup contract."""

    def _make_batchsource(self, name: str, table: Any = None, query: Any = None, query_file: Any = None) -> Any:
        from snowflake.ml.feature_store.decl.spec_models import BatchSource

        kwargs: dict[str, Any] = {"name": name}
        if table is not None:
            kwargs["table"] = table
        if query is not None:
            kwargs["query"] = query
        if query_file is not None:
            kwargs["query_file"] = query_file
        return BatchSource(**kwargs)

    def test_unique_table_maps_to_logical_name(self) -> None:
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        specs = [
            self._make_batchsource("EVENTS_FG_DECL", table="RAW_EVENTS_FG_DECL"),
        ]
        lookup = _build_datasources_by_table(specs)
        assert lookup == {"RAW_EVENTS_FG_DECL": "EVENTS_FG_DECL"}

    def test_table_lookup_is_uppercased(self) -> None:
        """The recovery side normalises the recovered table identifier
        (uppercased unqualified ident — see ``_classify_dt_body``).  The
        lookup map MUST follow the same convention so the keys collide
        on identical objects regardless of YAML-side casing."""
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        specs = [
            self._make_batchsource("events_fg_decl", table="raw_events_fg_decl"),
        ]
        lookup = _build_datasources_by_table(specs)
        assert "RAW_EVENTS_FG_DECL" in lookup
        # The logical name preserves operator-authored case so the
        # exported YAML matches what the operator typed.
        assert lookup["RAW_EVENTS_FG_DECL"] == "events_fg_decl"

    def test_multi_match_emits_collision_marker(self) -> None:
        """Two ``BatchSource`` YAMLs claim the same physical table.
        The helper must NOT silently pick one — it stores the list of
        conflicting names so the recovery helper can raise with both
        candidates in the message."""
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        specs = [
            self._make_batchsource("EVENTS_FG_DECL", table="RAW_EVENTS_FG_DECL"),
            self._make_batchsource("EVENTS_BACKUP", table="RAW_EVENTS_FG_DECL"),
        ]
        lookup = _build_datasources_by_table(specs)
        # The collision marker is a list of the conflicting logical
        # names (deterministic order — sorted) so the downstream error
        # message is reproducible regardless of spec walk order.
        assert isinstance(lookup["RAW_EVENTS_FG_DECL"], list)
        assert sorted(lookup["RAW_EVENTS_FG_DECL"]) == ["EVENTS_BACKUP", "EVENTS_FG_DECL"]

    def test_query_backed_sources_are_skipped(self) -> None:
        """A ``BatchSource`` with ``query=`` (or ``query_file=``) has no
        physical table identity to participate in the lookup — only
        ``table``-backed sources are eligible."""
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        specs = [
            self._make_batchsource("EVENTS_FG_DECL", table="RAW_EVENTS_FG_DECL"),
            self._make_batchsource("EVENTS_BY_QUERY", query="SELECT * FROM X"),
            self._make_batchsource("EVENTS_BY_FILE", query_file="events.sql"),
        ]
        lookup = _build_datasources_by_table(specs)
        assert lookup == {"RAW_EVENTS_FG_DECL": "EVENTS_FG_DECL"}

    def test_non_batchsource_specs_are_ignored(self) -> None:
        """Entities / FeatureViews / etc. in the spec batch are walked
        through but have no effect on the lookup map."""
        from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        specs = [
            Entity(name="USER_ID", join_keys=[FSColumn(name="USER_ID", type="StringType")]),
            self._make_batchsource("EVENTS_FG_DECL", table="RAW_EVENTS_FG_DECL"),
        ]
        lookup = _build_datasources_by_table(specs)
        assert lookup == {"RAW_EVENTS_FG_DECL": "EVENTS_FG_DECL"}

    def test_empty_specs_returns_empty_map(self) -> None:
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        assert _build_datasources_by_table([]) == {}

    def test_unqualified_table_lookup_strips_db_schema(self) -> None:
        """If the operator authored ``DB.SCHEMA.RAW_EVENTS`` as the
        ``table:`` value, the lookup key must be the unqualified last
        segment so it matches the recovered ident from the legacy
        helper."""
        from snowflake.ml.feature_store.decl.state import _build_datasources_by_table

        specs = [
            self._make_batchsource("EVENTS_FG_DECL", table="JKEW_DB.JKEW_SCHEMA.RAW_EVENTS_FG_DECL"),
        ]
        lookup = _build_datasources_by_table(specs)
        assert lookup == {"RAW_EVENTS_FG_DECL": "EVENTS_FG_DECL"}


# ---------------------------------------------------------------------------
# Phase B1/B3 — metadata-backed source recovery (no regex / no DT-text)
#
# After Phase A landed ``FV_SOURCE_REFS`` as an authoritative metadata row
# and added a ``source_refs`` column to ``list_feature_views()``, the
# applied-state recovery layer reads source bindings directly from
# metadata instead of regex-matching the ``CREATE DYNAMIC TABLE`` body.
# These tests pin that contract.
# ---------------------------------------------------------------------------


class TestInjectBatchFvFields:
    """``_inject_batch_fv_fields_from_list_row`` populates cluster_by,
    refresh_mode, initialize, and the source binding from the
    imperative-API list-FV row and ``fv_obj`` (the result of
    :meth:`FeatureStore.get_feature_view`)."""

    def test_no_duplicate_refresh_mode_initialize_block(self) -> None:
        """Pin that the ``refresh_mode`` / ``initialize`` injection is
        emitted exactly once.

        A prior merge left an identical copy of both injection blocks
        back-to-back inside
        ``_inject_batch_fv_fields_from_list_row``.  The
        ``"refresh_mode" not in inner`` / ``"initialize" not in inner``
        guards made the second copy a permanent no-op, so no behavioural
        test could catch it — this structural pin keeps the dead
        duplicate from reappearing.

        Uses an AST walk (count assignments to the ``inner["refresh_mode"]``
        subscript and reads of ``fv_obj.initialize``) rather than string
        counting, so reformatting the assignment line does not
        false-fail the pin.
        """
        import ast
        import inspect
        import textwrap

        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        tree = ast.parse(textwrap.dedent(inspect.getsource(_inject_batch_fv_fields_from_list_row)))

        refresh_assignments = 0
        initialize_reads = 0
        for node in ast.walk(tree):
            # Count ``inner["refresh_mode"] = ...`` assignments.
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "inner"
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "refresh_mode"
                    ):
                        refresh_assignments += 1
            # Count ``getattr(fv_obj, "initialize", ...)`` reads.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "fv_obj"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "initialize"
            ):
                initialize_reads += 1

        assert refresh_assignments == 1, (
            "_inject_batch_fv_fields_from_list_row must assign "
            f"inner['refresh_mode'] exactly once (found {refresh_assignments}); "
            "remove the duplicated injection block."
        )
        assert initialize_reads == 1, (
            "_inject_batch_fv_fields_from_list_row must read "
            f"fv_obj.initialize exactly once (found {initialize_reads}); "
            "remove the duplicated injection block."
        )

    def test_uses_list_row_columns_no_regex(self) -> None:
        """Pin AS4: after Phase B1 the ``re`` module is never bound in
        ``decl/state.py``'s namespace — every recovery path consumes
        metadata.

        Inspects the already-imported module's globals via
        ``vars(state_module)``; intentionally avoids
        ``sys.modules.pop`` because that would invalidate concurrent
        ``patch.object(state_module, …)`` references in other tests.
        The AST-level scan in :mod:`test_no_regex_in_decl` is the
        source-time companion.
        """
        from snowflake.ml.feature_store.decl import state as state_module

        assert "re" not in vars(state_module), (
            "decl/state.py must not bind the ``re`` module in its "
            "namespace after Phase B1 — recovery is metadata-driven, "
            "not regex-driven."
        )

    def test_desc_injected_at_top_level_for_l3_round_trip(self) -> None:
        """Phase E gap 4: the row's ``desc`` (from
        ``list_feature_views().desc``, which mirrors the deployed DT's
        ``COMMENT``) lands at the TOP level of ``spec_payload`` so the
        planner's ``_resolve_applied_desc`` — which probes top-level
        first — finds the value after an
        ``FeatureStore.update_feature_view(desc=…)`` rewrites the
        deployed DT's comment.  Without this injection a desc-only
        edit closed correctly on the Snowflake side (``ALTER DT … SET
        COMMENT``) but the next ``snow feature plan`` still saw the
        recovered ``spec_payload.desc`` as empty and emitted a
        spurious ``UPDATE_FV`` every replan — the L3 round-trip
        regression that this test pins.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_DESC_L3", "version": "V1"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
        }
        row: dict[str, Any] = {
            "name": "BFV_DESC_L3",
            "version": "V1",
            "desc": "v2 docs",
            "cluster_by": "",
            "refresh_mode": "",
        }
        _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
        assert spec_payload["desc"] == "v2 docs"

        # Idempotent: a spec_payload that already carries ``desc`` (e.g.
        # from the imperative ``_serialize_batch_fv_spec`` enrichment)
        # is NOT overwritten — preserves the principle that the
        # injection helper is purely additive.
        spec_payload_with_desc = dict(spec_payload)
        spec_payload_with_desc["desc"] = "preserved"
        spec_payload_with_desc["spec"] = dict(spec_payload["spec"])
        _inject_batch_fv_fields_from_list_row(spec_payload_with_desc, row, fv_obj=None)
        assert spec_payload_with_desc["desc"] == "preserved"

        # Empty desc cell does not overwrite an existing value and does
        # not stamp a blank key (the planner treats missing == empty).
        spec_payload_empty: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": spec_payload["metadata"],
            "spec": dict(spec_payload["spec"]),
        }
        _inject_batch_fv_fields_from_list_row(
            spec_payload_empty,
            {"name": "BFV_DESC_L3", "version": "V1", "desc": "", "cluster_by": "", "refresh_mode": ""},
            fv_obj=None,
        )
        assert "desc" not in spec_payload_empty

    def test_desc_is_stripped_from_full_spec_hash(self) -> None:
        """Phase E gap 4 companion: ``desc`` / ``description`` must be in
        :data:`invariants._OPERATIONAL_FV_KEYS` so the structural hash
        ignores the operational knob (which has an in-place imperative
        update path via ``ALTER DYNAMIC TABLE … SET COMMENT``).  Two
        otherwise-identical specs differing only on top-level ``desc``
        must produce the same hash; otherwise the planner would emit
        ``RECREATE_FV`` on a pure desc edit instead of ``UPDATE_FV``.
        """
        from snowflake.ml.feature_store.decl.invariants import (
            _OPERATIONAL_FV_KEYS,
            _full_spec_hash,
        )

        assert "desc" in _OPERATIONAL_FV_KEYS
        assert "description" in _OPERATIONAL_FV_KEYS

        base_spec: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "FV", "version": "V1"},
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [{"table": "T", "columns": []}],
                "features": [],
            },
        }
        spec_no_desc = dict(base_spec)
        spec_with_desc = dict(base_spec)
        spec_with_desc["desc"] = "new comment"
        spec_with_description = dict(base_spec)
        spec_with_description["description"] = "new comment"
        assert _full_spec_hash(spec_no_desc) == _full_spec_hash(spec_with_desc)
        assert _full_spec_hash(spec_no_desc) == _full_spec_hash(spec_with_description)

    def test_warehouse_injected_from_list_row(self) -> None:
        """The deployed refresh ``warehouse`` (list-FV ``warehouse`` column)
        lands on ``spec.warehouse``.

        The SPECIFICATION payload never carries the warehouse (it's a DT
        property), so without this injection ``_warehouse_drifted`` sees a
        permanent authored-vs-absent drift and re-plans every time.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_WH", "version": "V1"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
        }
        row: dict[str, Any] = {
            "name": "BFV_WH",
            "version": "V1",
            "warehouse": "WH_AIML",
            "cluster_by": "",
            "refresh_mode": "",
            "desc": "",
        }
        _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
        assert spec_payload["spec"]["warehouse"] == "WH_AIML"

    def test_warehouse_injection_does_not_overwrite_existing(self) -> None:
        """The warehouse injection is additive: a ``spec.warehouse`` that
        the offline enrichment path (``_serialize_batch_fv_spec``) already
        populated is preserved, never clobbered by the list-row cell.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_WH", "version": "V1"},
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [],
                "features": [],
                "warehouse": "WH_PREEXISTING",
            },
        }
        row: dict[str, Any] = {
            "name": "BFV_WH",
            "version": "V1",
            "warehouse": "WH_FROM_ROW",
            "cluster_by": "",
            "refresh_mode": "",
            "desc": "",
        }
        _inject_batch_fv_fields_from_list_row(spec_payload, row, fv_obj=None)
        assert spec_payload["spec"]["warehouse"] == "WH_PREEXISTING"

    def test_empty_warehouse_cell_not_injected(self) -> None:
        """An empty / whitespace ``warehouse`` cell (realtime and static
        view-backed FVs report SQL ``NULL`` → ``""``) does not stamp a
        blank ``spec.warehouse`` key.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_WH", "version": "V1"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
        }
        _inject_batch_fv_fields_from_list_row(
            spec_payload,
            {"name": "BFV_WH", "version": "V1", "warehouse": "  ", "cluster_by": "", "refresh_mode": ""},
            fv_obj=None,
        )
        assert "warehouse" not in spec_payload["spec"]

    def test_append_only_injected_from_list_row_when_true(self) -> None:
        """A truthy ``append_only`` cell injects ``spec.append_only = True``,
        accepting the transport spellings ``_coerce_applied_bool`` normalises.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        for truthy in (True, "true", "TRUE", "1", 1):
            spec_payload: dict[str, Any] = {
                "kind": "BatchFeatureView",
                "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_AO", "version": "V1"},
                "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
            }
            _inject_batch_fv_fields_from_list_row(
                spec_payload,
                {"name": "BFV_AO", "version": "V1", "append_only": truthy, "cluster_by": "", "refresh_mode": ""},
                fv_obj=None,
            )
            assert spec_payload["spec"].get("append_only") is True, f"cell {truthy!r} should inject True"

    def test_append_only_not_injected_when_falsy_or_absent(self) -> None:
        """The default ``False`` (and its string / ``None`` / absent
        spellings) must NOT stamp ``spec.append_only``.  In particular the
        string ``"false"`` (truthy in Python) must not be read as enabled.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        for falsy in (False, "false", "FALSE", "0", 0, None):
            spec_payload: dict[str, Any] = {
                "kind": "BatchFeatureView",
                "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_AO", "version": "V1"},
                "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
            }
            _inject_batch_fv_fields_from_list_row(
                spec_payload,
                {"name": "BFV_AO", "version": "V1", "append_only": falsy, "cluster_by": "", "refresh_mode": ""},
                fv_obj=None,
            )
            assert "append_only" not in spec_payload["spec"], f"cell {falsy!r} must not inject"

        # Absent cell, no fv_obj → not injected.
        spec_payload_absent: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_AO", "version": "V1"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
        }
        _inject_batch_fv_fields_from_list_row(
            spec_payload_absent,
            {"name": "BFV_AO", "version": "V1", "cluster_by": "", "refresh_mode": ""},
            fv_obj=None,
        )
        assert "append_only" not in spec_payload_absent["spec"]

    def test_append_only_falls_back_to_fv_obj(self) -> None:
        """When the row omits ``append_only`` the rehydrated FeatureView's
        ``append_only`` attribute is the fallback source.
        """
        from snowflake.ml.feature_store.decl.state import (
            _inject_batch_fv_fields_from_list_row,
        )

        class _FvObj:
            append_only = True
            initialize = None

        spec_payload: dict[str, Any] = {
            "kind": "BatchFeatureView",
            "metadata": {"database": "DB", "schema": "SCH", "name": "BFV_AO", "version": "V1"},
            "spec": {"ordered_entity_column_names": ["USER_ID"], "sources": [], "features": []},
        }
        _inject_batch_fv_fields_from_list_row(
            spec_payload,
            {"name": "BFV_AO", "version": "V1", "cluster_by": "", "refresh_mode": ""},
            fv_obj=_FvObj(),
        )
        assert spec_payload["spec"].get("append_only") is True


class TestTiledBfvSpecRecovery:
    """Phase B2: tiled BatchFV recovery enriches the spec_payload from
    ``FvSourceRefsMetadata`` + ``list_feature_views()`` columns +
    :meth:`FeatureStore.get_feature_view` (via the new
    ``fetch_feature_view_object`` helper) instead of DT-text parsing.
    """

    def test_features_granularity_timestamp_secondary_keys_recovered_from_imperative_api(self) -> None:
        """For an offline-only tiled BFV, the recovered spec_payload
        must surface ``feature_granularity``, ``timestamp_col``, and
        ``aggregation_secondary_keys`` from the imperative API path
        (``spec_text`` enrichment via ``fetch_feature_view_rows``).

        Pin: ``feature_granularity_sec`` is derived locally from the
        granularity string via :func:`interval_utils.interval_to_seconds`
        (no SQL parsing required).
        """
        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        # Tiled BFV row in the shape ``fetch_feature_view_rows`` emits
        # post-enrichment: ``spec_text`` is the SPECIFICATION-shaped
        # dict ``_serialize_batch_fv_spec`` produces.
        fv_row: dict[str, Any] = {
            "name": "TILED_FV",
            "version": "V1",
            "database_name": "DB",
            "schema_name": "SCH",
            "kind": "BATCH",
            "entities": ["USER"],
            "online_enabled": False,
            "target_lag": "",
            "refresh_freq": "300 seconds",
            "warehouse": "WH",
            "desc": "",
            "physical_dt_name": "TILED_FV$V1",
            "spec_text": {
                "kind": "BatchFeatureView",
                "metadata": {
                    "database": "DB",
                    "schema": "SCH",
                    "name": "TILED_FV",
                    "version": "V1",
                },
                "offline_configs": [
                    {
                        "store_type": "snowflake",
                        "table_type": "BatchTable",
                        "database": "DB",
                        "schema": "SCH",
                        "table": "TILED_FV$V1",
                        "columns": [],
                    }
                ],
                "spec": {
                    "ordered_entity_column_names": ["USER_ID"],
                    "sources": [],
                    "features": [
                        {
                            "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                            "output_column": {"name": "_PARTIAL_SUM_AMOUNT", "type": "DoubleType"},
                            "function": "SUM",
                            "window_sec": 300,
                        }
                    ],
                    "target_lag_sec": 300,
                    "timestamp_col": "EVENT_TS",
                    "feature_granularity": "60s",
                    "feature_granularity_sec": 60,
                    "aggregation_secondary_keys": ["DEVICE_ID"],
                },
            },
        }

        state = fetch_applied_state(
            [],
            None,
            feature_view_rows=[fv_row],
            default_database="DB",
            default_schema="SCH",
        )

        bfv_objs = [o for o in state.objects.values() if o.kind == "BatchFeatureView"]
        assert len(bfv_objs) == 1, bfv_objs
        inner = bfv_objs[0].spec_payload["spec"]
        assert inner["feature_granularity"] == "60s"
        assert inner["feature_granularity_sec"] == 60
        assert inner["timestamp_col"] == "EVENT_TS"
        assert inner["aggregation_secondary_keys"] == ["DEVICE_ID"]


class TestSourceRefsConsumption:
    """Phase B3: ``_inject_batch_fv_source_from_metadata`` populates
    ``spec.sources[]`` directly from the ``source_refs`` column on a
    ``list_feature_views()`` row.  Closes Bug 1 (column round-trip)
    and Bug 3 (tiled BFV source name)."""

    _BATCH_SPEC: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "my_batch_fv",
            "version": "v1",
            "database": "DB",
            "schema": "SCH",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [],
            "target_lag_sec": 60,
        },
        "offline_configs": [
            {
                "database": "DB",
                "schema": "SCH",
                "table": "MY_BATCH_FV$V1",
                "table_type": "BatchSource",
                "store_type": "snowflake",
            }
        ],
    }

    def _bfv_row(self, source_refs: Any) -> Any:
        """Build a minimal ``feature_view_rows`` entry carrying the
        FV metadata + ``spec_text`` + ``source_refs`` for the
        applied-state path that the orchestrator exercises.

        Args:
            source_refs: The ``FV_SOURCE_REFS`` payload (a
                ``list[dict]``) to attach to the row.  Tests pass
                whatever shape they want to exercise — fully-populated
                Batch refs, empty list, etc.

        Returns:
            A dict matching the
            :func:`imperative_executor.fetch_feature_view_rows` row
            shape.
        """
        return {
            "name": "my_batch_fv",
            "version": "v1",
            "database_name": "DB",
            "schema_name": "SCH",
            "kind": "BATCH",
            "entities": ["USER_ID"],
            "online_enabled": False,
            "target_lag": "",
            "refresh_freq": "60 seconds",
            "warehouse": "WH",
            "desc": "",
            "physical_dt_name": "MY_BATCH_FV$V1",
            "spec_text": copy.deepcopy(self._BATCH_SPEC),
            "source_refs": source_refs,
        }

    def test_columns_round_trip_from_metadata(self) -> None:
        """Authored ``columns`` ride through ``FV_SOURCE_REFS`` unchanged
        and land on ``spec.sources[0].columns``.

        Pin for AS1: ``columns`` are authoritative from metadata, no
        symmetry shim required."""
        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        source_refs = [
            {
                "name": "EVENTS_FG_DECL",
                "source_type": "Batch",
                "table": "RAW_EVENTS_FG_DECL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ]
        row = self._bfv_row(source_refs)
        state = fetch_applied_state(
            [],
            None,
            feature_view_rows=[row],
            default_database="DB",
            default_schema="SCH",
        )
        bfv = next(o for o in state.objects.values() if o.kind == "BatchFeatureView")
        sources = bfv.spec_payload["spec"]["sources"]
        assert len(sources) == 1
        assert sources[0]["name"] == "EVENTS_FG_DECL"
        assert sources[0]["table"] == "RAW_EVENTS_FG_DECL"
        assert sources[0]["source_type"] == "Batch"
        col_names = [c["name"] for c in sources[0].get("columns") or []]
        assert col_names == ["USER_ID", "EVENT_TS", "AMOUNT"]

    def test_query_shape_uses_authored_name_not_synthetic(self) -> None:
        """A query-shape BatchFV (``BatchSource.query``) preserves the
        operator-authored ``name`` instead of the synthetic
        ``<FV>__SOURCE`` placeholder the regex path used to emit.

        Pin for Bug 1: the source key in applied state matches the
        local YAML's ``name:`` field so re-plan resolves NO_CHANGE."""
        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        source_refs = [
            {
                "name": "EVENTS_QUERY_SRC",
                "source_type": "Batch",
                "query": "SELECT * FROM RAW.EVENTS WHERE ts > '2024-01-01'",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                ],
            }
        ]
        row = self._bfv_row(source_refs)
        state = fetch_applied_state(
            [],
            None,
            feature_view_rows=[row],
            default_database="DB",
            default_schema="SCH",
        )
        bfv = next(o for o in state.objects.values() if o.kind == "BatchFeatureView")
        sources = bfv.spec_payload["spec"]["sources"]
        assert len(sources) == 1
        assert sources[0]["name"] == "EVENTS_QUERY_SRC"
        assert sources[0]["query"] == "SELECT * FROM RAW.EVENTS WHERE ts > '2024-01-01'"
        assert "table" not in sources[0]
        # No synthetic fallback name (the regex path's contract); the
        # operator-authored name is the source of truth.
        assert not any(s["name"].endswith("__SOURCE") for s in sources)

    def test_tiled_bfv_source_name_is_operator_authored_not_inner_materialization(self) -> None:
        """A tiled BatchFV's recovered source ``name`` equals the
        operator's authored ``BatchSource.name`` — NOT the inner
        materialised DT name (the regex path's wrong answer).

        Pin for Bug 3 (AS3): tiled DT body ``SELECT … FROM
        (SELECT * FROM RAW_EVENTS_TBL) GROUP BY …`` used to leak
        the inner ``RAW_EVENTS_TBL`` identifier into ``sources[0].name``
        whenever ``datasources_by_table`` had no entry."""
        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        # The tiled DT inner-materialisation table is "RAW_EVENTS_TBL".
        # The operator authored the source as "EVENTS_STREAM_SRC"
        # pointing at the same table — metadata preserves that.
        source_refs = [
            {
                "name": "EVENTS_STREAM_SRC",
                "source_type": "Batch",
                "table": "RAW_EVENTS_TBL",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ]
        row = self._bfv_row(source_refs)
        state = fetch_applied_state(
            [],
            None,
            feature_view_rows=[row],
            default_database="DB",
            default_schema="SCH",
        )
        bfv = next(o for o in state.objects.values() if o.kind == "BatchFeatureView")
        sources = bfv.spec_payload["spec"]["sources"]
        assert sources[0]["name"] == "EVENTS_STREAM_SRC"
        assert sources[0]["table"] == "RAW_EVENTS_TBL"
        # The Datasource AppliedObject derived from the metadata path
        # must carry the same authored name (Bug 3's recovery side).
        ds_objs = [o for o in state.objects.values() if o.kind == "Datasource"]
        ds_names = {o.name for o in ds_objs}
        assert "EVENTS_STREAM_SRC" in ds_names
        # No synthetic source key derived from the physical table.
        assert "RAW_EVENTS_TBL" not in ds_names


class TestLegacyShimFallback:
    """Legacy (pre-``FV_SOURCE_REFS``) source-recovery behaviour.  When a
    list-FV row carries ``source_refs`` the metadata path supplies
    ``sources[0]`` and no warning is emitted.  When the column is absent
    (legacy pre-A1 deployment) ``sources`` stays empty and
    :func:`state._warn_if_sources_unrecovered` warns that the FV will be
    recreated on the next apply to stamp the metadata (the
    ``_build_datasources_by_table`` name-lookup shim was removed)."""

    _BATCH_SPEC: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "metadata": {
            "name": "legacy_fv",
            "version": "v1",
            "database": "DB",
            "schema": "SCH",
        },
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [],
            "features": [],
            "target_lag_sec": 60,
        },
        "offline_configs": [
            {
                "database": "DB",
                "schema": "SCH",
                "table": "LEGACY_FV$V1",
                "table_type": "BatchSource",
                "store_type": "snowflake",
            }
        ],
    }

    def test_fv_without_source_refs_falls_back_to_shim_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A FV row missing ``source_refs`` (legacy deployment) keeps
        ``sources = []`` and emits a ``logger.warning`` telling the
        operator the FV will be recreated on the next apply to stamp the
        metadata.

        Pin for AS10: the unrecovered-source warning is gated on the
        recovered ``sources`` still being empty and is visible at WARNING
        level.

        Args:
            caplog: Pytest log-capture fixture used to assert that
                ``decl.state`` emits the unrecovered-source warning.
        """
        import logging

        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        row: dict[str, Any] = {
            "name": "legacy_fv",
            "version": "v1",
            "database_name": "DB",
            "schema_name": "SCH",
            "kind": "BATCH",
            "entities": ["USER_ID"],
            "online_enabled": False,
            "target_lag": "",
            "refresh_freq": "60 seconds",
            "warehouse": "WH",
            "desc": "",
            "physical_dt_name": "LEGACY_FV$V1",
            "spec_text": copy.deepcopy(self._BATCH_SPEC),
            # No ``source_refs`` key — simulates a pre-A1 deployment.
        }
        caplog.set_level(logging.WARNING, logger="snowflake.ml.feature_store.decl.state")
        state = fetch_applied_state(
            [],
            None,
            feature_view_rows=[row],
            default_database="DB",
            default_schema="SCH",
        )
        bfv = next(o for o in state.objects.values() if o.kind == "BatchFeatureView")
        # Legacy fallback leaves ``sources`` empty; the operator must run apply
        # once so the FV is recreated and the source metadata row is stamped.
        assert bfv.spec_payload["spec"]["sources"] == []
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("legacy_fv" in m and "will be recreated" in m.lower() for m in warn_msgs), warn_msgs

    def test_fv_with_source_refs_skips_shim(self, caplog: pytest.LogCaptureFixture) -> None:
        """When ``source_refs`` is present the legacy fallback is NOT
        invoked — no warning is logged and the metadata path supplies
        ``sources[0]`` end-to-end.

        Args:
            caplog: Pytest log-capture fixture used to assert that
                ``decl.state`` does NOT emit the legacy-fallback
                warning when authoritative metadata is available.
        """
        import logging

        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        row: dict[str, Any] = {
            "name": "modern_fv",
            "version": "v1",
            "database_name": "DB",
            "schema_name": "SCH",
            "kind": "BATCH",
            "entities": ["USER_ID"],
            "online_enabled": False,
            "target_lag": "",
            "refresh_freq": "60 seconds",
            "warehouse": "WH",
            "desc": "",
            "physical_dt_name": "MODERN_FV$V1",
            "spec_text": copy.deepcopy(self._BATCH_SPEC),
            "source_refs": [
                {
                    "name": "MY_LOGICAL_SRC",
                    "source_type": "Batch",
                    "table": "MY_PHYSICAL_TBL",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
        }
        caplog.set_level(logging.WARNING, logger="snowflake.ml.feature_store.decl.state")
        state = fetch_applied_state(
            [],
            None,
            feature_view_rows=[row],
            default_database="DB",
            default_schema="SCH",
        )
        bfv = next(o for o in state.objects.values() if o.kind == "BatchFeatureView")
        assert bfv.spec_payload["spec"]["sources"][0]["name"] == "MY_LOGICAL_SRC"
        assert bfv.spec_payload["spec"]["sources"][0]["table"] == "MY_PHYSICAL_TBL"
        # No legacy-shim warning because source_refs supplied everything.
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("modern_fv" in m and "re-apply" in m.lower() for m in warn_msgs), warn_msgs

    def test_fv_without_source_refs_warning_says_will_be_recreated(self, caplog: pytest.LogCaptureFixture) -> None:
        """Bug C (Fix 2): when a legacy FV has no source_refs, the warning
        must tell the operator the FV will be recreated once to stamp metadata.

        The old shim message said 're-apply'; the new actionable message says
        'will be recreated' so operators understand what happens next.

        Args:
            caplog: Pytest log-capture fixture.
        """
        import copy
        import logging

        from snowflake.ml.feature_store.decl.state import fetch_applied_state

        row: dict[str, Any] = {
            "name": "legacy_fv",
            "version": "v1",
            "database_name": "DB",
            "schema_name": "SCH",
            "kind": "BATCH",
            "entities": ["USER_ID"],
            "online_enabled": False,
            "target_lag": "",
            "refresh_freq": "60 seconds",
            "warehouse": "WH",
            "desc": "",
            "physical_dt_name": "LEGACY_FV$V1",
            "spec_text": copy.deepcopy(self._BATCH_SPEC),
            # No source_refs — simulates a pre-A1 deployment.
        }
        caplog.set_level(logging.WARNING, logger="snowflake.ml.feature_store.decl.state")
        fetch_applied_state(
            [],
            None,
            feature_view_rows=[row],
            default_database="DB",
            default_schema="SCH",
        )
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "legacy_fv" in m and "will be recreated" in m.lower() for m in warn_msgs
        ), f"Expected 'will be recreated' in warning; got: {warn_msgs}"


# ---------------------------------------------------------------------------
# Bug A regression: phantom source suppression
# ---------------------------------------------------------------------------

_DB = "JKEW_DB"
_SCH = "JKEW_SCHEMA"


def _fg_backed_bfv_spec_payload(fv_name: str = "USER_CLICKS_FG_DECL") -> dict[str, Any]:
    """Return a spec_payload dict for an FG-backed BFV with phantom source.

    The sources[] entry has name == fv_name, the executor artifact that
    triggers the phantom-source bug (Bug A).

    Args:
        fv_name: Feature view name; also used as the source name in sources[].

    Returns:
        A spec_payload dict matching what fetch_applied_state builds for an
        FG-backed BFV.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {"database": _DB, "schema": _SCH, "name": fv_name, "version": "V1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [{"name": fv_name, "kind": "FeatureGroup"}],
            "features": [],
        },
    }


def _regular_bfv_spec_payload(
    fv_name: str = "MY_BATCH_FV",
    source_name: str = "RAW_EVENTS",
    source_table: str = "RAW_EVENTS_TABLE",
) -> dict[str, Any]:
    """Return a spec_payload dict for a BFV with an independent BatchSource.

    The source name differs from the FV name (not an FG-backed artifact).

    Args:
        fv_name: Feature view name.
        source_name: Logical name for the operator-authored BatchSource.
        source_table: Physical table backing the source.

    Returns:
        A spec_payload dict whose sources[0].name != fv_name.
    """
    return {
        "kind": "BatchFeatureView",
        "metadata": {"database": _DB, "schema": _SCH, "name": fv_name, "version": "V1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": source_name,
                    "source_type": "Batch",
                    "table": source_table,
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                }
            ],
            "features": [],
        },
    }


class TestDatasourceObjectsPhantomSourceSuppression:
    """Bug A regression: FG-backed BFV must not produce a phantom BatchSource."""

    def test_fg_backed_bfv_does_not_produce_phantom_source(self) -> None:
        """_datasource_objects_from_specs must not create a Datasource whose
        name matches an already-recovered FeatureView name (FG-backed BFV pattern).
        """
        fv_name = "USER_CLICKS_FG_DECL"
        known_fv_names = {fv_name.upper()}
        result = _datasource_objects_from_specs(
            [_fg_backed_bfv_spec_payload(fv_name)], _DB, _SCH, known_fv_names=known_fv_names
        )
        assert fv_name.upper() not in [
            o.name for o in result
        ], f"Phantom Datasource for FG-backed BFV {fv_name!r} must not appear in applied state."

    def test_fg_backed_bfv_without_filter_produces_phantom(self) -> None:
        """Without known_fv_names filter, the phantom source IS created.

        This documents the pre-fix behaviour and confirms the test harness
        can observe the phantom — i.e. the fix actually suppresses something
        real rather than testing a vacuous condition.
        """
        fv_name = "USER_CLICKS_FG_DECL"
        result = _datasource_objects_from_specs([_fg_backed_bfv_spec_payload(fv_name)], _DB, _SCH)
        assert fv_name.upper() in [
            o.name for o in result
        ], "Without the filter, the phantom source should be present (documents pre-fix behaviour)."

    def test_independent_batch_source_is_preserved(self) -> None:
        """A source whose name does NOT match any FV name must still appear."""
        fv_name = "MY_BATCH_FV"
        source_name = "RAW_EVENTS"
        result = _datasource_objects_from_specs(
            [_regular_bfv_spec_payload(fv_name, source_name)], _DB, _SCH, known_fv_names={fv_name.upper()}
        )
        assert source_name.upper() in [
            o.name for o in result
        ], f"Independent BatchSource {source_name!r} must survive the filter."

    def test_multiple_fg_backed_bfvs_all_suppressed(self) -> None:
        """All FG-backed phantom sources are suppressed when the full FV name
        set is provided — mirrors the USER_CLICKS + USER_AMOUNTS pair from the
        live symptom described in plans/bug_a_phantom_source_drop.md.
        """
        fv_names = ["USER_CLICKS_FG_DECL", "USER_AMOUNTS_FG_DECL"]
        known_fv_names = {n.upper() for n in fv_names}
        result = _datasource_objects_from_specs(
            [_fg_backed_bfv_spec_payload(n) for n in fv_names], _DB, _SCH, known_fv_names=known_fv_names
        )
        source_names = [o.name for o in result]
        for fv_name in fv_names:
            assert fv_name.upper() not in source_names, f"Phantom for {fv_name!r} must be suppressed."


class TestParseClusterByIdentifierResolution:
    """`_parse_cluster_by_list` must resolve quoted identifiers via the
    Snowflake identifier library rather than a naive ``.strip('"')``.

    A quoted identifier with an internal quote comes back from Snowflake as
    ``"FOO""BAR"`` (the doubled quote is the escaped inner quote).  A plain
    ``.strip('"')`` yields the mangled ``FOO""BAR``; ``resolve_identifier``
    keeps it as the canonical ``"FOO""BAR"`` instead.
    """

    def test_json_array_bare_uppercase_columns(self) -> None:
        """Canonical applied shape — a JSON-array literal of bare columns."""
        assert _parse_cluster_by_list('["USER_ID","TILE_START"]') == ["USER_ID", "TILE_START"]

    def test_json_array_single_element_list_wrapping(self) -> None:
        """Snowpark hands back a single-element list wrapping the JSON literal."""
        assert _parse_cluster_by_list(['["USER_ID","TILE_START"]']) == ["USER_ID", "TILE_START"]

    def test_json_array_internal_quote_is_not_mangled(self) -> None:
        """``"FOO""BAR"`` must round-trip as-is, not the mangled ``FOO""BAR``.

        Snowflake emits the ``cluster_by`` cell via ``json.dumps`` of the
        identifier list, so the escaped inner quotes arrive as a JSON-array
        literal string (``["\\"FOO\\"\\"BAR\\""]``).
        """
        raw = json.dumps(['"FOO""BAR"'])
        assert _parse_cluster_by_list(raw) == ['"FOO""BAR"']

    def test_json_array_case_sensitive_identifier_kept_quoted(self) -> None:
        """A case-sensitive quoted identifier stays quoted after resolution."""
        raw = json.dumps(['"MyCol"'])
        assert _parse_cluster_by_list(raw) == ['"MyCol"']

    def test_native_list_quoted_uppercase_unquoted(self) -> None:
        """Native ARRAY of quoted uppercase identifiers resolves to bare form."""
        assert _parse_cluster_by_list(['"USER_ID"', '"SESSION_ID"']) == ["USER_ID", "SESSION_ID"]

    def test_native_list_internal_quote_is_not_mangled(self) -> None:
        """Internal-quote identifier in a native list keeps its escaped form."""
        assert _parse_cluster_by_list(['"FOO""BAR"']) == ['"FOO""BAR"']

    def test_comma_separated_unquoted_is_uppercased(self) -> None:
        """Legacy comma-separated unquoted identifiers resolve to UPPER case."""
        assert _parse_cluster_by_list("user_id, session_id") == ["USER_ID", "SESSION_ID"]

    def test_none_and_empty_return_none(self) -> None:
        """No usable value yields ``None`` (unchanged contract)."""
        assert _parse_cluster_by_list(None) is None
        assert _parse_cluster_by_list("") is None
        assert _parse_cluster_by_list([]) is None


class TestOrphanedOftDiagnostic:
    """Diagnostic-only SHOW OFT pass (list-driven-discovery cutover).

    After the FV-retrieval unification, ``list_feature_views()`` is the
    authoritative discovery source and ``SHOW ONLINE FEATURE TABLES`` is
    demoted to a diagnostic side channel.  :func:`orphaned_oft_warnings`
    flags any OFT whose ``(name, version)`` has no matching
    ``list_feature_views`` row (nor a ``list_feature_groups`` row) — the
    genuinely unrecoverable case where the backing Dynamic Table was
    dropped but the OFT remained.  It never warns for a consistent OFT
    and never suppresses discovery (that is the list-driven path's job).
    """

    _DB_ = "JKEW_DB"
    _SCH_ = "JKEW_SCHEMA"

    def _oft_row(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "database_name": self._DB_,
            "schema_name": self._SCH_,
            "created_on": "2024-01-01 00:00:00",
        }

    def _fv_row(self, name: str, version: str) -> dict[str, Any]:
        return {
            "name": name,
            "version": version,
            "database_name": self._DB_,
            "schema_name": self._SCH_,
            "kind": "BATCH",
            "entities": ["USER_ID"],
            "online_enabled": True,
            "physical_dt_name": f"{name}${version}",
        }

    def test_consistent_oft_emits_no_warning(self) -> None:
        """An OFT whose FV IS listed by ``list_feature_views`` is
        consistent — no diagnostic warning.
        """
        warnings = orphaned_oft_warnings(
            [self._oft_row("USER_CLICKS$V1$ONLINE")],
            feature_view_rows=[self._fv_row("USER_CLICKS", "V1")],
        )
        assert warnings == []

    def test_orphan_oft_emits_named_warning(self) -> None:
        """An OFT with NO matching ``list_feature_views`` row (backing DT
        dropped) must surface exactly one named diagnostic warning.
        """
        warnings = orphaned_oft_warnings(
            [self._oft_row("GHOST_FV$V1$ONLINE")],
            feature_view_rows=[self._fv_row("USER_CLICKS", "V1")],
        )
        assert len(warnings) == 1
        assert "GHOST_FV" in warnings[0]

    def test_feature_group_backing_oft_not_flagged(self) -> None:
        """A FeatureGroup registers an OFT that ``list_feature_views``
        does not return; cross-checking ``feature_group_rows`` avoids a
        false-positive orphan warning.
        """
        warnings = orphaned_oft_warnings(
            [self._oft_row("MY_FG$V1$ONLINE")],
            feature_view_rows=[self._fv_row("USER_CLICKS", "V1")],
            feature_group_rows=[{"name": "MY_FG", "version": "V1"}],
        )
        assert warnings == []

    def test_none_feature_view_rows_disables_crosscheck(self) -> None:
        """When ``feature_view_rows`` is ``None`` the caller did not fetch
        the list — the diagnostic cannot cross-check and must stay silent
        rather than false-flag every OFT.
        """
        warnings = orphaned_oft_warnings(
            [self._oft_row("USER_CLICKS$V1$ONLINE")],
            feature_view_rows=None,
        )
        assert warnings == []

    def test_empty_feature_view_rows_flags_every_oft(self) -> None:
        """An explicit empty list means "no FVs are registered"; every
        OFT is therefore orphaned.
        """
        warnings = orphaned_oft_warnings(
            [self._oft_row("A$V1$ONLINE"), self._oft_row("B$V2$ONLINE")],
            feature_view_rows=[],
        )
        assert len(warnings) == 2


if __name__ == "__main__":
    pytest_driver.main()
