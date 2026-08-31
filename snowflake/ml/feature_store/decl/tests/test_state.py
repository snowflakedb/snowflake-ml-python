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
        """Key must be StreamingFeatureView:DB.SCH:CLICK_FV:V1 (versioned identity)."""
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
    """Phase B4: ``_build_datasources_by_table`` becomes a fallback-only
    shim.  When a list-FV row carries ``source_refs`` we skip the shim
    entirely.  When the column is absent (legacy pre-A1 deployment) we
    emit a once-per-FV warning and fall back to the shim path."""

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
        ``sources = []`` and emits a ``logger.warning`` recommending
        re-apply so the operator notices that source recovery is
        degraded.

        Pin for AS10: legacy shim engagement is gated on the
        absence of ``source_refs`` and is visible at WARNING level.

        Args:
            caplog: Pytest log-capture fixture used to assert that
                ``decl.state`` emits the legacy-fallback warning.
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
        # Legacy fallback leaves ``sources`` empty (the shim is only a
        # name lookup, not a source synthesizer) — this is the signal
        # to the operator that re-apply is needed.
        assert bfv.spec_payload["spec"]["sources"] == []
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("legacy_fv" in m and "re-apply" in m.lower() for m in warn_msgs), warn_msgs

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
        "metadata": {
            "database": _DB,
            "schema": _SCH,
            "name": fv_name,
            "version": "V1",
        },
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
        "metadata": {
            "database": _DB,
            "schema": _SCH,
            "name": fv_name,
            "version": "V1",
        },
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
    """Bug A regression: FG-backed BFV must not produce phantom BatchSource."""

    def test_fg_backed_bfv_does_not_produce_phantom_source(self) -> None:
        """_datasource_objects_from_specs must not create a Datasource whose
        name matches an already-recovered FeatureView name (FG-backed BFV pattern).
        """
        fv_name = "USER_CLICKS_FG_DECL"
        spec_payload = _fg_backed_bfv_spec_payload(fv_name)
        known_fv_names = {fv_name.upper()}

        result = _datasource_objects_from_specs(
            [spec_payload],
            _DB,
            _SCH,
            known_fv_names=known_fv_names,
        )
        source_names = [o.name for o in result]
        assert (
            fv_name.upper() not in source_names
        ), f"Phantom Datasource for FG-backed BFV {fv_name!r} must not appear in applied state."

    def test_fg_backed_bfv_without_filter_produces_phantom(self) -> None:
        """Without known_fv_names filter, the phantom source IS created.

        This documents the pre-fix behaviour and confirms the test harness
        can observe the phantom — i.e. the fix actually suppresses something
        real rather than testing a vacuous condition.
        """
        fv_name = "USER_CLICKS_FG_DECL"
        spec_payload = _fg_backed_bfv_spec_payload(fv_name)

        result = _datasource_objects_from_specs(
            [spec_payload],
            _DB,
            _SCH,
            # no known_fv_names — legacy/unfixed path
        )
        source_names = [o.name for o in result]
        assert fv_name.upper() in source_names, (
            "Without the filter, the phantom source should be present " "(documents pre-fix behaviour)."
        )

    def test_independent_batch_source_is_preserved(self) -> None:
        """A source whose name does NOT match any FV name must still appear."""
        fv_name = "MY_BATCH_FV"
        source_name = "RAW_EVENTS"
        spec_payload = _regular_bfv_spec_payload(fv_name, source_name)
        # FV name in the filter set, but source name (RAW_EVENTS) is not.
        known_fv_names = {fv_name.upper()}

        result = _datasource_objects_from_specs(
            [spec_payload],
            _DB,
            _SCH,
            known_fv_names=known_fv_names,
        )
        source_names = [o.name for o in result]
        assert source_name.upper() in source_names, f"Independent BatchSource {source_name!r} must survive the filter."

    def test_multiple_fg_backed_bfvs_all_suppressed(self) -> None:
        """All FG-backed phantom sources are suppressed when the full FV name
        set is provided — mirrors the USER_CLICKS + USER_AMOUNTS pair from the
        live symptom described in plans/bug_a_phantom_source_drop.md.
        """
        fv_names = ["USER_CLICKS_FG_DECL", "USER_AMOUNTS_FG_DECL"]
        specs = [_fg_backed_bfv_spec_payload(n) for n in fv_names]
        known_fv_names = {n.upper() for n in fv_names}

        result = _datasource_objects_from_specs(
            specs,
            _DB,
            _SCH,
            known_fv_names=known_fv_names,
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


if __name__ == "__main__":
    pytest_driver.main()
