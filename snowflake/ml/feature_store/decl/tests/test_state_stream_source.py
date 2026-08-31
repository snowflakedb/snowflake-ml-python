"""Wave 1C + Wave 2A tests — stream-source applied-state read path.

Wave 1C pins the contract that a single row produced by
:func:`imperative_executor.fetch_stream_source_rows` (canonical shape
``{"name", "schema", "desc", "owner"}``) is reified as one
``AppliedObject(kind="Datasource")`` whose ``key`` / ``spec_payload``
shape collides cleanly with the shape that
:func:`state._datasource_objects_from_specs` already produces.

Wave 2A pins the ``fetch_applied_state(stream_source_rows=...)`` merge
contract from ``plans/stream_source_contract.md`` §5b: runtime rows
are authoritative on key collision, FV-derived datasource entries
only fill keys the runtime did not already claim, and the existing
FV-derived-only behaviour is preserved when the new kwarg is omitted
or empty.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from snowflake.ml.feature_store.decl.invariants import structural_fingerprint_hash
from snowflake.ml.feature_store.decl.state import (
    _build_stream_source_object,
    fetch_applied_state,
)
from snowflake.ml.feature_store.decl.types import ObjectKind
from snowflake.ml.test_utils import pytest_driver


def _row(
    *,
    name: str = "CLICKSTREAM_EVENTS",
    schema: list[dict[str, str]] | None = None,
    desc: str = "click events",
    owner: str = "ROLE_X",
) -> dict[str, Any]:
    """Canonical row shape returned by ``fetch_stream_source_rows``."""
    if schema is None:
        schema = [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT_TS", "type": "TimestampType"},
        ]
    return {
        "name": name,
        "schema": schema,
        "desc": desc,
        "owner": owner,
    }


class TestBuildStreamSourceObject:
    """Contract: §5a of plans/stream_source_contract.md."""

    def test_happy_path_shape(self) -> None:
        row = _row()
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        assert obj.kind == ObjectKind.DATASOURCE
        assert obj.kind == "Datasource"
        assert obj.key == "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"
        assert obj.name == "CLICKSTREAM_EVENTS"
        assert obj.version is None
        assert obj.from_specification is True

        spec_payload = obj.spec_payload
        assert spec_payload["kind"] == "Datasource"
        assert spec_payload["name"] == "CLICKSTREAM_EVENTS"
        assert spec_payload["database"] == "JKEW_DB"
        assert spec_payload["schema"] == "JKEW_SCHEMA"
        assert spec_payload["source_type"] == "Stream"
        assert spec_payload["columns"] == row["schema"]
        assert spec_payload["description"] == "click events"

        assert obj.content_hash == structural_fingerprint_hash(spec_payload)

        assert obj.details["source_type"] == "Stream"
        assert obj.details["column_count"] == len(row["schema"])

    def test_lowercase_name_is_uppercased_in_name_and_key(self) -> None:
        row = _row(name="clickstream_events")
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        assert obj.name == "CLICKSTREAM_EVENTS"
        assert obj.key == "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"
        assert obj.spec_payload["name"] == "CLICKSTREAM_EVENTS"

    def test_lowercase_db_and_schema_are_uppercased(self) -> None:
        row = _row(name="My_Source")
        obj = _build_stream_source_object(
            row,
            default_db="jkew_db",
            default_schema="jkew_schema",
        )
        assert obj.key == "Datasource:JKEW_DB.JKEW_SCHEMA:MY_SOURCE"
        assert obj.spec_payload["database"] == "JKEW_DB"
        assert obj.spec_payload["schema"] == "JKEW_SCHEMA"
        assert obj.spec_payload["name"] == "MY_SOURCE"

    def test_empty_desc_yields_empty_description(self) -> None:
        row = _row(desc="")
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        assert obj.spec_payload["description"] == ""

    def test_missing_desc_key_yields_empty_description(self) -> None:
        row = {
            "name": "CLICKSTREAM_EVENTS",
            "schema": [{"name": "USER_ID", "type": "StringType"}],
            "owner": "ROLE_X",
        }
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        assert obj.spec_payload["description"] == ""

    def test_empty_schema_does_not_crash(self) -> None:
        row = _row(schema=[])
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        assert obj.spec_payload["columns"] == []
        assert obj.details["column_count"] == 0
        assert obj.content_hash == structural_fingerprint_hash(obj.spec_payload)

    def test_columns_preserve_case_as_authored(self) -> None:
        row = _row(
            schema=[
                {"name": "user_id", "type": "StringType"},
                {"name": "EventTs", "type": "TimestampType"},
            ]
        )
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        assert obj.spec_payload["columns"] == row["schema"]

    def test_no_type_key_in_spec_payload(self) -> None:
        # The runtime metadata does not record the producer protocol
        # ("type": "REST" on the local side); §5a explicitly excludes
        # it so the diff helper's stripped-payload comparison stays
        # symmetric.
        row = _row()
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        assert "type" not in obj.spec_payload

    def test_key_collides_with_datasource_objects_from_specs_shape(self) -> None:
        # Round-trip with a FV-derived datasource entry: both paths
        # must agree on ``key`` so the W2A merge step can dedup
        # cleanly. The FV-derived shape comes from
        # ``state._datasource_objects_from_specs`` (verified by reading
        # the source); we replicate its key formula here to lock the
        # collision contract from the *consumer* side.
        row = _row(name="clickstream_events")
        obj = _build_stream_source_object(
            row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )
        expected_key = f"{ObjectKind.DATASOURCE}:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"
        assert obj.key == expected_key


# ---------------------------------------------------------------------------
# Wave 2A — ``fetch_applied_state(stream_source_rows=...)`` merge tests.
#
# Contract: ``plans/stream_source_contract.md`` §5b.
#
# Runtime-authoritative merge: when both a runtime ``stream_source_rows``
# entry and a FV-derived ``_datasource_objects_from_specs`` entry surface
# the same ``Datasource:DB.SCHEMA:NAME`` key, the runtime row's
# ``AppliedObject`` wins.  When the runtime kwarg is absent or empty,
# today's FV-derived-only behaviour is preserved.
# ---------------------------------------------------------------------------


def _show_row(*, name: str = "CLICK_FV", version: str = "V1") -> dict[str, Any]:
    """Build a SHOW ONLINE FEATURE TABLES row keyed on ``DB.SCH``."""
    return {
        "name": f"{name}${version}$ONLINE",
        "database_name": "JKEW_DB",
        "schema_name": "JKEW_SCHEMA",
        "created_on": "2024-01-01 00:00:00",
    }


def _fv_spec_with_source(
    *,
    fv_name: str = "click_fv",
    version: str = "v1",
    source_name: str = "CLICKSTREAM_EVENTS",
) -> dict[str, Any]:
    """Build a StreamingFeatureView SPECIFICATION-JSON payload.

    The payload's ``spec.sources[0].name`` equals *source_name* so the
    FV-derived ``_datasource_objects_from_specs`` pass inside
    ``fetch_applied_state`` emits a ``Datasource`` entry keyed by
    *source_name* under the FV's database/schema — letting the merge
    tests force a collision against a runtime ``stream_source_rows``
    entry of the same name.

    Args:
        fv_name: Feature-view name written into the spec metadata.
        version: Feature-view version written into the spec metadata.
        source_name: Source name written into ``spec.sources[0].name``.

    Returns:
        A SPECIFICATION-JSON dict suitable for ``specification_map``.
    """
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "name": fv_name,
            "version": version,
            "database": "JKEW_DB",
            "schema": "JKEW_SCHEMA",
        },
        "spec": {
            "ordered_entity_column_names": ["user_id"],
            "sources": [
                {
                    "name": source_name,
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "EVENT_TS", "type": "TimestampType"},
                    ],
                },
            ],
            "features": [{"output_column": {"name": "event", "type": "StringType"}}],
        },
    }


class TestFetchAppliedStateStreamSourceMerge:
    """§5b of plans/stream_source_contract.md."""

    def test_none_falls_back_to_fv_derived_only(self) -> None:
        # Backward compat: when ``stream_source_rows`` is omitted (None),
        # ``fetch_applied_state`` must produce the same datasource entries
        # the FV-derived path emitted before W2A landed.
        show_row = _show_row()
        spec = _fv_spec_with_source(source_name="CLICKSTREAM_EVENTS")

        state = fetch_applied_state(
            [show_row],
            None,
            specification_map={show_row["name"]: spec},
            default_database="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        ds_objs = [o for o in state.objects.values() if o.kind == "Datasource"]
        assert len(ds_objs) == 1
        assert ds_objs[0].name == "CLICKSTREAM_EVENTS"
        assert ds_objs[0].key == "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"

    def test_empty_stream_source_rows_preserves_fv_derived_entries(self) -> None:
        # Explicit empty list is equivalent to None for the runtime side:
        # zero runtime entries contributed, FV-derived entries still emerge.
        show_row = _show_row()
        spec = _fv_spec_with_source(source_name="CLICKSTREAM_EVENTS")

        state = fetch_applied_state(
            [show_row],
            None,
            specification_map={show_row["name"]: spec},
            stream_source_rows=[],
            default_database="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        ds_objs = [o for o in state.objects.values() if o.kind == "Datasource"]
        assert len(ds_objs) == 1
        assert ds_objs[0].name == "CLICKSTREAM_EVENTS"

    def test_runtime_row_emerges_even_without_fv_reference(self) -> None:
        # A registered stream source that no FV currently references must
        # still appear in ``applied_state.objects`` so the planner can
        # diff it against the local YAML (CREATE_SOURCE / UPDATE_SOURCE /
        # RECREATE_SOURCE / DROP_SOURCE).
        runtime_row = _row(name="CLICKSTREAM_EVENTS")

        state = fetch_applied_state(
            [],
            None,
            stream_source_rows=[runtime_row],
            default_database="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        key = "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"
        assert key in state.objects
        obj = state.objects[key]
        assert obj.kind == "Datasource"
        assert obj.name == "CLICKSTREAM_EVENTS"
        assert obj.spec_payload["source_type"] == "Stream"

    def test_runtime_row_wins_on_collision_with_fv_derived(self) -> None:
        # The merge-collision contract: both paths surface the same key,
        # runtime row is authoritative.  The fingerprint differs because
        # ``_build_stream_source_object`` stamps ``description`` from the
        # runtime row while ``_datasource_objects_from_specs`` does not.
        show_row = _show_row()
        spec = _fv_spec_with_source(source_name="CLICKSTREAM_EVENTS")
        runtime_row = _row(name="CLICKSTREAM_EVENTS", desc="runtime-authoritative")

        runtime_expected = _build_stream_source_object(
            runtime_row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        state = fetch_applied_state(
            [show_row],
            None,
            specification_map={show_row["name"]: spec},
            stream_source_rows=[runtime_row],
            default_database="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        key = "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"
        assert key in state.objects
        merged = state.objects[key]
        # Runtime AppliedObject wins on collision: the description /
        # content_hash come from the runtime row, not the FV-derived
        # payload (which has no description field at all).
        assert merged.content_hash == runtime_expected.content_hash
        assert merged.spec_payload.get("description") == "runtime-authoritative"
        # Exactly one Datasource entry survives the merge — the FV-derived
        # duplicate must NOT shadow the runtime entry under a different key.
        ds_objs = [o for o in state.objects.values() if o.kind == "Datasource"]
        assert len(ds_objs) == 1

    def test_different_sources_coexist(self) -> None:
        # Runtime row for source A + FV-derived row for source B (no
        # collision) → both appear in ``applied_state.objects``.
        show_row = _show_row(name="CLICK_FV", version="V1")
        spec = _fv_spec_with_source(source_name="USER_EVENTS")
        runtime_row = _row(name="CLICKSTREAM_EVENTS")

        state = fetch_applied_state(
            [show_row],
            None,
            specification_map={show_row["name"]: spec},
            stream_source_rows=[runtime_row],
            default_database="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        ds_keys = {k for k, o in state.objects.items() if o.kind == "Datasource"}
        assert "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS" in ds_keys
        assert "Datasource:JKEW_DB.JKEW_SCHEMA:USER_EVENTS" in ds_keys
        assert len(ds_keys) == 2

    def test_merge_order_runtime_first_fv_derived_skipped_on_collision(self) -> None:
        # Pin the merge ORDER, not just the outcome: ``fetch_applied_state``
        # must insert the runtime entry FIRST, then walk the FV-derived
        # results and skip when the key is already taken.  Spying on
        # ``_build_stream_source_object`` and ``_datasource_objects_from_specs``
        # locks the call sequence so a future refactor that flips the
        # order (FV-derived first, runtime overwrite second) is caught.
        show_row = _show_row()
        spec = _fv_spec_with_source(source_name="CLICKSTREAM_EVENTS")
        runtime_row = _row(name="CLICKSTREAM_EVENTS", desc="runtime-authoritative")

        from snowflake.ml.feature_store.decl import state as state_module

        runtime_obj = _build_stream_source_object(
            runtime_row,
            default_db="JKEW_DB",
            default_schema="JKEW_SCHEMA",
        )

        call_order: list[str] = []
        # Capture the real implementations BEFORE patching so the
        # FV-derived spy can delegate without recursing through the
        # patched name in the ``state`` module namespace.
        real_runtime_builder = state_module._build_stream_source_object
        real_fv_derived = state_module._datasource_objects_from_specs

        def _runtime_builder_spy(row: dict[str, Any], default_db: str, default_schema: str) -> Any:
            call_order.append("runtime")
            return real_runtime_builder(row, default_db, default_schema)

        def _fv_derived_spy(
            specs: list[dict[str, Any]], db: str, schema: str, *, known_fv_names: Any = None
        ) -> list[Any]:
            call_order.append("fv_derived")
            return real_fv_derived(specs, db, schema, known_fv_names=known_fv_names)

        with patch.object(state_module, "_build_stream_source_object", side_effect=_runtime_builder_spy,), patch.object(
            state_module,
            "_datasource_objects_from_specs",
            side_effect=_fv_derived_spy,
        ):
            state = fetch_applied_state(
                [show_row],
                None,
                specification_map={show_row["name"]: spec},
                stream_source_rows=[runtime_row],
                default_database="JKEW_DB",
                default_schema="JKEW_SCHEMA",
            )

        assert call_order == ["runtime", "fv_derived"], call_order
        # And the merged AppliedObject is the runtime one (sanity check
        # that the spy didn't break the contract from the previous test).
        key = "Datasource:JKEW_DB.JKEW_SCHEMA:CLICKSTREAM_EVENTS"
        assert state.objects[key].content_hash == runtime_obj.content_hash


if __name__ == "__main__":
    pytest_driver.main()
