"""Phase 3 RED tests — ``state.fetch_applied_state(feature_group_rows=...)``.

Pin the contract that an FG row from
:func:`imperative_executor.fetch_feature_group_rows` is reified as one
``AppliedObject(kind="FeatureGroup")`` per row, with:

* ``key``  = ``"FeatureGroup:<DB>.<SCHEMA>:<NAME_UPPER>"``.
* ``name``, ``version`` populated from the row.
* ``content_hash`` computed via :func:`invariants.fg_content_hash` over
  the reconstructed declarative-shape spec payload.  This makes the
  planner's hash basis match by construction.
* ``from_specification`` is False (FG state lives in
  ``FeatureGroupMetadata``, not in an OFT specification).
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store.decl.invariants import fg_content_hash
from snowflake.ml.feature_store.decl.state import fetch_applied_state


def _fg_row(
    *,
    name: str = "USER_FRAUD_FG",
    version: str = "V1",
    desc: str = "",
    auto_prefix: bool = True,
    sources: list[dict[str, Any]] | None = None,
    output_columns: list[str] | None = None,
    database_name: str = "DB",
    schema_name: str = "SCH",
) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "desc": desc,
        "owner": "ROLE",
        "auto_prefix": auto_prefix,
        "sources": sources if sources is not None else [{"fv_name": "FV_A", "fv_version": "V1"}],
        "output_columns": output_columns,
        "database_name": database_name,
        "schema_name": schema_name,
    }


class TestStateFetchAppliedFeatureGroup:
    def test_one_applied_object_per_row(self) -> None:
        rows = [_fg_row(name="FG_A"), _fg_row(name="FG_B")]
        state = fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database="DB",
            default_schema="SCH",
        )
        fg_objs = [o for o in state.objects.values() if o.kind == "FeatureGroup"]
        assert {o.name for o in fg_objs} == {"FG_A", "FG_B"}

    def test_key_is_fully_qualified(self) -> None:
        rows = [_fg_row(name="MY_FG", database_name="DB", schema_name="SCH")]
        state = fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database="DB",
            default_schema="SCH",
        )
        assert "FeatureGroup:DB.SCH:MY_FG" in state.objects

    def test_content_hash_matches_planner_basis(self) -> None:
        sources: list[dict[str, Any]] = [
            {"fv_name": "FV_A", "fv_version": "V1"},
            {
                "fv_name": "FV_B",
                "fv_version": "V1",
                "slice_columns": ["X"],
                "alias": "b",
            },
        ]
        rows = [
            _fg_row(
                name="FG_X",
                version="V1",
                desc="hello",
                auto_prefix=False,
                sources=sources,
            )
        ]
        state = fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database="DB",
            default_schema="SCH",
        )
        ao = state.objects["FeatureGroup:DB.SCH:FG_X"]
        # Reconstruct the declarative-shape spec the planner will see at
        # plan time and verify the applied hash matches by construction.
        decl_payload = {
            "kind": "FeatureGroup",
            "name": "FG_X",
            "version": "V1",
            "database": "DB",
            "schema": "SCH",
            "desc": "hello",
            "auto_prefix": False,
            "feature_views": [
                {"name": "FV_A", "version": "V1"},
                {
                    "name": "FV_B",
                    "version": "V1",
                    "slice_columns": ["X"],
                    "alias": "b",
                },
            ],
        }
        assert ao.content_hash == fg_content_hash(decl_payload)

    def test_from_specification_is_false(self) -> None:
        rows = [_fg_row(name="FG_A")]
        state = fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database="DB",
            default_schema="SCH",
        )
        ao = next(o for o in state.objects.values() if o.kind == "FeatureGroup")
        assert ao.from_specification is False

    def test_alias_empty_string_preserved_in_hash_basis(self) -> None:
        rows = [
            _fg_row(
                name="FG_A",
                sources=[
                    {"fv_name": "FV_A", "fv_version": "V1", "alias": ""},
                ],
            )
        ]
        state = fetch_applied_state(
            raw_show_results=[],
            feature_group_rows=rows,
            default_database="DB",
            default_schema="SCH",
        )
        ao = next(o for o in state.objects.values() if o.kind == "FeatureGroup")
        # The reconstructed spec must preserve alias="" so the hash basis
        # round-trips with a local YAML carrying the same value.
        decl_payload_no_alias = {
            "kind": "FeatureGroup",
            "name": "FG_A",
            "version": "V1",
            "database": "DB",
            "schema": "SCH",
            "desc": "",
            "auto_prefix": True,
            "feature_views": [{"name": "FV_A", "version": "V1"}],
        }
        decl_payload_empty_alias = dict(decl_payload_no_alias)
        decl_payload_empty_alias["feature_views"] = [{"name": "FV_A", "version": "V1", "alias": ""}]
        assert ao.content_hash == fg_content_hash(decl_payload_empty_alias)
        assert ao.content_hash != fg_content_hash(decl_payload_no_alias)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
