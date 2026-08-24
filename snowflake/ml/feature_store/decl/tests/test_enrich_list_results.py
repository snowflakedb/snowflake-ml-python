"""Tests for ``decl.api.enrich_list_results`` multi-kind list enrichment.

The function is responsible for turning raw Snowflake query results
(``SHOW ONLINE FEATURE TABLES``, ``SHOW TAGS``, ``DESCRIBE ... TYPE =
SPECIFICATION``) into a single, ordered list of display rows that the
CLI can render in a table with a ``type`` column for FeatureView,
Entity, and Datasource objects.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.api import enrich_list_results
from snowflake.ml.feature_store.decl.types import ObjectKind

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SHOW_ROW = {
    "name": "USER_CLICKS$V1$ONLINE",
    "database_name": "MYDB",
    "schema_name": "PUBLIC",
    "created_on": "2024-01-01 00:00:00",
    "scheduling_state": "ACTIVE",
}

_DESCRIBE_ROWS = [
    {"name": "USER_ID", "type": "VARCHAR", "primary key": "Y"},
    {"name": "EVENT", "type": "VARCHAR", "primary key": "N"},
]

_ENTITY_ROW_USER = {
    "name": "SNOWML_FEATURE_STORE_ENTITY_USER",
    "database_name": "MYDB",
    "schema_name": "PUBLIC",
    "allowed_values": '["USER_ID"]',
}

_ENTITY_ROW_SESSION = {
    "name": "SNOWML_FEATURE_STORE_ENTITY_SESSION",
    "database_name": "MYDB",
    "schema_name": "PUBLIC",
    "allowed_values": '["SESSION_ID", "USER_ID"]',
}

_FULL_SPEC = {
    "kind": "StreamingFeatureView",
    "metadata": {
        "database": "MYDB",
        "schema": "PUBLIC",
        "name": "user_clicks",
        "version": "v1",
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
            }
        ],
        "features": [],
    },
}


# ---------------------------------------------------------------------------
# FeatureView rows (legacy compatibility surface)
# ---------------------------------------------------------------------------


class TestEnrichListResultsFeatureViews:
    """FeatureView rows must keep all original SHOW columns *and* gain
    ``type``, ``feature_view``, ``version``, and ``entities`` fields."""

    def test_single_show_row_plus_describe_map(self) -> None:
        result = enrich_list_results([_SHOW_ROW], {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS})

        assert len(result) == 1
        row = result[0]
        assert row["type"] == ObjectKind.FEATURE_VIEW
        assert row["type"] == "FeatureView"
        # Names are preserved in Snowflake's canonical form (upper-case
        # for unquoted identifiers), exactly as ``_parse_oft_name``
        # returned them from ``SHOW ONLINE FEATURE TABLES``.  The
        # function does not client-side case-fold identifiers.
        assert row["feature_view"] == "USER_CLICKS"
        assert row["version"] == "V1"
        assert row["entities"] == "USER_ID"
        # Original SHOW columns survive.
        assert row["database_name"] == "MYDB"
        assert row["schema_name"] == "PUBLIC"
        assert row["created_on"] == "2024-01-01 00:00:00"
        assert row["scheduling_state"] == "ACTIVE"
        # ``name`` is the user-facing name (consistent with Entity/Datasource);
        # the raw OFT name is preserved as ``oft_name``.
        assert row["name"] == "USER_CLICKS"
        assert row["oft_name"] == "USER_CLICKS$V1$ONLINE"

    def test_positional_call_is_backward_compatible(self) -> None:
        # Old call site: positional (show_rows, describe_map).
        result = enrich_list_results([_SHOW_ROW], {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS})

        assert len(result) == 1
        assert result[0]["type"] == "FeatureView"
        assert result[0]["feature_view"] == "USER_CLICKS"
        assert result[0]["version"] == "V1"

    def test_empty_show_rows_returns_empty(self) -> None:
        result = enrich_list_results([], {})
        assert result == []

    def test_describe_map_can_be_none(self) -> None:
        # Even with no DESCRIBE rows we still produce a FV row.
        result = enrich_list_results([_SHOW_ROW])
        assert len(result) == 1
        row = result[0]
        assert row["type"] == "FeatureView"
        assert row["entities"] == ""

    def test_specification_map_provides_entities(self) -> None:
        """When ``specification_map`` carries the spec, ``entities`` comes
        from ``ordered_entity_column_names``, not from PK columns.

        ``_FULL_SPEC`` carries ``kind: StreamingFeatureView``, so the
        emitted row's ``type`` reflects that subkind rather than the
        generic ``FeatureView`` — see
        ``TestEnrichListResultsFeatureViewSubkind`` for the contract.
        """
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        assert len(result) >= 1
        fv_row = next(r for r in result if r["type"] == "StreamingFeatureView")
        assert fv_row["entities"] == "user_id"

    def test_describe_map_fallback_when_spec_missing(self) -> None:
        """If ``specification_map`` is empty for this OFT, fall back to
        primary-key columns from ``describe_map``."""
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
            specification_map={},
        )
        fv_row = next(r for r in result if r["type"] == "FeatureView")
        assert fv_row["entities"] == "USER_ID"


class TestEnrichListResultsFeatureViewSubkind:
    """When the FV's spec carries a ``kind`` field, the enriched row's
    ``type`` should surface the specific subkind (``StreamingFeatureView``
    / ``RealtimeFeatureView`` / ``BatchFeatureView``) instead of the
    generic ``FeatureView``.  The CLI displays this directly in the
    ``type`` column so operators can tell stream / realtime / batch
    FVs apart at a glance.

    Resolution order:
      1. ``specification_map[oft_name].kind`` (from
         ``DESCRIBE … TYPE = SPECIFICATION``)
      2. ``json.loads(show_row['specification']).kind`` (legacy
         embedded spec column on the SHOW row)
      3. Fallback: ``"FeatureView"``
    """

    def _spec_with_kind(self, kind: str) -> dict[str, Any]:
        spec: dict[str, Any] = json.loads(json.dumps(_FULL_SPEC))
        spec["kind"] = kind
        return spec

    def test_specification_map_streaming_kind_surfaces_as_type(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": self._spec_with_kind("StreamingFeatureView"),
            },
        )
        fv_row = next(r for r in result if r.get("oft_name") == "USER_CLICKS$V1$ONLINE")
        assert fv_row["type"] == "StreamingFeatureView"

    def test_specification_map_realtime_kind_surfaces_as_type(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": self._spec_with_kind("RealtimeFeatureView"),
            },
        )
        fv_row = next(r for r in result if r.get("oft_name") == "USER_CLICKS$V1$ONLINE")
        assert fv_row["type"] == "RealtimeFeatureView"

    def test_specification_map_batch_kind_surfaces_as_type(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": self._spec_with_kind("BatchFeatureView"),
            },
        )
        fv_row = next(r for r in result if r.get("oft_name") == "USER_CLICKS$V1$ONLINE")
        assert fv_row["type"] == "BatchFeatureView"

    def test_embedded_specification_column_on_show_row_resolves_subkind(self) -> None:
        """When ``DESCRIBE … TYPE = SPECIFICATION`` is unavailable but
        the SHOW row carries the legacy embedded ``specification`` JSON
        column, the FV subkind must still be recovered from there."""
        show_row = dict(_SHOW_ROW)
        show_row["specification"] = json.dumps(self._spec_with_kind("BatchFeatureView"))
        result = enrich_list_results(
            [show_row],
            None,
            # No specification_map entry — force the embedded column path.
        )
        fv_row = next(r for r in result if r.get("oft_name") == "USER_CLICKS$V1$ONLINE")
        assert fv_row["type"] == "BatchFeatureView"

    def test_fallback_to_generic_featureview_when_no_spec_anywhere(self) -> None:
        """No ``specification_map`` entry, no embedded ``specification``
        column → the row is still produced but with the generic
        ``FeatureView`` type (matches existing legacy behavior)."""
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
        )
        fv_row = next(r for r in result if r.get("oft_name") == "USER_CLICKS$V1$ONLINE")
        assert fv_row["type"] == "FeatureView"

    def test_unknown_kind_value_falls_back_to_generic_featureview(self) -> None:
        """Defensive: if a future / unknown kind string lands in the
        spec, we don't blindly trust it — we surface the generic
        ``FeatureView`` so downstream consumers (CLI, planner) only ever
        see one of the four canonical strings."""
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": self._spec_with_kind("MysteryFeatureView"),
            },
        )
        fv_row = next(r for r in result if r.get("oft_name") == "USER_CLICKS$V1$ONLINE")
        assert fv_row["type"] == "FeatureView"


# ---------------------------------------------------------------------------
# Entity rows
# ---------------------------------------------------------------------------


class TestEnrichListResultsEntities:
    """Entity rows are derived from ``SHOW TAGS`` rows whose ``name``
    starts with ``SNOWML_FEATURE_STORE_ENTITY_``."""

    def test_two_entity_rows_produce_two_entries(self) -> None:
        result = enrich_list_results(
            [],
            None,
            entity_rows=[_ENTITY_ROW_USER, _ENTITY_ROW_SESSION],
        )

        entity_rows = [r for r in result if r["type"] == "Entity"]
        assert len(entity_rows) == 2

        # Entity names are preserved in Snowflake's canonical form —
        # the tag suffix (``USER`` / ``SESSION``) comes back from
        # ``SHOW TAGS`` already upper-cased for unquoted identifiers.
        names = [r["name"] for r in entity_rows]
        assert names == ["USER", "SESSION"]

        # First row: single join key.
        assert entity_rows[0]["entities"] == "USER_ID"
        # Second row: two join keys, comma-separated.
        assert entity_rows[1]["entities"] == "SESSION_ID, USER_ID"

    def test_entity_row_carries_database_schema_and_empty_version(self) -> None:
        result = enrich_list_results(
            [],
            None,
            entity_rows=[_ENTITY_ROW_USER],
        )

        ent = result[0]
        assert ent["type"] == "Entity"
        assert ent["name"] == "USER"
        assert ent["version"] == ""
        assert ent["database_name"] == "MYDB"
        assert ent["schema_name"] == "PUBLIC"
        # ``feature_view`` column must exist for the CLI display projection.
        assert ent["feature_view"] == ""

    def test_entity_row_with_comment_populates_details(self) -> None:
        row = dict(_ENTITY_ROW_USER)
        row["comment"] = "primary user identifier"
        result = enrich_list_results([], None, entity_rows=[row])

        ent = result[0]
        assert ent["details"].get("comment") == "primary user identifier"

    def test_non_entity_tags_are_filtered_out(self) -> None:
        unrelated = {
            "name": "SOME_OTHER_TAG",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "allowed_values": "[]",
        }
        result = enrich_list_results(
            [],
            None,
            entity_rows=[unrelated, _ENTITY_ROW_USER],
        )
        entity_rows = [r for r in result if r["type"] == "Entity"]
        assert len(entity_rows) == 1
        assert entity_rows[0]["name"] == "USER"


# ---------------------------------------------------------------------------
# Datasource rows
# ---------------------------------------------------------------------------


class TestEnrichListResultsDatasources:
    """Datasource rows are unioned across all FV ``spec.sources[]``,
    deduplicated case-insensitively (Snowflake unquoted identifiers are
    case-insensitive) but rendered with the original case as stored in
    the recovered spec JSON."""

    def test_single_fv_spec_produces_one_datasource_row(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        ds_rows = [r for r in result if r["type"] == "Datasource"]
        assert len(ds_rows) == 1
        assert ds_rows[0]["name"] == "user_events"
        assert ds_rows[0]["version"] == ""
        assert ds_rows[0]["entities"] == ""
        assert ds_rows[0]["database_name"] == "MYDB"
        assert ds_rows[0]["schema_name"] == "PUBLIC"

    def test_two_fvs_share_source_dedupe_to_one_row(self) -> None:
        spec_a = json.loads(json.dumps(_FULL_SPEC))
        spec_b = json.loads(json.dumps(_FULL_SPEC))
        spec_b["metadata"]["name"] = "user_clicks_b"
        spec_b["metadata"]["version"] = "v2"

        show_rows = [
            _SHOW_ROW,
            {
                "name": "USER_CLICKS_B$V2$ONLINE",
                "database_name": "MYDB",
                "schema_name": "PUBLIC",
                "scheduling_state": "ACTIVE",
            },
        ]
        result = enrich_list_results(
            show_rows,
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": spec_a,
                "USER_CLICKS_B$V2$ONLINE": spec_b,
            },
        )
        ds_rows = [r for r in result if r["type"] == "Datasource"]
        assert len(ds_rows) == 1
        assert ds_rows[0]["name"] == "user_events"

    def test_two_fvs_with_case_different_source_names_dedupe_to_one_row(self) -> None:
        """Snowflake unquoted identifiers are case-insensitive, so two
        FVs that spell a shared source's name in different cases must
        collapse to a single Datasource row.  The displayed ``name``
        carries the case of the first FV to reference the source — we
        do not client-side normalise to upper / lower."""
        spec_a = json.loads(json.dumps(_FULL_SPEC))
        spec_a["spec"]["sources"][0]["name"] = "user_events"
        spec_b = json.loads(json.dumps(_FULL_SPEC))
        spec_b["spec"]["sources"][0]["name"] = "USER_EVENTS"
        spec_b["metadata"]["name"] = "user_clicks_b"
        spec_b["metadata"]["version"] = "v2"

        show_rows = [
            _SHOW_ROW,
            {
                "name": "USER_CLICKS_B$V2$ONLINE",
                "database_name": "MYDB",
                "schema_name": "PUBLIC",
                "scheduling_state": "ACTIVE",
            },
        ]
        result = enrich_list_results(
            show_rows,
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": spec_a,
                "USER_CLICKS_B$V2$ONLINE": spec_b,
            },
        )
        ds_rows = [r for r in result if r["type"] == "Datasource"]
        assert len(ds_rows) == 1
        # First-seen wins for the displayed name; case is preserved.
        assert ds_rows[0]["name"] == "user_events"

    def test_datasource_details_carry_source_type_and_column_count(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            None,
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        ds = next(r for r in result if r["type"] == "Datasource")
        assert ds["details"]["source_type"] == "Stream"
        assert ds["details"]["column_count"] == 2

    def test_datasource_rows_sorted_by_name(self) -> None:
        spec_a = json.loads(json.dumps(_FULL_SPEC))
        spec_a["spec"]["sources"][0]["name"] = "z_source"
        spec_b = json.loads(json.dumps(_FULL_SPEC))
        spec_b["spec"]["sources"][0]["name"] = "a_source"
        spec_b["metadata"]["name"] = "user_clicks_b"

        show_row_b = {
            "name": "USER_CLICKS_B$V1$ONLINE",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
        }
        result = enrich_list_results(
            [_SHOW_ROW, show_row_b],
            None,
            specification_map={
                "USER_CLICKS$V1$ONLINE": spec_a,
                "USER_CLICKS_B$V1$ONLINE": spec_b,
            },
        )
        ds_names = [r["name"] for r in result if r["type"] == "Datasource"]
        assert ds_names == ["a_source", "z_source"]


# ---------------------------------------------------------------------------
# Combined ordering: FV → Entity → Datasource
# ---------------------------------------------------------------------------


class TestEnrichListResultsCombined:
    """All three kinds together emit the right counts and ordering."""

    def test_ordering_fv_entity_datasource(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
            entity_rows=[_ENTITY_ROW_USER, _ENTITY_ROW_SESSION],
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )

        types_in_order = [r["type"] for r in result]
        # FV first, Entities next, Datasources last.  The FV type
        # surfaces ``_FULL_SPEC.kind`` (``StreamingFeatureView``)
        # rather than the generic ``FeatureView``.
        assert types_in_order[0] == "StreamingFeatureView"
        assert types_in_order[1] == "Entity"
        assert types_in_order[2] == "Entity"
        assert types_in_order[-1] == "Datasource"

    def test_combined_counts(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
            entity_rows=[_ENTITY_ROW_USER, _ENTITY_ROW_SESSION],
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        # ``_FULL_SPEC.kind`` is ``StreamingFeatureView``.
        assert len([r for r in result if r["type"] == "StreamingFeatureView"]) == 1
        assert len([r for r in result if r["type"] == "Entity"]) == 2
        assert len([r for r in result if r["type"] == "Datasource"]) == 1
        assert len(result) == 4

    def test_type_column_uses_canonical_strings(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
            entity_rows=[_ENTITY_ROW_USER],
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        types = {r["type"] for r in result}
        # ``_FULL_SPEC.kind`` flows through to the FV row's ``type``.
        assert types == {"StreamingFeatureView", "Entity", "Datasource"}

    def test_every_row_has_required_display_columns(self) -> None:
        """Every emitted row carries the columns used by the CLI's table
        projection (``feature_view``, ``version``, ``entities``,
        ``database_name``, ``schema_name``) plus the new ``type`` and
        ``details``."""
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
            entity_rows=[_ENTITY_ROW_USER],
            specification_map={"USER_CLICKS$V1$ONLINE": _FULL_SPEC},
        )
        required = {
            "type",
            "name",
            "version",
            "feature_view",
            "entities",
            "database_name",
            "schema_name",
            "details",
        }
        for row in result:
            missing = required - row.keys()
            assert not missing, f"Row missing keys {missing}: {row}"


# ---------------------------------------------------------------------------
# Backward compatibility with the legacy two-arg call
# ---------------------------------------------------------------------------


class TestEnrichListResultsBackwardCompat:
    """When the caller uses the legacy ``(show_rows, describe_map)``
    signature, no Entity or Datasource rows appear because the new
    keyword args were not supplied.  This guarantees the CLI manager's
    existing call sites keep producing the same output shape until they
    are migrated."""

    def test_legacy_call_produces_only_feature_view_rows(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
        )
        types = {r["type"] for r in result}
        assert types == {"FeatureView"}

    def test_legacy_call_keeps_legacy_columns(self) -> None:
        result = enrich_list_results(
            [_SHOW_ROW],
            {"USER_CLICKS$V1$ONLINE": _DESCRIBE_ROWS},
        )
        row = result[0]
        # Legacy columns the CLI's table projection still relies on.
        for key in (
            "feature_view",
            "version",
            "entities",
            "database_name",
            "schema_name",
            "created_on",
            "scheduling_state",
        ):
            assert key in row, f"Legacy column {key} missing"

    def test_legacy_call_signature_is_positional(self) -> None:
        # The CLI manager test asserts ``mock_decl.enrich_list_results``
        # is callable with positional args only.  This double-checks the
        # function signature still allows that.
        result = enrich_list_results([], {})
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
