"""Tests for state._datasource_objects_from_specs phantom-source suppression.

Pins the fix for Bug A: FG-backed BFVs emit a ``sources[0].name`` entry
that equals the FV name itself (an internal executor artifact, not an
operator-authored BatchSource).  Without the guard, every plan run
schedules ``DROP_SOURCE <fv_name>`` because the phantom entry appears
in the applied state but not in any local spec file.

See ``plans/bug_a_phantom_source_drop.md`` for the full root-cause
analysis and fix specification.
"""

from __future__ import annotations

import json

from snowflake.ml.feature_store.decl.state import (
    _datasource_objects_from_specs,
    _parse_cluster_by_list,
)

_DB = "JKEW_DB"
_SCH = "JKEW_SCHEMA"


def _fg_backed_bfv_spec_payload(fv_name: str = "USER_CLICKS_FG_DECL") -> dict:
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
) -> dict:
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
