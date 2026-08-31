"""Tests for ``invariants.compute_source_diff_kind``.

Pins the truth table from
``plans/stream_source_contract.md`` §2:

| Local vs. Applied              | structural | desc | result              |
|--------------------------------|------------|------|---------------------|
| identical YAML reload          | yes        | yes  | ``no_change``       |
| desc edited only               | yes        | no   | ``update_desc_only``|
| columns added                  | no         | yes  | ``recreate``        |
| ``source_type`` changed        | no         | n/a  | ``recreate``        |
| ``table`` / ``query`` changed  | no         | n/a  | ``recreate``        |
| ``type`` (REST vs other)       | no         | n/a  | ``recreate``        |

The applied side is the runtime row shape produced by
``_build_stream_source_object`` — it canonicalises ``kind`` to
``"Datasource"`` and never carries a ``type`` field.  The local side
mirrors the authoring YAML — ``kind: StreamingSource`` (or
``BatchSource``) and an authored ``type: REST``.  The diff helper
canonicalises both sides to ``Datasource`` and strips the producer-
protocol ``type`` from the local payload before structural hashing so a
clean round-trip emits ``no_change``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.invariants import compute_source_diff_kind
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_streaming_source(
    *,
    name: str = "USER_CLICKS",
    columns: list[dict[str, Any]] | None = None,
    description: str = "user clickstream events",
    type_: str = "REST",
) -> dict[str, Any]:
    """Build a local-side YAML-shape payload for a StreamingSource.

    Mirrors what ``model_to_dict(StreamingSource(...))`` produces:
    ``kind: StreamingSource``, ``type: REST``, ``columns`` list, and a
    top-level ``description``.

    Args:
        name: Source name (uppercased to mirror the YAML round-trip).
        columns: Optional override for the ``columns`` list; defaults to
            two string columns matching the applied helper.
        description: Top-level ``description`` text; an empty string
            simulates an absent description.
        type_: The producer-protocol ``type`` field (e.g. ``"REST"``).

    Returns:
        Plain dict in the local-YAML shape suitable as the first
        argument to ``compute_source_diff_kind``.
    """
    return {
        "kind": "StreamingSource",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "type": type_,
        "columns": columns
        if columns is not None
        else [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT", "type": "StringType"},
        ],
        "description": description,
    }


def _applied_datasource(
    *,
    name: str = "USER_CLICKS",
    columns: list[dict[str, Any]] | None = None,
    desc: str = "user clickstream events",
    source_type: str = "Stream",
) -> dict[str, Any]:
    """Build an applied-state shape payload for a Datasource.

    Mirrors what ``_build_stream_source_object`` will produce: ``kind:
    Datasource``, ``source_type: Stream``, ``columns`` list, and a
    top-level ``description`` that originated from the runtime ``DESC``
    column.  No ``type`` field on the applied side.

    Args:
        name: Source name (uppercased to mirror the runtime row shape).
        columns: Optional override for the ``columns`` list; defaults
            to two string columns matching the local helper.
        desc: Description text written into the ``description`` key
            (mirrors ``_build_stream_source_object`` storing the DESC
            row column there).
        source_type: ``"Stream"`` for streaming-derived rows;
            ``"Batch"`` is also valid.

    Returns:
        Plain dict in the applied-state shape suitable as the second
        argument to ``compute_source_diff_kind``.
    """
    return {
        "kind": "Datasource",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "source_type": source_type,
        "columns": columns
        if columns is not None
        else [
            {"name": "USER_ID", "type": "StringType"},
            {"name": "EVENT", "type": "StringType"},
        ],
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Truth table — primary cases
# ---------------------------------------------------------------------------


class TestComputeSourceDiffKindTruthTable:
    """Each case in the §2 truth table gets one targeted test."""

    def test_identical_yaml_reload_returns_no_change(self) -> None:
        """Round-trip — local YAML and applied row carry the same content."""
        local = _local_streaming_source()
        applied = _applied_datasource()
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_desc_only_change_returns_update_desc_only(self) -> None:
        """Edit only the description — structural hashes match."""
        local = _local_streaming_source(description="edited clickstream description")
        applied = _applied_datasource(desc="user clickstream events")
        assert compute_source_diff_kind(local, applied) == "update_desc_only"

    def test_column_added_returns_recreate(self) -> None:
        """Add a new column on the local side — structural hashes diverge."""
        local = _local_streaming_source(
            columns=[
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT", "type": "StringType"},
                {"name": "PAGE_URL", "type": "StringType"},
            ],
        )
        applied = _applied_datasource()
        assert compute_source_diff_kind(local, applied) == "recreate"

    def test_source_type_changed_returns_recreate(self) -> None:
        """``source_type`` differs (Stream vs Batch) — recreate."""
        local = _local_streaming_source()
        local["source_type"] = "Stream"
        applied = _applied_datasource(source_type="Batch")
        assert compute_source_diff_kind(local, applied) == "recreate"

    def test_table_changed_returns_recreate(self) -> None:
        """A table-backed source whose ``table`` changed — recreate."""
        local = _local_streaming_source()
        local["table"] = "DB.SCH.NEW_TABLE"
        applied = _applied_datasource()
        applied["table"] = "DB.SCH.OLD_TABLE"
        assert compute_source_diff_kind(local, applied) == "recreate"

    def test_query_changed_returns_recreate(self) -> None:
        """A query-backed source whose ``query`` changed — recreate."""
        local = _local_streaming_source()
        local["query"] = "SELECT * FROM CLICKSTREAM_V2"
        applied = _applied_datasource()
        applied["query"] = "SELECT * FROM CLICKSTREAM_V1"
        assert compute_source_diff_kind(local, applied) == "recreate"

    def test_type_changed_returns_recreate(self) -> None:
        """``type`` (REST vs other) differs across sides — recreate.

        The applied side does not normally carry ``type`` at runtime, but a
        synthetic test asserts the helper does NOT silently match when the
        applied payload happens to carry a different value (defence in
        depth — the authored YAML's protocol must round-trip cleanly).
        """
        local = _local_streaming_source(type_="REST")
        applied = _applied_datasource()
        applied["type"] = "GRPC"
        assert compute_source_diff_kind(local, applied) == "recreate"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestComputeSourceDiffKindEdgeCases:
    def test_kind_canonicalisation_streaming_vs_datasource(self) -> None:
        """``kind: StreamingSource`` (local) and ``kind: Datasource`` (applied)
        must NOT trigger ``recreate`` when the rest of the payload matches.

        ``spec_key`` already canonicalises both forms to ``Datasource``;
        the diff helper must apply the same canonicalisation before
        structural hashing so a clean round-trip emits ``no_change``.
        """
        local = _local_streaming_source()
        assert local["kind"] == "StreamingSource"
        applied = _applied_datasource()
        assert applied["kind"] == "Datasource"
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_kind_canonicalisation_batch_source_vs_datasource(self) -> None:
        """Same canonicalisation rule applies to ``BatchSource``."""
        local = _local_streaming_source()
        local["kind"] = "BatchSource"
        applied = _applied_datasource(source_type="Batch")
        # Realign the source_type on the local side to match the applied
        # batch shape so the only meaningful difference is the kind alias.
        local["source_type"] = "Batch"
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_missing_description_on_both_sides_returns_no_change(self) -> None:
        """Both sides omit the description fields → desc compares as ``""``."""
        local = _local_streaming_source(description="")
        local.pop("description", None)
        applied = _applied_datasource(desc="")
        applied.pop("description", None)
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_missing_description_on_local_only_returns_update_desc_only(self) -> None:
        """Local missing description → ``""``; applied has text → desc differs."""
        local = _local_streaming_source()
        local.pop("description", None)
        applied = _applied_datasource(desc="user clickstream events")
        assert compute_source_diff_kind(local, applied) == "update_desc_only"

    def test_missing_description_on_applied_only_returns_update_desc_only(self) -> None:
        """Applied missing description → ``""``; local has text → desc differs."""
        local = _local_streaming_source(description="local-side description")
        applied = _applied_datasource()
        applied.pop("description", None)
        assert compute_source_diff_kind(local, applied) == "update_desc_only"

    def test_desc_field_is_consulted_when_description_absent(self) -> None:
        """Either ``description`` or ``desc`` may carry the source text.

        ``local`` uses the YAML-shape ``description`` field; ``applied``
        carries it under the runtime row's ``desc`` field.  Both must
        be considered — when both resolve to the same value the diff is
        ``no_change``.
        """
        local = _local_streaming_source(description="abc")
        applied = _applied_datasource()
        applied.pop("description", None)
        applied["desc"] = "abc"
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_desc_compare_is_case_sensitive(self) -> None:
        """``"abc"`` and ``"ABC"`` must not collapse to the same desc."""
        local = _local_streaming_source(description="abc")
        applied = _applied_datasource(desc="ABC")
        assert compute_source_diff_kind(local, applied) == "update_desc_only"

    def test_desc_compare_is_whitespace_trimmed(self) -> None:
        """Leading / trailing whitespace differences must collapse to ``no_change``."""
        local = _local_streaming_source(description="  user clickstream events  ")
        applied = _applied_datasource(desc="user clickstream events")
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_recreate_short_circuits_desc_compare(self) -> None:
        """A structural diff returns ``recreate`` even when the desc also drifts.

        The contract says desc drift is "free" inside a recreate — the
        helper must not return ``update_desc_only`` for a payload whose
        structure already requires a recreate.
        """
        local = _local_streaming_source(
            columns=[
                {"name": "USER_ID", "type": "StringType"},
                {"name": "EVENT", "type": "StringType"},
                {"name": "PAGE_URL", "type": "StringType"},
            ],
            description="freshly-authored description",
        )
        applied = _applied_datasource(desc="originally-deployed description")
        assert compute_source_diff_kind(local, applied) == "recreate"

    def test_type_omission_on_applied_does_not_trigger_recreate(self) -> None:
        """Local YAML carries ``type: REST``; applied row omits ``type`` entirely.

        This is the canonical clean-round-trip shape: the runtime
        ``list_stream_sources()`` row has no producer-protocol field, so
        the helper must strip ``type`` from the local payload before
        structural hashing.
        """
        local = _local_streaming_source(type_="REST")
        applied = _applied_datasource()
        assert "type" not in applied
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_returns_one_of_three_literal_strings(self) -> None:
        """Sanity — every output is one of the three contract literals."""
        valid = {"no_change", "update_desc_only", "recreate"}
        local = _local_streaming_source()
        applied = _applied_datasource()
        assert compute_source_diff_kind(local, applied) in valid

        edited_desc = copy.deepcopy(local)
        edited_desc["description"] = "different"
        assert compute_source_diff_kind(edited_desc, applied) in valid

        added_col = copy.deepcopy(local)
        added_col["columns"] = added_col["columns"] + [{"name": "EXTRA", "type": "StringType"}]
        assert compute_source_diff_kind(added_col, applied) in valid


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------


class TestComputeSourceDiffKindSignature:
    """The helper accepts plain dicts; nothing snowpark / connector-shaped."""

    def test_accepts_plain_dicts(self) -> None:
        """Smoke — the function takes two ``dict`` args and returns a str."""
        result = compute_source_diff_kind(
            _local_streaming_source(),
            _applied_datasource(),
        )
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        ("local", "applied", "expected"),
        [
            pytest.param(
                _local_streaming_source(),
                _applied_datasource(),
                "no_change",
                id="round-trip",
            ),
            pytest.param(
                _local_streaming_source(description="new"),
                _applied_datasource(desc="old"),
                "update_desc_only",
                id="desc-only",
            ),
            pytest.param(
                _local_streaming_source(
                    columns=[{"name": "X", "type": "StringType"}],
                ),
                _applied_datasource(),
                "recreate",
                id="columns-diverge",
            ),
        ],
    )
    def test_returns_expected_literal(self, local: dict[str, Any], applied: dict[str, Any], expected: str) -> None:
        assert compute_source_diff_kind(local, applied) == expected


# ---------------------------------------------------------------------------
# Query-backed BatchSource: SQL canonicalization (PR A — A2)
# ---------------------------------------------------------------------------


def _batch_source(*, query: str, description: str = "orders rollup") -> dict[str, Any]:
    """Local YAML-shape payload for a query-backed BatchSource."""
    return {
        "kind": "BatchSource",
        "name": "ORDERS",
        "query": query,
        "description": description,
    }


def _applied_batch_datasource(*, query: str, desc: str = "orders rollup") -> dict[str, Any]:
    """Applied Datasource row for a query-backed BatchSource (DT-recovered)."""
    return {
        "kind": "Datasource",
        "name": "ORDERS",
        "source_type": "Batch",
        "query": query,
        "description": desc,
    }


class TestComputeSourceDiffKindQueryCanonicalization:
    """A comment / keyword-case / whitespace-only change to a query-backed
    ``BatchSource`` must NOT surface as ``recreate`` — the SQL is
    canonicalized before structural hashing so cosmetic edits are
    ``no_change`` while a genuine query change is still ``recreate``."""

    def test_comment_only_change_is_no_change(self) -> None:
        local = _batch_source(query="SELECT id, amount FROM orders -- v2 edited")
        applied = _applied_batch_datasource(query="SELECT id, amount FROM orders")
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_keyword_case_only_change_is_no_change(self) -> None:
        local = _batch_source(query="select id, amount from orders")
        applied = _applied_batch_datasource(query="SELECT id, amount FROM orders")
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_whitespace_only_change_is_no_change(self) -> None:
        local = _batch_source(query="SELECT id,\n   amount\nFROM orders")
        applied = _applied_batch_datasource(query="SELECT id, amount FROM orders")
        assert compute_source_diff_kind(local, applied) == "no_change"

    def test_semantic_query_change_is_recreate(self) -> None:
        local = _batch_source(query="SELECT id, amount, tax FROM orders")
        applied = _applied_batch_datasource(query="SELECT id, amount FROM orders")
        assert compute_source_diff_kind(local, applied) == "recreate"


if __name__ == "__main__":
    pytest_driver.main()
