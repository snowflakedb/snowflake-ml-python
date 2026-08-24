"""Tests for the ``decl.api`` stream-source surface.

Pins two contracts the CLI manager relies on:

1. :func:`decl_api.fetch_stream_source_rows` is a thin facade that
   lazy-imports and forwards to
   :func:`imperative_executor.fetch_stream_source_rows` — mirrors the
   existing :func:`fetch_entity_rows` / :func:`fetch_feature_view_rows`
   / :func:`fetch_feature_group_rows` facades (see
   ``plans/stream_source_contract.md`` §6).
2. :func:`decl_api.fetch_applied_state` threads the new
   ``stream_source_rows`` kwarg straight through to
   :func:`state.fetch_applied_state` without inspecting it — the
   merge contract lives in ``state.py`` (Wave 2A) and is verified
   there.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.feature_store.decl import api as decl_api


class TestFetchStreamSourceRowsFacade:
    """The ``decl_api.fetch_stream_source_rows`` thin facade delegates
    to ``imperative_executor.fetch_stream_source_rows`` (the only place
    that may lazy-import ``snowflake.ml.feature_store``).
    """

    def test_decl_api_facade_delegates_to_executor(self) -> None:
        session = MagicMock(name="session")
        expected_rows = [
            {
                "name": "CLICKSTREAM_EVENTS",
                "schema": [{"name": "USER_ID", "type": "StringType"}],
                "desc": "Clickstream events",
                "owner": "ROLE_X",
            }
        ]
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.fetch_stream_source_rows",
            return_value=expected_rows,
        ) as mock_fetch:
            result = decl_api.fetch_stream_source_rows(session, "DB", "SCH", "WH")

        mock_fetch.assert_called_once_with(session, "DB", "SCH", "WH")
        assert result is expected_rows

    def test_decl_api_facade_accepts_default_warehouse(self) -> None:
        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.fetch_stream_source_rows",
            return_value=[],
        ) as mock_fetch:
            result = decl_api.fetch_stream_source_rows(session, "DB", "SCH")

        mock_fetch.assert_called_once_with(session, "DB", "SCH", "")
        assert result == []

    def test_decl_api_facade_is_callable(self) -> None:
        assert callable(decl_api.fetch_stream_source_rows)


class TestFetchAppliedStateStreamSourcePassthrough:
    """The ``decl_api.fetch_applied_state`` facade forwards the new
    ``stream_source_rows`` kwarg verbatim to
    :func:`state.fetch_applied_state`.  Existing callers that omit the
    kwarg must continue to work (kwarg defaults to ``None``).
    """

    def test_forwards_stream_source_rows_kwarg_when_provided(self) -> None:
        rows = [
            {
                "name": "CLICKSTREAM_EVENTS",
                "schema": [{"name": "USER_ID", "type": "StringType"}],
                "desc": "",
                "owner": "ROLE_X",
            }
        ]
        with patch(
            "snowflake.ml.feature_store.decl.state.fetch_applied_state",
            return_value="SENTINEL_APPLIED_STATE",
        ) as mock_state:
            result = decl_api.fetch_applied_state(
                raw_show_results=[],
                stream_source_rows=rows,
            )

        mock_state.assert_called_once()
        kwargs = mock_state.call_args.kwargs
        assert kwargs.get("stream_source_rows") is rows, (
            "api.fetch_applied_state must forward stream_source_rows by-reference into "
            f"state.fetch_applied_state; got kwargs={kwargs!r}"
        )
        assert result == "SENTINEL_APPLIED_STATE"  # type: ignore[comparison-overlap]

    def test_default_stream_source_rows_is_none_when_caller_omits_it(self) -> None:
        with patch(
            "snowflake.ml.feature_store.decl.state.fetch_applied_state",
            return_value="SENTINEL_APPLIED_STATE",
        ) as mock_state:
            decl_api.fetch_applied_state(raw_show_results=[])

        kwargs = mock_state.call_args.kwargs
        assert kwargs.get("stream_source_rows", "<unset>") is None, (
            "Omitting stream_source_rows must surface as stream_source_rows=None inside "
            f"state.fetch_applied_state; got kwargs={kwargs!r}"
        )

    def test_forwards_empty_stream_source_rows_list(self) -> None:
        with patch(
            "snowflake.ml.feature_store.decl.state.fetch_applied_state",
            return_value="SENTINEL_APPLIED_STATE",
        ) as mock_state:
            decl_api.fetch_applied_state(
                raw_show_results=[],
                stream_source_rows=[],
            )

        kwargs = mock_state.call_args.kwargs
        assert kwargs.get("stream_source_rows") == [], (
            "api.fetch_applied_state must distinguish stream_source_rows=[] from "
            f"stream_source_rows=None; got kwargs={kwargs!r}"
        )

    def test_existing_callers_unaffected(self) -> None:
        """Callers passing only the legacy kwargs must continue to work."""
        rows = [{"name": "ENT_A"}]
        with patch(
            "snowflake.ml.feature_store.decl.state.fetch_applied_state",
            return_value="SENTINEL_APPLIED_STATE",
        ) as mock_state:
            decl_api.fetch_applied_state(
                raw_show_results=[],
                entity_rows=rows,
                default_database="DB",
                default_schema="SCH",
            )

        kwargs = mock_state.call_args.kwargs
        assert kwargs.get("entity_rows") is rows
        assert kwargs.get("default_database") == "DB"
        assert kwargs.get("default_schema") == "SCH"
        assert kwargs.get("stream_source_rows", "<unset>") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
