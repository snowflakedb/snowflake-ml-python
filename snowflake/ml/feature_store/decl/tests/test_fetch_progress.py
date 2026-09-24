"""RED tests — optional ``on_progress`` callback on the fetch facades.

The CLI renders a stderr progress bar for ``snow feature plan`` / ``list``
by driving a per-row callback threaded through the imperative read
helpers.  The library itself never prints — it only invokes the
callback the caller supplies.  These tests pin the callback contract for
all four fetch functions:

* :func:`fetch_feature_view_rows`
* :func:`fetch_entity_rows`
* :func:`fetch_feature_group_rows`
* :func:`fetch_stream_source_rows`

Contract (see ``plans/plan_list_progress_bar_35464193.plan.md`` §Library
callback contract):

* The callback signature is ``on_progress(completed, total, label)``.
* Immediately after the listing ``collect()`` returns, the helper calls
  ``on_progress(0, n, "")`` once with ``n = len(listed)`` — the moment
  ``total`` first exists.
* After each row is translated, the helper calls
  ``on_progress(i, n, name)`` with a 1-based ``i`` and the row's name.
* An empty listing still emits the single ``(0, 0, "")`` call so the CLI
  can skip the phase.
* ``on_progress=None`` (the default) emits no calls and leaves the
  returned rows byte-identical to the no-callback contract.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from snowflake.ml.test_utils import pytest_driver


class _Recorder:
    """Records every ``on_progress(completed, total, label)`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def __call__(self, completed: int, total: int, label: str) -> None:
        self.calls.append((completed, total, label))


def _fv_row(name: str, version: str = "V1") -> dict[str, Any]:
    """Streaming-kind ``list_feature_views`` row (no ``get_feature_view``)."""
    return {
        "name": name,
        "version": version,
        "database_name": "DB",
        "schema_name": "SCH",
        "created_on": "2024-01-01 00:00:00",
        "owner": "ROLE_X",
        "desc": "",
        "entities": ["USER_ID"],
        "refresh_freq": "1 day",
        "refresh_mode": "INCREMENTAL",
        "scheduling_state": "ACTIVE",
        "warehouse": "WH",
        "cluster_by": None,
        "online_config": json.dumps({"enable": True}),
        "storage_config": '{"format": "snowflake"}',
        "stream_config": None,
        "kind": "STREAMING",
        "target_lag": "1 day",
    }


def _entity_row(name: str) -> dict[str, Any]:
    return {"NAME": name, "JOIN_KEYS": '["USER_ID"]', "DESC": "", "OWNER": "ROLE_X"}


def _fg_row(name: str, version: str = "V1") -> dict[str, Any]:
    return {
        "NAME": name,
        "VERSION": version,
        "DESC": "",
        "OWNER": "ROLE_X",
        "AUTO_PREFIX": True,
        "SOURCES": json.dumps([{"fv_name": "FV_A", "fv_version": "V1"}]),
        "OUTPUT_COLUMNS": None,
    }


def _stream_row(name: str) -> dict[str, Any]:
    return {
        "NAME": name,
        "SCHEMA": json.dumps([{"name": "USER_ID", "type": "StringType"}]),
        "DESC": "",
        "OWNER": "ROLE_X",
    }


def _fs_with(list_attr: str, rows: list[dict[str, Any]]) -> MagicMock:
    df = MagicMock(name="DataFrame")
    df.collect.return_value = rows
    fs = MagicMock(name="FeatureStore")
    getattr(fs, list_attr).return_value = df
    return fs


class TestFetchFeatureViewRowsProgress:
    def test_reports_total_then_per_row(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_feature_views", [_fv_row("FV_A"), _fv_row("FV_B")])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert len(rows) == 2
        assert rec.calls[0] == (0, 2, "")
        assert rec.calls[1] == (1, 2, "FV_A")
        assert rec.calls[2] == (2, 2, "FV_B")
        assert len(rec.calls) == 3

    def test_empty_listing_emits_zero_total(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_feature_views", [])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert rows == []
        assert rec.calls == [(0, 0, "")]

    def test_none_callback_emits_nothing(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        fs = _fs_with("list_feature_views", [_fv_row("FV_A")])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(MagicMock(name="session"), "DB", "SCH", "WH")

        assert len(rows) == 1


class TestFetchEntityRowsProgress:
    def test_reports_total_then_per_row(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_entity_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_entities", [_entity_row("USER"), _entity_row("DEVICE")])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_entity_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert len(rows) == 2
        assert rec.calls[0] == (0, 2, "")
        assert rec.calls[1] == (1, 2, "USER")
        assert rec.calls[2] == (2, 2, "DEVICE")
        assert len(rec.calls) == 3

    def test_empty_listing_emits_zero_total(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_entity_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_entities", [])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_entity_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert rows == []
        assert rec.calls == [(0, 0, "")]


class TestFetchFeatureGroupRowsProgress:
    def test_reports_total_then_per_row(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_feature_groups", [_fg_row("FG_A"), _fg_row("FG_B")])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_group_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert len(rows) == 2
        assert rec.calls[0] == (0, 2, "")
        assert rec.calls[1] == (1, 2, "FG_A")
        assert rec.calls[2] == (2, 2, "FG_B")
        assert len(rec.calls) == 3

    def test_empty_listing_emits_zero_total(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_feature_groups", [])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_group_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert rows == []
        assert rec.calls == [(0, 0, "")]


class TestFetchStreamSourceRowsProgress:
    def test_reports_total_then_per_row(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_stream_source_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_stream_sources", [_stream_row("SS_A"), _stream_row("SS_B")])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_stream_source_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert len(rows) == 2
        assert rec.calls[0] == (0, 2, "")
        assert rec.calls[1] == (1, 2, "SS_A")
        assert rec.calls[2] == (2, 2, "SS_B")
        assert len(rec.calls) == 3

    def test_empty_listing_emits_zero_total(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_stream_source_rows,
        )

        rec = _Recorder()
        fs = _fs_with("list_stream_sources", [])
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_stream_source_rows(MagicMock(name="session"), "DB", "SCH", "WH", on_progress=rec)

        assert rows == []
        assert rec.calls == [(0, 0, "")]


class TestDeclApiFacadesForwardProgress:
    """The thin ``decl_api`` facades must forward ``on_progress`` to the
    executor implementations (CLI only ever calls the facade)."""

    def test_api_fetch_feature_view_rows_forwards(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        rec = _Recorder()
        with patch("snowflake.ml.feature_store.decl.imperative_executor.fetch_feature_view_rows") as impl:
            impl.return_value = []
            decl_api.fetch_feature_view_rows(MagicMock(), "DB", "SCH", "WH", on_progress=rec)
        _, kwargs = impl.call_args
        assert kwargs.get("on_progress") is rec

    def test_api_fetch_entity_rows_forwards(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        rec = _Recorder()
        with patch("snowflake.ml.feature_store.decl.imperative_executor.fetch_entity_rows") as impl:
            impl.return_value = []
            decl_api.fetch_entity_rows(MagicMock(), "DB", "SCH", "WH", on_progress=rec)
        _, kwargs = impl.call_args
        assert kwargs.get("on_progress") is rec

    def test_api_fetch_feature_group_rows_forwards(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        rec = _Recorder()
        with patch("snowflake.ml.feature_store.decl.imperative_executor.fetch_feature_group_rows") as impl:
            impl.return_value = []
            decl_api.fetch_feature_group_rows(MagicMock(), "DB", "SCH", "WH", on_progress=rec)
        _, kwargs = impl.call_args
        assert kwargs.get("on_progress") is rec

    def test_api_fetch_stream_source_rows_forwards(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        rec = _Recorder()
        with patch("snowflake.ml.feature_store.decl.imperative_executor.fetch_stream_source_rows") as impl:
            impl.return_value = []
            decl_api.fetch_stream_source_rows(MagicMock(), "DB", "SCH", "WH", on_progress=rec)
        _, kwargs = impl.call_args
        assert kwargs.get("on_progress") is rec


if __name__ == "__main__":
    pytest_driver.main()
