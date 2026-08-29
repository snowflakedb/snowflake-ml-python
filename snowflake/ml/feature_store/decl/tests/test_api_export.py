"""Tests for the ``decl.api.export_specs`` facade.

Pins the contract that the public facade in :mod:`decl.api` forwards
its ``entity_rows`` kwarg verbatim to :func:`decl.exporter.export_specs`.
The CLI manager goes through this facade only — never directly into
``decl.exporter`` — so the facade is the single seam where the
``entity_rows`` plumbing must be respected.
"""

from __future__ import annotations

from unittest.mock import patch

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.test_utils import pytest_driver


class TestApiExportSpecsForwardsEntityRows:
    """Pin the api -> exporter forwarding contract for entity_rows."""

    def test_forwards_entity_rows_kwarg_when_provided(self) -> None:
        """A non-empty entity_rows must be forwarded verbatim to the exporter."""
        rows = [
            {
                "name": "SNOWML_FEATURE_STORE_ENTITY_USER_ID",
                "database_name": "DB",
                "schema_name": "SCH",
                "allowed_values": '["USER_ID"]',
                "comment": "user identifier",
            }
        ]
        with patch(
            "snowflake.ml.feature_store.decl.exporter.export_specs",
            return_value={"status": "exported", "directory": "", "files": []},
        ) as mock_exporter:
            decl_api.export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir="/tmp/never-written",
                database="DB",
                schema="SCH",
                specification_map={},
                entity_rows=rows,
            )

        mock_exporter.assert_called_once()
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("entity_rows") is rows, (
            "api.export_specs must forward entity_rows by-reference into the exporter; " f"got kwargs={kwargs!r}"
        )

    def test_default_entity_rows_is_none_when_caller_omits_it(self) -> None:
        """Omitting the kwarg must surface inside the exporter as entity_rows=None."""
        with patch(
            "snowflake.ml.feature_store.decl.exporter.export_specs",
            return_value={"status": "exported", "directory": "", "files": []},
        ) as mock_exporter:
            decl_api.export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir="/tmp/never-written",
                database="DB",
                schema="SCH",
            )

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("entity_rows", "<unset>") is None, (
            "Omitting entity_rows must surface as entity_rows=None inside "
            f"the exporter (it normalises None to [] internally); got {kwargs!r}"
        )

    def test_forwards_empty_entity_rows_list(self) -> None:
        """An explicit empty list is forwarded verbatim (caller signalled 'no entities')."""
        with patch(
            "snowflake.ml.feature_store.decl.exporter.export_specs",
            return_value={"status": "exported", "directory": "", "files": []},
        ) as mock_exporter:
            decl_api.export_specs(
                show_rows=[],
                describe_rows_by_oft={},
                output_dir="/tmp/never-written",
                database="DB",
                schema="SCH",
                entity_rows=[],
            )

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs.get("entity_rows") == [], (
            "api.export_specs must distinguish entity_rows=[] from entity_rows=None; " f"got kwargs={kwargs!r}"
        )


if __name__ == "__main__":
    pytest_driver.main()
