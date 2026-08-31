"""Phase 5b RED tests — ``enrich_list_results`` surfaces FG rows.

Pin the contract that ``decl_api.enrich_list_results`` accepts the
``feature_group_rows`` kwarg (Phase 5 already did) AND emits one
``type=FeatureGroup`` row per input row, with the canonical CLI-table
columns (``name``, ``version``, ``database_name``, ``schema_name``,
``entities``, ``feature_view``) plus an FG-specific ``details`` block
that exposes ``source_count`` and the ordered ``sources`` summary.

The display ordering is: existing FV rows, existing Entity rows,
NEW FG rows, then existing Datasource rows.  FG rows are inserted
between Entity and Datasource so the table groups "things deployed
in the schema" (FV / Entity / FG) above the derived Datasource block.
"""

from __future__ import annotations

from typing import Any

from snowflake.ml.feature_store.decl.api import enrich_list_results
from snowflake.ml.feature_store.decl.types import ObjectKind
from snowflake.ml.test_utils import pytest_driver


def _fg_row(
    name: str = "USER_FRAUD_FG",
    *,
    version: str = "V1",
    sources: list[dict[str, Any]] | None = None,
    desc: str = "",
    auto_prefix: bool = True,
    database_name: str = "MYDB",
    schema_name: str = "PUBLIC",
) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "desc": desc,
        "owner": "ROLE",
        "auto_prefix": auto_prefix,
        "sources": sources or [{"fv_name": "FV_A", "fv_version": "V1"}],
        "output_columns": None,
        "database_name": database_name,
        "schema_name": schema_name,
    }


class TestEnrichEmitsFGRows:
    def test_single_fg_row_emits_one_output_row(self) -> None:
        out = enrich_list_results(
            show_rows=[],
            entity_rows=[],
            feature_group_rows=[_fg_row(name="USER_FRAUD_FG")],
        )
        fg_rows = [r for r in out if r["type"] == ObjectKind.FEATURE_GROUP]
        assert len(fg_rows) == 1
        r = fg_rows[0]
        assert r["name"] == "USER_FRAUD_FG"
        assert r["version"] == "V1"
        assert r["database_name"] == "MYDB"
        assert r["schema_name"] == "PUBLIC"

    def test_fg_row_summarises_sources_in_details(self) -> None:
        sources = [
            {"fv_name": "FV_A", "fv_version": "V1"},
            {"fv_name": "FV_B", "fv_version": "V2", "alias": "b"},
        ]
        out = enrich_list_results(
            show_rows=[],
            entity_rows=[],
            feature_group_rows=[_fg_row(name="MULTI", sources=sources)],
        )
        r = next(r for r in out if r["type"] == ObjectKind.FEATURE_GROUP)
        details = r["details"]
        assert details["source_count"] == 2
        assert details["sources"] == ["FV_A:V1", "FV_B:V2"]

    def test_fg_rows_appear_after_entity_rows(self) -> None:
        ent_row = {
            "name": "SNOWML_FEATURE_STORE_ENTITY_USER",
            "database_name": "MYDB",
            "schema_name": "PUBLIC",
            "allowed_values": '["USER_ID"]',
        }
        out = enrich_list_results(
            show_rows=[],
            entity_rows=[ent_row],
            feature_group_rows=[_fg_row(name="FG1")],
        )
        types = [r["type"] for r in out]
        assert ObjectKind.ENTITY in types
        assert ObjectKind.FEATURE_GROUP in types
        # FG must be after Entity in display order.
        assert types.index(ObjectKind.FEATURE_GROUP) > types.index(ObjectKind.ENTITY)

    def test_no_fg_rows_when_kwarg_omitted(self) -> None:
        out = enrich_list_results(show_rows=[], entity_rows=[])
        assert all(r["type"] != ObjectKind.FEATURE_GROUP for r in out)

    def test_no_fg_rows_when_kwarg_is_empty_list(self) -> None:
        out = enrich_list_results(show_rows=[], entity_rows=[], feature_group_rows=[])
        assert all(r["type"] != ObjectKind.FEATURE_GROUP for r in out)


if __name__ == "__main__":
    pytest_driver.main()
