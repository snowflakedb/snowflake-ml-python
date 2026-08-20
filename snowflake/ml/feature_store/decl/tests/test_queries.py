"""Tests for decl/queries.py — SQL string factory functions."""

from __future__ import annotations

import pytest

from snowflake.ml.feature_store.decl.queries import (
    describe_columns_query,
    describe_query,
    describe_specification_query,
    drop_queries,
    list_query,
    list_state_queries,
    state_queries,
)

# ---------------------------------------------------------------------------
# state_queries
# ---------------------------------------------------------------------------


class TestStateQueries:
    def test_returns_dict(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert isinstance(result, dict)

    def test_show_ofts_key_present(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "show_ofts" in result

    def test_show_tables_key_present(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "show_tables" in result

    def test_show_ofts_contains_db_and_schema(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "MYDB" in result["show_ofts"]
        assert "MYSCHEMA" in result["show_ofts"]

    def test_show_tables_contains_db_and_schema(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "MYDB" in result["show_tables"]
        assert "MYSCHEMA" in result["show_tables"]

    def test_show_ofts_is_show_online_feature_tables(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "SHOW ONLINE FEATURE TABLES" in result["show_ofts"]

    def test_show_tables_is_show_tables_like(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "SHOW TABLES" in result["show_tables"]
        assert "%" in result["show_tables"]

    def test_show_ofts_exact(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert result["show_ofts"] == "SHOW ONLINE FEATURE TABLES IN SCHEMA MYDB.MYSCHEMA"

    def test_show_tables_exact(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert result["show_tables"] == "SHOW TABLES LIKE '%' IN SCHEMA MYDB.MYSCHEMA"

    def test_different_db_and_schema(self) -> None:
        result = state_queries("PROD_DB", "FS_SCHEMA")
        assert "PROD_DB" in result["show_ofts"]
        assert "FS_SCHEMA" in result["show_ofts"]
        assert "PROD_DB" in result["show_tables"]
        assert "FS_SCHEMA" in result["show_tables"]

    def test_show_dynamic_tables_key_present(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "show_dynamic_tables" in result, (
            "state_queries() must surface SHOW DYNAMIC TABLES so the offline-DT "
            "text column (the FROM clause) is available to "
            "fetch_applied_state — the deployed BatchFV SPECIFICATION JSON "
            "always has sources=[] (lossy round-trip), so DT text parsing is "
            "the only way to recover the source-table binding."
        )

    def test_show_dynamic_tables_is_show_dynamic_tables(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "SHOW DYNAMIC TABLES" in result["show_dynamic_tables"]

    def test_show_dynamic_tables_contains_db_and_schema(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert "MYDB" in result["show_dynamic_tables"]
        assert "MYSCHEMA" in result["show_dynamic_tables"]

    def test_show_dynamic_tables_exact(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        assert result["show_dynamic_tables"] == "SHOW DYNAMIC TABLES IN SCHEMA MYDB.MYSCHEMA"


# ---------------------------------------------------------------------------
# list_query
# ---------------------------------------------------------------------------


class TestListQuery:
    def test_returns_string(self) -> None:
        result = list_query("MYDB", "MYSCHEMA")
        assert isinstance(result, str)

    def test_contains_db_and_schema(self) -> None:
        result = list_query("MYDB", "MYSCHEMA")
        assert "MYDB" in result
        assert "MYSCHEMA" in result

    def test_exact_sql(self) -> None:
        result = list_query("MYDB", "MYSCHEMA")
        assert result == "SHOW ONLINE FEATURE TABLES IN SCHEMA MYDB.MYSCHEMA"

    def test_is_show_online_feature_tables(self) -> None:
        result = list_query("MYDB", "MYSCHEMA")
        assert result.startswith("SHOW ONLINE FEATURE TABLES")

    def test_different_db_and_schema(self) -> None:
        result = list_query("PROD_DB", "FS_SCHEMA")
        assert "PROD_DB" in result
        assert "FS_SCHEMA" in result


# ---------------------------------------------------------------------------
# describe_query
# ---------------------------------------------------------------------------


class TestDescribeQuery:
    def test_returns_string(self) -> None:
        result = describe_query("my_table", "MYDB", "MYSCHEMA")
        assert isinstance(result, str)

    def test_contains_name(self) -> None:
        result = describe_query("my_table", "MYDB", "MYSCHEMA")
        assert "my_table" in result

    def test_contains_db_and_schema(self) -> None:
        result = describe_query("my_table", "MYDB", "MYSCHEMA")
        assert "MYDB" in result
        assert "MYSCHEMA" in result

    def test_exact_sql(self) -> None:
        result = describe_query("my_table", "MYDB", "MYSCHEMA")
        assert result == "SHOW ONLINE FEATURE TABLES LIKE 'my_table' IN SCHEMA MYDB.MYSCHEMA"

    def test_name_is_single_quoted(self) -> None:
        result = describe_query("my_table", "MYDB", "MYSCHEMA")
        assert "'my_table'" in result

    def test_name_with_underscores_and_digits(self) -> None:
        result = describe_query("my_fv_v1", "DB", "SCH")
        assert "my_fv_v1" in result

    def test_is_show_online_feature_tables_like(self) -> None:
        result = describe_query("tbl", "MYDB", "MYSCHEMA")
        assert "SHOW ONLINE FEATURE TABLES LIKE" in result


# ---------------------------------------------------------------------------
# describe_columns_query
# ---------------------------------------------------------------------------


class TestDescribeColumnsQuery:
    def test_returns_string(self) -> None:
        result = describe_columns_query("my_table", "MYDB", "MYSCHEMA")
        assert isinstance(result, str)

    def test_contains_name(self) -> None:
        result = describe_columns_query("my_table", "MYDB", "MYSCHEMA")
        assert "my_table" in result

    def test_contains_db_and_schema(self) -> None:
        result = describe_columns_query("my_table", "MYDB", "MYSCHEMA")
        assert "MYDB" in result
        assert "MYSCHEMA" in result

    def test_starts_with_describe(self) -> None:
        result = describe_columns_query("my_table", "MYDB", "MYSCHEMA")
        assert result.startswith("DESCRIBE ")

    def test_exact_sql(self) -> None:
        result = describe_columns_query("my_table", "MYDB", "MYSCHEMA")
        assert result == 'DESCRIBE ONLINE FEATURE TABLE "MYDB"."MYSCHEMA"."my_table"'

    def test_components_are_double_quoted(self) -> None:
        result = describe_columns_query("my_table", "MYDB", "MYSCHEMA")
        assert '"MYDB"' in result
        assert '"MYSCHEMA"' in result
        assert '"my_table"' in result

    def test_name_with_hyphens_is_quoted(self) -> None:
        result = describe_columns_query("my-table", "MYDB", "MYSCHEMA")
        assert '"my-table"' in result

    def test_name_with_spaces_is_quoted(self) -> None:
        result = describe_columns_query("my table", "MYDB", "MYSCHEMA")
        assert '"my table"' in result


# ---------------------------------------------------------------------------
# drop_queries
# ---------------------------------------------------------------------------


class TestDropQueries:
    def test_returns_list(self) -> None:
        result = drop_queries(["tbl1"], "MYDB", "MYSCHEMA")
        assert isinstance(result, list)

    def test_empty_names_returns_empty_list(self) -> None:
        result = drop_queries([], "MYDB", "MYSCHEMA")
        assert result == []

    def test_one_name_returns_one_sql(self) -> None:
        result = drop_queries(["tbl1"], "MYDB", "MYSCHEMA")
        assert len(result) == 1

    def test_multiple_names_returns_multiple_sqls(self) -> None:
        result = drop_queries(["tbl1", "tbl2", "tbl3"], "MYDB", "MYSCHEMA")
        assert len(result) == 3

    def test_each_sql_contains_its_name(self) -> None:
        result = drop_queries(["alpha", "beta"], "MYDB", "MYSCHEMA")
        assert any("alpha" in s for s in result)
        assert any("beta" in s for s in result)

    def test_each_sql_contains_db_and_schema(self) -> None:
        result = drop_queries(["tbl1"], "MYDB", "MYSCHEMA")
        assert "MYDB" in result[0]
        assert "MYSCHEMA" in result[0]

    def test_each_sql_is_drop_if_exists(self) -> None:
        result = drop_queries(["tbl1"], "MYDB", "MYSCHEMA")
        assert "DROP ONLINE FEATURE TABLE IF EXISTS" in result[0]

    def test_exact_sql(self) -> None:
        result = drop_queries(["my_table"], "MYDB", "MYSCHEMA")
        assert result[0] == 'DROP ONLINE FEATURE TABLE IF EXISTS "MYDB"."MYSCHEMA"."my_table"'

    def test_names_are_double_quoted(self) -> None:
        result = drop_queries(["my-table"], "MYDB", "MYSCHEMA")
        assert '"my-table"' in result[0]

    def test_order_preserved(self) -> None:
        names = ["zzz", "aaa", "mmm"]
        result = drop_queries(names, "MYDB", "MYSCHEMA")
        assert "zzz" in result[0]
        assert "aaa" in result[1]
        assert "mmm" in result[2]

    def test_each_sql_is_string(self) -> None:
        result = drop_queries(["tbl1", "tbl2"], "MYDB", "MYSCHEMA")
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# describe_specification_query
# ---------------------------------------------------------------------------


class TestDescribeSpecificationQuery:
    def test_returns_string(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert isinstance(result, str)

    def test_contains_name(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert "my_table" in result

    def test_contains_db_and_schema(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert "MYDB" in result
        assert "MYSCHEMA" in result

    def test_starts_with_describe(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert result.startswith("DESCRIBE ")

    def test_includes_type_specification_clause(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert "TYPE = SPECIFICATION" in result

    def test_exact_sql(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert result == 'DESCRIBE ONLINE FEATURE TABLE "MYDB"."MYSCHEMA"."my_table" TYPE = SPECIFICATION'

    def test_components_are_double_quoted(self) -> None:
        result = describe_specification_query("MYDB", "MYSCHEMA", "my_table")
        assert '"MYDB"' in result
        assert '"MYSCHEMA"' in result
        assert '"my_table"' in result


# ---------------------------------------------------------------------------
# list_state_queries
#
# The historical ``list_entities_query`` factory and the ``show_entities``
# key in ``list_state_queries`` were deleted in the entity round-trip
# migration: entity rows are now fetched via the imperative
# ``FeatureStore.list_entities()`` (through :func:`decl.api.fetch_entity_rows`)
# rather than a duplicated SQL string here.
# ---------------------------------------------------------------------------


class TestListStateQueries:
    def test_returns_dict(self) -> None:
        result = list_state_queries("MYDB", "MYSCHEMA")
        assert isinstance(result, dict)

    def test_includes_show_ofts(self) -> None:
        result = list_state_queries("MYDB", "MYSCHEMA")
        assert "show_ofts" in result
        assert result["show_ofts"] == "SHOW ONLINE FEATURE TABLES IN SCHEMA MYDB.MYSCHEMA"

    def test_does_not_include_show_entities(self) -> None:
        """``show_entities`` was removed when entity-row fetching moved to
        the imperative path; pin the absence so future regressions can't
        silently re-introduce a duplicated SHOW TAGS query."""
        result = list_state_queries("MYDB", "MYSCHEMA")
        assert "show_entities" not in result

    def test_includes_describe_specification_template(self) -> None:
        result = list_state_queries("MYDB", "MYSCHEMA")
        assert "describe_specification_template" in result
        assert "{name}" in result["describe_specification_template"]

    def test_describe_specification_template_formats_correctly(self) -> None:
        result = list_state_queries("MYDB", "MYSCHEMA")
        sql = result["describe_specification_template"].format(name="my_table")
        assert sql == 'DESCRIBE ONLINE FEATURE TABLE "MYDB"."MYSCHEMA"."my_table" TYPE = SPECIFICATION'


# ---------------------------------------------------------------------------
# state_queries — extended for SPECIFICATION
# ---------------------------------------------------------------------------


class TestStateQueriesSpecificationTemplate:
    def test_state_queries_now_includes_describe_specification_template(self) -> None:
        """state_queries gains the SPECIFICATION template so apply can do full-spec diffs."""
        result = state_queries("MYDB", "MYSCHEMA")
        assert "describe_specification_template" in result

    def test_state_queries_describe_specification_template_formats(self) -> None:
        result = state_queries("MYDB", "MYSCHEMA")
        sql = result["describe_specification_template"].format(name="my_table")
        assert "TYPE = SPECIFICATION" in sql
        assert '"my_table"' in sql


# ---------------------------------------------------------------------------
# dynamic_tables_query — offline-DT text column for BatchFV source recovery.
#
# The deployed BatchFV SPECIFICATION JSON always returns ``sources: []`` (a
# lossy snowml-core round-trip on the FROM SPECIFICATION path).  The offline
# Dynamic Table's ``text`` column preserves the real source binding in the
# ``SELECT ... FROM <db>.<schema>.<table>`` clause, so the declarative state
# fetcher recovers ``sources[0].table`` from this row.  See
# docs/BATCH_FV_BUG_BASH.md §7/§8 for the user-visible motivation.
# ---------------------------------------------------------------------------


class TestDynamicTablesQuery:
    def test_state_queries_now_includes_show_dynamic_tables(self) -> None:
        """state_queries gains a SHOW DYNAMIC TABLES query so fetch_applied_state
        can recover the BatchFV source-table binding from the offline DT's
        ``text`` (FROM clause).
        """
        from snowflake.ml.feature_store.decl.queries import dynamic_tables_query

        result = state_queries("MYDB", "MYSCHEMA")
        assert "show_dynamic_tables" in result
        assert result["show_dynamic_tables"] == dynamic_tables_query("MYDB", "MYSCHEMA")

    def test_dynamic_tables_query_is_show_dynamic_tables(self) -> None:
        from snowflake.ml.feature_store.decl.queries import dynamic_tables_query

        sql = dynamic_tables_query("MYDB", "MYSCHEMA")
        assert sql == "SHOW DYNAMIC TABLES IN SCHEMA MYDB.MYSCHEMA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
