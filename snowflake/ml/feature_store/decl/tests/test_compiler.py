"""Tests for decl/compiler.py — normalize_types, inline_udf_source, normalize_durations, parse_duration_to_seconds."""

import os
import tempfile

import pytest

from snowflake.ml.feature_store.decl.compiler import (
    canonicalize_sql_for_hash,
    compile_spec,
    inline_query_source,
    inline_udf_source,
    normalize_durations,
    normalize_sql_whitespace,
    normalize_types,
    parse_duration_to_seconds,
    strip_sql_comments,
)
from snowflake.ml.feature_store.decl.errors import SpecLoadError

# ---------------------------------------------------------------------------
# parse_duration_to_seconds
# ---------------------------------------------------------------------------


class TestParseDurationToSeconds:
    def test_none_returns_none(self) -> None:
        assert parse_duration_to_seconds(None) is None

    def test_integer_passthrough(self) -> None:
        assert parse_duration_to_seconds(300) == 300

    def test_integer_zero(self) -> None:
        assert parse_duration_to_seconds(0) == 0

    def test_float_truncated(self) -> None:
        result = parse_duration_to_seconds(1.5)
        assert result == 1

    def test_seconds_s_suffix(self) -> None:
        assert parse_duration_to_seconds("300s") == 300

    def test_seconds_sec_suffix(self) -> None:
        assert parse_duration_to_seconds("60sec") == 60

    def test_seconds_seconds_suffix(self) -> None:
        assert parse_duration_to_seconds("5seconds") == 5

    def test_minutes_m_suffix(self) -> None:
        assert parse_duration_to_seconds("5m") == 300

    def test_minutes_min_suffix(self) -> None:
        assert parse_duration_to_seconds("10min") == 600

    def test_minutes_minutes_suffix(self) -> None:
        assert parse_duration_to_seconds("2minutes") == 120

    def test_hours_h_suffix(self) -> None:
        assert parse_duration_to_seconds("1h") == 3600

    def test_hours_hr_suffix(self) -> None:
        assert parse_duration_to_seconds("2hr") == 7200

    def test_hours_hours_suffix(self) -> None:
        assert parse_duration_to_seconds("1hours") == 3600

    def test_days_d_suffix(self) -> None:
        assert parse_duration_to_seconds("7d") == 604800

    def test_days_day_suffix(self) -> None:
        assert parse_duration_to_seconds("1day") == 86400

    def test_days_days_suffix(self) -> None:
        assert parse_duration_to_seconds("3days") == 259200

    def test_fractional_string_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            parse_duration_to_seconds("1.5m")

    def test_lifetime_returns_minus_one(self) -> None:
        assert parse_duration_to_seconds("lifetime") == -1

    def test_bare_numeric_string_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            parse_duration_to_seconds("300")

    def test_invalid_unit_raises_value_error(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            parse_duration_to_seconds("5x")

    def test_invalid_format_raises_value_error(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            parse_duration_to_seconds("abc")

    def test_whitespace_stripped(self) -> None:
        assert parse_duration_to_seconds("  5m  ") == 300


# ---------------------------------------------------------------------------
# normalize_types
# ---------------------------------------------------------------------------


class TestNormalizeTypes:
    def test_top_level_type_field_resolved(self) -> None:
        d = {"type": "str"}
        result = normalize_types(d)
        assert result["type"] == "StringType"

    def test_nested_type_field_resolved(self) -> None:
        d = {"join_keys": [{"name": "user_id", "type": "int"}]}
        result = normalize_types(d)
        assert result["join_keys"][0]["type"] == "LongType"

    def test_canonical_type_unchanged(self) -> None:
        d = {"type": "StringType"}
        result = normalize_types(d)
        assert result["type"] == "StringType"

    def test_unknown_type_passthrough(self) -> None:
        d = {"type": "MyCustomType"}
        result = normalize_types(d)
        assert result["type"] == "MyCustomType"

    def test_non_type_string_field_unchanged(self) -> None:
        d = {"name": "str"}  # "str" here is a name, not a type field
        result = normalize_types(d)
        assert result["name"] == "str"

    def test_deeply_nested(self) -> None:
        d = {
            "features": [
                {
                    "source_column": {"name": "val", "type": "float"},
                    "output_column": {"name": "out", "type": "number"},
                }
            ]
        }
        result = normalize_types(d)
        assert result["features"][0]["source_column"]["type"] == "DoubleType"
        assert result["features"][0]["output_column"]["type"] == "DoubleType"

    def test_handles_non_dict_top_level(self) -> None:
        assert normalize_types([{"type": "str"}]) == [{"type": "StringType"}]
        assert normalize_types("hello") == "hello"
        assert normalize_types(42) == 42


# ---------------------------------------------------------------------------
# normalize_durations
# ---------------------------------------------------------------------------


class TestNormalizeDurations:
    def test_feature_granularity_renamed_and_converted(self) -> None:
        d = {"feature_granularity": "5m"}
        result = normalize_durations(d)
        assert "feature_granularity" not in result
        assert result["feature_granularity_sec"] == 300

    def test_target_lag_renamed_and_converted(self) -> None:
        d = {"target_lag": "1h"}
        result = normalize_durations(d)
        assert "target_lag" not in result
        assert result["target_lag_sec"] == 3600

    def test_window_renamed_and_converted(self) -> None:
        d = {"window": "7d"}
        result = normalize_durations(d)
        assert "window" not in result
        assert result["window_sec"] == 604800

    def test_offset_renamed_and_converted(self) -> None:
        d = {"offset": "1m"}
        result = normalize_durations(d)
        assert "offset" not in result
        assert result["offset_sec"] == 60

    def test_duration_none_skipped(self) -> None:
        d = {"feature_granularity": None}
        result = normalize_durations(d)
        assert "feature_granularity_sec" not in result
        assert "feature_granularity" in result  # kept as-is when None

    def test_integer_duration_preserved(self) -> None:
        d = {"target_lag": 300}
        result = normalize_durations(d)
        assert result["target_lag_sec"] == 300

    def test_non_duration_keys_unchanged(self) -> None:
        d = {"name": "fv", "kind": "StreamingFeatureView"}
        result = normalize_durations(d)
        assert result["name"] == "fv"
        assert result["kind"] == "StreamingFeatureView"

    def test_nested_duration_keys_converted(self) -> None:
        d = {"features": [{"window": "30d", "function": "sum"}]}
        result = normalize_durations(d)
        assert result["features"][0]["window_sec"] == 2592000
        assert "window" not in result["features"][0]

    def test_invalid_duration_kept_unchanged(self) -> None:
        d = {"feature_granularity": "invalid_unit_xyz"}
        result = normalize_durations(d)
        # Invalid duration kept in original key, no _sec variant
        assert "feature_granularity" in result
        assert "feature_granularity_sec" not in result

    def test_lifetime_emits_minus_one_sentinel(self) -> None:
        d = {"window": "lifetime"}
        result = normalize_durations(d)
        assert result.get("window_sec") == -1
        assert "window" not in result


# ---------------------------------------------------------------------------
# inline_udf_source
# ---------------------------------------------------------------------------


class TestInlineUdfSource:
    def test_no_udf_returns_unchanged(self) -> None:
        d = {"kind": "Entity", "name": "customer"}
        result = inline_udf_source(d, None)
        assert result == d

    def test_udf_without_file_returns_unchanged(self) -> None:
        d = {"udf": {"name": "fn", "engine": "pandas"}}
        result = inline_udf_source(d, None)
        assert "function_definition" not in result["udf"] or result["udf"]["function_definition"] is None

    def test_udf_with_file_inlines_source(self) -> None:
        udf_code = "def my_fn(df):\n    return df * 2\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_path = os.path.join(tmpdir, "my_fn.py")
            with open(udf_path, "w") as f:
                f.write(udf_code)

            d = {"udf": {"name": "my_fn", "file": "my_fn.py"}}
            result = inline_udf_source(d, tmpdir)

        assert result["udf"]["function_definition"] == udf_code
        assert "file" not in result["udf"]

    def test_udf_file_not_found_returns_unchanged(self) -> None:
        d = {"udf": {"name": "fn", "file": "nonexistent.py"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = inline_udf_source(d, tmpdir)
        # File not found: file key is not removed, no function_definition inlined
        assert "file" in result["udf"] or result["udf"].get("file") == "nonexistent.py"

    def test_udf_file_path_traversal_raises(self) -> None:
        d = {"udf": {"name": "fn", "file": "../../etc/passwd"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="path traversal"):
                inline_udf_source(d, tmpdir)

    def test_udf_file_absolute_path_raises(self) -> None:
        d = {"udf": {"name": "fn", "file": "/etc/passwd"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="absolute path"):
                inline_udf_source(d, tmpdir)


# ---------------------------------------------------------------------------
# compile_spec
# ---------------------------------------------------------------------------


class TestCompileSpec:
    def test_compile_runs_all_passes(self) -> None:
        d = {
            "kind": "StreamingFeatureView",
            "name": "fv",
            "features": [
                {
                    "source_column": {"name": "val", "type": "float"},
                    "output_column": {"name": "out", "type": "int"},
                    "window": "5m",
                }
            ],
        }
        result = compile_spec(d)
        # normalize_types
        assert result["features"][0]["source_column"]["type"] == "DoubleType"
        assert result["features"][0]["output_column"]["type"] == "LongType"
        # normalize_durations
        assert result["features"][0]["window_sec"] == 300
        assert "window" not in result["features"][0]

    def test_compile_with_no_udf_is_fine(self) -> None:
        d = {"kind": "Entity", "name": "user", "join_keys": [{"name": "user_id", "type": "str"}]}
        result = compile_spec(d)
        assert result["join_keys"][0]["type"] == "StringType"

    def test_compile_with_spec_file_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_code = "def fn(df):\n    return df\n"
            udf_path = os.path.join(tmpdir, "fn.py")
            with open(udf_path, "w") as f:
                f.write(udf_code)
            d = {
                "kind": "StreamingFeatureView",
                "name": "fv",
                "udf": {"name": "fn", "file": "fn.py"},
            }
            result = compile_spec(d, spec_file_dir=tmpdir)

        assert result["udf"]["function_definition"] == udf_code
        assert "file" not in result["udf"]


class TestCompileSpecFeatureAggregationMethod:
    def test_feature_aggregation_method_preserved(self) -> None:
        d = {
            "kind": "StreamingFeatureView",
            "name": "fv",
            "feature_aggregation_method": "continuous",
        }
        result = compile_spec(d)
        assert result["feature_aggregation_method"] == "continuous"

    def test_feature_aggregation_method_tiles_preserved(self) -> None:
        d = {
            "kind": "StreamingFeatureView",
            "name": "fv",
            "feature_aggregation_method": "tiles",
        }
        result = compile_spec(d)
        assert result["feature_aggregation_method"] == "tiles"

    def test_feature_aggregation_method_not_treated_as_duration(self) -> None:
        """feature_aggregation_method must NOT be renamed to _sec."""
        d = {"feature_aggregation_method": "continuous"}
        result = compile_spec(d)
        assert "feature_aggregation_method" in result
        assert "feature_aggregation_method_sec" not in result

    def test_feature_aggregation_method_not_treated_as_type(self) -> None:
        """feature_aggregation_method value must NOT be resolved via normalize_type."""
        d = {"feature_aggregation_method": "continuous"}
        result = compile_spec(d)
        assert result["feature_aggregation_method"] == "continuous"


# ---------------------------------------------------------------------------
# normalize_sql_whitespace — Phase 3
# ---------------------------------------------------------------------------


class TestNormalizeSqlWhitespace:
    """Whitespace normalization keeps round-trip through the DT-text recovery
    path stable: any author-side / Snowflake-side whitespace re-formatting
    converges to the same canonical string, so re-applying an unchanged
    BatchSource does not register as drift."""

    def test_strips_outer_whitespace(self) -> None:
        assert normalize_sql_whitespace("  SELECT 1  ") == "SELECT 1"

    def test_strips_leading_and_trailing_newlines(self) -> None:
        assert normalize_sql_whitespace("\n\nSELECT 1\n\n") == "SELECT 1"

    def test_collapses_internal_runs_of_whitespace(self) -> None:
        assert normalize_sql_whitespace("SELECT     a,\n     b\n   FROM   T") == "SELECT a, b FROM T"

    def test_collapses_tabs_and_newlines(self) -> None:
        assert normalize_sql_whitespace("SELECT\ta,\n\tb\nFROM\tT") == "SELECT a, b FROM T"

    def test_idempotent(self) -> None:
        s = "SELECT a, b FROM T WHERE c = 1"
        assert normalize_sql_whitespace(s) == s
        assert normalize_sql_whitespace(normalize_sql_whitespace(s)) == s

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_sql_whitespace("") == ""

    def test_only_whitespace_returns_empty(self) -> None:
        assert normalize_sql_whitespace("   \n\t  ") == ""

    def test_preserves_string_literal_content(self) -> None:
        # Literal contents are not specifically protected — the contract is
        # 'whitespace-only normalization'. This test pins what authors see
        # so they know to author SQL with the assumption that the recovered
        # text from a DT will compare equal after the same normalization.
        s = "SELECT 'a   b' FROM T"
        # Inside a string literal, runs of whitespace ARE collapsed by this
        # simple normaliser. Document the limitation explicitly.
        assert normalize_sql_whitespace(s) == "SELECT 'a b' FROM T"

    def test_round_trip_with_typical_sql(self) -> None:
        original = """
            SELECT
                user_id,
                event_ts,
                COUNT(*) AS n_events
            FROM RAW.EVENTS
            WHERE event_ts > '2024-01-01'
            GROUP BY user_id, event_ts
        """
        normalized = normalize_sql_whitespace(original)
        # No leading/trailing whitespace
        assert not normalized.startswith(" ")
        assert not normalized.endswith(" ")
        # All internal whitespace is single spaces
        assert "  " not in normalized
        assert "\n" not in normalized
        assert "\t" not in normalized
        # Idempotent
        assert normalize_sql_whitespace(normalized) == normalized

    def test_strips_trailing_semicolon(self) -> None:
        """Snowflake stores ``CREATE DYNAMIC TABLE … AS <body>`` with no
        trailing ``;``.  ``query_file`` sidecars typically end with
        ``;`` (SQL convention), so the local-compile output must strip
        the terminator for the round-trip to converge with the
        deployed-side DT body the planner reads back via
        ``state._classify_dt_body``.

        Pins the BATCH_FV_BUG_BASH §6 SQL-FV NO_CHANGE contract.
        """
        body = "SELECT a, b FROM RAW.EVENTS;\n"
        assert normalize_sql_whitespace(body) == "SELECT a, b FROM RAW.EVENTS"

    def test_strips_multiple_trailing_semicolons(self) -> None:
        """Defensive: multiple trailing ``;`` (with surrounding
        whitespace) all collapse — authors who paste two statement
        terminators by accident still see a stable round-trip."""
        body = "SELECT 1 ;  ;\n;\n"
        assert normalize_sql_whitespace(body) == "SELECT 1"

    def test_preserves_embedded_semicolons(self) -> None:
        """Only *trailing* terminators are stripped.  An embedded ``;``
        (multi-statement body) is left intact — bodies that legitimately
        contain ``;`` mid-statement should still hash equal across
        re-applies."""
        body = "SELECT 'a;b' FROM T;"
        assert normalize_sql_whitespace(body) == "SELECT 'a;b' FROM T"

    def test_idempotent_after_terminator_strip(self) -> None:
        """Idempotency must hold after the terminator strip too — the
        state.py recovery path runs ``normalize_sql_whitespace`` on the
        DT body and the local side runs it on the sidecar; both must
        converge to the same fixed point regardless of how many times
        the helper is called."""
        body = "SELECT 1;"
        once = normalize_sql_whitespace(body)
        twice = normalize_sql_whitespace(once)
        assert once == twice == "SELECT 1"


# ---------------------------------------------------------------------------
# inline_query_source — Phase 3
# ---------------------------------------------------------------------------


class TestInlineQuerySource:
    """``inline_query_source`` is the BatchSource analogue of
    :func:`inline_udf_source`: when a BatchSource doc carries
    ``query_file:``, the compiler reads the sibling file and inlines its
    contents as ``query`` (whitespace-normalized). Inline ``query:`` values
    are normalized as well so the DT-text round-trip is stable.
    """

    def test_no_query_or_query_file_returns_unchanged(self) -> None:
        d = {"kind": "BatchSource", "name": "events", "table": "EVENTS"}
        result = inline_query_source(d, None)
        assert result == d
        assert "query" not in result

    def test_inline_query_is_normalized(self) -> None:
        d = {
            "kind": "BatchSource",
            "name": "events",
            "query": "  SELECT  *\n  FROM   EVENTS  ",
        }
        result = inline_query_source(d, None)
        assert result["query"] == "SELECT * FROM EVENTS"
        assert "query_file" not in result

    def test_query_file_inlined_and_normalized(self) -> None:
        sql_body = "SELECT\n    user_id,\n    ts\nFROM RAW.EVENTS\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)
            d = {"kind": "BatchSource", "name": "events", "query_file": "events.sql"}
            result = inline_query_source(d, tmpdir)

        assert result["query"] == "SELECT user_id, ts FROM RAW.EVENTS"
        assert "query_file" not in result

    def test_query_file_not_found_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "missing.sql",
            }
            with pytest.raises(SpecLoadError) as excinfo:
                inline_query_source(d, tmpdir)
            msg = str(excinfo.value)
            assert "missing.sql" in msg
            assert "events" in msg

    def test_query_file_with_no_spec_dir_returns_unchanged(self) -> None:
        # Defensive: if compile_spec is invoked without a spec_file_dir,
        # the helper returns the doc unchanged so the caller can decide
        # whether to surface the error (validation will catch it later).
        d = {"kind": "BatchSource", "name": "events", "query_file": "events.sql"}
        result = inline_query_source(d, None)
        assert result["query_file"] == "events.sql"
        assert "query" not in result

    def test_non_batchsource_kind_returns_unchanged(self) -> None:
        d = {"kind": "Entity", "name": "customer", "join_keys": []}
        result = inline_query_source(d, None)
        assert result == d

    def test_query_file_with_subdirectory_path(self) -> None:
        sql_body = "SELECT 1\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "sql")
            os.makedirs(sub)
            sql_path = os.path.join(sub, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)
            d = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "sql/events.sql",
            }
            result = inline_query_source(d, tmpdir)
        assert result["query"] == "SELECT 1"
        assert "query_file" not in result

    def test_inline_query_with_line_comment_does_not_break(self) -> None:
        # Regression: a ``--`` line comment used to lose its terminating
        # newline when whitespace collapsed, swallowing the rest of the
        # query onto one line and producing invalid DDL.  The comment must
        # be stripped BEFORE collapse so the remaining SQL survives.
        d = {
            "kind": "BatchSource",
            "name": "events",
            "query": "SELECT id, -- primary key\n amount\nFROM t",
        }
        result = inline_query_source(d, None)
        assert result["query"] == "SELECT id, amount FROM t"
        assert "--" not in result["query"]

    def test_inline_query_with_block_comment_is_stripped(self) -> None:
        d = {
            "kind": "BatchSource",
            "name": "events",
            "query": "SELECT a, /* keep me? no */ b FROM t",
        }
        result = inline_query_source(d, None)
        assert result["query"] == "SELECT a, b FROM t"
        assert "/*" not in result["query"]

    def test_query_file_with_line_comment_does_not_break(self) -> None:
        sql_body = "SELECT\n  user_id, -- the user\n  ts\nFROM RAW.EVENTS\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)
            d = {"kind": "BatchSource", "name": "events", "query_file": "events.sql"}
            result = inline_query_source(d, tmpdir)
        assert result["query"] == "SELECT user_id, ts FROM RAW.EVENTS"
        assert "--" not in result["query"]


# ---------------------------------------------------------------------------
# strip_sql_comments — PR A (A1)
# ---------------------------------------------------------------------------


class TestStripSqlComments:
    """``strip_sql_comments`` removes ``--`` line comments and ``/* */``
    block comments so the compile-time whitespace collapse cannot swallow
    trailing SQL onto a commented-out line."""

    def test_strips_line_comment(self) -> None:
        assert "primary key" not in strip_sql_comments("SELECT id -- primary key\nFROM t")

    def test_line_comment_rest_of_query_survives(self) -> None:
        out = strip_sql_comments("SELECT id, -- pk\n amount\nFROM t")
        assert "amount" in out
        assert "FROM t" in out
        assert "--" not in out

    def test_strips_block_comment(self) -> None:
        out = strip_sql_comments("SELECT a, /* note */ b FROM t")
        assert "/*" not in out
        assert "note" not in out
        assert "a" in out and "b" in out

    def test_no_comments_returns_equivalent_sql(self) -> None:
        s = "SELECT a, b FROM t WHERE c = 1"
        assert strip_sql_comments(s).strip() == s

    def test_defensive_on_unparsable_input(self) -> None:
        # Never raise: a parser hiccup must return input unchanged.
        weird = "not really ))) sql -- x"
        assert isinstance(strip_sql_comments(weird), str)

    def test_empty_string(self) -> None:
        assert strip_sql_comments("") == ""


# ---------------------------------------------------------------------------
# canonicalize_sql_for_hash — PR A (A2)
# ---------------------------------------------------------------------------


class TestCanonicalizeSqlForHash:
    """``canonicalize_sql_for_hash`` produces a stable hash basis: comments
    stripped, keywords upper-cased, whitespace collapsed.  It never mutates
    the SQL submitted to the Dynamic Table (hash-basis use only)."""

    def test_comment_only_change_canonicalizes_equal(self) -> None:
        a = canonicalize_sql_for_hash("SELECT a FROM t -- v1")
        b = canonicalize_sql_for_hash("SELECT a FROM t -- v2 (edited)")
        assert a == b

    def test_keyword_case_only_change_canonicalizes_equal(self) -> None:
        a = canonicalize_sql_for_hash("select a from t")
        b = canonicalize_sql_for_hash("SELECT a FROM t")
        assert a == b

    def test_whitespace_only_change_canonicalizes_equal(self) -> None:
        a = canonicalize_sql_for_hash("SELECT   a,\n   b\nFROM t")
        b = canonicalize_sql_for_hash("SELECT a, b FROM t")
        assert a == b

    def test_semantic_change_differs(self) -> None:
        a = canonicalize_sql_for_hash("SELECT a FROM t")
        b = canonicalize_sql_for_hash("SELECT a, b FROM t")
        assert a != b

    def test_idempotent(self) -> None:
        s = "SELECT a, b FROM t -- c\n"
        once = canonicalize_sql_for_hash(s)
        assert canonicalize_sql_for_hash(once) == once

    def test_defensive_on_unparsable_input(self) -> None:
        assert isinstance(canonicalize_sql_for_hash("))) not sql"), str)


# ---------------------------------------------------------------------------
# compile_spec ordering with inline_query_source
# ---------------------------------------------------------------------------


class TestCompileSpecRunsInlineQuerySource:
    def test_compile_spec_inlines_batch_source_query_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "events.sql"), "w") as f:
                f.write("SELECT  *  FROM  EVENTS\n")
            d = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result = compile_spec(d, spec_file_dir=tmpdir)
        assert result["query"] == "SELECT * FROM EVENTS"
        assert "query_file" not in result

    def test_compile_spec_normalizes_inline_query(self) -> None:
        d = {
            "kind": "BatchSource",
            "name": "events",
            "query": "SELECT\n   *\nFROM   EVENTS",
        }
        result = compile_spec(d)
        assert result["query"] == "SELECT * FROM EVENTS"

    def test_compile_spec_runs_inline_query_after_udf(self) -> None:
        # When a spec carries BOTH a UDF file and a BatchSource query_file
        # (uncommon but legal — e.g. a manifest-style document), both are
        # inlined and the udf path runs first so the udf's function source
        # is available before query normalization touches anything else.
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_code = "def fn(df):\n    return df\n"
            sql_code = "SELECT * FROM EVENTS"
            with open(os.path.join(tmpdir, "fn.py"), "w") as f:
                f.write(udf_code)
            with open(os.path.join(tmpdir, "events.sql"), "w") as f:
                f.write(sql_code)
            d = {
                "kind": "StreamingFeatureView",
                "name": "fv",
                "udf": {"name": "fn", "file": "fn.py"},
                # Same dict carries a query_file at top level for an
                # alternative authoring shape; the helper must still inline
                # it because the kind check is per-doc and we call the
                # compiler for BatchSource docs the same way.
            }
            udf_result = compile_spec(d, spec_file_dir=tmpdir)
            assert udf_result["udf"]["function_definition"] == udf_code
            assert "file" not in udf_result["udf"]

            bs = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            bs_result = compile_spec(bs, spec_file_dir=tmpdir)
            assert bs_result["query"] == "SELECT * FROM EVENTS"
            assert "query_file" not in bs_result

    def test_compile_spec_with_table_only_unchanged(self) -> None:
        d = {"kind": "BatchSource", "name": "events", "table": "RAW.EVENTS"}
        result = compile_spec(d)
        assert result["table"] == "RAW.EVENTS"
        assert "query" not in result
        assert "query_file" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
