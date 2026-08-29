"""Tests for Jinja2 templating of companion sidecar files.

The compiler reads two kinds of sidecar files from disk and inlines
their bytes into the spec dict before the loader hands the result to
pydantic validation:

* ``UDF`` body — referenced by ``<NAME>.yaml`` via ``udf.file: <NAME>.py``
  (:func:`compiler.inline_udf_source`).
* ``BatchSource.query`` body — referenced by ``<NAME>.yaml`` via
  ``query_file: <NAME>.sql`` (:func:`compiler.inline_query_source`).

These tests exercise hypotheses H3, H4, and the H7 extension from
``plans/TEMPLATING_HYPOTHESES.md`` — both sidecar kinds MUST be
Jinja2-rendered using the same merged template variables as the parent
YAML, with the same ``StrictUndefined`` semantics so missing variables
fail loudly naming the sidecar file.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from snowflake.ml.feature_store.decl.compiler import (
    compile_spec,
    inline_query_source,
    inline_udf_source,
)
from snowflake.ml.feature_store.decl.errors import SpecLoadError
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# inline_query_source — sidecar templating (H4)
# ---------------------------------------------------------------------------


class TestSqlSidecarTemplating:
    """``inline_query_source`` MUST accept a ``template_vars`` kwarg and
    render any Jinja2 syntax in the referenced ``.sql`` body before the
    whitespace-normalisation pass.  The pre-existing whitespace contract
    (`normalize_sql_whitespace`) is applied AFTER rendering so the
    structural fingerprint hashes the post-render text.
    """

    _SQL_BODY_WITH_TEMPLATE = "SELECT USER_ID, EVENT_TS, METRIC_VAL\n" "FROM RAW_EVENTS\n" "LIMIT {{ row_limit }}\n"

    def test_sql_sidecar_renders_template_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(self._SQL_BODY_WITH_TEMPLATE)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result = inline_query_source(data, tmpdir, template_vars={"row_limit": 42})

        # Sidecar rendered: the integer 42 appears in the inlined query.
        assert "LIMIT 42" in result["query"], result["query"]
        assert "{{" not in result["query"]
        assert "query_file" not in result
        # Whitespace normalised after rendering.
        assert "\n" not in result["query"]

    def test_sql_sidecar_undefined_variable_raises_with_filepath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(self._SQL_BODY_WITH_TEMPLATE)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            # No row_limit in template_vars → StrictUndefined.
            with pytest.raises(SpecLoadError) as excinfo:
                inline_query_source(data, tmpdir, template_vars={})

        msg = str(excinfo.value)
        assert "row_limit" in msg
        assert "events.sql" in msg

    def test_sql_sidecar_without_jinja_unchanged_by_template_vars(self) -> None:
        sql_body = "SELECT * FROM RAW_EVENTS\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result = inline_query_source(data, tmpdir, template_vars={"row_limit": 42})

        # No Jinja in the file → result is the (whitespace-normalised) body,
        # regardless of what template_vars contains.
        assert result["query"] == "SELECT * FROM RAW_EVENTS"

    def test_sql_sidecar_none_template_vars_treats_as_plain_passthrough(self) -> None:
        # Symmetric with current ``process_file`` behaviour: a SQL body
        # without Jinja markers MUST inline cleanly when no template_vars
        # are supplied. A SQL body WITH Jinja markers but no template_vars
        # MUST raise (SpecLoadError) so the operator notices the missing
        # config rather than seeing silently-undefined output.
        sql_body = "SELECT * FROM RAW_EVENTS\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result = inline_query_source(data, tmpdir, template_vars=None)

        assert result["query"] == "SELECT * FROM RAW_EVENTS"

    def test_sql_sidecar_with_jinja_but_no_template_vars_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(self._SQL_BODY_WITH_TEMPLATE)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            with pytest.raises(SpecLoadError) as excinfo:
                inline_query_source(data, tmpdir, template_vars=None)

        msg = str(excinfo.value)
        # The same "Undefined variables" error surfaced for top-level
        # spec files MUST surface for sidecars too, so operators get a
        # consistent diagnostic regardless of which file the placeholder
        # lives in.
        assert "row_limit" in msg
        assert "events.sql" in msg

    def test_sql_sidecar_rendering_happens_before_whitespace_normalization(self) -> None:
        # The renderer produces ``LIMIT 99``; the whitespace normaliser
        # then collapses indentation in the surrounding lines. If the
        # render pass ran AFTER normalisation the surviving ``\n`` in
        # the template body would have been collapsed to a single space
        # already and the rendered string would still need a second
        # normalisation pass — wire-order is what we pin here.
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(self._SQL_BODY_WITH_TEMPLATE)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result = inline_query_source(data, tmpdir, template_vars={"row_limit": 99})

        # No newlines or runs of internal whitespace survive.
        assert "  " not in result["query"]
        assert "\n" not in result["query"]
        assert "LIMIT 99" in result["query"]


# ---------------------------------------------------------------------------
# inline_udf_source — sidecar templating (H3)
# ---------------------------------------------------------------------------


class TestUdfSidecarTemplating:
    """``inline_udf_source`` MUST accept a ``template_vars`` kwarg and
    render any Jinja2 syntax in the referenced ``.py`` body before
    inlining it as ``udf.function_definition``.
    """

    _UDF_BODY_WITH_TEMPLATE = (
        "def compute(df):\n"
        '    """UDF compiled for {{ env_suffix }}."""\n'
        "    threshold = {{ threshold }}\n"
        "    return df[df['val'] > threshold]\n"
    )

    def test_udf_sidecar_renders_template_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_path = os.path.join(tmpdir, "compute.py")
            with open(udf_path, "w") as f:
                f.write(self._UDF_BODY_WITH_TEMPLATE)

            data = {"udf": {"name": "compute", "file": "compute.py"}}
            result = inline_udf_source(
                data,
                tmpdir,
                template_vars={"env_suffix": "_DEV", "threshold": 7},
            )

        rendered = result["udf"]["function_definition"]
        assert "UDF compiled for _DEV" in rendered
        assert "threshold = 7" in rendered
        assert "{{" not in rendered
        assert "file" not in result["udf"]

    def test_udf_sidecar_undefined_variable_raises_with_filepath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_path = os.path.join(tmpdir, "compute.py")
            with open(udf_path, "w") as f:
                f.write(self._UDF_BODY_WITH_TEMPLATE)

            data = {"udf": {"name": "compute", "file": "compute.py"}}
            with pytest.raises(SpecLoadError) as excinfo:
                inline_udf_source(
                    data,
                    tmpdir,
                    template_vars={"env_suffix": "_DEV"},  # missing threshold
                )

        msg = str(excinfo.value)
        assert "threshold" in msg
        assert "compute.py" in msg

    def test_udf_sidecar_without_jinja_unchanged_by_template_vars(self) -> None:
        udf_body = "def compute(df):\n    return df * 2\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_path = os.path.join(tmpdir, "compute.py")
            with open(udf_path, "w") as f:
                f.write(udf_body)

            data = {"udf": {"name": "compute", "file": "compute.py"}}
            result = inline_udf_source(
                data,
                tmpdir,
                template_vars={"unused": "value"},
            )

        # Body verbatim — no normalisation pass exists for .py sidecars.
        assert result["udf"]["function_definition"] == udf_body

    def test_udf_sidecar_with_jinja_but_no_template_vars_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_path = os.path.join(tmpdir, "compute.py")
            with open(udf_path, "w") as f:
                f.write(self._UDF_BODY_WITH_TEMPLATE)

            data = {"udf": {"name": "compute", "file": "compute.py"}}
            with pytest.raises(SpecLoadError) as excinfo:
                inline_udf_source(data, tmpdir, template_vars=None)

        msg = str(excinfo.value)
        assert "env_suffix" in msg or "threshold" in msg
        assert "compute.py" in msg


# ---------------------------------------------------------------------------
# compile_spec — threads template_vars to both sidecar inliners
# ---------------------------------------------------------------------------


class TestCompileSpecPassesTemplateVars:
    """``compile_spec`` MUST accept a ``template_vars`` kwarg and thread
    it through to both :func:`inline_udf_source` and
    :func:`inline_query_source`.  Callers in :mod:`decl.loader` rely on
    this so the on-disk sidecar text matches the rendered top-level
    YAML.
    """

    def test_compile_spec_threads_template_vars_to_inline_query(self) -> None:
        sql_body = "SELECT * FROM RAW_EVENTS LIMIT {{ row_limit }}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result = compile_spec(data, spec_file_dir=tmpdir, template_vars={"row_limit": 25})

        assert "LIMIT 25" in result["query"]
        assert "query_file" not in result

    def test_compile_spec_threads_template_vars_to_inline_udf(self) -> None:
        udf_body = "def fn(df):\n    return df.head({{ row_limit }})\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            udf_path = os.path.join(tmpdir, "fn.py")
            with open(udf_path, "w") as f:
                f.write(udf_body)

            data = {
                "kind": "StreamingFeatureView",
                "name": "fv",
                "udf": {"name": "fn", "file": "fn.py"},
            }
            result = compile_spec(data, spec_file_dir=tmpdir, template_vars={"row_limit": 33})

        rendered = result["udf"]["function_definition"]
        assert "df.head(33)" in rendered
        assert "file" not in result["udf"]

    def test_compile_spec_with_no_template_vars_preserves_existing_behaviour(self) -> None:
        # Backward-compat: callers that pass no template_vars (or omit
        # the kwarg entirely) must continue to inline plain sidecars
        # exactly as before — the template_vars kwarg is purely additive.
        sql_body = "SELECT * FROM EVENTS\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = os.path.join(tmpdir, "events.sql")
            with open(sql_path, "w") as f:
                f.write(sql_body)

            data = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result_default = compile_spec(data, spec_file_dir=tmpdir)
            assert result_default["query"] == "SELECT * FROM EVENTS"

            # Calling again with an explicit template_vars=None matches the
            # default kwarg behaviour exactly.
            data2 = {
                "kind": "BatchSource",
                "name": "events",
                "query_file": "events.sql",
            }
            result_explicit_none = compile_spec(data2, spec_file_dir=tmpdir, template_vars=None)
            assert result_explicit_none["query"] == "SELECT * FROM EVENTS"


if __name__ == "__main__":
    pytest_driver.main()
