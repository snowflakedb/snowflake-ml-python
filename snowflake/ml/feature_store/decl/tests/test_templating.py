"""Tests for decl/templating.py — is_template, load_config, render_template, detect_and_render."""

import os
import tempfile

import pytest

from snowflake.ml.feature_store.decl.errors import SpecLoadError
from snowflake.ml.feature_store.decl.templating import (
    detect_and_render,
    is_template,
    load_config,
    render_template,
)

# ---------------------------------------------------------------------------
# is_template
# ---------------------------------------------------------------------------


class TestIsTemplate:
    def test_double_brace_variable(self) -> None:
        assert is_template("hello {{ name }}") is True

    def test_block_tag(self) -> None:
        assert is_template("{% if x %}yes{% endif %}") is True

    def test_plain_string(self) -> None:
        assert is_template("hello world") is False

    def test_empty_string(self) -> None:
        assert is_template("") is False

    def test_partial_brace(self) -> None:
        assert is_template("{not_a_template}") is False

    def test_multiline_with_template(self) -> None:
        content = "kind: Entity\nname: {{ entity_name }}\n"
        assert is_template(content) is True

    def test_multiline_no_template(self) -> None:
        content = "kind: Entity\nname: customer\n"
        assert is_template(content) is False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_none_returns_empty_dict(self) -> None:
        result = load_config(None)
        assert result == {}

    def test_env_returns_environ(self) -> None:
        result = load_config("env")
        assert isinstance(result, dict)
        # os.environ is non-empty
        assert len(result) > 0

    def test_env_contains_path(self) -> None:
        result = load_config("env")
        assert "PATH" in result or "HOME" in result or len(result) > 0

    def test_inline_json_object(self) -> None:
        result = load_config('{"app": "myapp", "env": "prod"}')
        assert result == {"app": "myapp", "env": "prod"}

    def test_yaml_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("database: PROD\nschema: PUBLIC\n")
            path = f.name
        try:
            result = load_config(path)
            assert result["database"] == "PROD"
            assert result["schema"] == "PUBLIC"
        finally:
            os.unlink(path)

    def test_nonexistent_path_treated_as_json_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            load_config("/nonexistent/path/config.yaml")

    def test_inline_json_array_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            load_config("[1, 2, 3]")


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_basic_variable(self) -> None:
        result = render_template("hello {{ name }}", {"name": "world"})
        assert result == "hello world"

    def test_multiple_variables(self) -> None:
        result = render_template(
            "{{ database }}.{{ schema }}",
            {"database": "PROD", "schema": "PUBLIC"},
        )
        assert result == "PROD.PUBLIC"

    def test_undefined_variable_raises_spec_load_error(self) -> None:
        import pytest

        with pytest.raises(SpecLoadError):
            render_template("hello {{ undefined_var }}", {})

    def test_conditional_block(self) -> None:
        result = render_template(
            "{% if version is defined %}v={{ version }}{% endif %}",
            {"version": "v1"},
        )
        assert "v=v1" in result

    def test_no_template_passthrough(self) -> None:
        result = render_template("plain text", {})
        assert result == "plain text"

    def test_default_filter(self) -> None:
        result = render_template("{{ x | default('fallback') }}", {})
        assert result == "fallback"

    def test_trailing_newline_preserved(self) -> None:
        result = render_template("line1\n", {})
        assert result.endswith("\n")


# ---------------------------------------------------------------------------
# detect_and_render
# ---------------------------------------------------------------------------


class TestDetectAndRender:
    def test_non_template_no_config_passthrough(self) -> None:
        content = "kind: Entity\nname: customer\n"
        result = detect_and_render(content, None, "/some/file.yaml")
        assert result == content

    def test_non_template_with_config_passthrough(self) -> None:
        content = "kind: Entity\nname: customer\n"
        result = detect_and_render(content, '{"x": 1}', "/some/file.yaml")
        assert result == content

    def test_template_with_config_rendered(self) -> None:
        content = "name: {{ app_name }}"
        result = detect_and_render(content, '{"app_name": "myapp"}', "/some/file.yaml")
        assert result == "name: myapp"

    def test_template_without_config_raises_spec_load_error(self) -> None:
        import pytest

        content = "name: {{ app_name }}"
        with pytest.raises(SpecLoadError, match="app_name"):
            detect_and_render(content, None, "/some/file.yaml")

    def test_error_message_contains_filepath(self) -> None:
        import pytest

        content = "name: {{ missing_var }}"
        with pytest.raises(SpecLoadError) as exc_info:
            detect_and_render(content, None, "/some/file.yaml")
        assert "/some/file.yaml" in str(exc_info.value)

    def test_error_message_lists_undefined_variables(self) -> None:
        import pytest

        content = "db: {{ database }}\nschema: {{ schema_name }}"
        with pytest.raises(SpecLoadError) as exc_info:
            detect_and_render(content, None, "/some/file.yaml")
        error_msg = str(exc_info.value)
        assert "database" in error_msg
        assert "schema_name" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
