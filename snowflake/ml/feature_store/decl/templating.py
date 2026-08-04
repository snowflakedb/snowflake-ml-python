"""Jinja2 templating utilities for declarative feature store spec files.

Handles template detection, config loading, and rendering for spec files
that contain Jinja2 placeholder syntax (``{{ }}`` or ``{% %}``).

All functions are self-contained — no imports from snowflake.ml.feature_store.spec,
snowflake.snowpark, or snowflake.connector.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import jinja2
import jinja2.meta
import yaml

from snowflake.ml.feature_store.decl.errors import SpecLoadError

# Pattern matching Jinja2 template markers: {{ ... }} or {% ... %}
_JINJA2_PATTERN = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL)


def is_template(content: str) -> bool:
    """Return ``True`` if *content* contains Jinja2 template markers.

    Detects ``{{ variable }}`` and ``{% block %}`` syntax.

    Args:
        content: The string to inspect.

    Returns:
        ``True`` if Jinja2 syntax is found, ``False`` otherwise.
    """
    return bool(_JINJA2_PATTERN.search(content))


def load_config(config_source: str | None) -> dict[str, Any]:
    """Load template variables from a config source.

    Supported sources:

    - ``None`` → empty dict
    - ``"env"`` → ``dict(os.environ)``
    - A path to a YAML file → ``yaml.safe_load()``
    - An inline JSON string → ``json.loads()``

    Args:
        config_source: Specifies where to load config from.

    Returns:
        A dict of template variables.

    Raises:
        ValueError: If the source is not a file, ``"env"``, or valid JSON.
    """
    if config_source is None:
        return {}

    if config_source == "env":
        return dict(os.environ)

    config_path = os.path.expanduser(config_source)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping, got {type(data).__name__}")
        return data

    try:
        data = json.loads(config_source)
        if isinstance(data, dict):
            return data
        raise ValueError(f"Inline JSON config must be an object, got {type(data).__name__}")
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not load config: '{config_source}' is not a file, 'env', or valid JSON")


def render_template(content: str, variables: dict[str, Any]) -> str:
    """Render a Jinja2 template string with the given variables.

    Uses ``jinja2.StrictUndefined`` so that any unresolved variable raises
    a ``SpecLoadError`` rather than silently producing an empty string.

    Args:
        content: A Jinja2 template string.
        variables: Template variables to substitute.

    Returns:
        The rendered string.

    Raises:
        SpecLoadError: If any template variable is undefined.
    """
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.from_string(content)
    try:
        return template.render(variables)
    except jinja2.UndefinedError as exc:
        raise SpecLoadError(f"Template rendering failed: {exc}") from exc


def detect_and_render(content: str, config_source: str | None, filepath: str) -> str:
    """Render *content* as a Jinja2 template if it contains template syntax.

    If *content* is not a template, it is returned unchanged regardless of
    whether a *config_source* is provided.

    If *content* is a template but *config_source* is ``None``, a
    ``SpecLoadError`` is raised listing the undefined variables found and
    the *filepath* for context.

    Args:
        content: File content that may or may not contain Jinja2 syntax.
        config_source: Config source string passed to :func:`load_config`, or
            ``None`` if no config was provided.
        filepath: Path to the file being processed (used in error messages).

    Returns:
        The original content if it is not a template, or the rendered string.

    Raises:
        SpecLoadError: If the content is a template but no config was provided,
            or if rendering fails due to undefined variables.
    """
    if not is_template(content):
        return content

    if config_source is None:
        # Identify undefined variables for a helpful error message
        env = jinja2.Environment()
        ast = env.parse(content)
        undefined_vars = sorted(jinja2.meta.find_undeclared_variables(ast))
        var_list = ", ".join(undefined_vars)
        raise SpecLoadError(
            f"File '{filepath}' contains Jinja2 template variables but no "
            f"--config was provided. Undefined variables: {var_list}"
        )

    variables = load_config(config_source)
    try:
        return render_template(content, variables)
    except SpecLoadError as exc:
        # render_template raises ``Template rendering failed: '<var>' is
        # undefined`` without the source filepath; re-raise here so the
        # operator-facing message names the offending file (matters most
        # for companion sidecars where a missing variable would otherwise
        # surface a one-line generic error with no file context).
        raise SpecLoadError(f"File '{filepath}': {exc}") from exc
