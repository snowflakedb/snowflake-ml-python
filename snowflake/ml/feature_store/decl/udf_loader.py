"""Compile a UDF source string from a declarative spec into a named callable.

The :func:`compile_udf_callable` helper bridges the gap between the spec
payload's ``udf.function_definition`` (a plain Python source string inlined
by ``decl/compiler.py::inline_udf_source``) and a real callable that
satisfies snowml-core's ``StreamConfig`` validators in
``snowflake.ml.feature_store.stream_config``:

1. ``callable(fn)`` is True.
2. ``fn.__name__`` matches the requested name (lambdas rejected).
3. ``inspect.getsource(fn)`` returns the original source — required for
   the AST-walk import guard inside ``StreamConfig.__post_init__``.

The third requirement is the subtle one: a callable produced by ``exec``
into a fresh namespace is invisible to ``inspect.getsource`` because
``inspect`` looks the function up via ``__module__`` / ``__file__`` /
``linecache``.  We therefore (a) host the source in a synthetic
``ModuleType`` whose ``__file__`` is a unique sentinel path, and (b)
register the source lines with :mod:`linecache` against that path so
``inspect.getsource`` can read them back.

This module is pure Python with no Snowflake dependencies — it is
imported lazily from :mod:`snowflake.ml.feature_store.decl.imperative_executor`
when a streaming feature view is being built.
"""

from __future__ import annotations

import linecache
import textwrap
import types
import uuid
from typing import Any, Callable, cast

_MODULE_PREFIX = "snowflake.ml.feature_store.decl._udf_compiled_"


def compile_udf_callable(function_definition: str, name: str) -> Callable[..., Any]:
    """Compile a UDF source string into a named, ``getsource``-able callable.

    Args:
        function_definition: Plain-text Python source containing a
            module-level ``def {name}(...)`` (and any imports that source
            requires).  The same string the spec compiler inlines from
            ``udf.file`` into ``payload['udf']['function_definition']``.
        name: The function name to extract from the compiled namespace.
            Must match a top-level ``def`` in ``function_definition``.

    Returns:
        The compiled function object.  Its ``__module__`` points at a
        synthetic module registered in :data:`sys.modules`, its
        ``__qualname__`` matches ``name``, and ``inspect.getsource`` will
        recover the original source.

    Raises:
        ValueError: If ``function_definition`` contains no top-level
            ``def {name}``, or compilation / execution fails.
    """
    source = textwrap.dedent(function_definition)
    if not source.strip():
        raise ValueError("function_definition is empty.")

    fake_module_name = f"{_MODULE_PREFIX}{uuid.uuid4().hex}"
    fake_filename = f"<{fake_module_name}>"

    module = types.ModuleType(fake_module_name)
    module.__file__ = fake_filename

    # Register the source with linecache BEFORE exec so that any error
    # tracebacks (and inspect.getsource) can resolve fake_filename to the
    # original source lines.
    line_list = [line + "\n" for line in source.splitlines()]
    linecache.cache[fake_filename] = (
        len(source),
        None,
        line_list,
        fake_filename,
    )

    import sys

    sys.modules[fake_module_name] = module

    try:
        code = compile(source, fake_filename, "exec")
        exec(code, module.__dict__)  # noqa: S102 — declarative UDF execution surface
    except SyntaxError as e:
        raise ValueError(f"Failed to compile UDF source for '{name}': {e}") from e

    fn = module.__dict__.get(name)
    if fn is None:
        raise ValueError(
            f"UDF source does not define a function named '{name}'. "
            f"Found: {sorted(k for k in module.__dict__ if not k.startswith('_'))}"
        )
    if not callable(fn):
        raise ValueError(f"'{name}' in UDF source is not callable.")

    return cast(Callable[..., Any], fn)
