"""Unified ``compute_fn`` / ``transformation_fn`` validation.

This module is the single source of truth for the policy applied to user
functions registered on a streaming feature view (``StreamConfig``) or a
realtime feature view (``RealtimeConfig``). The policy is split into two
layers, intentionally complementary:

- :func:`validate_compute_fn_source` — static AST checks on the dedented
  source text. Enforces that the source is a single top-level
  ``def``-style function, with no nested ``def`` / ``async def`` / lambda,
  no calls to forbidden builtins, and only imports from
  :data:`ALLOWED_MODULES`.
- :func:`validate_compute_fn_callable` — closure-vars / free-name check on
  the live callable. Enforces that every free name resolves to a key in
  :data:`ALLOWED_NAMESPACE_KEYS` (or ``__builtins__``), with no closure
  references to enclosing scopes.

Each error message starts with the caller-supplied ``kind`` label
(``streaming feature view`` or ``realtime feature view``) so the user gets
a domain-appropriate prefix.

This is a deterministic guardrail, not a security sandbox — obfuscated
bypasses (``getattr(builtins, "__import__")``, etc.) are out of scope.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable, Optional

# Top-level packages permitted in import statements inside compute_fn.
ALLOWED_MODULES: frozenset[str] = frozenset({"numpy", "pandas", "re", "copy"})

# Names that may appear as free globals inside compute_fn. Mirrors the
# runtime namespace populated by RTFV reconstruction so the same source
# code runs identically at author time and at warehouse-side execution.
ALLOWED_NAMESPACE_KEYS: frozenset[str] = frozenset({"pd", "pandas", "np", "numpy", "re", "copy"})

# Builtins whose call is rejected by the AST layer. Union of the two
# pre-existing per-subsystem sets — the strict / RTFV-superset policy.
FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
    }
)


def validate_compute_fn_source(source: str, *, kind: str) -> None:
    """Run the AST layer of compute_fn validation against *source*.

    Enforces, in this order:

    1. *source* is valid Python.
    2. *source* contains exactly one top-level ``ast.FunctionDef`` (no
       multi-statement modules, no decorators-only modules, no top-level
       ``async def``).
    3. No nested ``ast.FunctionDef`` / ``ast.AsyncFunctionDef`` and no
       ``ast.Lambda`` anywhere inside the body.
    4. No ``ast.Call`` whose target name (Name or Attribute) is in
       :data:`FORBIDDEN_BUILTINS`.
    5. Every ``ast.Import`` / ``ast.ImportFrom`` references a top-level
       package in :data:`ALLOWED_MODULES`.

    Args:
        source: Plain-text Python source for the user function (will be
            dedented internally).
        kind: Domain prefix prepended to every error message; one of
            ``"streaming feature view"`` / ``"realtime feature view"``.

    Raises:
        ValueError: If any rule above is violated. The error message
            names the offending construct and its line number.
    """
    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
    except SyntaxError as e:
        raise ValueError(f"{kind}: compute_fn source is not valid Python: {e.msg} (line {e.lineno}).") from e

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(
            f"{kind}: compute_fn source must contain exactly one top-level "
            "function definition. Decorators-only modules, multi-statement "
            "modules, and async function definitions at the top level are "
            "not supported."
        )
    outer_fn = tree.body[0]

    for node in ast.walk(outer_fn):
        if node is outer_fn:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(
                f"{kind}: compute_fn must not contain nested function "
                f"definitions; found {node.name!r} at line {node.lineno}. "
                f"Inline the logic or import helpers from one of the allowed "
                f"runtime namespace modules: {sorted(ALLOWED_NAMESPACE_KEYS)}."
            )
        if isinstance(node, ast.Lambda):
            raise ValueError(
                f"{kind}: compute_fn must not contain lambda expressions; " f"found one at line {node.lineno}."
            )
        if isinstance(node, ast.Call):
            func = node.func
            fname: Optional[str]
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            else:
                fname = None
            if fname is not None and fname in FORBIDDEN_BUILTINS:
                raise ValueError(
                    f"{kind}: compute_fn must not call {fname!r}; found call "
                    f"at line {node.lineno}. Forbidden builtins: "
                    f"{sorted(FORBIDDEN_BUILTINS)}."
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in ALLOWED_MODULES:
                    raise ValueError(
                        f"{kind}: import {alias.name!r} is not allowed in "
                        f"compute_fn. Allowed modules: {sorted(ALLOWED_MODULES)}."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module not in ALLOWED_MODULES:
                    raise ValueError(
                        f"{kind}: import from {node.module!r} is not allowed "
                        f"in compute_fn. Allowed modules: {sorted(ALLOWED_MODULES)}."
                    )


def validate_compute_fn_callable(fn: Callable[..., Any], *, kind: str) -> None:
    """Run the closure-vars layer of compute_fn validation against *fn*.

    Uses :func:`inspect.getclosurevars` to inspect the live callable. Names
    bound by in-body imports and attribute lookups are filtered out before
    the namespace check (see :func:`_collect_ast_local_names`) so
    ``import pandas`` and ``pd.DataFrame`` do not trip the check.

    Three rules are enforced:

    - ``cvars.nonlocals`` is empty: no closure references to an enclosing
      scope.
    - ``cvars.unbound`` (after filtering) is empty: every free name
      resolves somewhere.
    - ``cvars.globals`` (after filtering) live inside
      :data:`ALLOWED_NAMESPACE_KEYS` ∪ ``{"__builtins__"}``.

    Args:
        fn: The user-supplied callable.
        kind: Domain prefix prepended to every error message.

    Raises:
        ValueError: If *fn* references closures, unbound names, or globals
            outside the runtime namespace. The message names each offender.
    """
    try:
        cvars = inspect.getclosurevars(fn)
    except TypeError as e:
        raise ValueError(f"{kind}: compute_fn closure variables are not inspectable: {e}") from e

    if cvars.nonlocals:
        offenders = sorted(cvars.nonlocals.keys())
        raise ValueError(
            f"{kind}: compute_fn must not reference enclosing-scope variables "
            f"(closures); offending names: {offenders}. Inline the values or "
            f"import from one of the allowed runtime namespace modules: "
            f"{sorted(ALLOWED_NAMESPACE_KEYS)}."
        )

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = ""
    import_names, attr_names = _collect_ast_local_names(source)
    filterable = import_names | attr_names

    effective_unbound = cvars.unbound - filterable
    if effective_unbound:
        offenders = sorted(effective_unbound)
        raise ValueError(
            f"{kind}: compute_fn references names that resolve nowhere: "
            f"{offenders}. Check for typos, or import the symbol from one of "
            f"the allowed runtime namespace modules: {sorted(ALLOWED_NAMESPACE_KEYS)}."
        )

    allowed_globals = set(ALLOWED_NAMESPACE_KEYS) | {"__builtins__"}
    effective_globals = set(cvars.globals.keys()) - filterable
    extra_globals = sorted(effective_globals - allowed_globals)
    if extra_globals:
        raise ValueError(
            f"{kind}: compute_fn references symbols outside the allowed "
            f"runtime namespace: {extra_globals}. Allowed namespace keys: "
            f"{sorted(ALLOWED_NAMESPACE_KEYS)}. Inline the values or import "
            f"the symbol from one of the allowed modules."
        )


def _collect_ast_local_names(source: str) -> tuple[set[str], set[str]]:
    """Return ``(import-name set, attribute-name set)`` for free-name filtering.

    :func:`inspect.getclosurevars` reports every name read from
    ``co_names`` as either resolved or unbound. That set includes module
    names from ``import x`` (``IMPORT_NAME`` reads them) and attribute
    names from ``obj.attr`` (``LOAD_ATTR`` reads them). Neither class is
    a real free name, so callers strip them before applying the namespace
    check.

    Args:
        source: Dedented source of the user function. An empty / unparsable
            string yields two empty sets (the closure check still runs).

    Returns:
        Tuple ``(imported_module_names, attribute_names)``.
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return set(), set()
    import_names: set[str] = set()
    attr_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                import_names.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_names.add(node.module.split(".")[0])
            for alias in node.names:
                if alias.name != "*":
                    import_names.add(alias.name)
        elif isinstance(node, ast.Attribute):
            attr_names.add(node.attr)
    return import_names, attr_names
