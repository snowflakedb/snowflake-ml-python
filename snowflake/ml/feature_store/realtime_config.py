"""RealtimeConfig — configuration for realtime feature views.

This module provides:

- ``RealtimeConfig``: User-facing dataclass for defining realtime feature view
  configuration (compute function, sources, output schema).
- ``_RTFV_RUNTIME_NAMESPACE``: The complete runtime namespace shipped to the
  ``compute_fn`` (``pd``/``pandas``, ``np``/``numpy``, ``re``, ``copy``). Used
  by the rehydration helper to build the exec namespace; the *keys* that drive
  validation live in :mod:`snowflake.ml.feature_store._compute_fn_validation`.
- ``_rehydrate_realtime_compute_fn``: ``exec``-based reconstruction of a
  ``compute_fn`` from its persisted source text, executed in a fresh copy of
  ``_RTFV_RUNTIME_NAMESPACE``. Used at both author time and reconstruction
  time so the same code path runs in both places.

``compute_fn`` validation runs through the shared
:mod:`snowflake.ml.feature_store._compute_fn_validation` policy so SFV and
RTFV registration enforce the exact same rules. This is a deterministic
guardrail, not a security sandbox — obfuscated bypasses
(``getattr(builtins, "__import__")``, etc.) are out of scope.
"""

from __future__ import annotations

import copy as _copy_module
import hashlib
import inspect
import linecache
import re as _re_module
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from snowflake.ml.feature_store._compute_fn_validation import (
    validate_compute_fn_callable,
    validate_compute_fn_source,
)
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import StructType

# ---------------------------------------------------------------------------
# Runtime namespace
# ---------------------------------------------------------------------------

# Complete runtime namespace for compute_fn. Every key in
# ``inspect.getclosurevars(compute_fn).globals`` must be one of these (plus
# ``__builtins__``). This set is a subset of the conda packages bundled in
# the warehouse Python env that runs ``map_in_pandas``, so the same source
# code runs identically in author-time exec, reconstruction-time exec, and
# warehouse-time ``map_in_pandas``. The validation allow-list itself lives in
# :mod:`snowflake.ml.feature_store._compute_fn_validation` as
# ``ALLOWED_NAMESPACE_KEYS``; this dict is the runtime construct only.
_RTFV_RUNTIME_NAMESPACE: dict[str, Any] = {
    "pd": pd,
    "pandas": pd,
    "np": np,
    "numpy": np,
    "re": _re_module,
    "copy": _copy_module,
}


# ---------------------------------------------------------------------------
# Rehydration (used by RealtimeConfig.__post_init__ and by reconstruction
# code in realtime_registration.py)
# ---------------------------------------------------------------------------


def _rehydrate_realtime_compute_fn(source: str, name: str) -> Callable[..., pd.DataFrame]:
    """Rebuild a ``compute_fn`` callable from its persisted source text.

    Builds a fresh copy of :data:`_RTFV_RUNTIME_NAMESPACE` (so callers cannot
    pollute each other's namespace), execs the dedented *source* in it, and
    looks up *name*. Used by both :meth:`RealtimeConfig.__post_init__` and the
    realtime reconstruction branch in ``FeatureStore._compose_feature_view``;
    the single code path guarantees bit-identical behavior across exec sites.

    Args:
        source: Dedented Python source text containing a top-level
            ``def <name>(...)`` definition.
        name: ``__name__`` of the function to extract from the executed module.

    Returns:
        The reconstructed callable.

    Raises:
        ValueError: If the source fails to parse, the named function is not
            defined after exec, or the resolved attribute is not callable.
    """
    namespace = dict(_RTFV_RUNTIME_NAMESPACE)
    dedented = textwrap.dedent(source)
    # Register the source in ``linecache`` under a pseudo-filename keyed by
    # (name, source hash) so ``inspect.getsource(fn)`` succeeds at every later
    # call site -- in particular the read-time path where
    # ``compose_rtfv_from_metadata`` constructs a new ``RealtimeConfig`` from
    # the persisted source and ``__post_init__`` re-validates by extracting
    # the source from the rehydrated callable. Without this, the exec'd
    # function has no on-disk source and ``inspect.getsource`` raises
    # ``OSError``. ``mtime=None`` instructs ``linecache.checkcache`` to treat
    # the entry as loader-backed and skip the file-system invalidation check.
    src_hash = hashlib.sha256(dedented.encode("utf-8")).hexdigest()[:12]
    pseudo_filename = f"<realtime-config:{name}:{src_hash}>"
    cached_lines = [line + "\n" for line in dedented.splitlines()] or [""]
    linecache.cache[pseudo_filename] = (
        len(dedented),
        None,
        cached_lines,
        pseudo_filename,
    )
    try:
        exec(compile(dedented, pseudo_filename, "exec"), namespace)
    except Exception as e:
        raise ValueError(
            f"realtime feature view: compute_fn source failed to execute during "
            f"reconstruction (function name {name!r}): {e}"
        ) from e
    fn = namespace.get(name)
    if fn is None:
        raise ValueError(
            f"realtime feature view: compute_fn source does not define a top-level " f"function named {name!r}."
        )
    if not callable(fn):
        raise ValueError(
            f"realtime feature view: name {name!r} resolved to a non-callable value of type "
            f"{type(fn).__name__} after compute_fn source was executed."
        )
    return fn  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# RealtimeConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RealtimeConfig:
    """Configuration for realtime feature views.

    A ``RealtimeConfig`` bundles a ``compute_fn`` (the request-time
    transformation), the list of inputs it receives (one or more upstream
    :class:`FeatureView` / :class:`FeatureViewSlice` references, optionally
    preceded by a single :class:`RequestSource` at position 0), and the
    output schema describing the columns ``compute_fn`` returns.

    The ``compute_fn`` source is validated at construction time. Validation
    fires up-front in this order so the most user-facing error wins:

    1. Author shape: must be callable, must be a named function (lambdas /
       callable classes / dynamically generated functions are rejected),
       source must be extractable via ``inspect.getsource``.
    2. Source layer (shared with SFV): single top-level ``def``, no nested
       ``def``/``async def``/``lambda``, no forbidden builtin calls, only
       imports from the allow-list. See
       :func:`snowflake.ml.feature_store._compute_fn_validation.validate_compute_fn_source`.
    3. Callable layer (shared with SFV): no closures, no unbound names,
       every free name resolves to ``ALLOWED_NAMESPACE_KEYS`` ∪
       ``{"__builtins__"}``. See
       :func:`snowflake.ml.feature_store._compute_fn_validation.validate_compute_fn_callable`.
    4. Source rules: at most one :class:`RequestSource` (at position 0 if
       present) plus one or more :class:`FeatureView` /
       :class:`FeatureViewSlice`. ``FeatureGroup`` is **not** a valid source.
    5. Output schema non-empty.
    6. Round-trip canary: source is exec'd in a fresh copy of the runtime
       namespace and the resulting callable replaces ``compute_fn``. The
       reconstructed callable has the same observable behavior as the
       original because the namespace is closed.
    7. Positional arity equals ``len(sources)``.

    Args:
        compute_fn: A **named** Python function that returns a
            ``pd.DataFrame``. Lambdas, callable classes, and interactively
            defined functions are rejected.
        sources: Inputs to ``compute_fn``. If a :class:`RequestSource` is
            provided, it must be at ``sources[0]`` and there can be at most
            one. Remaining entries must be :class:`FeatureView` or
            :class:`FeatureViewSlice` (and **not** :class:`FeatureGroup`),
            and at least one such upstream is required. ``compute_fn`` is
            invoked with one positional argument per source, in this order.
        output_schema: The Snowpark ``StructType`` describing the columns
            ``compute_fn`` returns. Must be non-empty.

    Raises:
        ValueError: If any validation rule above fails.

    Example::

        >>> import pandas as pd
        >>> from snowflake.snowpark.types import (
        ...     StructType, StructField, DoubleType, StringType,
        ... )
        >>> def risk_score(req, txn_features):
        ...     return pd.DataFrame({
        ...         "risk_score": req["amount"] / (txn_features["avg_amount"] + 1),
        ...         "risk_bucket": ["high"] * len(req),
        ...     })
        >>> rt = RealtimeConfig(
        ...     compute_fn=risk_score,
        ...     sources=[request_source, txn_fv],
        ...     output_schema=StructType([
        ...         StructField("risk_score", DoubleType()),
        ...         StructField("risk_bucket", StringType()),
        ...     ]),
        ... )
    """

    compute_fn: Callable[..., pd.DataFrame]
    sources: list[Any] = field(default_factory=list)
    output_schema: StructType = field(default_factory=lambda: StructType([]))

    def __post_init__(self) -> None:
        # Lazy imports to avoid circular dependency
        from snowflake.ml.feature_store.feature_view import (
            FeatureView,
            FeatureViewSlice,
        )

        fn = self.compute_fn

        # Step 1: author shape (callable + named function).
        if not callable(fn):
            raise ValueError("realtime feature view: compute_fn must be callable.")
        if not hasattr(fn, "__name__") or fn.__name__ == "<lambda>":
            raise ValueError(
                "realtime feature view: compute_fn must be a named function. Lambdas "
                "and callable class instances are not supported."
            )

        try:
            source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError) as e:
            raise ValueError(
                "realtime feature view: cannot extract source code from compute_fn. "
                "It must be a named function defined at module level. Interactively "
                f"defined or dynamically generated functions are not supported: {e}"
            ) from e

        # Steps 2-3: shared compute_fn policy (AST + closure-vars layers).
        validate_compute_fn_source(source, kind="realtime feature view")
        validate_compute_fn_callable(fn, kind="realtime feature view")

        # Step 4: source rules.
        sources_list = list(self.sources)
        if not sources_list:
            raise ValueError(
                "realtime feature view: sources must be non-empty. Provide at least one "
                "upstream FeatureView, optionally preceded by a single RequestSource at "
                "position 0."
            )
        # Defense in depth against FeatureGroup-as-source. Imported lazily for
        # circular-import reasons.
        from snowflake.ml.feature_store.feature_group import FeatureGroup

        has_request_source = isinstance(sources_list[0], RequestSource)
        upstream_sources = sources_list[1:] if has_request_source else sources_list
        upstream_offset = 1 if has_request_source else 0
        if not upstream_sources:
            raise ValueError(
                "realtime feature view: sources must include at least one upstream " "FeatureView or FeatureViewSlice."
            )
        for offset, src in enumerate(upstream_sources):
            i = offset + upstream_offset
            if isinstance(src, FeatureGroup):
                raise ValueError(
                    f"realtime feature view: sources[{i}] is a FeatureGroup, which is "
                    "not a valid RealtimeFeatureView source. Use the underlying "
                    "FeatureViews directly."
                )
            if isinstance(src, RequestSource):
                if has_request_source:
                    raise ValueError(
                        f"realtime feature view: sources contains more than one RequestSource "
                        f"(found at positions 0 and {i}). At most one RequestSource is allowed, "
                        "and it must be sources[0]."
                    )
                raise ValueError(
                    f"realtime feature view: sources[{i}] is a RequestSource. RequestSource "
                    "must be at sources[0] when provided."
                )
            if not isinstance(src, (FeatureView, FeatureViewSlice)):
                raise ValueError(
                    f"realtime feature view: sources[{i}] must be a FeatureView or "
                    f"FeatureViewSlice; got {type(src).__name__}."
                )

        # Step 5: output_schema non-empty.
        if not self.output_schema or not self.output_schema.fields:
            raise ValueError("realtime feature view: output_schema must be non-empty (at least one " "StructField).")

        # Step 6: round-trip exec — replace compute_fn with the version that
        # ran inside the runtime namespace. This guarantees the callable
        # observed by every later consumer (registration, reconstruction,
        # warehouse-side map_in_pandas) is the same one that the namespace
        # produces.
        rehydrated = _rehydrate_realtime_compute_fn(source, fn.__name__)

        # Step 7: positional arity matches len(sources).
        try:
            sig = inspect.signature(rehydrated)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"realtime feature view: cannot inspect signature of compute_fn after " f"reconstruction: {e}"
            ) from e
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
            raise ValueError(
                "realtime feature view: compute_fn must not use *args; each source maps "
                "to a single positional argument."
            )
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            raise ValueError(
                "realtime feature view: compute_fn must not use **kwargs; each source "
                "maps to a single positional argument."
            )
        if any(p.kind == inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values()):
            raise ValueError(
                "realtime feature view: compute_fn must not declare keyword-only "
                "arguments; each source maps to a single positional argument."
            )
        if len(positional_params) != len(sources_list):
            raise ValueError(
                f"realtime feature view: compute_fn has {len(positional_params)} "
                f"positional argument(s) but RealtimeConfig was given {len(sources_list)} "
                f"source(s). compute_fn receives one positional argument per source, in "
                f"sources[] order (RequestSource first if present)."
            )

        # Step 8: install the rehydrated callable in place of the original.
        # Frozen-dataclass workaround. The rehydrated callable was produced
        # via ``exec(...)`` so :func:`inspect.getsource` cannot find it on
        # disk — cache the dedented source under ``_compute_fn_source`` so
        # later consumers (registration metadata, ``get_function_source``)
        # round-trip without going back through ``inspect``.
        object.__setattr__(self, "compute_fn", rehydrated)
        object.__setattr__(self, "_compute_fn_source", source)

    # ------------------------------------------------------------------
    # Helpers (mirror StreamConfig)
    # ------------------------------------------------------------------

    def get_function_name(self) -> str:
        """Return the ``__name__`` of ``compute_fn``."""
        return self.compute_fn.__name__

    def get_function_source(self) -> str:
        """Return the dedented plain-text source code of ``compute_fn``.

        Cached during :meth:`__post_init__` because the live ``compute_fn``
        is the ``exec``-produced rehydrated version, which is not on disk.

        Returns:
            Dedented source string captured at construction time.
        """
        return self._compute_fn_source  # type: ignore[no-any-return,attr-defined]

    @property
    def request_source(self) -> Optional[RequestSource]:
        """The :class:`RequestSource` at ``sources[0]``, or ``None`` if not provided."""
        if not self.sources:
            return None
        first = self.sources[0]
        return first if isinstance(first, RequestSource) else None

    @property
    def feature_view_sources(self) -> list[Union[Any, Any]]:
        """Upstream :class:`FeatureView` / :class:`FeatureViewSlice` references."""
        return [s for s in self.sources if not isinstance(s, RequestSource)]
