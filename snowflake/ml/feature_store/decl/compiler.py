"""Compilation passes for declarative feature store spec dicts.

Transforms authoring-format spec dicts (human-friendly types and durations)
into normalized output ready for validation and planning.

All functions are self-contained — no imports from snowflake.ml.feature_store.spec,
snowflake.snowpark, or snowflake.connector.  String duration parsing is
delegated to the shared, stdlib-only
:mod:`snowflake.ml.feature_store.interval_utils` module so the imperative
and declarative paths agree on accepted units and the ``"lifetime"``
sentinel.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from snowflake.ml.feature_store.decl.enums import normalize_type
from snowflake.ml.feature_store.decl.templating import detect_and_render
from snowflake.ml.feature_store.interval_utils import interval_to_seconds

# Whitespace normalization regex shared across SQL inlining and DT-text
# round-trip helpers (state.py, exporter.py). Defined locally — the
# imperative library's _DT_OR_VIEW_QUERY_PATTERN is intentionally NOT
# imported per the wheel-isolation rule (decl must not depend on
# feature_store.feature_store).
_SQL_WHITESPACE_RUN = re.compile(r"\s+")
# Trailing statement terminators (one or more ``;`` plus surrounding
# whitespace) Snowflake strips when storing the ``CREATE DYNAMIC TABLE
# … AS <body>`` body in ``SHOW DYNAMIC TABLES.text``.  Authors commonly
# end ``BatchSource.query_file`` sidecars with ``;`` (SQL statement
# convention), but the deployed-side DT body never carries one.  Both
# sides must converge after :func:`normalize_sql_whitespace` so the
# planner does not surface a phantom RECREATE_FV on a clean re-apply.
_SQL_TRAILING_TERMINATORS = re.compile(r"(?:\s*;\s*)+$")

# Duration keys that should be renamed to their _sec counterparts.
_DURATION_KEYS = frozenset({"feature_granularity", "target_lag", "window", "offset"})


def parse_duration_to_seconds(duration: str | int | float | None) -> int | None:
    """Parse a human-friendly duration value to integer seconds.

    The declarative authoring layer accepts ``None`` and pre-converted
    ``int`` / ``float`` values directly; every string is delegated to
    :func:`snowflake.ml.feature_store.interval_utils.interval_to_seconds`,
    which is the canonical duration parser shared with the imperative
    aggregation layer.

    Accepted shapes:

    - ``None`` → ``None``
    - ``int`` / ``float`` → ``int(value)`` (assumed already in seconds)
    - ``"5m"``, ``"30 minutes"``, ``"1h"``, ``"7d"``
    - ``"lifetime"`` → ``-1`` (sentinel matching the imperative layer)

    Fractional string values like ``"1.5m"`` are rejected by the shared
    ``interval_to_seconds`` parser; pass an ``int``/``float`` already
    in seconds if a sub-unit precision is required.

    Bare numeric strings (``"300"``) are intentionally rejected; the
    imperative ``parse_interval`` rejects them as well.  Authoring code
    should pass an ``int`` if the value is already in seconds.

    Args:
        duration: A duration value to parse.

    Returns:
        Integer seconds, or ``None`` if input is ``None``.
    """
    if duration is None:
        return None
    if isinstance(duration, (int, float)):
        return int(duration)
    return interval_to_seconds(str(duration))


def normalize_types(obj: Any) -> Any:
    """Recursively resolve type aliases to FSBaseType values.

    Walks all ``type`` fields in a dict/list tree and normalises the values
    via :func:`~snowflake.ml.feature_store.decl.enums.normalize_type`.

    Args:
        obj: A dict, list, or scalar to normalise.

    Returns:
        The same structure with all ``type`` field values resolved.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str):
                result[k] = normalize_type(v)
            else:
                result[k] = normalize_types(v)
        return result
    elif isinstance(obj, list):
        return [normalize_types(item) for item in obj]
    return obj


def normalize_durations(obj: Any) -> Any:
    """Recursively normalize human-friendly duration fields to integer seconds.

    Renames the following keys to their ``_sec`` equivalents and converts
    the value using :func:`parse_duration_to_seconds`:

    - ``feature_granularity`` → ``feature_granularity_sec``
    - ``target_lag`` → ``target_lag_sec``
    - ``window`` → ``window_sec``
    - ``offset`` → ``offset_sec``

    If the value is ``None`` the key is kept unchanged (no ``_sec`` variant).
    If the value cannot be parsed (e.g. ``"invalid"``) the key is kept unchanged.

    Special case: when the *current* dict carries
    ``kind: "StreamingFeatureView"`` or ``kind: "RealtimeFeatureView"``,
    both ``target_lag`` and ``target_lag_sec`` are dropped entirely
    (no rename, no copy-through).  Streaming and realtime FVs always
    run at 0 seconds target lag — the Snowflake runtime stamps
    ``target_lag_sec: 0`` onto the deployed SPECIFICATION regardless
    of the authored value — and the Pydantic-layer validator
    (``FeatureView._reject_target_lag_on_stream_or_realtime``) rejects
    any authored value on the Python / loader path.  Stripping here is
    defence-in-depth: a YAML carrying the runtime-stamped key (e.g. a
    legacy export from before the exporter fix) is silently cleaned up
    on the wire path so the rest of the compiler / planner / executor
    pipeline never sees the offending key.

    Args:
        obj: A dict, list, or scalar to normalise.

    Returns:
        The same structure with duration fields renamed and converted
        (and ``target_lag`` / ``target_lag_sec`` stripped from any
        streaming / realtime FV dicts).
    """
    if isinstance(obj, dict):
        kind = obj.get("kind", "") if isinstance(obj.get("kind"), str) else ""
        strip_target_lag = "Streaming" in kind or "Realtime" in kind
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if strip_target_lag and k in ("target_lag", "target_lag_sec"):
                continue
            if k in _DURATION_KEYS:
                if v is None:
                    result[k] = v
                else:
                    try:
                        result[f"{k}_sec"] = parse_duration_to_seconds(v)
                    except ValueError:
                        result[k] = v
            else:
                result[k] = normalize_durations(v)
        return result
    elif isinstance(obj, list):
        return [normalize_durations(item) for item in obj]
    return obj


def inline_udf_source(
    data: dict[str, Any],
    spec_file_dir: str | None,
    *,
    template_vars: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Read an external UDF file and inline its source as ``function_definition``.

    If ``data["udf"]["file"]`` is set, reads the file relative to
    ``spec_file_dir`` and stores the source in ``data["udf"]["function_definition"]``.
    The ``file`` key is removed from the compiled output.

    When ``template_vars`` is supplied (or the UDF body contains Jinja2
    markers and the caller wants StrictUndefined-style failure semantics),
    the inlined source is run through
    :func:`snowflake.ml.feature_store.decl.templating.detect_and_render`
    *before* it is stored on ``udf.function_definition``.  This matches
    the rendering semantics already applied to top-level YAML / JSON
    specs in :func:`loader.process_file` so UDF bodies pick up the same
    merged manifest variables as the parent FV YAML.  A UDF body with
    Jinja markers but no ``template_vars`` (or a missing variable under
    StrictUndefined) surfaces a :class:`SpecLoadError` from
    :func:`detect_and_render` naming the ``.py`` filepath.

    If the file is not found, a warning is printed and the spec is returned
    unchanged.

    Args:
        data: A spec dict that may contain a ``udf`` sub-dict.
        spec_file_dir: Directory to resolve relative ``udf.file`` paths against.
        template_vars: Merged manifest template variables (per D5
            precedence ``defaults < configurations < runtime``).  ``None``
            means "no templating context"; a UDF body without Jinja markers
            still passes through unchanged, but a body WITH Jinja markers
            triggers the :class:`SpecLoadError` described above.

    Returns:
        The spec dict with UDF source inlined (mutated in place and returned).

    Raises:
        ValueError: If ``udf.file`` is an absolute path or resolves outside ``spec_file_dir``.
    """
    import os

    udf = data.get("udf")
    if not udf or not isinstance(udf, dict):
        return data

    udf_file = udf.get("file")
    if not udf_file:
        return data

    if spec_file_dir is None:
        return data

    if os.path.isabs(udf_file):
        raise ValueError(f"udf.file must be a relative path, got absolute path: {udf_file!r}")

    udf_path = os.path.join(spec_file_dir, udf_file)
    _spec_root = os.path.realpath(spec_file_dir) + os.sep
    if not os.path.realpath(udf_path).startswith(_spec_root):
        raise ValueError(f"udf.file path traversal detected: {udf_file!r} resolves outside spec directory")

    if not os.path.exists(udf_path):
        import warnings

        warnings.warn(f"UDF file not found: {udf_path}", stacklevel=2)
        return data

    with open(udf_path) as f:
        source = f.read()

    source = detect_and_render(
        source,
        json.dumps(template_vars) if template_vars else None,
        udf_path,
    )

    udf["function_definition"] = source
    del udf["file"]
    return data


def normalize_sql_whitespace(sql: str) -> str:
    """Normalize SQL whitespace and trailing statement terminators.

    Strips leading / trailing whitespace, collapses every internal run
    of whitespace (spaces, tabs, newlines) to a single space, and
    removes any trailing run of ``;`` separators (with surrounding
    whitespace).  The operation is idempotent:
    ``normalize(normalize(s)) == normalize(s)``.

    This is the contract that keeps ``BatchSource.query`` round-tripping
    stable across re-applies: when the imperative layer creates a
    ``DYNAMIC TABLE … AS <body>``, Snowflake records the submitted text
    in the ``SHOW DYNAMIC TABLES`` ``text`` column **without** the
    statement-terminating ``;``.  The decl recovery path (Phase 4)
    extracts ``<body>`` and applies this normalizer; the locally-
    authored / sidecared SQL is normalized at compile time.  Authors
    routinely end ``query_file`` sidecars with ``;`` (SQL convention),
    so the trailing-terminator strip is what makes both sides converge
    on identical strings — without it the planner emits a phantom
    ``RECREATE_FV`` for a SQL-backed BFV on a clean re-apply (see
    ``declarative_feature_store/BATCH_FV_BUG_BASH.md`` §6/§7/§8).

    Limitations:

    - **String-literal contents are NOT preserved**. A literal like
      ``'a   b'`` collapses to ``'a b'``. Authors who need exact
      whitespace inside quoted literals should use SQL functions like
      ``REPLACE`` or escape sequences instead.
    - **Comments are preserved as-is** only insofar as the regex does
      not strip them; ``-- foo`` and ``/* foo */`` survive but any
      whitespace inside them collapses.
    - **Embedded** ``;`` separators (i.e. multi-statement bodies) are
      preserved.  Only a *trailing* run of ``;`` is removed, mirroring
      how Snowflake stores the ``AS <body>`` portion.

    Args:
        sql: The SQL string to normalize.

    Returns:
        The whitespace-normalized SQL string with no trailing ``;``.
    """
    collapsed = _SQL_WHITESPACE_RUN.sub(" ", sql).strip()
    return _SQL_TRAILING_TERMINATORS.sub("", collapsed)


def inline_query_source(
    data: dict[str, Any],
    spec_file_dir: str | None,
    *,
    template_vars: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Inline a ``BatchSource.query_file`` sidecar and normalize ``query``.

    Mirrors :func:`inline_udf_source` for ``BatchSource`` SQL sidecars:

    * If the doc's ``kind`` is not ``BatchSource``, return unchanged.
    * If the doc carries ``query_file:``, read the sibling file
      (resolved relative to ``spec_file_dir``), Jinja-render it with
      ``template_vars`` when supplied, set ``query`` to the
      whitespace-normalized contents, and drop ``query_file``.
    * If the doc already carries an inline ``query:``, normalize it in
      place. This is what keeps the DT-text round-trip stable on
      re-apply (see :func:`normalize_sql_whitespace`).
    * Missing sidecar files raise :class:`SpecLoadError` naming both
      the source and the file. ``spec_file_dir=None`` returns the doc
      unchanged so the caller can defer error surfacing to validation.

    Jinja rendering runs *before* :func:`normalize_sql_whitespace` so
    the structural fingerprint hashes the post-render text — this keeps
    templated SQL with ``LIMIT {{ row_limit }}`` round-tripping cleanly
    on re-apply as long as ``row_limit`` is stable for the target.

    Args:
        data: A spec dict that may be a ``BatchSource``.
        spec_file_dir: Directory to resolve relative ``query_file``
            paths against. ``None`` skips file inlining (compiled-from-
            python paths that have no on-disk sidecar).
        template_vars: Merged manifest template variables (per D5
            precedence ``defaults < configurations < runtime``).  ``None``
            means "no templating context"; a SQL body without Jinja
            markers still passes through, but a body WITH Jinja markers
            raises :class:`SpecLoadError` so operators don't deploy
            unrendered SQL.

    Returns:
        The spec dict with ``query`` inlined and normalized
        (mutated in place and returned).

    Raises:
        SpecLoadError: If ``query_file`` is set but the file is missing,
            if the SQL body contains Jinja2 markers and no
            ``template_vars`` were supplied, or if a referenced variable
            is undefined under StrictUndefined semantics.
    """
    import os

    if data.get("kind") != "BatchSource":
        return data

    query_file = data.get("query_file")
    if isinstance(query_file, str) and query_file:
        if spec_file_dir is None:
            return data
        query_path = os.path.join(spec_file_dir, query_file)
        if not os.path.exists(query_path):
            from snowflake.ml.feature_store.decl.errors import SpecLoadError

            raise SpecLoadError(
                f"BatchSource '{data.get('name', '<unknown>')}' references "
                f"query_file '{query_file}' but the file was not found at "
                f"'{query_path}'."
            )
        with open(query_path) as f:
            source = f.read()
        source = detect_and_render(
            source,
            json.dumps(template_vars) if template_vars else None,
            query_path,
        )
        data["query"] = normalize_sql_whitespace(source)
        del data["query_file"]
        return data

    inline_query = data.get("query")
    if isinstance(inline_query, str) and inline_query:
        data["query"] = normalize_sql_whitespace(inline_query)

    return data


def compile_spec(
    data: dict[str, Any],
    spec_file_dir: str | None = None,
    *,
    template_vars: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compile an authoring-format spec dict into normalized output.

    Runs four normalization passes in order:

    1. :func:`normalize_types` — resolve type aliases to ``FSBaseType`` values.
    2. :func:`inline_udf_source` — inline external UDF ``.py`` files
       (Jinja-rendered when ``template_vars`` is supplied).
    3. :func:`inline_query_source` — inline ``BatchSource.query_file`` sidecars
       (Jinja-rendered when ``template_vars`` is supplied) and whitespace-
       normalize ``query`` values so the DT-text round-trip stays stable on
       re-apply.
    4. :func:`normalize_durations` — convert human-friendly duration strings to
       integer seconds and rename keys to ``_sec`` variants.

    Args:
        data: The raw authoring-format spec dict.
        spec_file_dir: Directory for resolving relative ``udf.file`` and
            ``query_file`` paths. May be ``None`` when no inlining is needed.
        template_vars: Merged manifest template variables (per D5
            precedence) threaded through to both sidecar inliners so the
            on-disk ``.py`` UDF bodies and ``.sql`` query bodies pick up
            the same context as the parent YAML.  ``None`` matches the
            pre-templating behaviour exactly — plain sidecars inline as
            before, Jinja-bearing sidecars raise.

    Returns:
        The compiled, normalized spec dict.
    """
    data = normalize_types(data)
    data = inline_udf_source(data, spec_file_dir, template_vars=template_vars)
    data = inline_query_source(data, spec_file_dir, template_vars=template_vars)
    data = normalize_durations(data)
    return data
