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
from typing import Any

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Where
from sqlparse.tokens import DML, Keyword, Literal, Name, Wildcard

from snowflake.ml._internal.utils import identifier
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

# Top-level set-operation keywords.  A query joined by any of these has no
# single projection list to compare, so the SELECT-list analysis bails to the
# predicate-strip proof instead (see :func:`extract_select_output_columns`).
_SQL_SET_OPERATIONS: frozenset[str] = frozenset({"UNION", "UNION ALL", "INTERSECT", "EXCEPT", "MINUS"})


def _decode_join_keys(raw: Any) -> list[str]:
    """Decode a join-key cell to column names (case preserved).

    Shared 3-tier contract: a list verbatim, else a JSON list (a JSON
    scalar is no keys), else a legacy comma-separated fallback.

    Args:
        raw: A ``JOIN_KEYS`` / ``allowed_values`` cell (list, JSON, or string).

    Returns:
        The join-key column names; ``[]`` when none parse.
    """
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return [v.strip() for v in text.strip("[]").split(",") if v.strip()]
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    return []


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
    template_vars: dict[str, Any] | None = None,
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
    - **Comments are NOT handled here.** This helper only touches
      whitespace, so a ``--`` line comment would swallow the rest of the
      query once its terminating newline collapses.  The compile-time
      caller (:func:`inline_query_source`) therefore runs
      :func:`strip_sql_comments` *before* this normaliser so the deployed
      SQL is comment-free and valid; the hash-basis canonicaliser
      (:func:`canonicalize_sql_for_hash`) strips comments too.
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


def strip_sql_comments(sql: str) -> str:
    """Remove SQL comments (``--`` line and ``/* ... */`` block) from ``sql``.

    Applied at compile time by :func:`inline_query_source` *before*
    :func:`normalize_sql_whitespace` collapses newlines.  Without this a
    ``--`` line comment loses its terminating newline when whitespace is
    collapsed and swallows the rest of the query onto a single line,
    producing invalid ``CREATE DYNAMIC TABLE … AS <body>`` DDL.

    The transform is defensive: any parser error returns the input
    unchanged so a malformed SQL body never breaks compilation.

    Args:
        sql: The SQL string to strip comments from.

    Returns:
        The SQL with comments removed (whitespace otherwise untouched;
        the caller normalises whitespace separately).  Empty / falsy
        input is returned unchanged.
    """
    if not sql:
        return sql
    try:
        # sqlparse.format is untyped (returns Any); coerce for mypy.
        return str(sqlparse.format(sql, strip_comments=True))
    except Exception:  # noqa: BLE001 — never break compilation on a parser hiccup
        return sql


def canonicalize_sql_for_hash(sql: str) -> str:
    """Canonicalize SQL into a stable **hash basis** (never the submitted DDL).

    Strips comments, upper-cases keywords, and collapses whitespace so a
    comment-only / keyword-case-only / formatting-only edit to a
    query-backed ``BatchSource`` does not change the structural hash —
    which would otherwise emit a spurious ``RECREATE_SOURCE`` /
    ``RECREATE_FV`` (and, for append-only BatchFVs, wipe the companion
    ``$SNAPSHOTS`` history).

    This is used *only* at hashing boundaries in
    :mod:`snowflake.ml.feature_store.decl.invariants`; the SQL actually
    submitted to the Dynamic Table keeps its authored keyword case /
    quoting verbatim.  Idempotent, and defensive: any parser error falls
    back to :func:`normalize_sql_whitespace` so hashing never crashes.

    Args:
        sql: The SQL string to canonicalize for hashing.

    Returns:
        The canonicalized SQL string; empty / falsy input is returned
        unchanged.
    """
    if not sql:
        return sql
    try:
        formatted = sqlparse.format(
            sql,
            strip_comments=True,
            keyword_case="upper",
            strip_whitespace=True,
        )
    except Exception:  # noqa: BLE001 — fall back to whitespace-only normalisation
        formatted = sql
    return normalize_sql_whitespace(formatted)


def _select_output_name(item: Identifier) -> str | None:
    """Resolve one ``SELECT`` projection item to its output column name.

    The output name is the item's explicit ``AS`` alias when present, else the
    bare column name (``t.USER_ID`` → ``USER_ID``).  The name is read from the
    raw ``Name`` / quoted-identifier token so a ``"quoted"`` identifier keeps
    its case and quoting, then resolved through
    :func:`snowflake.ml._internal.utils.identifier.resolve_identifier` — the
    same rule the applied-state recovery uses — so an unquoted authored name
    and the core-upper-cased name collapse to one canonical form.

    An unaliased expression (``a + b``, ``SUM(b)``) has no provable output
    name and returns ``None`` so the caller falls back to the predicate-strip
    proof rather than guessing.

    Args:
        item: A single ``sqlparse`` ``Identifier`` from the projection list.

    Returns:
        The canonical output column name, or ``None`` when it cannot be proven.
    """
    flat = item.flatten()
    names = [t for t in flat if t.ttype is Name or t.ttype is Literal.String.Symbol]
    if not names:
        return None
    if item.get_alias():
        raw = names[-1].value
    else:
        real = item.get_real_name()
        if not real:
            return None
        raw = next((n.value for n in reversed(names) if n.value.strip('"') == real), None)
        if raw is None:
            return None
    try:
        return identifier.resolve_identifier(raw)
    except ValueError:
        return None


def extract_select_output_columns(sql: str) -> list[str] | None:
    """Return the ordered output column names of a single ``SELECT``, or ``None``.

    Used by the append-only ``BatchFeatureView`` evolution classifier to decide
    whether a query-body edit changed the projected output schema (a drop /
    reorder / retype → destructive ``RECREATE_FV``) or only its predicates (a
    ``WHERE``-only rewrite → in-place ``EVOLVE_FV``).  Because
    :func:`spec_compiler.compile_to_spec` parks a query-backed source's SQL in
    ``spec.sources`` and leaves ``spec.features`` empty, the SELECT list is the
    only place the output schema is visible.

    Returns ``None`` (the caller then falls back to the predicate-strip proof)
    whenever the shape cannot be *proven*:

    * not a single statement, or not a ``SELECT`` (after an optional leading
      ``WITH`` common-table-expression block, which is skipped so a CTE-backed
      query parses like a plain one);
    * the outer query is joined by a set operation (``UNION`` / ``INTERSECT`` /
      ``EXCEPT`` / ``MINUS``) — a set operation nested *inside* a CTE definition
      does not count, since only the outer projection list matters;
    * the projection contains a ``*`` wildcard, a non-identifier item, or an
      unaliased expression (see :func:`_select_output_name`).

    Args:
        sql: The query body (already comment-stripped / whitespace-normalised
            at compile time, but this helper does not rely on that).

    Returns:
        The ordered list of canonical output column names, or ``None`` when the
        projection cannot be proven.
    """
    if not sql:
        return None
    try:
        statements = sqlparse.parse(sql)
    except Exception:  # noqa: BLE001 — never raise from a hashing / planning boundary
        return None
    if len(statements) != 1:
        return None
    toks = [t for t in statements[0].tokens if not t.is_whitespace]
    idx = 0
    # Skip a leading ``WITH <cte-defs>`` block so the outer SELECT is reached.
    if idx < len(toks) and toks[idx].ttype is Keyword.CTE:
        idx += 1
        if idx < len(toks) and isinstance(toks[idx], (Identifier, IdentifierList)):
            idx += 1
    if idx >= len(toks) or toks[idx].ttype is not DML or toks[idx].value.upper() != "SELECT":
        return None
    # Outer-query set-operation scan: only top-level keyword tokens count, so a
    # set operation buried inside a CTE definition does not disqualify a query
    # whose outer projection list is perfectly readable.
    for tok in toks[idx:]:
        if tok.ttype is Keyword and tok.value.upper() in _SQL_SET_OPERATIONS:
            return None
    idx += 1
    projection: list[Any] = []
    for tok in toks[idx:]:
        if tok.ttype is Keyword and tok.value.upper() == "FROM":
            break
        if tok.ttype is Keyword and tok.value.upper() in ("DISTINCT", "ALL"):
            continue
        projection.append(tok)
    items: list[Any] = []
    for tok in projection:
        if isinstance(tok, IdentifierList):
            items.extend(tok.get_identifiers())
        else:
            items.append(tok)
    if not items:
        return None
    out: list[str] = []
    for item in items:
        if getattr(item, "ttype", None) is Wildcard:
            return None
        if not isinstance(item, Identifier):
            return None
        name = _select_output_name(item)
        if name is None:
            return None
        out.append(name)
    return out or None


def _count_set_operations(sql: str) -> int:
    """Count top-level-flattened set-operation keywords in *sql* (``-1`` on error)."""
    try:
        parsed = sqlparse.parse(sql)
    except Exception:  # noqa: BLE001 — defensive; see :func:`strip_sql_predicates`
        return -1
    if not parsed:
        return -1
    count = 0
    for tok in parsed[0].flatten():
        if tok.ttype is Keyword and tok.value.upper() in _SQL_SET_OPERATIONS:
            count += 1
    return count


def strip_sql_predicates(sql: str) -> str | None:
    """Return *sql* with every ``WHERE`` clause removed, canonicalised, or ``None``.

    The append-only evolution classifier uses this as a cheap *proof of SELECT-
    list equality* for queries :func:`extract_select_output_columns` cannot read
    (set operations, unaliased expressions).  If the ``WHERE``-stripped,
    :func:`canonicalize_sql_for_hash`-normalised remainders of two query bodies
    are byte-identical, then only predicates changed and the projected schema is
    untouched — so a ``WHERE``-only rewrite of such a query still evolves in
    place.

    Every ``sqlparse`` ``Where`` group is dropped recursively.  A ``Where``
    group runs to end-of-statement, so on Snowflake-only trailing clauses
    (``QUALIFY`` / ``ORDER BY`` / ``LIMIT``) it swallows them too; that is
    harmless because those clauses sit *after* the projection and never carry
    output columns.  The one dangerous case — a ``Where`` group swallowing
    across a set-operation boundary and deleting an entire ``UNION`` branch — is
    rejected by requiring the set-operation count to survive the strip.

    Args:
        sql: The query body to strip predicates from.

    Returns:
        The canonicalised, ``WHERE``-free SQL, or ``None`` when *sql* is empty,
        is not a single statement, cannot be parsed, or the strip would change
        the set-operation count.
    """
    if not sql:
        return None
    try:
        statements = sqlparse.parse(sql)
    except Exception:  # noqa: BLE001 — never raise from a hashing / planning boundary
        return None
    if len(statements) != 1:
        return None

    def _walk(token: Any) -> str:
        parts: list[str] = []
        for child in token.tokens:
            if isinstance(child, Where):
                continue
            parts.append(_walk(child) if child.is_group else child.value)
        return "".join(parts)

    try:
        stripped = _walk(statements[0])
    except Exception:  # noqa: BLE001 — defensive against malformed parse trees
        return None
    if _count_set_operations(stripped) != _count_set_operations(sql):
        return None
    return canonicalize_sql_for_hash(stripped)


def to_authoring_identifier(value: Any) -> Any:
    """Re-quote a Snowflake stored-form identifier back into SQL syntax.

    The inverse of ``invariants._to_stored_form(authored=True)``: re-quotes a
    case-sensitive name (``userId`` -> ``"userId"``) and leaves an all-upper
    name untouched (``USERID`` -> ``USERID``), so the authored hash half (which
    resolves its input as SQL syntax) round-trips a stored value instead of
    folding ``userId`` to ``USERID``. Non-string/empty/parser-rejected values
    are returned unchanged.

    Args:
        value: A single stored-form identifier cell (typically a ``str``).

    Returns:
        The SQL-syntax (authoring-form) identifier, or *value* unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    stripped = value.strip()
    # Already SQL syntax: re-inferring a quoted name would double-escape it
    # (``"userId"`` -> ``"""userId"""``), so the boundary stays idempotent.
    if identifier.QUOTED_IDENTIFIER_RE.match(stripped):
        return stripped
    try:
        return identifier.get_inferred_name(stripped)
    except ValueError:
        return value


def to_comparison_identifier(value: Any) -> str:
    """Fold an identifier of either provenance to one case-insensitive key.

    Entity join keys reach the apply/export comparisons as Snowflake stored
    form (bare, case-preserved); secondary keys and re-quoted entity columns
    reach them as SQL syntax (quotes still encoding case). Both fold here to a
    single unquoted, upper-cased key so a comparison never depends on which
    form a value arrived in. Parser-rejected and non-string values fall back to
    a bare strip so no recovery path raises.

    Args:
        value: A single identifier cell (``str`` or anything ``str()`` accepts).

    Returns:
        The unquoted, upper-cased comparison key; ``""`` for empty input.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return identifier.get_unescaped_names(text).upper()
    except ValueError:
        return text.strip('"').upper()


def inline_query_source(
    data: dict[str, Any],
    spec_file_dir: str | None,
    *,
    template_vars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inline a ``BatchSource.query_file`` sidecar and normalize ``query``.

    Mirrors :func:`inline_udf_source` for ``BatchSource`` SQL sidecars:

    * If the doc's ``kind`` is not ``BatchSource``, return unchanged.
    * If the doc carries ``query_file:``, read the sibling file
      (resolved relative to ``spec_file_dir``), Jinja-render it with
      ``template_vars`` when supplied, set ``query`` to the
      comment-stripped, whitespace-normalized contents, and drop
      ``query_file``.
    * If the doc already carries an inline ``query:``, strip comments and
      normalize whitespace in place. Comment stripping (via
      :func:`strip_sql_comments`) runs BEFORE whitespace collapse so a
      ``--`` line comment cannot swallow the rest of the query; the
      whitespace normalisation keeps the DT-text round-trip stable on
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
        data["query"] = normalize_sql_whitespace(strip_sql_comments(source))
        del data["query_file"]
        return data

    inline_query = data.get("query")
    if isinstance(inline_query, str) and inline_query:
        data["query"] = normalize_sql_whitespace(strip_sql_comments(inline_query))

    return data


def compile_spec(
    data: dict[str, Any],
    spec_file_dir: str | None = None,
    *,
    template_vars: dict[str, Any] | None = None,
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
