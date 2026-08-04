"""File loading utilities for the declarative feature store.

Handles loading spec files from disk (Python, YAML, JSON), applying
Jinja2 templating, running compilation passes, and assembling a
``SpecBatch`` for validation and planning.

All functions are self-contained — no imports from snowflake.ml.feature_store.spec,
snowflake.snowpark, or snowflake.connector.
"""

from __future__ import annotations

import ast
import glob
import importlib.util
import json
import os
import pathlib
import sys
from typing import Any, Optional

import yaml

from snowflake.ml.feature_store.decl.compiler import compile_spec
from snowflake.ml.feature_store.decl.errors import SpecLoadError
from snowflake.ml.feature_store.decl.manifest import SOURCES_FOLDER
from snowflake.ml.feature_store.decl.serializer import spec_to_dict
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureGroup,
    FeatureView,
    SpecBase,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.templating import detect_and_render
from snowflake.ml.feature_store.decl.types import SpecBatch

_SPEC_EXTENSIONS = frozenset({".py", ".yaml", ".yml", ".json"})

# Project-mode (Phase 1C) ---------------------------------------------------
# These constants pin the layout enforced by ``load_from_project``. The
# ordering of ``_PROJECT_SUBDIRS`` is significant: it matches the
# dependency DAG (Entity -> Datasource -> FeatureView), so spec batches
# come out in dependency order without a separate sort.

_PROJECT_SUBDIRS: tuple[str, ...] = (
    "entities",
    "datasources",
    "feature_views",
    "feature_groups",
)
_RESERVED_SUBDIR = "macros"
_RESERVED_SUBDIR_MESSAGE = (
    "sources/macros/ is reserved for a future release; "
    "remove the directory or upgrade snowflake-ml-feature-store-decl."
)


def expand_input_files(patterns: list[str]) -> list[str]:
    """Expand a list of file patterns to a flat list of spec file paths.

    Supports:
    - ``./...`` or ``<dir>/...`` — recursive walk for ``.py``/``.yaml``/
      ``.yml``/``.json`` files.
    - Standard glob patterns (``*.yaml``, ``entities/*.py``).
    - Direct file paths.

    Non-spec file extensions are excluded from recursive glob results.

    Args:
        patterns: List of file paths or glob patterns.

    Returns:
        Sorted, de-duplicated list of resolved file paths.
    """
    result: list[str] = []
    for pattern in patterns:
        # A path that resolves to an existing directory (with no trailing
        # ``/...``) is treated as ``<dir>/...`` — a recursive walk.  Without
        # this rewrite the ``glob.glob`` branch below returns the directory
        # path itself, which then crashes ``process_file`` with
        # ``IsADirectoryError`` and the bare ``except Exception: pass`` in
        # :func:`load_specs` silently drops it, producing an empty
        # ``SpecBatch`` and a misleading "no changes" plan.  Auto-expansion
        # gives bare directories the same recursive semantics as ``./...``.
        normalized = pattern.rstrip("/")
        is_recursive_marker = pattern.endswith("/...") or pattern == "./..."
        if not is_recursive_marker and os.path.isdir(normalized):
            pattern = f"{normalized}/..."
            is_recursive_marker = True

        if is_recursive_marker:
            if pattern == "./...":
                root = "."
            else:
                root = pattern.rsplit("/...", 1)[0]
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in sorted(filenames):
                    if os.path.splitext(fname)[1] in _SPEC_EXTENSIONS:
                        result.append(os.path.join(dirpath, fname))
        else:
            expanded = glob.glob(pattern)
            if expanded:
                result.extend(sorted(expanded))
            else:
                result.append(pattern)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in result:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _get_defined_names(filepath: str) -> set[str]:
    """Get names of variables assigned at module level (not imported).

    Parses the file with ``ast`` to distinguish local assignments from
    import statements, so that imported objects are not mistakenly treated
    as locally-defined specs.

    Args:
        filepath: Path to the Python file.

    Returns:
        Set of names assigned at module level.
    """
    with open(filepath) as f:
        source = f.read()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def load_python_file(filepath: str) -> list[tuple[str, Any]]:
    """Load a Python spec file and extract locally-defined spec objects.

    Picks up every module-level instance of :class:`Entity`,
    :class:`StreamingSource`, :class:`BatchSource`, :class:`FeatureView`
    (including the three Python-form subclasses
    :class:`StreamingFeatureView` / :class:`BatchFeatureView` /
    :class:`RealtimeFeatureView`), :class:`FeatureGroup`, or
    :class:`SpecBase` — the class IS the kind (Q2 of
    ``plans/python_form/python_authoring_form.md``).

    Only returns objects that are *locally defined* in the file (not imported).
    This allows spec files to import source objects for inline use without
    those sources being treated as new specs to apply.

    Sibling directories relative to the file's parent are added to
    ``sys.path`` so that cross-directory imports work (e.g. a feature view
    importing a data source from a sibling directory).

    Args:
        filepath: Absolute or relative path to the Python spec file.

    Returns:
        List of ``(name, spec_object)`` tuples for locally-defined specs.

    Raises:
        ValueError: If the file cannot be loaded as a Python module.
    """
    filepath = os.path.abspath(filepath)
    file_dir = os.path.dirname(filepath)
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    orig_path = list(sys.path)
    try:
        # Append (not insert at 0) so stdlib/third-party modules are not shadowed.
        # Paths are restored in the finally block so sys.path is not permanently
        # mutated — preventing CWE-426 code injection via sibling directories.
        if file_dir not in sys.path:
            sys.path.append(file_dir)

        parent_dir = os.path.dirname(file_dir)
        if os.path.isdir(parent_dir):
            for entry in sorted(os.listdir(parent_dir)):
                sibling = os.path.join(parent_dir, entry)
                if os.path.isdir(sibling) and sibling not in sys.path:
                    sys.path.append(sibling)

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load Python file: {filepath}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = orig_path

    defined_names = _get_defined_names(filepath)
    known_types = (FeatureView, FeatureGroup, StreamingSource, BatchSource, Entity, SpecBase)
    objects: list[tuple[str, Any]] = []

    for attr_name in dir(module):
        if attr_name.startswith("_") or attr_name not in defined_names:
            continue
        obj = getattr(module, attr_name)
        if isinstance(obj, known_types):
            objects.append((attr_name, obj))

    return objects


def process_file(
    filepath: str,
    config_source: Optional[str] = None,
    *,
    template_vars: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Load a single spec file and return a list of compiled spec dicts.

    Steps:
    1. Read file content.
    2. Apply Jinja2 template rendering if needed (via ``templating.detect_and_render``).
    3. Dispatch by extension:
       - ``.py`` → :func:`load_python_file` then :func:`~compiler.compile_spec`
       - ``.yaml`` / ``.yml`` → ``yaml.safe_load`` then ``compile_spec``
       - ``.json`` → ``json.loads`` then ``compile_spec``

    The ``template_vars`` kwarg is threaded into
    :func:`~compiler.compile_spec` so companion sidecars
    (``udf.file`` / ``query_file``) pick up the same merged manifest
    context as the top-level YAML/JSON/Python body.  When
    ``template_vars`` is omitted but ``config_source`` is a JSON object
    (the legacy single-argument form used by callers that pre-encoded
    their context), the helper recovers the dict so sidecar templating
    still works.

    Args:
        filepath: Path to the spec file.
        config_source: Jinja2 config source string (passed to :func:`load_config`),
            or ``None`` if no templating is needed.
        template_vars: Pre-merged manifest template variables.  When
            supplied, this dict is forwarded to ``compile_spec`` so the
            on-disk ``.py`` UDF and ``.sql`` query sidecars render with
            the same context as the parent file.

    Returns:
        List of compiled spec dicts.
    """
    filepath = os.path.abspath(filepath)
    spec_dir = os.path.dirname(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath) as f:
        raw = f.read()

    rendered = detect_and_render(raw, config_source, filepath)

    # Recover template_vars from config_source when only the legacy
    # positional argument was supplied. ``config_source`` may also be a
    # path to a YAML/JSON file or the special ``"env"`` sentinel; those
    # legacy shapes are not threaded into sidecar templating to keep
    # the new contract narrow — manifest-driven callers always supply
    # both arguments via ``load_from_project``.
    effective_template_vars = template_vars
    if effective_template_vars is None and isinstance(config_source, str) and config_source:
        try:
            maybe = json.loads(config_source)
            if isinstance(maybe, dict):
                effective_template_vars = maybe
        except json.JSONDecodeError:
            pass

    if ext == ".py":
        # For Python files with template rendering, write to a temp file
        if rendered != raw:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=spec_dir,
                prefix=f".{os.path.splitext(os.path.basename(filepath))[0]}_",
                delete=False,
            ) as tmp:
                tmp.write(rendered)
                tmp_path = tmp.name
            try:
                objects = load_python_file(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            objects = load_python_file(filepath)

        results = []
        for _name, obj in objects:
            data = spec_to_dict(obj)
            data = compile_spec(data, spec_file_dir=spec_dir, template_vars=effective_template_vars)
            results.append(data)
        return results

    elif ext in (".yaml", ".yml"):
        data = yaml.safe_load(rendered)
        if not isinstance(data, dict):
            return []
        data = compile_spec(data, spec_file_dir=spec_dir, template_vars=effective_template_vars)
        return [data]

    elif ext == ".json":
        data = json.loads(rendered)
        if not isinstance(data, dict):
            return []
        data = compile_spec(data, spec_file_dir=spec_dir, template_vars=effective_template_vars)
        return [data]

    else:
        return []


#: Fields the loader preserves when degrading a malformed spec dict to
#: a bare :class:`SpecBase`.  These are exactly the fields declared on
#: :class:`SpecBase` itself (see ``spec_models.py``); kind-specific
#: fields (``feature_views``, ``sources``, ``entities``, ``features``,
#: ...) are intentionally dropped because ``SpecBase`` doesn't model
#: them — Pydantic would reject them as extras.
#:
#: ``schema_`` (with trailing underscore) is the model field name —
#: YAML ``schema`` is renamed upstream by ``_inject_database_schema``
#: (loader.py near line 669) before reaching :func:`_dict_to_spec`, so
#: the fallback reads the already-normalized key.
_SPECBASE_IDENTITY_FIELDS: tuple[str, ...] = (
    "kind",
    "name",
    "version",
    "database",
    "schema_",
    "description",
)


def _specbase_fallback(data: dict[str, Any], kind: str) -> SpecBase:
    """Build a bare :class:`SpecBase` from a spec dict, preserving identity fields.

    Used by :func:`_dict_to_spec` when concrete-model validation fails
    for a non-allowlisted reason (or when ``kind`` is unknown).

    The defensive contract: every field that :class:`SpecBase` models
    (``kind``, ``name``, ``version``, ``database``, ``schema_``,
    ``description``) is preserved if present in ``data`` — so a
    downstream validator sees the version that was on disk and can
    report the real shape problem instead of a misleading
    ``MISSING_VERSION`` (the user-visible symptom that motivated this
    helper; see ``plans/fg_export_version_fix_loader_fallback_preserves_version.plan.md``).

    Kind-specific fields are intentionally dropped — :class:`SpecBase`
    doesn't model them and Pydantic would reject them as extras.

    Args:
        data: The compiled spec dict that failed concrete-model validation.
        kind: The kind string the caller is degrading from (used as the
            ``kind`` value on the returned ``SpecBase`` so the
            discriminator survives even when ``data`` lacks the key).

    Returns:
        SpecBase carrying only the identity fields preserved from ``data``.
    """
    fallback: dict[str, Any] = {"kind": kind}
    for field in _SPECBASE_IDENTITY_FIELDS:
        if field == "kind":
            continue
        if field in data:
            fallback[field] = data[field]
    return SpecBase.model_validate(fallback)


def _dict_to_spec(data: dict[str, Any]) -> SpecBase:
    """Convert a compiled spec dict to the appropriate SpecBase subclass.

    Q8 of ``plans/python_form/python_authoring_form.md`` maps the three
    feature-view kind strings to the three new concrete subclasses so a
    YAML / JSON / Python load all produce the same ``isinstance`` chain.

    Migration errors (renamed / removed authoring fields) are re-raised
    so the author sees the actionable migration message instead of a
    confusing "feature view has no sources" downstream error from the
    degraded ``SpecBase`` fallback.  The stable substrings
    ``"has been renamed to"`` (renames) and ``"has been removed"``
    (removals) in the validator error messages are the explicit
    contract for which errors propagate.

    Defensive contract on the SpecBase fallback (delegated to
    :func:`_specbase_fallback`): when concrete-model validation fails
    for a non-allowlisted reason, the returned bare ``SpecBase``
    preserves identity fields (``kind``, ``name``, ``version``,
    ``database``, ``schema_``, ``description``) from ``data`` so the
    downstream validator sees the real shape — and ``MISSING_VERSION``
    only fires when the input truly has no version.  Kind-specific
    fields are intentionally dropped (see ``_specbase_fallback``).

    Args:
        data: Compiled spec dict carrying a ``kind`` discriminator.

    Returns:
        The matching Pydantic model instance, or a bare ``SpecBase`` if
        the kind is unknown or model validation fails for a non-migration
        reason.

    Raises:
        ValidationError: When the input dict triggers a migration error
            (legacy authoring key still present); the original
            ``pydantic.ValidationError`` is re-raised unchanged so
            callers see the full Pydantic context.
    """
    from snowflake.ml.feature_store.decl.spec_models import (
        BatchFeatureView,
        BatchSource,
        Entity,
        FeatureGroup,
        RealtimeFeatureView,
        StreamingFeatureView,
        StreamingSource,
    )

    kind = data.get("kind", "")
    kind_map: dict[str, type[SpecBase]] = {
        "Entity": Entity,
        "StreamingSource": StreamingSource,
        "BatchSource": BatchSource,
        "StreamingFeatureView": StreamingFeatureView,
        "RealtimeFeatureView": RealtimeFeatureView,
        "BatchFeatureView": BatchFeatureView,
        "FeatureGroup": FeatureGroup,
    }
    from pydantic import ValidationError

    cls = kind_map.get(kind, SpecBase)
    try:
        return cls.model_validate(data)
    except ValidationError as exc:
        msg = str(exc)
        # Allowlist of error-message substrings that MUST propagate to the
        # caller (rather than degrading to a bare SpecBase fallback) so
        # authors see actionable validator messages instead of a silent
        # downstream "spec has no sources" surprise.
        #
        # - "has been renamed to" / "has been removed": migration errors
        #   from ``FeatureView._normalize_legacy_authoring_keys`` (etc.).
        # - "is not valid on": kind-targeted rejection from
        #   ``FeatureView._reject_target_lag_on_stream_or_realtime``
        #   (and any future ``_reject_X_on_Y`` validators that follow
        #   the same message convention).
        if "has been renamed to" in msg or "has been removed" in msg or "is not valid on" in msg:
            raise
        return _specbase_fallback(data, kind)
    except Exception:
        return _specbase_fallback(data, kind)


def load_specs(
    files: list[str],
    config: Optional[dict[str, Any]] = None,
) -> SpecBatch:
    """Load and parse spec files into a :class:`~types.SpecBatch`.

    Accepts ``.py``, ``.yaml``, ``.yml``, and ``.json`` files. Glob patterns
    and ``./...`` recursive expansion are resolved before loading.

    Files that fail due to unresolved imports are retried after all other
    files have been processed — this handles dependency ordering automatically
    (e.g. a feature group that imports a feature view from a sibling directory).

    Args:
        files: List of file paths or glob patterns.
        config: Jinja2 template variables as a dict. Converted to an inline
            JSON string for passing to :func:`process_file`. May be ``None``.

    Returns:
        SpecBatch with all loaded specs and source file paths.
    """
    import json as _json

    config_source: Optional[str] = None
    if config:
        config_source = _json.dumps(config)

    expanded = expand_input_files(files)
    all_specs: list[SpecBase] = []
    source_files: list[str] = []

    pending = [(fp, None) for fp in expanded]
    max_passes = len(pending) + 1

    for _ in range(max_passes):
        still_pending: list[tuple[str, Any]] = []
        made_progress = False

        for filepath, _prev_err in pending:
            if not os.path.exists(filepath):
                continue
            try:
                spec_dicts = process_file(filepath, config_source=config_source)
                for d in spec_dicts:
                    all_specs.append(_dict_to_spec(d))
                if filepath not in source_files:
                    source_files.append(filepath)
                made_progress = True
            except (ImportError, SyntaxError) as e:
                still_pending.append((filepath, e))
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"Skipping spec file '{filepath}': {exc}",
                    stacklevel=2,
                )

        if not still_pending:
            break
        if not made_progress:
            break
        pending = still_pending

    return SpecBatch(specs=all_specs, source_files=source_files)


# ---------------------------------------------------------------------------
# Project-mode entry point (Phase 1C)
# ---------------------------------------------------------------------------


def _is_udf_companion_py(py_path: str) -> bool:
    """Return True if ``py_path`` is a UDF body referenced by a sibling YAML.

    A ``<NAME>.py`` file is treated as the UDF body for a FeatureView
    (and therefore not a Python spec to ``importlib`` -load) when any
    sibling ``.yaml`` / ``.yml`` file in the same directory parses
    cleanly and carries a top-level ``udf.file:`` string whose basename
    equals the ``.py`` file's basename.

    Historically (pre-Q9 of ``plans/python_form/python_authoring_form.md``)
    this helper only checked the same-stem YAML peer (``<NAME>.yaml``),
    which was correct for the convention where a UDF body's filename
    matches the YAML FV that owns it.  Q9's spec-first-with-UDF-fallback
    rule relaxes that constraint: any sibling YAML that names the ``.py``
    via ``udf.file:`` counts as the owner, so authors are free to name
    UDF bodies independently of their owning FV's YAML.

    The compiler's :func:`compiler.inline_udf_source` reads the file
    as plain text, so the loader must skip it during spec collection.

    Args:
        py_path: Absolute path to a ``.py`` file under
            ``<project_root>/sources/{entities,datasources,feature_views,feature_groups}/``.

    Returns:
        ``True`` if any sibling YAML claims ``py_path`` via
        ``udf.file:``; ``False`` otherwise (orphan ``.py``, no sibling
        YAML, or the sibling YAML is unparsable, e.g. due to Jinja2
        placeholders).
    """
    py_dir = os.path.dirname(py_path)
    py_basename = os.path.basename(py_path)

    if not os.path.isdir(py_dir):
        return False

    for fname in os.listdir(py_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".yaml", ".yml"):
            continue
        yaml_path = os.path.join(py_dir, fname)
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        udf = data.get("udf")
        if not isinstance(udf, dict):
            continue
        udf_file = udf.get("file")
        if isinstance(udf_file, str) and os.path.basename(udf_file) == py_basename:
            return True
    return False


def _is_query_companion_sql(sql_path: str) -> bool:
    """Return True if ``sql_path`` is a BatchSource ``query_file`` sidecar.

    Mirrors :func:`_is_udf_companion_py` for SQL sidecars. A
    ``<NAME>.sql`` file is treated as the SQL body for a sibling
    ``BatchSource`` YAML (and therefore not a candidate for spec
    discovery) when:

    1. A sibling ``<NAME>.yaml`` or ``<NAME>.yml`` exists in the same
       directory (basename minus extension equals ``<NAME>``), AND
    2. that sibling YAML parses cleanly with ``yaml.safe_load``, AND
    3. its top-level ``query_file:`` is a string whose basename equals
       the ``.sql`` file's basename.

    The compiler's :func:`compiler.inline_query_source` reads the file
    as plain text and inlines it as ``query``; the loader uses this
    helper to distinguish a deliberately-sidecared SQL body from a
    stray ``.sql`` file the user dropped into the spec tree.

    Note: ``.sql`` is not in :data:`_SPEC_EXTENSIONS`, so even an
    "orphan" ``.sql`` is silently skipped during spec discovery; this
    helper exists so other components (exporter, diagnostics) can
    distinguish the two cases.

    Args:
        sql_path: Absolute path to a ``.sql`` file under
            ``<project_root>/sources/{entities,datasources,feature_views}/``.

    Returns:
        ``True`` if ``sql_path`` is a sidecar referenced by a sibling
        YAML's ``query_file:``; ``False`` otherwise (orphan SQL, missing
        / mismatched ``query_file:``, or unparsable YAML peer).
    """
    sql_dir = os.path.dirname(sql_path)
    sql_basename = os.path.basename(sql_path)
    sql_stem = os.path.splitext(sql_basename)[0]

    for yaml_ext in (".yaml", ".yml"):
        yaml_path = os.path.join(sql_dir, sql_stem + yaml_ext)
        if not os.path.isfile(yaml_path):
            continue
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        query_file = data.get("query_file")
        if isinstance(query_file, str) and os.path.basename(query_file) == sql_basename:
            return True
    return False


def _collect_subdir_spec_files(subdir: pathlib.Path) -> list[str]:
    """Return all spec files under ``subdir`` in lexicographic order.

    Walks ``subdir`` recursively, filters to the known spec extensions
    (``.py`` / ``.yaml`` / ``.yml`` / ``.json``), and sorts the result
    by the path relative to ``subdir`` so the output is deterministic
    across filesystems with non-sorted ``readdir`` order.

    Per Q9 of ``plans/python_form/python_authoring_form.md`` the
    eager "skip ``.py`` if it is a UDF companion" pre-filter is
    intentionally GONE: under the spec-first-with-UDF-fallback
    contract, every ``.py`` is given a chance to load as a Python
    authoring spec module first.  The companion check now lives in
    :func:`load_from_project`'s per-file try/except so it only kicks
    in when exec actually fails, and on the zero-spec-instances path
    it kicks in via the source-files-record gate (so a YAML
    ``udf.file:`` companion that exec's cleanly still does not get
    counted as a spec source).

    ``.sql`` files are NOT in :data:`_SPEC_EXTENSIONS` and are
    therefore implicitly skipped from spec discovery. The companion
    helper :func:`_is_query_companion_sql` exists for documentation
    and exporter / diagnostic use; if a future change ever adds
    ``.sql`` to :data:`_SPEC_EXTENSIONS`, the explicit guard below
    keeps companion sidecars from being treated as spec files.

    Args:
        subdir: Path to one of the canonical sub-directories under
            ``<project_root>/sources/`` (``entities``, ``datasources``,
            ``feature_views``, or ``feature_groups``).

    Returns:
        Sorted list of absolute file paths for every spec-extension
        file found under ``subdir``.
    """
    if not subdir.is_dir():
        return []

    collected: list[tuple[str, str]] = []
    base = str(subdir)
    for dirpath, _dirnames, filenames in os.walk(base):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            absolute = os.path.join(dirpath, fname)
            # Defensive: if .sql ever joins _SPEC_EXTENSIONS, sidecars
            # paired with a BatchSource YAML's query_file: must still
            # be skipped. Today this branch never fires (.sql is not in
            # the set), but the guard documents intent and keeps the
            # invariant local to one place.
            if ext == ".sql" and _is_query_companion_sql(absolute):
                continue
            if ext not in _SPEC_EXTENSIONS:
                continue
            relative = os.path.relpath(absolute, base)
            collected.append((relative, absolute))

    collected.sort(key=lambda pair: pair[0])
    return [absolute for _relative, absolute in collected]


def _inject_target_qualifier(
    spec_dict: dict[str, Any],
    *,
    database: str,
    schema: str,
) -> dict[str, Any]:
    """Inject ``database`` / ``schema`` into a compiled spec dict.

    Honors a user-set value (``database``, ``schema``, or the Pydantic
    field alias ``schema_``) over the supplied kwargs. After injection
    the ``schema`` key is renamed to ``schema_`` so the value flows into
    the Pydantic model field of the same name during ``model_validate``.

    Args:
        spec_dict: Compiled spec dict (output of :func:`process_file`).
            Mutated in place AND returned for chaining.
        database: Fallback target database (used when the dict does not
            already declare one).
        schema: Fallback target schema (used when the dict does not
            already declare one).

    Returns:
        The mutated ``spec_dict``.
    """
    if "database" not in spec_dict:
        spec_dict["database"] = database

    user_set_schema = "schema" in spec_dict or "schema_" in spec_dict
    if not user_set_schema:
        spec_dict["schema"] = schema

    # The Pydantic model field is named ``schema_`` (``schema`` is reserved
    # by BaseModel). Translate the YAML-style key so model_validate picks
    # the value up; otherwise it would be silently dropped as an extra
    # field and the resulting spec would carry ``schema_ = None``.
    if "schema" in spec_dict and "schema_" not in spec_dict:
        spec_dict["schema_"] = spec_dict.pop("schema")

    return spec_dict


def load_from_project(
    project_root: pathlib.Path,
    *,
    database: str,
    schema: str,
    template_vars: Optional[dict[str, Any]] = None,
) -> SpecBatch:
    """Load every spec under ``<project_root>/sources/`` (project mode).

    Walks the three canonical sub-directories of
    ``<project_root>/sources/``: ``entities/``, ``datasources/``, and
    ``feature_views/`` — in that order, recursively, with lexicographic
    sorting within each. ``.yaml``, ``.yml``, ``.json``, and ``.py``
    files are picked up; every other extension is silently skipped.
    Files outside the three canonical sub-directories (e.g.
    ``<project_root>/manifest.yml``, ``<project_root>/sources/notes/``,
    or anything else in ``<project_root>/``) are NOT loaded.

    Before each parsed spec dict is converted to a Pydantic model,
    ``database`` and ``schema`` are injected from the kwargs unless the
    spec already supplies its own value — the user's explicit
    ``database:`` / ``schema:`` field in YAML / JSON / Python wins.

    Args:
        project_root: Resolved project root (the directory containing
            ``manifest.yml``). ``<project_root>/sources/`` MUST exist.
        database: Active target database (per D2). Injected as the
            ``database`` field on every loaded spec dict before
            Pydantic validation if not already present.
        schema: Active target schema (per D2). Same injection rule as
            ``database``, but the YAML key is renamed to ``schema_``
            internally so the value lands on the Pydantic ``schema_``
            field.
        template_vars: Jinja2 template variables, already merged from
            the manifest's ``templating: defaults`` /
            ``configurations`` and any ``--variable key=value``
            runtime overrides. ``None`` means "no templating", which
            matches the contract of :func:`process_file`.

    Returns:
        ``SpecBatch`` with every loaded spec in deterministic order
        (entities, then datasources, then feature_views; lexicographic
        within each sub-directory). ``source_files`` carries every
        on-disk path that contributed to the batch.

    Raises:
        SpecLoadError: If ``<project_root>/sources/`` is missing, if
            ``<project_root>/sources/macros/`` exists (D7 reserved),
            or if a spec file under one of the canonical
            sub-directories fails to parse.
    """
    root = pathlib.Path(project_root)
    sources_root = root / SOURCES_FOLDER

    if not sources_root.is_dir():
        raise SpecLoadError(f"No 'sources/' directory under '{root}'.")

    if (sources_root / _RESERVED_SUBDIR).exists():
        raise SpecLoadError(_RESERVED_SUBDIR_MESSAGE)

    config_source: Optional[str] = None
    if template_vars:
        config_source = json.dumps(template_vars)

    all_specs: list[SpecBase] = []
    source_files: list[str] = []

    for subdir_name in _PROJECT_SUBDIRS:
        subdir = sources_root / subdir_name
        for filepath in _collect_subdir_spec_files(subdir):
            is_py = filepath.lower().endswith(".py")

            try:
                spec_dicts = process_file(
                    filepath,
                    config_source=config_source,
                    template_vars=template_vars,
                )
            except SpecLoadError:
                raise
            except Exception as exc:
                # Q9 spec-first with UDF fallback: every ``.py`` is
                # tried as a spec module first.  If exec fails AND a
                # sibling YAML claims the file via ``udf.file:`` the
                # file is treated as a UDF body — silently skipped
                # from spec discovery (the YAML's compiler still reads
                # the .py as text downstream).  All other exec
                # failures bubble up as ``SpecLoadError``.
                if is_py and _is_udf_companion_py(filepath):
                    continue
                raise SpecLoadError(f"Failed to load spec file '{filepath}': {exc}") from exc

            # Q9 zero-spec-instances rule for ``.py``: a file that
            # exec'd cleanly but declared no module-level spec
            # instances is a pure UDF body — drop it entirely from
            # the batch (no specs, no source-files entry) so the
            # YAML's ``udf.file:`` companion mechanism remains the
            # only consumer of that file's text.
            if is_py and not spec_dicts:
                continue

            for raw_dict in spec_dicts:
                spec_dict = _inject_target_qualifier(
                    raw_dict,
                    database=database,
                    schema=schema,
                )
                all_specs.append(_dict_to_spec(spec_dict))

            if filepath not in source_files:
                source_files.append(filepath)

    return SpecBatch(specs=all_specs, source_files=source_files)
