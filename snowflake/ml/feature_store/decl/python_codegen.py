"""Python code generation for declarative feature store spec dicts.

Converts authoring-format spec dicts (as produced by
:mod:`snowflake.ml.feature_store.decl.exporter` helpers) into complete
``.py`` file bodies whose module-level variable assignments instantiate
the Pydantic spec classes.  Generated files are immediately loadable by
:func:`snowflake.ml.feature_store.decl.loader.load_python_file` and
produce a ``NO_CHANGE`` re-plan when passed to ``snow feature plan``.

Public surface: :func:`spec_dict_to_python_source` only.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any


def spec_dict_to_python_source(
    kind: str,
    obj_name: str,
    spec_dict: dict[str, Any],
) -> str:
    """Generate a complete ``.py`` file body for one spec object.

    Args:
        kind: The spec kind string — one of ``"Entity"``,
            ``"BatchSource"``, ``"StreamingSource"``,
            ``"BatchFeatureView"``, ``"StreamingFeatureView"``,
            ``"RealtimeFeatureView"``, ``"FeatureGroup"``.
        obj_name: Module-level variable name to assign the spec to
            (e.g. ``"user_id"``).
        spec_dict: Authoring-format dict as returned by the exporter
            helpers (``_entity_yaml_from_row``,
            ``_build_full_fidelity_fv``, ``_collect_datasources``, etc.).

    Returns:
        Complete Python file source as a string, including imports and
        optional UDF ``def`` block.

    Raises:
        ValueError: If *kind* is not one of the supported spec kinds, or if
            a feature-view spec has a UDF block with no ``function_definition``
            (or a body containing no ``def``).  Either case would emit
            ``function_definition=,`` / an undefined name — unrecoverable,
            not degraded.
    """
    if kind == "Entity":
        return _render_entity(obj_name, spec_dict)
    if kind == "BatchSource":
        return _render_batch_source(obj_name, spec_dict)
    if kind == "StreamingSource":
        return _render_streaming_source(obj_name, spec_dict)
    if kind in ("BatchFeatureView", "StreamingFeatureView", "RealtimeFeatureView"):
        return _render_feature_view(kind, obj_name, spec_dict)
    if kind == "FeatureGroup":
        return _render_feature_group(obj_name, spec_dict)
    raise ValueError(f"python_codegen: unsupported spec kind {kind!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_fscolumn(col: dict[str, Any]) -> str:
    """Render one FSColumn constructor call as a single-line string."""
    parts = [f"name={col['name']!r}", f"type={col['type']!r}"]
    # ``element_type`` MUST be rendered: ArrayType aggregation outputs
    # (last_n / last_distinct_n and every continuous-secondary-key output)
    # carry an inner element type that the applied DESCRIBE spec records.
    # Dropping it here made the python-form round-trip re-plan as
    # RECREATE_FV / "spec.features changed" for legacy streaming FVs.
    for extra in ("length", "precision", "scale", "tz", "default", "element_type"):
        if col.get(extra) is not None:
            parts.append(f"{extra}={col[extra]!r}")
    return "FSColumn(" + ", ".join(parts) + ")"


def _render_fscolumn_list(cols: list[dict[str, Any]], *, indent: int = 8) -> str:
    """Render a list of FSColumn constructors, one per line."""
    if not cols:
        return "[]"
    pad = " " * indent
    inner = (",\n" + pad).join(_render_fscolumn(c) for c in cols)
    return "[\n" + pad + inner + ",\n" + " " * (indent - 4) + "]"


def _render_source_ref(src: dict[str, Any]) -> str:
    """Render one SourceRef constructor (no columns — resolved from datasource files)."""
    parts = [f"name={src['name']!r}", f"source_type={src['source_type']!r}"]
    # Carry batch-source location fields when present (needed for BatchFVs)
    for fld in ("table", "query", "source_database", "source_schema"):
        val = src.get(fld)
        if val is not None:
            parts.append(f"{fld}={val!r}")
    return "SourceRef(" + ", ".join(parts) + ")"


def _render_feature(feat: dict[str, Any]) -> str:
    """Render one Feature constructor call."""
    parts: list[str] = []
    sc = feat.get("source_column")
    if isinstance(sc, dict):
        parts.append(f"source_column={_render_fscolumn(sc)}")
    oc = feat.get("output_column")
    if isinstance(oc, dict):
        parts.append(f"output_column={_render_fscolumn(oc)}")
    for fld in ("function", "window_sec", "offset_sec", "window", "offset"):
        val = feat.get(fld)
        if val is not None:
            parts.append(f"{fld}={val!r}")
    params = feat.get("function_params")
    if isinstance(params, dict) and params:
        parts.append(f"function_params={params!r}")
    inner = ", ".join(parts)
    return f"Feature({inner})"


def _render_feature_list(features: list[dict[str, Any]], *, indent: int = 8) -> str:
    """Render a list of Feature constructors, one per line."""
    if not features:
        return "[]"
    pad = " " * indent
    inner = (",\n" + pad).join(_render_feature(f) for f in features)
    return "[\n" + pad + inner + ",\n" + " " * (indent - 4) + "]"


def _render_source_ref_list(sources: list[dict[str, Any]], *, indent: int = 8) -> str:
    """Render a list of SourceRef constructors, one per line."""
    if not sources:
        return "[]"
    pad = " " * indent
    inner = (",\n" + pad).join(_render_source_ref(s) for s in sources)
    return "[\n" + pad + inner + ",\n" + " " * (indent - 4) + "]"


def _render_fv_ref(ref: dict[str, Any]) -> str:
    """Render one FeatureViewRef constructor call."""
    parts = [f"name={ref['name']!r}", f"version={ref['version']!r}"]
    slice_cols = ref.get("slice_columns")
    if slice_cols:
        parts.append(f"slice_columns={slice_cols!r}")
    # alias is emitted even when "" (semantically distinct from absent)
    if "alias" in ref:
        parts.append(f"alias={ref['alias']!r}")
    return "FeatureViewRef(" + ", ".join(parts) + ")"


def _render_fv_ref_list(refs: list[dict[str, Any]], *, indent: int = 8) -> str:
    """Render a list of FeatureViewRef constructors, one per line."""
    if not refs:
        return "[]"
    pad = " " * indent
    inner = (",\n" + pad).join(_render_fv_ref(r) for r in refs)
    return "[\n" + pad + inner + ",\n" + " " * (indent - 4) + "]"


def _extract_udf_func_name(source_code: str) -> str:
    """Return the function name from a ``def`` source-code string.

    Args:
        source_code: The UDF ``function_definition`` body, expected to contain
            a top-level ``def``.

    Returns:
        The name of the first ``def`` in *source_code*.

    Raises:
        ValueError: If *source_code* contains no ``def``.  The generated module
            references the callable by name (``function_definition=<name>``), so
            a body with no function would emit an undefined name that fails to
            load — a UDF body with no function is unrecoverable, not degraded.
    """
    match = re.search(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", source_code, re.MULTILINE)
    if match:
        return match.group(1)
    raise ValueError(
        "UDF function_definition contains no 'def'; cannot reference a callable "
        "by name. The UDF body is unrecoverable — re-register the feature view."
    )


def _render_udf_block(udf: dict[str, Any], func_name: str) -> str:
    """Render UDF constructor call referencing *func_name* as callable."""
    parts = [f"name={udf['name']!r}"]
    if udf.get("engine"):
        parts.append(f"engine={udf['engine']!r}")
    # Reference the callable by name — NOT by source-code string.
    parts.append(f"function_definition={func_name}")
    out_cols = udf.get("output_columns") or []
    if out_cols:
        # The ``UDF(`` constructor args sit at column 8, so the nested
        # ``output_columns`` list items must be at 12 (and its closing ``]`` at
        # 8) — otherwise the list visibly dedents out of its own constructor.
        parts.append(f"output_columns={_render_fscolumn_list(out_cols, indent=12)}")
    return "UDF(\n        " + ",\n        ".join(parts) + ",\n    )"


def _assemble_file(
    imports: list[str],
    *,
    def_block: str | None = None,
    assignment: str,
) -> str:
    """Assemble the final file body from parts."""
    sections: list[str] = ["from __future__ import annotations", ""]
    sections.append("\n".join(imports))
    sections.append("")
    if def_block:
        sections.append("")
        sections.append(textwrap.dedent(def_block).rstrip())
        sections.append("")
    sections.append("")
    sections.append(assignment)
    sections.append("")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Kind-specific renderers
# ---------------------------------------------------------------------------


def _render_entity(obj_name: str, spec: dict[str, Any]) -> str:
    join_keys = spec.get("join_keys") or []
    needed = ["Entity"]
    if join_keys:
        needed.append("FSColumn")
    imports = [f"from snowflake.ml.feature_store.decl import {', '.join(needed)}"]
    args: list[str] = [f"    name={spec['name']!r}"]
    args.append(f"    join_keys={_render_fscolumn_list(join_keys, indent=8)}")
    if spec.get("description"):
        args.append(f"    description={spec['description']!r}")
    assignment = f"{obj_name} = Entity(\n" + ",\n".join(args) + ",\n)"
    return _assemble_file(imports, assignment=assignment)


def _render_batch_source(obj_name: str, spec: dict[str, Any]) -> str:
    needed = ["BatchSource"]
    columns = spec.get("columns") or []
    if columns:
        needed.append("FSColumn")
    imports = [f"from snowflake.ml.feature_store.decl import {', '.join(needed)}"]

    args: list[str] = [f"    name={spec['name']!r}"]
    for fld in ("source_database", "source_schema", "table", "query"):
        val = spec.get(fld)
        if val is not None:
            args.append(f"    {fld}={val!r}")
    if columns:
        args.append(f"    columns={_render_fscolumn_list(columns, indent=8)}")
    assignment = f"{obj_name} = BatchSource(\n" + ",\n".join(args) + ",\n)"
    return _assemble_file(imports, assignment=assignment)


def _render_streaming_source(obj_name: str, spec: dict[str, Any]) -> str:
    needed = ["StreamingSource"]
    columns = spec.get("columns") or []
    if columns:
        needed.append("FSColumn")
    imports = [f"from snowflake.ml.feature_store.decl import {', '.join(needed)}"]

    args: list[str] = [f"    name={spec['name']!r}"]
    if spec.get("type"):
        args.append(f"    type={spec['type']!r}")
    if columns:
        args.append(f"    columns={_render_fscolumn_list(columns, indent=8)}")
    assignment = f"{obj_name} = StreamingSource(\n" + ",\n".join(args) + ",\n)"
    return _assemble_file(imports, assignment=assignment)


def _render_feature_view(kind: str, obj_name: str, spec: dict[str, Any]) -> str:  # noqa: C901
    needed: list[str] = [kind]

    sources = spec.get("sources") or []
    features = spec.get("features") or []
    udf_block = spec.get("udf")
    entities = spec.get("entities") or []

    has_source_ref = bool(sources)
    has_features = bool(features)
    has_udf = isinstance(udf_block, dict) and bool(udf_block)
    has_fscolumn = has_features or bool(isinstance(udf_block, dict) and udf_block.get("output_columns"))

    if has_source_ref:
        needed.append("SourceRef")
    if has_features:
        needed.append("Feature")
    if has_fscolumn:
        needed.append("FSColumn")
    if has_udf:
        needed.append("UDF")

    # UDF: extract def block and function name
    def_block: str | None = None
    func_name = ""
    if has_udf:
        udf_source_code = udf_block.get("function_definition") or ""  # type: ignore[union-attr]
        if not udf_source_code:
            raise ValueError(
                "UDF has no function_definition; cannot reference a callable "
                "by name. The UDF body is unrecoverable — re-register the feature view."
            )
        def_block = udf_source_code
        func_name = _extract_udf_func_name(udf_source_code)

    args: list[str] = [f"    name={spec['name']!r}"]
    if spec.get("version"):
        args.append(f"    version={spec['version']!r}")
    if spec.get("database"):
        args.append(f"    database={spec['database']!r}")
    # The Pydantic field is ``schema_`` (``schema`` collides with
    # ``BaseModel.schema``) and there is no ``schema`` alias, so a generated
    # ``schema=`` kwarg is silently discarded on ``load_python_file``.  Emit
    # ``schema_=`` so the binding actually takes effect.
    schema_val = spec.get("schema") or spec.get("schema_")
    if schema_val:
        args.append(f"    schema_={schema_val!r}")

    # online flag: only emit True explicitly; False is the default and is omitted
    if spec.get("online") is True:
        args.append("    online=True")

    # ``description`` (operational) mirrors the YAML export path so a described
    # FeatureView does not drift to a perpetual ``UPDATE_FV`` on a clean re-plan
    # (``description`` is stripped from the structural hash).  Emitted only for a
    # non-empty value, matching the YAML guard in ``_build_full_fidelity_fv``.
    # Placed right after ``online`` to match ``_FV_TOP_LEVEL_ORDER`` in the YAML
    # form so the two exports stay visually consistent.
    desc_val = spec.get("description")
    if isinstance(desc_val, str) and desc_val.strip():
        args.append(f"    description={desc_val.strip()!r}")

    if entities:
        args.append(f"    entities={entities!r}")

    if spec.get("timestamp_col"):
        args.append(f"    timestamp_col={spec['timestamp_col']!r}")

    for fld in ("feature_granularity_sec", "feature_aggregation_method", "refresh_freq", "target_lag"):
        val = spec.get(fld)
        if val is not None:
            args.append(f"    {fld}={val!r}")

    if has_source_ref:
        rendered_sources = _render_source_ref_list(sources, indent=8)
        args.append(f"    sources={rendered_sources}")

    if has_features:
        rendered_feats = _render_feature_list(features, indent=8)
        args.append(f"    features={rendered_feats}")

    if has_udf:
        udf_rendered = _render_udf_block(udf_block, func_name)  # type: ignore[arg-type]
        args.append(f"    udf={udf_rendered}")

    # Advanced BFV fields
    for fld in ("warehouse", "cluster_by", "refresh_mode", "initialize", "aggregation_secondary_keys"):
        val = spec.get(fld)
        if val is not None:
            args.append(f"    {fld}={val!r}")

    # ``append_only`` is emitted only when truthy so the generated Python mirrors
    # the YAML authoring form (the default ``False`` is never authored).
    if spec.get("append_only"):
        args.append("    append_only=True")

    storage_config = spec.get("storage_config")
    if isinstance(storage_config, dict) and storage_config:
        sc_parts = ", ".join(f"{k}={v!r}" for k, v in storage_config.items() if v is not None)
        args.append(f"    storage_config=StorageConfig({sc_parts})")
        if "StorageConfig" not in needed:
            needed.append("StorageConfig")

    backfill = spec.get("backfill")
    if isinstance(backfill, dict) and backfill:
        bp_parts = ", ".join(f"{k}={v!r}" for k, v in backfill.items() if v is not None)
        args.append(f"    backfill=Backfill({bp_parts})")
        if "Backfill" not in needed:
            needed.append("Backfill")

    # Build the import line once, after every ``needed`` append above, so a
    # future field that extends ``needed`` cannot be dropped by a stale rebuild.
    imports = [f"from snowflake.ml.feature_store.decl import {', '.join(needed)}"]
    assignment = f"{obj_name} = {kind}(\n" + ",\n".join(args) + ",\n)"
    return _assemble_file(imports, def_block=def_block, assignment=assignment)


def _render_feature_group(obj_name: str, spec: dict[str, Any]) -> str:
    imports = ["from snowflake.ml.feature_store.decl import FeatureGroup, FeatureViewRef"]
    refs = spec.get("feature_views") or []

    args: list[str] = [f"    name={spec['name']!r}"]
    if spec.get("version"):
        args.append(f"    version={spec['version']!r}")
    if spec.get("database"):
        args.append(f"    database={spec['database']!r}")
    # ``schema_`` (not ``schema``) is the bound field name — see
    # ``_render_feature_view`` for the rationale.
    schema_val = spec.get("schema") or spec.get("schema_")
    if schema_val:
        args.append(f"    schema_={schema_val!r}")
    # desc: always emit (empty string is valid)
    args.append(f"    desc={spec.get('desc', '')!r}")
    # auto_prefix: emit only when explicitly False; True is the default
    if spec.get("auto_prefix") is False:
        args.append("    auto_prefix=False")
    args.append(f"    feature_views={_render_fv_ref_list(refs, indent=8)}")

    assignment = f"{obj_name} = FeatureGroup(\n" + ",\n".join(args) + ",\n)"
    return _assemble_file(imports, assignment=assignment)
