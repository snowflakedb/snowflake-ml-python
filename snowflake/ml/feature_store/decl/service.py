"""Pure parsing and SQL-building functions for feature store service management.

No IO — no SQL execution, no HTTP calls, no file writes.
"""

from __future__ import annotations

import json
from typing import Any, Optional

try:
    from snowflake.ml.feature_store.online_service import (
        _parse_status_payload as _online_service_parse_status,
    )

    _HAS_ONLINE_SERVICE = True
except ImportError:
    _online_service_parse_status = None  # type: ignore[assignment]
    _HAS_ONLINE_SERVICE = False


# Raw-payload top-level keys that the rich status display consumes but
# that snowml-core's ``OnlineServiceStatus`` dataclass does not expose.
# Both ``parse_service_status`` branches forward these verbatim from the
# parsed JSON so the CLI rendering stays consistent regardless of which
# parser path is active.
_PASSTHROUGH_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "runtime_id",
    "compute_pool",
    "postgres",
    "service",
    "network_rules",
)


def _endpoint_to_dict(ep: Any) -> dict[str, Any]:
    """Project an ``OnlineServiceEndpoint`` dataclass to a CLI-friendly dict.

    Carries the optional ``privatelink_url`` / ``internal_url`` fields only
    when they were supplied by the runtime so downstream consumers can
    distinguish "no PrivateLink configured" from "PrivateLink endpoint
    intentionally empty".

    Args:
        ep: ``OnlineServiceEndpoint`` instance from snowml-core.

    Returns:
        Dict with ``name`` / ``url`` plus the optional URL variants when set.
    """
    out: dict[str, Any] = {"name": ep.name, "url": ep.url}
    pl = getattr(ep, "privatelink_url", None)
    if pl:
        out["privatelink_url"] = pl
    iu = getattr(ep, "internal_url", None)
    if iu:
        out["internal_url"] = iu
    return out


def parse_service_status(raw_json: str) -> dict[str, Any]:
    """Parse the JSON string from SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS.

    Returns dict with: status, message, endpoints (list of dicts with
    ``name`` / ``url`` plus optional ``privatelink_url`` / ``internal_url``),
    created_at, updated_at, and the nested component dicts the rich CLI
    display consumes (``runtime_id``, ``compute_pool``, ``postgres``,
    ``service``, ``network_rules``).  The ``secret`` payload is
    intentionally not forwarded so the Postgres password is never
    collected.

    Uses ``online_service._parse_status_payload`` when available to
    benefit from the canonical typed parser, then layers the raw-payload
    fields it does not expose back on top.  Falls back to ``json.loads``
    when snowml-core's online_service module is not importable.

    Args:
        raw_json: JSON string returned by the system function.

    Returns:
        Parsed status dict.
    """
    parsed = json.loads(raw_json)

    if _HAS_ONLINE_SERVICE and _online_service_parse_status is not None:
        svc = _online_service_parse_status(parsed)
        result: dict[str, Any] = {
            "status": svc.status,
            "message": svc.message,
            "endpoints": [_endpoint_to_dict(ep) for ep in svc.endpoints],
            "created_at": svc.created_at,
            "updated_at": svc.updated_at,
        }
        for key in _PASSTHROUGH_TOP_LEVEL_KEYS:
            if key in parsed:
                result[key] = parsed[key]
        return result

    endpoints_raw = parsed.get("endpoints", [])
    result = {
        "status": parsed.get("status", ""),
        "message": parsed.get("message", ""),
        "endpoints": endpoints_raw,
    }
    if "created_at" in parsed:
        result["created_at"] = parsed["created_at"]
    if "updated_at" in parsed:
        result["updated_at"] = parsed["updated_at"]
    for key in _PASSTHROUGH_TOP_LEVEL_KEYS:
        if key in parsed:
            result[key] = parsed[key]
    return result


def get_service_endpoint(status: dict[str, Any], name: str) -> Optional[str]:
    """Extract an endpoint URL by name from a parsed status dict.

    Args:
        status: Dict returned by :func:`parse_service_status`.
        name: Endpoint name to look for (e.g. ``"ingest"`` or ``"query"``).

    Returns:
        URL string, or ``None`` if the endpoint is not found.
    """
    for ep in status.get("endpoints", []):
        if isinstance(ep, dict) and ep.get("name") == name:
            return ep.get("url")
    return None


def service_sql(
    database: str,
    schema: str,
    producer_role: str = "ACCOUNTADMIN",
    consumer_role: str = "PUBLIC",
) -> dict[str, str]:
    """Return SQL strings for service management, keyed by operation.

    Args:
        database: Snowflake database name.
        schema: Snowflake schema name.
        producer_role: Snowflake role for producing features.
        consumer_role: Snowflake role for consuming features.

    Returns:
        Dict with keys: get_status, create, drop, show_ofts,
        drop_oft_template.
    """
    location = f"{database}.{schema}"
    config = json.dumps(
        {
            "roles": {
                "producer_role_name": producer_role,
                "consumer_role_name": consumer_role,
            }
        }
    ).replace("'", "\\'")
    return {
        "get_status": (f"SELECT SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('{location}')"),
        "create": (f"SELECT SYSTEM$CREATE_FEATURE_STORE_ONLINE_SERVICE('{location}', '{config}')"),
        "drop": f"SELECT SYSTEM$DROP_FEATURE_STORE_ONLINE_SERVICE('{location}')",
        "show_ofts": f"SHOW ONLINE FEATURE TABLES IN SCHEMA {location}",
        "drop_oft_template": (f"DROP ONLINE FEATURE TABLE IF EXISTS {location}.{{name}}"),
    }


# ---------------------------------------------------------------------------
# Rich status display
# ---------------------------------------------------------------------------


def format_status_display(
    status: dict[str, Any],
    user: str = "",
    database: str = "",
    schema: str = "",
    *,
    verbose: bool = False,
) -> str:
    """Format the parsed online-service status into a rich multi-line display.

    Two layouts share this single rendering path:

    * **Default-compact** (``verbose=False``) — header banner, the
      umbrella ``+ Online Service`` heading row, single-line heading
      summaries for the Compute Pool / Postgres / Service
      sub-components, and the Endpoints block with every URL variant
      the runtime supplied.  The deep
      per-component detail (``Name`` / ``Family`` / ``Nodes`` /
      ``Min/Max`` / ``Auto-suspend`` / ``Host`` / ``Image`` /
      per-instance rows / ``Upgrading:`` flag) and the Network Rules
      block are omitted.  This is the surface most operators want.
    * **Verbose** (``verbose=True``) — adds every detail row and
      block back in.  Use this when diagnosing a misbehaving runtime.

    The Postgres password is never collected or displayed —
    ``parse_service_status`` does not forward the secret payload.

    The heading-only single-line summaries always render as long as
    the parsed payload carries the corresponding nested dict; this
    keeps the default view honest about whether a component is
    healthy without dragging in its detail rows.

    Ported from ``ofs_quake/tests/utils/sf_provisioner.py::print_status()``.

    Args:
        status: Dict returned by :func:`parse_service_status` (with nested data).
        user: Current Snowflake user (for the header).
        database: Current database (for the header).
        schema: Current schema (for the header).
        verbose: When ``True``, render every detail row and ancillary
            block.  When ``False`` (the default), render the compact
            layout described above.

    Returns:
        Formatted multi-line string ready to print.
    """
    LBL = 18
    lines: list[str] = []

    def _icon(ok: bool) -> str:
        return "+" if ok else "-"

    def _heading(icon: str, title: str, state: str) -> None:
        lines.append(f"  {icon} {title:<24s} {state}")

    def _kv(key: str, val: object) -> None:
        lines.append(f"     {key + ':':<{LBL}s} {val}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  Feature Store — Online Service Status")
    if user or database or schema:
        parts = []
        if user:
            parts.append(f"User: {user}")
        if database:
            parts.append(f"DB: {database}")
        if schema:
            parts.append(f"Schema: {schema}")
        lines.append("  " + "  |  ".join(parts))
    lines.append("=" * 80)

    # Umbrella status block.  The underlying parsed dict still calls
    # this ``runtime_id`` (matching the raw SYSTEM$ payload key) but
    # operators interact with this surface as "the online service",
    # which is also the command name (``snow feature
    # online-service``).  ``Message`` and ``Runtime ID`` move under
    # the verbose flag — the compact view shows only the heading row
    # so the umbrella state clusters tightly with the three
    # sub-component summaries below.
    state = status.get("status", "UNKNOWN")
    running = state == "RUNNING"
    _heading(_icon(running), "Online Service", state)
    if verbose:
        _kv("Message", status.get("message", ""))
        _kv("Runtime ID", status.get("runtime_id", "?"))
        lines.append("")

    # Compute Pool / Postgres / Service
    #
    # The trailing blank line under each component heading is part of
    # the verbose layout — it separates the per-component detail
    # blocks visually.  In compact mode there are no detail rows under
    # the headings, so emitting the blank line just opens a vertical
    # gap between the three single-line summaries.  We gate the blank
    # line on ``verbose`` here so the compact view renders the three
    # heading rows back-to-back and inserts a single blank line before
    # the Endpoints block below.
    cp = status.get("compute_pool", {})
    if cp:
        cp_ok = cp.get("status", "") in ("ACTIVE", "IDLE", "RESIZING")
        _heading(_icon(cp_ok), "Compute Pool", cp.get("status", "?"))
        if verbose:
            _kv("Name", cp.get("name", "?"))
            _kv("Family", cp.get("instance_family", "?"))
            _kv("Nodes", f"{cp.get('active_nodes', '?')} active / {cp.get('total_nodes', '?')} total")
            _kv("Min/Max", f"{cp.get('min_nodes', '?')} / {cp.get('max_nodes', '?')}")
            _kv("Auto-suspend", f"{cp.get('auto_suspend_secs', '?')}s")
            lines.append("")

    pg = status.get("postgres", {})
    if pg:
        pg_ok = pg.get("status", "") == "READY"
        _heading(_icon(pg_ok), "Postgres", pg.get("status", "?"))
        if verbose:
            _kv("Name", pg.get("name", "?"))
            _kv("Family", pg.get("compute_family", "?"))
            _kv("Version", f"PG {pg.get('postgres_version', '?')}")
            _kv("Storage", f"{pg.get('storage_gb', '?')} GB")
            _kv("Host", pg.get("host", "?"))
            lines.append("")

    svc = status.get("service", {})
    if svc:
        svc_ok = svc.get("status", "") in ("RUNNING", "READY")
        _heading(_icon(svc_ok), "Service", svc.get("status", "?"))
        if verbose:
            _kv("Name", svc.get("name", "?"))
            _kv("Image", svc.get("image_version", "?"))
            _kv("Pool", svc.get("compute_pool_name", "?"))
            _kv(
                "Instances",
                (
                    f"{svc.get('current_instances', '?')} running "
                    f"(min={svc.get('min_instances', '?')}, max={svc.get('max_instances', '?')})"
                ),
            )
            if svc.get("is_upgrading"):
                _kv("Upgrading", "YES")
            for inst in svc.get("instances", []):
                lines.append(
                    f"       [{inst.get('instance_id', '?')}]  "
                    f"{inst.get('status', '?'):<8s}  "
                    f"ip={inst.get('ip', '?'):<16s}  "
                    f"started={inst.get('start_time', '?')}"
                )
            lines.append("")

    # Bridge blank line between the (possibly compact) component
    # cluster and the Endpoints block.  Verbose mode already ended the
    # last component block with its own trailing blank; compact mode
    # needs us to add one here so Endpoints doesn't crash into the
    # Service heading.
    if not verbose and (cp or pg or svc):
        lines.append("")

    # Endpoints — render every URL variant the runtime supplied
    # (public, PrivateLink, SPCS-internal).  Only present variants
    # appear; the labels match the keys returned by snowml-core's
    # OnlineServiceEndpoint dataclass.
    endpoints = status.get("endpoints", [])
    if endpoints:
        lines.append("  Endpoints:")
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            name = ep.get("name", "?")
            for label, key in (
                (name, "url"),
                ("privatelink", "privatelink_url"),
                ("internal", "internal_url"),
            ):
                value = ep.get(key)
                if not value:
                    continue
                if not str(value).startswith("https://"):
                    value = f"https://{value}"
                if key == "url":
                    lines.append(f"     {label}: {value}")
                else:
                    lines.append(f"       {label}: {value}")
        lines.append("")

    # Network Rules — diagnostic-only block gated behind the verbose flag.
    rules = status.get("network_rules", [])
    if verbose:
        if rules:
            lines.append("  Network Rules:")
            for rule in rules:
                _heading(_icon(True), rule.get("name", "?"), rule.get("mode", "?"))
                _kv("Type", rule.get("type", "?"))
                _kv("Purpose", rule.get("purpose", "?"))
                values = rule.get("value_list", "")
                if values:
                    for v in values.split(","):
                        lines.append(f"       - {v.strip()}")
            lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rich describe display
# ---------------------------------------------------------------------------


def format_describe_display(
    fv_name: str,
    version: str,
    database: str,
    schema: str,
    oft_name: str,
    entities: list[str],
    describe_rows: list[dict[str, Any]],
    show_row: Optional[dict[str, Any]] = None,
    spec: Optional[dict[str, Any]] = None,
    examples: Optional[list[str]] = None,
) -> str:
    """Format a rich describe display for a feature view.

    Shows all associated objects, reference names, columns, and curl examples.

    Args:
        fv_name: Feature view name (lowercase).
        version: Version string.
        database: Snowflake database.
        schema: Snowflake schema.
        oft_name: Online Feature Table name (e.g. ``USER_EVENT_FEATURES$V1$ONLINE``).
        entities: List of entity (PK) column names.
        describe_rows: Rows from DESCRIBE ONLINE FEATURE TABLE.
        show_row: Optional row from SHOW ONLINE FEATURE TABLES for this OFT.
        spec: Optional parsed YAML spec dict.
        examples: Optional list of curl example strings.

    Returns:
        Formatted multi-line string.
    """
    LBL = 22
    lines: list[str] = []

    def _kv(key: str, val: object) -> None:
        lines.append(f"  {key + ':':<{LBL}s} {val}")

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  Feature View: {fv_name}")
    lines.append("=" * 80)

    # --- Identifiers ---
    lines.append("")
    lines.append("  Identifiers:")
    _kv("  Feature View", fv_name)
    _kv("  Version", version)
    _kv("  Database", database)
    _kv("  Schema", schema)
    _kv("  REST API name", fv_name)
    _kv("  SQL (OFT)", f"{database}.{schema}.{oft_name}")
    lines.append("")

    # --- Entities ---
    lines.append("  Entities (Primary Key):")
    for ent in entities:
        lines.append(f"    - {ent}")
    lines.append("")

    # --- Source ---
    if spec:
        sources = spec.get("sources", [])
        if sources:
            src = sources[0] if isinstance(sources[0], dict) else {}
            src_name = src.get("name", "?")
            src_type = src.get("source_type", "?")
            lines.append("  Source:")
            _kv("  Name", src_name)
            _kv("  Type", src_type)
            src_cols = src.get("columns", [])
            if src_cols:
                lines.append("    Columns (ingest schema):")
                for col in src_cols:
                    cname = col.get("name", "?") if isinstance(col, dict) else "?"
                    ctype = col.get("type", "?") if isinstance(col, dict) else "?"
                    lines.append(f"      - {cname}: {ctype}")
            lines.append("")

    # --- Features (output columns) ---
    if spec and spec.get("features"):
        lines.append("  Features (query output):")
        for feat in spec["features"]:
            oc = feat.get("output_column", {})
            oname = oc.get("name", "?")
            otype = oc.get("type", "?")
            func = feat.get("function", "")
            window = feat.get("window", "")
            detail = f" ({func}" if func else ""
            if window:
                detail += f", {window})"
            elif detail:
                detail += ")"
            lines.append(f"    - {oname}: {otype}{detail}")
        lines.append("")

    # --- UDF ---
    if spec and spec.get("udf"):
        udf = spec["udf"]
        lines.append("  UDF:")
        _kv("  Name", udf.get("name", "?"))
        _kv("  Engine", udf.get("engine", "?"))
        lines.append("")

    # --- OFT Columns (from DESCRIBE) ---
    if describe_rows:
        lines.append("  OFT Columns (DESCRIBE):")
        for col in describe_rows:
            cname = col.get("name", col.get("NAME", "?"))
            ctype = col.get("type", col.get("TYPE", "?"))
            is_pk = False
            for pk_key in ("primary key", "PRIMARY KEY", "primary_key"):
                val = col.get(pk_key, "")
                if val and str(val).upper() in ("Y", "YES", "TRUE", "1"):
                    is_pk = True
                    break
            pk_marker = " [PK]" if is_pk else ""
            lines.append(f"    - {cname}: {ctype}{pk_marker}")
        lines.append("")

    # --- SHOW metadata ---
    if show_row:
        lines.append("  Metadata:")
        for key in ("created_on", "scheduling_state", "owner", "target_lag"):
            val = show_row.get(key, show_row.get(key.upper()))
            if val is not None:
                _kv(f"  {key}", val)
        lines.append("")

    # --- Curl examples ---
    if examples:
        lines.append("-" * 80)
        for ex in examples:
            lines.append(ex)
            lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample value generation for describe examples
# ---------------------------------------------------------------------------

# Map SQL type prefixes → sample values for example curl commands.
# Longer prefixes first so TIMESTAMP_NTZ matches before TIMESTAMP.
_SQL_TYPE_SAMPLES: list[tuple[str, Any]] = [
    ("VARCHAR", "example_text"),
    ("STRING", "example_text"),
    ("TEXT", "example_text"),
    ("NUMBER", 42),
    ("BIGINT", 123),
    ("INT", 123),
    ("FLOAT", 3.14),
    ("DOUBLE", 3.14),
    ("REAL", 3.14),
    ("BOOLEAN", True),
    ("TIMESTAMP_NTZ", "2025-01-15T12:00:00"),
    ("TIMESTAMP_LTZ", "2025-01-15T12:00:00Z"),
    ("TIMESTAMP_TZ", "2025-01-15T12:00:00Z"),
    ("TIMESTAMP", "2025-01-15T12:00:00Z"),
    ("DATE", "2025-01-15"),
    ("DECIMAL", 99.99),
]


def _sample_value_for_type(sql_type: str) -> Any:
    """Return a representative sample value for a SQL type string.

    Args:
        sql_type: SQL type string from DESCRIBE (e.g. ``"VARCHAR(100)"``).

    Returns:
        A sample value suitable for JSON serialization.
    """
    upper = sql_type.upper().strip()
    for prefix, val in _SQL_TYPE_SAMPLES:
        if upper.startswith(prefix):
            return val
    return "sample_value"


def _join_key_names(entities: list[Any]) -> list[str]:
    """Flatten FeatureView ``entities`` entries (strings or Entity dicts) to join-key names."""
    names: list[str] = []
    for ent in entities:
        if isinstance(ent, str):
            names.append(ent)
        elif isinstance(ent, dict):
            keys = [jk["name"] for jk in (ent.get("join_keys") or []) if isinstance(jk, dict) and jk.get("name")]
            names.extend(keys or ([ent["name"]] if ent.get("name") else []))
    return names


def build_describe_examples(
    fv_name: str,
    version: str,
    source_name: str,
    describe_rows: list[dict[str, Any]],
    ingest_url: Optional[str],
    query_url: Optional[str],
    spec: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Build example curl commands for ingest and query.

    When a *spec* dict (parsed YAML) is provided, examples use the UDF
    output columns for ingest and the entity columns for query — these
    are the actual API-level column names.  Without a spec, falls back
    to DESCRIBE rows (less accurate).

    Auth uses the SPCS ``Snowflake Token="<PAT>"`` format.

    Args:
        fv_name: Feature view name (user-facing, lowercase).
        version: Feature view version (e.g. ``"v1"``).
        source_name: Source name for ingest (e.g. ``"user_events"``).
        describe_rows: Rows from DESCRIBE ONLINE FEATURE TABLE (fallback).
        ingest_url: Base ingest endpoint URL, or ``None``.
        query_url: Base query endpoint URL, or ``None``.
        spec: Optional parsed YAML spec dict for the feature view.

    Returns:
        List of example strings (may be empty if no endpoints available).
    """
    # Determine ingest columns and entity columns from spec or DESCRIBE
    ingest_cols: list[dict[str, str]] = []
    entity_cols: list[dict[str, str]] = []

    if spec:
        # Use source columns for ingest (raw input schema, not UDF output)
        sources = spec.get("sources", [])
        if sources and isinstance(sources, list):
            src = sources[0] if isinstance(sources[0], dict) else {}
            src_cols = src.get("columns", [])
            if src_cols:
                for col in src_cols:
                    ingest_cols.append(
                        {
                            "name": col.get("name", ""),
                            "type": col.get("type", "StringType"),
                        }
                    )

        # Fall back to UDF output_columns if source has no columns
        if not ingest_cols:
            udf = spec.get("udf", {})
            if udf and isinstance(udf, dict):
                for col in udf.get("output_columns", []):
                    ingest_cols.append(
                        {
                            "name": col.get("name", ""),
                            "type": col.get("type", "StringType"),
                        }
                    )

        entity_names = _join_key_names(spec.get("entities", []))

        # Fall back to features + entity columns if still empty
        if not ingest_cols:
            for name in entity_names:
                ingest_cols.append({"name": name, "type": "StringType"})
            for feat in spec.get("features", []):
                src = feat.get("source_column", {})
                if src:
                    ingest_cols.append(
                        {
                            "name": src.get("name", ""),
                            "type": src.get("type", "StringType"),
                        }
                    )

        # Entity columns for query
        for name in entity_names:
            entity_cols.append({"name": name, "type": "StringType"})
    else:
        # Fallback: use DESCRIBE rows
        for col in describe_rows:
            col_name = col.get("name", col.get("NAME", ""))
            col_type = col.get("type", col.get("TYPE", "VARCHAR"))
            is_pk = False
            for pk_key in ("primary key", "PRIMARY KEY", "primary_key"):
                val = col.get(pk_key, "")
                if val and str(val).upper() in ("Y", "YES", "TRUE", "1"):
                    is_pk = True
                    break
            ingest_cols.append({"name": col_name, "type": col_type})
            if is_pk:
                entity_cols.append({"name": col_name, "type": col_type})

    # Build sample ingest record
    sample_record: dict[str, Any] = {}
    for col in ingest_cols:
        sample_record[col["name"]] = _sample_for_fs_type(col["type"])

    # Build entity keys for query
    sample_entity: dict[str, Any] = {}
    for col in entity_cols:
        sample_entity[col["name"]] = _sample_for_fs_type(col["type"])

    examples: list[str] = []

    if ingest_url:
        ingest_endpoint = ingest_url.rstrip("/") + "/api/v1/ingest"
        body = json.dumps({"records": {source_name: [sample_record]}}, indent=2)
        examples.append(
            f"# Ingest example\n"
            f"curl -X POST '{ingest_endpoint}' \\\n"
            f'  -H "Authorization: Snowflake Token=\\"$SNOWFLAKE_PAT\\"" \\\n'
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{body}'"
        )

    if query_url:
        query_endpoint = query_url.rstrip("/") + "/api/v1/query"
        body = json.dumps(
            {
                "name": fv_name,
                "version": version,
                "object_type": "feature_view",
                "request_rows": [{"entity": sample_entity}],
            },
            indent=2,
        )
        examples.append(
            f"# Query example\n"
            f"curl -X POST '{query_endpoint}' \\\n"
            f'  -H "Authorization: Snowflake Token=\\"$SNOWFLAKE_PAT\\"" \\\n'
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{body}'"
        )

    return examples


# FSBaseType → sample values (for spec-derived examples)
_FS_TYPE_SAMPLES: dict[str, Any] = {
    "StringType": "example_text",
    "LongType": 123,
    "DoubleType": 3.14,
    "DecimalType": 99.99,
    "BooleanType": True,
    "TimestampType": "2025-01-15T12:00:00Z",
}


def _sample_for_fs_type(fs_type: str) -> Any:
    """Return a sample value for an FSBaseType or SQL type string."""
    # Try FSBaseType first
    val = _FS_TYPE_SAMPLES.get(fs_type)
    if val is not None:
        return val
    # Fall back to SQL type matching
    return _sample_value_for_type(fs_type)
