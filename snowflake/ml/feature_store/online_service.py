"""Online Service management and online feature read/write operations.

Provides helpers for creating, querying status, and dropping the Online Service,
as well as reading online features and ingesting records for streaming feature views.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils

logger = logging.getLogger(__name__)

_QUERY_API_REL_PATH = "api/v1/query"
_INGEST_API_REL_PATH = "api/v1/ingest"
_MAX_QUERY_REQUEST_ROWS = 10

_ONLINE_SERVICE_NOT_READY_USER_MESSAGE = (
    "Online Service is not RUNNING or the query endpoint is not available. "
    "Online reads for Postgres-backed online feature tables require a running Online Service with a query endpoint. "
    "Use create_online_service and poll get_online_service_status() until status is RUNNING, "
    "then call get_feature_view(...) again."
)


@dataclass(frozen=True)
class OnlineServiceEndpoint:
    name: str
    url: str


@dataclass(frozen=True)
class OnlineServiceStatus:
    status: str
    message: Optional[str] = None
    endpoints: tuple[OnlineServiceEndpoint, ...] = field(default_factory=tuple)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass(frozen=True)
class OnlineServiceResult:
    status: str
    message: str


def online_service_not_ready_message() -> str:
    return _ONLINE_SERVICE_NOT_READY_USER_MESSAGE


def _parse_status_payload(data: dict[str, Any]) -> OnlineServiceStatus:
    """Parse Online Service status JSON.

    Only ``status``, ``message``, ``endpoints`` (each item: ``name``, ``url``),
    ``created_at``, and ``updated_at`` are read. Any other top-level or endpoint
    keys are ignored (Snowflake may return a superset on some accounts).

    Args:
        data: Parsed JSON object from the system function.

    Returns:
        OnlineServiceStatus built from ``data``.
    """
    endpoints_raw = data.get("endpoints") or []
    endpoints: list[OnlineServiceEndpoint] = []
    if isinstance(endpoints_raw, list):
        for item in endpoints_raw:
            if not isinstance(item, dict):
                continue
            n = item.get("name")
            u = item.get("url")
            if isinstance(n, str) and isinstance(u, str):
                endpoints.append(OnlineServiceEndpoint(name=n, url=u))
    return OnlineServiceStatus(
        status=str(data.get("status", "")),
        message=data.get("message") if isinstance(data.get("message"), str) else None,
        endpoints=tuple(endpoints),
        created_at=str(data["created_at"]) if data.get("created_at") is not None else None,
        updated_at=str(data["updated_at"]) if data.get("updated_at") is not None else None,
    )


def _parse_mutation_payload(data: dict[str, Any]) -> OnlineServiceResult:
    """Parse Online Service mutation JSON; only ``status`` and ``message`` are used."""
    return OnlineServiceResult(
        status=str(data.get("status", "")),
        message=str(data.get("message", "")),
    )


def _first_cell(rows: list[Any]) -> str:
    if not rows:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError("Expected a row from Online Service."),
        )
    return str(rows[0][0])


def _escaped_feature_store_locator(database: SqlIdentifier, schema: SqlIdentifier) -> str:
    """Single-quoted SQL literal content for SYSTEM$ first argument (same form as FeatureStore full_schema_path)."""
    locator = f"{database}.{schema}"
    return str(snowpark_utils.escape_single_quotes(locator))  # type: ignore[no-untyped-call]


def _call_system_function(
    session: Session,
    sql: str,
    *,
    operation: str,
    statement_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Execute a SYSTEM$ SQL function and return its parsed JSON dict."""
    raw = _first_cell(session.sql(sql).collect(statement_params=statement_params))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError(f"Invalid JSON from Online Service {operation} response: {raw!r}"),
        ) from e
    if not isinstance(data, dict):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError(f"Unexpected {operation} payload type: {type(data)}"),
        )
    return data


def fetch_online_service_status(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceStatus:
    loc = _escaped_feature_store_locator(database, schema)
    data = _call_system_function(
        session,
        f"SELECT SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('{loc}')",
        operation="status",
        statement_params=statement_params,
    )
    st = _parse_status_payload(data)
    return st


def get_online_service_status(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceStatus:
    st = fetch_online_service_status(session, database, schema, statement_params=statement_params)
    if st.status == "ERROR":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=RuntimeError(st.message or "Online Service status error."),
        )
    return st


def create_online_service(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    producer_role: str,
    consumer_role: str,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceResult:
    if not producer_role.strip() or not consumer_role.strip():
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("producer_role and consumer_role must be non-empty."),
        )
    if producer_role.strip() == consumer_role.strip():
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("producer_role and consumer_role must be different roles."),
        )
    payload = json.dumps(
        {
            "roles": {
                "producer_role_name": producer_role.strip(),
                "consumer_role_name": consumer_role.strip(),
            }
        }
    )
    properties_escaped = snowpark_utils.escape_single_quotes(payload)  # type: ignore[no-untyped-call]
    loc = _escaped_feature_store_locator(database, schema)
    data = _call_system_function(
        session,
        f"SELECT SYSTEM$CREATE_FEATURE_STORE_ONLINE_SERVICE('{loc}', '{properties_escaped}')",
        operation="create",
        statement_params=statement_params,
    )
    result = _parse_mutation_payload(data)
    if result.status != "SUCCESS":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=RuntimeError(result.message or "Online Service creation failed."),
        )
    return OnlineServiceResult(
        status=result.status,
        message="Online Service created. Poll get_online_service_status() until RUNNING.",
    )


def drop_online_service(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceResult:
    loc = _escaped_feature_store_locator(database, schema)
    data = _call_system_function(
        session,
        f"SELECT SYSTEM$DROP_FEATURE_STORE_ONLINE_SERVICE('{loc}')",
        operation="drop",
        statement_params=statement_params,
    )
    result = _parse_mutation_payload(data)
    if result.status != "SUCCESS":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=RuntimeError(result.message or "Online Service drop failed."),
        )
    return OnlineServiceResult(
        status=result.status,
        message="Online Service dropped.",
    )


def endpoint_url(status: OnlineServiceStatus, name: str) -> Optional[str]:
    for ep in status.endpoints:
        if ep.name == name:
            return ep.url
    return None


def _query_api_url(query_base_url: str) -> str:
    """Append ``api/v1/query`` to the Online Service ``query`` endpoint URL from status JSON."""
    base = query_base_url.strip().rstrip("/") + "/"
    return urllib.parse.urljoin(base, _QUERY_API_REL_PATH)


def _ingest_api_url(ingest_base_url: str) -> str:
    """Append ``api/v1/ingest`` to the Online Service ``ingest`` endpoint URL from status JSON."""
    base = ingest_base_url.strip().rstrip("/") + "/"
    return urllib.parse.urljoin(base, _INGEST_API_REL_PATH)


def _online_service_query_api_pat_from_env() -> str:
    """PAT for ``Authorization: Snowflake Token="..."`` on the Online Service REST APIs (Query + Ingest).

    Only ``SNOWFLAKE_PAT`` is supported so behavior is explicit; create a PAT in Snowflake and set this env var.

    Returns:
        Programmatic access token from ``SNOWFLAKE_PAT``.

    Raises:
        SnowflakeMLException: If ``SNOWFLAKE_PAT`` is unset or empty.
    """
    pat = os.environ.get("SNOWFLAKE_PAT", "").strip()
    if pat:
        return pat
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INVALID_ARGUMENT,
        original_exception=RuntimeError(
            "Online Service requires a Snowflake Programmatic Access Token (PAT). "
            "Set the SNOWFLAKE_PAT environment variable "
            "(export in your shell or set os.environ in your notebook)."
        ),
    )


def _session_rest_auth_headers(session: Session) -> dict[str, str]:
    """Build HTTP headers for the Online Service REST APIs (Query + Ingest); ``session`` is reserved for future use."""
    _ = session
    token = _online_service_query_api_pat_from_env()
    return {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _json_serialize_value(value: Any) -> Any:
    """JSON-serialize a single field value for Online Service request bodies."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, bytes):
        raise TypeError(f"bytes values are not supported in Online Service requests: {value!r}")
    return str(value)


_MAX_RESPONSE_BYTES = 1024 * 1024  # 1 MB cap on response reads to prevent unbounded memory use.


def _http_post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
) -> tuple[int, bytes]:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, e.read(_MAX_RESPONSE_BYTES)
    except (urllib.error.URLError, OSError) as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWFLAKE_API_ERROR,
            original_exception=RuntimeError(f"Online Service is unreachable: {e}"),
        ) from e
    try:
        return resp.getcode(), resp.read(_MAX_RESPONSE_BYTES)
    finally:
        resp.close()


def _try_extract_request_id_from_json_payload(parsed: Any) -> Optional[str]:
    """Best-effort ``request_id`` / ``requestId`` from Query or Ingest API JSON (top-level or under ``error``)."""
    if not isinstance(parsed, dict):
        return None
    for key in ("request_id", "requestId"):
        v = parsed.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    err = parsed.get("error")
    if isinstance(err, dict):
        for key in ("request_id", "requestId"):
            v = err.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _raise_online_service_http_error(status: int, body: bytes) -> None:
    """Raise for failed Online Service HTTP calls; include server ``request_id`` when present."""
    text = body.decode("utf-8", errors="replace")
    message = text
    err_code = ""
    server_request_id: Optional[str] = None
    try:
        parsed = json.loads(text)
        server_request_id = _try_extract_request_id_from_json_payload(parsed)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                err_code = str(err.get("code", "") or "")
                if isinstance(err.get("message"), str):
                    message = err["message"]
    except json.JSONDecodeError:
        pass

    preview = text if len(text) <= 800 else text[:800] + "..."
    logger.warning(
        "Online Service HTTP error: status=%s request_id=%s body=%r",
        status,
        server_request_id,
        preview,
    )

    if status == 404 or err_code == "NOT_FOUND":
        ec = error_codes.NOT_FOUND
    elif status in (400, 422) or err_code == "INVALID_ARGUMENT":
        ec = error_codes.INVALID_ARGUMENT
    elif status in (401, 403):
        ec = error_codes.SNOWML_READ_FAILED
    elif status == 429:
        ec = error_codes.INTERNAL_SNOWFLAKE_API_ERROR
    else:
        ec = error_codes.INTERNAL_SNOWFLAKE_API_ERROR
    rid_suffix = f" [request_id: {server_request_id}]" if server_request_id else ""
    if status in (401, 403):
        message = f"Authentication/authorization failed: {message}"
    elif status == 429:
        message = f"Rate limited by Online Service: {message}"
    raise snowml_exceptions.SnowflakeMLException(
        error_code=ec,
        original_exception=RuntimeError(f"Online Service error (HTTP {status}): {message}{rid_suffix}"),
    )


def _assert_online_service_running(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    endpoint_name: str,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceStatus:
    """Assert the Online Service is RUNNING and has the named endpoint."""
    st = fetch_online_service_status(session, database, schema, statement_params=statement_params)
    if st.status == "ERROR":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=RuntimeError(st.message or "Online Service status error."),
        )
    if st.status == "NOT_FOUND":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "No Online Service for this schema. Call create_online_service(...) first, "
                "then poll get_online_service_status() until RUNNING."
            ),
        )
    if st.status != "RUNNING":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Online Service is not RUNNING (current status: {st.status}). "
                "Poll get_online_service_status() until RUNNING."
            ),
        )
    if not endpoint_url(st, endpoint_name):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Online Service is RUNNING but no {endpoint_name} endpoint was returned. "
                "Wait for endpoints to be available and try again."
            ),
        )
    return st


def assert_online_service_running_with_query_endpoint(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceStatus:
    return _assert_online_service_running(session, database, schema, "query", statement_params=statement_params)


def assert_online_service_running_with_ingest_endpoint(
    session: Session,
    database: SqlIdentifier,
    schema: SqlIdentifier,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> OnlineServiceStatus:
    return _assert_online_service_running(session, database, schema, "ingest", statement_params=statement_params)


def _parse_ingest_response_count(data: dict[str, Any], *, stream_source_name: str) -> int:
    """Extract accepted record count from Ingest API JSON (``IngestResponse``).

    Used for HTTP **200** and **207** responses; same schema per Ingest OpenAPI.

    **Online Service (ofs_quake)** shape::

        {
          "request_id": "...",
          "status": "success",
          "sources": {
            "<STREAM_SOURCE_NAME>": {
              "records": {"total": N, "failed": M, ...},
              ...
            }
          }
        }

    Older/alternate handlers may return a flat integer field; those are still accepted.

    Args:
        data: Parsed Ingest API JSON body.
        stream_source_name: Key under ``sources`` for this stream (or sole source when only one entry).

    Returns:
        Accepted record count (``total - failed`` when nested, else top-level count fields).

    Raises:
        SnowflakeMLException: If no recognized count is present in the payload.
    """
    sources = data.get("sources")
    if isinstance(sources, dict) and stream_source_name:
        src = sources.get(stream_source_name)
        if src is None and len(sources) == 1:
            src = next(iter(sources.values()))
        if isinstance(src, dict):
            rec = src.get("records")
            if isinstance(rec, dict):
                total = rec.get("total")
                failed = rec.get("failed", 0)
                if isinstance(total, int) and not isinstance(total, bool):
                    f = failed if isinstance(failed, int) and not isinstance(failed, bool) else 0
                    return max(0, total - f)

    for key in ("ingested_count", "records_ingested", "count", "accepted", "num_records"):
        v = data.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    logger.debug(
        "Online Service ingest response missing recognized count for stream_source=%r: keys=%r body=%r",
        stream_source_name,
        list(data.keys()),
        data,
    )
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INTERNAL_SNOWML_ERROR,
        original_exception=RuntimeError(
            f"Online Service returned an unexpected response for stream source {stream_source_name!r}."
        ),
    )


def _raise_on_retryable_ingest_errors(data: dict[str, Any], stream_source_name: str, request_id: Any) -> None:
    """Raise on HTTP 207 responses that contain retryable record or feature_view errors."""
    sources = data.get("sources")
    if not isinstance(sources, dict):
        return
    src = sources.get(stream_source_name)
    if src is None and len(sources) == 1:
        src = next(iter(sources.values()))
    if not isinstance(src, dict):
        return
    errors: list[str] = []
    rec = src.get("records")
    if isinstance(rec, dict) and rec.get("failed", 0) > 0:
        errors.append(f"records: {rec}")
    fvs = src.get("feature_views")
    if isinstance(fvs, dict) and fvs.get("failed", 0) > 0:
        for r in fvs.get("results", []):
            if isinstance(r, dict) and r.get("status") == "failed":
                err = r.get("error", {})
                errors.append(f"feature_view {r.get('name')}/{r.get('version')}: {err.get('message', err)}")
    if errors:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError(f"Ingest partial failure (request_id={request_id}): {'; '.join(errors)}"),
        )


def stream_ingest_records(
    session: Session,
    ingest_base_url: str,
    stream_source_name: str,
    records: list[dict[str, Any]],
    *,
    timeout_sec: float = 120.0,
) -> int:
    """Send records to the Online Service for a given stream source.

    Requires ``SNOWFLAKE_PAT`` environment variable.

    Args:
        session: Snowpark session (reserved for future use).
        ingest_base_url: Base URL for the ingest endpoint from Online Service status.
        stream_source_name: Registered stream source name.
        records: Non-empty list of row dicts (keys must match the stream source schema).
        timeout_sec: Request timeout in seconds.

    Returns:
        Count of accepted records. On partial success, this may be less than the
        number of rows sent.

    Raises:
        SnowflakeMLException: On request failure, invalid response, or if the response
            lacks a recognizable accepted-record count.
    """
    if not records:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("records must be non-empty for stream ingest."),
        )
    resolved = _ingest_api_url(ingest_base_url)
    logger.debug(
        "Ingest API: url=%r stream_source=%r num_records=%d",
        resolved,
        stream_source_name,
        len(records),
    )
    serializable = [{k: _json_serialize_value(v) for k, v in row.items()} for row in records]
    body: dict[str, Any] = {"records": {stream_source_name: serializable}}
    headers = _session_rest_auth_headers(session)
    status, raw = _http_post_json(resolved, headers, body, timeout_sec)
    # 200 = full success, 207 = partial success.
    if status not in (200, 207):
        _raise_online_service_http_error(status, raw)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("Invalid response from Online Service."),
        ) from e
    if not isinstance(data, dict):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("Unexpected response format from Online Service."),
        )
    rid = data.get("request_id")
    logger.info(
        "Feature Store Ingest API response request_id=%r http_status=%s stream_source=%r",
        rid,
        status,
        stream_source_name,
    )
    if status == 207 or data.get("status") == "partial_success":
        logger.warning(
            "Feature Store Ingest API partial success: request_id=%r stream_source=%r sources=%r",
            rid,
            stream_source_name,
            data.get("sources"),
        )
        _raise_on_retryable_ingest_errors(data, stream_source_name, rid)
    return _parse_ingest_response_count(data, stream_source_name=stream_source_name)


def _extract_query_api_data_type_raw(item: dict[str, Any]) -> Optional[str]:
    """Read type string from Query API ``metadata.features[]`` (Snowflake REST / SQL API ``type`` field)."""
    for key in ("data_type", "dataType", "type", "snowflake_type", "snowflakeType"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            inner = v.get("type") or v.get("name")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return None


def _snowpark_type_from_sql_style_string(raw: str) -> Optional[Any]:
    """Map SQL-style type strings (e.g. ``FLOAT``, ``TIMESTAMP_NTZ(9)``) to Snowpark types."""
    from snowflake.snowpark.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        LongType,
        StringType,
        TimestampType,
        TimeType,
    )

    s = raw.strip().upper()
    if not s:
        return None
    base = s.split("(")[0].strip()

    if base in ("FLOAT", "FLOAT4", "FLOAT8", "DOUBLE", "DOUBLE PRECISION", "REAL", "FLOAT64"):
        return DoubleType()
    if base in ("BOOLEAN", "BOOL"):
        return BooleanType()
    if base.startswith("TIMESTAMP"):
        return TimestampType()
    if base == "DATE":
        return DateType()
    if base == "TIME":
        return TimeType()
    if base in ("VARCHAR", "CHAR", "CHARACTER", "STRING", "TEXT"):
        return StringType()
    if base in ("BINARY", "VARBINARY"):
        return BinaryType()
    if base in ("BIGINT", "LONG"):
        return LongType()
    if base in ("INT", "INTEGER", "SMALLINT", "TINYINT", "BYTEINT", "NUMBER", "DECIMAL", "NUMERIC"):
        return DecimalType(38, 0)
    return None


def _snowflake_query_api_type_to_snowpark(raw: str, item: dict[str, Any]) -> Optional[Any]:
    """Map Snowflake REST / SQL API ``rowType``-style metadata to Snowpark types.

    Feature Store Query service emits the same shape as SQL API ``rowType`` (see
    https://docs.snowflake.com/en/developer-guide/sql-api/handling-responses ).
    Server-side ``FSType.RESTDataType()`` maps roughly as:

    - ``DoubleType`` → ``{"type": "real"}``
    - ``LongType`` → ``{"type": "fixed", "precision": 38, "scale": 0}``
    - ``DecimalType(p,s)`` → ``{"type": "fixed", "precision": p, "scale": s}``
    - ``StringType(n)`` → ``{"type": "text", "length": n}``
    - ``BooleanType`` → ``{"type": "boolean"}``
    - ``TimestampType`` → ``{"type": "timestamp_ntz"}`` (``RESTTypeTimestampNTZ``)

    Unknown server types fall back to ``text`` on the service; this client mirrors
    that for unlisted tokens by deferring to SQL-style parsing, then view schema.

    Args:
        raw: Type string from metadata (REST token or SQL-style text after strip).
        item: Metadata dict for the column (e.g. ``precision``, ``scale``, ``length``).

    Returns:
        Matching Snowpark type, or ``None`` if ``raw`` is empty after stripping.
    """
    from snowflake.snowpark.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        StringType,
        TimestampType,
        TimeType,
        VariantType,
    )

    stripped = raw.strip()
    if not stripped:
        return None

    rest = stripped.lower()
    if rest == "fixed":
        prec = item.get("precision")
        scale = item.get("scale")
        if isinstance(prec, int) and prec > 0:
            sc = scale if isinstance(scale, int) and scale >= 0 else 0
            return DecimalType(prec, sc)
        return DecimalType(38, 0)
    if rest in ("real", "double", "double precision"):
        return DoubleType()
    if rest == "float":
        return FloatType()
    if rest == "text":
        length = item.get("length")
        if isinstance(length, int) and length > 0:
            return StringType(length)
        return StringType()
    if rest in ("boolean", "bool"):
        return BooleanType()
    if rest == "date":
        return DateType()
    if rest == "time":
        return TimeType()
    if rest in ("binary", "varbinary"):
        return BinaryType()
    if rest in ("variant", "object", "array"):
        return VariantType()
    if rest.startswith("timestamp"):
        return TimestampType()

    return _snowpark_type_from_sql_style_string(stripped)


def _snowpark_type_from_query_metadata_item(item: dict[str, Any]) -> Optional[Any]:
    raw = _extract_query_api_data_type_raw(item)
    if not raw:
        return None
    return _snowflake_query_api_type_to_snowpark(raw, item)


def _snowpark_types_from_query_metadata_features(meta_features: list[dict[str, Any]]) -> dict[str, Any]:
    """Map feature name -> Snowpark type from Query API metadata only; unknown -> StringType."""
    from snowflake.snowpark.types import StringType

    out: dict[str, Any] = {}
    for item in meta_features:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        name = item["name"]
        sp = _snowpark_type_from_query_metadata_item(item)
        out[name] = sp if sp is not None else StringType()
    return out


def _resolve_output_feature_names(
    names_from_server: list[str],
    feature_names: Optional[list[str]],
) -> list[str]:
    """Feature columns to return (server spelling). ``feature_names`` must match case-insensitively."""
    if not feature_names:
        return list(names_from_server)
    by_resolved: dict[str, str] = {}
    for n in names_from_server:
        by_resolved.setdefault(SqlIdentifier(n).resolved(), n)
    out: list[str] = []
    seen: set[str] = set()
    for f in feature_names:
        r = SqlIdentifier(f).resolved()
        if r not in by_resolved:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Requested feature {f!r} not found in Online Service; "
                    f"available features: {sorted(by_resolved.keys())!r}"
                ),
            )
        canon = by_resolved[r]
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _dedupe_feature_names_preserve_order(names: list[str]) -> list[str]:
    """Deduplicate feature names case-insensitively while preserving first-seen spelling."""
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        r = SqlIdentifier(n).resolved()
        if r in seen:
            continue
        seen.add(r)
        out.append(n)
    return out


def _postgres_online_query_result_schema(
    join_key_names: list[str],
    join_key_field_types: dict[str, Any],
    output_feature_names: list[str],
    feature_types_by_name: dict[str, Any],
) -> Any:
    """StructType for Postgres Query API rows (join keys + features)."""
    from snowflake.snowpark.types import StringType, StructField, StructType

    jfields = []
    for jk in join_key_names:
        dt = join_key_field_types.get(jk)
        jfields.append(StructField(jk, dt if dt is not None else StringType()))
    ffields = [StructField(fn, feature_types_by_name.get(fn, StringType())) for fn in output_feature_names]
    return StructType(jfields + ffields)


def _parse_query_batch_response(
    raw: bytes,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Parse and validate a single Query API batch response.

    Args:
        raw: Raw HTTP response body (UTF-8 JSON).
        batch_size: Expected number of entries in ``results``.

    Returns:
        ``(data, results, names_from_server, metadata_dict_items)``.

    Raises:
        SnowflakeMLException: If JSON is invalid, top-level shape is wrong, ``results`` length mismatches
            ``batch_size``, or feature metadata is missing.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("Invalid response from Online Service."),
        ) from e
    if not isinstance(data, dict):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("Unexpected response format from Online Service."),
        )

    results = data.get("results")
    if not isinstance(results, list) or len(results) != batch_size:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError(
                "Online Service returned unexpected number of results: "
                f"expected {batch_size} results, got {len(results) if isinstance(results, list) else type(results)}."
            ),
        )

    meta = data.get("metadata")
    meta_features = meta.get("features") if isinstance(meta, dict) else None
    if not isinstance(meta_features, list) or not meta_features:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("Online Service response missing feature metadata."),
        )
    names_from_server: list[str] = []
    dict_items = [it for it in meta_features if isinstance(it, dict)]
    for item in dict_items:
        if not isinstance(item.get("name"), str):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError("Online Service response contains invalid feature metadata."),
            )
        names_from_server.append(item["name"])

    return data, results, names_from_server, dict_items


def _extract_result_rows(
    results: list[dict[str, Any]],
    names_from_server: list[str],
    output_feature_names: list[str],
    join_key_names: list[str],
    batch: list[list[Any]],
) -> list[dict[str, Any]]:
    """Assemble row dicts from Query API results, combining join keys with feature values."""
    rows: list[dict[str, Any]] = []
    for i, res in enumerate(results):
        if not isinstance(res, dict):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError("Online Service returned an invalid result row."),
            )
        feats = res.get("features")
        if not isinstance(feats, list) or len(feats) != len(names_from_server):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError("Online Service returned mismatched feature values and metadata."),
            )
        name_to_val = dict(zip(names_from_server, feats))
        row_out: dict[str, Any] = {}
        for jk, jv in zip(join_key_names, batch[i]):
            row_out[jk] = jv
        for fname in output_feature_names:
            row_out[fname] = name_to_val.get(fname)
        rows.append(row_out)
    return rows


def read_postgres_online_features(
    session: Session,
    query_url: str,
    feature_view_name: str,
    feature_view_version: str,
    join_key_names: list[str],
    keys: list[list[Any]],
    feature_names: Optional[list[str]],
    *,
    join_key_field_types: dict[str, Any],
    timeout_sec: float = 120.0,
) -> tuple[list[dict[str, Any]], Any]:
    """Query Postgres-backed online features via the Online Service ``POST /api/v1/query`` endpoint.

    If ``feature_names`` is set, the JSON body includes ``features: [...]`` so the Query API can narrow
    the payload; if ``feature_names`` is ``None``, ``features`` is omitted and the server chooses its
    default columns. Returned row dicts and the Snowpark :class:`StructType` always reflect only the
    requested subset when ``feature_names`` is set (extra columns from the server are dropped). Values
    are aligned to ``metadata.features`` order. Join-key columns use ``join_key_field_types`` from the
    feature view (they are not part of ``metadata.features``).

    Args:
        session: Snowpark session (reserved for future use). Auth requires ``SNOWFLAKE_PAT``; see
            :func:`_online_service_query_api_pat_from_env`.
        query_url: Base URL for the ``query`` endpoint from Online Service status.
        feature_view_name: Logical feature view name.
        feature_view_version: Feature view version string (e.g. ``v1``).
        join_key_names: Flattened entity join key column names (same order as each row in ``keys``).
        keys: Non-empty list of entity rows; internally chunked to the API limit (10 rows per request).
        feature_names: If set, only these features are included (must appear in ``metadata.features``,
            matched case-insensitively); order follows this list. If ``None``, all features from
            ``metadata.features`` are included, in response order.
        join_key_field_types: Snowpark types for join keys (from feature view ``output_schema``).
        timeout_sec: Per-request timeout.

    Returns:
        ``(rows, schema)`` where each row has join keys plus selected feature columns,
        and ``schema`` is the matching :class:`StructType`
        for :func:`~snowflake.ml.feature_store.feature_store._coerce_row_values_for_snowpark_local_schema`.

    Raises:
        SnowflakeMLException: For invalid arguments, HTTP/API errors, unknown requested features,
            malformed or inconsistent responses, or bad result rows.
    """
    if not keys:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("keys must be non-empty for online feature reads."),
        )
    if len(join_key_names) == 0:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("join_key_names must be non-empty."),
        )
    for row in keys:
        if len(row) != len(join_key_names):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Each keys row must have {len(join_key_names)} values (join keys {join_key_names}), "
                    f"got {len(row)}."
                ),
            )

    resolved_query_url = _query_api_url(query_url)
    logger.debug("Query API url=%r", resolved_query_url)
    url = resolved_query_url
    headers = _session_rest_auth_headers(session)

    out: list[dict[str, Any]] = []
    baseline_names: Optional[list[str]] = None
    output_feature_names: Optional[list[str]] = None
    schema: Any = None

    for start in range(0, len(keys), _MAX_QUERY_REQUEST_ROWS):
        batch = keys[start : start + _MAX_QUERY_REQUEST_ROWS]
        request_rows = [
            {"entity": {jk: _json_serialize_value(kv) for jk, kv in zip(join_key_names, row)}} for row in batch
        ]
        body: dict[str, Any] = {
            "name": str(feature_view_name),
            "version": str(feature_view_version),
            "object_type": "feature_view",
            "metadata_options": {"include_names": True, "include_data_types": True},
            "request_rows": request_rows,
        }
        if feature_names:
            body["features"] = _dedupe_feature_names_preserve_order(feature_names)

        status, raw = _http_post_json(url, headers, body, timeout_sec)
        if status not in (200, 207):
            _raise_online_service_http_error(status, raw)

        data, results, names_from_server, dict_items = _parse_query_batch_response(raw, len(batch))

        rid = data.get("request_id")
        if isinstance(rid, str) and rid.strip():
            logger.info(
                "Query API response request_id=%r batch_start=%s batch_size=%s",
                rid.strip(),
                start,
                len(batch),
            )

        if baseline_names is None:
            baseline_names = names_from_server
            output_feature_names = _resolve_output_feature_names(names_from_server, feature_names)
            feature_types = _snowpark_types_from_query_metadata_features(dict_items)
            schema = _postgres_online_query_result_schema(
                join_key_names,
                join_key_field_types,
                output_feature_names,
                feature_types,
            )
        elif names_from_server != baseline_names:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(
                    "Online Service returned inconsistent feature metadata across requests; "
                    f"expected {baseline_names!r}, got {names_from_server!r}."
                ),
            )

        if output_feature_names is None or schema is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(
                    "Online Service query loop completed without initializing feature metadata."
                ),
            )

        out.extend(_extract_result_rows(results, names_from_server, output_feature_names, join_key_names, batch))

    if schema is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError("Online Service returned no batches; cannot build result schema."),
        )
    return out, schema
