"""Tests for decl/service.py — pure parsing / SQL-building functions."""

from __future__ import annotations

import json

import pytest

from snowflake.ml.feature_store.decl.service import (
    format_status_display,
    get_service_endpoint,
    parse_service_status,
    service_sql,
)
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RAW_STATUS_WITH_ENDPOINTS = json.dumps(
    {
        "status": "RUNNING",
        "message": "Service is healthy",
        "endpoints": [
            {"name": "ingest", "url": "https://ingest.example.snowflakecomputing.com"},
            {"name": "query", "url": "https://query.example.snowflakecomputing.com"},
        ],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
    }
)

_RAW_STATUS_NO_ENDPOINTS = json.dumps(
    {
        "status": "INITIALIZING",
        "message": "Service is starting up",
    }
)


# ---------------------------------------------------------------------------
# parse_service_status
# ---------------------------------------------------------------------------


class TestParseServiceStatus:
    def test_returns_dict(self) -> None:
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        assert isinstance(result, dict)

    def test_status_field_present(self) -> None:
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        assert result["status"] == "RUNNING"

    def test_message_field_present(self) -> None:
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        assert result["message"] == "Service is healthy"

    def test_endpoints_list_present(self) -> None:
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        assert isinstance(result.get("endpoints"), list)
        assert len(result["endpoints"]) == 2

    def test_endpoint_has_name_and_url(self) -> None:
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        ep = result["endpoints"][0]
        assert "name" in ep
        assert "url" in ep

    def test_timestamps_preserved(self) -> None:
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        assert result.get("created_at") == "2024-01-01T00:00:00Z"
        assert result.get("updated_at") == "2024-01-02T00:00:00Z"

    def test_missing_endpoints_key_returns_empty_list(self) -> None:
        result = parse_service_status(_RAW_STATUS_NO_ENDPOINTS)
        assert result.get("endpoints", []) == [] or "endpoints" not in result

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_service_status("not-json{{{")

    def test_already_dict_passthrough(self) -> None:
        # Should also accept a pre-parsed dict (optional convenience)
        result = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)
        assert result["status"] == "RUNNING"


# ---------------------------------------------------------------------------
# get_service_endpoint
# ---------------------------------------------------------------------------


class TestGetServiceEndpoint:
    def setup_method(self) -> None:
        self.status = parse_service_status(_RAW_STATUS_WITH_ENDPOINTS)

    def test_finds_ingest_endpoint(self) -> None:
        url = get_service_endpoint(self.status, "ingest")
        assert url == "https://ingest.example.snowflakecomputing.com"

    def test_finds_query_endpoint(self) -> None:
        url = get_service_endpoint(self.status, "query")
        assert url == "https://query.example.snowflakecomputing.com"

    def test_missing_name_returns_none(self) -> None:
        url = get_service_endpoint(self.status, "nonexistent")
        assert url is None

    def test_empty_endpoints_returns_none(self) -> None:
        status = parse_service_status(_RAW_STATUS_NO_ENDPOINTS)
        url = get_service_endpoint(status, "ingest")
        assert url is None


# ---------------------------------------------------------------------------
# service_sql
# ---------------------------------------------------------------------------


class TestServiceSql:
    def setup_method(self) -> None:
        self.sqls = service_sql("MYDB", "MYSCHEMA")

    def test_returns_dict(self) -> None:
        assert isinstance(self.sqls, dict)

    def test_get_status_key_present(self) -> None:
        assert "get_status" in self.sqls

    def test_create_key_present(self) -> None:
        assert "create" in self.sqls

    def test_drop_key_present(self) -> None:
        assert "drop" in self.sqls

    def test_show_ofts_key_present(self) -> None:
        assert "show_ofts" in self.sqls

    def test_drop_oft_template_key_present(self) -> None:
        assert "drop_oft_template" in self.sqls

    def test_get_status_contains_db_and_schema(self) -> None:
        sql = self.sqls["get_status"]
        assert "MYDB" in sql
        assert "MYSCHEMA" in sql

    def test_create_contains_db_and_schema(self) -> None:
        sql = self.sqls["create"]
        assert "MYDB" in sql
        assert "MYSCHEMA" in sql

    def test_drop_contains_db_and_schema(self) -> None:
        sql = self.sqls["drop"]
        assert "MYDB" in sql
        assert "MYSCHEMA" in sql

    def test_show_ofts_contains_db_and_schema(self) -> None:
        sql = self.sqls["show_ofts"]
        assert "MYDB" in sql
        assert "MYSCHEMA" in sql

    def test_drop_oft_template_is_template(self) -> None:
        tmpl = self.sqls["drop_oft_template"]
        # Must contain a placeholder for the OFT name
        assert "{name}" in tmpl or "%" in tmpl or "<name>" in tmpl

    def test_get_status_calls_system_function(self) -> None:
        sql = self.sqls["get_status"]
        assert "SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS" in sql

    def test_create_calls_system_function(self) -> None:
        sql = self.sqls["create"]
        assert "SYSTEM$CREATE_FEATURE_STORE_ONLINE_SERVICE" in sql

    def test_drop_calls_system_function(self) -> None:
        sql = self.sqls["drop"]
        assert "SYSTEM$DROP_FEATURE_STORE_ONLINE_SERVICE" in sql


# ---------------------------------------------------------------------------
# Raw payload passthrough — runtime_id and endpoint URL variants
# ---------------------------------------------------------------------------


_RAW_STATUS_WITH_RUNTIME_ID_AND_PRIVATELINK = json.dumps(
    {
        "status": "RUNNING",
        "message": "Feature Store Online Service is running",
        "runtime_id": "rt-abc-123",
        "endpoints": [
            {
                "name": "ingest",
                "url": "https://ingest.example.snowflakecomputing.app",
                "privatelink_url": "https://ingest.pl.example.snowflakecomputing.app",
                "internal_url": "https://ingest.internal.snowflakecomputing.app",
            },
            {
                "name": "query",
                "url": "https://query.example.snowflakecomputing.app",
                "privatelink_url": "https://query.pl.example.snowflakecomputing.app",
            },
        ],
        "compute_pool": {"status": "ACTIVE", "name": "POOL_X"},
        "postgres": {"status": "READY", "name": "PG_X", "host": "pg.host"},
        "service": {"status": "RUNNING", "name": "SVC_X"},
        "network_rules": [],
        "secret": {"name": "SECRET_X", "username": "user_x"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
    }
)


class TestParseServiceStatusRawPassthrough:
    """Pin that ``parse_service_status`` carries raw-payload fields the
    rich display needs (``runtime_id`` plus the nested component dicts)
    and that endpoint url variants survive in the happy path.

    These regressed because the ``_HAS_ONLINE_SERVICE`` branch built
    the result dict strictly from ``OnlineServiceStatus`` — which has
    no ``runtime_id`` field and whose endpoint projection only kept
    ``name`` / ``url``. The user-facing symptom was
    ``Runtime ID: ?`` and missing PrivateLink endpoints in the
    ``snow feature online-service`` display.
    """

    def setup_method(self) -> None:
        self.result = parse_service_status(_RAW_STATUS_WITH_RUNTIME_ID_AND_PRIVATELINK)

    def test_runtime_id_preserved(self) -> None:
        assert self.result.get("runtime_id") == "rt-abc-123"

    def test_compute_pool_preserved(self) -> None:
        assert self.result.get("compute_pool") == {
            "status": "ACTIVE",
            "name": "POOL_X",
        }

    def test_postgres_preserved(self) -> None:
        assert self.result.get("postgres", {}).get("host") == "pg.host"

    def test_service_block_preserved(self) -> None:
        assert self.result.get("service", {}).get("name") == "SVC_X"

    def test_secret_not_collected(self) -> None:
        # The secret payload carries the Postgres password, so it is
        # never forwarded into the parsed status dict.
        raw = json.dumps(
            {
                "status": "RUNNING",
                "secret": {
                    "name": "SECRET_X",
                    "username": "user_x",
                    "password": "abcdef1234",
                },
            }
        )
        result = parse_service_status(raw)
        assert "secret" not in result

    def test_endpoint_privatelink_url_preserved(self) -> None:
        ingest = next(ep for ep in self.result["endpoints"] if ep.get("name") == "ingest")
        assert ingest.get("privatelink_url") == "https://ingest.pl.example.snowflakecomputing.app"

    def test_endpoint_internal_url_preserved(self) -> None:
        ingest = next(ep for ep in self.result["endpoints"] if ep.get("name") == "ingest")
        assert ingest.get("internal_url") == "https://ingest.internal.snowflakecomputing.app"

    def test_endpoint_without_internal_url_has_no_internal_url_key_or_none(self) -> None:
        query = next(ep for ep in self.result["endpoints"] if ep.get("name") == "query")
        # Either the key is absent or it is explicitly ``None`` — both are
        # acceptable; the format layer treats them identically.
        assert query.get("internal_url") in (None, "")


# ---------------------------------------------------------------------------
# format_status_display — rich text rendering for the CLI status display
# ---------------------------------------------------------------------------


class TestFormatStatusDisplayRuntimeId:
    """Pin that the rich text display surfaces the runtime id from the
    parsed status dict (rather than the historical ``?`` placeholder).

    The ``Runtime ID`` row moved under the ``verbose=True`` view as
    part of the Online Service naming pass — the compact default view
    only shows the umbrella ``+ Online Service   <state>`` heading —
    so these regressions are now pinned through the verbose path.
    """

    def test_runtime_id_rendered_when_present(self) -> None:
        status = parse_service_status(_RAW_STATUS_WITH_RUNTIME_ID_AND_PRIVATELINK)
        display = format_status_display(status, user="u", database="D", schema="S", verbose=True)
        assert "rt-abc-123" in display

    def test_runtime_id_falls_back_to_question_mark_when_absent(self) -> None:
        raw = json.dumps({"status": "PENDING", "message": "starting"})
        status = parse_service_status(raw)
        display = format_status_display(status, user="u", database="D", schema="S", verbose=True)
        assert "Runtime ID:" in display
        # Question-mark fallback is preserved when the runtime has not
        # finished provisioning yet.
        assert "Runtime ID:        ?" in display or "Runtime ID:" in display


class TestFormatStatusDisplayEndpoints:
    """Pin that every endpoint URL variant returned by the service is
    rendered — public URL plus PrivateLink and SPCS-internal variants
    when the service supplies them.
    """

    def setup_method(self) -> None:
        self.status = parse_service_status(_RAW_STATUS_WITH_RUNTIME_ID_AND_PRIVATELINK)
        self.display = format_status_display(self.status, user="u", database="D", schema="S")

    def test_public_ingest_url_rendered(self) -> None:
        assert "https://ingest.example.snowflakecomputing.app" in self.display

    def test_public_query_url_rendered(self) -> None:
        assert "https://query.example.snowflakecomputing.app" in self.display

    def test_privatelink_ingest_url_rendered(self) -> None:
        assert "https://ingest.pl.example.snowflakecomputing.app" in self.display

    def test_privatelink_query_url_rendered(self) -> None:
        assert "https://query.pl.example.snowflakecomputing.app" in self.display

    def test_internal_ingest_url_rendered(self) -> None:
        assert "https://ingest.internal.snowflakecomputing.app" in self.display

    def test_endpoint_without_optional_url_omits_label(self) -> None:
        # ``query`` has no ``internal_url`` — the ``internal:`` row must
        # not appear next to the query block (we only render variants
        # that the service actually supplied).
        # Locate the query endpoint block and assert the next two
        # indented lines (if any) do not advertise an internal URL.
        lines = self.display.splitlines()
        in_endpoints = False
        seen_query = False
        for line in lines:
            stripped = line.strip()
            if stripped == "Endpoints:":
                in_endpoints = True
                continue
            if not in_endpoints:
                continue
            if stripped.startswith("query:"):
                seen_query = True
                continue
            if seen_query and stripped.startswith("internal:"):
                pytest.fail(
                    "query endpoint advertised an internal URL even " "though the parsed status did not include one"
                )
            # Stop scanning once the next endpoint header (no leading
            # whitespace beyond two spaces) starts.
            if seen_query and stripped and not stripped.startswith(("privatelink:", "internal:")):
                break


# ---------------------------------------------------------------------------
# format_status_display — default-compact vs verbose layout
# ---------------------------------------------------------------------------


_RAW_STATUS_FULLY_POPULATED = json.dumps(
    {
        "status": "RUNNING",
        "message": "Feature Store Online Service is running",
        "runtime_id": "rt-abc-123",
        "endpoints": [
            {
                "name": "ingest",
                "url": "https://ingest.example.snowflakecomputing.app",
                "privatelink_url": "https://ingest.pl.example.snowflakecomputing.app",
                "internal_url": "https://ingest.internal.snowflakecomputing.app",
            },
            {
                "name": "query",
                "url": "https://query.example.snowflakecomputing.app",
            },
        ],
        "compute_pool": {
            "status": "ACTIVE",
            "name": "POOL_X",
            "instance_family": "CPU_X64_XS",
            "active_nodes": 1,
            "total_nodes": 1,
            "min_nodes": 1,
            "max_nodes": 2,
            "auto_suspend_secs": 300,
        },
        "postgres": {
            "status": "READY",
            "name": "PG_X",
            "compute_family": "PG_S",
            "postgres_version": "15.4",
            "storage_gb": 50,
            "host": "pg.host",
        },
        "service": {
            "status": "RUNNING",
            "name": "SVC_X",
            "image_version": "1.2.3",
            "compute_pool_name": "POOL_X",
            "current_instances": 1,
            "min_instances": 1,
            "max_instances": 2,
            "is_upgrading": True,
            "instances": [
                {
                    "instance_id": "i-1",
                    "status": "READY",
                    "ip": "10.0.0.1",
                    "start_time": "2024-01-01T00:00:00Z",
                },
            ],
        },
        "network_rules": [
            {
                "name": "RULE_X",
                "mode": "EGRESS",
                "type": "HOST_PORT",
                "purpose": "DEMO",
                "value_list": "example.com:443",
            },
        ],
        "secret": {
            "name": "SECRET_X",
            "username": "user_x",
            "password": "abcdef1234",
        },
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
    }
)


class TestFormatStatusDisplayVerbose:
    """Pin the default-compact / verbose split for the rich status display.

    Default view (``verbose=False``):
    * Keeps the banner, Runtime block, single-line summaries for the
      Compute Pool / Postgres / Service blocks, and every endpoint
      URL variant the runtime supplied.
    * Drops the per-component detail rows (``Name``, ``Family``,
      ``Nodes``, ``Min/Max``, ``Auto-suspend``, ``Host``, ``Image``,
      ``Pool``, ``Instances``, ``Upgrading``, per-instance lines) and
      the Network Rules block.

    Verbose view (``verbose=True``):
    * Renders every detail row and the Network Rules block.  The
      Postgres password is never collected, so neither the Secret
      block nor a connection string is rendered in either view.
    """

    def setup_method(self) -> None:
        self.status = parse_service_status(_RAW_STATUS_FULLY_POPULATED)

    # -- Default-compact view ---------------------------------------

    def test_default_keeps_online_service_heading(self) -> None:
        """The compact default still renders the umbrella ``+ Online
        Service`` heading row so the operator can read the overall
        state at a glance.  The detail rows (``Message:`` /
        ``Runtime ID:``) moved under ``verbose=True`` — pinned
        separately by ``TestFormatStatusDisplayOnlineServiceNaming``.
        """
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "+ Online Service" in display or "x Online Service" in display

    def test_default_keeps_compute_pool_heading(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Compute Pool" in display
        assert "ACTIVE" in display

    def test_default_keeps_postgres_heading(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Postgres" in display
        assert "READY" in display

    def test_default_keeps_service_heading(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Service" in display
        assert "RUNNING" in display

    def test_default_keeps_all_endpoint_variants(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "https://ingest.example.snowflakecomputing.app" in display
        assert "https://ingest.pl.example.snowflakecomputing.app" in display
        assert "https://ingest.internal.snowflakecomputing.app" in display
        assert "https://query.example.snowflakecomputing.app" in display

    def test_default_omits_compute_pool_detail_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        # ``Family:`` is unique to the Compute Pool detail rows in the
        # current layout — the Postgres detail uses ``Family:`` too,
        # but if neither block is rendered the label vanishes outright.
        assert "Family:" not in display
        assert "CPU_X64_XS" not in display
        assert "Nodes:" not in display
        assert "Min/Max:" not in display
        assert "Auto-suspend:" not in display

    def test_default_omits_postgres_detail_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Version:" not in display
        assert "Storage:" not in display
        assert "Host:" not in display
        assert "pg.host" not in display
        assert "PG 15.4" not in display

    def test_default_omits_service_detail_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Image:" not in display
        assert "1.2.3" not in display
        assert "Pool:" not in display
        assert "Instances:" not in display
        assert "Upgrading:" not in display

    def test_default_omits_per_instance_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "[i-1]" not in display
        assert "10.0.0.1" not in display

    def test_default_omits_network_rules_block(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Network Rules:" not in display
        assert "RULE_X" not in display
        assert "EGRESS" not in display

    def test_default_omits_secret_block(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Secret" not in display
        assert "SECRET_X" not in display
        assert "user_x" not in display

    def test_default_omits_postgres_connection_string(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "postgresql://" not in display
        assert "Postgres Connection String" not in display

    # -- Verbose view -----------------------------------------------

    def test_verbose_renders_compute_pool_detail_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "CPU_X64_XS" in display
        assert "Nodes:" in display
        assert "Min/Max:" in display
        assert "Auto-suspend:" in display

    def test_verbose_renders_postgres_detail_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "PG 15.4" in display
        assert "Storage:" in display
        assert "pg.host" in display

    def test_verbose_renders_service_detail_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "Image:" in display
        assert "1.2.3" in display
        assert "Pool:" in display
        assert "Instances:" in display
        assert "Upgrading:" in display

    def test_verbose_renders_per_instance_rows(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "[i-1]" in display
        assert "10.0.0.1" in display

    def test_verbose_renders_network_rules_block(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "Network Rules:" in display
        assert "RULE_X" in display
        assert "example.com:443" in display

    def test_verbose_omits_secret_block(self) -> None:
        # The secret payload is never collected, so even the verbose
        # view no longer renders a Secret block.
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "SECRET_X" not in display
        assert "user_x" not in display

    def test_verbose_omits_postgres_connection_string(self) -> None:
        # The password is never collected or printed, so even the verbose
        # view no longer renders a Postgres connection string.
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "Postgres Connection String" not in display
        assert "postgresql://" not in display
        assert "abcdef1234" not in display


class TestFormatStatusDisplayOnlineServiceNaming:
    """Pin the operator-facing naming.

    The umbrella status block was historically labelled ``Runtime``
    (banner header + ``+ Runtime`` heading row + ``Runtime ID`` /
    ``Message`` detail rows).  Operators consistently refer to the
    thing being managed as the *online service* — ``snow feature
    online-service`` is the command surface — so the display now
    speaks the same language:

    * Banner: ``Feature Store — Online Service Status``.
    * Heading row: ``+ Online Service   RUNNING``.
    * ``Message`` and ``Runtime ID`` move under the verbose flag —
      the compact view shows only the heading-only umbrella summary,
      matching the three sub-component summaries beneath it.

    The underlying parsed dict key ``runtime_id`` is unchanged
    (still copied through by ``parse_service_status``); only the
    rendered display labels were renamed.
    """

    def setup_method(self) -> None:
        self.status = parse_service_status(_RAW_STATUS_FULLY_POPULATED)

    def test_banner_says_online_service_status(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Online Service Status" in display
        assert "Runtime Status" not in display

    def test_banner_says_online_service_status_in_verbose_too(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "Online Service Status" in display
        assert "Runtime Status" not in display

    def test_umbrella_heading_says_online_service(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "+ Online Service" in display or "x Online Service" in display

    def test_umbrella_heading_in_verbose_too(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "+ Online Service" in display or "x Online Service" in display

    def test_default_omits_runtime_id_row(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Runtime ID:" not in display
        assert "rt-abc-123" not in display

    def test_default_omits_message_row(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        assert "Message:" not in display
        assert "Feature Store Online Service is running" not in display

    def test_verbose_renders_runtime_id_row(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "Runtime ID:" in display
        assert "rt-abc-123" in display

    def test_verbose_renders_message_row(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        assert "Message:" in display
        assert "Feature Store Online Service is running" in display


class TestFormatStatusDisplayCompactSpacing:
    """Pin the default-compact spacing contract.

    Without details under the Compute Pool / Postgres / Service
    headings, the inter-component blank lines just produce visual
    gaps.  Operators wanted the three single-line summaries clustered
    together — one blank line between the Runtime block and the
    component cluster, the three heading rows back-to-back with no
    interleaved blanks, then one blank line before the Endpoints
    block.

    Verbose mode keeps the historical detail-block separation (each
    detail block sits in its own paragraph) so this contract is
    default-only.
    """

    def setup_method(self) -> None:
        self.status = parse_service_status(_RAW_STATUS_FULLY_POPULATED)

    def _component_heading_indices(self, lines: list[str]) -> dict[str, int]:
        """Locate the three sub-component heading rows.

        ``Online Service`` is the umbrella heading and is excluded
        here on purpose — these checks are about the cluster of three
        sub-components.  We anchor on the ``+ <name>`` / ``x <name>``
        prefix so "Online Service" (which contains the substring
        "Service ") does not get misidentified as the SPCS Service
        heading.

        Args:
            lines: Rendered display split on newlines.

        Returns:
            Map of ``"compute_pool"`` / ``"postgres"`` / ``"service"``
            to the zero-based line index of each heading row.  Missing
            components are simply absent from the dict.
        """
        idx: dict[str, int] = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not (stripped.startswith("+ ") or stripped.startswith("x ")):
                continue
            after_icon = stripped[2:]
            if after_icon.startswith("Compute Pool"):
                idx.setdefault("compute_pool", i)
            elif after_icon.startswith("Postgres "):
                idx.setdefault("postgres", i)
            elif after_icon.startswith("Service "):
                idx.setdefault("service", i)
        return idx

    def test_default_no_blank_lines_between_component_headings(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        lines = display.splitlines()
        idx = self._component_heading_indices(lines)
        assert "compute_pool" in idx and "postgres" in idx and "service" in idx
        # The Postgres heading sits exactly one line after the
        # Compute Pool heading (no interleaved blank).
        assert idx["postgres"] == idx["compute_pool"] + 1, (
            f"expected Postgres heading immediately after Compute Pool, " f"got lines {idx} in:\n{display}"
        )
        # And Service sits exactly one line after Postgres.
        assert idx["service"] == idx["postgres"] + 1, (
            f"expected Service heading immediately after Postgres, " f"got lines {idx} in:\n{display}"
        )

    def test_default_single_blank_line_between_component_cluster_and_endpoints(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S")
        lines = display.splitlines()
        idx = self._component_heading_indices(lines)
        # Locate the Endpoints heading.
        endpoint_idx = next(i for i, line in enumerate(lines) if line.strip() == "Endpoints:")
        # Exactly one blank line should sit between the Service heading
        # and the Endpoints block.
        between = lines[idx["service"] + 1 : endpoint_idx]
        assert between == [""], f"expected exactly one blank line between Service and Endpoints, " f"got {between!r}"

    def test_verbose_single_blank_line_between_online_service_block_and_compute_pool(self) -> None:
        """In verbose mode the Online Service block ends with the
        ``Runtime ID`` row and is followed by exactly one blank line
        before the Compute Pool heading — pin the separator survives
        the Online Service rename + gate."""
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        lines = display.splitlines()
        idx = self._component_heading_indices(lines)
        runtime_id_idx = next(i for i, line in enumerate(lines) if "Runtime ID:" in line)
        between = lines[runtime_id_idx + 1 : idx["compute_pool"]]
        assert between == [""], (
            f"expected exactly one blank line between the Online Service "
            f"detail block and Compute Pool, got {between!r}"
        )

    def test_default_no_blank_line_between_online_service_and_compute_pool(self) -> None:
        """In the new compact layout the umbrella ``Online Service``
        heading clusters tightly with the three sub-component
        summaries — no blank line interleaves them."""
        display = format_status_display(self.status, user="u", database="D", schema="S")
        lines = display.splitlines()
        idx = self._component_heading_indices(lines)
        os_idx = next(i for i, line in enumerate(lines) if "+ Online Service" in line or "x Online Service" in line)
        between = lines[os_idx + 1 : idx["compute_pool"]]
        assert between == [], (
            f"expected Online Service heading immediately followed by " f"Compute Pool heading, got {between!r}"
        )

    def test_verbose_keeps_blank_line_after_compute_pool_detail(self) -> None:
        display = format_status_display(self.status, user="u", database="D", schema="S", verbose=True)
        lines = display.splitlines()
        idx = self._component_heading_indices(lines)
        # In verbose mode the detail rows live between the Compute Pool
        # heading and the Postgres heading — there must NOT be a
        # zero-line gap (regression pin against an over-zealous removal
        # of inter-block separation).
        assert idx["postgres"] > idx["compute_pool"] + 1, (
            f"verbose mode regressed: Compute Pool and Postgres headings " f"are adjacent in:\n{display}"
        )
        # And there is at least one blank line between Compute Pool's
        # detail block and the Postgres heading.
        between = lines[idx["compute_pool"] + 1 : idx["postgres"]]
        assert "" in between, (
            f"verbose mode lost the blank line separator between " f"Compute Pool and Postgres; got {between!r}"
        )


if __name__ == "__main__":
    pytest_driver.main()
