"""HTTP client for the Online Service REST APIs (Query + Ingest).

Wraps ``httpx`` with HTTP/2 enabled. One ``httpx.Client`` is cached per
``(scheme://host, proxy)`` so concurrent reads share a single multiplexed
connection per origin. Auth headers are rebuilt per request so a rotated
``SNOWFLAKE_PAT`` takes effect immediately.

This module owns the entire HTTP surface for the Online Service: URL
building, PAT lookup, header construction, proxy resolution, and the
client itself. Callers in ``online_service.py`` get a typed
``(status_code, body_bytes)`` tuple from :meth:`OnlineServiceHttpClient.post_json`
and never touch ``httpx`` directly.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_QUERY_API_REL_PATH = "api/v1/query"
_INGEST_API_REL_PATH = "api/v1/ingest"


def query_api_url(query_base_url: str) -> str:
    """Append ``api/v1/query`` to the Online Service ``query`` endpoint URL from status JSON."""
    base = query_base_url.strip().rstrip("/") + "/"
    return urllib.parse.urljoin(base, _QUERY_API_REL_PATH)


def ingest_api_url(ingest_base_url: str) -> str:
    """Append ``api/v1/ingest`` to the Online Service ``ingest`` endpoint URL from status JSON."""
    base = ingest_base_url.strip().rstrip("/") + "/"
    return urllib.parse.urljoin(base, _INGEST_API_REL_PATH)


def online_service_pat_from_env() -> str:
    """PAT for ``Authorization: Snowflake Token="..."`` on the Online Service REST APIs.

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


def auth_headers(session: Optional[Session] = None) -> dict[str, str]:
    """Build HTTP headers for the Online Service REST APIs (Query + Ingest); ``session`` is unused."""
    _ = session
    token = online_service_pat_from_env()
    return {
        "Authorization": f'Snowflake Token="{token}"',
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _proxy_for_url(url: str) -> Optional[str]:
    """Proxy URL for ``url`` per ``NO_PROXY`` and the env-derived proxy table; ``None`` for direct."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if host and urllib.request.proxy_bypass(host):
        return None
    proxies = urllib.request.getproxies()
    return proxies.get(parsed.scheme) or proxies.get("all")


class OnlineServiceHttpClient:
    """HTTP/2-enabled httpx client for the Online Service, cached per origin+proxy.

    Caches one ``httpx.Client(http2=True)`` per ``(scheme://host, proxy)`` so concurrent reads share a
    single multiplexed connection per origin. Falls back to HTTP/1.1 transparently if the server's
    ALPN does not advertise h2. Honors ``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``NO_PROXY`` via
    :func:`_proxy_for_url`. Auth headers are rebuilt per request so a rotated PAT is picked up
    without re-init.
    """

    def __init__(
        self,
        *,
        max_connections: int = 8,
        _transport_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._max_connections = max_connections
        self._clients_by_key: dict[tuple[str, Optional[str]], Any] = {}
        self._http2_logged_for_origin: set[str] = set()
        self._transport_factory = _transport_factory
        self._closed = False

    def _client_for_url(self, url: str) -> Any:
        """Cached ``httpx.Client(http2=True)`` keyed by (scheme://host, proxy)."""
        import httpx

        parsed = urllib.parse.urlparse(url)
        proxy = _proxy_for_url(url)
        key = (f"{parsed.scheme}://{parsed.hostname or ''}", proxy)
        client = self._clients_by_key.get(key)
        if client is not None:
            return client

        limits = httpx.Limits(
            max_keepalive_connections=self._max_connections,
            max_connections=self._max_connections,
        )
        kwargs: dict[str, Any] = {
            "http2": True,
            "limits": limits,
            "timeout": None,  # per-request timeout passed to .post()
            "trust_env": False,  # proxy resolution handled by _proxy_for_url
        }
        if proxy:
            kwargs["proxy"] = proxy
        if self._transport_factory is not None:
            kwargs["transport"] = self._transport_factory(proxy=proxy)
        client = httpx.Client(**kwargs)
        self._clients_by_key[key] = client
        return client

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """POST ``body`` as JSON and return ``(status_code, body_bytes)``; only network errors raise."""
        if self._closed:
            raise RuntimeError("OnlineServiceHttpClient is closed")

        import httpx

        payload = json.dumps(body).encode("utf-8")
        headers = auth_headers()
        client = self._client_for_url(url)

        try:
            resp = client.post(url, content=payload, headers=headers, timeout=httpx.Timeout(timeout))
        except (httpx.HTTPError, OSError) as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWFLAKE_API_ERROR,
                original_exception=RuntimeError(f"Online Service is unreachable: {e}"),
            ) from e

        # Confirm h2 negotiation in production by logging once per origin.
        parsed = urllib.parse.urlparse(url)
        origin = f"{parsed.scheme}://{parsed.hostname or ''}"
        if origin not in self._http2_logged_for_origin:
            self._http2_logged_for_origin.add(origin)
            logger.info(
                "Online Service HTTP negotiated %s for origin %s",
                getattr(resp, "http_version", "unknown"),
                origin,
            )

        return int(resp.status_code), resp.content

    def close(self) -> None:
        """Close all underlying ``httpx.Client`` instances. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for client in self._clients_by_key.values():
            try:
                client.close()
            except Exception:  # noqa: BLE001 - best-effort close, never raise from cleanup
                logger.debug("OnlineServiceHttpClient close suppressed", exc_info=True)
        self._clients_by_key.clear()
