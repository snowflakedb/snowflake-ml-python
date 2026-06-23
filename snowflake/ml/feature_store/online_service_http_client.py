"""HTTP client for the Online Service REST APIs (Query + Ingest).

HTTP/2 ``httpx`` client cached per ``(scheme://host, proxy)``. Auth resolution order:
in-session PAT (``authenticator='PROGRAMMATIC_ACCESS_TOKEN'``) → ``SNOWFLAKE_PAT`` env var → raise.

NOTE: We intentionally do NOT issue a session token from the Snowpark connection.
A session token reuses the caller's existing GS session; when that session was created
by the Python connector (``clientAppId=PythonSnowpark``), GS encodes the
``system$snowservices_resolve_ingress`` result in ARROW format, which the SPCS ingress
proxy cannot parse (it only reads ``rowset``, null for ARROW) → "empty data" → HTTP 500.
A real PAT forces a fresh GS session (``clientAppId=SnowServices Ingress``) → JSON → works.
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
    """Read PAT from ``SNOWFLAKE_PAT``; legacy lookup used by the env-var fallback path.

    Returns:
        PAT string from ``SNOWFLAKE_PAT``.

    Raises:
        SnowflakeMLException: When ``SNOWFLAKE_PAT`` is unset or empty.
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


def _pat_token_from_session(session: Session) -> Optional[str]:
    """Pull a PAT from the session when ``authenticator='PROGRAMMATIC_ACCESS_TOKEN'``.

    Args:
        session: Snowpark session.

    Returns:
        PAT string, or ``None`` if the session isn't PAT-authed.
    """
    try:
        from snowflake.connector.auth.pat import AuthByPAT

        auth_cls = session.connection.auth_class
        if not isinstance(auth_cls, AuthByPAT):
            return None
        pat = auth_cls.assertion_content
        if isinstance(pat, str) and pat.strip():
            return pat.strip()
    except Exception:
        logger.debug("PAT extraction from session failed", exc_info=True)
    return None


_NO_AUTH_MESSAGE = (
    "Online Service requires a Programmatic Access Token (PAT). Either create the "
    "Snowpark Session with authenticator='PROGRAMMATIC_ACCESS_TOKEN' (PAT in password), "
    "or set the SNOWFLAKE_PAT environment variable. A session token cannot be used: it "
    "reuses the caller's GS session and triggers an ARROW response the ingress proxy "
    "cannot parse."
)


def auth_headers(session: Optional[Session] = None) -> dict[str, str]:
    """Build HTTP headers via a one-shot auth resolution; per-request and not cached.

    Prefer :class:`OnlineServiceHttpClient` for the hot path — it captures the session
    once at construction time and reuses it. This standalone helper is for callers
    that need a quick auth-validation pre-flight.

    Resolution: in-session PAT (if ``authenticator='PROGRAMMATIC_ACCESS_TOKEN'``) \u2192
    ``SNOWFLAKE_PAT`` env var. A session token is intentionally never used (see module docstring).

    Args:
        session: Snowpark session for the resolution chain; ``None`` falls through to env var.

    Returns:
        Dict with ``Authorization``, ``Content-Type``, and ``Accept`` headers.

    Raises:
        SnowflakeMLException: When no auth path resolves a token.
    """
    token: Optional[str] = None
    if session is not None:
        token = _pat_token_from_session(session)
    if not token:
        token = os.environ.get("SNOWFLAKE_PAT", "").strip() or None
    if not token:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=RuntimeError(_NO_AUTH_MESSAGE),
        )
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

    Caches one ``httpx.Client(http2=True)`` per ``(scheme://host, proxy)`` so concurrent reads
    share a single multiplexed connection per origin. Falls back to HTTP/1.1 transparently if
    the server's ALPN does not advertise h2. Honors ``HTTPS_PROXY`` / ``HTTP_PROXY`` /
    ``NO_PROXY`` via :func:`_proxy_for_url`.

    When ``session`` is provided and was created with
    ``authenticator='PROGRAMMATIC_ACCESS_TOKEN'``, the in-session PAT is used. Otherwise
    falls back to the ``SNOWFLAKE_PAT`` env var. A session token is intentionally never
    used — it reuses the caller's GS session and triggers an ARROW response the ingress
    proxy cannot parse (see module docstring).
    """

    def __init__(
        self,
        *,
        session: Optional[Session] = None,
        max_connections: int = 8,
        _transport_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._max_connections = max_connections
        self._clients_by_key: dict[tuple[str, Optional[str]], Any] = {}
        self._http2_logged_for_origin: set[str] = set()
        self._transport_factory = _transport_factory
        self._closed = False
        self._session: Optional[Session] = session
        self._session_pat: Optional[str] = _pat_token_from_session(session) if session is not None else None

    def _resolve_token(self, url: str) -> str:
        """Resolve the bearer token: in-session PAT → ``SNOWFLAKE_PAT`` env var.

        A session token is intentionally never used — it reuses the caller's GS session
        and triggers an ARROW response the ingress proxy cannot parse (see module docstring).

        Args:
            url: Online Service URL (unused; kept for signature symmetry).

        Returns:
            Token string for the ``Authorization`` header.

        Raises:
            SnowflakeMLException: when no auth path can produce a token.
        """
        del url
        if self._session_pat:
            logger.info("Online Service auth: resolved via session PAT (len=%d)", len(self._session_pat))
            return self._session_pat
        env_pat = os.environ.get("SNOWFLAKE_PAT", "").strip()
        if env_pat:
            logger.info("Online Service auth: resolved via SNOWFLAKE_PAT env var (len=%d)", len(env_pat))
            return env_pat
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=RuntimeError(_NO_AUTH_MESSAGE),
        )

    def _build_auth_headers(self, url: str) -> dict[str, str]:
        return {
            "Authorization": f'Snowflake Token="{self._resolve_token(url)}"',
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

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

    def post_json(
        self,
        url: str,
        body: dict[str, Any],
        timeout: float,
    ) -> tuple[int, bytes]:
        """POST ``body`` as JSON and return ``(status_code, body_bytes)``; only network errors raise.

        Args:
            url: Online Service endpoint URL.
            body: JSON-serializable payload.
            timeout: Per-request timeout in seconds.

        Returns:
            ``(status_code, body_bytes)`` from the server response.

        Raises:
            RuntimeError: When the client has been closed.
            SnowflakeMLException: On transport errors or unresolvable authentication.
        """
        if self._closed:
            raise RuntimeError("OnlineServiceHttpClient is closed")

        import httpx

        payload = json.dumps(body).encode("utf-8")
        headers = self._build_auth_headers(url)
        client = self._client_for_url(url)

        try:
            resp = client.post(url, content=payload, headers=headers, timeout=httpx.Timeout(timeout))
        except (httpx.HTTPError, OSError) as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWFLAKE_API_ERROR,
                original_exception=RuntimeError(f"Online Service is unreachable: {e}"),
            ) from e

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
