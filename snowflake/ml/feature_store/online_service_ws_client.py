"""Single-connection WebSocket client for the Online Service query API.

One instance owns one connection. It satisfies the ``post_json`` / ``close`` surface in
:mod:`online_service_transport`, so it substitutes for the HTTP client without the query
path changing: request bodies, batching, response parsing and error mapping are all reused.

Why a second transport at all: the query endpoint authenticates every HTTP request, so a
read of N batches costs N authentications. A WebSocket authenticates once, at the handshake,
and every frame after that rides an already-authorized connection.

Two wire-format differences from HTTP, both handled here:

* A frame carries no headers, so the correlation id travels in the body as ``request_id``
  and the error envelope carries ``retry_after_seconds`` where HTTP uses a header.
* The envelope has no status field, so :func:`status_for_error_code` reconstructs the
  status the HTTP path would have returned. Callers therefore see identical statuses and
  raise identical exceptions on both transports.

Query only. The ingest API has no WebSocket endpoint, and its callers read 207 and
``partial_success``, which :func:`decode_ws_response` cannot express: every non-error envelope
decodes as 200. This client must never be handed to the ingest path.

``websocket-client`` is imported lazily so an install without the ``feature_store`` extra
cannot break the HTTP path.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import uuid
from typing import Any, Callable

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.feature_store import online_service_http_client
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_WS_QUERY_REL_PATH = "ws"

# Upgrade response header carrying this connection's lifetime in whole seconds. Matched
# case-insensitively; websocket-client lower-cases response header names.
WS_MAX_CONN_AGE_HEADER = "x-ws-max-conn-age-seconds"

# Error codes the service emits, mapped to the status the HTTP path would have returned for
# the same failure. Keeping the two transports on one status vocabulary is what lets the
# query path reuse its existing error-mapping table unchanged.
_ERROR_CODE_TO_STATUS: dict[str, int] = {
    "INVALID_ARGUMENT": 400,
    "MISSING_FIELD": 400,
    "INVALID_TIMESTAMP": 400,
    "INVALID_FIELD_TYPE": 400,
    "STRING_TOO_LONG": 400,
    "UNSUPPORTED_TYPE": 400,
    "FAILED_PRECONDITION": 400,
    "UNAUTHENTICATED": 401,
    "PERMISSION_DENIED": 403,
    "NOT_FOUND": 404,
    "RESOURCE_EXHAUSTED": 429,
    "INTERNAL": 500,
    "UDF_EXECUTION_FAILED": 500,
    "UNAVAILABLE": 503,
    "UPSTREAM_UNAVAILABLE": 503,
    "RT_READER_NOT_READY": 503,
    "WRITE_TIMEOUT": 504,
}

_DEFAULT_STATUS_FOR_UNKNOWN_ERROR = 500

# Floor for a derived socket timeout. A timeout of exactly 0 means "non-blocking" to a socket,
# which fails in a different shape than a timeout and would be reported as an unreachable
# service rather than a slow one.
_MIN_SOCKET_TIMEOUT_SECONDS = 0.001


def status_for_error_code(code: str) -> int:
    """Map a service error code to the status the HTTP path would have returned.

    Args:
        code: Error code from the response envelope, e.g. ``RESOURCE_EXHAUSTED``.

    Returns:
        The equivalent status code; 500 when the code is unrecognized.
    """
    return _ERROR_CODE_TO_STATUS.get(code.strip().upper(), _DEFAULT_STATUS_FOR_UNKNOWN_ERROR)


def _dial(url: str, headers: list[str], timeout: float) -> Any:
    """Open a real WebSocket, trusting the same CA bundle the HTTP transport does.

    httpx (the HTTP transport) verifies against ``certifi.where()``; websocket-client on its
    own falls back to the interpreter's system store. On hosts whose system store lacks the
    service's root the WebSocket dial fails with CERTIFICATE_VERIFY_FAILED while HTTP reads
    to the same host succeed, so pin both transports to one trust store.

    Args:
        url: The WebSocket query URL.
        headers: Handshake header lines.
        timeout: Handshake timeout in seconds.

    Returns:
        The connected ``websocket.WebSocket``.
    """
    # Imported lazily: an install without the feature_store extra must still be
    # able to use the HTTP transport.
    import certifi
    import websocket

    return websocket.create_connection(url, header=headers, timeout=timeout, sslopt={"ca_certs": certifi.where()})


def ws_query_url(query_base_url: str) -> str:
    """Build the WebSocket query URL from the Online Service ``query`` endpoint URL.

    Derived from :func:`online_service_http_client.query_api_url` rather than assembled
    independently, so the two transports cannot drift apart on the path.

    Args:
        query_base_url: The ``query`` endpoint URL from the service status payload.

    Returns:
        The WebSocket URL for the query API, ending in ``api/v1/query/ws``.

    Raises:
        SnowflakeMLException: When the endpoint URL has no http or https scheme.
    """
    http_url = online_service_http_client.query_api_url(query_base_url)
    parsed = urllib.parse.urlparse(http_url)
    if parsed.scheme == "https":
        ws_scheme = "wss"
    elif parsed.scheme == "http":
        ws_scheme = "ws"
    else:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Online Service query endpoint must use http or https, got {parsed.scheme!r}"
            ),
        )
    base = http_url.rstrip("/") + "/"
    joined = urllib.parse.urljoin(base, _WS_QUERY_REL_PATH)
    return ws_scheme + joined[len(parsed.scheme) :]


def retry_after_of_handshake(error: BaseException) -> int:
    """Read ``Retry-After`` out of a refused handshake's response headers.

    A refusal at the handshake is still an HTTP response, so the backoff guidance arrives as a
    header rather than in a frame body. Without reading it here the caller can tell *that* the
    service is at capacity but not how long it asked to be left alone, and every client redials
    on its own schedule instead of the one the service requested.

    Args:
        error: The exception raised by the handshake.

    Returns:
        The requested delay in whole seconds, or 0 when absent or unparsable. Only the
        delta-seconds form is read; the HTTP-date form is treated as absent, because a clock
        skew would turn a misread date into an unbounded wait.
    """
    headers = getattr(error, "resp_headers", None)
    if headers is None:
        return 0
    items = headers.items() if hasattr(headers, "items") else headers
    try:
        for name, value in items:
            if str(name).strip().lower() != "retry-after":
                continue
            try:
                return max(0, int(float(str(value).strip())))
            except (TypeError, ValueError):
                return 0
    except (AttributeError, TypeError, ValueError):
        return 0
    return 0


class OnlineServiceWsDialError(RuntimeError):
    """A WebSocket handshake was refused, carrying the status so callers can classify it.

    The three refusals mean different things and warrant different behaviour: 404 when the
    transport is switched off server-side, 503 when the per-instance connection cap is full
    and the caller should back off, 401 when the credential was rejected before the request
    reached the service.
    """

    def __init__(self, message: str, *, status: int | None = None, retry_after_seconds: int = 0) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after_seconds = retry_after_seconds

    @property
    def transport_unavailable(self) -> bool:
        """True when the WebSocket transport is not served at all, so retrying is pointless."""
        return self.status == 404

    @property
    def at_capacity(self) -> bool:
        """True when the service is at its connection cap and a later redial may succeed."""
        return self.status == 503

    @property
    def auth_rejected(self) -> bool:
        """True when the credential was rejected, so redialing without new credentials is pointless."""
        return self.status in (401, 403)


class _ServerClosedConnection(RuntimeError):
    """The service closed the connection, which ``websocket-client`` reports as an empty ``recv()``.

    An empty frame is a close, never a response, and must leave the exchange as an exception: a
    close that returns normally would decode as an empty success, leaving the transport to count a
    dead socket as served and keep it available for reuse.
    """


def _deadline_exceeded(when: str, kind: type[TimeoutError] = TimeoutError) -> snowml_exceptions.SnowflakeMLException:
    """Build the error for a call that ran out of its timeout.

    Args:
        when: Where the deadline was found spent, e.g. ``"before sending a frame"``.
        kind: The ``TimeoutError`` subclass to wrap, so a caller can tell where the time went.

    Returns:
        The exception to raise.
    """
    return snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INTERNAL_SNOWFLAKE_API_ERROR,
        original_exception=kind(
            f"Online Service batch exceeded its timeout {when}; the timeout covers the whole batch, not each frame"
        ),
    )


def _is_socket_timeout(error: BaseException) -> bool:
    """Whether a send or recv failed because its socket timeout expired.

    ``websocket-client`` catches the socket's ``TimeoutError`` and re-raises it as
    ``WebSocketTimeoutException``, which is not a ``TimeoutError`` subclass. The deadline nearly
    always expires while a recv is blocked, so without this the usual overrun would read as an
    unreachable service rather than a slow one.

    Args:
        error: The exception raised by the socket operation.

    Returns:
        True for a timeout from either the socket or ``websocket-client``.
    """
    if isinstance(error, TimeoutError):
        return True
    try:
        import websocket
    except ImportError:  # only test fakes reach here without the library
        return False
    return isinstance(error, websocket.WebSocketTimeoutException)


def _transport_failure(error: BaseException) -> snowml_exceptions.SnowflakeMLException:
    """Classify a send or recv failure as a timeout or as an unreachable service.

    Args:
        error: The exception raised by the socket operation.

    Returns:
        The exception to raise.
    """
    if _is_socket_timeout(error):
        return _deadline_exceeded("while waiting on the service")
    return snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INTERNAL_SNOWFLAKE_API_ERROR,
        original_exception=RuntimeError(f"Online Service is unreachable: {error}"),
    )


class OnlineServiceWsClient:
    """One WebSocket connection to the Online Service query API.

    Satisfies :class:`online_service_transport.OnlineServiceTransport` for the query API only
    (see the module docstring). Not thread-safe: a WebSocket permits one reader and one
    writer, so callers must not share an instance across threads.
    :mod:`online_service_ws_pool` is what hands out one per caller.
    """

    def __init__(
        self,
        *,
        session: Session | None = None,
        connect_timeout: float = 30.0,
        max_in_flight: int | None = None,
        _ws_factory: Callable[..., Any] | None = None,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        """Prepare a client. No connection is opened until the first request.

        Args:
            session: Snowpark session used only for auth resolution.
            connect_timeout: Handshake timeout in seconds.
            max_in_flight: Most requests :meth:`post_json_many` leaves unanswered at once;
                defaults to :data:`online_service_ws_pipeline.DEFAULT_MAX_IN_FLIGHT`.
            _ws_factory: Test seam. Called as ``factory(url, header=[...], timeout=...)``
                and expected to return an object with ``send``, ``recv`` and ``close``.
            _clock: Test seam returning monotonic seconds, so batch-deadline behaviour can be
                asserted without waiting.
        """
        self._session = session
        self._connect_timeout = connect_timeout
        self._max_in_flight = max_in_flight
        self._ws_factory = _ws_factory
        self._clock = _clock or time.monotonic
        self._ws: Any | None = None
        self._url: str | None = None
        self._closed = False
        self._request_count = 0
        self._server_hostname: str | None = None
        self._server_max_conn_age_seconds: float | None = None
        # When the current socket was dialed. The service's advertised age runs from the dial,
        # and this client re-dials transparently after a mid-request failure, so a pool holding
        # this client cannot date the connection from when it first handed it out.
        self._connected_at: float | None = None

    # ----- connection ------------------------------------------------------

    def _handshake_headers(self, url: str) -> list[str]:
        """Build handshake headers by reusing the HTTP client's auth resolution.

        Going through the shared helper keeps session-PAT and environment-variable auth
        working identically on both transports, and means a future change to the auth
        scheme is picked up here automatically.

        Args:
            url: The WebSocket URL being dialed.

        Returns:
            Header lines in ``Name: value`` form.
        """
        del url
        headers = dict(online_service_http_client.auth_headers(self._session))
        # Content-Type describes a request body; a handshake has none.
        headers.pop("Content-Type", None)
        return [f"{name}: {value}" for name, value in headers.items()]

    def connect(self, url: str, timeout: float | None = None) -> None:
        """Open the connection if it is not already open.

        Args:
            url: The WebSocket query URL, from :func:`ws_query_url`.
            timeout: Seconds to allow for the handshake. Bounded by ``connect_timeout`` either
                way; a caller that passes its own remaining budget gets the smaller of the two,
                so the dial cannot overrun a deadline the caller has already started.

        Raises:
            RuntimeError: When the client has been closed.
            OnlineServiceWsDialError: When the handshake is refused or the dial fails.
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsClient is closed")
        if self._ws is not None:
            return

        headers = self._handshake_headers(url)
        dial_timeout = self._connect_timeout if timeout is None else min(timeout, self._connect_timeout)
        try:
            if self._ws_factory is not None:
                self._ws = self._ws_factory(url, header=headers, timeout=dial_timeout)
            else:
                self._ws = _dial(url, headers, dial_timeout)
        except Exception as e:  # noqa: BLE001 - library raises several unrelated shapes
            status = getattr(e, "status_code", None)
            raise OnlineServiceWsDialError(
                f"Online Service WebSocket connection failed: {e}",
                status=int(status) if isinstance(status, int) else None,
                retry_after_seconds=retry_after_of_handshake(e),
            ) from e

        self._url = url
        self._server_max_conn_age_seconds = _advertised_conn_age(self._ws)
        self._connected_at = self._clock()
        logger.info("Online Service WebSocket connected")

    @property
    def is_connected(self) -> bool:
        """True when a connection is open."""
        return self._ws is not None and not self._closed

    @property
    def request_count(self) -> int:
        """Requests answered on this connection — the reuse the transport exists for."""
        return self._request_count

    @property
    def server_hostname(self) -> str | None:
        """The serving instance this connection landed on, once a response has reported one.

        The gateway assigns an instance per connection at random and the client cannot choose
        or query it, so the only way to observe where a connection landed is to read what the
        responses say. Without this, uneven spread is invisible from the client side.

        Returns:
            The instance hostname, or None when no response has reported one.
        """
        return self._server_hostname

    @property
    def server_max_conn_age_seconds(self) -> float | None:
        """How long the service said this connection will live, from the upgrade response.

        The service closes every connection at its own jittered age and advertises that age on
        the upgrade, so the pool can retire the connection just before the service would close
        it rather than guessing from a hardcoded constant.

        Returns:
            Seconds, or None when not connected or the service did not advertise one.
        """
        return self._server_max_conn_age_seconds

    @property
    def connected_at(self) -> float | None:
        """When the current socket was dialed, on this client's clock.

        The advertised age runs from the dial, and this client re-dials transparently on the
        next use after a mid-request failure (see :meth:`_discard_connection`). A pool that
        dated the connection from when it first handed the client out would therefore measure a
        fresh socket against the previous one's dial and retire it at birth.

        Returns:
            Monotonic seconds, or None when no socket is open.
        """
        return self._connected_at

    # ----- transport surface -----------------------------------------------

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Send one request body and return ``(status_code, body_bytes)``.

        Connects on first use. A ``request_id`` is added to the body when the caller did not
        supply one, because a frame has no headers to carry a correlation id.

        Args:
            url: The WebSocket query URL.
            body: JSON-serializable payload.
            timeout: Per-request timeout in seconds.

        Returns:
            ``(status_code, body_bytes)``, shaped so callers cannot tell which transport
            answered.

        Raises:
            RuntimeError: When the client has been closed.
            SnowflakeMLException: When the connection fails mid-request.

        # noqa: DAR401
        # noqa: DAR402
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsClient is closed")

        # Built before connecting, as the HTTP client does, so a body that cannot be
        # serialized raises to the caller without costing a healthy connection.
        request_id = request_id_for(body)
        request_body = dict(body)
        request_body["request_id"] = request_id
        frame = json.dumps(request_body)

        # Same deadline rule as the batch path, for the same reason: a send and a recv are two
        # socket operations, so one settimeout(timeout) would allow up to twice the timeout the
        # caller asked for. Started *before* the dial, because the handshake is a socket
        # operation too -- on a cold connection it is the most expensive one -- and a deadline
        # that begins after it lets a slow upgrade overrun the caller's timeout by the whole
        # connect_timeout before the budget it asked for has even started.
        deadline = self._clock() + timeout

        self.connect(url, self._deadline_timeout(deadline, "the handshake"))
        ws = self._ws
        assert ws is not None  # connect() either set it or raised

        try:
            self._apply_timeout(ws, self._deadline_timeout(deadline, "sending the request"))
            ws.send(frame)
            self._apply_timeout(ws, self._deadline_timeout(deadline, "reading the response"))
            raw = ws.recv()
            if not raw:
                raise _ServerClosedConnection("the service closed the connection")
        except snowml_exceptions.SnowflakeMLException:
            # Already classified -- a deadline overrun from _deadline_timeout arrives here, and
            # relabelling it "unreachable" would hide a slow service behind a dead one. The
            # stream may hold an unread response either way, so the connection still goes.
            self._discard_connection()
            raise
        except Exception as e:  # noqa: BLE001 - library raises several unrelated shapes
            self._discard_connection()
            raise _transport_failure(e) from e

        self._request_count += 1
        try:
            status, payload = decode_ws_response(raw)
            _assert_response_matches(payload, request_id)
        except snowml_exceptions.SnowflakeMLException:
            # Same reasoning as :meth:`post_json_many`: a response carrying another request's
            # correlation id means this connection's stream is misaligned, so the response this
            # request was owed is still queued and would be read by the *next* request on this
            # connection. Returning it to the pool would let one bad exchange poison every later
            # caller that draws it until age or budget retires it, and a caller that retries
            # would re-send on the same misaligned socket and fail again. Dropping the socket
            # leaves the client reusable, so the next use re-upgrades onto a fresh stream.
            self._discard_connection()
            raise

        self._record_server_hostname(payload)
        return status, payload

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Pipeline every body over this connection, and return results in request order.

        Requests go out through a bounded window rather than one round trip each. Responses are
        matched to requests by ``request_id`` in :func:`online_service_ws_pipeline.exchange_frames`,
        because pipelining means they can arrive in any order.

        Args:
            url: The WebSocket query URL.
            bodies: JSON-serializable payloads, in the order results are wanted.
            timeout: Timeout in seconds for the whole batch.

        Returns:
            One ``(status_code, body_bytes)`` per entry in ``bodies``, in the same order.

        Raises:
            RuntimeError: When the client has been closed.
            SnowflakeMLException: When the caller supplied duplicate request ids (raised before
                connecting), or when a response cannot be matched to the request it belongs
                to, re-raised from the exchange.
            snowml_exceptions.SnowflakeMLException: When the connection fails mid-batch.

        # noqa: DAR401
        # noqa: DAR402
        """
        if self._closed:
            raise RuntimeError("OnlineServiceWsClient is closed")
        if not bodies:
            return []
        if len(bodies) == 1:
            # One request in flight needs no correlation, and requiring it here would add a
            # failure mode the sequential path does not have.
            return [self.post_json(url, bodies[0], timeout)]

        # Imported here rather than at module scope: the pipeline module imports this one to
        # reuse decode_ws_response, so a top-level import would be circular.
        from snowflake.ml.feature_store import online_service_ws_pipeline

        # Validated before connecting: duplicate ids are the caller's error, and raising one
        # from inside the exchange would discard a healthy connection and force a re-handshake.
        framed = online_service_ws_pipeline.frame_requests(bodies)
        max_in_flight = (
            online_service_ws_pipeline.DEFAULT_MAX_IN_FLIGHT if self._max_in_flight is None else self._max_in_flight
        )

        # One deadline for the whole batch, re-derived before every socket operation. Setting
        # the caller's timeout once would bound each send and each recv separately, so a batch
        # of N could take N times the timeout the caller asked for -- the documented contract
        # here and on the transport Protocol is that it covers the batch. The handshake is
        # inside it too, for the reason given in :meth:`post_json`.
        deadline = self._clock() + timeout

        self.connect(url, self._deadline_timeout(deadline, "the handshake"))
        ws = self._ws
        assert ws is not None  # connect() either set it or raised

        def send(frame: str) -> None:
            self._apply_timeout(ws, self._deadline_timeout(deadline, "sending a frame"))
            ws.send(frame)

        def recv() -> str | bytes:
            self._apply_timeout(ws, self._deadline_timeout(deadline, "reading a response"))
            received: str | bytes = ws.recv()
            if not received:
                raise _ServerClosedConnection("the service closed the connection mid-batch")
            return received

        try:
            results = online_service_ws_pipeline.exchange_frames(send, recv, framed, max_in_flight=max_in_flight)
        except snowml_exceptions.SnowflakeMLException:
            # A correlation or completeness failure means this connection's stream is no
            # longer trustworthy — unread responses would land in the next batch.
            self._discard_connection()
            raise
        except Exception as e:  # noqa: BLE001 - library raises several unrelated shapes
            self._discard_connection()
            raise _transport_failure(e) from e

        self._request_count += len(bodies)
        return results

    def close(self) -> None:
        """Close the connection. Idempotent, and never raises."""
        if self._closed:
            return
        self._closed = True
        self._discard_connection()

    # ----- internals -------------------------------------------------------

    def _apply_timeout(self, ws: Any, timeout: float) -> None:
        """Apply a socket timeout, tolerating stubs that do not support it.

        ``settimeout`` bounds each socket operation, not a sequence of them, so on the batch
        path this is the *remaining* budget rather than the caller's whole timeout -- see
        :meth:`_deadline_timeout`.

        Args:
            ws: The live connection object.
            timeout: Seconds to allow for the next socket operation.
        """
        settimeout = getattr(ws, "settimeout", None)
        if not callable(settimeout):
            # A stub with no settimeout has no timeout to go stale, so there is nothing to warn
            # about: the budget simply is not applicable.
            logger.debug("Online Service WebSocket connection does not support settimeout")
            return
        try:
            settimeout(timeout)
        except Exception:  # noqa: BLE001 - a closed socket must not break the call
            # Louder than the stub case, and deliberately: a live socket that refuses the
            # budget keeps whatever timeout it already had -- on a fresh connection that is
            # connect_timeout -- so the deadline just derived is not being enforced and an
            # overrun here would otherwise have no trace above DEBUG.
            logger.warning(
                "Online Service WebSocket could not apply a %.3fs timeout; the socket keeps its "
                "previous timeout and this request's deadline is not enforced",
                timeout,
                exc_info=True,
            )

    def _deadline_timeout(self, deadline: float, what: str) -> float:
        """Seconds left until ``deadline``, raising once it has passed.

        ``settimeout`` is per socket operation, so applying the caller's timeout once would
        bound each send and each recv separately rather than the batch. A 100-frame batch with
        a 120s timeout could then block for over three hours while still honouring every
        individual operation -- and both the transport Protocol and this client document the
        value as covering the whole batch. Re-deriving the remaining budget before each
        operation is what makes that documentation true.

        Args:
            deadline: Monotonic time the whole exchange must finish by.
            what: Operation name, for the error message.

        Returns:
            Seconds remaining, never zero or negative.

        Raises:
            SnowflakeMLException: When the deadline has already passed.

        # noqa: DAR401
        # noqa: DAR402
        """
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise _deadline_exceeded(f"before {what}")
        # Floor it: a vanishingly small positive timeout is a non-blocking socket, which fails
        # differently from a timeout and would be reported as an unreachable service.
        return max(remaining, _MIN_SOCKET_TIMEOUT_SECONDS)

    def _record_server_hostname(self, payload: bytes) -> None:
        """Remember which instance answered, when the response says.

        Args:
            payload: Raw response bytes.
        """
        if self._server_hostname is not None:
            return
        try:
            parsed = json.loads(payload.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict):
            return
        diagnostics = parsed.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return
        hostname = diagnostics.get("hostname")
        if isinstance(hostname, str) and hostname.strip():
            self._server_hostname = hostname

    def _discard_connection(self) -> None:
        """Drop the current connection, suppressing close failures."""
        ws, self._ws = self._ws, None
        self._server_max_conn_age_seconds = None
        self._connected_at = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception:  # noqa: BLE001 - best-effort close, never raise from cleanup
            logger.debug("Online Service WebSocket close suppressed", exc_info=True)


def _advertised_conn_age(ws: Any) -> float | None:
    """Read the connection's advertised lifetime off the upgrade response.

    Args:
        ws: The connected socket. Test fakes may not expose response headers at all.

    Returns:
        Positive seconds, or None when absent or unparsable -- the caller then keeps its
        own configured age, which is what it did before the service advertised one.
    """
    getheaders = getattr(ws, "getheaders", None)
    headers = getheaders() if callable(getheaders) else None
    if not headers:
        return None
    try:
        for name, value in headers.items():
            if str(name).strip().lower() == WS_MAX_CONN_AGE_HEADER:
                seconds = float(str(value).strip())
                return seconds if seconds > 0 else None
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def decode_ws_response(raw: str | bytes) -> tuple[int, bytes]:
    """Turn one response frame into the ``(status, bytes)`` pair the query path expects.

    A success envelope becomes ``(200, payload)``. An error envelope becomes the status the
    HTTP path would have returned for the same error code, with the body passed through
    untouched so the caller's error mapping reads ``error.code``, ``error.message`` and
    ``request_id`` exactly as it does over HTTP.

    Args:
        raw: The frame payload, text or bytes.

    Returns:
        ``(status_code, body_bytes)``.
    """
    payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    try:
        parsed = json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        # Let the caller's parser produce the "invalid JSON" error it already has, rather
        # than inventing a second one here.
        return 200, payload

    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        code = parsed["error"].get("code", "")
        return status_for_error_code(str(code)), payload
    return 200, payload


def request_id_for(body: dict[str, Any]) -> str:
    """Return the correlation id to send with ``body``: the caller's when usable, else a fresh one.

    Assigned rather than ``setdefault``-ed: ``setdefault`` only fills an *absent* key, so a
    caller passing ``request_id=""`` or ``None`` would keep it, and a blank id cannot be
    correlated -- which would silently disable the one guard against receiving another
    request's feature values. Both the single-request and the pipelined path use this.

    Args:
        body: The request payload.

    Returns:
        A non-blank id.
    """
    supplied = body.get("request_id")
    return supplied if isinstance(supplied, str) and supplied.strip() else str(uuid.uuid4())


def response_request_id(payload: bytes) -> str | None:
    """Read the correlation id out of a response, wherever the envelope puts it.

    The one extractor both the single-request and the pipelined path use, so they cannot
    disagree on which response belongs to which request. Checks ``request_id`` then
    ``requestId``, at the top level and then under ``error``. Unlike
    ``online_service._try_extract_request_id_from_json_payload``, which reads ids for log
    messages, the value is not stripped: it is compared to the id that was sent, verbatim.

    What to do with ``None`` is the caller's rule, and both paths apply the same one: an
    id-less response is attributed to the request only when exactly one is unanswered.

    Args:
        payload: Raw response bytes.

    Returns:
        The id, or None when the payload is not a JSON object or carries no usable id.
    """
    try:
        parsed = json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    for envelope in (parsed, error if isinstance(error, dict) else {}):
        for key in ("request_id", "requestId"):
            value = envelope.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _assert_response_matches(payload: bytes, expected_request_id: str) -> None:
    """Fail loudly when a response carries a different request's correlation id.

    On a single request/response exchange the ids should always agree. They would not if two
    callers ever shared one connection, and the consequence of letting that pass is the worst
    failure this transport has: a caller receives another request's feature values with no
    error raised. The pool serializes access to prevent it; this makes any residual mismatch
    loud instead of silent.

    A response with no id is accepted: with one request in flight it can only be that
    request's answer. The service sends one when it could not parse a frame far enough to
    read its id (malformed JSON, oversized frame).

    Args:
        payload: Raw response bytes.
        expected_request_id: The id that was sent.

    Raises:
        SnowflakeMLException: When the response carries a different id.
    """
    seen = response_request_id(payload)
    if seen is not None and seen != expected_request_id:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError(
                "Feature retrieval received a response belonging to a different request; "
                "the connection is not being used exclusively"
            ),
        )
