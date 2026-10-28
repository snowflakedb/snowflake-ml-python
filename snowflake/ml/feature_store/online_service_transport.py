"""Transport protocol for the Online Service query and ingest APIs.

The query and ingest paths in :mod:`online_service` have always been duck-typed against
:class:`online_service_http_client.OnlineServiceHttpClient`: they call only ``post_json``
and ``close``. This module states that contract explicitly so a second transport can
satisfy it and be type-checked against it.

``post_json_many`` is an optional capability. A transport that can answer several request
bodies in one round trip (a WebSocket connection pipelining a burst of frames, for example)
may implement it, and say through ``send_ahead_window`` how many bodies are worth sending before
the first response is checked. Callers go through :func:`send_ahead_window`, which is 1 -- send,
check, send -- for any transport that does not batch.

The protocol covers both APIs, but a given transport need not serve both. The WebSocket
client serves the query API only: ingest has no WebSocket endpoint and depends on 207 and
``partial_success``, which a WebSocket frame does not carry.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OnlineServiceTransport(Protocol):
    """The two-method surface the Online Service query and ingest paths depend on."""

    def post_json(self, url: str, body: dict[str, Any], timeout: float) -> tuple[int, bytes]:
        """Send one request body and return ``(status_code, body_bytes)``.

        Implementations return the status rather than raising on it: status interpretation
        belongs to the caller, which owns the error-mapping table.

        The pair is ``(status_code, body_bytes)``.

        Args:
            url: Online Service endpoint URL.
            body: JSON-serializable payload.
            timeout: Per-request timeout in seconds.
        """
        ...

    def close(self) -> None:
        """Release transport resources. Must be idempotent."""
        ...


@runtime_checkable
class BatchingOnlineServiceTransport(OnlineServiceTransport, Protocol):
    """A transport that can answer several bodies in fewer round trips than one each."""

    def post_json_many(self, url: str, bodies: list[dict[str, Any]], timeout: float) -> list[tuple[int, bytes]]:
        """Send several request bodies and return their results in request order.

        Implementations may send all bodies before reading any response, so the returned
        list is ordered to match ``bodies`` regardless of the order answers arrived in.

        One ``(status_code, body_bytes)`` comes back per entry in ``bodies``, same order.

        Args:
            url: Online Service endpoint URL.
            bodies: JSON-serializable payloads, in the order the caller wants results.
            timeout: Timeout in seconds for the whole batch.
        """
        ...


def supports_batching(transport: object) -> bool:
    """Report whether ``transport`` can answer several bodies in one batch.

    Args:
        transport: Any transport instance.

    Returns:
        True when ``post_json_many`` is available and callable.
    """
    return callable(getattr(transport, "post_json_many", None))


def send_ahead_window(transport: object) -> int:
    """How many request bodies a caller may send before checking any response.

    Sending ahead is only worth it when the transport answers several bodies in one round trip
    right now. Otherwise each body costs its own round trip anyway, and sending the rest before
    looking at the first turns a fast failure (a 400 on batch 1) into one that waits on every
    batch, holds every response in memory, and adds load exactly when the service is struggling.

    Args:
        transport: Any transport instance.

    Returns:
        The transport's ``send_ahead_window`` when it can batch, otherwise 1.
    """
    if not supports_batching(transport):
        return 1
    window = getattr(transport, "send_ahead_window", 1)
    return window if isinstance(window, int) and window > 1 else 1
