"""Pipelined request/response exchange over one WebSocket connection.

Pipelining means sending requests without waiting for each one's response. It removes
the round-trip gap between sends, which on a reads-heavy path is a larger term than any
server-side change: a read of N batches costs one round trip instead of N.

The cost is that **responses no longer arrive in the order the requests were sent**, so
every response must be matched to its own request by ``request_id``. Getting that wrong is
the one failure in this transport that is silent — the caller receives a well-formed
response carrying another request's feature values, with no error raised. That is why the
matching logic lives here, in a module small enough to read in full, and why
:func:`pipelined_exchange` takes plain callables so it can be tested without a socket.

The burst is bounded. At most ``max_in_flight`` requests are unanswered at once; past that the
exchange alternates one read with one send. An unbounded burst deadlocks against a server that
answers frames in order with blocking writes: once both sides' socket buffers fill, the server
stops reading, the client's ``send`` blocks, and a large read times out where the same read over
HTTP succeeds. A window of a few dozen keeps nearly all of the round-trip saving.

The exchange is deliberately synchronous. Interleaving sends and reads needs no event loop,
which keeps this consistent with the synchronous HTTP client beside it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.feature_store import online_service_ws_client

logger = logging.getLogger(__name__)

SendFrame = Callable[[str], None]
RecvFrame = Callable[[], "str | bytes"]

# Requests allowed unanswered at once on one connection. Large enough that the window rarely
# waits on a round trip, small enough that the outstanding frames and their responses fit in
# ordinary socket buffers, so a server writing responses in order never blocks on us.
DEFAULT_MAX_IN_FLIGHT = 32

# One request's correlation id and its serialized frame.
FramedRequest = tuple[str, str]


def frame_requests(bodies: list[dict[str, Any]]) -> list[FramedRequest]:
    """Assign one correlation id per body and serialize each, without touching a socket.

    Separate from the exchange so the caller can reject bad input before it connects: a caller
    error must not cost a healthy connection its handshake.

    Args:
        bodies: Request payloads, in caller order.

    Returns:
        One ``(request_id, frame)`` per body, positionally aligned.

    Raises:
        SnowflakeMLException: When the caller supplied duplicate ids, which would make
            correlation ambiguous.
    """
    ids = [online_service_ws_client.request_id_for(body) for body in bodies]

    if len(set(ids)) != len(ids):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Pipelined feature retrieval requires a distinct request_id per request; "
                "duplicate ids make it impossible to tell which response belongs to which request"
            ),
        )

    framed: list[FramedRequest] = []
    for body, request_id in zip(bodies, ids):
        request_body = dict(body)
        request_body["request_id"] = request_id
        framed.append((request_id, json.dumps(request_body)))
    return framed


def pipelined_exchange(
    send: SendFrame,
    recv: RecvFrame,
    bodies: list[dict[str, Any]],
    *,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
) -> list[tuple[int, bytes]]:
    """Pipeline every body over one connection and match each response by ``request_id``.

    Args:
        send: Sends one serialized frame.
        recv: Blocks until one response frame arrives.
        bodies: Request payloads. Results come back in this order regardless of arrival
            order.
        max_in_flight: Most requests left unanswered at once.

    Returns:
        One ``(status_code, body_bytes)`` per entry in ``bodies``, in request order.
    """
    return exchange_frames(send, recv, frame_requests(bodies), max_in_flight=max_in_flight)


def exchange_frames(
    send: SendFrame,
    recv: RecvFrame,
    framed: list[FramedRequest],
    *,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
) -> list[tuple[int, bytes]]:
    """Send frames from :func:`frame_requests` through a bounded window and read one response each.

    Up to ``max_in_flight`` frames go out before the first read; after that each response read
    frees one slot for the next send.

    Args:
        send: Sends one serialized frame.
        recv: Blocks until one response frame arrives. A peer that stops answering surfaces
            here, as whatever ``recv`` raises.
        framed: Output of :func:`frame_requests`, in the order results are wanted.
        max_in_flight: Most requests left unanswered at once.

    Returns:
        One ``(status_code, body_bytes)`` per frame, in request order.

    Raises:
        SnowflakeMLException: When ``max_in_flight`` is below 1, when a response cannot be
            attributed to an outstanding request, or when two responses answer the same request.
    """
    if max_in_flight < 1:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(f"max_in_flight must be at least 1, got {max_in_flight}"),
        )
    total = len(framed)
    if not total:
        return []

    index_by_id = {request_id: position for position, (request_id, _) in enumerate(framed)}
    results: list[tuple[int, bytes] | None] = [None] * total
    outstanding: set[str] = set()
    next_to_send = 0

    for _ in range(total):
        while next_to_send < total and len(outstanding) < max_in_flight:
            request_id, frame = framed[next_to_send]
            send(frame)
            outstanding.add(request_id)
            next_to_send += 1

        status, payload = online_service_ws_client.decode_ws_response(recv())
        response_id = online_service_ws_client.response_request_id(payload)
        if response_id is None and len(outstanding) == 1:
            # Same rule as the single-request path: with exactly one request unanswered, an
            # id-less response can only be its answer. The service sends one for a frame it
            # could not parse far enough to read the id from (malformed JSON, oversized).
            response_id = next(iter(outstanding))

        if response_id is None or response_id not in index_by_id:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(
                    "Feature retrieval response could not be matched to a request; "
                    f"{len(outstanding)} of {total} requests were still awaiting a response"
                ),
            )
        if response_id not in outstanding:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError("Feature retrieval returned two responses for the same request"),
            )

        outstanding.discard(response_id)
        results[index_by_id[response_id]] = (status, payload)

    # Every slot is filled: one response was read per frame, each matched a distinct
    # outstanding id, and none was rejected above.
    return [result for result in results if result is not None]
