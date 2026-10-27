"""Unit tests for online_service_ws_pipeline (request/response correlation under pipelining).

These are the highest-value tests in the WebSocket transport. Pipelining is the only place
where a defect is *silent*: a caller receives a well-formed response carrying another
request's feature values and no error is raised. Every other failure in this transport throws.

:func:`pipelined_exchange` takes plain callables, so a scripted peer can deliver responses in
a deliberately wrong order — something a real server would not do on demand.
"""

import json
from typing import Any

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store import online_service_ws_pipeline


def _scripted_peer(order: str = "reverse") -> tuple[Any, Any, list[dict[str, Any]]]:
    """Build send/recv callables that echo each request's marker back, in a chosen order.

    Args:
        order: ``"reverse"``, ``"forward"``, or ``"last_first"`` delivery order.

    Returns:
        ``(send, recv, sent)`` where ``sent`` accumulates the decoded request bodies.
    """
    sent: list[dict[str, Any]] = []
    queued: list[str] = []

    def send(frame: str) -> None:
        sent.append(json.loads(frame))

    def recv() -> str:
        if not queued:
            indices = list(range(len(sent)))
            if order == "reverse":
                indices.reverse()
            elif order == "last_first" and len(indices) > 1:
                indices = [indices[-1]] + indices[:-1]
            for index in indices:
                queued.append(
                    json.dumps(
                        {
                            "request_id": sent[index]["request_id"],
                            "status": "SUCCESS",
                            "marker_echo": sent[index]["marker"],
                        }
                    )
                )
        return queued.pop(0)

    return send, recv, sent


def _bodies(count: int) -> list[dict[str, Any]]:
    """Build request bodies carrying a distinct marker each.

    Args:
        count: How many bodies.

    Returns:
        Bodies with distinct ``marker`` values.
    """
    return [{"name": "my_fv", "version": "v1", "marker": index} for index in range(count)]


class PipelinedExchangeTest(absltest.TestCase):
    def test_out_of_order_responses_each_match_their_own_request(self) -> None:
        """The correctness gate: results are positioned by request_id, not by arrival order.

        Every body carries a distinct marker and the peer answers in reverse, so a handler
        that matched positionally would return every value in the wrong slot and still look
        well-formed.
        """
        bodies = _bodies(8)
        markers = [body["marker"] for body in bodies]
        # Guard: without distinct markers this test could pass while correlation is broken.
        self.assertEqual(len(set(markers)), len(markers))

        send, recv, _ = _scripted_peer(order="reverse")
        results = online_service_ws_pipeline.pipelined_exchange(send, recv, bodies)

        self.assertLen(results, len(bodies))
        for position, (status, payload) in enumerate(results):
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["marker_echo"], markers[position])

    def test_depth_one_behaves_like_a_single_request(self) -> None:
        """A one-body batch is the sequential path, so the default configuration is covered."""
        send, recv, sent = _scripted_peer(order="forward")
        results = online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(1))

        self.assertLen(results, 1)
        self.assertLen(sent, 1)
        self.assertEqual(json.loads(results[0][1])["marker_echo"], 0)

    def test_empty_batch_sends_nothing(self) -> None:
        """An empty batch must not touch the connection."""
        send, recv, sent = _scripted_peer()
        self.assertEqual(online_service_ws_pipeline.pipelined_exchange(send, recv, []), [])
        self.assertEmpty(sent)

    def test_every_request_gets_a_request_id(self) -> None:
        """Correlation is only possible if each frame carries a distinct id."""
        send, recv, sent = _scripted_peer()
        online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(5))

        ids = [body["request_id"] for body in sent]
        self.assertLen(ids, 5)
        self.assertLen(set(ids), 5)

    def test_caller_supplied_request_ids_are_preserved(self) -> None:
        """A caller that sets its own ids keeps them, so its own correlation still works."""
        bodies = _bodies(3)
        for index, body in enumerate(bodies):
            body["request_id"] = f"caller-{index}"

        send, recv, sent = _scripted_peer(order="reverse")
        results = online_service_ws_pipeline.pipelined_exchange(send, recv, bodies)

        self.assertEqual([body["request_id"] for body in sent], ["caller-0", "caller-1", "caller-2"])
        for position, (_, payload) in enumerate(results):
            self.assertEqual(json.loads(payload)["marker_echo"], position)

    def test_duplicate_caller_request_ids_are_rejected(self) -> None:
        """Ambiguous ids must fail loudly rather than silently mismatching responses."""
        bodies = _bodies(2)
        for body in bodies:
            body["request_id"] = "same-id"

        send, recv, _ = _scripted_peer()
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service_ws_pipeline.pipelined_exchange(send, recv, bodies)
        self.assertIn("distinct request_id", str(ctx.exception.original_exception))

    def test_unmatched_response_id_raises(self) -> None:
        """A response matching no outstanding request means correlation is broken."""
        sent: list[dict[str, Any]] = []

        def send(frame: str) -> None:
            sent.append(json.loads(frame))

        def recv() -> str:
            return json.dumps({"request_id": "never-sent", "status": "SUCCESS"})

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(2))
        self.assertIn("could not be matched", str(ctx.exception.original_exception))

    def test_response_with_no_request_id_raises_while_several_are_outstanding(self) -> None:
        """With two requests unanswered an id-less response cannot be attributed, so it must not be accepted."""

        def send(frame: str) -> None:
            del frame

        def recv() -> str:
            return json.dumps({"status": "SUCCESS"})

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(2))
        self.assertIn("could not be matched", str(ctx.exception.original_exception))

    def test_id_less_response_is_attributed_to_the_sole_outstanding_request(self) -> None:
        """Same rule as the single-request path: one request unanswered means the answer is unambiguous.

        The service answers a frame it could not parse (oversized, malformed) with an empty
        request_id. Rejecting that would turn a per-request 413 into a failed batch. The window
        is 1 here, so every response arrives with exactly one request outstanding.
        """
        sent: list[dict[str, Any]] = []

        def send(frame: str) -> None:
            sent.append(json.loads(frame))

        def recv() -> str:
            if len(sent) == 2:
                return json.dumps({"request_id": "", "error": {"code": "RESOURCE_EXHAUSTED", "message": "too big"}})
            return json.dumps({"request_id": sent[-1]["request_id"], "marker_echo": sent[-1]["marker"]})

        results = online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(3), max_in_flight=1)

        self.assertEqual([status for status, _ in results], [200, 429, 200])
        self.assertEqual(json.loads(results[2][1])["marker_echo"], 2)

    def test_duplicate_response_for_one_request_raises(self) -> None:
        """Answering one request twice would leave another unanswered; fail rather than guess."""
        sent: list[dict[str, Any]] = []

        def send(frame: str) -> None:
            sent.append(json.loads(frame))

        def recv() -> str:
            return json.dumps({"request_id": sent[0]["request_id"], "status": "SUCCESS"})

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(2))
        self.assertIn("two responses for the same request", str(ctx.exception.original_exception))

    def test_error_envelope_keeps_its_status_and_position(self) -> None:
        """A per-request error must land in its own slot, not fail the whole batch."""
        sent: list[dict[str, Any]] = []
        queued: list[str] = []

        def send(frame: str) -> None:
            sent.append(json.loads(frame))

        def recv() -> str:
            if not queued:
                queued.append(
                    json.dumps(
                        {
                            "request_id": sent[1]["request_id"],
                            "error": {"code": "RESOURCE_EXHAUSTED", "message": "slow down"},
                            "retry_after_seconds": 2,
                        }
                    )
                )
                queued.append(json.dumps({"request_id": sent[0]["request_id"], "status": "SUCCESS"}))
            return queued.pop(0)

        results = online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(2))

        self.assertEqual(results[0][0], 200)
        self.assertEqual(results[1][0], 429)
        self.assertEqual(json.loads(results[1][1])["retry_after_seconds"], 2)

    def test_error_envelope_request_id_is_read_from_the_error_object(self) -> None:
        """Some error envelopes carry the id inside ``error``; correlation must still work."""
        sent: list[dict[str, Any]] = []

        def send(frame: str) -> None:
            sent.append(json.loads(frame))

        def recv() -> str:
            return json.dumps({"error": {"code": "NOT_FOUND", "request_id": sent[0]["request_id"]}})

        results = online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(1))
        self.assertEqual(results[0][0], 404)


def _backpressured_peer(window: int) -> tuple[Any, Any, list[dict[str, Any]], list[int]]:
    """Build a peer that answers in order and refuses a send once ``window`` frames are unanswered.

    Models a server that handles one connection's frames in order with blocking writes: past
    a point it stops reading, and the client's next send would block forever. Raising makes
    that a test failure instead of a hang.

    Args:
        window: Most frames the peer accepts unanswered.

    Returns:
        ``(send, recv, sent, unanswered_at_each_send)``.
    """
    sent: list[dict[str, Any]] = []
    unanswered: list[dict[str, Any]] = []
    unanswered_at_each_send: list[int] = []

    def send(frame: str) -> None:
        if len(unanswered) >= window:
            raise AssertionError(f"send with {len(unanswered)} frames unanswered; the peer would block")
        body = json.loads(frame)
        sent.append(body)
        unanswered.append(body)
        unanswered_at_each_send.append(len(unanswered))

    def recv() -> str:
        body = unanswered.pop(0)
        return json.dumps({"request_id": body["request_id"], "marker_echo": body["marker"]})

    return send, recv, sent, unanswered_at_each_send


class InFlightWindowTest(absltest.TestCase):
    def test_never_exceeds_the_window_and_keeps_request_order(self) -> None:
        """A large read must not deadlock against a server that stops reading once its writes back up."""
        send, recv, sent, unanswered_at_each_send = _backpressured_peer(window=4)

        results = online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(50), max_in_flight=4)

        self.assertEqual(len(sent), 50)
        self.assertLessEqual(max(unanswered_at_each_send), 4)
        self.assertEqual([json.loads(payload)["marker_echo"] for _, payload in results], list(range(50)))

    def test_the_window_is_filled_before_the_first_read(self) -> None:
        """The bound must not cost the round-trip saving: a batch within the window still goes out in one burst."""
        sends_before_first_read: list[int] = []
        send, peer_recv, sent, _ = _backpressured_peer(window=8)

        def recv() -> str:
            if not sends_before_first_read:
                sends_before_first_read.append(len(sent))
            received: str = peer_recv()
            return received

        online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(8), max_in_flight=8)

        self.assertEqual(sends_before_first_read, [8])

    def test_default_window_bounds_an_unconfigured_call(self) -> None:
        """Callers that never pass max_in_flight are the ones most likely to send a large burst."""
        window = online_service_ws_pipeline.DEFAULT_MAX_IN_FLIGHT
        send, recv, sent, unanswered_at_each_send = _backpressured_peer(window=window)

        online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(window * 3))

        self.assertEqual(len(sent), window * 3)
        self.assertEqual(max(unanswered_at_each_send), window)

    def test_out_of_order_answers_within_the_window_still_match(self) -> None:
        sent: list[dict[str, Any]] = []
        unanswered: list[dict[str, Any]] = []

        def send(frame: str) -> None:
            body = json.loads(frame)
            sent.append(body)
            unanswered.append(body)

        def recv() -> str:
            body = unanswered.pop()  # newest first
            return json.dumps({"request_id": body["request_id"], "marker_echo": body["marker"]})

        results = online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(10), max_in_flight=3)

        self.assertEqual([json.loads(payload)["marker_echo"] for _, payload in results], list(range(10)))

    def test_a_window_below_one_is_rejected(self) -> None:
        send, recv, sent, _ = _backpressured_peer(window=4)

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service_ws_pipeline.pipelined_exchange(send, recv, _bodies(2), max_in_flight=0)

        self.assertIn("max_in_flight", str(ctx.exception.original_exception))
        self.assertEqual(sent, [])


class FrameRequestsTest(absltest.TestCase):
    def test_each_frame_carries_its_id_and_the_caller_body(self) -> None:
        framed = online_service_ws_pipeline.frame_requests([{"marker": 0, "request_id": "mine"}, {"marker": 1}])

        self.assertEqual(framed[0][0], "mine")
        for request_id, frame in framed:
            self.assertEqual(json.loads(frame)["request_id"], request_id)
        self.assertEqual([json.loads(frame)["marker"] for _, frame in framed], [0, 1])

    def test_blank_or_non_string_ids_are_replaced(self) -> None:
        framed = online_service_ws_pipeline.frame_requests(
            [{"request_id": ""}, {"request_id": None}, {"request_id": 7}]
        )

        ids = [request_id for request_id, _ in framed]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(isinstance(i, str) and i.strip() for i in ids))


if __name__ == "__main__":
    absltest.main()
