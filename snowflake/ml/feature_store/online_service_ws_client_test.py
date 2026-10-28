"""Unit tests for online_service_ws_client (URL derivation, handshake auth, status synthesis).

The client's job is to be indistinguishable from the HTTP client at the ``post_json`` seam, so
most of these tests assert equivalence rather than WebSocket-specific behaviour: the same
status codes, the same closed-client error, the same auth scheme.
"""

import json
import os
from typing import Any
from unittest.mock import patch

import websocket
from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store import online_service_ws_client

_UNITTEST_SNOWFLAKE_PAT = "unittest-snowflake-pat-token"


class _FakeWebSocket:
    """Stand-in for a live connection, scripted with the frames it should return."""

    def __init__(self, frames: list[str] | None = None, *, recv_error: Exception | None = None) -> None:
        self.frames = list(frames or [])
        self.recv_error = recv_error
        self.sent: list[str] = []
        self.timeouts: list[float] = []
        self.closed = False

    def send(self, frame: str) -> None:
        self.sent.append(frame)

    def recv(self) -> str:
        if self.recv_error is not None:
            raise self.recv_error
        if self.frames:
            return self.frames.pop(0)
        # Echo the id back, as the service does, so correlation can be exercised.
        last = json.loads(self.sent[-1]) if self.sent else {}
        return json.dumps({"status": "SUCCESS", "request_id": last.get("request_id", "")})

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def close(self) -> None:
        self.closed = True


def _factory_for(ws: Any, *, dial_error: Exception | None = None) -> Any:
    """Build a ``_ws_factory`` that records the dial arguments.

    Args:
        ws: The connection object to return.
        dial_error: When set, raised instead of connecting.

    Returns:
        A factory with a ``calls`` list attribute.
    """
    calls: list[dict[str, Any]] = []

    def factory(
        url: str,
        header: list[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        calls.append({"url": url, "header": list(header or []), "timeout": timeout})
        if dial_error is not None:
            raise dial_error
        return ws

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


def _client(ws: Any, **kwargs: Any) -> online_service_ws_client.OnlineServiceWsClient:
    """Build a client wired to a fake connection.

    Args:
        ws: The fake connection.
        **kwargs: Passed through to the client.

    Returns:
        The client.
    """
    return online_service_ws_client.OnlineServiceWsClient(_ws_factory=_factory_for(ws), **kwargs)


class WsQueryUrlTest(absltest.TestCase):
    def test_https_becomes_wss_with_the_query_ws_path(self) -> None:
        """The WebSocket URL is derived from the HTTP query path so the two cannot drift."""
        self.assertEqual(
            online_service_ws_client.ws_query_url("https://svc.example.app"),
            "wss://svc.example.app/api/v1/query/ws",
        )

    def test_trailing_slash_is_idempotent(self) -> None:
        """Endpoint URLs from the status payload may or may not have a trailing slash."""
        self.assertEqual(
            online_service_ws_client.ws_query_url("https://svc.example.app/"),
            "wss://svc.example.app/api/v1/query/ws",
        )

    def test_http_becomes_ws(self) -> None:
        """Plain http is supported so a local service can be targeted."""
        self.assertEqual(
            online_service_ws_client.ws_query_url("http://localhost:8081"),
            "ws://localhost:8081/api/v1/query/ws",
        )

    def test_non_http_scheme_is_rejected(self) -> None:
        """A malformed endpoint URL should fail with a clear argument error."""
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service_ws_client.ws_query_url("ftp://svc.example.app")
        self.assertIn("http or https", str(ctx.exception.original_exception))


class StatusSynthesisTest(absltest.TestCase):
    def test_known_codes_map_to_their_http_equivalents(self) -> None:
        """The envelope carries no status, so codes must reconstruct what HTTP would return."""
        for code, expected in (
            ("INVALID_ARGUMENT", 400),
            ("UNAUTHENTICATED", 401),
            ("PERMISSION_DENIED", 403),
            ("NOT_FOUND", 404),
            ("RESOURCE_EXHAUSTED", 429),
            ("INTERNAL", 500),
            ("UNAVAILABLE", 503),
        ):
            self.assertEqual(online_service_ws_client.status_for_error_code(code), expected, code)

    def test_unknown_code_is_treated_as_a_server_error(self) -> None:
        """A code this client does not know must not be mistaken for success."""
        self.assertEqual(online_service_ws_client.status_for_error_code("SOMETHING_NEW"), 500)

    def test_code_matching_ignores_case_and_padding(self) -> None:
        """Defensive: a code should not fall through to 500 over whitespace."""
        self.assertEqual(online_service_ws_client.status_for_error_code("  not_found "), 404)

    def test_success_envelope_decodes_as_200_with_the_body_untouched(self) -> None:
        """The caller parses the body itself, so it must arrive byte-identical."""
        payload = json.dumps({"status": "SUCCESS", "results": []})
        status, raw = online_service_ws_client.decode_ws_response(payload)
        self.assertEqual(status, 200)
        self.assertEqual(raw, payload.encode("utf-8"))

    def test_bytes_frames_are_accepted(self) -> None:
        """Some libraries hand back bytes rather than str."""
        status, raw = online_service_ws_client.decode_ws_response(b'{"status":"SUCCESS"}')
        self.assertEqual(status, 200)
        self.assertEqual(raw, b'{"status":"SUCCESS"}')

    def test_unparsable_frame_is_left_to_the_callers_parser(self) -> None:
        """Inventing a second invalid-JSON error here would duplicate one the caller has."""
        status, raw = online_service_ws_client.decode_ws_response("not json")
        self.assertEqual(status, 200)
        self.assertEqual(raw, b"not json")


class OnlineServiceWsClientTest(absltest.TestCase):
    def test_no_connection_is_opened_until_the_first_request(self) -> None:
        """Construction must stay cheap, matching the HTTP client."""
        ws = _FakeWebSocket()
        client = _client(ws)
        self.assertFalse(client.is_connected)

    def test_handshake_reuses_auth_headers_without_content_type(self) -> None:
        """Auth is reused so both transports keep working; a handshake has no body to type."""
        ws = _FakeWebSocket()
        factory = _factory_for(ws)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc.example.app/api/v1/query/ws", {"name": "fv"}, 30.0)

        headers = factory.calls[0]["header"]
        self.assertIn(f'Authorization: Snowflake Token="{_UNITTEST_SNOWFLAKE_PAT}"', headers)
        self.assertNotIn("Content-Type: application/json", headers)

    def test_real_dial_trusts_the_same_ca_bundle_as_http(self) -> None:
        """Without this the dial uses the system store and fails TLS where HTTP succeeds."""
        import certifi

        ws = _FakeWebSocket()
        client = online_service_ws_client.OnlineServiceWsClient()

        with patch.object(websocket, "create_connection", return_value=ws) as create, patch.dict(
            os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False
        ):
            client.post_json("wss://svc.example.app/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertEqual(create.call_args.kwargs["sslopt"], {"ca_certs": certifi.where()})

    def test_request_id_is_added_when_the_caller_did_not_set_one(self) -> None:
        """A frame has no headers, so the correlation id has to travel in the body."""
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertIn("request_id", json.loads(ws.sent[0]))

    def test_caller_request_id_is_not_overwritten(self) -> None:
        """A caller that supplies an id keeps it, so its own correlation still works."""
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": "mine"}, 30.0)

        self.assertEqual(json.loads(ws.sent[0])["request_id"], "mine")

    def test_caller_body_is_not_mutated(self) -> None:
        """Adding an id must not surprise a caller that reuses its body dict."""
        ws = _FakeWebSocket()
        client = _client(ws)
        body: dict[str, Any] = {"name": "fv"}

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", body, 30.0)

        self.assertNotIn("request_id", body)

    def test_error_envelope_becomes_its_synthesized_status(self) -> None:
        """Callers branch on status, so an error frame must not read as a success."""
        frame = json.dumps({"error": {"code": "NOT_FOUND", "message": "no such feature view"}})
        ws = _FakeWebSocket([frame])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            status, raw = client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertEqual(status, 404)
        self.assertEqual(json.loads(raw)["error"]["code"], "NOT_FOUND")

    def test_connection_is_reused_across_requests(self) -> None:
        """Reuse is the entire point: one handshake is one authentication."""
        ws = _FakeWebSocket()
        factory = _factory_for(ws)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            for _ in range(3):
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertLen(factory.calls, 1)
        self.assertEqual(client.request_count, 3)

    def test_per_request_timeout_is_applied(self) -> None:
        """A request must not be able to block for longer than the caller allowed.

        One budget per socket operation rather than one for the request: ``settimeout`` bounds
        each operation, so a single call would have allowed the send and the recv a full timeout
        each. Both are derived from one deadline, so neither exceeds the caller's budget and the
        second is no larger than the first. Values are approximate because this case runs on the
        real clock -- ``BatchTimeoutIsADeadlineTest`` pins the arithmetic exactly with an
        injected one.
        """
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 12.5)

        self.assertLen(ws.timeouts, 2, "the send and the recv should each get a derived budget")
        for applied in ws.timeouts:
            self.assertLessEqual(applied, 12.5, "an operation was given more than the caller allowed")
            self.assertAlmostEqual(applied, 12.5, places=2)
        self.assertLessEqual(ws.timeouts[1], ws.timeouts[0])

    def test_mid_request_failure_wraps_and_drops_the_connection(self) -> None:
        """A broken connection must not be reused, and must raise the shared unreachable error."""
        ws = _FakeWebSocket(recv_error=OSError("socket went away"))
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertIn("unreachable", str(ctx.exception.original_exception))
        self.assertFalse(client.is_connected)
        self.assertTrue(ws.closed)

    def test_close_is_idempotent_and_post_close_calls_raise(self) -> None:
        """Matches OnlineServiceHttpClient so callers can treat the two identically."""
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)
        client.close()
        client.close()

        self.assertTrue(ws.closed)
        with self.assertRaises(RuntimeError):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

    def test_batch_of_one_matches_a_single_request(self) -> None:
        """Depth 1 is the shipped default, so it must behave exactly as post_json does."""
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            results = client.post_json_many("wss://svc/api/v1/query/ws", [{"name": "fv"}], 30.0)

        self.assertLen(results, 1)
        self.assertEqual(results[0][0], 200)

    def test_empty_batch_does_not_connect(self) -> None:
        """An empty read should cost nothing."""
        ws = _FakeWebSocket()
        factory = _factory_for(ws)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory)
        self.assertEqual(client.post_json_many("wss://svc/api/v1/query/ws", [], 30.0), [])
        self.assertEmpty(factory.calls)


class DialErrorClassificationTest(absltest.TestCase):
    """The three refusals mean different things and warrant different client behaviour."""

    def _dial(self, status: int | None) -> online_service_ws_client.OnlineServiceWsDialError:
        """Dial against a factory that refuses with ``status``.

        Args:
            status: Handshake status to simulate, or None for a bare failure.

        Returns:
            The raised dial error.
        """
        error: Exception = RuntimeError("refused")
        if status is not None:
            error.status_code = status  # type: ignore[attr-defined]
        client = online_service_ws_client.OnlineServiceWsClient(
            _ws_factory=_factory_for(None, dial_error=error),
        )
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError) as ctx:
                client.connect("wss://svc/api/v1/query/ws")
        refused: online_service_ws_client.OnlineServiceWsDialError = ctx.exception
        return refused

    def test_404_means_the_transport_is_not_served(self) -> None:
        """Retrying a transport that is switched off server-side is pointless."""
        error = self._dial(404)
        self.assertTrue(error.transport_unavailable)
        self.assertFalse(error.at_capacity)

    def test_503_means_at_capacity_and_worth_retrying(self) -> None:
        """A later redial may succeed, so this must not be confused with 404."""
        error = self._dial(503)
        self.assertTrue(error.at_capacity)
        self.assertFalse(error.transport_unavailable)

    def test_401_and_403_mean_credentials_were_rejected(self) -> None:
        """Redialing without new credentials would just repeat the rejection."""
        self.assertTrue(self._dial(401).auth_rejected)
        self.assertTrue(self._dial(403).auth_rejected)

    def test_bare_failure_classifies_as_nothing_specific(self) -> None:
        """A network failure with no status must not be read as any of the three refusals."""
        error = self._dial(None)
        self.assertIsNone(error.status)
        self.assertFalse(error.transport_unavailable)
        self.assertFalse(error.at_capacity)
        self.assertFalse(error.auth_rejected)


class ResponseCorrelationGuardTest(absltest.TestCase):
    def test_a_response_for_another_request_raises_instead_of_being_returned(self) -> None:
        """The worst failure this transport could have is returning another request's feature
        values silently. The pool prevents the cause; this makes any residual mismatch loud."""
        ws = _FakeWebSocket([json.dumps({"status": "SUCCESS", "request_id": "somebody-elses-id"})])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": "mine"}, 30.0)

        self.assertIn("different request", str(ctx.exception.original_exception))

    def test_a_mismatched_response_discards_the_connection(self) -> None:
        """A mismatch means the stream is misaligned, so the socket must not be reused.

        The response this request was owed is still queued, and the next request on this
        connection would read it -- so keeping the connection turns one bad exchange into a
        failure for every later caller that draws it. Raising alone is not enough: without the
        discard, a caller that retries re-sends on the same misaligned socket.
        """
        ws = _FakeWebSocket([json.dumps({"status": "SUCCESS", "request_id": "somebody-elses-id"})])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": "mine"}, 30.0)

            self.assertFalse(client.is_connected, "a misaligned connection was kept")
            self.assertTrue(ws.closed, "the misaligned socket was not closed")
            # The client itself is not closed, only its socket, so the next use re-upgrades
            # rather than raising "closed" from here on.
            client.connect("wss://svc/api/v1/query/ws")
            self.assertTrue(client.is_connected, "the client could not re-upgrade after a discard")

    def test_a_blank_caller_supplied_id_does_not_disable_the_check(self) -> None:
        """A caller passing request_id="" must not silently turn the correlation guard off.

        ``setdefault`` only fills an absent key, so a blank id used to survive and the guard
        skips blank ids -- meaning a caller could disable the only protection against receiving
        another request's feature values without any sign it had happened.
        """
        for supplied in ("", "   ", None, 12345):
            with self.subTest(supplied=supplied):
                ws = _FakeWebSocket([json.dumps({"status": "SUCCESS", "request_id": "somebody-elses-id"})])
                client = _client(ws)

                with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
                    with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                        client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": supplied}, 30.0)

                self.assertIn("different request", str(ctx.exception.original_exception))
                # A real id was sent in place of the blank one.
                sent = json.loads(ws.sent[0])
                self.assertIsInstance(sent["request_id"], str)
                self.assertTrue(sent["request_id"].strip())

    def test_a_caller_supplied_id_is_preserved(self) -> None:
        """A usable id from the caller must be sent unchanged, so their correlation still works."""
        ws = _FakeWebSocket([json.dumps({"status": "SUCCESS", "request_id": "mine"})])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": "mine"}, 30.0)

        self.assertEqual("mine", json.loads(ws.sent[0])["request_id"])

    def test_a_response_with_no_request_id_is_accepted(self) -> None:
        """Older services may not echo the id; that must not break a single exchange."""
        ws = _FakeWebSocket([json.dumps({"status": "SUCCESS"})])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            status, _ = client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)
        self.assertEqual(status, 200)

    def test_a_mismatched_id_inside_an_error_envelope_raises(self) -> None:
        """Both paths read the id with one extractor, so an id under ``error`` is checked here too.

        The single-request guard used to read only the top-level ``request_id``, so an error
        envelope belonging to another request passed it -- while the pipelined path, reading
        ``error.request_id``, would have rejected the same frame.
        """
        for envelope in (
            {"error": {"code": "NOT_FOUND", "request_id": "somebody-elses-id"}},
            {"error": {"code": "NOT_FOUND"}, "requestId": "somebody-elses-id"},
        ):
            with self.subTest(envelope=envelope):
                ws = _FakeWebSocket([json.dumps(envelope)])
                client = _client(ws)

                with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
                    with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                        client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": "mine"}, 30.0)

                self.assertIn("different request", str(ctx.exception.original_exception))
                self.assertFalse(client.is_connected)

    def test_both_paths_extract_the_same_id(self) -> None:
        cases = {
            b'{"request_id": "a"}': "a",
            b'{"requestId": "a"}': "a",
            b'{"error": {"request_id": "a"}}': "a",
            b'{"error": {"requestId": "a"}}': "a",
            b'{"request_id": "", "error": {"request_id": "a"}}': "a",
            b'{"request_id": " a "}': " a ",
            b'{"request_id": ""}': None,
            b'{"request_id": {"nested": 1}}': None,
            b"[1, 2]": None,
            b"not json": None,
        }
        for payload, expected in cases.items():
            with self.subTest(payload=payload):
                self.assertEqual(online_service_ws_client.response_request_id(payload), expected)


class CallerErrorKeepsConnectionTest(absltest.TestCase):
    """A caller's mistake is not evidence the socket is bad, so it must not cost a re-handshake."""

    def test_duplicate_batch_ids_raise_before_dialing(self) -> None:
        ws = _FakeWebSocket()
        factory = _factory_for(ws)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json_many(
                    "wss://svc/api/v1/query/ws",
                    [{"name": "fv", "request_id": "same"}, {"name": "fv", "request_id": "same"}],
                    30.0,
                )

        self.assertIn("distinct request_id", str(ctx.exception.original_exception))
        self.assertEqual(factory.calls, [], "a caller error opened a connection")
        self.assertEqual(ws.sent, [])

    def test_duplicate_batch_ids_keep_an_open_connection(self) -> None:
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json_many(
                    "wss://svc/api/v1/query/ws",
                    [{"name": "fv", "request_id": "same"}, {"name": "fv", "request_id": "same"}],
                    30.0,
                )

        self.assertTrue(client.is_connected, "a caller error discarded a healthy connection")
        self.assertFalse(ws.closed)

    def test_an_unserializable_body_keeps_an_open_connection(self) -> None:
        """Matches the HTTP client, which serializes before it touches the network."""
        ws = _FakeWebSocket()
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)
            with self.assertRaises(TypeError):
                client.post_json("wss://svc/api/v1/query/ws", {"name": object()}, 30.0)

        self.assertTrue(client.is_connected)
        self.assertFalse(ws.closed)
        self.assertEqual(len(ws.sent), 1)


class _CountingWebSocket(_FakeWebSocket):
    """Answers in order, recording how many frames were unanswered at each send."""

    def __init__(self) -> None:
        super().__init__()
        self.unanswered: list[str] = []
        self.unanswered_at_each_send: list[int] = []

    def send(self, frame: str) -> None:
        super().send(frame)
        self.unanswered.append(json.loads(frame)["request_id"])
        self.unanswered_at_each_send.append(len(self.unanswered))

    def recv(self) -> str:
        return json.dumps({"status": "SUCCESS", "request_id": self.unanswered.pop(0)})


class InFlightWindowClientTest(absltest.TestCase):
    def test_configured_window_reaches_the_exchange(self) -> None:
        ws = _CountingWebSocket()
        client = _client(ws, max_in_flight=2)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            results = client.post_json_many("wss://svc/api/v1/query/ws", [{"name": "fv"}] * 7, 30.0)

        self.assertEqual(len(results), 7)
        self.assertEqual(max(ws.unanswered_at_each_send), 2)
        self.assertEqual(client.request_count, 7)


class ServerHostnameTest(absltest.TestCase):
    def test_hostname_is_recorded_from_response_diagnostics(self) -> None:
        """The gateway picks the instance and never tells the client, so the only way to see
        where a connection landed is to read what its responses report."""
        frame = json.dumps({"status": "SUCCESS", "request_id": "x", "diagnostics": {"hostname": "statefulset-1"}})
        ws = _FakeWebSocket([frame])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "request_id": "x"}, 30.0)

        self.assertEqual(client.server_hostname, "statefulset-1")

    def test_hostname_is_none_when_the_server_does_not_report_one(self) -> None:
        """The field is optional on the wire, so its absence must not be invented."""
        ws = _FakeWebSocket([json.dumps({"status": "SUCCESS"})])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertIsNone(client.server_hostname)


class HandshakeRetryAfterTest(absltest.TestCase):
    """A refused handshake is still an HTTP response, so Retry-After arrives as a header."""

    def test_reads_delta_seconds_from_response_headers(self) -> None:
        error = RuntimeError("503")
        error.resp_headers = {"Retry-After": "7", "Server": "envoy"}  # type: ignore[attr-defined]
        self.assertEqual(online_service_ws_client.retry_after_of_handshake(error), 7)

    def test_header_name_is_matched_case_insensitively(self) -> None:
        error = RuntimeError("503")
        error.resp_headers = [("retry-after", "3")]  # type: ignore[attr-defined]
        self.assertEqual(online_service_ws_client.retry_after_of_handshake(error), 3)

    def test_absent_or_unparsable_reads_as_no_guidance(self) -> None:
        """Zero rather than a guess: an invented wait is worse than none."""
        for headers in (None, {}, {"Retry-After": ""}, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}):
            with self.subTest(headers=headers):
                error = RuntimeError("503")
                if headers is not None:
                    error.resp_headers = headers  # type: ignore[attr-defined]
                self.assertEqual(online_service_ws_client.retry_after_of_handshake(error), 0)

    def test_negative_values_clamp_to_zero(self) -> None:
        error = RuntimeError("503")
        error.resp_headers = {"Retry-After": "-5"}  # type: ignore[attr-defined]
        self.assertEqual(online_service_ws_client.retry_after_of_handshake(error), 0)

    def test_dial_error_carries_the_delay_off_the_handshake(self) -> None:
        """End to end: a refused dial surfaces both the status and the requested wait."""

        def _refuse(*_args: object, **_kwargs: object) -> object:
            error = RuntimeError("at capacity")
            error.status_code = 503  # type: ignore[attr-defined]
            error.resp_headers = {"Retry-After": "9"}  # type: ignore[attr-defined]
            raise error

        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=_refuse)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(online_service_ws_client.OnlineServiceWsDialError) as caught:
                client.connect("wss://svc.example.app/api/v1/query/ws")

        self.assertTrue(caught.exception.at_capacity)
        self.assertEqual(caught.exception.retry_after_seconds, 9)


class ServerClosedConnectionTest(absltest.TestCase):
    """A close must surface as a failure, never as an empty success.

    ``websocket-client``'s ``recv()`` returns ``""`` on a close frame instead of raising, so a
    graceful close is the one disconnect that can reach the decoder as if it were data. Decoded it
    becomes an empty ``200``: the transport counts a dead socket as served, does not redial, and does
    not fall back to HTTP. A reset or hard kill raises from ``recv()`` and needs none of this, which
    is what makes the well-behaved close the dangerous one.
    """

    def test_empty_frame_raises_instead_of_decoding_as_an_empty_200(self) -> None:
        ws = _FakeWebSocket(frames=[""])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json("wss://svc.example.app/api/v1/query/ws", {"feature_view": "fv"}, timeout=5.0)

    def test_empty_frame_discards_the_connection_so_the_next_call_redials(self) -> None:
        ws = _FakeWebSocket(frames=[""])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json("wss://svc.example.app/api/v1/query/ws", {"feature_view": "fv"}, timeout=5.0)

        self.assertFalse(client.is_connected, "a closed connection must not stay available for reuse")
        self.assertTrue(ws.closed, "the dead socket should have been closed")

    def test_empty_frame_is_not_counted_as_a_served_request(self) -> None:
        ws = _FakeWebSocket(frames=[""])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json("wss://svc.example.app/api/v1/query/ws", {"feature_view": "fv"}, timeout=5.0)

        self.assertEqual(client.request_count, 0, "a close is not a served request")

    def test_empty_frame_mid_batch_discards_the_connection(self) -> None:
        # The batch path reaches this outcome through pipelined correlation as well, so this pins
        # existing behaviour rather than a change. Two bodies keep it off the single-request path.
        ws = _FakeWebSocket(frames=[""])
        client = _client(ws)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json_many(
                    "wss://svc.example.app/api/v1/query/ws",
                    [{"feature_view": "fv"}, {"feature_view": "fv2"}],
                    timeout=5.0,
                )

        self.assertFalse(client.is_connected)


class _HeaderedWebSocket(_FakeWebSocket):
    """A fake that also exposes upgrade response headers, as websocket-client's socket does."""

    def __init__(self, headers: dict[str, str] | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.headers = headers

    def getheaders(self) -> dict[str, str] | None:
        return self.headers


class AdvertisedConnAgeTest(absltest.TestCase):
    """The service advertises each connection's lifetime on the upgrade; the pool retires by it."""

    def _connect(self, ws: Any) -> online_service_ws_client.OnlineServiceWsClient:
        client = _client(ws)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.connect("wss://svc.example.app/api/v1/query/ws")
        return client

    def test_advertised_age_is_read_off_the_upgrade(self) -> None:
        client = self._connect(_HeaderedWebSocket({"x-ws-max-conn-age-seconds": "57"}))
        self.assertEqual(client.server_max_conn_age_seconds, 57.0)

    def test_header_name_is_matched_case_insensitively(self) -> None:
        client = self._connect(_HeaderedWebSocket({"X-WS-Max-Conn-Age-Seconds": " 57 "}))
        self.assertEqual(client.server_max_conn_age_seconds, 57.0)

    def test_absent_unparsable_or_non_positive_is_none(self) -> None:
        """None makes the pool keep its configured age, which is what it did before this existed."""
        for headers in (None, {}, {"x-ws-max-conn-age-seconds": "soon"}, {"x-ws-max-conn-age-seconds": "0"}):
            client = self._connect(_HeaderedWebSocket(headers))
            self.assertIsNone(client.server_max_conn_age_seconds, headers)

    def test_a_socket_without_response_headers_is_none(self) -> None:
        self.assertIsNone(self._connect(_FakeWebSocket()).server_max_conn_age_seconds)

    def test_dropping_the_connection_forgets_the_age(self) -> None:
        """A redial gets its own age; the dead socket's must not outlive it."""
        ws = _HeaderedWebSocket({"x-ws-max-conn-age-seconds": "57"}, frames=[""])
        client = self._connect(ws)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json("wss://svc.example.app/api/v1/query/ws", {"name": "fv"}, 30.0)
        self.assertIsNone(client.server_max_conn_age_seconds)


class _Clock:
    """Monotonic clock that advances only when told, so deadlines are testable without waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _SlowWebSocket(_FakeWebSocket):
    """Echoes like the real service, but each operation costs time on an injected clock."""

    def __init__(self, clock: _Clock, per_recv_seconds: float, per_send_seconds: float = 0.0) -> None:
        super().__init__()
        self._clock = clock
        self._per_recv_seconds = per_recv_seconds
        self._per_send_seconds = per_send_seconds

    def send(self, frame: str) -> None:
        self._clock.advance(self._per_send_seconds)
        super().send(frame)

    def recv(self) -> str:
        self._clock.advance(self._per_recv_seconds)
        return super().recv()


class BatchTimeoutIsADeadlineTest(absltest.TestCase):
    """``timeout`` must bound the whole exchange, not each socket operation.

    ``settimeout`` applies per send and per recv, so applying the caller's timeout once would let
    a batch of N take N times what the caller asked for -- hours, for a large read against a slow
    service, while every individual operation stayed inside its own limit. Both this client and
    the transport Protocol document the value as covering the batch.
    """

    def _bodies(self, count: int) -> list[dict[str, Any]]:
        return [{"name": "fv", "version": "v1", "request_id": f"r{i}"} for i in range(count)]

    def test_the_socket_timeout_shrinks_as_the_deadline_approaches(self) -> None:
        """Each operation gets the time remaining, not the full timeout."""
        clock = _Clock()
        ws = _SlowWebSocket(clock, per_recv_seconds=1.0)
        client = online_service_ws_client.OnlineServiceWsClient(
            _ws_factory=_factory_for(ws), _clock=clock, max_in_flight=1
        )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json_many("wss://svc/api/v1/query/ws", self._bodies(4), 100.0)

        self.assertTrue(ws.timeouts, "no socket timeout was ever applied")
        # Non-increasing: every operation re-derives the remaining budget.
        self.assertEqual(sorted(ws.timeouts, reverse=True), ws.timeouts, f"timeouts did not shrink: {ws.timeouts}")
        self.assertLess(ws.timeouts[-1], ws.timeouts[0])
        self.assertLessEqual(max(ws.timeouts), 100.0, "an operation was allowed more than the whole batch budget")

    def test_a_batch_that_outlives_its_timeout_raises_instead_of_running_on(self) -> None:
        """The failure the deadline exists to prevent: N frames x the timeout each."""
        clock = _Clock()
        # 10s per response against a 25s budget. The old per-operation behaviour would have
        # spent 200s here happily, since no single recv exceeded its own 25s limit.
        ws = _SlowWebSocket(clock, per_recv_seconds=10.0)
        client = online_service_ws_client.OnlineServiceWsClient(
            _ws_factory=_factory_for(ws), _clock=clock, max_in_flight=1
        )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json_many("wss://svc/api/v1/query/ws", self._bodies(20), 25.0)

        message = str(ctx.exception.original_exception)
        self.assertIn("whole batch", message)
        self.assertNotIn("unreachable", message, "a deadline overrun was relabelled as a dead service")
        self.assertLessEqual(clock.now, 40.0, "the call ran well past its deadline before giving up")
        self.assertFalse(client.is_connected, "a timed-out batch leaves unread responses; discard the socket")

    def test_a_single_request_deadline_covers_both_operations(self) -> None:
        """One send plus one recv is two operations, so a single settimeout would allow 2x.

        The send is given a cost here so the recv's budget demonstrably excludes it; with a free
        send the two budgets are legitimately equal and the test would prove nothing.
        """
        clock = _Clock()
        ws = _SlowWebSocket(clock, per_recv_seconds=1.0, per_send_seconds=4.0)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=_factory_for(ws), _clock=clock)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertLen(ws.timeouts, 2, "the send and the recv should each get a derived budget")
        self.assertEqual(30.0, ws.timeouts[0])
        self.assertEqual(26.0, ws.timeouts[1], "the recv budget should exclude the 4s the send took")


class _TimingOutWebSocket(_FakeWebSocket):
    """Raises what ``websocket-client`` raises when a blocked operation outlives its socket timeout.

    The clock-advancing fakes above only reach the deadline *between* operations. A real
    overrun nearly always lands while a recv is blocked, where the library raises instead.
    """

    def __init__(self, *, on: str = "recv", error: Exception | None = None) -> None:
        super().__init__()
        self._on = on
        self._error = error or websocket.WebSocketTimeoutException("Connection timed out")

    def send(self, frame: str) -> None:
        if self._on == "send":
            raise self._error
        super().send(frame)

    def recv(self) -> str:
        if self._on == "recv":
            raise self._error
        return super().recv()


class SocketTimeoutIsReportedAsTimeoutTest(absltest.TestCase):
    """A timeout raised mid-operation must read as a slow service, not a dead one."""

    def _assert_timeout(self, call: Any, client: online_service_ws_client.OnlineServiceWsClient) -> None:
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                call()
        message = str(ctx.exception.original_exception)
        self.assertIsInstance(ctx.exception.original_exception, TimeoutError)
        self.assertIn("exceeded its timeout", message)
        self.assertNotIn("unreachable", message)
        self.assertFalse(client.is_connected, "a timed-out exchange may leave a response unread")

    def test_a_recv_timeout_on_a_single_request(self) -> None:
        client = _client(_TimingOutWebSocket())
        self._assert_timeout(lambda: client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0), client)

    def test_a_recv_timeout_mid_batch(self) -> None:
        client = _client(_TimingOutWebSocket())
        self._assert_timeout(
            lambda: client.post_json_many("wss://svc/api/v1/query/ws", [{"name": "fv"}, {"name": "fv"}], 30.0),
            client,
        )

    def test_a_send_timeout(self) -> None:
        client = _client(_TimingOutWebSocket(on="send"))
        self._assert_timeout(lambda: client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0), client)

    def test_a_plain_socket_timeout(self) -> None:
        client = _client(_TimingOutWebSocket(error=TimeoutError("timed out")))
        self._assert_timeout(lambda: client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0), client)

    def test_other_failures_still_read_as_unreachable(self) -> None:
        client = _client(_TimingOutWebSocket(error=ConnectionResetError("reset by peer")))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv"}, 30.0)

        self.assertIn("unreachable", str(ctx.exception.original_exception))


class _CostlyDialFactory:
    """A ``_ws_factory`` whose dial costs time on an injected clock, like a real upgrade."""

    def __init__(self, ws: Any, clock: _Clock, cost_seconds: float) -> None:
        self._ws = ws
        self._clock = clock
        self._cost_seconds = cost_seconds
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, header: list[str] | None = None, timeout: float | None = None) -> Any:
        self.calls.append({"url": url, "header": list(header or []), "timeout": timeout})
        self._clock.advance(self._cost_seconds)
        return self._ws


class HandshakeIsInsideTheDeadlineTest(absltest.TestCase):
    """The dial is a socket operation, and on a cold connection the most expensive one.

    Charging only the send and the recv left the handshake bounded by ``connect_timeout`` and
    outside the caller's budget entirely: a 5s request against a gateway slow to upgrade could
    block for ``connect_timeout`` and then still be granted a full 5s of send and recv. That is
    the overrun the deadline exists to prevent, moved one layer out -- and the pool makes it the
    common case, since it dials a fresh connection on every pool-grow and every retirement.
    """

    def _body(self) -> dict[str, Any]:
        return {"name": "fv", "version": "v1"}

    def test_the_handshake_gets_the_callers_budget_when_it_is_the_smaller(self) -> None:
        clock = _Clock()
        factory = _CostlyDialFactory(_FakeWebSocket(), clock, cost_seconds=0.0)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory, _clock=clock, connect_timeout=30.0)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", self._body(), 5.0)

        self.assertEqual(5.0, factory.calls[0]["timeout"], "the dial was not bounded by the caller's timeout")

    def test_connect_timeout_still_caps_a_generous_caller(self) -> None:
        """The configured ceiling stays a ceiling: a 10-minute timeout is not a 10-minute dial."""
        clock = _Clock()
        factory = _CostlyDialFactory(_FakeWebSocket(), clock, cost_seconds=0.0)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory, _clock=clock, connect_timeout=30.0)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", self._body(), 600.0)

        self.assertEqual(30.0, factory.calls[0]["timeout"])

    def test_a_slow_dial_leaves_only_what_is_left_for_the_exchange(self) -> None:
        """The send and the recv get the remainder, not a budget that restarts after the dial."""
        clock = _Clock()
        ws = _FakeWebSocket()
        factory = _CostlyDialFactory(ws, clock, cost_seconds=4.0)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory, _clock=clock, connect_timeout=30.0)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("wss://svc/api/v1/query/ws", self._body(), 10.0)

        self.assertTrue(ws.timeouts, "no socket timeout was applied")
        self.assertLessEqual(max(ws.timeouts), 6.0, f"an operation was granted more than the dial left: {ws.timeouts}")

    def test_a_dial_that_consumes_the_whole_budget_raises_before_sending(self) -> None:
        """The failure this exists for: the dial alone outlasts the timeout, and the request still went."""
        clock = _Clock()
        ws = _FakeWebSocket()
        factory = _CostlyDialFactory(ws, clock, cost_seconds=6.0)
        client = online_service_ws_client.OnlineServiceWsClient(_ws_factory=factory, _clock=clock, connect_timeout=30.0)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as caught:
                client.post_json("wss://svc/api/v1/query/ws", self._body(), 5.0)

        self.assertIsInstance(caught.exception.original_exception, TimeoutError)
        self.assertEmpty(ws.sent, "a request went out after its deadline had already passed")

    def test_the_batch_path_charges_the_handshake_too(self) -> None:
        clock = _Clock()
        ws = _FakeWebSocket()
        factory = _CostlyDialFactory(ws, clock, cost_seconds=6.0)
        client = online_service_ws_client.OnlineServiceWsClient(
            _ws_factory=factory, _clock=clock, connect_timeout=30.0, max_in_flight=2
        )
        bodies = [{"name": "fv", "version": "v1", "request_id": f"r{i}"} for i in range(3)]

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                client.post_json_many("wss://svc/api/v1/query/ws", bodies, 5.0)

        self.assertEmpty(ws.sent, "a batch went out after its deadline had already passed")


class ConnectedAtTracksTheSocketTest(absltest.TestCase):
    """The dial time belongs to the socket, because this client replaces the socket silently.

    :mod:`online_service_ws_pool` retires a connection a margin under the lifetime the service
    advertised for it, measured from the dial. This client re-dials on the next use after a
    mid-request failure, so a pool dating the connection from when it first handed the client out
    would measure a seconds-old socket against the previous one's dial and retire it at birth --
    one extra handshake per transient failure, against a service whose handshake is the one
    authenticated request of a connection's life.
    """

    def _post(self, client: online_service_ws_client.OnlineServiceWsClient) -> None:
        client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "version": "v1"}, 30.0)

    def test_connected_at_is_none_until_the_dial(self) -> None:
        self.assertIsNone(_client(_FakeWebSocket()).connected_at)

    def test_connected_at_is_the_dial_time(self) -> None:
        clock = _Clock()
        clock.advance(17.0)
        client = online_service_ws_client.OnlineServiceWsClient(
            _ws_factory=_factory_for(_FakeWebSocket()), _clock=clock
        )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            self._post(client)

        self.assertEqual(17.0, client.connected_at)

    def test_a_redial_restarts_the_clock(self) -> None:
        """The point of the whole field: a replacement socket is dated from its own dial."""
        clock = _Clock()
        client = online_service_ws_client.OnlineServiceWsClient(
            _ws_factory=_factory_for(_FakeWebSocket()), _clock=clock
        )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            self._post(client)
            self.assertEqual(0.0, client.connected_at)

            # What every mid-request failure path does, and what leaves the client reusable.
            client._discard_connection()  # noqa: SLF001 - the transparent redial under test
            self.assertIsNone(client.connected_at, "a discarded socket still reported a dial time")

            clock.advance(595.0)
            self._post(client)

        self.assertEqual(595.0, client.connected_at)


class TimeoutApplicationIsAuditableTest(absltest.TestCase):
    """A live socket that refuses the budget keeps its old one, so it must not fail silently."""

    class _RefusesTimeouts(_FakeWebSocket):
        def settimeout(self, timeout: float) -> None:
            raise OSError("socket will not take a timeout")

    def test_a_socket_that_refuses_a_timeout_warns_rather_than_debugs(self) -> None:
        client = _client(self._RefusesTimeouts())

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertLogs(online_service_ws_client.__name__, level="WARNING") as logs:
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "version": "v1"}, 30.0)

        self.assertTrue(
            any("deadline is not enforced" in line for line in logs.output),
            f"an unenforced deadline was not reported above DEBUG: {logs.output}",
        )

    def test_a_stub_without_settimeout_does_not_warn(self) -> None:
        """Tolerating a stub is deliberate; warning about one would drown the real warning."""

        class _NoSettimeout:
            def __init__(self) -> None:
                self._inner = _FakeWebSocket()

            def send(self, frame: str) -> None:
                self._inner.send(frame)

            def recv(self) -> str:
                return self._inner.recv()

            def close(self) -> None:
                self._inner.close()

        client = _client(_NoSettimeout())
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertNoLogs(online_service_ws_client.__name__, level="WARNING"):
                client.post_json("wss://svc/api/v1/query/ws", {"name": "fv", "version": "v1"}, 30.0)


if __name__ == "__main__":
    absltest.main()
