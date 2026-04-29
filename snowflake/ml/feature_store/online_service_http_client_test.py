"""Unit tests for online_service_http_client (HTTP/2 client + URL/auth helpers)."""

import json
import logging
import os
from typing import Any, Optional
from unittest.mock import create_autospec, patch

import httpx
from absl.testing import absltest

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.feature_store import online_service_http_client
from snowflake.snowpark import Session

_UNITTEST_SNOWFLAKE_PAT = "unittest-snowflake-pat-token"


def _make_response(
    body: bytes,
    *,
    status: int = 200,
    http_version: str = "HTTP/2",
    headers: Optional[dict[str, str]] = None,
) -> httpx.Response:
    """Build an httpx.Response with a forced http_version (httpx defaults vary by transport)."""
    resp = httpx.Response(status_code=status, content=body, headers=headers or {})
    # http_version is a read-only computed property; set the underlying ext dict to override.
    resp.extensions["http_version"] = http_version.encode("ascii")
    return resp


def _transport_for(handler: Any) -> Any:
    """Wrap a request handler in an httpx.MockTransport factory the OnlineServiceHttpClient can use."""

    def _factory(*, proxy: Optional[str] = None) -> httpx.MockTransport:
        return httpx.MockTransport(handler)

    return _factory


def _ok_handler(body: bytes = b"{}", *, status: int = 200, http_version: str = "HTTP/2") -> Any:
    """Handler factory that always returns the same body/status; tests use it for happy-path posts."""

    def _h(request: httpx.Request) -> httpx.Response:
        return _make_response(body, status=status, http_version=http_version)

    return _h


class UrlAndAuthHelpersTest(absltest.TestCase):
    def test_query_api_url_appends_relative_path(self) -> None:
        self.assertEqual(
            online_service_http_client.query_api_url("https://q.example/base"),
            "https://q.example/base/api/v1/query",
        )
        self.assertEqual(
            online_service_http_client.query_api_url("https://q.example/base/"),
            "https://q.example/base/api/v1/query",
        )

    def test_ingest_api_url_appends_relative_path(self) -> None:
        self.assertEqual(
            online_service_http_client.ingest_api_url("https://i.example/"),
            "https://i.example/api/v1/ingest",
        )

    def test_auth_headers_uses_snowflake_pat_env(self) -> None:
        session = create_autospec(Session)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": "pat-from-env"}, clear=False):
            headers = online_service_http_client.auth_headers(session)
        self.assertEqual(headers["Authorization"], 'Snowflake Token="pat-from-env"')
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "application/json")

    def test_online_service_pat_from_env_raises_when_unset(self) -> None:
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": ""}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                online_service_http_client.online_service_pat_from_env()
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("SNOWFLAKE_PAT", str(ctx.exception.original_exception))


class OnlineServiceHttpClientTest(absltest.TestCase):
    """Unit tests for :class:`online_service_http_client.OnlineServiceHttpClient`."""

    def test_client_is_lazy_until_first_post(self) -> None:
        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        self.assertEqual(client._clients_by_key, {})

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("https://q.example/", {}, timeout=1.0)
        self.assertEqual(len(client._clients_by_key), 1)

    def test_post_json_returns_status_and_body_for_2xx(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _make_response(b'{"ok": true}')

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(handler))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            status, body = client.post_json("https://q/", {"a": 1}, timeout=2.0)

        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"ok": true}')
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].method, "POST")
        self.assertEqual(str(captured[0].url), "https://q/")
        self.assertEqual(captured[0].content, json.dumps({"a": 1}).encode("utf-8"))
        self.assertEqual(captured[0].headers["Authorization"], f'Snowflake Token="{_UNITTEST_SNOWFLAKE_PAT}"')
        self.assertEqual(captured[0].headers["Content-Type"], "application/json")
        self.assertEqual(captured[0].headers["Accept"], "application/json")

    def test_post_json_returns_status_and_body_for_4xx_without_raising(self) -> None:
        err_body = json.dumps({"error": {"message": "bad keys"}, "request_id": "r-9"}).encode()
        client = online_service_http_client.OnlineServiceHttpClient(
            _transport_factory=_transport_for(_ok_handler(err_body, status=400))
        )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            status, body = client.post_json("https://q/", {"a": 1}, timeout=2.0)

        self.assertEqual(status, 400)
        self.assertEqual(body, err_body)

    def test_post_json_wraps_httpx_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connreset")

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(handler))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json("https://q/", {}, timeout=1.0)

        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_SNOWFLAKE_API_ERROR)
        self.assertIn("Online Service is unreachable", str(ctx.exception.original_exception))

    def test_post_json_wraps_os_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise OSError("socket closed")

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(handler))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json("https://q/", {}, timeout=1.0)

        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_SNOWFLAKE_API_ERROR)

    def test_headers_are_rebuilt_per_request(self) -> None:
        """Rebuilding per request is what lets a rotated ``SNOWFLAKE_PAT`` take effect without re-init."""
        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(
                online_service_http_client,
                "auth_headers",
                wraps=online_service_http_client.auth_headers,
            ) as spy:
                for _ in range(3):
                    client.post_json("https://q/", {"k": 1}, timeout=1.0)

        self.assertEqual(spy.call_count, 3)

    def test_rotated_pat_is_picked_up_on_next_request(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return _make_response(b"{}")

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(handler))
        rotated_pat = _UNITTEST_SNOWFLAKE_PAT + "-rotated"

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("https://q/", {}, timeout=1.0)
            os.environ["SNOWFLAKE_PAT"] = rotated_pat
            client.post_json("https://q/", {}, timeout=1.0)

        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0].headers["Authorization"], f'Snowflake Token="{_UNITTEST_SNOWFLAKE_PAT}"')
        self.assertEqual(captured[1].headers["Authorization"], f'Snowflake Token="{rotated_pat}"')

    def test_missing_pat_raises_without_allocating_client(self) -> None:
        """Error surface for a missing PAT must match the non-pooled path (INVALID_ARGUMENT, no client built)."""
        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                client.post_json("https://q/", {}, timeout=1.0)
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertEqual(client._clients_by_key, {})

    def test_request_uses_httpx_timeout_for_per_phase_budget(self) -> None:
        """``post_json`` calls ``client.post(..., timeout=httpx.Timeout(timeout))`` so each phase honors the budget."""
        captured: list[Any] = []
        real_post = httpx.Client.post

        def _spy_post(self_client: Any, url: str, *args: Any, **kwargs: Any) -> Any:
            captured.append(kwargs.get("timeout"))
            return real_post(self_client, url, *args, **kwargs)

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(httpx.Client, "post", _spy_post):
                client.post_json("https://q/", {}, timeout=2.5)
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0], httpx.Timeout)
        # All four phases must inherit the caller's budget.
        self.assertAlmostEqual(captured[0].connect, 2.5)
        self.assertAlmostEqual(captured[0].read, 2.5)
        self.assertAlmostEqual(captured[0].write, 2.5)
        self.assertAlmostEqual(captured[0].pool, 2.5)

    def test_no_proxy_means_no_proxy_kwarg_to_httpx_client(self) -> None:
        """When no proxy applies, the httpx.Client is built without a ``proxy`` argument."""
        captured_kwargs: list[dict[str, Any]] = []
        real_client_cls = httpx.Client

        def _spy_client(**kwargs: Any) -> httpx.Client:
            captured_kwargs.append(kwargs)
            return real_client_cls(**kwargs)

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch("urllib.request.getproxies", return_value={}):
                with patch.object(httpx, "Client", side_effect=_spy_client):
                    client.post_json("https://q.example/", {}, timeout=1.0)

        self.assertEqual(len(captured_kwargs), 1)
        self.assertNotIn("proxy", captured_kwargs[0])
        self.assertTrue(captured_kwargs[0]["http2"])

    def test_proxy_kwarg_passed_to_httpx_client_when_https_proxy_applies(self) -> None:
        """When a proxy applies, it is forwarded as ``proxy=`` to ``httpx.Client``."""
        captured_kwargs: list[dict[str, Any]] = []
        real_client_cls = httpx.Client

        def _spy_client(**kwargs: Any) -> httpx.Client:
            captured_kwargs.append(dict(kwargs))
            # httpx forbids passing both ``proxy`` and ``transport``; keep the explicit
            # transport for in-memory mocking and drop ``proxy`` after capture.
            kwargs.pop("proxy", None)
            return real_client_cls(**kwargs)

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch("urllib.request.getproxies", return_value={"https": "http://proxy.example:3128"}):
                with patch("urllib.request.proxy_bypass", return_value=False):
                    with patch.object(httpx, "Client", side_effect=_spy_client):
                        client.post_json("https://q.example/", {}, timeout=1.0)

        self.assertEqual(len(captured_kwargs), 1)
        self.assertEqual(captured_kwargs[0].get("proxy"), "http://proxy.example:3128")

    def test_no_proxy_bypass_skips_proxy_kwarg(self) -> None:
        """``NO_PROXY`` must override ``HTTPS_PROXY`` for matching hosts."""
        captured_kwargs: list[dict[str, Any]] = []
        real_client_cls = httpx.Client

        def _spy_client(**kwargs: Any) -> httpx.Client:
            captured_kwargs.append(kwargs)
            return real_client_cls(**kwargs)

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch("urllib.request.getproxies", return_value={"https": "http://proxy.example:3128"}):
                with patch("urllib.request.proxy_bypass", return_value=True):
                    with patch.object(httpx, "Client", side_effect=_spy_client):
                        client.post_json("https://q.example/", {}, timeout=1.0)

        self.assertEqual(len(captured_kwargs), 1)
        self.assertNotIn("proxy", captured_kwargs[0])

    def test_clients_are_cached_per_host_and_proxy(self) -> None:
        """Same (scheme://host, proxy) reuses one httpx.Client; different hosts build separate clients."""
        construct_count = 0
        real_client_cls = httpx.Client

        def _spy_client(**kwargs: Any) -> httpx.Client:
            nonlocal construct_count
            construct_count += 1
            return real_client_cls(**kwargs)

        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch("urllib.request.getproxies", return_value={}):
                with patch.object(httpx, "Client", side_effect=_spy_client):
                    client.post_json("https://a.example/", {}, timeout=1.0)
                    client.post_json("https://a.example/other", {}, timeout=1.0)
                    client.post_json("https://b.example/", {}, timeout=1.0)

        self.assertEqual(construct_count, 2)
        self.assertEqual(len(client._clients_by_key), 2)

    def test_max_connections_translates_to_httpx_limits(self) -> None:
        """``max_connections`` translates into ``httpx.Limits`` and ``http2=True`` is always set."""
        captured_kwargs: list[dict[str, Any]] = []
        real_client_cls = httpx.Client

        def _spy_client(**kwargs: Any) -> httpx.Client:
            captured_kwargs.append(kwargs)
            return real_client_cls(**kwargs)

        client = online_service_http_client.OnlineServiceHttpClient(
            max_connections=7,
            _transport_factory=_transport_for(_ok_handler()),
        )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(httpx, "Client", side_effect=_spy_client):
                client.post_json("https://q/", {}, timeout=1.0)

        self.assertEqual(len(captured_kwargs), 1)
        self.assertTrue(captured_kwargs[0]["http2"])
        limits = captured_kwargs[0]["limits"]
        self.assertEqual(limits.max_connections, 7)
        self.assertEqual(limits.max_keepalive_connections, 7)

    def test_close_is_idempotent_and_rejects_subsequent_posts(self) -> None:
        client = online_service_http_client.OnlineServiceHttpClient(_transport_factory=_transport_for(_ok_handler()))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            client.post_json("https://q/", {}, timeout=1.0)
            client.close()
            client.close()
            with self.assertRaises(RuntimeError):
                client.post_json("https://q/", {}, timeout=1.0)

    def test_http2_negotiation_logged(self) -> None:
        """Each origin emits one info log with the negotiated HTTP version on first response."""
        client = online_service_http_client.OnlineServiceHttpClient(
            _transport_factory=_transport_for(_ok_handler(http_version="HTTP/2"))
        )
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertLogs(online_service_http_client.logger, level=logging.INFO) as cm:
                client.post_json("https://q.example/", {}, timeout=1.0)
                # Second call must NOT re-log for the same origin.
                client.post_json("https://q.example/again", {}, timeout=1.0)
        joined = "\n".join(cm.output)
        self.assertIn("HTTP/2", joined)
        self.assertIn("https://q.example", joined)
        self.assertEqual(joined.count("Online Service HTTP negotiated"), 1)


if __name__ == "__main__":
    absltest.main()
