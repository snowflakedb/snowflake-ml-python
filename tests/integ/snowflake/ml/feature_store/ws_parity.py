"""Run an existing online-read integ suite over WebSocket and prove its results do not change.

The WebSocket transport is only safe to enable if every online read returns exactly what HTTP does,
for every feature-view shape the suites already cover: streaming, tiled, secondary-key, all
supported types, feature groups with realtime sources. Re-writing those cases for WebSocket would
duplicate them and drift; instead :func:`install` wraps ``read_feature_view`` (online only) and
``read_feature_group`` for the duration of a test, so each read the suite already makes is served
twice -- once over HTTP, once over WebSocket, through the real ``FeatureStore`` -- and fails unless
the two frames are identical, the WebSocket arm was actually served over WebSocket
(``ws_requests`` moved) and nothing fell back to HTTP (``http_fallback_requests`` did not).

The suite's own assertions then run on the HTTP result, unchanged.

Use it by subclassing a suite and restricting it to the tests that read online::

    class StreamingFeatureViewWsParityIntegTest(StreamingFeatureViewIntegTest):
        def setUp(self) -> None:
            super().setUp()
            ws_parity.install(self)

    ws_parity.restrict_to(StreamingFeatureViewWsParityIntegTest, ["test_..."])
"""

import os
import time
from typing import Any, Callable
from unittest import mock

import pandas as pd
from absl.testing import absltest
from common_utils import skip_unless_ws_route_served

from snowflake.ml.feature_store import (
    feature_view as fv_mod,
    online_service_ws_transport,
)
from snowflake.ml.feature_store.feature_store import FeatureStore

# The two arms are separate reads against a multi-replica service, so right after a write they can
# legitimately see different replicas. A mismatch is retried before it is reported.
_PARITY_ATTEMPTS = 4
_PARITY_BACKOFF_SECONDS = 3.0


def restrict_to(cls: type, test_names: list[str]) -> None:
    """Keep only the listed tests (and their parameterized variants) on a parity subclass.

    The rest are inherited from the HTTP suite and already run there; running them again would
    only double the suite's runtime.

    Args:
        cls: The parity subclass.
        test_names: Test method names to keep; a parameterized variant is kept with its base name.
    """
    for name in dir(cls):
        if name.startswith("test") and not any(name == kept or name.startswith(kept + "_") for kept in test_names):
            setattr(cls, name, None)


def install(test: absltest.TestCase) -> None:
    """Serve every online read in ``test`` over both transports and require identical results.

    Args:
        test: A test from a suite built on ``StreamingFeatureViewIntegTestBase``. Skipped when its
            Online Service does not serve the WebSocket route.
    """
    skip_unless_ws_route_served(test, test.fs)  # type: ignore[attr-defined]

    arms: dict[int, tuple[Any, Any]] = {}

    def transports(fs: FeatureStore) -> tuple[Any, Any]:
        # One of each per FeatureStore for the whole test, so the WebSocket arm reuses its pool
        # instead of paying a handshake per read.
        if id(fs) not in arms:
            # The transport this FeatureStore already had, handed back untouched at cleanup. Both
            # arms built below are closed when the test ends, and the wrapper leaves ``fs`` holding
            # the HTTP arm -- so without this the shared instance is left pointing at a closed
            # client and every later online read or ingest in the shard fails with "is closed".
            # Registered before the closes, and addCleanup runs last-in-first-out, so it restores
            # after both arms are shut.
            incoming = getattr(fs, "_online_http_client", None)  # noqa: SLF001
            test.addCleanup(setattr, fs, "_online_http_client", incoming)

            saved = os.environ.get(online_service_ws_transport.WS_ENABLED_ENV_VAR)
            try:
                os.environ[online_service_ws_transport.WS_ENABLED_ENV_VAR] = ""
                fs._online_http_client = None  # noqa: SLF001
                http = fs._get_or_create_online_http_client()  # noqa: SLF001
                os.environ[online_service_ws_transport.WS_ENABLED_ENV_VAR] = "true"
                fs._online_http_client = None  # noqa: SLF001
                ws = fs._get_or_create_online_http_client()  # noqa: SLF001
            finally:
                if saved is None:
                    os.environ.pop(online_service_ws_transport.WS_ENABLED_ENV_VAR, None)
                else:
                    os.environ[online_service_ws_transport.WS_ENABLED_ENV_VAR] = saved
            test.assertIsInstance(
                ws, online_service_ws_transport.OnlineServiceWsTransport, "WS_ENABLED did not build the WS transport"
            )
            test.assertNotIsInstance(http, online_service_ws_transport.OnlineServiceWsTransport)
            arms[id(fs)] = (http, ws)
            test.addCleanup(http.close)
            test.addCleanup(ws.close)
        return arms[id(fs)]

    def parity(original: Callable[..., Any], default_store: fv_mod.StoreType) -> Callable[..., Any]:
        def wrapper(fs: FeatureStore, *args: Any, **kwargs: Any) -> Any:
            if not _is_online(kwargs.get("store_type", default_store)):
                return original(fs, *args, **kwargs)

            http, ws = transports(fs)
            last_diff = ""
            for attempt in range(_PARITY_ATTEMPTS):
                fs._online_http_client = http  # noqa: SLF001
                via_http = original(fs, *args, **kwargs)  # an HTTP error is the suite's to handle

                before_ws, before_fallback = ws.ws_requests, ws.http_fallback_requests
                fs._online_http_client = ws  # noqa: SLF001
                try:
                    via_ws = original(fs, *args, **kwargs)
                finally:
                    fs._online_http_client = http  # noqa: SLF001
                test.assertGreater(ws.ws_requests, before_ws, "the WebSocket arm was not served over WebSocket")
                test.assertEqual(
                    before_fallback, ws.http_fallback_requests, "the WebSocket arm fell back to HTTP for a batch"
                )

                last_diff = _frame_diff(via_http, via_ws)
                if not last_diff:
                    return via_http
                if attempt + 1 < _PARITY_ATTEMPTS:
                    time.sleep(_PARITY_BACKOFF_SECONDS)
            raise AssertionError(f"WebSocket and HTTP returned different results: {last_diff}")

        return wrapper

    for name, original, default_store in (
        # The real signatures' defaults: read_feature_view is offline unless asked, read_feature_group online.
        ("read_feature_view", _READ_FV, fv_mod.StoreType.OFFLINE),
        ("read_feature_group", _READ_FG, fv_mod.StoreType.ONLINE),
    ):
        patcher = mock.patch.object(FeatureStore, name, parity(original, default_store))
        patcher.start()
        test.addCleanup(patcher.stop)


_READ_FV = FeatureStore.read_feature_view
_READ_FG = FeatureStore.read_feature_group


def _is_online(store_type: Any) -> bool:
    """Whether a ``store_type`` argument, enum or string, selects the online store.

    Args:
        store_type: What the caller passed.

    Returns:
        True for the online store.
    """
    if isinstance(store_type, fv_mod.StoreType):
        return store_type == fv_mod.StoreType.ONLINE
    return str(store_type).lower() == str(fv_mod.StoreType.ONLINE.value).lower()


def _frame_diff(via_http: Any, via_ws: Any) -> str:
    """Describe how two online-read results differ.

    Args:
        via_http: Result served over HTTP.
        via_ws: Result served over WebSocket.

    Returns:
        An empty string when they are identical, otherwise the first difference found.
    """
    left = via_http.to_pandas() if hasattr(via_http, "to_pandas") else via_http
    right = via_ws.to_pandas() if hasattr(via_ws, "to_pandas") else via_ws
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        return "" if left == right else f"{left!r} != {right!r}"
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    except AssertionError as e:
        return str(e)[:2000]
    return ""
