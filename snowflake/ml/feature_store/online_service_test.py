"""Unit tests for online_service helpers."""

import email.message
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

from absl.testing import absltest

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import online_service
from snowflake.snowpark import Row, Session
from snowflake.snowpark.types import StringType

_UNITTEST_SNOWFLAKE_PAT = "unittest-snowflake-pat-token"


def _locator_fragment(database: str = "DB", schema: str = "SC") -> str:
    """Substring that appears in SYSTEM$ calls for the given logical db/schema names."""
    return f"{SqlIdentifier(database)}.{SqlIdentifier(schema)}"


class _FakeHttpOk:
    """Minimal response object for ``urlopen`` success path."""

    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = body
        self._status = status

    def getcode(self) -> int:
        return self._status

    def read(self, size: int = -1) -> bytes:
        if size >= 0:
            return self._body[:size]
        return self._body

    def close(self) -> None:
        pass


class OnlineServiceTest(absltest.TestCase):
    def test_fetch_online_service_status_parses_endpoints(self) -> None:
        payload = json.dumps(
            {
                "status": "RUNNING",
                "message": "ok",
                "endpoints": [{"name": "query", "url": "https://q.example"}, {"name": "ingest", "url": "https://i"}],
                "created_at": "1",
                "updated_at": "2",
            }
        )
        session = create_autospec(Session)

        def sql_side_effect(query: str, *args: object, **kwargs: object) -> MagicMock:
            m = MagicMock()
            qn = query.replace("\n", " ")
            loc = _locator_fragment()
            self.assertIn(f"SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('{loc}')", qn)
            m.collect.return_value = [Row(payload)]
            return m

        session.sql.side_effect = sql_side_effect
        st = online_service.fetch_online_service_status(session, SqlIdentifier("DB"), SqlIdentifier("SC"))
        self.assertEqual(st.status, "RUNNING")
        self.assertEqual(online_service.endpoint_url(st, "query"), "https://q.example")

    def test_fetch_online_service_status_ignores_extra_fields(self) -> None:
        """Richer account payloads may include keys we do not model; parsing stays canonical-only."""
        payload = json.dumps(
            {
                "status": "RUNNING",
                "message": "ok",
                "endpoints": [
                    {
                        "name": "query",
                        "url": "https://q.example",
                        "internal_port": 8443,
                        "diag": {"nested": [1, 2]},
                    },
                    {"name": "ingest", "url": "https://i", "region": "us-west"},
                ],
                "created_at": "1",
                "updated_at": "2",
                "trace_id": "ignored",
                "deployment": {"shard": 7},
            }
        )
        session = create_autospec(Session)

        def sql_side_effect(query: str, *args: object, **kwargs: object) -> MagicMock:
            m = MagicMock()
            qn = query.replace("\n", " ")
            loc = _locator_fragment()
            self.assertIn(f"SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('{loc}')", qn)
            m.collect.return_value = [Row(payload)]
            return m

        session.sql.side_effect = sql_side_effect
        st = online_service.fetch_online_service_status(session, SqlIdentifier("DB"), SqlIdentifier("SC"))
        self.assertEqual(st.status, "RUNNING")
        self.assertEqual(st.message, "ok")
        self.assertEqual(st.created_at, "1")
        self.assertEqual(st.updated_at, "2")
        self.assertEqual(len(st.endpoints), 2)
        self.assertEqual(online_service.endpoint_url(st, "query"), "https://q.example")
        self.assertEqual(online_service.endpoint_url(st, "ingest"), "https://i")

    def test_session_rest_auth_headers_uses_snowflake_pat_env(self) -> None:
        session = create_autospec(Session)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": "pat-from-env"}, clear=False):
            headers = online_service._session_rest_auth_headers(session)
        self.assertEqual(headers["Authorization"], 'Snowflake Token="pat-from-env"')

    def test_online_service_query_api_pat_from_env_raises_when_unset(self) -> None:
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": ""}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                online_service._online_service_query_api_pat_from_env()
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("SNOWFLAKE_PAT", str(ctx.exception.original_exception))

    def test_read_postgres_online_features_requires_snowflake_pat(self) -> None:
        session = create_autospec(Session)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": ""}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                online_service.read_postgres_online_features(
                    session,
                    "https://q/",
                    "fv",
                    "v1",
                    ["k"],
                    [["1"]],
                    None,
                    join_key_field_types={"k": StringType()},
                )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("SNOWFLAKE_PAT", str(ctx.exception.original_exception))

    def test_create_online_service_ignores_extra_fields_in_response(self) -> None:
        session = create_autospec(Session)
        create_payload = json.dumps(
            {
                "status": "SUCCESS",
                "message": "created",
                "internal": {"job_id": "j1"},
                "trace_id": "t9",
            }
        )

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            qn = query.replace("\n", " ")
            loc = _locator_fragment()
            self.assertIn("SYSTEM$CREATE_FEATURE_STORE_ONLINE_SERVICE(", qn)
            self.assertIn(f"'{loc}'", qn)
            self.assertEqual(qn.count("', '"), 1)
            m.collect.return_value = [Row(create_payload)]
            return m

        session.sql.side_effect = sql_side_effect
        result = online_service.create_online_service(
            session, SqlIdentifier("DB"), SqlIdentifier("SC"), "producer_r", "consumer_r"
        )
        self.assertEqual(result.status, "SUCCESS")
        self.assertIn("get_online_service_status", result.message)

    def test_create_online_service_validates_same_roles(self) -> None:
        session = create_autospec(Session)
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.create_online_service(
                session,
                SqlIdentifier("DB"),
                SqlIdentifier("SC"),
                "R1",
                "R1",
            )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)

    def test_get_online_service_status_raises_on_error_status(self) -> None:
        session = create_autospec(Session)

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            qn = query.replace("\n", " ")
            loc = _locator_fragment()
            self.assertIn(f"SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('{loc}')", qn)
            m.collect.return_value = [Row(json.dumps({"status": "ERROR", "message": "boom"}))]
            return m

        session.sql.side_effect = sql_side_effect
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.get_online_service_status(session, SqlIdentifier("DB"), SqlIdentifier("SC"))
        self.assertIn("boom", str(ctx.exception.original_exception))

    def test_read_postgres_online_features_requires_keys(self) -> None:
        session = create_autospec(Session)
        session.connection.rest.token = "t"
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.read_postgres_online_features(
                session,
                "https://q.example/base/",
                "my_fv",
                "v1",
                ["user_id"],
                [],
                None,
                join_key_field_types={"user_id": StringType()},
            )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)

    def test_read_postgres_online_features_posts_query_api(self) -> None:
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        response_payload = {
            "request_id": "r1",
            "status": "success",
            "results": [
                {"features": [10.5, 99]},
                {"features": [20.0, 100]},
            ],
            "metadata": {
                "features": [
                    {"name": "purchase_amount"},
                    {"name": "score"},
                ],
            },
        }

        captured: list[urllib.request.Request] = []

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            captured.append(req)
            self.assertTrue(req.full_url.endswith("/api/v1/query") or "/api/v1/query" in req.full_url)
            self.assertIn("Snowflake Token=", req.headers["Authorization"])
            assert isinstance(req.data, bytes)
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(body["name"], "my_fv")
            self.assertEqual(body["version"], "v1")
            self.assertEqual(body["object_type"], "feature_view")
            self.assertEqual(
                body["metadata_options"],
                {"include_names": True, "include_data_types": True},
            )
            self.assertEqual(len(body["request_rows"]), 2)
            self.assertNotIn("features", body)
            return _FakeHttpOk(json.dumps(response_payload).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                rows, _schema = online_service.read_postgres_online_features(
                    session,
                    "https://q.example/svc",
                    "my_fv",
                    "v1",
                    ["user_id"],
                    [[1], [2]],
                    None,
                    join_key_field_types={"user_id": StringType()},
                )

        self.assertEqual(len(captured), 1)
        self.assertEqual(
            rows,
            [
                {"user_id": 1, "purchase_amount": 10.5, "score": 99},
                {"user_id": 2, "purchase_amount": 20.0, "score": 100},
            ],
        )

    def test_read_postgres_online_features_feature_name_subset(self) -> None:
        """Optional ``feature_names`` restricts columns to a subset of ``metadata.features`` (server spelling)."""
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        response_payload = {
            "request_id": "r1",
            "status": "success",
            "results": [{"features": [10.5, 99]}],
            "metadata": {
                "features": [
                    {"name": "purchase_amount"},
                    {"name": "score"},
                ],
            },
        }

        captured: list[urllib.request.Request] = []

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            captured.append(req)
            assert isinstance(req.data, bytes)
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(body["features"], ["score"])
            return _FakeHttpOk(json.dumps(response_payload).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                rows, _schema = online_service.read_postgres_online_features(
                    session,
                    "https://q.example/",
                    "my_fv",
                    "v1",
                    ["user_id"],
                    [[1]],
                    ["score"],
                    join_key_field_types={"user_id": StringType()},
                )

        self.assertEqual(len(captured), 1)
        self.assertEqual(rows, [{"user_id": 1, "score": 99}])

    def test_read_postgres_online_features_values_follow_metadata_features_order(self) -> None:
        """``results[].features`` values align with ``metadata.features`` order (not client-side column order)."""
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        response_payload = {
            "request_id": "r1",
            "status": "success",
            "results": [{"features": [10.5, 99]}],
            "metadata": {
                "features": [
                    {"name": "score"},
                    {"name": "purchase_amount"},
                ],
            },
        }

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            return _FakeHttpOk(json.dumps(response_payload).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                rows, _schema = online_service.read_postgres_online_features(
                    session,
                    "https://q.example/",
                    "my_fv",
                    "v1",
                    ["user_id"],
                    [[1]],
                    None,
                    join_key_field_types={"user_id": StringType()},
                )

        self.assertEqual(rows, [{"user_id": 1, "score": 10.5, "purchase_amount": 99}])

    def test_read_postgres_online_features_maps_values_by_metadata_order(self) -> None:
        """Values pair with ``metadata.features`` positions (no client-side permutation)."""
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        response_payload = {
            "request_id": "r1",
            "status": "success",
            "results": [{"features": [999.0, "2024-06-01T12:00:00Z"]}],
            "metadata": {
                "features": [
                    {"name": "AMOUNT"},
                    {"name": "EVENT_TIME"},
                ],
            },
        }

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            return _FakeHttpOk(json.dumps(response_payload).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                rows, schema = online_service.read_postgres_online_features(
                    session,
                    "https://q.example/",
                    "my_fv",
                    "v1",
                    ["USER_ID"],
                    [["k1"]],
                    None,
                    join_key_field_types={"USER_ID": StringType()},
                )

        self.assertEqual(
            rows,
            [{"USER_ID": "k1", "AMOUNT": 999.0, "EVENT_TIME": "2024-06-01T12:00:00Z"}],
        )
        self.assertEqual([f.name for f in schema.fields], ["USER_ID", "AMOUNT", "EVENT_TIME"])

    def test_read_postgres_online_features_uses_metadata_data_types_without_fv_types(self) -> None:
        """``include_data_types`` response fields can supply Snowpark mapping without ``feature_field_types``."""
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        response_payload = {
            "request_id": "r1",
            "status": "success",
            "results": [{"features": [999.0, "2024-06-01T12:00:00Z"]}],
            "metadata": {
                "features": [
                    {"name": "AMOUNT", "type": "real", "precision": None, "scale": None},
                    {"name": "EVENT_TIME", "type": "timestamp_ntz", "precision": 0, "scale": 3},
                ],
            },
        }

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            return _FakeHttpOk(json.dumps(response_payload).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                rows, _schema = online_service.read_postgres_online_features(
                    session,
                    "https://q.example/",
                    "my_fv",
                    "v1",
                    ["USER_ID"],
                    [["k1"]],
                    None,
                    join_key_field_types={"USER_ID": StringType()},
                )

        self.assertEqual(
            rows,
            [{"USER_ID": "k1", "AMOUNT": 999.0, "EVENT_TIME": "2024-06-01T12:00:00Z"}],
        )

    def test_read_postgres_online_features_batches_eleven_keys(self) -> None:
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        call_count = 0

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            nonlocal call_count
            call_count += 1
            assert isinstance(req.data, bytes)
            body = json.loads(req.data.decode("utf-8"))
            self.assertNotIn("features", body)
            n = len(body["request_rows"])
            self.assertLessEqual(n, 10)
            results = [{"features": [float(i)]} for i in range(n)]
            payload = {
                "request_id": f"b{call_count}",
                "status": "success",
                "results": results,
                "metadata": {"features": [{"name": "x"}]},
            }
            return _FakeHttpOk(json.dumps(payload).encode("utf-8"))

        keys = [[i] for i in range(11)]
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                rows, _schema = online_service.read_postgres_online_features(
                    session,
                    "https://q.example/",
                    "fv",
                    "v1",
                    ["id"],
                    keys,
                    None,
                    join_key_field_types={"id": StringType()},
                )
        self.assertEqual(call_count, 2)
        self.assertEqual(len(rows), 11)

    def test_read_postgres_online_features_http_error(self) -> None:
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        err_body = json.dumps({"error": {"code": "NOT_FOUND", "message": "missing fv"}}).encode()

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", hdrs=email.message.Message(), fp=io.BytesIO(err_body)
            )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.read_postgres_online_features(
                        session,
                        "https://q.example/",
                        "fv",
                        "v1",
                        ["k"],
                        [["1"]],
                        None,
                        join_key_field_types={"k": StringType()},
                    )
        self.assertEqual(ctx.exception.error_code, error_codes.NOT_FOUND)

    def test_read_postgres_online_features_http_500_includes_correlation_ids(self) -> None:
        """Failed Online Service responses include server request_id when present."""
        session = create_autospec(Session)
        session.connection.rest.token = "tok"

        err_body = json.dumps(
            {
                "request_id": "01abc-server-trace",
                "error": {"message": "Failed to retrieve features"},
            }
        ).encode()

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            raise urllib.error.HTTPError(
                req.full_url, 500, "Server Error", hdrs=email.message.Message(), fp=io.BytesIO(err_body)
            )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.read_postgres_online_features(
                        session,
                        "https://q.example/",
                        "fv",
                        "v1",
                        ["k"],
                        [["1"]],
                        None,
                        join_key_field_types={"k": StringType()},
                    )
        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_SNOWFLAKE_API_ERROR)
        msg = str(ctx.exception.original_exception)
        self.assertIn("request_id: 01abc-server-trace", msg)
        self.assertIn("Failed to retrieve features", msg)

    def test_stream_ingest_records_requires_snowflake_pat(self) -> None:
        session = create_autospec(Session)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": ""}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                online_service.stream_ingest_records(
                    session,
                    "https://ingest.example/",
                    "my_stream",
                    [{"a": 1}],
                )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("SNOWFLAKE_PAT", str(ctx.exception.original_exception))

    def test_stream_ingest_records_posts_ingest_api(self) -> None:
        session = create_autospec(Session)
        captured: list[urllib.request.Request] = []

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            captured.append(req)
            self.assertTrue(req.full_url.endswith("/api/v1/ingest") or "/api/v1/ingest" in req.full_url)
            self.assertIn("Snowflake Token=", req.headers["Authorization"])
            assert isinstance(req.data, bytes)
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(
                body["records"],
                {"MY_STREAM": [{"USER_ID": "u1", "AMOUNT": 2.5}]},
            )
            resp = {
                "request_id": "r1",
                "status": "success",
                "sources": {
                    "MY_STREAM": {
                        "records": {"total": 1, "failed": 0},
                        "feature_views": {"failed": 0, "results": []},
                    }
                },
            }
            return _FakeHttpOk(json.dumps(resp).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                n = online_service.stream_ingest_records(
                    session,
                    "https://ingest.example/svc",
                    "MY_STREAM",
                    [{"USER_ID": "u1", "AMOUNT": 2.5}],
                )
        self.assertEqual(n, 1)
        self.assertEqual(len(captured), 1)

    def test_stream_ingest_records_http_207_partial_success_raises(self) -> None:
        """207 with failed records raises SnowflakeMLException."""
        session = create_autospec(Session)

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            resp = {
                "request_id": "r2",
                "status": "partial_success",
                "sources": {
                    "MY_STREAM": {
                        "records": {
                            "total": 10,
                            "failed": 2,
                            "errors": [
                                {
                                    "index": 3,
                                    "error": {
                                        "code": "MISSING_REQUIRED_FIELD",
                                        "message": "Record is missing required field: timestamp",
                                    },
                                },
                            ],
                        },
                        "feature_views": {"failed": 0, "results": []},
                    }
                },
            }
            return _FakeHttpOk(json.dumps(resp).encode("utf-8"), status=207)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.stream_ingest_records(
                        session,
                        "https://ingest.example/svc",
                        "MY_STREAM",
                        [{"USER_ID": "u1"}],
                    )
        self.assertIn("partial failure", str(ctx.exception.original_exception))

    def test_stream_ingest_records_http_207_all_records_failed_raises(self) -> None:
        session = create_autospec(Session)

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            resp = {
                "request_id": "r3",
                "status": "partial_success",
                "sources": {
                    "MY_STREAM": {
                        "records": {
                            "total": 1,
                            "failed": 1,
                            "errors": [
                                {
                                    "index": 0,
                                    "error": {
                                        "code": "INVALID_ARGUMENT",
                                        "message": "Unknown stream source: 'MY_STREAM'",
                                    },
                                },
                            ],
                        },
                        "feature_views": {"failed": 0, "results": []},
                    }
                },
            }
            return _FakeHttpOk(json.dumps(resp).encode("utf-8"), status=207)

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.stream_ingest_records(
                        session,
                        "https://ingest.example/svc",
                        "MY_STREAM",
                        [{"USER_ID": "u1"}],
                    )
        self.assertIn("partial failure", str(ctx.exception.original_exception))

    def test_stream_ingest_records_empty_raises(self) -> None:
        session = create_autospec(Session)
        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                online_service.stream_ingest_records(session, "https://i/", "s", [])
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)

    def test_stream_ingest_records_http_error_uses_ingest_api_label(self) -> None:
        session = create_autospec(Session)
        err_body = json.dumps({"error": {"code": "INVALID_ARGUMENT", "message": "bad row"}}).encode()

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", hdrs=email.message.Message(), fp=io.BytesIO(err_body)
            )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.stream_ingest_records(
                        session,
                        "https://ingest.example/",
                        "s",
                        [{"x": 1}],
                    )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("Online Service", str(ctx.exception.original_exception))

    # --- Online Service error path tests ---

    def test_create_online_service_rejects_empty_roles(self) -> None:
        session = create_autospec(Session)
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.create_online_service(
                session, SqlIdentifier("DB"), SqlIdentifier("SC"), producer_role="", consumer_role="consumer"
            )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("non-empty", str(ctx.exception.original_exception))

    def test_drop_online_service_invalid_json_response(self) -> None:
        session = create_autospec(Session)

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            m.collect.return_value = [Row("NOT-VALID-JSON")]
            return m

        session.sql.side_effect = sql_side_effect
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.drop_online_service(session, SqlIdentifier("DB"), SqlIdentifier("SC"))
        self.assertEqual(ctx.exception.error_code, error_codes.INTERNAL_SNOWML_ERROR)

    def test_fetch_online_service_status_not_found(self) -> None:
        session = create_autospec(Session)
        payload = json.dumps({"status": "NOT_FOUND", "message": "no service"})

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            m.collect.return_value = [Row(payload)]
            return m

        session.sql.side_effect = sql_side_effect
        st = online_service.fetch_online_service_status(session, SqlIdentifier("DB"), SqlIdentifier("SC"))
        self.assertEqual(st.status, "NOT_FOUND")

    def test_assert_running_not_found_raises(self) -> None:
        session = create_autospec(Session)
        payload = json.dumps({"status": "NOT_FOUND"})

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            m.collect.return_value = [Row(payload)]
            return m

        session.sql.side_effect = sql_side_effect
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.assert_online_service_running_with_query_endpoint(
                session, SqlIdentifier("DB"), SqlIdentifier("SC")
            )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("create_online_service", str(ctx.exception.original_exception))

    def test_assert_running_not_running_raises(self) -> None:
        session = create_autospec(Session)
        payload = json.dumps({"status": "PROVISIONING"})

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            m.collect.return_value = [Row(payload)]
            return m

        session.sql.side_effect = sql_side_effect
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.assert_online_service_running_with_query_endpoint(
                session, SqlIdentifier("DB"), SqlIdentifier("SC")
            )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("not RUNNING", str(ctx.exception.original_exception))

    def test_assert_running_no_endpoint_raises(self) -> None:
        session = create_autospec(Session)
        payload = json.dumps({"status": "RUNNING", "endpoints": []})

        def sql_side_effect(query: str, *a: object, **kw: object) -> MagicMock:
            m = MagicMock()
            m.collect.return_value = [Row(payload)]
            return m

        session.sql.side_effect = sql_side_effect
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
            online_service.assert_online_service_running_with_query_endpoint(
                session, SqlIdentifier("DB"), SqlIdentifier("SC")
            )
        self.assertEqual(ctx.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn("no query endpoint", str(ctx.exception.original_exception))

    # --- Stream ingest edge case tests ---

    def test_stream_ingest_records_datetime_serialization(self) -> None:
        import datetime

        session = create_autospec(Session)
        captured_body: list[dict[str, Any]] = []

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            assert isinstance(req.data, bytes)
            captured_body.append(json.loads(req.data.decode("utf-8")))
            resp = {"request_id": "r1", "status": "success", "sources": {"S": {"records": {"total": 1, "failed": 0}}}}
            return _FakeHttpOk(json.dumps(resp).encode("utf-8"))

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                online_service.stream_ingest_records(
                    session,
                    "https://ingest.example/",
                    "S",
                    [{"ts": datetime.datetime(2024, 6, 1, 12, 0, 0), "val": 1}],
                )
        row = captured_body[0]["records"]["S"][0]
        self.assertEqual(row["ts"], "2024-06-01T12:00:00")
        self.assertEqual(row["val"], 1)

    def test_stream_ingest_records_http_500_includes_request_id(self) -> None:
        session = create_autospec(Session)
        err_body = json.dumps({"error": {"message": "boom"}, "request_id": "srv-123"}).encode()

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            raise urllib.error.HTTPError(
                req.full_url, 500, "Internal", hdrs=email.message.Message(), fp=io.BytesIO(err_body)
            )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.stream_ingest_records(session, "https://i/", "s", [{"x": 1}])
        msg = str(ctx.exception.original_exception)
        self.assertIn("request_id: srv-123", msg)
        self.assertIn("boom", msg)

    def test_stream_ingest_records_non_json_error_body(self) -> None:
        session = create_autospec(Session)
        html_body = b"<html><body>502 Bad Gateway</body></html>"

        def urlopen_impl(req: urllib.request.Request, timeout: float | None = None) -> _FakeHttpOk:
            raise urllib.error.HTTPError(
                req.full_url, 502, "Bad Gateway", hdrs=email.message.Message(), fp=io.BytesIO(html_body)
            )

        with patch.dict(os.environ, {"SNOWFLAKE_PAT": _UNITTEST_SNOWFLAKE_PAT}, clear=False):
            with patch.object(urllib.request, "urlopen", side_effect=urlopen_impl):
                with self.assertRaises(snowml_exceptions.SnowflakeMLException) as ctx:
                    online_service.stream_ingest_records(session, "https://i/", "s", [{"x": 1}])
        msg = str(ctx.exception.original_exception)
        self.assertIn("502", msg)
        self.assertIn("Bad Gateway", msg)


if __name__ == "__main__":
    absltest.main()
