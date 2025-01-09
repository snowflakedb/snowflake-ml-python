import http.server
import json
import logging
import re
import threading
import time
import unittest
from dataclasses import dataclass
from io import BytesIO
from types import GeneratorType
from typing import Any, Dict, Iterable, Iterator, cast

import _test_util
from absl.testing import absltest
from requests.exceptions import HTTPError

from snowflake import snowpark
from snowflake.cortex import _complete
from snowflake.snowpark import functions, types

_OPTIONS = _complete.CompleteOptions(  # random params
    max_tokens=10,
    temperature=0.7,
    top_p=1,
)

# Use of this model name triggers a 400 error and missing model response.
_MISSING_MODEL_NAME = "missing_model"
_RETRY_FOREVER_MODEL_NAME = "429_forever"
_RETRY_UNTIL_TIME_MODEL_NAME = "429_until_time_"  # follow by time.time() value
_MISSING_MODEL_RESPONSE = '{"message": "unknown model", "error_code": "X-123", "request_id": "123-456"}'


def retry_until_model_name(deadline: float) -> str:
    return _RETRY_UNTIL_TIME_MODEL_NAME + str(int(deadline))


_UNEXPECTED_RESPONSE_FORMAT_MODEL_NAME = "unexpected_format_response_model"

logger = logging.getLogger(__name__)


@dataclass
class FakeToken:
    token: str = "abc"


@dataclass
class FakeConnParams:
    rest: FakeToken
    scheme: str
    host: str


@dataclass
class FakeSession:
    connection: FakeConnParams


class FakeResponse:  # needed for testing, imitates some of requests.Response behaviors
    def __init__(self, content: bytes, headers: Dict[str, str], data: bytes) -> None:
        self.content = BytesIO(content)
        self.headers = headers
        self.data = data

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        while True:
            chunk = self.content.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def __iter__(self) -> Iterator[bytes]:
        return self.iter_content()


class CompleteSQLBackendTest(absltest.TestCase):
    model = "|model|"
    custom_model_stage = "@my.custom.model/stage"
    custom_model_entity = "my.custom.model_entity"
    all_models = [model, custom_model_stage, custom_model_entity]
    prompt = "|prompt|"

    @staticmethod
    def complete_for_test(
        model: str,
        prompt: str,
    ) -> str:
        return f"answered: {model}, {prompt}"

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.complete_for_test,
            name="complete",
            return_type=types.StringType(),
            input_types=[types.StringType(), types.StringType()],
            session=self._session,
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function complete(string,string)").collect()
        self._session.close()

    def test_complete_snowpark_mode(self) -> None:
        """Test complete call with a single dataframe argument with columns for model
        and prompt."""
        df_in = self._session.create_dataframe(
            [snowpark.Row(model=model, prompt=self.prompt) for model in self.all_models]
        )
        df_out = df_in.select(
            _complete._complete_impl(functions.col("model"), functions.col("prompt"), function="complete")
        )
        for row_index in range(len(self.all_models)):
            res = df_out.collect()[row_index][0]
            self.assertEqual(self.complete_for_test(self.all_models[row_index], self.prompt), res)


class MockIpifyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTPServer mock request handler"""

    def do_POST(self) -> None:
        token: str = cast(str, self.headers.get("Authorization"))
        if "Snowflake Token" not in token:
            self.send_response(401)
            self.end_headers()
            return
        assert self.path == "/api/v2/cortex/inference:complete"
        content_length = int(cast(int, self.headers.get("Content-Length")))

        post_data = self.rfile.read(content_length).decode("utf-8")
        params = json.loads(post_data)
        model = params["model"]
        stream = params["stream"]

        logger.info(f"model: {model} stream: {stream}")

        if model == _MISSING_MODEL_NAME:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_MISSING_MODEL_RESPONSE.encode("utf-8"))
            return

        if model == _RETRY_FOREVER_MODEL_NAME:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return

        if model.startswith(_RETRY_UNTIL_TIME_MODEL_NAME):
            deadline = float(model.replace(_RETRY_UNTIL_TIME_MODEL_NAME, ""))
            if time.time() < deadline:
                logger.info("haven't hit deadline, sending 429")
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-after", "1")
                self.end_headers()
                return
            logger.info("sending successful response")

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            if model == _UNEXPECTED_RESPONSE_FORMAT_MODEL_NAME:
                self.wfile.write(b"data: {}\n\n")
                self.wfile.write(b'data: {"choices": [{"delta": {"content": "msg"}}]}\n\n')
                self.wfile.flush()
                return

            # Simulate streaming by sending the response in chunks
            data_out = "This is a streaming response"
            chunk_size = 4
            for i in range(0, len(data_out), chunk_size):
                json_msg = json.dumps({"choices": [{"delta": {"content": data_out[i : i + chunk_size]}}]})
                self.wfile.write(f"data: {json_msg}\n\n".encode())
                self.wfile.flush()
            return

        response_json = {"choices": [{"message": {"content": "This is a non streaming response"}}]}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if model == _UNEXPECTED_RESPONSE_FORMAT_MODEL_NAME:
            self.wfile.write(b"{}")
            return

        self.wfile.write(json.dumps(response_json).encode("utf-8"))


# This is a fake implementation of the function that sends the request to the Snowflake API via XP.
def fake_xp_request_handler(
    method: str,
    url: str,
    queryParams: Dict[str, str],
    headers: Dict[str, str],
    body: Dict[str, Any],
    postParams: Dict[str, str],
    timeoutMs: Any,
) -> Any:
    assert method == "POST"
    assert "/cortex/" in url

    if body["model"] == "empty_content":
        return {
            "status": 500,
            "content": "",
            "headers": {
                "Content-Type": "application/json",
                "Content-Length": "243",
                "Date": "Thu, 05 Dec 2024 16:51:28 GMT",
                "X-Snowflake-Request-ID": "80b66f5c-f955-42f7-8d6d-e524533f4f1a",
            },
        }

    if body["model"] == "unknown_model":
        # Response from a real request.
        return {
            "status": 400,
            "content": """{
                "code":	"300014",
                    "message":	"unknown model: \\"fake_model\\"",
                    "request_id":	"80b66f5c-f955-42f7-8d6d-e524533f4f1a",
                    "error_code":	"300014"
                }""",
            "headers": {
                "Content-Type": "application/json",
                "Content-Length": "243",
                "Date": "Thu, 05 Dec 2024 16:51:28 GMT",
                "X-Snowflake-Request-ID": "80b66f5c-f955-42f7-8d6d-e524533f4f1a",
            },
        }

    # Response from a real request.
    return {
        "status": 200,
        "content": """[{
            "data":	{
                "id":	"6e577fa0-9673-4214-84d9-20f1a9033df8",
                "created":	1733417829,
                "model":	"mistral-large",
                "choices":	[{
                        "delta":	{
                            "content":	" Sure"
                        }
                    }],
                "usage":	{
                    "prompt_tokens":	14,
                    "completion_tokens":	1,
                    "total_tokens":	15
                }
            }
        }, {
            "data":	{
                "id":	"6e577fa0-9673-4214-84d9-20f1a9033df8",
                "created":	1733417829,
                "model":	"mistral-large",
                "choices":	[{
                        "delta":	{
                            "content":	","
                        }
                    }],
                "usage":	{
                    "prompt_tokens":	14,
                    "completion_tokens":	2,
                    "total_tokens":	16
                }
            }
        }]""",
        "headers": {
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Snowflake-Request-ID": "6e577fa0-9673-4214-84d9-20f1a9033df8",
            "Date": "Thu, 05 Dec 2024 16:57:09 GMT",
            "Content-Type": "text/event-stream",
        },
    }


def replace_uuid(input: str) -> str:
    # Matches a UUID, e.g. 6e577fa0-9673-4214-84d9-20f1a9033df8.
    return re.sub(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", "[UUID]", input
    )


class CompleteRESTBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockIpifyHTTPRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        faketoken = FakeToken()
        fakeconnectionparameters = FakeConnParams(
            scheme="http", host=f"127.0.0.1:{self.server.server_address[1]}", rest=faketoken
        )
        self.session = cast(snowpark.Session, FakeSession(fakeconnectionparameters))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server_thread.join()

    def test_streaming(self) -> None:
        result = _complete._complete_impl(model="my_models", prompt="test_prompt", session=self.session, stream=True)
        self.assertIsInstance(result, GeneratorType)
        output = "".join(list(cast(Iterable[str], result)))
        self.assertEqual("This is a streaming response", output)

    def test_streaming_with_options(self) -> None:
        result = _complete._complete_impl(
            model="my_models",
            prompt="test_prompt",
            options=_OPTIONS,
            session=self.session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        output = "".join(list(cast(Iterable[str], result)))
        self.assertEqual("This is a streaming response", output)

    def test_streaming_with_empty_options(self) -> None:
        result = _complete._complete_impl(
            model="my_models",
            prompt="test_prompt",
            options=_complete.CompleteOptions(),
            session=self.session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        output = "".join(list(cast(Iterable[str], result)))
        self.assertEqual("This is a streaming response", output)

    def test_streaming_unexpected_response_format(self) -> None:
        response = _complete._complete_impl(
            model=_UNEXPECTED_RESPONSE_FORMAT_MODEL_NAME,
            prompt="test_prompt",
            session=self.session,
            stream=True,
        )
        assert isinstance(response, Iterator)
        message = ""
        for part in response:
            message += part
        self.assertEqual("msg", message)

    def test_streaming_error(self) -> None:
        try:
            _complete._complete_impl(
                model=_MISSING_MODEL_NAME,
                prompt="test_prompt",
                session=self.session,
                stream=True,
            )
        except HTTPError as e:
            self.assertEqual(400, e.response.status_code)
            self.assertEqual(_MISSING_MODEL_RESPONSE, e.response.text)

    def test_streaming_timeout(self) -> None:
        self.assertRaises(
            TimeoutError,
            lambda: _complete._complete_impl(
                model=_RETRY_FOREVER_MODEL_NAME,
                prompt="test_prompt",
                session=self.session,
                stream=True,
                timeout=1,
            ),
        )

    def test_complete_non_streaming_mode(self) -> None:
        result = _complete._complete_impl(
            model="my_models",
            prompt="test_prompt",
            options=_complete.CompleteOptions(),
            session=self.session,
        )
        self.assertIsInstance(result, str)
        self.assertEqual("This is a streaming response", result)

    def test_deadline(self) -> None:
        self.assertRaises(
            TimeoutError,
            lambda: _complete._complete_impl(
                model=_RETRY_FOREVER_MODEL_NAME,
                prompt="test_prompt",
                session=self.session,
                stream=True,
                deadline=time.time() + 1,
            ),
        )

    def test_streaming_retry_until_success(self) -> None:
        result = _complete._complete_impl(
            model=retry_until_model_name(time.time() + 1),
            prompt="test_prompt",
            session=self.session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        output = "".join(list(cast(Iterable[str], result)))
        self.assertEqual("This is a streaming response", output)

    def test_xp(self) -> None:
        result = _complete._complete_impl(
            snow_api_xp_request_handler=fake_xp_request_handler,
            model="my_models",
            prompt="test_prompt",
            session=self.session,
            stream=False,
        )
        self.assertEqual(" Sure,", result)

    def test_xp_stream(self) -> None:
        result = _complete._complete_impl(
            snow_api_xp_request_handler=fake_xp_request_handler,
            model="my_models",
            prompt="test_prompt",
            session=self.session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        output = "".join(list(cast(Iterable[str], result)))
        self.assertEqual(" Sure,", output)

    def test_xp_unknown_model(self) -> None:
        with self.assertRaises(ValueError) as ar:
            _complete._complete_impl(
                snow_api_xp_request_handler=fake_xp_request_handler,
                model="unknown_model",
                prompt="test_prompt",
                session=self.session,
            )
        self.assertEqual(
            'Request failed: unknown model: "fake_model" (request id: [UUID])', replace_uuid(str(ar.exception))
        )

    def test_xp_empty_content(self) -> None:
        with self.assertRaises(ValueError) as ar:
            _complete._complete_impl(
                snow_api_xp_request_handler=fake_xp_request_handler,
                model="empty_content",
                prompt="test_prompt",
                session=self.session,
            )
        self.assertEqual("Request failed (request id: [UUID])", replace_uuid(str(ar.exception)))


if __name__ == "__main__":
    absltest.main()
