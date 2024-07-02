import http.server
import json
import threading
import unittest
from dataclasses import dataclass
from io import BytesIO
from string import Template
from types import GeneratorType
from typing import Dict, Iterator, Optional, Union, cast

import _test_util
import requests
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _complete
from snowflake.cortex._util import CompleteOptions, process_rest_response
from snowflake.snowpark import functions, types

_OPTIONS = CompleteOptions(  # random params
    max_tokens=10,
    temperature=0.7,
    topP=1,
)


@dataclass
class FakeToken:
    token: str = "abc"


@dataclass
class FakeConnParams:
    rest: FakeToken
    host: str


@dataclass
class FakeSession:
    connection: FakeConnParams


class FakeResponse:  # needed for testing, imitates some of requests.Response behaviors
    def __init__(self, content: bytes) -> None:
        self.content = BytesIO(content)

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        while True:
            chunk = self.content.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def __iter__(self) -> Iterator[bytes]:
        return self.iter_content()


class CompleteTest(absltest.TestCase):
    model = "|model|"
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

    def test_complete_str(self) -> None:
        res = _complete._complete_impl("complete", self.model, self.prompt, session=self._session)
        self.assertEqual(self.complete_for_test(self.model, self.prompt), res)

    def test_complete_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model=self.model, prompt=self.prompt)])
        df_out = df_in.select(_complete._complete_impl("complete", functions.col("model"), functions.col("prompt")))
        res = df_out.collect()[0][0]
        self.assertEqual(self.complete_for_test(self.model, self.prompt), res)


class CompleteOptionsTest(absltest.TestCase):
    model = "|model|"
    prompt = "|prompt|"

    @staticmethod
    def format_as_complete(model: str, prompt: str, options: CompleteOptions) -> str:
        resp = Template("answered: $model, [{'content': '$prompt', 'role': 'user'}] with options: $options").substitute(
            model=model, prompt=prompt, options=options
        )
        return str(resp)

    @staticmethod
    def complete_for_test(model: str, prompt: str, options: Dict[str, float]) -> str:
        resp = Template("answered: $model, $prompt with options: $options").substitute(
            model=model, prompt=prompt, options=options
        )
        return str(resp)

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.complete_for_test,
            name="complete",
            return_type=types.StringType(),
            input_types=[types.StringType(), types.ArrayType(), types.MapType()],
            session=self._session,
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function complete(string,array,object)").collect()
        self._session.close()

    def test_populated_options(self) -> None:
        res = _complete._complete_impl("complete", self.model, self.prompt, _OPTIONS, session=self._session)
        self.assertEqual(self.format_as_complete(self.model, self.prompt, _OPTIONS), res)


class MockIpifyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTPServer mock request handler"""

    def do_POST(self) -> None:
        token: str = cast(str, self.headers.get("Authorization"))
        if "Snowflake Token" not in token:
            self.send_response(401)
            self.end_headers()
            return
        assert self.path == "/api/v2/cortex/inference/complete"
        content_length = int(cast(int, self.headers.get("Content-Length")))

        post_data = self.rfile.read(content_length).decode("utf-8")
        params = json.loads(post_data)
        stream = params["stream"]

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            # Simulate streaming by sending the response in chunks
            data_out = "This is a streaming response"
            chunk_size = 4
            for i in range(0, len(data_out), chunk_size):
                response_line = (
                    f"data: {json.dumps( {'choices': [{'delta': {'content': data_out[i:i + chunk_size]}}]})}\n\n"
                )
                self.wfile.write(response_line.encode("utf-8"))
                self.wfile.flush()
            return

        response_json = {"choices": [{"message": {"content": "This is a non streaming response"}}]}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_json).encode("utf-8"))


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockIpifyHTTPRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server_thread.join()

    def send_request(
        self, stream: bool = False, options: Optional[CompleteOptions] = None
    ) -> Union[str, Iterator[str]]:
        faketoken = FakeToken()
        fakeconnectionparameters = FakeConnParams(
            host=f"http://127.0.0.1:{self.server.server_address[1]}/", rest=faketoken
        )
        session = FakeSession(fakeconnectionparameters)
        response = _complete.call_rest_function(  # type: ignore[attr-defined]
            function="complete",
            model="my_models",
            prompt="test_prompt",
            options=options,
            session=cast(snowpark.Session, session),
            stream=stream,
        )
        return process_rest_response(response, stream=stream)

    def test_non_streaming(self) -> None:
        result = self.send_request(stream=False)
        self.assertEqual("This is a non streaming response", result)

    def test_wrong_token(self) -> None:
        headers = {"Authorization": "Wrong Token=123"}
        data = {"stream": "hh"}

        # Send the POST request
        response = requests.post(
            f"http://127.0.0.1:{self.server.server_address[1]}/api/v2/cortex/inference/complete",
            headers=headers,
            json=data,
        )
        self.assertEqual(response.status_code, 401)

    def test_streaming(self) -> None:
        result = self.send_request(stream=True)
        output = "".join(list(result))
        self.assertEqual("This is a streaming response", output)

    def test_streaming_with_options(self) -> None:
        result = self.send_request(stream=True, options=_OPTIONS)
        output = "".join(list(result))
        self.assertEqual("This is a streaming response", output)

    def test_non_streaming_with_options(self) -> None:
        result = self.send_request(stream=False, options=_OPTIONS)
        self.assertEqual("This is a non streaming response", result)

    def test_streaming_type(self) -> None:
        result = self.send_request(stream=True)
        self.assertIsInstance(result, GeneratorType)


if __name__ == "__main__":
    absltest.main()
