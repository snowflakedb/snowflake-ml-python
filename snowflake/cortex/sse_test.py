from typing import List, cast

import requests
from absl.testing import absltest

from snowflake.cortex._sse_client import SSEClient
from snowflake.cortex.complete_test import FakeResponse


def _streaming_messages(response_data: bytes) -> List[str]:
    client = SSEClient(cast(requests.Response, FakeResponse(response_data)))
    out = []
    for event in client.events():
        out.append(event.data)
    return out


class SSETest(absltest.TestCase):
    def test_empty_response(self) -> None:
        # Set up the mock streaming response with no data
        response_data = b""

        result = _streaming_messages(response_data)

        assert result == []

    def test_empty_response_many_newlines(self) -> None:
        # Set up the mock streaming response with no data (in the form of newlines)
        response_data = b"\n\n\n\n\n\n"

        result = _streaming_messages(response_data)

        assert result == []

    def test_whitespace_handling(self) -> None:
        # Set up the mock streaming response with leading and trailing whitespace
        response_data = b"  \n \ndata: Message 1\n\n   \n \n \n \n"

        result = _streaming_messages(response_data)

        expected_message = ["Message 1"]
        assert expected_message == result

    def test_ignore_anything_but_message_event(self) -> None:
        # check that only "data" is considered

        response_data = (
            b"data: data\n\n"
            b"event: event\n\n"
            b"id: id\n\n"
            b"retry: retry\n\n"
            b"some_other_message: some_other_message\n\n"
        )

        result = _streaming_messages(response_data)

        expected_message = ["data"]
        assert result == expected_message

    def test_colon_cases(self) -> None:
        response_data_colon_middle = b"data: choices: middle_colon\n\n"
        response_many_colons = b"data: choices: middle_colon: last_colon\n\n"

        result_data_colon_middle = _streaming_messages(response_data_colon_middle)
        result_many_colons = _streaming_messages(response_many_colons)

        expected_colon_middle = ["choices: middle_colon"]
        expected_many_colons = ["choices: middle_colon: last_colon"]

        assert result_data_colon_middle == expected_colon_middle
        assert result_many_colons == expected_many_colons

    def test_line_separator(self) -> None:
        # test if data is not combined if it has trailing \n\n
        # fmt: off
        response_not_combined = (
            b"data: one\n\n"
            b"data: two\n"
            b"data: three\n\n"
        )
        # fmt: on

        result_parsed = _streaming_messages(response_not_combined)

        assert result_parsed == ["one", "two\nthree"]  # not combined

    def test_combined_data(self) -> None:
        # test if data is combined if it has trailing \n
        # fmt: off
        response_not_combined = (
            b"data: jeden\n"
            b"data: dwa\n\n"
        )
        # fmt: on

        result_parsed = _streaming_messages(response_not_combined)
        assert result_parsed == ["jeden\ndwa"]  # combined due to only one \n

    def test_commented_data(self) -> None:
        # test if data is treated as comment if it starts with a :
        response_not_combined = b": jeden\n\n"

        result_parsed = _streaming_messages(response_not_combined)

        assert result_parsed == []  # not combined

    def test_ignore_other_event_types(self) -> None:
        # test if data is ignored if its event is not message
        # fmt: off
        response_sth_else = (
            b"data: jeden\n"
            b"event: sth_else\n\n"
        )
        # fmt: on

        result_parsed = _streaming_messages(response_sth_else)

        assert result_parsed == []  # ignore anything that is not message

    def test_empty_data_json(self) -> None:
        response_sth_else = b"data: {}"

        result_parsed = _streaming_messages(response_sth_else)

        assert result_parsed == ["{}"]


if __name__ == "__main__":
    absltest.main()
