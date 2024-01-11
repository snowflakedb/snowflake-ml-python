import json
from typing import cast

import requests
from absl.testing import absltest, parameterized
from absl.testing.absltest import mock

from snowflake.ml._internal.container_services.image_registry import (
    http_client as image_registry_http_client,
)
from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import session


class ImageRegistryHttpClientTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.m_repo_url = "https://org-account.registry.snowflakecomputing.com"

    def _get_mock_response(self, *, status_code: int, text: str) -> mock.Mock:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_response.text = text
        return mock_response

    @parameterized.parameters(("head",), ("get",), ("put",), ("post",), ("patch",))  # type: ignore[misc]
    def test_http_method_succeed_in_one_request(self, http_method: str) -> None:
        http_client = image_registry_http_client.ImageRegistryHttpClient(
            session=cast(session.Session, self.m_session), repo_url=self.m_repo_url
        )
        api_url = "https://org-account.registry.snowflakecomputing.com/v2/"

        dummy_token = "fake_token"
        mock_token_response = self._get_mock_response(status_code=200, text=json.dumps({"token": dummy_token}))
        mock_response = self._get_mock_response(status_code=200, text="succeed")

        with mock.patch.object(http_client, "_login", return_value=mock_token_response), mock.patch.object(
            http_client._retryable_http, http_method, return_value=mock_response
        ):
            res = getattr(http_client, http_method)(api_url, headers={})
            self.assertEqual(res, mock_response)
            getattr(http_client._retryable_http, http_method).assert_called_once_with(
                api_url, headers={"Authorization": f"Bearer {dummy_token}"}
            )

    @parameterized.parameters(("head",), ("get",), ("put",), ("post",), ("patch",))  # type: ignore[misc]
    def test_http_method_retry_on_401(self, http_method: str) -> None:
        http_client = image_registry_http_client.ImageRegistryHttpClient(
            session=cast(session.Session, self.m_session), repo_url=self.m_repo_url
        )
        api_url = "https://org-account.registry.snowflakecomputing.com/v2/"

        dummy_token_1 = "fake_token_1"
        dummy_token_2 = "fake_token_2"
        mock_token_response_1 = self._get_mock_response(status_code=200, text=json.dumps({"token": dummy_token_1}))
        mock_token_response_2 = self._get_mock_response(status_code=200, text=json.dumps({"token": dummy_token_2}))

        mock_response_1 = self._get_mock_response(status_code=401, text="401 FAILED")
        mock_response_2 = self._get_mock_response(status_code=200, text="Succeed")

        mock_token_responses = [mock_token_response_1, mock_token_response_2]
        mock_responses = [mock_response_1, mock_response_2]

        with mock.patch.object(http_client, "_login", side_effect=mock_token_responses), mock.patch.object(
            http_client._retryable_http, http_method, side_effect=mock_responses
        ):
            res = getattr(http_client, http_method)(api_url, headers={})
            self.assertEqual(res, mock_response_2)
            getattr(http_client._retryable_http, http_method).assert_has_calls(
                [
                    mock.call(api_url, headers={"Authorization": f"Bearer {dummy_token_1}"}),
                    mock.call(api_url, headers={"Authorization": f"Bearer {dummy_token_2}"}),
                ],
                any_order=False,
            )

    # Running only 1 method to reduce test time; in general other test case already guarantees that each method will
    # have decorator @retry_on_401 set.
    @parameterized.parameters(("head",))  # type: ignore[misc]
    def test_http_method_fail_after_max_retries(self, http_method: str) -> None:
        http_client = image_registry_http_client.ImageRegistryHttpClient(
            session=cast(session.Session, self.m_session), repo_url=self.m_repo_url
        )
        api_url = "https://org-account.registry.snowflakecomputing.com/v2/"
        dummy_token = "fake_token"
        mock_token_responses = [
            self._get_mock_response(status_code=200, text=json.dumps({"token": f"dummy_token{i}"}))
            for i in range(image_registry_http_client._MAX_RETRIES)
        ]
        mock_responses = [
            self._get_mock_response(status_code=401, text="401 FAILED")
            for _ in range(image_registry_http_client._MAX_RETRIES)
        ]

        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as context:
            with mock.patch.object(http_client, "_login", side_effect=mock_token_responses), mock.patch.object(
                http_client._retryable_http, http_method, side_effect=mock_responses
            ):
                getattr(http_client, http_method)(api_url, headers={})

                getattr(http_client._retryable_http, http_method).assert_has_calls(
                    [
                        mock.call(api_url, headers={"Authorization": f"Bearer {dummy_token}{i}"})
                        for i in range(image_registry_http_client._MAX_RETRIES)
                    ],
                    any_order=False,
                )

                expected_error_message = "Failed to authenticate to registry after max retries"
                self.assertIn(expected_error_message, str(context.exception))

    @parameterized.parameters(("head",))  # type: ignore[misc]
    def test_should_not_retry_on_non_401(self, http_method: str) -> None:
        http_client = image_registry_http_client.ImageRegistryHttpClient(
            session=cast(session.Session, self.m_session), repo_url=self.m_repo_url
        )
        api_url = "https://org-account.registry.snowflakecomputing.com/v2/"

        dummy_token_1 = "fake_token_1"
        mock_token_response = self._get_mock_response(status_code=200, text=json.dumps({"token": dummy_token_1}))
        mock_response = self._get_mock_response(status_code=403, text="403 FAILED")

        with mock.patch.object(http_client, "_login", return_value=mock_token_response), mock.patch.object(
            http_client._retryable_http, http_method, return_value=mock_response
        ):
            getattr(http_client, http_method)(api_url, headers={})
            # There should only be a single call for non-401 http code
            getattr(http_client._retryable_http, http_method).assert_has_calls(
                [
                    mock.call(api_url, headers={"Authorization": f"Bearer {dummy_token_1}"}),
                ],
                any_order=False,
            )


if __name__ == "__main__":
    absltest.main()
