from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.utils import image_registry_client
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import session


class ImageRegistryClientTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))

    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.get")  # type: ignore
    def test_successful_login(self, mock_get: mock.MagicMock) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = '{"token": "dummy_token"}'
        mock_get.return_value = mock_response

        repo_url = "https://org-account.registry-dev.snowflakecomputing.com/v2/db/schema/repo"
        registry_cred = "dummy_credentials"
        token = self.client.login(repo_url, registry_cred)

        # Assertions
        self.assertEqual(token, "dummy_token")
        mock_get.assert_called_once_with(
            "https://org-account.registry-dev.snowflakecomputing.com/login",
            headers={"Authorization": f"Basic {registry_cred}"},
        )

    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.get")  # type: ignore
    def test_failed_login(self, mock_get: mock.MagicMock) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        repo_url = "https://org-account.registry-dev.snowflakecomputing.com/v2/db/schema/repo"
        registry_cred = "dummy_credentials"

        with self.assertRaises(RuntimeError):
            self.client.login(repo_url, registry_cred)

        mock_get.assert_called_once_with(
            "https://org-account.registry-dev.snowflakecomputing.com/login",
            headers={"Authorization": f"Basic {registry_cred}"},
        )

    def test_convert_to_v2_head_manifests_url(self) -> None:
        full_image_name = "org-account.registry-dev.snowflakecomputing.com/db/schema/repo/image:latest"
        actual = self.client.convert_to_v2_head_manifests_url(full_image_name=full_image_name)
        expected = "https://org-account.registry-dev.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
        self.assertEqual(actual, expected)

    def test_convert_to_v2_head_manifests_url_with_invalid_full_image_name(self) -> None:
        image_name_without_tag = "org-account.registry-dev.snowflakecomputing.com/db/schema/repo/image"
        with self.assertRaises(AssertionError):
            self.client.convert_to_v2_head_manifests_url(full_image_name=image_name_without_tag)

    @mock.patch(
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.login"
    )  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.head")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry")  # type: ignore
    def test_image_exists(
        self, mock_spcs_image_registry: mock.MagicMock, mock_head: mock.MagicMock, mock_login: mock.MagicMock
    ) -> None:
        mock_head_response = mock.Mock()
        mock_head_response.status_code = 200
        mock_head.return_value = mock_head_response

        mock_bearer_token = "dummy_bearer_token"
        mock_registry_cred = "dummy_registry_cred"
        mock_login.return_value = mock_bearer_token

        with mock.patch.object(mock_spcs_image_registry, "generate_image_registry_credential") as m_generate:
            m_generate.return_value.__enter__.return_value = mock_registry_cred
            full_image_name = "org-account.registry-dev.snowflakecomputing.com/db/schema/repo/image:latest"
            url = "https://org-account.registry-dev.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            self.assertEqual(self.client.image_exists(full_image_name=full_image_name), True)
            mock_login.assert_called_once_with(url, mock_registry_cred)
            mock_head.assert_called_once_with(
                url,
                headers={
                    "Authorization": f"Bearer {mock_bearer_token}",
                    "Accept": "application/vnd.oci.image.manifest.v1+json",
                },
            )

    @mock.patch(
        "snowflake.ml.model._deploy_client.utils.image_registry_client" ".ImageRegistryClient.login"
    )  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.head")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry")  # type: ignore
    def test_image_exists_with_two_head_requests(
        self, mock_spcs_image_registry: mock.MagicMock, mock_head: mock.MagicMock, mock_login: mock.MagicMock
    ) -> None:
        mock_head_response_success = mock.Mock()
        mock_head_response_success.status_code = 200
        mock_head_response_fail = mock.Mock()
        mock_head_response_fail.status_code = 404

        # Simulate that first head request fails, but second succeeded with the different header.
        mock_head.side_effect = [mock_head_response_fail, mock_head_response_success]

        mock_bearer_token = "dummy_bearer_token"
        mock_registry_cred = "dummy_registry_cred"
        mock_login.return_value = mock_bearer_token

        with mock.patch.object(mock_spcs_image_registry, "generate_image_registry_credential") as m_generate:
            m_generate.return_value.__enter__.return_value = mock_registry_cred
            full_image_name = "org-account.registry-dev.snowflakecomputing.com/db/schema/repo/image:latest"
            url = "https://org-account.registry-dev.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            self.assertEqual(self.client.image_exists(full_image_name=full_image_name), True)
            mock_login.assert_called_once_with(url, mock_registry_cred)
            self.assertEqual(mock_head.call_count, 2)
            expected_calls = [
                mock.call(
                    url,
                    headers={
                        "Authorization": f"Bearer {mock_bearer_token}",
                        "Accept": "application/vnd.oci.image.manifest.v1+json",
                    },
                ),
                mock.call(
                    url,
                    headers={
                        "Authorization": f"Bearer {mock_bearer_token}",
                        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                    },
                ),
            ]
            mock_head.assert_has_calls(expected_calls)


if __name__ == "__main__":
    absltest.main()
