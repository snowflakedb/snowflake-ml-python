from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.utils import image_registry_client
from snowflake.ml.test_utils import exception_utils, mock_session
from snowflake.snowpark import session


class ImageRegistryClientTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))

    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.get")  # type: ignore[misc]
    def test_successful_login(self, mock_get: mock.MagicMock) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.text = '{"token": "dummy_token"}'
        mock_get.return_value = mock_response

        repo_url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo"
        registry_cred = "dummy_credentials"
        token = self.client.login(repo_url, registry_cred)

        # Assertions
        self.assertEqual(token, "dummy_token")
        mock_get.assert_called_once_with(
            "https://org-account.registry.snowflakecomputing.com/login",
            headers={"Authorization": f"Basic {registry_cred}"},
        )

    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.get")  # type: ignore[misc]
    def test_failed_login(self, mock_get: mock.MagicMock) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        repo_url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo"
        registry_cred = "dummy_credentials"

        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
            self.client.login(repo_url, registry_cred)

        mock_get.assert_called_once_with(
            "https://org-account.registry.snowflakecomputing.com/login",
            headers={"Authorization": f"Basic {registry_cred}"},
        )

    def test_convert_to_v2_head_manifests_url(self) -> None:
        full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
        actual = self.client.convert_to_v2_manifests_url(full_image_name=full_image_name)
        expected = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
        self.assertEqual(actual, expected)

    def test_convert_to_v2_head_manifests_url_with_invalid_full_image_name(self) -> None:
        image_name_without_tag = "org-account.registry.snowflakecomputing.com/db/schema/repo/image"
        with self.assertRaises(AssertionError):
            self.client.convert_to_v2_manifests_url(full_image_name=image_name_without_tag)

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.login"
    )
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.head")  # type: ignore[misc]
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry"
    )
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
            full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
            url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            self.assertEqual(self.client.image_exists(full_image_name=full_image_name), True)
            mock_login.assert_called_once_with(url, mock_registry_cred)
            mock_head.assert_called_once_with(
                url,
                headers={
                    "Authorization": f"Bearer {mock_bearer_token}",
                    "Accept": image_registry_client.MANIFEST_V2_HEADER,
                },
            )

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client" ".ImageRegistryClient.login"
    )
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.head")  # type: ignore[misc]
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry"
    )
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
            full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
            url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            self.assertEqual(self.client.image_exists(full_image_name=full_image_name), True)
            mock_login.assert_called_once_with(url, mock_registry_cred)
            self.assertEqual(mock_head.call_count, 2)
            expected_calls = [
                mock.call(
                    url,
                    headers={
                        "Authorization": f"Bearer {mock_bearer_token}",
                        "Accept": image_registry_client.MANIFEST_V2_HEADER,
                    },
                ),
                mock.call(
                    url,
                    headers={
                        "Authorization": f"Bearer {mock_bearer_token}",
                        "Accept": image_registry_client.MANIFEST_V1_HEADER,
                    },
                ),
            ]
            mock_head.assert_has_calls(expected_calls)

    @mock.patch(
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.login"
    )  # type: ignore[misc]
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.get")  # type: ignore[misc]
    @mock.patch("snowflake.ml.model._deploy_client.utils.image_registry_client.requests.put")  # type: ignore[misc]
    @mock.patch(
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.image_exists"
    )  # type: ignore[misc]
    @mock.patch(
        "snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry"
    )  # type: ignore[misc]
    def test_add_tag_to_remote_image(
        self,
        mock_spcs_image_registry: mock.MagicMock,
        mock_image_exists: mock.MagicMock,
        mock_put: mock.MagicMock,
        mock_get: mock.MagicMock,
        mock_login: mock.MagicMock,
    ) -> None:
        mock_image_exists.side_effect = [False, True]

        test_manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "size": 4753,
                "digest": "sha256:fb56ac2b330e2adaefd71f89e5a7e09a415bb55929d6b14b7a0bca5096479ad1",
            },
            "layers": [
                {
                    "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    "size": 143,
                    "digest": "sha256:24b9c0f433244f171ec20a922de94fc83d401d5d471ec13ca75f5a5fa7867426",
                },
            ],
            "tag": "tag-v1",
        }
        mock_get_response = mock.Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = test_manifest
        mock_get.return_value = mock_get_response

        mock_put_response = mock.Mock()
        mock_put_response.status_code = 201
        mock_put.return_value = mock_put_response

        mock_bearer_token = "dummy_bearer_token"
        mock_registry_cred = "dummy_registry_cred"
        mock_login.return_value = mock_bearer_token

        with mock.patch.object(mock_spcs_image_registry, "generate_image_registry_credential") as m_generate:
            m_generate.return_value.__enter__.return_value = mock_registry_cred
            new_tag = "new_tag"
            full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
            new_image_name = f"org-account.registry.snowflakecomputing.com/db/schema/repo/image:{new_tag}"
            url_old_tag = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            url = f"https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/{new_tag}"
            self.client.add_tag_to_remote_image(original_full_image_name=full_image_name, new_tag=new_tag)

            headers = {
                "Authorization": f"Bearer {mock_bearer_token}",
                "Accept": image_registry_client.MANIFEST_V2_HEADER,
            }
            mock_login.assert_called_once_with(url, mock_registry_cred)
            mock_get.assert_called_once_with(url_old_tag, headers=headers)
            test_manifest_copy = test_manifest.copy()
            test_manifest_copy["tag"] = new_tag

            put_headers = {
                **headers,
                "Content-Type": image_registry_client.MANIFEST_V2_HEADER,
            }

            mock_put.assert_called_once_with(url, headers=put_headers, json=test_manifest_copy)

            # First call to check existence before adding tag; second call to validate tag indeed added.
            mock_image_exists.assert_has_calls(
                [
                    mock.call(new_image_name),
                    mock.call(new_image_name),
                ]
            )


if __name__ == "__main__":
    absltest.main()
