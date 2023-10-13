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

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.retryable_http.get_http_client"
    )
    def test_successful_login(self, mock_get_http_client: mock.MagicMock) -> None:
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
        mock_response = mock.MagicMock(status_code=200, text='{"token": "dummy_token"}')
        mock_get_http_client.return_value.get.return_value = mock_response
        repo_url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo"
        registry_cred = "dummy_credentials"
        token = client.login(repo_url, registry_cred)
        self.assertEqual(token, "dummy_token")
        mock_get_http_client.return_value.get.assert_called_once_with(
            "https://org-account.registry.snowflakecomputing.com/login",
            headers={"Authorization": f"Basic {registry_cred}"},
        )

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.retryable_http.get_http_client"
    )
    def test_failed_login(self, mock_get_http_client: mock.MagicMock) -> None:
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
        mock_response = mock.MagicMock(status_code=401, text="Unauthorized")
        mock_get_http_client.return_value.get.return_value = mock_response
        repo_url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo"
        registry_cred = "dummy_credentials"

        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
            client.login(repo_url, registry_cred)

        mock_get_http_client.return_value.get.assert_called_once_with(
            "https://org-account.registry.snowflakecomputing.com/login",
            headers={"Authorization": f"Basic {registry_cred}"},
        )

    def test_convert_to_v2_head_manifests_url(self) -> None:
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
        full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
        actual = client.convert_to_v2_manifests_url(full_image_name=full_image_name)
        expected = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
        self.assertEqual(actual, expected)

    def test_convert_to_v2_head_manifests_url_with_invalid_full_image_name(self) -> None:
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
        image_name_without_tag = "org-account.registry.snowflakecomputing.com/db/schema/repo/image"
        with self.assertRaises(AssertionError):
            client.convert_to_v2_manifests_url(full_image_name=image_name_without_tag)

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.login"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.retryable_http.get_http_client"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry"
    )
    def test_image_exists(
        self, mock_spcs_image_registry: mock.MagicMock, mock_get_http_client: mock.MagicMock, mock_login: mock.MagicMock
    ) -> None:
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
        mock_response = mock.MagicMock(status_code=200)
        mock_get_http_client.return_value.head.return_value = mock_response
        mock_bearer_token = "dummy_bearer_token"
        mock_registry_cred = "dummy_registry_cred"
        mock_login.return_value = mock_bearer_token

        with mock.patch.object(mock_spcs_image_registry, "generate_image_registry_credential") as m_generate:
            m_generate.return_value.__enter__.return_value = mock_registry_cred
            full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
            url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            self.assertEqual(client.image_exists(full_image_name=full_image_name), True)
            mock_login.assert_called_once_with(url, mock_registry_cred)
            mock_get_http_client.return_value.head.assert_called_once_with(
                url,
                headers={
                    "Authorization": f"Bearer {mock_bearer_token}",
                    "Accept": image_registry_client.MANIFEST_V2_HEADER,
                },
            )

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client" ".ImageRegistryClient.login"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.retryable_http.get_http_client"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry"
    )
    def test_image_exists_with_two_head_requests(
        self, mock_spcs_image_registry: mock.MagicMock, mock_get_http_client: mock.MagicMock, mock_login: mock.MagicMock
    ) -> None:
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
        mock_get_http_client.return_value.head.side_effect = [
            mock.Mock(status_code=404),
            mock.Mock(status_code=200),
        ]

        mock_bearer_token = "dummy_bearer_token"
        mock_registry_cred = "dummy_registry_cred"
        mock_login.return_value = mock_bearer_token

        with mock.patch.object(mock_spcs_image_registry, "generate_image_registry_credential") as m_generate:
            m_generate.return_value.__enter__.return_value = mock_registry_cred
            full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
            url = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
            self.assertEqual(client.image_exists(full_image_name=full_image_name), True)
            mock_login.assert_called_once_with(url, mock_registry_cred)

            # Modify the assertion to check for two calls to head
            self.assertEqual(mock_get_http_client.return_value.head.call_count, 2)

            # Modify the expected calls to match both head requests
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

            # Assert that the expected calls were made
            mock_get_http_client.return_value.head.assert_has_calls(expected_calls)

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.login"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.retryable_http.get_http_client"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client." "ImageRegistryClient.image_exists"
    )
    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.utils.image_registry_client.spcs_image_registry"
    )
    def test_add_tag_to_remote_image(
        self,
        mock_spcs_image_registry: mock.MagicMock,
        mock_image_exists: mock.MagicMock,
        mock_get_http_client: mock.MagicMock,
        mock_login: mock.MagicMock,
    ) -> None:
        mock_image_exists.side_effect = [False, True]
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session))
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
        mock_get_response = mock.Mock(status_code=200, json=lambda: test_manifest)
        mock_put_response = mock.Mock(status_code=201)
        mock_get_http_client.return_value.get.return_value = mock_get_response
        mock_get_http_client.return_value.put.return_value = mock_put_response

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
            client.add_tag_to_remote_image(original_full_image_name=full_image_name, new_tag=new_tag)

            headers = {
                "Authorization": f"Bearer {mock_bearer_token}",
                "Accept": image_registry_client.MANIFEST_V2_HEADER,
            }
            mock_login.assert_called_once_with(url, mock_registry_cred)
            mock_get_http_client.return_value.get.assert_called_once_with(url_old_tag, headers=headers)
            test_manifest_copy = test_manifest.copy()
            test_manifest_copy["tag"] = new_tag

            put_headers = {
                **headers,
                "Content-Type": image_registry_client.MANIFEST_V2_HEADER,
            }

            mock_get_http_client.return_value.put.assert_called_once_with(
                url, headers=put_headers, json=test_manifest_copy
            )

            # First call to check existence before adding tag; second call to validate tag indeed added.
            mock_image_exists.assert_has_calls(
                [
                    mock.call(new_image_name),
                    mock.call(new_image_name),
                ]
            )


if __name__ == "__main__":
    absltest.main()
