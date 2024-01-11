from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml._internal.container_services.image_registry import (
    registry_client as image_registry_client,
)
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import session


class ImageRegistryClientTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_convert_to_v2_head_manifests_url(self) -> None:
        full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session), full_image_name)
        actual = client._convert_to_v2_manifests_url(full_image_name=full_image_name)
        expected = "https://org-account.registry.snowflakecomputing.com/v2/db/schema/repo/image/manifests/latest"
        self.assertEqual(actual, expected)

    def test_convert_to_v2_head_manifests_url_with_invalid_full_image_name(self) -> None:
        image_name_without_tag = "org-account.registry.snowflakecomputing.com/db/schema/repo/image"
        with self.assertRaises(AssertionError):
            image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session), image_name_without_tag)

    def test_image_exists(self) -> None:
        full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session), full_image_name)
        url = client._convert_to_v2_manifests_url(full_image_name)

        with mock.patch.object(client.image_registry_http_client, "head", return_value=mock.MagicMock(status_code=200)):
            self.assertEqual(client.image_exists(full_image_name=full_image_name), True)

            client.image_registry_http_client.head.assert_called_once_with(  # type: ignore[attr-defined]
                url, headers=client._get_accept_headers()
            )

    def test_add_tag_to_remote_image(self) -> None:
        # Test case for updating the tag on an image that initially doesn't exist.
        # Retrieves the manifest, updates it with a new tag, and pushes the manifest to add the tag.
        # Covers the scenario where it takes 2 put requests to update the tag.
        full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
        client = image_registry_client.ImageRegistryClient(cast(session.Session, self.m_session), full_image_name)
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
        with mock.patch.object(client, "image_exists", side_effect=[False, True]), mock.patch.object(
            client, "_get_manifest", return_value=test_manifest
        ), mock.patch.object(
            client.image_registry_http_client,
            "put",
            side_effect=[mock.Mock(status_code=400), mock.Mock(status_code=201)],
        ):
            new_tag = "new_tag"
            new_image_name = f"org-account.registry.snowflakecomputing.com/db/schema/repo/image:{new_tag}"
            client.add_tag_to_remote_image(original_full_image_name=full_image_name, new_tag=new_tag)
            headers = client._get_accept_headers()
            put_header_v1 = {
                **headers,
                "Content-Type": image_registry_client._MANIFEST_V1_HEADER,
            }
            put_header_v2 = {
                **headers,
                "Content-Type": image_registry_client._MANIFEST_V2_HEADER,
            }
            url = client._convert_to_v2_manifests_url(new_image_name)

            test_manifest_copy = test_manifest.copy()
            test_manifest_copy["tag"] = new_tag
            client.image_registry_http_client.put.assert_has_calls(  # type: ignore[attr-defined]
                [
                    mock.call(url, headers=put_header_v1, json=test_manifest_copy),
                    mock.call(url, headers=put_header_v2, json=test_manifest_copy),
                ],
                any_order=False,
            )

            # First call to check existence before adding tag; second call to validate tag indeed added.
            client.image_exists.assert_has_calls(  # type: ignore[attr-defined]
                [
                    mock.call(new_image_name),
                    mock.call(new_image_name),
                ]
            )


if __name__ == "__main__":
    absltest.main()
