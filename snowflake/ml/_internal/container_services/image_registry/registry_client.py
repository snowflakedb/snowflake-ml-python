import http
import logging
from typing import Dict, Optional, cast
from urllib.parse import urlunparse

from snowflake.ml._internal.container_services.image_registry import (
    http_client as image_registry_http_client,
    imagelib,
)
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils

_MANIFEST_V1_HEADER = "application/vnd.oci.image.manifest.v1+json"
_MANIFEST_V2_HEADER = "application/vnd.docker.distribution.manifest.v2+json"
_SUPPORTED_MANIFEST_HEADERS = [_MANIFEST_V1_HEADER, _MANIFEST_V2_HEADER]

logger = logging.getLogger(__name__)


class ImageRegistryClient:
    """
    A partial implementation of an SPCS image registry client. The client utilizes the ImageRegistryHttpClient under
    the hood, incorporating a retry mechanism to handle intermittent 401 errors from the SPCS image registry.
    """

    def __init__(self, session: Session, full_dest_image_name: str) -> None:
        """Initialization

        Args:
            session: Snowpark session
            full_dest_image_name: Based on dest image name, repo url can be inferred.
        """
        self.image_registry_http_client = image_registry_http_client.ImageRegistryHttpClient(
            session=session,
            repo_url=self._convert_to_v2_manifests_url(full_image_name=full_dest_image_name),
        )

    def _convert_to_v2_manifests_url(self, full_image_name: str) -> str:
        """Converts a full image name to a Docker Registry HTTP API V2 URL:
        https://docs.docker.com/registry/spec/api/#existing-manifests

        org-account.registry-dev.snowflakecomputing.com/db/schema/repo/image_name:tag becomes
        https://org-account.registry-dev.snowflakecomputing.com/v2/db/schema/repo/image_name/manifests/tag

        Args:
            full_image_name: a string consists of image name and image tag.

        Returns:
            Docker HTTP V2 URL for checking manifest existence.
        """
        scheme = "https"
        full_image_name_parts = full_image_name.split(":")
        assert len(full_image_name_parts) == 2, "full image name should include both image name and tag"

        image_name = full_image_name_parts[0]
        tag = full_image_name_parts[1]
        image_name_parts = image_name.split("/")
        domain = image_name_parts[0]
        rest = "/".join(image_name_parts[1:])
        path = f"/v2/{rest}/manifests/{tag}"
        url_tuple = (scheme, domain, path, "", "", "")
        return urlunparse(url_tuple)

    def _get_accept_headers(self) -> Dict[str, str]:
        # Depending on the built image, the media type of the image manifest might be either
        # application/vnd.oci.image.manifest.v1+json or application/vnd.docker.distribution.manifest.v2+json
        # Hence we need to check for both, otherwise it could result in false negative.
        return {"Accept": ",".join(_SUPPORTED_MANIFEST_HEADERS)}

    def image_exists(self, full_image_name: str) -> bool:
        """Check whether image already exists in the registry.

        Args:
            full_image_name: Full image name consists of image name and image tag.

        Returns:
            Boolean value. True when image already exists, else False.

        """
        # When running in SPROC, the Sproc session connection will not have _rest object associated, which makes it
        # unable to fetch session token needed to authenticate to SPCS image registry.
        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            return False
        v2_api_url = self._convert_to_v2_manifests_url(full_image_name)
        headers = self._get_accept_headers()
        status = self.image_registry_http_client.head(v2_api_url, headers=headers).status_code
        return status == http.HTTPStatus.OK

    def _get_manifest(self, full_image_name: str) -> Dict[str, str]:
        """Retrieve image manifest file. Given Docker manifest comes with two versions, and for each version the
        corresponding request header is required for a successful HTTP response.

        Args:
            full_image_name: Full image name.

        Returns:
            Full manifest content as a python dict.

        Raises:
            SnowflakeMLException: when failed to retrieve manifest.
        """

        v2_api_url = self._convert_to_v2_manifests_url(full_image_name)
        res = self.image_registry_http_client.get(v2_api_url, headers=self._get_accept_headers())
        if res.status_code == http.HTTPStatus.OK:
            return cast(Dict[str, str], res.json())
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
            original_exception=ValueError(
                f"Failed to retrieve manifest for {full_image_name}. \n"
                f"HTTP status code: {res.status_code}. Full response: {res.text}."
            ),
        )

    def add_tag_to_remote_image(self, original_full_image_name: str, new_tag: str) -> None:
        """Add a tag to an image in the registry.

        Args:
            original_full_image_name: The full image name is required to fetch manifest.
            new_tag:  New tag to be added to the image.

        Returns:
            None

        Raises:
            SnowflakeMLException: when failed to push the newly updated manifest.
        """

        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            return None

        full_image_name_parts = original_full_image_name.split(":")
        assert len(full_image_name_parts) == 2, "full image name should include both image name and tag"
        new_full_image_name = ":".join([full_image_name_parts[0], new_tag])
        if self.image_exists(new_full_image_name):
            # Early return if image with the associated tag already exists.
            return
        api_url = self._convert_to_v2_manifests_url(new_full_image_name)
        manifest = self._get_manifest(full_image_name=original_full_image_name)
        manifest_copy = manifest.copy()
        manifest_copy["tag"] = new_tag
        headers = self._get_accept_headers()
        # Http Content-Type does not support multi-value, hence need to construct separate header.
        put_header_v1 = {
            **headers,
            "Content-Type": _MANIFEST_V1_HEADER,
        }
        put_header_v2 = {
            **headers,
            "Content-Type": _MANIFEST_V2_HEADER,
        }

        res1 = self.image_registry_http_client.put(api_url, headers=put_header_v1, json=manifest_copy)
        if res1.status_code != http.HTTPStatus.CREATED:
            res2 = self.image_registry_http_client.put(api_url, headers=put_header_v2, json=manifest_copy)
            if res2.status_code != http.HTTPStatus.CREATED:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
                    original_exception=ValueError(
                        f"Failed to push manifest for {new_full_image_name}. Two requests filed: \n"
                        f"HTTP status code 1: {res1.status_code}. Full response 1: {res1.text}. \n"
                        f"HTTP status code 2: {res2.status_code}. Full response 2: {res2.text}"
                    ),
                )
        assert self.image_exists(
            new_full_image_name
        ), f"{new_full_image_name} should exist in image repo after a successful manifest update"

    def copy_image(
        self,
        source_image_with_digest: str,
        dest_image_with_tag: str,
        arch: Optional[imagelib._Arch] = None,
    ) -> None:
        """Util function to copy image across registry. Currently supported pulling from public docker image repo to
        SPCS image registry.

        Args:
            source_image_with_digest: source image with digest, e.g. gcr.io/kaniko-project/executor@sha256:b8c0977
            dest_image_with_tag: destination image with tag.
            arch: architecture of source image.

        Returns:
            None
        """
        logger.info(f"Copying image from {source_image_with_digest} to {dest_image_with_tag}")
        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            logger.warning(f"Running inside Sproc. Please ensure image already exists at {dest_image_with_tag}")
            return None

        arch = arch or imagelib._Arch("amd64", "linux")

        src_image = imagelib.convert_to_image_descriptor(source_image_with_digest, with_digest=True)
        dest_image = imagelib.convert_to_image_descriptor(
            dest_image_with_tag,
            with_tag=True,
        )
        # TODO[shchen]: Remove the imagelib, instead rely on the copy image system function later.
        imagelib.copy_image(
            src_image=src_image, dest_image=dest_image, arch=arch, retryable_http=self.image_registry_http_client
        )
        logger.info("Image copy completed successfully")
