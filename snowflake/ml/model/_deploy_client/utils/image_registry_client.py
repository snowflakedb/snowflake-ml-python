import http
import json
import logging
from typing import Dict, Optional, cast
from urllib.parse import urlparse, urlunparse

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import retryable_http, spcs_image_registry
from snowflake.ml.model._deploy_client.utils import image_auth_manager, imagelib
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils

MANIFEST_V1_HEADER = "application/vnd.oci.image.manifest.v1+json"
MANIFEST_V2_HEADER = "application/vnd.docker.distribution.manifest.v2+json"

logger = logging.getLogger(__name__)


class ImageRegistryClient:
    """
    A simple SPCS image registry HTTP client partial implementation. This client exists due to current unavailability
    of registry "list image" system function and lack of registry SDK.
    """

    def __init__(self, session: Session) -> None:
        """Initialization

        Args:
            session: Snowpark session
        """
        self.session = session
        self.http = retryable_http.get_http_client()

    def login(self, repo_url: str, registry_cred: str) -> str:
        """Log in to image registry

        Args:
            repo_url: image repo url.
            registry_cred: registry basic auth credential.

        Returns:
            Bearer token when login succeeded.

        Raises:
            SnowflakeMLException: when login failed.
        """
        parsed_url = urlparse(repo_url)
        scheme = parsed_url.scheme
        host = parsed_url.netloc

        login_path = "/login"  # Construct the login path
        url_tuple = (scheme, host, login_path, "", "", "")
        login_url = urlunparse(url_tuple)

        resp = self.http.get(login_url, headers={"Authorization": f"Basic {registry_cred}"})
        if resp.status_code != http.HTTPStatus.OK:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
                original_exception=RuntimeError(
                    f"Failed to login to the repository. Status {resp.status_code}," f"{str(resp.text)}"
                ),
            )

        return str(json.loads(resp.text)["token"])

    def convert_to_v2_manifests_url(self, full_image_name: str) -> str:
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

        with spcs_image_registry.generate_image_registry_credential(self.session) as registry_cred:
            v2_api_url = self.convert_to_v2_manifests_url(full_image_name)
            bearer_login = self.login(v2_api_url, registry_cred)

            headers_v1 = {
                "Authorization": f"Bearer {bearer_login}",
                "Accept": MANIFEST_V1_HEADER,
            }

            headers_v2 = {
                "Authorization": f"Bearer {bearer_login}",
                "Accept": MANIFEST_V2_HEADER,
            }
            # Depending on the built image, the media type of the image manifest might be either
            # application/vnd.oci.image.manifest.v1+json or application/vnd.docker.distribution.manifest.v2+json
            # Hence we need to check for both, otherwise it could result in false negative.
            if self.http.head(v2_api_url, headers=headers_v2).status_code == http.HTTPStatus.OK:
                return True
            elif self.http.head(v2_api_url, headers=headers_v1).status_code == http.HTTPStatus.OK:
                return True
            return False

    def _get_manifest(
        self, full_image_name: str, header_v1: Dict[str, str], header_v2: Dict[str, str]
    ) -> Dict[str, str]:
        """Retrieve image manifest file. Given Docker manifest comes with two versions, and for each version the
        corresponding request header is required for a successful HTTP response.

        Args:
            full_image_name: Full image name.
            header_v1: Docker manifest v1 header.
            header_v2: Docker manifest v2 header.

        Returns:
            Full manifest content as a python dict.

        Raises:
            SnowflakeMLException: when failed to retrieve manifest.
        """

        v2_api_url = self.convert_to_v2_manifests_url(full_image_name)
        res1 = self.http.get(v2_api_url, headers=header_v2)
        if res1.status_code == http.HTTPStatus.OK:
            return cast(Dict[str, str], res1.json())
        res2 = self.http.get(v2_api_url, headers=header_v1)
        if res2.status_code == http.HTTPStatus.OK:
            return cast(Dict[str, str], res2.json())
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
            original_exception=ValueError(
                f"Failed to retrieve manifest for {full_image_name}. Two requests filed: \n"
                f"HTTP status code 1: {res1.status_code}. Full response 1: {res1.text}. \n"
                f"HTTP status code 2: {res2.status_code}. Full response 2: {res2.text}"
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

        with spcs_image_registry.generate_image_registry_credential(self.session) as registry_cred:
            full_image_name_parts = original_full_image_name.split(":")
            assert len(full_image_name_parts) == 2, "full image name should include both image name and tag"
            new_full_image_name = ":".join([full_image_name_parts[0], new_tag])
            if self.image_exists(new_full_image_name):
                # Early return if image with the associated tag already exists.
                return
            api_url = self.convert_to_v2_manifests_url(new_full_image_name)
            # Login again to avoid token timeout issue.
            bearer_login = self.login(api_url, registry_cred)
            header_v1 = {
                "Authorization": f"Bearer {bearer_login}",
                "Accept": MANIFEST_V1_HEADER,
            }
            header_v2 = {"Authorization": f"Bearer {bearer_login}", "Accept": MANIFEST_V2_HEADER}

            manifest = self._get_manifest(
                full_image_name=original_full_image_name, header_v1=header_v1, header_v2=header_v2
            )
            manifest_copy = manifest.copy()
            manifest_copy["tag"] = new_tag

            put_header_v1 = {
                **header_v1,
                "Content-Type": MANIFEST_V1_HEADER,
            }
            put_header_v2 = {
                **header_v2,
                "Content-Type": MANIFEST_V2_HEADER,
            }

            res1 = self.http.put(api_url, headers=put_header_v2, json=manifest_copy)
            if res1.status_code != http.HTTPStatus.CREATED:
                res2 = self.http.put(api_url, headers=put_header_v1, json=manifest_copy)
                if res2.status_code != http.HTTPStatus.CREATED:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
                        original_exception=ValueError(
                            f"Failed to push manifest for {new_full_image_name}. Two requests filed: \n"
                            f"HTTP status code 1: {res1.status_code}. Full response 1: {res1.text}. \n"
                            f"HTTP status code 2: {res2.status_code}. Full response 2: {res2.text}"
                        ),
                    )
            assert self.image_exists(new_full_image_name), (
                f"{new_full_image_name} should exist in image repo after a" f"successful manifest update"
            )

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
        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            logger.warning(f"Running inside Sproc. Please ensure image already exists at {dest_image_with_tag}")
            return None

        arch = arch or imagelib._Arch("amd64", "linux")

        src_image = imagelib.convert_to_image_descriptor(source_image_with_digest, with_digest=True)
        dest_image = imagelib.convert_to_image_descriptor(
            dest_image_with_tag,
            with_tag=True,
            creds_manager=image_auth_manager.SnowflakeAuthManager(dest_image_with_tag.split("/")[0]),
        )
        imagelib.copy_image(src_image=src_image, dest_image=dest_image, arch=arch, session=self.session)
