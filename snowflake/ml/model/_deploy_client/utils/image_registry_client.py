import json
from urllib.parse import urlparse, urlunparse

import requests

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import spcs_image_registry
from snowflake.snowpark import Session


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

        resp = requests.get(login_url, headers={"Authorization": f"Basic {registry_cred}"})
        if resp.status_code != 200:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWFLAKE_IMAGE_REGISTRY_ERROR,
                original_exception=RuntimeError("Failed to login to the repository", resp.text),
            )

        return str(json.loads(resp.text)["token"])

    def convert_to_v2_head_manifests_url(self, full_image_name: str) -> str:
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

        with spcs_image_registry.generate_image_registry_credential(self.session) as registry_cred:
            v2_api_url = self.convert_to_v2_head_manifests_url(full_image_name)
            bearer_login = self.login(v2_api_url, registry_cred)

            headers_v1 = {
                "Authorization": f"Bearer {bearer_login}",
                "Accept": "application/vnd.oci.image.manifest.v1+json",
            }

            headers_v2 = {
                "Authorization": f"Bearer {bearer_login}",
                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            }
            # Depending on the built image, the media type of the image manifest might be either
            # application/vnd.oci.image.manifest.v1+json or application/vnd.docker.distribution.manifest.v2+json
            # Hence we need to check for both, otherwise it could result in false negative.
            if requests.head(v2_api_url, headers=headers_v1).status_code == 200:
                return True
            elif requests.head(v2_api_url, headers=headers_v2).status_code == 200:
                return True
            return False
