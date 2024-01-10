"""
A minimal pure python library to copy images between two remote registries.

This library only supports a limited set of features:
- Works only with docker and OCI manifests and manifest lists for multiarch images (most newer images)
  - Supported OCI manifest type: application/vnd.oci.image.manifest.v1+json
  - Supported Docker manifest type: application/vnd.docker.distribution.manifest.v2+json
- Supports only pulling a single architecture from a multiarch image. Does not support pulling all architectures.
- Supports only schemaVersion 2.
- Streams images from source to destination without any intermediate disk storage in chunks.
- Does not support copying in parallel.

It's recommended to use this library to copy previously tested images using sha256 to avoid surprises
with respect to compatibility.
"""
import dataclasses
import hashlib
import io
import json
import logging
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import requests

from snowflake.ml._internal.container_services.image_registry import (
    http_client as image_registry_http_client,
)

# Common HTTP headers
_CONTENT_LENGTH_HEADER = "content-length"
_CONTENT_TYPE_HEADER = "content-type"
_CONTENT_RANGE_HEADER = "content-range"
_LOCATION_HEADER = "location"
_AUTHORIZATION_HEADER = "Authorization"
_ACCEPT_HEADER = "accept"

_OCI_MANIFEST_LIST_TYPE = "application/vnd.oci.image.index.v1+json"
_DOCKER_MANIFEST_LIST_TYPE = "application/vnd.docker.distribution.manifest.list.v2+json"

_OCI_MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
_DOCKER_MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"

ALL_SUPPORTED_MEDIA_TYPES = [
    _OCI_MANIFEST_LIST_TYPE,
    _DOCKER_MANIFEST_LIST_TYPE,
    _OCI_MANIFEST_TYPE,
    _DOCKER_MANIFEST_TYPE,
]
_MANIFEST_SUPPORTED_KEYS = {"schemaVersion", "mediaType", "config", "layers"}

# Architecture descriptor as a named tuple
_Arch = namedtuple("_Arch", ["arch_name", "os"])

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ImageDescriptor:
    """
    Create an image descriptor.

    registry_name: the name of the registry like gcr.io
    repository_name: the name of the repository like kaniko-project/executor
    tag: the tag of the image like v1.6.0
    digest: the sha256 digest of the image like sha256:b8c0...
    protocol: the protocol to use, defaults to https

    Only a tag or a digest must be specified, not both.
    """

    registry_name: str
    repository_name: str
    tag: Optional[str] = None
    digest: Optional[str] = None
    protocol: str = "https"

    def __baseurl(self) -> str:
        return f"{self.protocol}://{self.registry_name}/v2/"

    def manifest_link(self) -> str:
        return f"{self.__baseurl()}{self.repository_name}/manifests/{self.tag or self.digest}"

    def blob_link(self, digest: str) -> str:
        return f"{self.__baseurl()}{self.repository_name}/blobs/{digest}"

    def blob_upload_link(self) -> str:
        return f"{self.__baseurl()}{self.repository_name}/blobs/uploads/"

    def manifest_upload_link(self, tag: str) -> str:
        return f"{self.__baseurl()}{self.repository_name}/manifests/{tag}"

    def __str__(self) -> str:
        return f"{self.registry_name}/{self.repository_name}@{self.tag or self.digest}"


class Manifest:
    def __init__(self, manifest_bytes: bytes, manifest_digest: str) -> None:
        """Create a manifest object from the manifest JSON dict.

        Args:
            manifest_bytes: manifest content in bytes.
            manifest_digest: SHA256 digest.
        """
        self.manifest_bytes = manifest_bytes
        self.manifest = json.loads(manifest_bytes.decode("utf-8"))
        self.__validate(self.manifest)

        self.manifest_digest = manifest_digest
        self.media_type = self.manifest["mediaType"]

    def get_blob_digests(self) -> List[str]:
        """
        Get the list of blob digests from the manifest including config and layers.
        """
        blobs = []
        blobs.extend([x["digest"] for x in self.manifest["layers"]])
        blobs.append(self.manifest["config"]["digest"])

        return blobs

    def __validate(self, manifest: Dict[str, str]) -> None:
        """
        Validate the manifest.
        """
        assert (
            manifest.keys() == _MANIFEST_SUPPORTED_KEYS
        ), f"Manifest must contain all keys and no more {_MANIFEST_SUPPORTED_KEYS}"
        assert int(manifest["schemaVersion"]) == 2, "Only manifest schemaVersion 2 is supported"
        assert manifest["mediaType"] in [
            _OCI_MANIFEST_TYPE,
            _DOCKER_MANIFEST_TYPE,
        ], f'Unsupported mediaType {manifest["mediaType"]}'

    def __str__(self) -> str:
        """
        Return the manifest as a string.
        """
        return json.dumps(self.manifest, indent=4)


@dataclasses.dataclass
class BlobTransfer:
    """
    Helper class to transfer a blob from one registry to another
    in small chunks using in-memory buffering.
    """

    # Uploads in chunks of 1MB
    chunk_size_bytes = 1024 * 1024

    src_image: ImageDescriptor
    dest_image: ImageDescriptor
    manifest: Manifest
    image_registry_http_client: image_registry_http_client.ImageRegistryHttpClient

    def upload_all_blobs(self) -> None:
        blob_digests = self.manifest.get_blob_digests()
        logger.debug(f"Found {len(blob_digests)} blobs for {self.src_image}")

        for blob_digest in blob_digests:
            logger.debug(f"Transferring blob {blob_digest} from {self.src_image} to {self.dest_image}")
            if self._should_upload(blob_digest):
                self._transfer(blob_digest)
            else:
                logger.debug(f"Blob {blob_digest} already exists in {self.dest_image}")

    def _should_upload(self, blob_digest: str) -> bool:
        """
        Check if the blob already exists in the destination registry.
        """
        resp = self.image_registry_http_client.head(self.dest_image.blob_link(blob_digest), headers={})
        return resp.status_code != 200

    def _fetch_blob(self, blob_digest: str) -> Tuple[io.BytesIO, int]:
        """
        Fetch a stream to the blob from the source registry.
        """
        src_blob_link = self.src_image.blob_link(blob_digest)
        headers = {_CONTENT_LENGTH_HEADER: "0"}
        resp = self.image_registry_http_client.get(src_blob_link, headers=headers)

        assert resp.status_code == 200, f"Blob GET failed with code {resp.status_code}"
        assert _CONTENT_LENGTH_HEADER in resp.headers, f"Blob does not contain {_CONTENT_LENGTH_HEADER}"

        return io.BytesIO(resp.content), int(resp.headers[_CONTENT_LENGTH_HEADER])

    def _get_upload_url(self) -> str:
        """
        Obtain the upload URL from the destination registry.
        """
        response = self.image_registry_http_client.post(self.dest_image.blob_upload_link())
        assert (
            response.status_code == 202
        ), f"Failed to get the upload URL to destination. Status {response.status_code}. {str(response.content)}"
        return str(response.headers[_LOCATION_HEADER])

    def _upload_blob(self, blob_digest: str, blob_data: io.BytesIO, content_length: int) -> None:
        """
        Upload a blob to the destination registry.
        """
        upload_url = self._get_upload_url()
        headers = {
            _CONTENT_TYPE_HEADER: "application/octet-stream",
        }

        # Use chunked transfer
        # This can be optimized to use a single PUT request for small blobs
        next_loc = upload_url
        start_byte = 0
        while start_byte < content_length:
            chunk = blob_data.read(self.chunk_size_bytes)
            chunk_length = len(chunk)
            end_byte = start_byte + chunk_length - 1

            headers[_CONTENT_RANGE_HEADER] = f"{start_byte}-{end_byte}"
            headers[_CONTENT_LENGTH_HEADER] = str(chunk_length)

            resp = self.image_registry_http_client.patch(next_loc, headers=headers, data=chunk)
            assert resp.status_code == 202, f"Blob PATCH failed with code {resp.status_code}"

            next_loc = resp.headers[_LOCATION_HEADER]
            start_byte += chunk_length

        # Finalize the upload
        resp = self.image_registry_http_client.put(f"{next_loc}&digest={blob_digest}")
        assert resp.status_code == 201, f"Blob PUT failed with code {resp.status_code}"

    def _transfer(self, blob_digest: str) -> None:
        """
        Transfer a blob from the source registry to the destination registry.
        """
        blob_data, content_length = self._fetch_blob(blob_digest)
        self._upload_blob(blob_digest, blob_data, content_length)


def get_bytes_with_sha_verification(resp: requests.Response, sha256_digest: str) -> Tuple[bytes, str]:
    """Get the bytes of a response and verify the sha256 digest.

    Args:
        resp: the response object
        sha256_digest: the expected sha256 digest in format "sha256:b8c0..."

    Returns:
        (res, sha256_digest)

    """
    digest = hashlib.sha256()
    chunks = []
    for chunk in resp.iter_content(chunk_size=8192):
        digest.update(chunk)
        chunks.append(chunk)

    calculated_digest = digest.hexdigest()
    assert not sha256_digest or sha256_digest.endswith(calculated_digest), "SHA256 digest does not match"

    content = b"".join(chunks)  # Minimize allocations by joining chunks
    return content, calculated_digest


def get_manifest(
    image_descriptor: ImageDescriptor, arch: _Arch, retryable_http: image_registry_http_client.ImageRegistryHttpClient
) -> Manifest:
    """Get the manifest of an image from the remote registry.

    Args:
        image_descriptor: the image descriptor
        arch: the architecture to filter for if it's a multi-arch image
        retryable_http: a retryable http client.

    Returns:
        Manifest object.

    """
    logger.debug(f"Getting manifest from {image_descriptor.manifest_link()}")

    headers = {_ACCEPT_HEADER: ",".join(ALL_SUPPORTED_MEDIA_TYPES)}

    response = retryable_http.get(image_descriptor.manifest_link(), headers=headers)
    assert response.status_code == 200, f"Manifest GET failed with code {response.status_code}, {response.text}"

    assert image_descriptor.digest
    manifest_bytes, manifest_digest = get_bytes_with_sha_verification(response, image_descriptor.digest)
    manifest_json = json.loads(manifest_bytes.decode("utf-8"))

    # If this is a manifest list, find the manifest for the specified architecture
    # and recurse till we find the real manifest
    if manifest_json["mediaType"] in [
        _OCI_MANIFEST_LIST_TYPE,
        _DOCKER_MANIFEST_LIST_TYPE,
    ]:
        logger.debug("Found a multiarch image. Following manifest reference.")

        assert "manifests" in manifest_json, "Manifest list does not contain manifests"
        qualified_manifests = [
            x
            for x in manifest_json["manifests"]
            if x["platform"]["architecture"] == arch.arch_name and x["platform"]["os"] == arch.os
        ]
        assert (
            len(qualified_manifests) == 1
        ), "Manifest list does not contain exactly one qualified manifest for this arch"

        manifest_object = qualified_manifests[0]
        manifest_digest = manifest_object["digest"]

        logger.debug(f"Found manifest reference for arch {arch}: {manifest_digest}")

        # Copy the image descriptor to fetch the arch-specific manifest
        descriptor_copy = ImageDescriptor(
            registry_name=image_descriptor.registry_name,
            repository_name=image_descriptor.repository_name,
            digest=manifest_digest,
            tag=None,
        )

        # Supports only one level of manifest list nesting to avoid infinite recursion
        return get_manifest(descriptor_copy, arch, retryable_http)

    return Manifest(manifest_bytes, manifest_digest)


def put_manifest(
    image_descriptor: ImageDescriptor,
    manifest: Manifest,
    retryable_http: image_registry_http_client.ImageRegistryHttpClient,
) -> None:
    """
    Upload the given manifest to the destination registry.
    """
    assert image_descriptor.tag is not None, "Tag must be specified for manifest upload"
    headers = {_CONTENT_TYPE_HEADER: manifest.media_type}
    url = image_descriptor.manifest_upload_link(image_descriptor.tag)
    logger.debug(f"Uploading manifest to {url}")
    response = retryable_http.put(url, headers=headers, data=manifest.manifest_bytes)
    assert response.status_code == 201, f"Manifest PUT failed with code {response.status_code}"


def copy_image(
    src_image: ImageDescriptor,
    dest_image: ImageDescriptor,
    arch: _Arch,
    retryable_http: image_registry_http_client.ImageRegistryHttpClient,
) -> None:
    logger.debug(f"Pulling image manifest for {src_image}")

    # 1. Get the manifest
    manifest = get_manifest(src_image, arch, retryable_http)
    logger.debug(f"Manifest pulled for {src_image} with digest {manifest.manifest_digest}")

    # 2: Retrieve all blob digests from manifest; fetch blob based on blob digest, then upload blob.
    blob_transfer = BlobTransfer(src_image, dest_image, manifest, image_registry_http_client=retryable_http)
    blob_transfer.upload_all_blobs()

    # 3. Upload the manifest
    logger.debug(f"All blobs copied successfully. Copying manifest for {src_image} to {dest_image}")
    put_manifest(dest_image, manifest, retryable_http)

    logger.debug(f"Image {src_image} copied to {dest_image}")


def convert_to_image_descriptor(
    image_name: str,
    with_digest: bool = False,
    with_tag: bool = False,
) -> ImageDescriptor:
    """Convert a full image name to a ImageDescriptor object.

    Args:
        image_name: name of image.
        with_digest: boolean to specify whether a digest is included in the image name
        with_tag: boolean to specify whether a tag is included in the image name.

    Returns:
        An ImageDescriptor instance
    """
    assert with_digest or with_tag, "image should contain either digest or tag"
    sep = "@" if with_digest else ":"
    parts = image_name.split("/")
    assert len(parts[-1].split(sep)) == 2, f"Image {image_name} missing digest/tag"
    tag_digest = parts[-1].split(sep)[1]
    return ImageDescriptor(
        registry_name=parts[0],
        repository_name="/".join(parts[1:-1] + [parts[-1].split(sep)[0]]),
        digest=tag_digest if with_digest else None,
        tag=tag_digest if with_tag else None,
    )
