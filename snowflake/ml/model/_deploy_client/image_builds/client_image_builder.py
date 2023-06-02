import tempfile
from enum import Enum

import docker

from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    docker_context,
)


class Platform(Enum):
    LINUX_AMD64 = "linux/amd64"


class ClientImageBuilder(base_image_builder.ImageBuilder):
    """
    Client-side image building and upload to model registry.

    Requires prior installation and running of Docker.

    See https://docs.docker.com/engine/install/ for official installation instructions.
    """

    def __init__(self, *, id: str, image_repo: str, model_dir: str, use_gpu: bool = False) -> None:
        """Initialization

        Args:
            id: A hexadecimal string used for naming the image tag.
            image_repo: Path to image repository.
            model_dir: Path to model directory.
            use_gpu: Boolean flag for generating the CPU or GPU base image.
        """
        self.image_tag = "/".join([image_repo.rstrip("/"), id])
        self.image_repo = image_repo
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self._docker_client = None

    @property
    def docker_client(self) -> docker.DockerClient:
        """Creates a Docker client object for interacting with the Docker daemon running on the local machine.

        Raises:
            ConnectionError: Occurs when Docker is not installed or is not running.

        Returns:
            A docker.DockerClient object representing the Docker client.
        """
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except docker.errors.DockerException:
                raise ConnectionError(
                    "Failed to initialize Docker client. Please ensure Docker is installed and running."
                )
        return self._docker_client

    def build_and_upload_image(self) -> None:
        """
        Builds and uploads an image to the model registry.
        """
        self._build()
        self._upload()

    def _build(self) -> None:
        """
        Constructs the Docker context directory and then builds a Docker image based on that context.
        """
        with tempfile.TemporaryDirectory() as context_dir:
            dc = docker_context.DockerContext(context_dir=context_dir, model_dir=self.model_dir, use_gpu=self.use_gpu)
            dc.build()
            self._build_image_from_context(context_dir)

    def _build_image_from_context(self, context_dir: str, *, platform: Platform = Platform.LINUX_AMD64) -> None:
        """Builds a Docker image based on provided context.

        Args:
            context_dir: Path to context directory.
            platform: Target platform for the build output, in the format "os[/arch[/variant]]".
        """
        self.docker_client.images.build(path=context_dir, tag=self.image_tag, platform=platform)

    def _upload(self) -> None:
        """
        Uploads image to image registry.
        """
        pass
