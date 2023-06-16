import logging
import os
import subprocess
import tempfile
from enum import Enum
from typing import List

from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    docker_context,
)


class Platform(Enum):
    LINUX_AMD64 = "linux/amd64"


class ClientImageBuilder(base_image_builder.ImageBuilder):
    """
    Client-side image building and upload to model registry.

    Usage requirements:
    Requires prior installation and running of Docker with BuildKit. See installation instructions in
        https://docs.docker.com/engine/install/


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

    def build_and_upload_image(self) -> None:
        """
        Builds and uploads an image to the model registry.
        """
        self.validate_docker_client_env()
        self._build()
        self._upload()

    def validate_docker_client_env(self) -> None:
        """Ensure docker client is running and BuildKit is enabled. Note that Buildx always uses BuildKit.
        - Ensure docker daemon is running through the "docker info" command on shell. When docker daemon is running,
        return code will be 0, else return code will be 1.
        - Ensure BuildKit is enabled by checking "docker buildx version".

        Raises:
            ConnectionError: Occurs when Docker is not installed or is not running.

        """
        info_command = "docker info"
        buildx_command = "docker buildx version"

        try:
            subprocess.check_call(info_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        except subprocess.CalledProcessError:
            raise ConnectionError("Failed to initialize Docker client. Please ensure Docker is installed and running.")

        try:
            subprocess.check_call(buildx_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        except subprocess.CalledProcessError:
            raise ConnectionError(
                "Please ensured Docker is installed with BuildKit by following "
                "https://docs.docker.com/build/buildkit/#getting-started"
            )

    def _build(self) -> None:
        """
        Constructs the Docker context directory and then builds a Docker image based on that context.
        """
        with tempfile.TemporaryDirectory() as context_dir:
            dc = docker_context.DockerContext(context_dir=context_dir, model_dir=self.model_dir, use_gpu=self.use_gpu)
            dc.build()
            self._build_image_from_context(context_dir)

    def _run_docker_commands(self, commands: List[str]) -> None:
        """Run docker commands in a new child process.

        Args:
            commands: List of commands to run.

        Raises:
            RuntimeError: Occurs when docker commands failed to execute.
        """
        proc = subprocess.Popen(commands, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if proc.stdout:
            for x in iter(proc.stdout.readline, ""):
                logging.info(x)

        if proc.wait():
            raise RuntimeError("Docker build failed.")

    def _build_image_from_context(self, context_dir: str, *, platform: Platform = Platform.LINUX_AMD64) -> None:
        """Builds a Docker image based on provided context.

        Args:
            context_dir: Path to context directory.
            platform: Target platform for the build output, in the format "os[/arch[/variant]]".
        """

        commands = [
            "docker",
            "buildx",
            "build",
            "--platform",
            platform.value,
            "--tag",
            f"{self.image_tag}",
            context_dir,
        ]

        self._run_docker_commands(commands)

    def _upload(self) -> None:
        """
        Uploads image to image registry.
        """
        pass
