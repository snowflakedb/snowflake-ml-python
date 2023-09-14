import json
import logging
import os
import subprocess
import tempfile
import time
from enum import Enum
from typing import List, Optional

from snowflake import snowpark
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import spcs_image_registry
from snowflake.ml.model import _model_meta
from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    docker_context,
)

logger = logging.getLogger(__name__)


class Platform(Enum):
    LINUX_AMD64 = "linux/amd64"


class ClientImageBuilder(base_image_builder.ImageBuilder):
    """
    Client-side image building and upload to model registry.

    Usage requirements:
    Requires prior installation and running of Docker with BuildKit. See installation instructions in
        https://docs.docker.com/engine/install/
    """

    def __init__(
        self,
        *,
        id: str,
        image_repo: str,
        model_meta: _model_meta.ModelMetadata,
        session: snowpark.Session,
        image_tag: Optional[str] = None,
    ) -> None:
        """Initialization

        Args:
            id: A hexadecimal string used for naming the image tag.
            image_repo: Path to image repository.
            model_meta: Model Metadata
            session: Snowpark session
            image_tag: Optional image tag name; when not provided, will use model id as the tag name.
        """
        self.image_tag = image_tag or "/".join([image_repo.rstrip("/"), id]) + ":latest"
        self.image_repo = image_repo
        self.model_meta = model_meta
        self.session = session

    def build_and_upload_image(self, image_to_pull: Optional[str] = None) -> str:
        """Builds and uploads an image to the model registry.

        Args:
            image_to_pull: When set, skips building image locally; instead, pull image directly from public
                repo. This is more of a workaround to support non-spcs-registry images.
                TODO[shchen] remove such logic when first-party-image is supported on snowservice registry.

        Returns:
            Snowservice registry image tag.

        Raises:
            SnowflakeMLException: Occurs when failed to build image or push to image registry.
        """

        def _setup_docker_config(docker_config_dir: str, registry_cred: str) -> None:
            """Set up a temporary docker config, which is used for running all docker commands. The format of config
            is based on the format that is compatible with docker credential helper:
            {
              "auths": {
                "https://index.docker.io/v1/": {
                  "auth": "<base_64_encoded_username_password>"
                }
              }
            }

            Args:
                docker_config_dir: Path to docker configuration directory, which stores the temporary session token.
                registry_cred: image registry basic auth credential.
            """
            content = {"auths": {self.image_tag: {"auth": registry_cred}}}
            config_path = os.path.join(docker_config_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(content, file)

        def _cleanup_local_image(docker_config_dir: str) -> None:
            try:
                image_exist_command = ["docker", "image", "inspect", self.image_tag]
                self._run_docker_commands(image_exist_command)
            except Exception:
                # Image does not exist, probably due to failed build step
                pass
            else:
                commands = ["docker", "--config", docker_config_dir, "rmi", self.image_tag]
                logger.debug(f"Removing local image: {self.image_tag}")
                self._run_docker_commands(commands)

        self.validate_docker_client_env()
        with spcs_image_registry.generate_image_registry_credential(
            self.session
        ) as registry_cred, tempfile.TemporaryDirectory() as docker_config_dir:
            try:
                _setup_docker_config(docker_config_dir=docker_config_dir, registry_cred=registry_cred)
                if not image_to_pull:
                    start = time.time()
                    self._build_and_tag(docker_config_dir)
                    end = time.time()
                    logger.info(f"Time taken to build the image on the client: {end - start:.2f} seconds")
                else:
                    self._pull_and_tag(image_to_pull=image_to_pull)
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_DOCKER_ERROR,
                    original_exception=RuntimeError("Failed to build docker image."),
                ) from e
            else:
                try:
                    start = time.time()
                    self._upload(docker_config_dir)
                    end = time.time()
                    logger.info(f"Time taken to upload the image to image registry: {end - start:.2f} seconds")
                except Exception as e:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_DOCKER_ERROR,
                        original_exception=RuntimeError("Failed to upload docker image to registry."),
                    ) from e
                finally:
                    _cleanup_local_image(docker_config_dir)
        return self.image_tag

    def validate_docker_client_env(self) -> None:
        """Ensure docker client is running and BuildKit is enabled. Note that Buildx always uses BuildKit.
        - Ensure docker daemon is running through the "docker info" command on shell. When docker daemon is running,
        return code will be 0, else return code will be 1.
        - Ensure BuildKit is enabled by checking "docker buildx version".

        Raises:
            SnowflakeMLException: Occurs when Docker is not installed or is not running.

        """
        try:
            self._run_docker_commands(["docker", "info"])
        except Exception:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.CLIENT_DEPENDENCY_MISSING_ERROR,
                original_exception=ConnectionError(
                    "Failed to initialize Docker client. Please ensure Docker is installed and running."
                ),
            )

        try:
            self._run_docker_commands(["docker", "buildx", "version"])
        except Exception:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.CLIENT_DEPENDENCY_MISSING_ERROR,
                original_exception=ConnectionError(
                    "Please ensured Docker is installed with BuildKit by following "
                    "https://docs.docker.com/build/buildkit/#getting-started"
                ),
            )

    def _build_and_tag(self, docker_config_dir: str) -> None:
        """Constructs the Docker context directory and then builds a Docker image based on that context.

        Args:
            docker_config_dir: Path to docker configuration directory, which stores the temporary session token.
        """

        with tempfile.TemporaryDirectory() as context_dir:
            dc = docker_context.DockerContext(context_dir=context_dir, model_meta=self.model_meta)
            dc.build()
            self._build_image_from_context(context_dir=context_dir, docker_config_dir=docker_config_dir)

    def _pull_and_tag(self, image_to_pull: str, platform: Platform = Platform.LINUX_AMD64) -> None:
        """Pull image from public docker hub repo. Then tag it with the specified image tag

        Args:
            image_to_pull: Name of image to download.
            platform: Specifies the target platform that matches the image to be downloaded
        """

        commands = ["docker", "pull", "--platform", platform.value, image_to_pull]
        logger.debug(f"Running {str(commands)}")
        self._run_docker_commands(commands)

        commands = ["docker", "tag", image_to_pull, self.image_tag]
        logger.debug(f"Running {str(commands)}")
        self._run_docker_commands(commands)

    def _run_docker_commands(self, commands: List[str]) -> None:
        """Run docker commands in a new child process.

        Args:
            commands: List of commands to run.

        Raises:
            SnowflakeMLException: Occurs when docker commands failed to execute.
        """
        proc = subprocess.Popen(
            commands, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False
        )
        output_lines = []

        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                output_lines.append(line)
                logger.debug(line)

        if proc.wait():
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_DOCKER_ERROR,
                original_exception=RuntimeError(f"Docker commands failed: \n {''.join(output_lines)}"),
            )

    def _build_image_from_context(
        self, context_dir: str, docker_config_dir: str, *, platform: Platform = Platform.LINUX_AMD64
    ) -> None:
        """Builds a Docker image based on provided context.

        Args:
            context_dir: Path to context directory.
            docker_config_dir: Path to docker configuration directory, which stores the temporary session token.
            platform: Target platform for the build output, in the format "os[/arch[/variant]]".
        """

        commands = [
            "docker",
            "--config",
            docker_config_dir,
            "buildx",
            "build",
            "--platform",
            platform.value,
            "--tag",
            f"{self.image_tag}",
            context_dir,
        ]

        self._run_docker_commands(commands)

    def _upload(self, docker_config_dir: str) -> None:
        """
        Uploads image to the image registry. This process requires a "docker login" followed by a "docker push". Remove
        local image at the end of the upload operation to save up local space. Image cache is kept for more performant
        built experience at the cost of small storage footprint.

        By default, Docker overwrites the local Docker config file "/.docker/config.json" whenever a docker login
        occurs. However, to ensure better isolation between Snowflake-managed Docker credentials and the user's own
        Docker credentials, we will not use the default Docker config. Instead, we will write the username and session
        token to a temporary file and use "docker --config" so that it only applies to the specific Docker command being
        executed, without affecting the user's local Docker setup. The credential file will be automatically removed
        at the end of upload operation.

        Args:
            docker_config_dir: Path to docker configuration directory, which stores the temporary session token.
        """
        commands = ["docker", "--config", docker_config_dir, "login", self.image_tag]
        self._run_docker_commands(commands)

        logger.debug(f"Pushing image to image repo {self.image_tag}")
        commands = ["docker", "--config", docker_config_dir, "push", self.image_tag]
        self._run_docker_commands(commands)
