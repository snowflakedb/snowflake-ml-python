import base64
import json
import logging
import os
import subprocess
import tempfile
from enum import Enum
from typing import List

from snowflake import snowpark
from snowflake.ml._internal.utils import query_result_checker
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

    def __init__(
        self, *, id: str, image_repo: str, model_dir: str, session: snowpark.Session, use_gpu: bool = False
    ) -> None:
        """Initialization

        Args:
            id: A hexadecimal string used for naming the image tag.
            image_repo: Path to image repository.
            model_dir: Local model directory, downloaded form stage and extracted.
            use_gpu: Boolean flag for generating the CPU or GPU base image.
            session: Snowpark session
        """
        self.image_tag = "/".join([image_repo.rstrip("/"), id]) + ":latest"
        self.image_repo = image_repo
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.session = session

    def build_and_upload_image(self) -> str:
        """
        Builds and uploads an image to the model registry.
        """

        def _setup_docker_config(docker_config_dir: str) -> None:
            """Set up a temporary docker config, which is used for running all docker commands.

            Args:
                docker_config_dir: Path to docker configuration directory, which stores the temporary session token.
            """
            ctx = self.session._conn._conn
            assert ctx._rest, "SnowflakeRestful is not set in session"
            token_data = ctx._rest._token_request("ISSUE")
            snowpark_session_token = token_data["data"]["sessionToken"]
            token_obj = {"token": snowpark_session_token}
            credentials = f"0sessiontoken:{json.dumps(token_obj)}"
            encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
            content = {"auths": {self.image_tag: {"auth": encoded_credentials}}}
            config_path = os.path.join(docker_config_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(content, file)

        def _cleanup_local_image() -> None:
            try:
                image_exist_command = f"docker image inspect {self.image_tag}"
                subprocess.check_call(
                    image_exist_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True
                )
            except subprocess.CalledProcessError:
                # Image does not exist, probably due to failed build step
                pass
            else:
                commands = ["docker", "--config", config_dir, "rmi", self.image_tag]
                logging.info(f"Removing local image: {self.image_tag}")
                self._run_docker_commands(commands)

        self.validate_docker_client_env()

        query_result = (
            query_result_checker.SqlResultValidator(
                self.session,
                query="SHOW PARAMETERS LIKE 'PYTHON_CONNECTOR_QUERY_RESULT_FORMAT' IN SESSION",
            )
            .has_dimensions(expected_rows=1)
            .validate()
        )
        prev_format = query_result[0].value

        with tempfile.TemporaryDirectory() as config_dir:
            try:
                # Workaround for SNOW-841699: Fail to authenticate to image registry with session token generated from
                # Snowpark. Need to temporarily set the json query format in order to process GS token response.
                self.session.sql("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'json'").collect()
                _setup_docker_config(config_dir)
                self._build(config_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to build docker image: {str(e)}")
            else:
                try:
                    self._upload(config_dir)
                except Exception as e:
                    raise RuntimeError(f"Failed to upload docker image to registry: {str(e)}")
                finally:
                    _cleanup_local_image()
            finally:
                self.session.sql(f"ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = '{prev_format}'").collect()
        return self.image_tag

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

    def _build(self, docker_config_dir: str) -> None:
        """Constructs the Docker context directory and then builds a Docker image based on that context.

        Args:
            docker_config_dir: Path to docker configuration directory, which stores the temporary session token.
        """

        with tempfile.TemporaryDirectory() as context_dir:
            dc = docker_context.DockerContext(context_dir=context_dir, model_dir=self.model_dir, use_gpu=self.use_gpu)
            dc.build()
            self._build_image_from_context(context_dir=context_dir, docker_config_dir=docker_config_dir)

    def _run_docker_commands(self, commands: List[str]) -> None:
        """Run docker commands in a new child process.

        Args:
            commands: List of commands to run.

        Raises:
            RuntimeError: Occurs when docker commands failed to execute.
        """
        proc = subprocess.Popen(commands, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output_lines = []

        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                output_lines.append(line)
                logging.info(line)

        if proc.wait():
            raise RuntimeError(f"Docker commands failed: \n {''.join(output_lines)}")

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

        For image registry authentication, we will use a session token obtained from the Snowpark session object.
        The token authentication mechanism is automatically used when the username is set to "0sessiontoken" according
        to the registry implementation detailed in the following link:
        https://github.com/snowflakedb/snowflake-image-registry/blob/277435c6fd79db2df9f863aa9d04dc875e034d85
        /AuthAdapter/src/main/java/com/snowflake/registry/service/AuthHeader.java#L122

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

        logging.info(f"Pushing image to image repo {self.image_tag}")
        commands = ["docker", "--config", docker_config_dir, "push", self.image_tag]
        self._run_docker_commands(commands)
