import importlib
import os
import shutil
import string
from abc import ABC

from snowflake.ml.model._deploy_client.utils import constants


class DockerContext(ABC):
    """
    Constructs the Docker context directory required for image building.
    """

    def __init__(self, context_dir: str, model_dir: str, *, use_gpu: bool = False) -> None:
        """Initialization

        Args:
            context_dir: Path to context directory.
            model_dir: Path to model directory.
            use_gpu: Boolean flag for generating the CPU or GPU base image.
        """
        self.context_dir = context_dir
        self.model_dir = model_dir
        # TODO(shchen): SNOW-825995, Define dockerfile template used for model deployment. use_gpu will be used.
        self.use_gpu = use_gpu

    def build(self) -> None:
        """
        Generates and/or moves resources into the Docker context directory.Rename the random model directory name to
        constant "model_dir" instead for better readability.
        """
        shutil.copytree(self.model_dir, "/".join([self.context_dir.rstrip("/"), constants.MODEL_DIR]))
        self._generate_inference_code()
        self._copy_entrypoint_script_to_docker_context()
        self._copy_snowml_source_code_to_docker_context()
        self._generate_docker_file()

    def _copy_snowml_source_code_to_docker_context(self) -> None:
        """Copy the entire snowflake/ml source code to docker context. This will be particularly useful for CI tests
        against latest changes.

        Note that we exclude the experimental directory mainly for development scenario; as experimental directory won't
        be included in the release.
        """
        snow_ml_source_dir = list(importlib.import_module("snowflake.ml").__path__)[0]
        shutil.copytree(
            snow_ml_source_dir,
            os.path.join(self.context_dir, "snowflake", "ml"),
            ignore=shutil.ignore_patterns("*.pyc", "experimental"),
        )

    def _copy_entrypoint_script_to_docker_context(self) -> None:
        """Copy gunicorn_run.sh entrypoint to docker context directory."""
        path = os.path.join(os.path.dirname(__file__), constants.ENTRYPOINT_SCRIPT)
        assert os.path.exists(path), f"Run script file missing at path: {path}"
        shutil.copy(path, os.path.join(self.context_dir, constants.ENTRYPOINT_SCRIPT))

    def _generate_docker_file(self) -> None:
        """
        Generates dockerfile based on dockerfile template.
        """
        docker_file_path = os.path.join(self.context_dir, "Dockerfile")
        docker_file_template = os.path.join(os.path.dirname(__file__), "templates/dockerfile_template")

        with open(docker_file_path, "w") as dockerfile, open(docker_file_template) as template:
            dockerfile_content = string.Template(template.read()).safe_substitute(
                {
                    # TODO(shchen): SNOW-835411, Support overwriting base image
                    "base_image": "mambaorg/micromamba:focal-cuda-11.7.1"
                    if self.use_gpu
                    else "mambaorg/micromamba:1.4.3",
                    "model_dir": constants.MODEL_DIR,
                    "inference_server_dir": constants.INFERENCE_SERVER_DIR,
                    "entrypoint_script": constants.ENTRYPOINT_SCRIPT,
                }
            )
            dockerfile.write(dockerfile_content)

    def _generate_inference_code(self) -> None:
        """
        Generates inference code based on the app template and creates a folder named 'server' to house the inference
        server code.
        """
        inference_server_folder_path = os.path.join(os.path.dirname(__file__), constants.INFERENCE_SERVER_DIR)
        destination_folder_path = os.path.join(self.context_dir, constants.INFERENCE_SERVER_DIR)
        ignore_patterns = shutil.ignore_patterns("BUILD.bazel", "*test.py", "*.\\.*", "__pycache__")
        shutil.copytree(inference_server_folder_path, destination_folder_path, ignore=ignore_patterns)
