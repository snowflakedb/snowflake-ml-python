import os
import shutil
import string
from abc import ABC

from packaging import version

from snowflake.ml.model import _model_meta
from snowflake.ml.model._deploy_client.utils import constants


class DockerContext(ABC):
    """
    Constructs the Docker context directory required for image building.
    """

    def __init__(self, context_dir: str, model_meta: _model_meta.ModelMetadata) -> None:
        """Initialization

        Args:
            context_dir: Path to context directory.
            model_meta: Model Metadata
        """
        self.context_dir = context_dir
        self.model_meta = model_meta

    def build(self) -> None:
        """
        Generates and/or moves resources into the Docker context directory.Rename the random model directory name to
        constant "model_dir" instead for better readability.
        """
        self._generate_inference_code()
        self._copy_entrypoint_script_to_docker_context()
        self._copy_model_env_dependency_to_docker_context()
        self._generate_docker_file()

    def _copy_entrypoint_script_to_docker_context(self) -> None:
        """Copy gunicorn_run.sh entrypoint to docker context directory."""
        path = os.path.join(os.path.dirname(__file__), constants.ENTRYPOINT_SCRIPT)
        assert os.path.exists(path), f"Run script file missing at path: {path}"
        shutil.copy(path, os.path.join(self.context_dir, constants.ENTRYPOINT_SCRIPT))

    def _copy_model_env_dependency_to_docker_context(self) -> None:
        """
        Convert model dependencies to files from model metadata.
        """
        self.model_meta.save_model_metadata(self.context_dir)

    def _generate_docker_file(self) -> None:
        """
        Generates dockerfile based on dockerfile template.
        """
        docker_file_path = os.path.join(self.context_dir, "Dockerfile")
        docker_file_template = os.path.join(os.path.dirname(__file__), "templates/dockerfile_template")

        if self.model_meta.cuda_version:
            cuda_version_parsed = version.parse(self.model_meta.cuda_version)
            cuda_version_str = f"{cuda_version_parsed.major}.{cuda_version_parsed.minor}"
        else:
            cuda_version_str = ""

        with open(docker_file_path, "w", encoding="utf-8") as dockerfile, open(
            docker_file_template, encoding="utf-8"
        ) as template:
            dockerfile_content = string.Template(template.read()).safe_substitute(
                {
                    # TODO(shchen): SNOW-835411, Support overwriting base image
                    "base_image": "mambaorg/micromamba:1.4.3",
                    "model_env_folder": constants.MODEL_ENV_FOLDER,
                    "inference_server_dir": constants.INFERENCE_SERVER_DIR,
                    "entrypoint_script": constants.ENTRYPOINT_SCRIPT,
                    # Instead of omitting this ENV var when no CUDA required, we explicitly set it to empty to override
                    # as no CUDA is detected thus it won't be affected by the existence of CUDA in base image.
                    # https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html
                    "cuda_override_env": cuda_version_str,
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
