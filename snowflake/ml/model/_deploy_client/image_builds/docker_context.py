import os
import posixpath
import shutil
import string
from typing import Optional

import importlib_resources

from snowflake.ml._internal import file_utils
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model._deploy_client import image_builds
from snowflake.ml.model._deploy_client.utils import constants
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.snowpark import FileOperation, Session


class DockerContext:
    """
    Constructs the Docker context directory required for image building.
    """

    def __init__(
        self,
        context_dir: str,
        model_meta: model_meta.ModelMetadata,
        session: Optional[Session] = None,
        model_zip_stage_path: Optional[str] = None,
    ) -> None:
        """Initialization

        Args:
            context_dir: Path to context directory.
            model_meta: Model Metadata.
            session: Snowpark session.
            model_zip_stage_path: Path to model zip file on stage.
        """
        self.context_dir = context_dir
        self.model_meta = model_meta
        assert (session is None) == (model_zip_stage_path is None)
        self.session = session
        self.model_zip_stage_path = model_zip_stage_path

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
        script_path = importlib_resources.files(image_builds).joinpath(  # type: ignore[no-untyped-call]
            constants.ENTRYPOINT_SCRIPT
        )
        target_path = os.path.join(self.context_dir, constants.ENTRYPOINT_SCRIPT)

        with open(script_path, encoding="utf-8") as source_file, file_utils.open_file(target_path, "w") as target_file:
            target_file.write(source_file.read())

    def _copy_model_env_dependency_to_docker_context(self) -> None:
        """
        Convert model dependencies to files from model metadata.
        """
        self.model_meta.save(self.context_dir)

    def _generate_docker_file(self) -> None:
        """
        Generates dockerfile based on dockerfile template.
        """
        docker_file_path = os.path.join(self.context_dir, "Dockerfile")
        docker_file_template = (
            importlib_resources.files(image_builds)
            .joinpath("templates/dockerfile_template")  # type: ignore[no-untyped-call]
            .read_text("utf-8")
        )

        if self.model_zip_stage_path is not None:
            norm_stage_path = posixpath.normpath(identifier.remove_prefix(self.model_zip_stage_path, "@"))
            assert self.session
            fop = FileOperation(self.session)
            # The explicit download here is inefficient but a compromise.
            # We could in theory reuse the download needed for metadata extraction, but it's hacky and will go away.
            # Ideally, the model download should happen as part of the server side image build,
            # but it requires have our own image builder since there's need to be logic downloading model
            # into the context directory.
            get_res_list = fop.get(stage_location=self.model_zip_stage_path, target_directory=self.context_dir)
            assert len(get_res_list) == 1, f"Single zip file should be returned, but got {len(get_res_list)} files."
            local_zip_file_path = os.path.basename(get_res_list[0].file)
            copy_model_statement = f"COPY {local_zip_file_path} ./{norm_stage_path}"
            extra_env_statement = f"ENV MODEL_ZIP_STAGE_PATH={norm_stage_path}"
        else:
            copy_model_statement = ""
            extra_env_statement = ""

        with open(docker_file_path, "w", encoding="utf-8") as dockerfile:
            base_image = "mambaorg/micromamba:1.4.3"
            tag = base_image.split(":")[1]
            assert tag != constants.LATEST_IMAGE_TAG, (
                "Base image tag should not be 'latest' as it might cause false" "positive image cache hit"
            )
            dockerfile_content = string.Template(docker_file_template).safe_substitute(
                {
                    "base_image": "mambaorg/micromamba:1.4.3",
                    "model_env_folder": constants.MODEL_ENV_FOLDER,
                    "inference_server_dir": constants.INFERENCE_SERVER_DIR,
                    "entrypoint_script": constants.ENTRYPOINT_SCRIPT,
                    # Instead of omitting this ENV var when no CUDA required, we explicitly set it to empty to override
                    # as no CUDA is detected thus it won't be affected by the existence of CUDA in base image.
                    # https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html
                    "cuda_override_env": self.model_meta.env.cuda_version if self.model_meta.env.cuda_version else "",
                    "copy_model_statement": copy_model_statement,
                    "extra_env_statement": extra_env_statement,
                }
            )
            dockerfile.write(dockerfile_content)

    def _generate_inference_code(self) -> None:
        """
        Generates inference code based on the app template and creates a folder named 'server' to house the inference
        server code.
        """
        with importlib_resources.as_file(
            importlib_resources.files(image_builds).joinpath(  # type: ignore[no-untyped-call]
                constants.INFERENCE_SERVER_DIR
            )
        ) as inference_server_folder_path:
            destination_folder_path = os.path.join(self.context_dir, constants.INFERENCE_SERVER_DIR)
            ignore_patterns = shutil.ignore_patterns("BUILD.bazel", "*test.py", "*.\\.*", "__pycache__")
            file_utils.copytree(
                inference_server_folder_path,
                destination_folder_path,
                ignore=ignore_patterns,
            )
