import os
import shutil
import string
from abc import ABC


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
        Generates and/or moves resources into the Docker context directory.
        """
        # TODO(shchen): SNOW-826705, Install SnowML wheel on the inference container
        shutil.copytree(self.model_dir, "/".join([self.context_dir.rstrip("/"), os.path.basename(self.model_dir)]))
        self._generate_docker_file()
        self._generate_inference_code()

    def _generate_docker_file(self) -> None:
        """
        Generates dockerfile based on dockerfile template.
        """
        docker_file_path = os.path.join(self.context_dir, "Dockerfile")
        docker_file_template = os.path.join(os.path.dirname(__file__), "templates/dockerfile_template")

        with open(docker_file_path, "w") as dockerfile, open(docker_file_template) as template:
            dockerfile_content = string.Template(template.read()).safe_substitute()
            dockerfile.write(dockerfile_content)

    def _generate_inference_code(self) -> None:
        """
        Generates inference code based on the app template and creates a folder named 'server' to house the inference
        server code.
        """
        inference_server_folder_path = os.path.join(os.path.dirname(__file__), "inference_server")
        destination_folder_path = os.path.join(self.context_dir, "inference_server")
        ignore_patterns = shutil.ignore_patterns("BUILD.bazel", "*test.py", "*.\\.*")
        shutil.copytree(inference_server_folder_path, destination_folder_path, ignore=ignore_patterns)
