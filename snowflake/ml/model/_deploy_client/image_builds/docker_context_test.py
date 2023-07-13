import os
import re
import shutil
import tempfile

from absl.testing import absltest

from snowflake.ml.model._deploy_client.image_builds import docker_context
from snowflake.ml.model._deploy_client.utils import constants


class DockerContextTest(absltest.TestCase):
    def setUp(self) -> None:
        self.context_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()
        self.use_gpu = False
        self.docker_context = docker_context.DockerContext(self.context_dir, model_dir=self.model_dir, use_gpu=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.context_dir)

    def test_build_results_in_correct_docker_context_file_structure(self) -> None:
        expected_files = [
            "Dockerfile",
            constants.INFERENCE_SERVER_DIR,
            constants.ENTRYPOINT_SCRIPT,
            "snowflake",
        ]
        self.docker_context.build()
        generated_files = os.listdir(self.context_dir)
        self.assertCountEqual(expected_files, generated_files)

        actual_inference_files = os.listdir(os.path.join(self.context_dir, constants.INFERENCE_SERVER_DIR))
        self.assertCountEqual(["main.py"], actual_inference_files)

        snow_ml_dir = os.path.join(self.context_dir, "snowflake", "ml")
        self.assertTrue(os.path.exists(snow_ml_dir))

        snow_ml_model_dir = os.path.join(self.context_dir, "snowflake", "ml", "model")
        self.assertTrue(os.path.exists(snow_ml_model_dir))

        experimental_dir = os.path.join(self.context_dir, "snowflake", "ml", "experimental")
        self.assertFalse(os.path.exists(experimental_dir))

    def test_docker_file_content(self) -> None:
        self.docker_context.build()
        dockerfile_path = os.path.join(self.context_dir, "Dockerfile")
        dockerfile_fixture_path = os.path.join(os.path.dirname(__file__), "test_fixtures", "dockerfile_test_fixture")
        with open(dockerfile_path, encoding="utf-8") as dockerfile, open(
            dockerfile_fixture_path, encoding="utf-8"
        ) as expected_dockerfile:
            actual = dockerfile.read()
            expected = expected_dockerfile.read()

            # Define a regular expression pattern to match comment lines
            comment_pattern = r"\s*#.*$"
            # Remove comments
            actual = re.sub(comment_pattern, "", actual, flags=re.MULTILINE)
            self.assertEqual(actual, expected, "Generated dockerfile is not aligned with the docker template")


if __name__ == "__main__":
    absltest.main()
