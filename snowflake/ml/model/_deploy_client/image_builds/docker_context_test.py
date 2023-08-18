import os
import re
import shutil
import tempfile

import sklearn.base
import sklearn.datasets as datasets
from absl.testing import absltest
from sklearn import neighbors

from snowflake.ml.model import _model as model_api
from snowflake.ml.model._deploy_client.image_builds import docker_context
from snowflake.ml.model._deploy_client.utils import constants

_IRIS = datasets.load_iris(as_frame=True)
_IRIS_X = _IRIS.data
_IRIS_Y = _IRIS.target


def _get_sklearn_model() -> "sklearn.base.BaseEstimator":
    knn_model = neighbors.KNeighborsClassifier()
    knn_model.fit(_IRIS_X, _IRIS_Y)
    return knn_model


class DockerContextTest(absltest.TestCase):
    def setUp(self) -> None:
        self.context_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()

        model_api.save_model(
            name="model",
            model_dir_path=self.model_dir,
            model=_get_sklearn_model(),
            sample_input=_IRIS_X,
        )

        self.docker_context = docker_context.DockerContext(self.context_dir, model_dir=self.model_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.context_dir)

    def test_build_results_in_correct_docker_context_file_structure(self) -> None:
        expected_files = ["Dockerfile", constants.INFERENCE_SERVER_DIR, constants.ENTRYPOINT_SCRIPT, "env"]
        self.docker_context.build()
        generated_files = os.listdir(self.context_dir)
        self.assertCountEqual(expected_files, generated_files)

        actual_inference_files = os.listdir(os.path.join(self.context_dir, constants.INFERENCE_SERVER_DIR))
        self.assertCountEqual(["main.py"], actual_inference_files)

        model_env_dir = os.path.join(self.context_dir, "env")
        self.assertTrue(os.path.exists(model_env_dir))

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
