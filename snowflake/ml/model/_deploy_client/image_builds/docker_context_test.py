import os
import re
import shutil
import tempfile
from unittest import mock

import sklearn.base
import sklearn.datasets as datasets
from absl.testing import absltest
from sklearn import neighbors

from snowflake.ml.model._deploy_client.image_builds import docker_context
from snowflake.ml.model._deploy_client.utils import constants
from snowflake.ml.model._packager import model_packager
from snowflake.snowpark import FileOperation, GetResult, Session

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

        self.packager = model_packager.ModelPackager(self.model_dir)
        self.packager.save(
            name="model",
            model=_get_sklearn_model(),
            sample_input_data=_IRIS_X,
        )
        assert self.packager.meta
        self.model_meta = self.packager.meta

        self.docker_context = docker_context.DockerContext(self.context_dir, model_meta=self.model_meta)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.context_dir)

    def test_build_results_in_correct_docker_context_file_structure(self) -> None:
        expected_files = [
            "Dockerfile",
            constants.INFERENCE_SERVER_DIR,
            constants.ENTRYPOINT_SCRIPT,
            "runtimes",
            "env",
            "model.yaml",
        ]
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


class DockerContextTestCuda(absltest.TestCase):
    def setUp(self) -> None:
        self.context_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()

        self.packager = model_packager.ModelPackager(self.model_dir)
        self.packager.save(
            name="model",
            model=_get_sklearn_model(),
            sample_input_data=_IRIS_X,
        )
        assert self.packager.meta
        self.model_meta = self.packager.meta

        self.model_meta.env.cuda_version = "11.7.1"

        self.docker_context = docker_context.DockerContext(self.context_dir, model_meta=self.model_meta)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.context_dir)

    def test_build_results_in_correct_docker_context_file_structure(self) -> None:
        expected_files = [
            "Dockerfile",
            constants.INFERENCE_SERVER_DIR,
            constants.ENTRYPOINT_SCRIPT,
            "env",
            "runtimes",
            "model.yaml",
        ]
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
        dockerfile_fixture_path = os.path.join(
            os.path.dirname(__file__), "test_fixtures", "dockerfile_test_fixture_with_CUDA"
        )
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


class DockerContextTestModelWeights(absltest.TestCase):
    def setUp(self) -> None:
        self.context_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()

        self.packager = model_packager.ModelPackager(self.model_dir)
        self.packager.save(
            name="model",
            model=_get_sklearn_model(),
            sample_input_data=_IRIS_X,
        )
        assert self.packager.meta
        self.model_meta = self.packager.meta

        self.model_meta.env.cuda_version = "11.7.1"

        self.mock_session = absltest.mock.MagicMock(spec=Session)
        self.model_zip_stage_path = "@model_repo/model.zip"

        self.docker_context = docker_context.DockerContext(
            self.context_dir,
            model_meta=self.model_meta,
            session=self.mock_session,
            model_zip_stage_path=self.model_zip_stage_path,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.context_dir)

    def test_build_results_in_correct_docker_context_file_structure(self) -> None:
        get_results = [GetResult(file="/tmp/model.zip", size="1", status="yes", message="hi")]
        with mock.patch.object(FileOperation, "get", return_value=get_results):
            expected_files = [
                "Dockerfile",
                constants.INFERENCE_SERVER_DIR,
                constants.ENTRYPOINT_SCRIPT,
                "env",
                "runtimes",
                "model.yaml",
            ]
            self.docker_context.build()
            generated_files = os.listdir(self.context_dir)
            self.assertCountEqual(expected_files, generated_files)

            actual_inference_files = os.listdir(os.path.join(self.context_dir, constants.INFERENCE_SERVER_DIR))
            self.assertCountEqual(["main.py"], actual_inference_files)

            model_env_dir = os.path.join(self.context_dir, "env")
            self.assertTrue(os.path.exists(model_env_dir))

    def test_docker_file_content(self) -> None:
        get_results = [GetResult(file="/tmp/model.zip", size="1", status="yes", message="hi")]
        with mock.patch.object(FileOperation, "get", return_value=get_results):
            self.docker_context.build()
            dockerfile_path = os.path.join(self.context_dir, "Dockerfile")
            dockerfile_fixture_path = os.path.join(
                os.path.dirname(__file__), "test_fixtures", "dockerfile_test_fixture_with_model"
            )
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
