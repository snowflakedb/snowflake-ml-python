import os
import shutil
import tempfile

from absl.testing import absltest

from snowflake.ml.model._deploy_client.image_builds.docker_context import DockerContext


class DockerContextTest(absltest.TestCase):
    def setUp(self) -> None:
        self.context_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()
        self.use_gpu = False
        self.docker_context = DockerContext(self.context_dir, model_dir=self.model_dir, use_gpu=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.context_dir)

    def test_build(self) -> None:
        expected_files = [os.path.basename(self.model_dir), "Dockerfile", "inference_server"]
        self.docker_context.build()
        generated_files = os.listdir(self.context_dir)
        self.assertCountEqual(expected_files, generated_files)

        actual_inference_files = os.listdir(os.path.join(self.context_dir, "inference_server"))
        self.assertCountEqual(["main.py"], actual_inference_files)


if __name__ == "__main__":
    absltest.main()
