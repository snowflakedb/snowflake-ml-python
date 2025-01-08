import os
import pathlib
import tempfile

from absl.testing import absltest

from snowflake.ml.model._model_composer.model_user_file import model_user_file


class ModelUserFileTest(absltest.TestCase):
    def test_user_files(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:

            filename = "file1"
            with open(os.path.join(tmpdir, filename), "w") as f:
                path = os.path.abspath(f.name)
                f.write("user file")

            subdir = "subdir/x/y/"

            res = model_user_file.ModelUserFile(pathlib.PurePosixPath(subdir), pathlib.Path(path)).save(
                pathlib.Path(workspace)
            )
            target_relative_path = os.path.join(subdir, filename)

            self.assertEqual(res, target_relative_path)

            target_file = os.path.join(
                workspace, model_user_file.ModelUserFile.USER_FILES_DIR_REL_PATH, target_relative_path
            )
            self.assertTrue(os.path.exists(target_file))


if __name__ == "__main__":
    absltest.main()
