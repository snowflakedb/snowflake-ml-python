import importlib
import os
import sys
import tempfile

from absl.testing import absltest, parameterized

from snowflake.ml._internal import file_utils
from tests.integ.snowflake.ml.test_utils import common_test_base


class FileUtilsIntegTest(common_test_base.CommonTestBase):
    @common_test_base.CommonTestBase.sproc_test()
    @parameterized.parameters({"content": "hello"}, {"content": "snowflake"})  # type: ignore[misc]
    def test_copytree(self, content: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            file_utils.copy_file_or_tree(leading_path, os.path.join(tmpdir, "my_copy"))

            self.assertListEqual(
                [ele[1:] for ele in os.walk(os.path.join(tmpdir, "my_copy", "test"))],
                [(["snowflake"], []), (["fake"], []), (["fake_module"], []), ([], ["p.py"])],
            )

    @common_test_base.CommonTestBase.sproc_test()
    def test_zip_python_package_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_module_filename = os.path.join(tmpdir, "snowml.zip")
            file_utils.zip_python_package(zip_module_filename, "snowflake.ml")
            sys.path.insert(0, os.path.abspath(zip_module_filename))

            mod = importlib.reload(file_utils)

            self.assertIn(zip_module_filename, mod.__file__)
            sys.path.remove(zip_module_filename)
            mod = importlib.reload(file_utils)

    @common_test_base.CommonTestBase.sproc_test()
    def test_zip_python_package_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_module_filename = os.path.join(tmpdir, "snowml.zip")
            file_utils.zip_python_package(zip_module_filename, "snowflake.ml._internal")
            sys.path.insert(0, os.path.abspath(zip_module_filename))

            mod = importlib.reload(file_utils)

            self.assertIn(zip_module_filename, mod.__file__)
            sys.path.remove(zip_module_filename)
            mod = importlib.reload(file_utils)


if __name__ == "__main__":
    absltest.main()
