import os
import tempfile

from absl.testing import absltest

from snowflake.ml._internal import file_utils
from tests.integ.snowflake.ml.test_utils import common_test_base


class FileUtilsIntegTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    @common_test_base.CommonTestBase.sproc_test()
    def test_copytree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write("Hello World")

            file_utils.copy_file_or_tree(leading_path, os.path.join(tmpdir, "my_copy"))

            self.assertListEqual(
                [ele[1:] for ele in os.walk(os.path.join(tmpdir, "my_copy", "test"))],
                [(["snowflake"], []), (["fake"], []), (["fake_module"], []), ([], ["p.py"])],
            )


if __name__ == "__main__":
    absltest.main()
