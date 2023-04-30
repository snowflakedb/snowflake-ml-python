import importlib
import os
import sys

from absl.testing import absltest

from snowflake.ml._internal import file_utils

PY_SRC = """\
def get_name():
    return __name__
def get_file():
    return __file__
"""


class UtilsTest(absltest.TestCase):
    def test_zip_file_or_directory_to_stream(self) -> None:
        tmpdir = self.create_tempdir()
        leading_path = os.path.join(tmpdir.full_path, "test")
        fake_mod_dirpath = os.path.join(leading_path, "snowflake", "snowpark", "fake_module")
        os.makedirs(fake_mod_dirpath)

        py_file_path = os.path.join(fake_mod_dirpath, "p.py")
        with open(py_file_path, "w") as f:
            f.write(PY_SRC)

        zip_module_filename = os.path.join(tmpdir.full_path, "fake_module.zip")
        with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
            with open(zip_module_filename, "wb") as f:
                f.write(input_stream.getbuffer())

        sys.path.insert(0, os.path.abspath(zip_module_filename))

        importlib.import_module("snowflake.snowpark.fake_module.p")

        sys.path.remove(os.path.abspath(zip_module_filename))

        with file_utils.zip_file_or_directory_to_stream(fake_mod_dirpath, leading_path) as input_stream:
            with open(zip_module_filename, "wb") as f:
                f.write(input_stream.getbuffer())

        sys.path.insert(0, os.path.abspath(zip_module_filename))

        importlib.import_module("snowflake.snowpark.fake_module.p")

        sys.path.remove(os.path.abspath(zip_module_filename))
