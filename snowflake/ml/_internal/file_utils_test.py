import importlib
import os
import shutil
import sys
import tempfile
import warnings
import zipimport
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal import file_utils
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import session

PY_SRC = """\
def get_name():
    return __name__
def get_file():
    return __file__
"""


class FileUtilsTest(absltest.TestCase):
    def test_zip_file_or_directory_to_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w") as f:
                f.write(PY_SRC)

            zip_module_filename = os.path.join(tmpdir, "fake_module.zip")
            with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
                with open(zip_module_filename, "wb") as f:
                    f.write(input_stream.getbuffer())

            sys.path.insert(0, os.path.abspath(zip_module_filename))

            importlib.import_module("snowflake.fake.fake_module.p")

            sys.path.remove(os.path.abspath(zip_module_filename))

            with file_utils.zip_file_or_directory_to_stream(fake_mod_dirpath, leading_path) as input_stream:
                with open(zip_module_filename, "wb") as f:
                    f.write(input_stream.getbuffer())

            sys.path.insert(0, os.path.abspath(zip_module_filename))

            importlib.import_module("snowflake.fake.fake_module.p")

            sys.path.remove(os.path.abspath(zip_module_filename))

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with file_utils.zip_file_or_directory_to_stream(fake_mod_dirpath, fake_mod_dirpath) as input_stream:
                    with open(zip_module_filename, "wb") as f:
                        f.write(input_stream.getbuffer())

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with file_utils.zip_file_or_directory_to_stream(leading_path, leading_path) as input_stream:
                    with open(zip_module_filename, "wb") as f:
                        f.write(input_stream.getbuffer())

    def test_unzip_stream_in_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w") as f:
                f.write(PY_SRC)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
                    with file_utils.unzip_stream_in_temp_dir(input_stream, temp_root=tmpdir) as sub_tempdir:
                        with open(os.path.join(sub_tempdir, "snowflake", "fake", "fake_module", "p.py")) as f:
                            self.assertEqual(f.read(), PY_SRC)

    def test_zip_snowml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_module_filename = os.path.join(tmpdir, "snowml.zip")
            with file_utils.zip_snowml() as (input_stream, _):
                with open(zip_module_filename, "wb") as f:
                    f.write(input_stream.getbuffer())
            sys.path.insert(0, os.path.abspath(zip_module_filename))

            mod = importlib.import_module("snowflake.ml._internal.file_utils_test")
            self.assertIsInstance(mod.__loader__, zipimport.zipimporter)

            sys.path.remove(os.path.abspath(zip_module_filename))

    def test_hash_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snowflake"), "w") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_0 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            hash_1 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snowflake"), "w") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_2 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snowflake"), "w") as f:
                f.write("Hello Taffy!")
                f.flush()
            hash_3 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snow"), "w") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_4 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "not-a-test"))
            with open(os.path.join(tmpdir, "not-a-test", "snowflake"), "w") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_5 = file_utils.hash_directory(os.path.join(tmpdir, "not-a-test"))
            shutil.rmtree(os.path.join(tmpdir, "not-a-test"))

            os.makedirs(os.path.join(tmpdir, "test", "test"))
            with open(os.path.join(tmpdir, "test", "test", "snowflake"), "w") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_6 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

        self.assertEqual(hash_0, hash_1)
        self.assertEqual(hash_0, hash_2)
        self.assertNotEqual(hash_0, hash_3)
        self.assertNotEqual(hash_0, hash_4)
        self.assertEqual(hash_0, hash_5)
        self.assertNotEqual(hash_0, hash_6)

    _MOCK_SHA1_RESULT = "10757e7c6da427f7eca0646fcd2a3883"

    def test_upload_snowml_session_stage(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(session.Session, m_session)
        with mock.patch.object(
            c_session, "get_session_stage", create=True, return_value="@mock_session_stage"
        ) as mock_session_stage:
            with mock.patch.object(c_session, "file", create=True) as mock_file_object:
                with mock.patch.object(mock_file_object, "put_stream") as mock_put_stream:
                    with mock.patch.object(
                        file_utils, "hash_directory", return_value=FileUtilsTest._MOCK_SHA1_RESULT
                    ) as mock_hash_directory:
                        file_location = file_utils.upload_snowml(c_session)
                        mock_session_stage.assert_called_once_with()
                        mock_hash_directory.assert_called_once()
                        mock_put_stream.assert_called_once_with(
                            mock.ANY,
                            stage_location=f"@mock_session_stage/snowml_{FileUtilsTest._MOCK_SHA1_RESULT}.zip",
                            auto_compress=False,
                            overwrite=False,
                        )
                        self.assertEqual(
                            file_location, f"@mock_session_stage/snowml_{FileUtilsTest._MOCK_SHA1_RESULT}.zip"
                        )

    def test_upload_snowml_provided_stage(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(session.Session, m_session)
        with mock.patch.object(
            c_session, "get_session_stage", create=True, return_value="@mock_session_stage"
        ) as mock_session_stage:
            with mock.patch.object(c_session, "file", create=True) as mock_file_object:
                with mock.patch.object(mock_file_object, "put_stream") as mock_put_stream:
                    with mock.patch.object(
                        file_utils, "hash_directory", return_value=FileUtilsTest._MOCK_SHA1_RESULT
                    ) as mock_hash_directory:
                        file_location = file_utils.upload_snowml(c_session, stage_location="@mystage")
                        mock_session_stage.assert_not_called()
                        mock_hash_directory.assert_called_once()
                        mock_put_stream.assert_called_once_with(
                            mock.ANY,
                            stage_location=f"@mystage/snowml_{FileUtilsTest._MOCK_SHA1_RESULT}.zip",
                            auto_compress=False,
                            overwrite=False,
                        )
                        self.assertEqual(file_location, f"@mystage/snowml_{FileUtilsTest._MOCK_SHA1_RESULT}.zip")


if __name__ == "__main__":
    absltest.main()
