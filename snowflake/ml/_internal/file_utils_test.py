import importlib
import os
import re
import shutil
import sys
import tempfile
import warnings
from datetime import datetime

from absl.testing import absltest

from snowflake.ml._internal import file_utils

PY_SRC = """\
def get_name():
    return __name__
def get_file():
    return __file__
"""


class FileUtilsTest(absltest.TestCase):
    def test_copytree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            file_utils.copy_file_or_tree(leading_path, os.path.join(tmpdir, "my_copy"))
            shutil.copytree(leading_path, os.path.join(tmpdir, "shutil_copy"))

            self.assertListEqual(
                [ele[1:] for ele in os.walk(os.path.join(tmpdir, "my_copy", "test"))],
                [ele[1:] for ele in os.walk(os.path.join(tmpdir, "shutil_copy"))],
            )

    def test_zip_file_or_directory_to_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            zip_module_filename = os.path.join(tmpdir, "fake_module.zip")
            with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
                with open(zip_module_filename, "wb") as f:
                    f.write(input_stream.getbuffer())

            sys.path.insert(0, os.path.abspath(zip_module_filename))

            importlib.import_module("snowflake.fake.fake_module.p")

            mod_path, start_path = file_utils.get_package_path("snowflake.fake.fake_module")
            self.assertEqual(mod_path, os.path.join(zip_module_filename, "snowflake", "fake", "fake_module"))
            self.assertEqual(start_path, zip_module_filename)

            sys.path.remove(os.path.abspath(zip_module_filename))

            with file_utils.zip_file_or_directory_to_stream(fake_mod_dirpath, leading_path) as input_stream:
                with open(zip_module_filename, "wb") as f:
                    f.write(input_stream.getbuffer())

            sys.path.insert(0, os.path.abspath(zip_module_filename))

            importlib.import_module("snowflake.fake.fake_module.p")

            mod_path, start_path = file_utils.get_package_path("snowflake.fake.fake_module")
            self.assertEqual(mod_path, os.path.join(zip_module_filename, "snowflake", "fake", "fake_module"))
            self.assertEqual(start_path, zip_module_filename)

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

            fake_mod_dirpath = os.path.join(leading_path, "❄️", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            with self.assertRaises(ValueError):
                zip_module_filename = os.path.join(tmpdir, "fake_module.zip")
                with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
                    with open(zip_module_filename, "wb") as f:
                        f.write(input_stream.getbuffer())

            py_file_path = os.path.join(fake_mod_dirpath, "❄️.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            with self.assertRaises(ValueError):
                zip_module_filename = os.path.join(tmpdir, "fake_module.zip")
                with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
                    with open(zip_module_filename, "wb") as f:
                        f.write(input_stream.getbuffer())

    def test_unzip_stream_in_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            leading_path = os.path.join(tmpdir, "test")
            fake_mod_dirpath = os.path.join(leading_path, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with file_utils.zip_file_or_directory_to_stream(py_file_path, leading_path) as input_stream:
                    with file_utils.unzip_stream_in_temp_dir(input_stream, temp_root=tmpdir) as sub_tempdir:
                        with open(
                            os.path.join(sub_tempdir, "snowflake", "fake", "fake_module", "p.py"), encoding="utf-8"
                        ) as f:
                            self.assertEqual(f.read(), PY_SRC)

    def test_hash_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snowflake"), "w", encoding="utf-8") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_0 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            hash_1 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snowflake"), "w", encoding="utf-8") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_2 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snowflake"), "w", encoding="utf-8") as f:
                f.write("Hello Taffy!")
                f.flush()
            hash_3 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "test"))
            with open(os.path.join(tmpdir, "test", "snow"), "w", encoding="utf-8") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_4 = file_utils.hash_directory(os.path.join(tmpdir, "test"))
            shutil.rmtree(os.path.join(tmpdir, "test"))

            os.mkdir(os.path.join(tmpdir, "not-a-test"))
            with open(os.path.join(tmpdir, "not-a-test", "snowflake"), "w", encoding="utf-8") as f:
                f.write("Hello Snowflake!")
                f.flush()
            hash_5 = file_utils.hash_directory(os.path.join(tmpdir, "not-a-test"))
            shutil.rmtree(os.path.join(tmpdir, "not-a-test"))

            os.makedirs(os.path.join(tmpdir, "test", "test"))
            with open(os.path.join(tmpdir, "test", "test", "snowflake"), "w", encoding="utf-8") as f:
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

    def test_hash_directory_with_excluded_files(self) -> None:
        def _populate_tmpdir(tmpdir: str) -> None:
            with open(os.path.join(tmpdir, "Dockerfile"), "w", encoding="utf-8") as f:
                f.write("FROM focal-cuda-11.6.2")
                f.flush()

            os.mkdir(os.path.join(tmpdir, "env"))
            with open(os.path.join(tmpdir, "env", "conda.yml"), "w", encoding="utf-8") as f:
                f.write("python==3.8.13")
                f.flush()

            os.mkdir(os.path.join(tmpdir, "server"))
            with open(os.path.join(tmpdir, "server", "main.py"), "w", encoding="utf-8") as f:
                f.write("import os")
                f.flush()

            with open(os.path.join(tmpdir, "model.yaml"), "w", encoding="utf-8") as f:
                f.write(f"creation_timestamp: {datetime.now().time().strftime('%H:%M:%S.%f')}")
                f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            _populate_tmpdir(tmpdir)
            hash_0 = file_utils.hash_directory(tmpdir)
            hash_0_with_exclude = file_utils.hash_directory(tmpdir, excluded_files=["model.yaml"])

        with tempfile.TemporaryDirectory() as tmpdir:
            _populate_tmpdir(tmpdir)
            hash_1 = file_utils.hash_directory(tmpdir)
            hash_1_with_exclude = file_utils.hash_directory(tmpdir, excluded_files=["model.yaml"])

        self.assertNotEqual(hash_0, hash_1)
        self.assertNotEqual(hash_0, hash_0_with_exclude)
        self.assertEqual(hash_0_with_exclude, hash_1_with_exclude)

    def test_hash_directory_with_ignore_hidden_file(self) -> None:
        def _populate_tmpdir(tmpdir: str) -> None:
            with open(os.path.join(tmpdir, "Dockerfile"), "w", encoding="utf-8") as f:
                f.write("FROM focal-cuda-11.6.2")
                f.flush()
            with open(os.path.join(tmpdir, ".DS_Store"), "w", encoding="utf-8") as f:
                f.write(f"creation_timestamp: {datetime.now().time().strftime('%H:%M:%S.%f')}")
                f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            _populate_tmpdir(tmpdir)
            hash_0 = file_utils.hash_directory(tmpdir)
            hash_0_ignore_hidden = file_utils.hash_directory(tmpdir, ignore_hidden=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            _populate_tmpdir(tmpdir)
            hash_1 = file_utils.hash_directory(tmpdir)
            hash_1_ignore_hidden = file_utils.hash_directory(tmpdir, ignore_hidden=True)

        self.assertNotEqual(hash_0, hash_1)
        self.assertNotEqual(hash_0, hash_0_ignore_hidden)
        self.assertEqual(hash_0_ignore_hidden, hash_1_ignore_hidden)

    def test_able_ascii_encode(self) -> None:
        self.assertTrue(file_utils._able_ascii_encode("abc"))
        self.assertFalse(file_utils._able_ascii_encode("❄️"))

    def test_resolve_zip_import_path(self) -> None:
        # Test when snowml is a directory
        snowflake_ml_path = "snowflake/ml/model/_deploy_client/image_builds/inference_server"
        input_path = f"/a/b/c/{snowflake_ml_path}"
        self.assertEqual(file_utils.resolve_zip_import_path(input_path), input_path)

        # Test when snowml is a zip
        snowml_path, snowml_start_path = file_utils.get_package_path("snowflake.ml", strategy="last")
        zip_file_name = "snowml.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            zipped_snowml_path = os.path.join(tmpdir, zip_file_name)
            with open(zipped_snowml_path, "wb") as f:
                with file_utils.zip_file_or_directory_to_stream(snowml_path, snowml_start_path) as zip_stream:
                    f.write(zip_stream.getbuffer())
            try:
                sys.path.append(zipped_snowml_path)
                tmp_parent_dir = os.path.dirname(tmpdir)
                resolved_path = file_utils.resolve_zip_import_path(f"{zipped_snowml_path}/{snowflake_ml_path}")
                self.assertTrue(zip_file_name not in resolved_path)
                # Note that this is based on the tempfile.TemporaryDirectory, in which the directory name might differ
                # between runs. But the suffix/prefix/dir will remain the same. Essentially, the test assumes that if
                # zip file is created at /tmp/a/b/snowml,zip, then the unzipped dir will be at /tmp/a/{some_dir}
                self.assertTrue(re.match(rf"{tmp_parent_dir}/(\w+)/{snowflake_ml_path}", resolved_path))
            finally:
                sys.path.remove(zipped_snowml_path)


if __name__ == "__main__":
    absltest.main()
