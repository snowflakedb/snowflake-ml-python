import importlib
import os
import shutil
import sys
import tempfile
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

    def test_make_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace:
            fake_mod_dirpath = os.path.join(tmpdir, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            file_utils.make_archive(os.path.join(workspace, "model.zip"), tmpdir)
            file_utils.make_archive(os.path.join(workspace, "model.tar"), tmpdir)
            file_utils.make_archive(os.path.join(workspace, "model.tar.gz"), tmpdir)
            file_utils.make_archive(os.path.join(workspace, "model.tar.bz2"), tmpdir)
            file_utils.make_archive(os.path.join(workspace, "model.tar.xz"), tmpdir)

            self.assertListEqual(
                sorted(os.listdir(workspace)),
                sorted(["model.zip", "model.tar", "model.tar.gz", "model.tar.bz2", "model.tar.xz"]),
            )

    def test_zip_python_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_mod_dirpath = os.path.join(tmpdir, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            sys.path.insert(0, os.path.abspath(tmpdir))
            importlib.import_module("snowflake.fake.fake_module.p")
            zip_module_filename = os.path.join(tmpdir, "fake_module.zip")
            file_utils.zip_python_package(zip_module_filename, "snowflake.fake.fake_module")
            sys.path.remove(os.path.abspath(tmpdir))

            sys.path.insert(0, os.path.abspath(zip_module_filename))
            importlib.import_module("snowflake.fake.fake_module.p")
            sys.path.remove(os.path.abspath(zip_module_filename))

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_mod_dirpath = os.path.join(tmpdir, "snowflake", "fake", "fake_module")
            os.makedirs(fake_mod_dirpath)

            py_file_path = os.path.join(fake_mod_dirpath, "p.py")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            py_file_path = os.path.join(fake_mod_dirpath, "❄️.txt")
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(PY_SRC)

            sys.path.insert(0, os.path.abspath(tmpdir))
            importlib.import_module("snowflake.fake.fake_module.p")
            zip_module_filename = os.path.join(tmpdir, "fake_module.zip")
            with self.assertRaises(ValueError):
                file_utils.zip_python_package(zip_module_filename, "snowflake.fake.fake_module")
            sys.path.remove(os.path.abspath(tmpdir))

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


if __name__ == "__main__":
    absltest.main()
