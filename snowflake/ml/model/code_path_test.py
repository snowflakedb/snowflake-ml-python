"""Unit tests for CodePath class."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from snowflake.ml.model.code_path import CodePath


class CodePathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        project = Path(self.temp_dir) / "project"
        (project / "src" / "utils").mkdir(parents=True)
        (project / "src" / "lib" / "helpers").mkdir(parents=True)

        (project / "src" / "__init__.py").touch()
        (project / "src" / "utils" / "__init__.py").touch()
        (project / "src" / "utils" / "helper.py").write_text("def helper(): pass\n")
        (project / "src" / "lib" / "helpers" / "__init__.py").touch()
        (project / "src" / "lib" / "helpers" / "utils.py").write_text("def util(): pass\n")
        (project / "src" / "config.py").write_text("CONFIG = {}\n")
        (project / "README.md").write_text("# Project\n")

        self.project_path = str(project)
        self.src_path = str(project / "src")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_no_filter_directory(self) -> None:
        cp = CodePath(self.src_path)
        source, dest = cp._resolve()

        self.assertTrue(os.path.isabs(source))
        self.assertTrue(source.endswith("src"))
        self.assertEqual(dest, "src")

    def test_no_filter_file(self) -> None:
        readme_path = os.path.join(self.project_path, "README.md")
        cp = CodePath(readme_path)
        source, dest = cp._resolve()

        self.assertTrue(source.endswith("README.md"))
        self.assertEqual(dest, "README.md")

    def test_no_filter_with_trailing_slash(self) -> None:
        cp = CodePath(self.src_path + "/")
        source, dest = cp._resolve()

        self.assertTrue(source.endswith("src"))
        self.assertEqual(dest, "src")

    def test_filter_simple_directory(self) -> None:
        cp = CodePath(self.src_path, filter="utils")
        source, dest = cp._resolve()

        self.assertTrue(source.endswith(os.path.join("src", "utils")))
        self.assertEqual(dest, "utils")

    def test_filter_nested_directory(self) -> None:
        cp = CodePath(self.src_path, filter="lib/helpers")
        source, dest = cp._resolve()

        self.assertTrue(source.endswith(os.path.join("src", "lib", "helpers")))
        self.assertEqual(dest, os.path.join("lib", "helpers"))

    def test_filter_single_file(self) -> None:
        cp = CodePath(self.src_path, filter="config.py")
        source, dest = cp._resolve()

        self.assertTrue(source.endswith(os.path.join("src", "config.py")))
        self.assertEqual(dest, "config.py")

    def test_filter_nested_file(self) -> None:
        cp = CodePath(self.src_path, filter="utils/helper.py")
        source, dest = cp._resolve()

        self.assertTrue(source.endswith(os.path.join("src", "utils", "helper.py")))
        self.assertEqual(dest, os.path.join("utils", "helper.py"))

    def test_filter_trailing_slash_normalized(self) -> None:
        cp1 = CodePath(self.src_path, filter="utils")
        cp2 = CodePath(self.src_path, filter="utils/")

        self.assertEqual(cp1._resolve(), cp2._resolve())

    def test_empty_filter_treated_as_none(self) -> None:
        cp1 = CodePath(self.src_path, filter=None)
        cp2 = CodePath(self.src_path, filter="")

        self.assertEqual(cp1._resolve(), cp2._resolve())

    def test_filter_escapes_root_error(self) -> None:
        cp = CodePath(self.src_path, filter="../README.md")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("escapes root directory", str(ctx.exception))

    def test_filter_on_file_root_error(self) -> None:
        readme_path = os.path.join(self.project_path, "README.md")
        cp = CodePath(readme_path, filter="something")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("cannot apply filter to a file root", str(ctx.exception))

    def test_nonexistent_root_error(self) -> None:
        cp = CodePath(os.path.join(self.project_path, "nonexistent"))

        with self.assertRaises(FileNotFoundError) as ctx:
            cp._resolve()

        self.assertIn("does not exist", str(ctx.exception))

    def test_nonexistent_filter_error(self) -> None:
        cp = CodePath(self.src_path, filter="nonexistent")

        with self.assertRaises(FileNotFoundError) as ctx:
            cp._resolve()

        self.assertIn("does not exist", str(ctx.exception))

    def test_absolute_filter_error(self) -> None:
        cp = CodePath(self.src_path, filter="/absolute/path")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("must be a relative path", str(ctx.exception))

    def test_wildcard_star_rejected(self) -> None:
        cp = CodePath(self.src_path, filter="*.py")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("Wildcards are not supported", str(ctx.exception))

    def test_wildcard_question_rejected(self) -> None:
        cp = CodePath(self.src_path, filter="config?.py")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("Wildcards are not supported", str(ctx.exception))

    def test_wildcard_glob_star_rejected(self) -> None:
        cp = CodePath(self.src_path, filter="**/*.py")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("Wildcards are not supported", str(ctx.exception))

    def test_home_directory_filter_rejected(self) -> None:
        cp = CodePath(self.src_path, filter="~/some/path")

        with self.assertRaises(ValueError) as ctx:
            cp._resolve()

        self.assertIn("home directory path", str(ctx.exception))

    def test_frozen_dataclass(self) -> None:
        cp = CodePath(self.src_path)

        with self.assertRaises(AttributeError):
            cp.root = "other/path"  # type: ignore[misc]

    def test_equality(self) -> None:
        cp1 = CodePath(self.src_path, filter="utils")
        cp2 = CodePath(self.src_path, filter="utils")
        cp3 = CodePath(self.src_path, filter="lib")

        self.assertEqual(cp1, cp2)
        self.assertNotEqual(cp1, cp3)

    def test_hash(self) -> None:
        cp1 = CodePath(self.src_path, filter="utils")
        cp2 = CodePath(self.src_path, filter="utils")

        code_paths_set = {cp1, cp2}
        self.assertEqual(len(code_paths_set), 1)

    def test_repr_without_filter(self) -> None:
        cp = CodePath("project/src")
        self.assertEqual(repr(cp), "CodePath('project/src')")

    def test_repr_with_filter(self) -> None:
        cp = CodePath("project/src", filter="utils")
        self.assertEqual(repr(cp), "CodePath('project/src', filter='utils')")


class CodePathEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        project = Path(self.temp_dir) / "project"
        (project / "src" / "utils").mkdir(parents=True)
        (project / "src" / "utils" / "helper.py").touch()

        self.project_path = str(project)
        self.src_path = str(project / "src")
        self.utils_path = str(project / "src" / "utils")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_string_and_codepath_equivalence(self) -> None:
        cp = CodePath(self.utils_path)
        _, dest = cp._resolve()

        self.assertEqual(dest, "utils")

    def test_codepath_with_filter_matches_direct_path(self) -> None:
        cp_direct = CodePath(self.utils_path)
        cp_filtered = CodePath(self.src_path, filter="utils")

        source_direct, dest_direct = cp_direct._resolve()
        source_filtered, dest_filtered = cp_filtered._resolve()

        self.assertEqual(source_direct, source_filtered)
        self.assertEqual(dest_direct, dest_filtered)


if __name__ == "__main__":
    unittest.main()
