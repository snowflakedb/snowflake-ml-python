import unittest
from importlib.machinery import SourceFileLoader

repo_paths_bzl = SourceFileLoader("repo_paths_bzl", "bazel/repo_paths.bzl").load_module()


class CheckForExperimentalDependencies(unittest.TestCase):
    """Testing for check_for_experimental_dependencies."""

    def testNoDeps(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_experimental_dependencies(":demo", {}))

    def testEmptyDeps(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_experimental_dependencies(":demo", {"deps": []}))

    def testAllowedDeps(self) -> None:
        self.assertTrue(
            repo_paths_bzl.check_for_experimental_dependencies(
                "//src/snowflake/ml/experimental/sdas",
                {"name": "demo", "deps": ["//src/snowflake/ml/utils:connection_params"]},
            )
        )

    def testDisallowedDeps(self) -> None:
        self.assertFalse(
            repo_paths_bzl.check_for_experimental_dependencies(
                "//src/snowflake/ml/utils",
                {"name": "demo", "deps": ["//src/snowflake/ml/experimental/sdas:connection_params"]},
            )
        )


class CheckForTestsDependencies(unittest.TestCase):
    """Testing for check_for_experimental_dependencies."""

    def testNoDeps(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_tests_dependencies("//", {"name": "demo"}))

    def testEmptyDeps(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_tests_dependencies("//pkg", {"name": "demo", "deps": []}))

    def testAllowedDeps(self) -> None:
        self.assertTrue(
            repo_paths_bzl.check_for_tests_dependencies(
                "//tests/snowflake/ml/utils",
                {"name": "demo", "deps": ["//src/snowflake/ml/utils:connection_params"]},
            )
        )

    def testDisallowedDeps(self) -> None:
        self.assertFalse(
            repo_paths_bzl.check_for_tests_dependencies(
                "//snowflake/ml/utils",
                {"name": "demo", "deps": ["//tests/snowflake/ml/utils:connection_params"]},
            )
        )

        self.assertFalse(
            repo_paths_bzl.check_for_tests_dependencies(
                "//tests/snowflake/ml/utils",
                {"name": "demo", "deps": ["//tests/snowflake/ml/utils:demo_test"]},
            )
        )


class CheckForTestsNames(unittest.TestCase):
    """Testing for check_for_test_name."""

    def testAllowed(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_test_name("//snowflake/ml/utils", {"name": "demo_test"}))

    def testDisallowed(self) -> None:
        self.assertFalse(repo_paths_bzl.check_for_test_name("//tests/snowflake/ml/utils", {"name": "demo"}))
        self.assertFalse(repo_paths_bzl.check_for_test_name("//snowflake/ml/utils", {"name": "demo"}))


if __name__ == "__main__":
    unittest.main()
