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
                "//src/snowflake/ml/experimental/sdas:demo",
                {"deps": ["//src/snowflake/ml/utils:connection_params"]},
            )
        )

    def testDisallowedDeps(self) -> None:
        self.assertFalse(
            repo_paths_bzl.check_for_experimental_dependencies(
                "//src/snowflake/ml/utils:demo",
                {"deps": ["//src/snowflake/ml/experimental/sdas:connection_params"]},
            )
        )


class CheckForTestsDependencies(unittest.TestCase):
    """Testing for check_for_experimental_dependencies."""

    def testNoDeps(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_tests_dependencies(":demo", {}))

    def testEmptyDeps(self) -> None:
        self.assertTrue(repo_paths_bzl.check_for_tests_dependencies(":demo", {"deps": []}))

    def testAllowedDeps(self) -> None:
        self.assertTrue(
            repo_paths_bzl.check_for_tests_dependencies(
                "//tests/snowflake/ml/utils:demo",
                {"deps": ["//src/snowflake/ml/utils:connection_params"]},
            )
        )

    def testDisallowedDeps(self) -> None:
        self.assertFalse(
            repo_paths_bzl.check_for_tests_dependencies(
                "//src/snowflake/ml/utils:demo",
                {"deps": ["//tests/snowflake/ml/utils:connection_params"]},
            )
        )


if __name__ == "__main__":
    unittest.main()
