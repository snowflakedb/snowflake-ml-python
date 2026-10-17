import pathlib
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import stage_utils

"""
StagePath inherits the PurePosixPath
this unit test is to make sure the normal and high frequent functions work as expected
"""


class StageUtilsTests(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src", "@test_stage/src/main.py", True),
        ("@test_stage/src/", "@test_stage/src/main.py", True),
        ("@test_stage/src/dir", "@test_stage/src/main.py", False),
        ("@test_stage/src/dir1/", "@test_stage/src/main.py", False),
        ("@test_stage/src///dir", "@test_stage/src///dir/main.py", True),
        ("@test_stage/src///dir//", "@test_stage/src///dir/main.py", True),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7/main.py", True),
        ("snow://headless/abc/versions/v9.8.7/dirs", "snow://headless/abc/versions/v9.8.7/main.py", False),
        ("snow://headless/abc/versions/v9.8.7///dirs", "snow://headless/abc/versions/v9.8.7/main.py", False),
        ('@repo/branches/"feature/configure-cicd"/src', '@repo/branches/"feature/configure-cicd"/src/main.py', True),
        ('@repo/branches/"feature/configure-cicd"/src', "@repo/branches/feature/configure-cicd/src/main.py", False),
    )
    def test_is_relative_to(self, path1: str, path2: str, expected: bool) -> None:
        stagePath1 = stage_utils.resolve_path(path1)
        stagePath2 = stage_utils.resolve_path(path2)
        self.assertEqual(stagePath2.is_relative_to(stagePath1), expected)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src", "@test_stage"),
        ("snow://headless/abc/versions/v9.8.7/dirs", "snow://headless/abc/versions/v9.8.7"),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7"),
        ("@test_stage/", "@test_stage"),
        ('@repo/branches/"feature/configure-cicd"/src', "@repo"),
    )
    def test_root(self, path: str, expected: str) -> None:
        self.assertEqual(stage_utils.resolve_path(path).root, expected)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage/src"),
        ("@test_stage/", "@test_stage"),
        ("snow://headless/abc/versions/v9.8.7/dirs/", "snow://headless/abc/versions/v9.8.7/dirs"),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7"),
    )
    def test_absolute(self, path: str, expected_path: str) -> None:
        self.assertEqual(stage_utils.resolve_path(path).absolute().as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage"),
        ("@test_stage/", "@test_stage"),
        ("@test_stage/src/main.py", "@test_stage/src"),
        ("snow://headless/abc/versions/v9.8.7/dirs/", "snow://headless/abc/versions/v9.8.7"),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7"),
        ("snow://headless/abc/versions/v9.8.7/dirs/main.py", "snow://headless/abc/versions/v9.8.7/dirs"),
        ('@repo/branches/"feature/configure-cicd"/src/main.py', '@repo/branches/"feature/configure-cicd"/src'),
    )
    def test_parent(self, path: str, expected_path: str) -> None:
        self.assertEqual(stage_utils.resolve_path(path).parent.as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage/src"),
        ("@test_stage/main.py", "@test_stage/main.py"),
        ("snow://headless/abc/versions/v9.8.7/dirs/", "snow://headless/abc/versions/v9.8.7/dirs"),
        (
            "snow://headless/abc/versions/v9.8.7/dir1/dir2/secondary.py",
            "snow://headless/abc/versions/v9.8.7/dir1/dir2/secondary.py",
        ),
        ("snow://headless/abc/versions/v9.8.7/dirs/dir", "snow://headless/abc/versions/v9.8.7/dirs/dir"),
        ('@repo/branches/"feature/configure-cicd"/src/', '@repo/branches/"feature/configure-cicd"/src'),
    )
    def test_posix(self, path: str, expected_path: str) -> None:
        self.assertEqual(stage_utils.resolve_path(path).as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage/src", True),
        ("@test_stage/main.py", "@test_stage/secondary.py", False),
        ("snow://headless/abc/versions/v9.8.7/src", "@test_stage/src/", False),
        (
            "snow://headless/abc/versions/v9.8.7/dir1/dir2/secondary.py",
            "snow://headless/abc/versions/v9.8.7/dir1/dir2/secondary.py",
            True,
        ),
    )
    def test_equal(self, path1: str, path2: str, expected_result: bool) -> None:
        self.assertEqual(stage_utils.resolve_path(path1) == stage_utils.resolve_path(path2), expected_result)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/dir1", ("dir2",), "@test_stage/dir1/dir2"),
        (
            "snow://headless/abc/versions/v9.8.7/dir1",
            ("dir2",),
            "snow://headless/abc/versions/v9.8.7/dir1/dir2",
        ),
        ("@test_stage/src/", ("dir1", "/dir2"), "@test_stage/dir2"),
        ('@repo/branches/"feature/configure-cicd"', ("src",), '@repo/branches/"feature/configure-cicd"/src'),
    )
    def test_joinpath(self, path1: str, paths: tuple[str], expected_path: str) -> None:
        stagePath1 = stage_utils.resolve_path(path1)
        stagePaths = []
        for path in paths:
            stagePaths.append(stage_utils.resolve_path(path))
        self.assertEqual(stagePath1.joinpath(*tuple(stagePaths)).as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/utils/main.py", "@test_stage/src", "utils/main.py"),
        (
            '@repo/branches/"feature/configure-cicd"/src/utils/main.py',
            '@repo/branches/"feature/configure-cicd"/src',
            "utils/main.py",
        ),
    )
    def test_relative_to(self, path: str, other: str, expected_path: str) -> None:
        self.assertEqual(stage_utils.resolve_path(path).relative_to(other).as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ('@repo/branches/"feature/configure-cicd"/src/main.py', stage_utils.StagePath),
        ('@repo/branches/"feature/configure-cicd/src/main.py', pathlib.Path),
        ("@repo/branches/'feature/configure-cicd'/src/main.py", pathlib.Path),
        ("@test_stage/my dir/main.py", pathlib.Path),
    )
    def test_resolve_path_type(self, path: str, expected_type: type) -> None:
        self.assertIsInstance(stage_utils.resolve_path(path), expected_type)

    def test_as_posix_normalizes_windows_separators(self) -> None:
        # Simulate Windows semantics even when running tests on non-Windows.
        with mock.patch.object(stage_utils, "Path", pathlib.PureWindowsPath):
            p = stage_utils.resolve_path("@test_stage/dir1/dir2")
            self.assertEqual(p.as_posix(), "@test_stage/dir1/dir2")
            self.assertNotIn("\\", p.as_posix())
            self.assertEqual(p.parent.as_posix(), "@test_stage/dir1")

    def test_build_versioned_stage_path(self) -> None:
        self.assertEqual(
            stage_utils.build_versioned_stage_path(
                namespace="notebook_project",
                stage_fqn="HEADLESS_STARTER_DB.HEADLESS_STARTER_SCHEMA.CB_ML_VERSIONED",
                version="VERSION$3",
            ),
            "snow://notebook_project/HEADLESS_STARTER_DB.HEADLESS_STARTER_SCHEMA.CB_ML_VERSIONED/versions/VERSION$3",
        )

    @parameterized.parameters(  # type: ignore[misc]
        (None, "DB.SCHEMA.NAME", "VERSION$1"),
        ("notebook_project", None, "VERSION$1"),
        ("notebook_project", "DB.SCHEMA.NAME", None),
        ("", "DB.SCHEMA.NAME", "VERSION$1"),
        ("notebook_project", "not a name", "VERSION$1"),
    )
    def test_try_build_versioned_stage_path_missing_or_invalid(
        self, namespace: str | None, stage_fqn: str | None, version: str | None
    ) -> None:
        self.assertIsNone(
            stage_utils.try_build_versioned_stage_path(namespace=namespace, stage_fqn=stage_fqn, version=version)
        )


if __name__ == "__main__":
    absltest.main()
