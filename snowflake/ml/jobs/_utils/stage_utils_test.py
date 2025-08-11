from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import payload_utils

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
    )
    def test_is_relative_to(self, path1: str, path2: str, expected: bool) -> None:
        stagePath1 = payload_utils.resolve_path(path1)
        stagePath2 = payload_utils.resolve_path(path2)
        self.assertEqual(stagePath2.is_relative_to(stagePath1), expected)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src", "@test_stage"),
        ("snow://headless/abc/versions/v9.8.7/dirs", "snow://headless/abc/versions/v9.8.7"),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7"),
        ("@test_stage/", "@test_stage"),
    )
    def test_root(self, path: str, expected: str) -> None:
        self.assertEqual(payload_utils.resolve_path(path).root, expected)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage/src"),
        ("@test_stage/", "@test_stage"),
        ("snow://headless/abc/versions/v9.8.7/dirs/", "snow://headless/abc/versions/v9.8.7/dirs"),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7"),
    )
    def test_absolute(self, path: str, expected_path: str) -> None:
        self.assertEqual(payload_utils.resolve_path(path).absolute().as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage"),
        ("@test_stage/", "@test_stage"),
        ("@test_stage/src/main.py", "@test_stage/src"),
        ("snow://headless/abc/versions/v9.8.7/dirs/", "snow://headless/abc/versions/v9.8.7"),
        ("snow://headless/abc/versions/v9.8.7", "snow://headless/abc/versions/v9.8.7"),
        ("snow://headless/abc/versions/v9.8.7/dirs/main.py", "snow://headless/abc/versions/v9.8.7/dirs"),
    )
    def test_parent(self, path: str, expected_path: str) -> None:
        self.assertEqual(payload_utils.resolve_path(path).parent.as_posix(), expected_path)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/src/", "@test_stage/src"),
        ("@test_stage/main.py", "@test_stage/main.py"),
        ("snow://headless/abc/versions/v9.8.7/dirs/", "snow://headless/abc/versions/v9.8.7/dirs"),
        (
            "snow://headless/abc/versions/v9.8.7/dir1/dir2/secondary.py",
            "snow://headless/abc/versions/v9.8.7/dir1/dir2/secondary.py",
        ),
        ("snow://headless/abc/versions/v9.8.7/dirs/dir", "snow://headless/abc/versions/v9.8.7/dirs/dir"),
    )
    def test_posix(self, path: str, expected_path: str) -> None:
        self.assertEqual(payload_utils.resolve_path(path).as_posix(), expected_path)

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
        self.assertEqual(payload_utils.resolve_path(path1) == payload_utils.resolve_path(path2), expected_result)

    @parameterized.parameters(  # type: ignore[misc]
        ("@test_stage/dir1", ("dir2",), "@test_stage/dir1/dir2"),
        (
            "snow://headless/abc/versions/v9.8.7/dir1",
            ("dir2",),
            "snow://headless/abc/versions/v9.8.7/dir1/dir2",
        ),
        ("@test_stage/src/", ("dir1", "/dir2"), "@test_stage/dir2"),
    )
    def test_joinpath(self, path1: str, paths: tuple[str], expected_path: str) -> None:
        stagePath1 = payload_utils.resolve_path(path1)
        stagePaths = []
        for path in paths:
            stagePaths.append(payload_utils.resolve_path(path))
        self.assertEqual(stagePath1.joinpath(*tuple(stagePaths)).as_posix(), expected_path)


if __name__ == "__main__":
    absltest.main()
