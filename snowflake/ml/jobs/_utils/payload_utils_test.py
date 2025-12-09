import contextlib
import functools
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import payload_utils, stage_utils, types
from snowflake.ml.jobs._utils.payload_utils_test_helper import dummy_function
from snowflake.ml.jobs._utils.test_file_helper import resolve_path

_CONSTANT_VALUE = "hello world"


def function_with_pos_arg(a, b: int):  # type: ignore[no-untyped-def]
    print(a, b + 1)  # noqa: T201: we need to print here.


def function_with_pos_arg_free_vars(a: str, b: int) -> None:
    print(a, b + 1, _CONSTANT_VALUE)  # noqa: T201: we need to print here.


def function_with_var_args(*args: str) -> None:
    print(args)  # noqa: T201: we need to print here.


def function_with_pos_arg_modules(a: str, b: int) -> None:
    print(a, b + 1, dummy_function())  # noqa: T201: we need to print here.


def function_with_opt_arg(a: str, b: int, c: float = 0.0, d: Optional[int] = None) -> None:
    print(a, b + 1, c * 2, d)  # noqa: T201: we need to print here.


def function_with_kw_arg(a: str, b: int, c: float = 0.0, *, named_arg: str, opt_named: str = "undefined") -> None:
    print(a, b + 1, c * 2, named_arg, "optional: " + opt_named)  # noqa: T201: we need to print here.


def function_with_unpacking_kw_arg(a: str, b: int, c: float = 0.0, *, named_arg: str, **kwargs: Any) -> None:
    print(a, b + 1, c * 2, named_arg, kwargs)  # noqa: T201: we need to print here.


def function_with_bool_args(a: bool, b: bool) -> None:
    print(a, b)  # noqa: T201: we need to print here.


def function_with_collection_args(a: list, b: dict, c: tuple) -> None:  # type: ignore[type-arg]
    print(a, b, c)  # noqa: T201: we need to print here.


def function_with_typing_collection_args(a: list[str], b: dict[str, int], c: tuple[int, int]) -> None:
    print(a, b, c)  # noqa: T201: we need to print here.


def function_with_object_arg(obj: object) -> None:
    print(obj)  # noqa: T201: we need to print here.


def function_with_any_arg(obj: Any) -> None:
    print(obj)  # noqa: T201: we need to print here.


@contextlib.contextmanager
def pushd(new_dir: str) -> Generator[None, None, None]:
    """Context manager to emulate pushd/popd behavior."""
    # Save the current working directory
    original_dir = os.getcwd()
    try:
        # Change to the new directory
        os.chdir(new_dir)
        yield  # Allow the test code to run inside the context
    finally:
        # Ensure we return to the original directory after the test
        os.chdir(original_dir)


class PayloadUtilsTests(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        ("file1.py", None, resolve_path("file1.py")),
        (resolve_path("file1.py"), None, resolve_path("file1.py")),
        (".", "file1.py", resolve_path("file1.py")),
        (resolve_path(""), "file1.py", resolve_path("file1.py")),
        ("src", "file2.py", resolve_path("src/file2.py")),
        ("src", "subdir1/file3.py", resolve_path("src/subdir1/file3.py")),
        ("src", resolve_path("src/subdir1/file3.py"), resolve_path("src/subdir1/file3.py")),
        ("src", resolve_path("src/src/subdir1/file3.py"), resolve_path("src/src/subdir1/file3.py")),
        ("src", "src/subdir1/file3.py", resolve_path("src/subdir1/file3.py")),  # Prefer more direct match
        ("src", resolve_path("src/src/subdir1/file5.py"), resolve_path("src/src/subdir1/file5.py")),
        ("src", "src/subdir1/file5.py", resolve_path("src/src/subdir1/file5.py")),
        (resolve_path("src"), "subdir1/file3.py", resolve_path("src/subdir1/file3.py")),
        (resolve_path("src"), "src/subdir1/file3.py", resolve_path("src/subdir1/file3.py")),
        (resolve_path("src"), resolve_path("src/subdir1/file3.py"), resolve_path("src/subdir1/file3.py")),
        ("@test_stage/src", "@test_stage/src/main.py", "@test_stage/src/main.py"),
        ("@test_stage/", "@test_stage/main.py", "@test_stage/main.py"),
        ("@test_stage/main.py", None, "@test_stage/main.py"),
        ("@test_stage/main.py", "@test_stage/main.py", "@test_stage/main.py"),
        ("@test_stage/src/dir", "@test_stage/src/dir/dir1/main.py", "@test_stage/src/dir/dir1/main.py"),
        ("@test_stage/src/dir/", "dir1/main.py", "@test_stage/src/dir/dir1/main.py"),
        ("snow://headless/abc/versions/v9.8.7/main.py", None, "snow://headless/abc/versions/v9.8.7/main.py"),
        (
            "snow://headless/abc/versions/v9.8.7/main.py",
            "snow://headless/abc/versions/v9.8.7/main.py",
            "snow://headless/abc/versions/v9.8.7/main.py",
        ),
        (
            "snow://headless/abc/versions/v9.8.7",
            "snow://headless/abc/versions/v9.8.7/main.py",
            "snow://headless/abc/versions/v9.8.7/main.py",
        ),
        (
            "snow://headless/abc/versions/v9.8.7/src",
            "snow://headless/abc/versions/v9.8.7/src/main.py",
            "snow://headless/abc/versions/v9.8.7/src/main.py",
        ),
        (
            "snow://headless/abc/versions/v9.8.7/src",
            "main.py",
            "snow://headless/abc/versions/v9.8.7/src/main.py",
        ),
    )
    def test_payload_validate(self, source: str, entrypoint: Optional[str], expected_entrypoint: str) -> None:
        with pushd(resolve_path("")):
            payload = payload_utils.JobPayload(source, entrypoint)
            resolved_source = payload_utils.resolve_source(payload.source)
            resolved_entrypoint = payload_utils.resolve_entrypoint(payload.source, payload.entrypoint)
            assert not callable(resolved_source)
            assert isinstance(resolved_entrypoint, types.PayloadEntrypoint)
            self.assertEqual(resolved_source.as_posix(), stage_utils.resolve_path(source).absolute().as_posix())
            self.assertEqual(resolved_entrypoint.file_path.as_posix(), expected_entrypoint)

    @parameterized.parameters(  # type: ignore[misc]
        ("not_exist", "file1.py", FileNotFoundError),  # not_exist/ does not exist
        ("src", "file1.py", FileNotFoundError),  # src/file1.py does not exist
        (resolve_path("src"), "file1.py", FileNotFoundError),  # src/file1.py does not exist
        (resolve_path("src"), resolve_path("src/file1.py"), FileNotFoundError),  # src/file1.py does not exist
        ("src", resolve_path("file1.py"), ValueError),  # file1.py is not under src
        (resolve_path("src"), resolve_path("file1.py"), ValueError),  # file1.py is not under src
        ("src/subdir1", resolve_path("src/subdir2/file4.py"), ValueError),  # subdir2/ is not under subdir1/
        ("src/subdir1", "src/subdir2/file4.py", FileNotFoundError),  # relative path resolution fails to find file
        (".", "script1.sh", ValueError),  # script1.sh does not have a .py extension
        ("@test_stage/src", "@test_stage/dir/main.py", ValueError),  # entrypoint is not under source
        ("@test_stage/src/dir1", "@test_stage/src/dir2/main.py", ValueError),  # entrypoint is not under source
        ("@test_stage/src/secondary.py", "@test_stage/src/main.py", ValueError),  # entrypoint is not under source
        (
            "@test_stage/src/secondary.py",
            "snow://headless/abc/versions/v9.8.8/main.py",
            ValueError,
        ),  # entrypoint is not under source
        ("@test_stage/src", "@test_stage/dir/main.java", ValueError),  # unsupported suffix
        ("@test_stage/src", None, ValueError),  # does not specify entrypoint when source is dir
        (
            "snow://headless/abc/versions/v9.8.7",
            "snow://headless/abc/versions/v9.8.8/main.py",
            ValueError,
        ),  # entrypoint is not under source
        (
            "snow://headless/abc/versions/v9.8.7/src/dir1",
            "snow://headless/abc/versions/v9.8.7/src/dir2/main.py",
            ValueError,
        ),  # entrypoint is not under source
        (
            "snow://headless/abc/versions/v9.8.7/secondary.py",
            "snow://headless/abc/versions/v9.8.7/main.py",
            ValueError,
        ),  # entrypoint is not under source
        (
            "snow://headless/abc/versions/v9.8.7/src",
            "snow://headless/abc/versions/v9.8.7/src/main.java",
            ValueError,
        ),  # unsupported suffix
        (
            "snow://headless/test/versions/v9.8.7/",
            None,
            ValueError,
        ),  # does not specify entrypoint when source is dir
        (
            "snowflake://headless/test/versions/v9.8.7/",
            "snowflake://headless/test/versions/v9.8.7/main.py",
            FileNotFoundError,
        ),  # incorrect protocol
        (
            "snow://headless/abc/",
            "snow://headless/test/versions/v9.8.7/main.py",
            FileNotFoundError,
        ),  # incomplete versioned stage
    )
    def test_payload_validate_negative(
        self, source: str, entrypoint: Optional[str], expected_error: type[Exception] = ValueError
    ) -> None:
        with pushd(resolve_path("")):
            payload = payload_utils.JobPayload(source, entrypoint)
            with self.assertRaises(expected_error):
                _ = payload_utils.resolve_source(payload.source)
                _ = payload_utils.resolve_entrypoint(payload.source, payload.entrypoint)

    @parameterized.parameters(  # type: ignore[misc]
        (function_with_pos_arg, ("Hello world", 100)),
        (function_with_pos_arg, ("Hello world", 100), None, True),
        (function_with_opt_arg, ("Hello world", 100)),
        (function_with_opt_arg, ("Hello world", 100, 1.5)),
        (function_with_opt_arg, ("Hello world", 100, 1.5, 0)),
        (function_with_kw_arg, ("Hello world", 100), {"named_arg": "true"}),
        (function_with_kw_arg, ("Hello world", 100), {"named_arg": "true", "opt_named": "provided"}),
        (function_with_kw_arg, ("Hello world", 100), {"named_arg": "true", "opt_named": "multiple words"}),
        (function_with_kw_arg, ("Hello world", 100), {"opt_named": "provided", "named_arg": "true"}),
        (function_with_kw_arg, ("Hello world", 100, 1.5), {"named_arg": "true"}),
        (function_with_pos_arg, tuple(), {"b": 100, "a": "Hello world"}),
        (function_with_kw_arg, ("Hello world",), {"named_arg": "true", "c": 1.5, "b": 100}),
        (function_with_kw_arg, ("Hello world",), {"named_arg": "true", "c": 1.5, "b": 100}, True),
        (functools.partial(function_with_kw_arg, "Hello world", named_arg=True, c=1.5, b=100), tuple(), dict(), True),
        (functools.partial(function_with_any_arg, {"key1": "val", "key2": "val"}), tuple(), dict(), True),
    )
    def test_generate_python_code(
        self,
        func: Callable[..., Any],
        args: list[Any],
        kwargs: Optional[dict[str, Any]] = None,
        source_code_display: bool = False,
    ) -> None:
        kwargs = kwargs or {}

        # Capture the expected stdout
        expected_stdout = io.StringIO()
        try:
            sys.stdout = expected_stdout
            func(*args, **kwargs)
        finally:
            sys.stdout = sys.__stdout__

        with tempfile.TemporaryDirectory() as temp_dir:
            with absltest.mock.patch("snowflake.ml.jobs._utils.constants.STAGE_VOLUME_MOUNT_PATH", temp_dir):
                generated = payload_utils.generate_python_code(func, source_code_display=source_code_display)
            if source_code_display:
                func_name = func.func.__name__ if isinstance(func, functools.partial) else func.__name__
                self.assertIn(f"def {func_name}", generated)

            # Write generated code to a temp file and execute temp file as a subprocess
            with tempfile.NamedTemporaryFile(prefix=temp_dir, suffix=".py") as temp_file:
                temp_file.write(generated.encode("utf-8"))
                temp_file.flush()
                arg_list = [str(arg) for arg in args]
                kwarg_list = [x for k, v in kwargs.items() for x in (f"--{k}", str(v))]
                result = subprocess.run(
                    [sys.executable, temp_file.name] + arg_list + kwarg_list,
                    capture_output=True,
                    text=True,
                )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(expected_stdout.getvalue().strip(), result.stdout.strip())

    @parameterized.parameters(  # type: ignore[misc]
        (function_with_var_args, NotImplementedError),
        (function_with_unpacking_kw_arg, NotImplementedError),
        (function_with_bool_args, ValueError),
        (function_with_collection_args, ValueError),
        (function_with_typing_collection_args, ValueError),
        (function_with_object_arg, ValueError),
        (function_with_any_arg, ValueError),
    )
    def test_generate_python_code_negative(
        self,
        func: Callable[..., Any],
        error_type: type[Exception] = ValueError,
    ) -> None:
        # Write generated code to a temp file and execute temp file as a subprocess
        with self.assertRaises(error_type):
            payload_utils.generate_python_code(func)

    @parameterized.parameters(  # type: ignore[misc]
        ((resolve_path("src/subdir1"), "src.subdir1"), [(resolve_path("src/subdir1"), "src/subdir1")]),
        ((resolve_path("src/subdir2"), "subdir2"), [(resolve_path("src/subdir2"), "subdir2")]),
        ((resolve_path("src/file2.py"), "file2"), [(resolve_path("src/file2.py"), "file2.py")]),
        (("@test_stage/src/file1.py", "src.file1"), [("@test_stage/src/file1.py", "src/file1")]),
        ((resolve_path("src/subdir1"), None), [(resolve_path("src/subdir1"), "subdir1")]),
    )
    def test_resolve_import_path(
        self, imports: tuple[str, Optional[str]], expected_imports: list[tuple[str, Optional[str]]]
    ) -> None:

        with pushd(resolve_path("")):
            source, import_path = imports
            resolved_imports = payload_utils.resolve_import_path(stage_utils.resolve_path(source), import_path)
            normalized = [(p.as_posix(), m) for p, m in resolved_imports]
            self.assertListEqual(expected_imports, normalized)

    @parameterized.parameters(  # type: ignore[misc]
        ("/path/to/archive.zip/nested_file.py", "/path/to/archive.zip"),
        ("/path/to/archive.zip", "/path/to/archive.zip"),
        ("/path/to/regular_file.py", "/path/to/regular_file.py"),
    )
    def test_get_zip_file_from_path(self, path: str, expected_zip_file: str) -> None:
        resolved_path = payload_utils.get_zip_file_from_path(Path(path))
        self.assertEqual(resolved_path.as_posix(), expected_zip_file)

    @parameterized.parameters(  # type: ignore[misc]
        (resolve_path("src/file2.py"), "file2"),
        (resolve_path("src/subdir1"), "lib/src/subdir1"),
    )
    def test_validate_import_path_negative(self, source: str, import_path: Optional[str]) -> None:
        with pushd(resolve_path("")):
            with self.assertRaises(ValueError):
                payload_utils.validate_import_path(source, import_path)

    @parameterized.parameters(  # type: ignore[misc]
        (resolve_path("src/file2.py"), "file2.py"),
        (resolve_path("src/subdir1"), "src/subdir1"),
    )
    def test_validate_import_path_positive(self, source: str, import_path: Optional[str]) -> None:
        with pushd(resolve_path("")):
            try:
                payload_utils.validate_import_path(source, import_path)
            except Exception as e:
                self.fail(f"validate_import_path() raised an exception unexpectedly: {e}")


if __name__ == "__main__":
    absltest.main()
