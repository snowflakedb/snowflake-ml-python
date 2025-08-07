import os
import re
from os import PathLike
from pathlib import Path, PurePath
from typing import Union

from snowflake.ml._internal.utils import identifier

PROTOCOL_NAME = "snow"
_SNOWURL_PATH_RE = re.compile(
    rf"^(?:(?:{PROTOCOL_NAME}://)?"
    r"(?<!@)(?P<domain>\w+)/"
    rf"(?P<name>(?:{identifier._SF_IDENTIFIER}\.){{,2}}{identifier._SF_IDENTIFIER})/)?"
    r"(?P<path>versions(?:/(?P<version>[^/]+)(?:/(?P<relpath>.*))?)?)$"
)

# Break long regex into two main parts
_STAGE_PATTERN = rf"~|%?(?:(?:{identifier._SF_IDENTIFIER}\.?){{,2}}{identifier._SF_IDENTIFIER})"
_RELPATH_PATTERN = r"[\w\-./]*"
_STAGEF_PATH_RE = re.compile(rf"^@(?P<stage>{_STAGE_PATTERN})(?:/(?P<relpath>{_RELPATH_PATTERN}))?$")


class StagePath:
    def __init__(self, path: str) -> None:
        stage_match = _SNOWURL_PATH_RE.fullmatch(path) or _STAGEF_PATH_RE.fullmatch(path)
        if not stage_match:
            raise ValueError(f"{path} is not a valid stage path")
        path = path.strip()
        self._raw_path = path
        relpath = stage_match.group("relpath")
        start, _ = stage_match.span("relpath")
        self._root = self._raw_path[0:start].rstrip("/") if relpath else self._raw_path.rstrip("/")
        self._path = Path(relpath or "")

    @property
    def parts(self) -> tuple[str, ...]:
        return self._path.parts

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def parent(self) -> "StagePath":
        if self._path.parent == Path(""):
            return StagePath(self._root)
        else:
            return StagePath(f"{self._root}/{self._path.parent}")

    @property
    def root(self) -> str:
        return self._root

    @property
    def suffix(self) -> str:
        return self._path.suffix

    def _compose_path(self, path: Path) -> str:
        # in pathlib, Path("") = "."
        if path == Path(""):
            return self.root
        else:
            return f"{self.root}/{path}"

    def is_relative_to(self, *other: Union[str, os.PathLike[str]]) -> bool:
        if not other:
            raise TypeError("is_relative_to() requires at least one argument")
        # For now, we only support a single argument, like pathlib.Path in Python < 3.12
        path = other[0]
        stage_path = path if isinstance(path, StagePath) else StagePath(os.fspath(path))
        if stage_path.root == self.root:
            return self._path.is_relative_to(stage_path._path)
        else:
            return False

    def relative_to(self, *other: Union[str, os.PathLike[str]]) -> PurePath:
        if not other:
            raise TypeError("relative_to() requires at least one argument")
        if not self.is_relative_to(*other):
            raise ValueError(f"{other} does not start with {self._raw_path}")
        path = other[0]
        stage_path = path if isinstance(path, StagePath) else StagePath(os.fspath(path))
        if self.root == stage_path.root:
            return self._path.relative_to(stage_path._path)
        else:
            raise ValueError(f"{self._raw_path} does not start with {stage_path._raw_path}")

    def absolute(self) -> "StagePath":
        return self

    def as_posix(self) -> str:
        return self._compose_path(self._path)

    # TODO Add actual implementation https://snowflakecomputing.atlassian.net/browse/SNOW-2112795
    def exists(self) -> bool:
        return True

    # TODO Add actual implementation https://snowflakecomputing.atlassian.net/browse/SNOW-2112795
    def is_file(self) -> bool:
        return True

    # TODO Add actual implementation https://snowflakecomputing.atlassian.net/browse/SNOW-2112795
    def is_dir(self) -> bool:
        return True

    def is_absolute(self) -> bool:
        return True

    def __str__(self) -> str:
        return self.as_posix()

    def __repr__(self) -> str:
        return f"StagePath('{self.as_posix()}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StagePath):
            raise NotImplementedError
        return bool(self.root == other.root and self._path == other._path)

    def __fspath__(self) -> str:
        return self._compose_path(self._path)

    def joinpath(self, *args: Union[str, PathLike[str]]) -> "StagePath":
        """
        Joins the given path arguments to the current path,
        mimicking the behavior of pathlib.Path.joinpath.
        If the argument is a stage path (i.e., an absolute path),
        it overrides the current path and is returned as the final path.
        If the argument is a normal path, it is joined with the current relative path
        using self._path.joinpath(arg).

        Args:
            *args: Path components to join.

        Returns:
            A new StagePath with the joined path.

        Raises:
            NotImplementedError: the argument is a stage path.
        """
        path = self
        for arg in args:
            if isinstance(arg, StagePath):
                raise NotImplementedError
            else:
                # the arg might be an absolute path, so we need to remove the leading '/'
                path = StagePath(f"{path.root}/{path._path.joinpath(arg).as_posix().lstrip('/')}")
        return path
