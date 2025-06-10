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

_STAGEF_PATH_RE = re.compile(r"^@(?P<stage>~|%?\w+)(?:/(?P<relpath>[\w\-./]*))?$")


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

    def is_relative_to(self, path: Union[str, PathLike[str], "StagePath"]) -> bool:
        stage_path = path if isinstance(path, StagePath) else StagePath(os.fspath(path))
        if stage_path.root == self.root:
            return self._path.is_relative_to(stage_path._path)
        else:
            return False

    def relative_to(self, path: Union[str, PathLike[str], "StagePath"]) -> PurePath:
        stage_path = path if isinstance(path, StagePath) else StagePath(os.fspath(path))
        if self.root == stage_path.root:
            return self._path.relative_to(stage_path._path)
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StagePath):
            raise NotImplementedError
        return bool(self.root == other.root and self._path == other._path)

    def __fspath__(self) -> str:
        return self._compose_path(self._path)

    def joinpath(self, *args: Union[str, PathLike[str], "StagePath"]) -> "StagePath":
        path = self
        for arg in args:
            path = path._make_child(arg)
        return path

    def _make_child(self, path: Union[str, PathLike[str], "StagePath"]) -> "StagePath":
        stage_path = path if isinstance(path, StagePath) else StagePath(os.fspath(path))
        if self.root == stage_path.root:
            child_path = self._path.joinpath(stage_path._path)
            return StagePath(self._compose_path(child_path))
        else:
            return stage_path


def identify_stage_path(path: str) -> Union[StagePath, Path]:
    try:
        stage_path = StagePath(path)
    except ValueError:
        return Path(path)
    return stage_path
