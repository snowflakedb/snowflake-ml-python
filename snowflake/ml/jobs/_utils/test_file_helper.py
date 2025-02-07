from typing import cast

import importlib_resources


class TestAsset:
    def __init__(self, name: str, resolve: bool = True) -> None:
        self.name = name
        self.path = resolve_path(name) if resolve else name

    def __repr__(self) -> str:
        return f"TestAsset({self.name})"


def resolve_path(path: str) -> str:
    traverse = importlib_resources.files("snowflake.ml.jobs._utils.test_files")
    if path:
        return cast(str, traverse.joinpath(path).as_posix())
    return cast(str, traverse._paths[0].as_posix())
