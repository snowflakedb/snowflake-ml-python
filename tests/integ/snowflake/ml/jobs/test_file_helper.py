import importlib_resources


class TestAsset:
    def __init__(self, name: str, resolve_path: bool = True) -> None:
        self.name = name
        self.path = (
            importlib_resources.files("tests.integ.snowflake.ml.jobs.test_files").joinpath(name)
            if resolve_path
            else name
        )

    def __repr__(self) -> str:
        return f"TestAsset({self.name})"
