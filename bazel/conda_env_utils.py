from typing import Dict, Iterable, List, Tuple


def merge_conda_envs(env_dicts: Iterable[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    channels_set = set()
    channels = []
    dependencies = _Dependencies()
    for env_dict in env_dicts:
        this_channels, this_deps = validate_and_get_conda_env(env_dict)
        # Dedup channels but keep the order as they appear.
        for channel in this_channels:
            if channel not in channels_set:
                channels_set.add(channel)
                channels.append(channel)
        for dep in this_deps:
            dependencies.add(dep)

    return {"channels": channels, "dependencies": dependencies.to_list()}


class _Dependencies:
    def __init__(self) -> None:
        # maps name to version
        self._deps = {}  # type: Dict[str, str]

    def add(self, dep_string: str) -> None:
        name_and_version = dep_string.split("==")
        if len(name_and_version) != 2:
            raise ValueError(f"Invalid dependency spec. Expected <name>==<version> but got {dep_string}")
        name, version = name_and_version
        existing_version = self._deps.get(name, None)
        if existing_version is not None:
            raise ValueError(
                f"Found duplicate package: {name}. Prefer specifying the package only"
                "in conda-env-snowflake.yml, if it is available in the Snowflake "
                "conda channel."
            )
        self._deps[name] = version

    def to_list(self) -> List[str]:
        deps = [f"{name}=={version}" for name, version in self._deps.items()]
        deps.sort()
        return deps


def validate_and_get_conda_env(env_dict: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    assert len(env_dict) == 2, "A conda env YAML must contain only two entries, 'channels' and 'dependencies'"
    assert "channels" in env_dict
    assert "dependencies" in env_dict

    return env_dict["channels"], env_dict["dependencies"]
