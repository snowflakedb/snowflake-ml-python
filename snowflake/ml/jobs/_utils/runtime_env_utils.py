from typing import Any, Optional, Union

from packaging.version import Version
from pydantic import BaseModel, Field, RootModel, field_validator


class SpcsContainerRuntime(BaseModel):
    python_version: Version = Field(alias="pythonVersion")
    hardware_type: str = Field(alias="hardwareType")
    runtime_container_image: str = Field(alias="runtimeContainerImage")

    @field_validator("python_version", mode="before")
    @classmethod
    def validate_python_version(cls, v: Union[str, Version]) -> Version:
        if isinstance(v, Version):
            return v
        try:
            return Version(v)
        except Exception:
            raise ValueError(f"Invalid Python version format: {v}")

    class Config:
        frozen = True
        extra = "allow"
        arbitrary_types_allowed = True


class RuntimeEnvironmentEntry(BaseModel):
    spcs_container_runtime: Optional[SpcsContainerRuntime] = Field(alias="spcsContainerRuntime", default=None)

    class Config:
        extra = "allow"
        frozen = True


class RuntimeEnvironmentsDict(RootModel[dict[str, RuntimeEnvironmentEntry]]):
    @field_validator("root", mode="before")
    @classmethod
    def _filter_to_dict_entries(cls, data: Any) -> dict[str, dict[str, Any]]:
        """
        Pre-validation hook: keep only those items at the root level
        whose values are dicts. Non-dict values will be dropped.

        Args:
            data: The input data to filter, expected to be a dictionary.

        Returns:
            A dictionary containing only the key-value pairs where values are dictionaries.

        Raises:
            ValueError: If input data is not a dictionary.
        """
        # If the entire root is not a dict, raise error immediately
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary data, but got {type(data).__name__}: {data}")

        # Filter out any key whose value is not a dict
        return {key: value for key, value in data.items() if isinstance(value, dict)}

    def get_spcs_container_runtimes(self) -> list[SpcsContainerRuntime]:
        return [
            entry.spcs_container_runtime for entry in self.root.values() if entry.spcs_container_runtime is not None
        ]
