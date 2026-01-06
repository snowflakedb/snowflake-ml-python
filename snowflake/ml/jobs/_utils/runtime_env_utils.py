import datetime
import logging
from typing import Any, Literal, Optional, Union

from packaging.version import Version
from pydantic import BaseModel, Field, RootModel, field_validator

from snowflake import snowpark
from snowflake.ml.jobs._utils import constants, query_helper


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
    created_on: datetime.datetime = Field(alias="createdOn")
    id: Optional[str] = Field(alias="id")

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

    def get_spcs_container_runtimes(
        self,
        *,
        hardware_type: Optional[str] = None,
        python_version: Optional[Version] = None,
    ) -> list[SpcsContainerRuntime]:
        # TODO(SNOW-2682000): parse version from NRE in a safer way, like relying on the label,id or image tag.
        entries: list[RuntimeEnvironmentEntry] = [
            entry
            for entry in self.root.values()
            if entry.spcs_container_runtime is not None
            and (hardware_type is None or entry.spcs_container_runtime.hardware_type.lower() == hardware_type.lower())
            and (
                python_version is None
                or (
                    entry.spcs_container_runtime.python_version.major == python_version.major
                    and entry.spcs_container_runtime.python_version.minor == python_version.minor
                )
            )
        ]
        entries.sort(key=lambda e: e.created_on, reverse=True)

        return [entry.spcs_container_runtime for entry in entries if entry.spcs_container_runtime is not None]


def _extract_image_tag(image_url: str) -> Optional[str]:
    image_tag = image_url.rsplit(":", 1)[-1]
    return image_tag


def find_runtime_image(
    session: snowpark.Session, target_hardware: Literal["CPU", "GPU"], target_python_version: Optional[str] = None
) -> Optional[str]:
    python_version = (
        Version(target_python_version) if target_python_version else Version(constants.DEFAULT_PYTHON_VERSION)
    )
    rows = query_helper.run_query(session, "CALL SYSTEM$NOTEBOOKS_FIND_LABELED_RUNTIMES()")
    if not rows:
        return None
    try:
        runtime_envs = RuntimeEnvironmentsDict.model_validate_json(rows[0][0])
        spcs_container_runtimes = runtime_envs.get_spcs_container_runtimes(
            hardware_type=target_hardware,
            python_version=python_version,
        )
    except Exception as e:
        logging.warning(f"Failed to parse runtime image name from {rows[0][0]}, error: {e}")
        return None

    selected_runtime = spcs_container_runtimes[0] if spcs_container_runtimes else None
    return selected_runtime.runtime_container_image if selected_runtime else None
