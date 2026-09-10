"""Adapter for validated legacy model metadata."""

import copy
from collections.abc import Mapping
from typing import Any, cast

from snowflake.ml.model._client.model_spec import model_spec
from snowflake.ml.model._packager.model_meta import model_meta_schema


class LegacyModelSpec(model_spec.ModelSpec):
    """Read-only access to a validated legacy model metadata dictionary."""

    def __init__(self, metadata: model_meta_schema.ModelMetadataDict) -> None:
        self._metadata = copy.deepcopy(metadata)

    @property
    def raw_spec(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def spec_version(self) -> str:
        return self._metadata["version"]

    @property
    def signatures(self) -> Mapping[str, dict[str, Any]]:
        return {name: copy.deepcopy(dict(signature)) for name, signature in self._metadata["signatures"].items()}

    @property
    def model_type(self) -> str:
        return cast(str, self._metadata["model_type"])

    @property
    def model_blobs(self) -> Mapping[str, Mapping[str, Any]]:
        return copy.deepcopy(self._metadata["models"])

    @property
    def model_options(self) -> Mapping[str, Mapping[str, Any]]:
        options: dict[str, Mapping[str, Any]] = {}
        for name, blob in self._metadata["models"].items():
            model_options = blob.get("options")
            options[name] = copy.deepcopy(model_options) if isinstance(model_options, Mapping) else {}
        return options

    @property
    def model_tasks(self) -> Mapping[str, str | None]:
        default_task = self._metadata.get("task")
        tasks: dict[str, str | None] = {}
        for name, blob in self._metadata["models"].items():
            model_options = blob.get("options")
            task = model_options.get("task", default_task) if isinstance(model_options, Mapping) else default_task
            tasks[name] = cast(str | None, task)
        return tasks

    @property
    def supports_gpu(self) -> bool:
        if self._metadata["env"].get("cuda_version"):
            return True
        return "gpu" in (self._metadata.get("runtimes") or {})

    def is_partitioned(self, function_name: str) -> bool:
        function_properties = self._metadata.get("function_properties") or {}
        properties = function_properties.get(function_name) or {}
        partitioned_key = model_meta_schema.FunctionProperties.PARTITIONED.value
        value = properties.get(partitioned_key, properties.get(partitioned_key.lower(), True))
        return bool(value)

    @property
    def method_options(self) -> Mapping[str, Mapping[str, Any]]:
        return copy.deepcopy(self._metadata.get("method_options") or {})

    @property
    def case_sensitive(self) -> bool:
        return bool(self._metadata.get("case_sensitive", False))
