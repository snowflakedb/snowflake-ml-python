"""Read-only access to model extension specification version 2.0."""

import copy
from collections.abc import Mapping
from typing import Any, cast

from snowflake.ml.model import target_platform
from snowflake.ml.model._client.model_spec import (
    model_extension_spec_schema,
    model_spec,
)

_SPEC_VERSION = "2.0"
_IS_PARTITION_PROPERTY = "is_partition"


class ModelExtensionSpecV2(model_spec.ModelSpec):
    """Read-only wrapper over a version 2.0 model specification."""

    def __init__(self, raw_spec: Mapping[str, Any]) -> None:
        validated = self._validate(raw_spec)
        self._raw_spec = copy.deepcopy(validated)

    @staticmethod
    def _validate(raw_spec: Mapping[str, Any]) -> model_extension_spec_schema.ModelExtensionSpecSchema:
        if str(raw_spec.get("version")) != _SPEC_VERSION:
            raise ValueError("model specification: version must be 2.0.")

        model_section = raw_spec.get("model")
        if not isinstance(model_section, Mapping):
            raise ValueError("model specification: model section must be a mapping.")

        serving_section = raw_spec.get("serving")
        if not isinstance(serving_section, Mapping):
            raise ValueError("model specification: serving section must be a mapping.")

        details = model_section.get("details")
        if not isinstance(details, Mapping):
            raise ValueError("model specification: model details must be a mapping.")

        models = details.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("model specification: model blobs must be a mapping.")

        for model_name, model_blob in models.items():
            if not isinstance(model_blob, Mapping):
                raise ValueError(f"model specification: model blob '{model_name}' must be a mapping.")
            model_options = model_blob.get("options")
            if model_options is not None and not isinstance(model_options, Mapping):
                raise ValueError(f"model specification: model blob '{model_name}' options must be a mapping.")

        client = model_section.get("client")
        if client is not None and not isinstance(client, Mapping):
            raise ValueError("model specification: client must be a mapping.")

        target_platforms = model_section.get("target_platforms")
        if target_platforms is not None and not isinstance(target_platforms, list):
            raise ValueError("model specification: target platforms must be a list.")

        serving_environments = serving_section.get("env")
        if serving_environments is not None:
            if not isinstance(serving_environments, Mapping):
                raise ValueError("model specification: serving environments must be a mapping.")
            for environment_name, environment in serving_environments.items():
                if not isinstance(environment, Mapping):
                    raise ValueError(
                        f"model specification: serving environment '{environment_name}' must be a mapping."
                    )

        functions = serving_section.get("functions")
        if not isinstance(functions, Mapping):
            raise ValueError("model specification: serving functions must be a mapping.")

        for function_name, function_spec in functions.items():
            if not isinstance(function_spec, Mapping):
                raise ValueError(f"model specification: function '{function_name}' must be a mapping.")
            if not isinstance(function_spec.get("signature"), Mapping):
                raise ValueError(f"model specification: function '{function_name}' signature must be a mapping.")
            function_properties = function_spec.get("properties")
            if function_properties is not None and not isinstance(function_properties, Mapping):
                raise ValueError(f"model specification: function '{function_name}' properties must be a mapping.")

        return cast(model_extension_spec_schema.ModelExtensionSpecSchema, raw_spec)

    @property
    def raw_spec(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._raw_spec)

    @property
    def spec_version(self) -> str:
        return _SPEC_VERSION

    @property
    def signatures(self) -> Mapping[str, dict[str, Any]]:
        functions = self._raw_spec["serving"]["functions"]
        return {name: copy.deepcopy(dict(function["signature"])) for name, function in functions.items()}

    @property
    def model_type(self) -> str:
        model_section = self._raw_spec["model"]
        raw_model_type = model_section.get("framework")
        if not raw_model_type:
            models = model_section["details"]["models"]
            raw_model_type = next(
                (blob.get("model_type") for blob in models.values() if blob.get("model_type")),
                None,
            )
        if not raw_model_type:
            raise ValueError("model specification: model type is missing.")

        # Version 2.0 declares framework names in upper case, while callers compare against the
        # lower case model type vocabulary that both specification schemas share.
        return str(raw_model_type).lower()

    @property
    def model_blobs(self) -> Mapping[str, Mapping[str, Any]]:
        return copy.deepcopy(self._raw_spec["model"]["details"]["models"])

    @property
    def model_options(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            name: copy.deepcopy(blob.get("options") or {})
            for name, blob in self._raw_spec["model"]["details"]["models"].items()
        }

    @property
    def model_tasks(self) -> Mapping[str, str | None]:
        model_section = self._raw_spec["model"]
        default_task = model_section.get("task")
        tasks: dict[str, str | None] = {}
        for name, blob in model_section["details"]["models"].items():
            options = blob.get("options") or {}
            tasks[name] = cast(str | None, options.get("task") or default_task)
        return tasks

    @property
    def supports_gpu(self) -> bool:
        target_platforms = self._raw_spec["model"].get("target_platforms") or []
        for platform in target_platforms:
            platform_value = platform.value if isinstance(platform, target_platform.TargetPlatform) else platform
            if str(platform_value).strip().upper() == target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value:
                return True

        environments = self._raw_spec["serving"].get("env") or {}
        return any(
            isinstance(environment, Mapping) and bool(environment.get("cuda_version"))
            for environment in environments.values()
        )

    def is_partitioned(self, function_name: str) -> bool:
        function = self._raw_spec["serving"]["functions"].get(function_name)
        if function is None:
            return True
        properties = function.get("properties") or {}
        return bool(properties.get(_IS_PARTITION_PROPERTY, False))

    @property
    def method_options(self) -> Mapping[str, Mapping[str, Any]]:
        options: dict[str, Mapping[str, Any]] = {}
        for function_name, function_spec in self._raw_spec["serving"]["functions"].items():
            properties = function_spec.get("properties") or {}
            if properties:
                options[function_name] = copy.deepcopy(dict(properties))
        return options

    @property
    def case_sensitive(self) -> bool:
        return any(
            bool((function_spec.get("properties") or {}).get("case_sensitive", False))
            for function_spec in self._raw_spec["serving"]["functions"].values()
        )
