"""Parse model extension specification version 2.0 into runtime metadata objects."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any, cast

from packaging import version

from snowflake.ml import version as snowml_version
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model_spec import model_extension_spec
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import model_blob_meta, model_meta_schema
from snowflake.ml.model._packager.model_runtime import model_runtime

_MODEL_DIR_PREFIX = "model/"
# Client load always reads the packaged artifact under the model directory.
_PACKAGED_ENV_CONDA = "env/conda.yml"
_PACKAGED_ENV_PIP = "env/requirements.txt"
_CPU_RUNTIME_CONDA = "runtimes/cpu/env/conda.yml"
_CPU_RUNTIME_PIP = "runtimes/cpu/env/requirements.txt"
_GPU_RUNTIME_CONDA = "runtimes/gpu/env/conda.yml"
_GPU_RUNTIME_PIP = "runtimes/gpu/env/requirements.txt"
_METHOD_OPTION_KEYS = ("case_sensitive", "max_batch_size")


@dataclasses.dataclass
class ParsedV2ModelMetadata:
    """Runtime objects parsed from a version 2.0 model extension spec."""

    name: str
    model_type: model_types.SupportedModelHandlerType
    env: model_env.ModelEnv
    runtimes: dict[str, model_runtime.ModelRuntime] | None
    signatures: dict[str, model_signature.ModelSignature]
    metadata: dict[str, str] | None
    min_snowpark_ml_version: str
    models: dict[str, model_blob_meta.ModelBlobMeta]
    original_metadata_version: str
    task: model_types.Task
    explain_algorithm: model_meta_schema.ModelExplainAlgorithm | None
    function_properties: dict[str, dict[str, Any]]
    method_options: dict[str, dict[str, Any]]
    case_sensitive: bool | None
    sample_input_file_paths: dict[str, str]
    packaged_env_dict: model_meta_schema.ModelEnvDict
    function_types: dict[str, str]


def parse(model_dir_path: str, raw_spec: Mapping[str, Any]) -> ParsedV2ModelMetadata:
    """Validate a version 2.0 spec and build the runtime objects it names.

    Args:
        model_dir_path: Directory that contains ``model.yaml`` and env files.
        raw_spec: Parsed ``model.yaml`` document.

    Returns:
        Signatures, blobs, env, runtimes, and function types from the spec.

    Raises:
        ValueError: If the spec is missing a required field.
    """
    validated = model_extension_spec.ModelExtensionSpecV2._validate(raw_spec)
    model_section = cast(Mapping[str, Any], validated["model"])
    serving_section = cast(Mapping[str, Any], validated["serving"])
    models_section = cast(Mapping[str, Any], model_section["details"]["models"])
    functions = cast(Mapping[str, Any], serving_section["functions"])
    serving_environments = cast(Mapping[str, Any] | None, serving_section.get("env")) or {}

    if not models_section:
        raise ValueError("model specification: model blobs must not be empty.")

    min_snowpark_ml_version = _min_snowpark_ml_version(model_section)
    _check_min_snowpark_ml_version(min_snowpark_ml_version)
    model_type = cast(model_types.SupportedModelHandlerType, _resolve_model_type(model_section, models_section))

    signatures, sample_input_file_paths = _convert_signatures(functions)
    function_properties, method_options, case_sensitive, function_types = _convert_function_metadata(functions)
    models = _convert_models(models_section)
    packaged_env_dict, runtime_dicts = _convert_environments(serving_environments)
    explainability = _extract_explainability(models_section)
    explain_algorithm = None
    if explainability:
        explain_algorithm = model_meta_schema.ModelExplainAlgorithm(explainability["algorithm"])
    task_value = str(model_section.get("task") or model_types.Task.UNKNOWN.value)

    model_dir = pathlib.Path(model_dir_path)
    env = model_env.ModelEnv()
    env.load_from_dict(model_dir, packaged_env_dict)
    runtimes = _load_runtimes(model_dir, env, runtime_dicts)

    return ParsedV2ModelMetadata(
        name=next(iter(models_section)),
        model_type=model_type,
        env=env,
        runtimes=runtimes,
        signatures=signatures,
        metadata=_convert_user_data(validated.get("user_data")),
        min_snowpark_ml_version=min_snowpark_ml_version,
        models=models,
        original_metadata_version=str(validated["version"]),
        task=model_types.Task(task_value),
        explain_algorithm=explain_algorithm,
        function_properties=function_properties,
        method_options=method_options,
        case_sensitive=case_sensitive,
        sample_input_file_paths=sample_input_file_paths,
        packaged_env_dict=packaged_env_dict,
        function_types=function_types,
    )


def _min_snowpark_ml_version(model_section: Mapping[str, Any]) -> str:
    client = model_section.get("client") or {}
    min_version = client.get("min_version") if isinstance(client, Mapping) else None
    if not min_version:
        return model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION
    return str(min_version)


def _check_min_snowpark_ml_version(min_snowpark_ml_version: str) -> None:
    if version.parse(min_snowpark_ml_version) > version.parse(snowml_version.VERSION):
        raise RuntimeError(
            f"The minimal version required to load the model is {min_snowpark_ml_version}, "
            f"while current version of Snowpark ML library is {snowml_version.VERSION}."
        )


def _resolve_model_type(model_section: Mapping[str, Any], models_section: Mapping[str, Any]) -> str:
    raw_model_type = model_section.get("framework")
    if not raw_model_type:
        raw_model_type = next(
            (
                blob.get("model_type")
                for blob in models_section.values()
                if isinstance(blob, Mapping) and blob.get("model_type")
            ),
            None,
        )
    if not raw_model_type:
        raise ValueError("model specification: model type is missing.")
    return str(raw_model_type).lower()


def _convert_models(models_section: Mapping[str, Any]) -> dict[str, model_blob_meta.ModelBlobMeta]:
    converted: dict[str, model_blob_meta.ModelBlobMeta] = {}
    for name, blob in models_section.items():
        if not isinstance(blob, Mapping):
            raise ValueError(f"model specification: model blob '{name}' must be a mapping.")
        model_type = blob.get("model_type")
        if not model_type:
            raise ValueError(f"model specification: model blob '{name}' is missing model_type.")
        path = blob.get("path")
        if path is None:
            raise ValueError(f"model specification: model blob '{name}' is missing path.")
        handler_version = blob.get("handler_version") or model_meta_schema.MODEL_METADATA_VERSION
        blob_kwargs: dict[str, Any] = {
            "name": name,
            "model_type": str(model_type).lower(),
            "path": str(path),
            "handler_version": str(handler_version),
            "artifacts": _convert_artifacts(blob.get("artifacts")),
            "options": dict(blob.get("options") or {}),
        }
        function_properties = blob.get("function_properties")
        if isinstance(function_properties, Mapping):
            blob_kwargs["function_properties"] = dict(function_properties)
        converted[name] = model_blob_meta.ModelBlobMeta(**blob_kwargs)
    return converted


def _convert_artifacts(artifacts: Any) -> dict[str, str]:
    if not artifacts:
        return {}
    if not isinstance(artifacts, Mapping):
        raise ValueError("model specification: model blob artifacts must be a mapping.")
    converted: dict[str, str] = {}
    for artifact_name, artifact_value in artifacts.items():
        if isinstance(artifact_value, Mapping):
            path = artifact_value.get("path")
            if path is None:
                raise ValueError(f"model specification: artifact '{artifact_name}' is missing path.")
            converted[str(artifact_name)] = str(path)
        else:
            converted[str(artifact_name)] = str(artifact_value)
    return converted


def _extract_explainability(models_section: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for blob in models_section.values():
        if not isinstance(blob, Mapping):
            continue
        explainability = blob.get("explainability")
        if isinstance(explainability, Mapping) and explainability:
            return explainability
    return None


def _convert_signatures(
    functions: Mapping[str, Any],
) -> tuple[dict[str, model_signature.ModelSignature], dict[str, str]]:
    signatures: dict[str, model_signature.ModelSignature] = {}
    sample_input_file_paths: dict[str, str] = {}
    for function_name, function_spec in functions.items():
        signature = dict(function_spec["signature"])
        sample_path = signature.get("sample_input_file_path")
        if sample_path:
            sample_input_file_paths[function_name] = str(sample_path)
        signatures[function_name] = model_signature.ModelSignature.from_dict(signature)
    return signatures, sample_input_file_paths


def _convert_function_metadata(
    functions: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool | None, dict[str, str]]:
    function_properties: dict[str, dict[str, Any]] = {}
    method_options: dict[str, dict[str, Any]] = {}
    function_types: dict[str, str] = {}
    case_sensitive_values: list[bool] = []

    for function_name, function_spec in functions.items():
        function_types[function_name] = function_spec["type"]

        properties = function_spec.get("properties") or {}
        if not isinstance(properties, Mapping):
            continue

        if bool(properties.get("is_partition", False)):
            function_properties[function_name] = {
                model_meta_schema.FunctionProperties.PARTITIONED.value: True,
            }

        options: dict[str, Any] = {}
        for key in _METHOD_OPTION_KEYS:
            if key in properties:
                options[key] = properties[key]
        volatility = function_spec.get("volatility")
        if isinstance(volatility, str) and volatility:
            options["volatility"] = volatility
        if options:
            method_options[function_name] = options

        if "case_sensitive" in properties:
            case_sensitive_values.append(bool(properties["case_sensitive"]))

    global_case_sensitive: bool | None = None
    if case_sensitive_values and all(value == case_sensitive_values[0] for value in case_sensitive_values):
        if len(case_sensitive_values) == len(functions):
            global_case_sensitive = case_sensitive_values[0]

    return function_properties, method_options, global_case_sensitive, function_types


def _convert_environments(
    serving_environments: Mapping[str, Any],
) -> tuple[model_meta_schema.ModelEnvDict, dict[str, model_meta_schema.ModelRuntimeDict] | None]:
    """Map serving environments onto the legacy metadata env and runtimes.

    Client load always uses the packaged files ``env/conda.yml`` and
    ``env/requirements.txt``. ``python_env`` is the warehouse runtime and is not
    part of this metadata. ``cpu_env`` becomes ``runtimes.cpu``. ``gpu_env`` becomes
    ``runtimes.gpu`` only when ``cpu_env`` is also present and ``cuda_version`` is set.
    ``gpu_env`` alone leaves runtimes unset so ``ModelMetadata.runtimes`` builds
    ``cpu`` and ``gpu`` from the packaged env.

    Args:
        serving_environments: ``serving.env`` mapping from the version 2.0 spec.

    Returns:
        The packaged env dict and runtime dicts. Runtime dicts are omitted when
        the spec has no ``cpu_env``.
    """
    packaged_env = _packaged_env_dict(serving_environments)
    runtimes: dict[str, model_meta_schema.ModelRuntimeDict] = {}

    cpu_environment = _named_environment(serving_environments, "cpu_env")
    if cpu_environment is not None:
        runtimes["cpu"] = _serving_env_to_runtime_dict(
            cpu_environment, default_conda=_CPU_RUNTIME_CONDA, default_pip=_CPU_RUNTIME_PIP
        )

    gpu_environment = _named_environment(serving_environments, "gpu_env")
    if cpu_environment is not None and gpu_environment is not None and gpu_environment.get("cuda_version"):
        runtimes["gpu"] = _serving_env_to_runtime_dict(
            gpu_environment, default_conda=_GPU_RUNTIME_CONDA, default_pip=_GPU_RUNTIME_PIP
        )

    return packaged_env, runtimes or None


def _named_environment(serving_environments: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    environment = serving_environments.get(name)
    if environment is None:
        return None
    if not isinstance(environment, Mapping):
        raise ValueError(f"model specification: serving environment '{name}' must be a mapping.")
    return environment


def _packaged_env_dict(serving_environments: Mapping[str, Any]) -> model_meta_schema.ModelEnvDict:
    version_source = (
        _named_environment(serving_environments, "python_env")
        or _named_environment(serving_environments, "cpu_env")
        or _named_environment(serving_environments, "gpu_env")
        or {}
    )
    packaged_env = model_meta_schema.ModelEnvDict(
        python_version=str(version_source.get("python_version") or ""),
        snowpark_ml_version=str(
            version_source.get("snowpark_ml_version") or model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION
        ),
        conda=_PACKAGED_ENV_CONDA,
        pip=_PACKAGED_ENV_PIP,
    )
    gpu_environment = _named_environment(serving_environments, "gpu_env")
    if gpu_environment is not None and gpu_environment.get("cuda_version"):
        packaged_env["cuda_version"] = str(gpu_environment["cuda_version"])
    return packaged_env


def _load_runtimes(
    model_dir: pathlib.Path,
    env: model_env.ModelEnv,
    runtime_dicts: dict[str, model_meta_schema.ModelRuntimeDict] | None,
) -> dict[str, model_runtime.ModelRuntime] | None:
    if not runtime_dicts:
        return None
    runtimes = {
        name: model_runtime.ModelRuntime.load(model_dir, name, env, runtime_dict)
        for name, runtime_dict in runtime_dicts.items()
    }
    # ModelRuntime.load leaves is_gpu false. The gpu entry is created only when cuda_version is set.
    gpu_runtime = runtimes.get("gpu")
    if gpu_runtime is not None:
        gpu_runtime.is_gpu = True
    return runtimes


def _serving_env_to_runtime_dict(
    environment: Mapping[str, Any],
    *,
    default_conda: str,
    default_pip: str,
) -> model_meta_schema.ModelRuntimeDict:
    dependencies: model_meta_schema.ModelRuntimeDependenciesDict = {}
    conda_path = environment.get("conda")
    pip_path = environment.get("pip")
    dependencies["conda"] = _to_model_dir_relpath(str(conda_path)) if conda_path else default_conda
    dependencies["pip"] = _to_model_dir_relpath(str(pip_path)) if pip_path else default_pip
    if "artifact_repository_map" in environment:
        dependencies["artifact_repository_map"] = environment.get("artifact_repository_map")

    imports = environment.get("imports") or []
    if not isinstance(imports, list):
        raise ValueError("model specification: serving environment imports must be a list.")

    runtime_dict: model_meta_schema.ModelRuntimeDict = {
        "imports": [str(item) for item in imports],
        "dependencies": dependencies,
    }
    if environment.get("resource_constraint") is not None:
        runtime_dict["resource_constraint"] = environment.get("resource_constraint")
    return runtime_dict


def _to_model_dir_relpath(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized.startswith(_MODEL_DIR_PREFIX):
        return normalized[len(_MODEL_DIR_PREFIX) :]
    return normalized


def _convert_user_data(user_data: Any) -> dict[str, str] | None:
    if not isinstance(user_data, Mapping) or not user_data:
        return None
    if all(isinstance(value, str) for value in user_data.values()):
        return {str(key): value for key, value in user_data.items()}
    return None
