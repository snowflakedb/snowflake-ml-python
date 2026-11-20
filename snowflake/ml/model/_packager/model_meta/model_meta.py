import collections
import os
import pathlib
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Generator, TypedDict, cast

import cloudpickle
import yaml
from packaging import requirements, version
from typing_extensions import Required

from snowflake.ml import version as snowml_version
from snowflake.ml._internal import env_utils, file_utils, platform_capabilities
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model_spec import model_extension_spec
from snowflake.ml.model._model_composer import huggingface_lazy_uploader
from snowflake.ml.model._model_composer.model_method import model_method_options
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta_schema,
    model_meta_v2,
)
from snowflake.ml.model._packager.model_meta_migrator import migrator_plans
from snowflake.ml.model._packager.model_runtime import model_runtime
from snowflake.ml.model.code_path import CodePath
from snowflake.snowpark import exceptions as snowpark_exceptions

MODEL_METADATA_FILE = "model.yaml"
_MODEL_EXTENSION_SPEC_VERSION = "2.0"
# Directory holding the packaged model inside the uploaded workspace. The version 2.0 specification resolves every
# file path from the workspace root.
_MODEL_DIR_IN_WORKSPACE = "model"
_WAREHOUSE_RUNTIME_NAME = "warehouse_runtime"
PYTHON_ENV_NAME = "python_env"
MODEL_CODE_DIR = "code"

_PACKAGING_REQUIREMENTS = ["cloudpickle"]

_SNOWFLAKE_PKG_NAME = "snowflake"
_SNOWFLAKE_ML_PKG_NAME = f"{_SNOWFLAKE_PKG_NAME}.ml"


@contextmanager
def create_model_metadata(
    *,
    model_dir_path: str,
    name: str,
    model_type: model_types.SupportedModelHandlerType,
    signatures: dict[str, model_signature.ModelSignature] | None = None,
    function_properties: dict[str, dict[str, Any]] | None = None,
    metadata: dict[str, str] | None = None,
    code_paths: list[model_types.CodePathLike] | None = None,
    ext_modules: list[ModuleType] | None = None,
    conda_dependencies: list[str] | None = None,
    pip_requirements: list[str] | None = None,
    artifact_repository_map: dict[str, str] | None = None,
    resource_constraint: dict[str, str] | None = None,
    target_platforms: list[model_types.TargetPlatform] | None = None,
    python_version: str | None = None,
    task: model_types.Task = model_types.Task.UNKNOWN,
    prefer_pip_for_automatic_dependencies: bool | None = None,
    **kwargs: Any,
) -> Generator["ModelMetadata", None, None]:
    """Create a generator for model metadata object. Use generator to ensure correct register and unregister for
        cloudpickle.

    Args:
        model_dir_path: Path to the directory containing the model to be packed.
        name: Name of the model.
        model_type: Type of the model.
        signatures: Signatures of the model. If None, it will be inferred after the model meta is created.
            Defaults to None.
        function_properties: Dict mapping function names to a dict of properties, mapping property key to value.
        metadata: User provided key-value metadata of the model. Defaults to None.
        code_paths: List of paths to additional codes that needs to be packed with. Defaults to None.
        ext_modules: List of names of modules that need to be pickled with the model. Defaults to None.
        conda_dependencies: List of conda requirements for running the model. Defaults to None.
        pip_requirements: List of pip Python packages requirements for running the model. Defaults to None.
        artifact_repository_map: A dict mapping from package channel to artifact repository name (e.g.
            {'pip': 'snowflake.snowpark.pypi_shared_repository'}).
        resource_constraint: Mapping of resource constraint keys and values, e.g. {"architecture": "x86"}.
        target_platforms: List of target platforms to run the model.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        task: The task of the Model Version. It is an enum class Task with values TABULAR_REGRESSION,
            TABULAR_BINARY_CLASSIFICATION, TABULAR_MULTI_CLASSIFICATION, TABULAR_RANKING, or UNKNOWN. By default,
            it is set to Task.UNKNOWN and may be overridden by inferring from the Model Object.
        prefer_pip_for_automatic_dependencies: When set (e.g. by Registry after reconciliation), use this for
            routing automatic packaging dependencies to pip vs conda. When ``None``, derive via
            :func:`~snowflake.ml.model._packager.model_env.model_env.resolve_prefer_pip_for_automatic_dependencies`.
        **kwargs: Dict of attributes and values of the metadata. Used when loading from file.

    Raises:
        ValueError: Raised when the code path contains reserved file or directory, or destination conflicts.
        FileNotFoundError: Raised when a code path does not exist.

    Yields:
        A model metadata object.
    """
    model_dir_path = os.path.normpath(model_dir_path)
    save_options: dict[str, Any] = dict(kwargs)
    embed_local_ml_library = kwargs.pop("embed_local_ml_library", False)

    if embed_local_ml_library:
        # Use the last one which is loaded first, that is mean, it is loaded from site-packages.
        # We could make sure that user does not overwrite our library with their code follow the same naming.
        snowml_path, snowml_start_path = file_utils.get_package_path(_SNOWFLAKE_ML_PKG_NAME, strategy="last")
        if os.path.isdir(snowml_start_path):
            path_to_copy = snowml_path
        # If the package is zip-imported, then the path will be `../path_to_zip.zip/snowflake/ml`
        # It is not a valid path in fact and we need to get the path to the zip file to verify it.
        elif os.path.isfile(snowml_start_path):
            extract_root = tempfile.mkdtemp()
            with zipfile.ZipFile(os.path.abspath(snowml_start_path), mode="r", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.extractall(path=extract_root)
            path_to_copy = os.path.join(extract_root, *(_SNOWFLAKE_ML_PKG_NAME.split(".")))
        else:
            raise ValueError("`snowflake.ml` is imported via a way that embedding local ML library is not supported.")

    resolved_prefer_pip_for_automatic_dependencies = (
        prefer_pip_for_automatic_dependencies
        if prefer_pip_for_automatic_dependencies is not None
        else model_env.resolve_prefer_pip_for_automatic_dependencies(
            target_platforms=target_platforms,
            conda_dependencies=conda_dependencies,
            force_conda_defaults=False,
        )
    )
    env = _create_env_for_model_metadata(
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        artifact_repository_map=artifact_repository_map,
        resource_constraint=resource_constraint,
        python_version=python_version,
        embed_local_ml_library=embed_local_ml_library,
        prefer_pip_for_automatic_dependencies=resolved_prefer_pip_for_automatic_dependencies,
        target_platforms=target_platforms,
    )

    if embed_local_ml_library:
        env.snowpark_ml_version = (
            f"{snowml_version.VERSION}+{file_utils.hash_directory(path_to_copy, excluded_files=['experimental'])}"
        )

    # Persist full method_options
    method_options: dict[str, dict[str, Any]] = kwargs.pop("method_options", {})
    case_sensitive = kwargs.pop("case_sensitive", None)
    relax_version = bool(kwargs.pop("relax_version", False))
    model_meta = ModelMetadata(
        name=name,
        env=env,
        metadata=metadata,
        model_type=model_type,
        signatures=signatures,
        function_properties=function_properties,
        task=task,
        method_options=method_options,
        case_sensitive=case_sensitive,
        target_platforms=target_platforms,
        relax_version=relax_version,
        save_options=save_options,
    )

    code_dir_path = os.path.join(model_dir_path, MODEL_CODE_DIR)
    if code_paths:
        os.makedirs(code_dir_path, exist_ok=True)

    if code_paths:
        # Resolve all code paths and check for conflicts
        resolved_paths: list[tuple[str, str]] = []  # (source, destination_relative)
        for code_path in code_paths:
            if isinstance(code_path, CodePath):
                source, dest_relative = code_path._resolve()
            else:
                # String path: keep existing behavior
                source = os.path.normpath(os.path.abspath(code_path))
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Code path '{code_path}' does not exist (resolved to {source}).")
                dest_relative = os.path.basename(source)
            resolved_paths.append((source, dest_relative))

        # Check for destination conflicts
        seen: dict[str, str] = {}
        for source, dest in resolved_paths:
            if dest in seen:
                raise ValueError(
                    f"Destination path conflict: '{dest}' is targeted by both '{seen[dest]}' and '{source}'."
                )
            seen[dest] = source

        # Copy files
        for source, dest_relative in resolved_paths:
            # Prevent reserved name conflicts
            dest_name = dest_relative.split(os.sep)[0] if os.sep in dest_relative else dest_relative
            if (os.path.isfile(source) and os.path.splitext(dest_name)[0] == _SNOWFLAKE_PKG_NAME) or (
                os.path.isdir(source) and dest_name == _SNOWFLAKE_PKG_NAME
            ):
                raise ValueError("`snowflake` is a reserved name and you cannot contain that into code path.")

            parent_dir = (
                os.path.join(code_dir_path, os.path.dirname(dest_relative))
                if os.path.dirname(dest_relative)
                else code_dir_path
            )
            os.makedirs(parent_dir, exist_ok=True)
            file_utils.copy_file_or_tree(source, parent_dir)

    try:
        imported_modules = []
        if ext_modules:
            registered_modules = cloudpickle.list_registry_pickle_by_value()
            for mod in ext_modules:
                if mod.__name__ not in registered_modules:
                    cloudpickle.register_pickle_by_value(mod)
                    imported_modules.append(mod)
        yield model_meta
        model_meta.save(model_dir_path)
    finally:
        for mod in imported_modules:
            cloudpickle.unregister_pickle_by_value(mod)


def _is_client_model_spec_v2_write_enabled() -> bool:
    try:
        return platform_capabilities.PlatformCapabilities.get_instance().is_client_model_spec_v2_write_enabled()
    except snowpark_exceptions.SnowparkSessionException:
        # Packaging can run without a Snowpark session; keep writing the legacy spec in that case.
        return False


def _workspace_relative_path(path: str) -> str:
    """Express a path that is relative to the model directory as relative to the workspace root."""
    model_dir_prefix = f"{_MODEL_DIR_IN_WORKSPACE}/"
    if path.startswith(model_dir_prefix):
        return path
    return f"{model_dir_prefix}{path}"


def _create_env_for_model_metadata(
    *,
    conda_dependencies: list[str] | None = None,
    pip_requirements: list[str] | None = None,
    artifact_repository_map: dict[str, str] | None = None,
    resource_constraint: dict[str, str] | None = None,
    python_version: str | None = None,
    embed_local_ml_library: bool = False,
    prefer_pip_for_automatic_dependencies: bool = False,
    target_platforms: list[model_types.TargetPlatform] | None = None,
) -> model_env.ModelEnv:
    env = model_env.ModelEnv(
        prefer_pip_for_automatic_dependencies=prefer_pip_for_automatic_dependencies,
        target_platforms=target_platforms,
    )

    # Mypy doesn't like getter and setter have different types. See python/mypy #3004
    env.conda_dependencies = conda_dependencies
    env.pip_requirements = pip_requirements
    env.artifact_repository_map = artifact_repository_map
    env.resource_constraint = resource_constraint
    env.python_version = python_version
    env.snowpark_ml_version = snowml_version.VERSION

    requirements_to_add = _PACKAGING_REQUIREMENTS

    if embed_local_ml_library:
        env.include_if_absent(
            [
                model_env.ModelDependency(requirement=dep, pip_name=requirements.Requirement(dep).name)
                for dep in requirements_to_add
            ],
            check_local_version=True,
        )
    else:
        env.include_if_absent(
            [
                model_env.ModelDependency(requirement=dep, pip_name=requirements.Requirement(dep).name)
                for dep in requirements_to_add + [env_utils.SNOWPARK_ML_PKG_NAME]
            ],
            check_local_version=True,
        )
    return env


def load_code_path(model_dir_path: str) -> None:
    """Load custom code in the code path into memory.

    Args:
        model_dir_path: Path to the directory containing the model to be loaded.

    """
    code_path = os.path.join(model_dir_path, MODEL_CODE_DIR)
    if os.path.exists(code_path):
        if code_path in sys.path:
            sys.path.remove(code_path)
        sys.path.insert(0, code_path)


class ModelMetadataTelemetryDict(TypedDict):
    model_name: Required[str]
    framework_type: Required[model_types.SupportedModelHandlerType]
    number_of_functions: Required[int]
    is_packaged_pip_only: Required[bool]


class ModelMetadata:
    """Model metadata for Snowflake native model packaged model.

    Attributes:
        name: Name of the model.
        model_type: Type of the model.
        env: ModelEnv object containing all environment related object
        models: Dict of model blob metadata
        signatures: A dict mapping from target function name to input and output signatures.
        function_properties: A dict mapping function names to dict mapping function property key to value.
        metadata: User provided key-value metadata of the model. Defaults to None.
        creation_timestamp: Unix timestamp when the model metadata is created.
        user_files: Dict mapping subdirectories to extra artifact file paths for files to include in the model.
        task: Model task like TABULAR_REGRESSION, tabular_classification, timeseries_forecasting etc.
    """

    def telemetry_metadata(self) -> ModelMetadataTelemetryDict:
        is_packaged_pip_only = self._packaged_env_dict is not None and model_meta_schema.env_dict_is_packaged_pip_only(
            self._packaged_env_dict
        )
        return ModelMetadataTelemetryDict(
            model_name=self.name,
            framework_type=self.model_type,
            number_of_functions=len(self.signatures.keys()),
            is_packaged_pip_only=is_packaged_pip_only,
        )

    def __init__(
        self,
        *,
        name: str,
        env: model_env.ModelEnv,
        model_type: model_types.SupportedModelHandlerType,
        runtimes: dict[str, model_runtime.ModelRuntime] | None = None,
        signatures: dict[str, model_signature.ModelSignature] | None = None,
        function_properties: dict[str, dict[str, Any]] | None = None,
        user_files: dict[str, list[str]] | None = None,
        metadata: dict[str, str] | None = None,
        creation_timestamp: str | None = None,
        min_snowpark_ml_version: str | None = None,
        models: dict[str, model_blob_meta.ModelBlobMeta] | None = None,
        original_metadata_version: str | None = model_meta_schema.MODEL_METADATA_VERSION,
        task: model_types.Task = model_types.Task.UNKNOWN,
        explain_algorithm: model_meta_schema.ModelExplainAlgorithm | None = None,
        method_options: dict[str, dict[str, Any]] | None = None,
        case_sensitive: bool | None = None,
        sample_input_file_paths: dict[str, str] | None = None,
        packaged_env_dict: model_meta_schema.ModelEnvDict | None = None,
        target_platforms: list[model_types.TargetPlatform] | None = None,
        relax_version: bool = False,
        save_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        # Distinguish omitted signatures=None from an explicit empty dict. Both become {}.
        self._signatures_supplied = signatures is not None
        self.signatures: dict[str, model_signature.ModelSignature] = signatures if signatures is not None else dict()
        self.function_properties = function_properties or {}
        self.user_files = user_files
        self.metadata = metadata
        self.model_type = model_type
        self.env = env
        self.creation_timestamp = (
            creation_timestamp if creation_timestamp else str(datetime.now(timezone.utc).replace(tzinfo=None))
        )
        self._min_snowpark_ml_version = version.parse(
            min_snowpark_ml_version
            if min_snowpark_ml_version
            else model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION
        )

        self.models: dict[str, model_blob_meta.ModelBlobMeta] = dict()
        if models:
            self.models = models

        self._runtimes = runtimes

        self.original_metadata_version = original_metadata_version

        self.task: model_types.Task = task
        self.explain_algorithm: model_meta_schema.ModelExplainAlgorithm | None = explain_algorithm
        self.method_options: dict[str, dict[str, Any]] = method_options or {}
        self.case_sensitive: bool | None = case_sensitive
        # Maps method name -> filename of the captured sample input data row,
        # written into model.yaml as the per-method ``sample_input_file_path`` field.
        self.sample_input_file_paths: dict[str, str] = sample_input_file_paths or {}
        self._packaged_env_dict = packaged_env_dict
        self.target_platforms = target_platforms
        self.relax_version = relax_version
        # Options the model was saved with; used to resolve per-function settings exactly as the manifest does.
        self.save_options: Mapping[str, Any] = save_options or {}
        # Whether save() wrote the version 2.0 specification. The manifest follows this so that the two files
        # can never describe different specification versions.
        self.wrote_v2_spec = False

        self._lazy_hf_upload: huggingface_lazy_uploader.LazyHFUpload | None = None

    def function_type(self, function_name: str) -> str | None:
        """Return the SQL function type recorded for a model function.

        Version 2.0 specifications store this as ``FUNCTION`` or ``TABLE_FUNCTION``.
        Legacy metadata leaves it unset.

        Args:
            function_name: Name of the model function.

        Returns:
            The SQL function type, or None when the metadata does not record one.
        """
        return cast(str | None, self.method_options.get(function_name, {}).get("function_type"))

    @property
    def min_snowpark_ml_version(self) -> str:
        return self._min_snowpark_ml_version.base_version

    @min_snowpark_ml_version.setter
    def min_snowpark_ml_version(self, min_snowpark_ml_version: str) -> None:
        parsed_min_snowpark_ml_version = version.parse(min_snowpark_ml_version)
        self._min_snowpark_ml_version = max(self._min_snowpark_ml_version, parsed_min_snowpark_ml_version)

    @property
    def runtimes(self) -> dict[str, model_runtime.ModelRuntime]:
        if self._runtimes and "cpu" in self._runtimes:
            return self._runtimes
        runtimes = {
            "cpu": model_runtime.ModelRuntime("cpu", self.env, is_warehouse=False),
        }
        if self.env.cuda_version:
            runtimes.update({"gpu": model_runtime.ModelRuntime("gpu", self.env, is_warehouse=False, is_gpu=True)})
        return runtimes

    def save(self, model_dir_path: str) -> None:
        """Save the model metadata

        Raises:
            RuntimeError: Raised when the metadata is not ready to save

        Args:
            model_dir_path: Path to the directory containing the model to be loaded.
        """
        if self.name not in self.models:
            raise RuntimeError("The meta data is not ready to save.")
        if (not self.signatures) and self.model_type != "peft_adapter":
            raise RuntimeError("The meta data is not ready to save.")

        if _is_client_model_spec_v2_write_enabled():
            self._save_v2(model_dir_path)
            return

        model_yaml_path = os.path.join(model_dir_path, MODEL_METADATA_FILE)
        model_dict = model_meta_schema.ModelMetadataDict(
            {
                "creation_timestamp": self.creation_timestamp,
                "env": self._save_packaged_env(pathlib.Path(model_dir_path)),
                "runtimes": {
                    runtime_name: runtime.save(pathlib.Path(model_dir_path), default_channel_override="conda-forge")
                    for runtime_name, runtime in self.runtimes.items()
                },
                "metadata": self.metadata,
                "model_type": self.model_type,
                "models": {model_name: blob.to_dict() for model_name, blob in self.models.items()},
                "name": self.name,
                "signatures": {
                    func_name: self._serialize_signature(func_name, sig) for func_name, sig in self.signatures.items()
                },
                "version": model_meta_schema.MODEL_METADATA_VERSION,
                "min_snowpark_ml_version": self.min_snowpark_ml_version,
                "task": self.task.value,
                "explainability": (
                    model_meta_schema.ExplainabilityMetadataDict(algorithm=self.explain_algorithm.value)
                    if self.explain_algorithm
                    else None
                ),
                "function_properties": self.function_properties,
                "method_options": self.method_options,
            }
        )
        if self.case_sensitive is not None:
            model_dict["case_sensitive"] = self.case_sensitive
        with open(model_yaml_path, "w", encoding="utf-8") as out:
            yaml.SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[method-assign]
            yaml.safe_dump(model_dict, stream=out, default_flow_style=False)

    def _save_v2(self, model_dir_path: str) -> None:
        """Write the unified version 2.0 extension specification.

        The packaged environment remains at ``model/env`` for client loading,
        but is intentionally not declared in the extension specification.

        Args:
            model_dir_path: Directory that will contain ``model.yaml``.

        Raises:
            ValueError: If two methods resolve to the same SQL name.
        """
        model_dir = pathlib.Path(model_dir_path)
        self._save_packaged_env(model_dir)

        warehouse_runtime = model_runtime.ModelRuntime(
            name=_WAREHOUSE_RUNTIME_NAME,
            env=self.env,
            imports=[f"{_MODEL_DIR_IN_WORKSPACE}/"],
            is_warehouse=True,
        )
        if self.relax_version:
            warehouse_runtime.runtime_env.relax_version()
        warehouse_runtime_dict = warehouse_runtime.save(
            model_dir, default_channel_override=env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
        )

        serving_environments: dict[str, dict[str, Any]] = {
            PYTHON_ENV_NAME: self._serving_environment(
                warehouse_runtime,
                warehouse_runtime_dict,
                environment_type="python",
                # Only declare pip requirements when the model has some, so that the warehouse environment
                # matches what the version 1.0 manifest declares for the same model.
                include_pip=len(self.env.pip_requirements) > 0,
            )
        }
        target_platform_values = {
            platform.value if isinstance(platform, model_types.TargetPlatform) else str(platform)
            for platform in self.target_platforms or []
        }
        supports_spcs = model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value in target_platform_values
        if supports_spcs:
            cpu_runtime = self.runtimes["cpu"]
            cpu_runtime_dict = cpu_runtime.save(
                model_dir, default_channel_override=env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
            )
            serving_environments["cpu_env"] = self._serving_environment(
                cpu_runtime,
                cpu_runtime_dict,
                environment_type="python",
            )
            if self.env.cuda_version:
                gpu_runtime = self.runtimes["gpu"]
                gpu_runtime_dict = gpu_runtime.save(
                    model_dir, default_channel_override=env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
                )
                serving_environments["gpu_env"] = self._serving_environment(
                    gpu_runtime,
                    gpu_runtime_dict,
                    environment_type="python",
                )

        function_env = [PYTHON_ENV_NAME]
        if supports_spcs:
            function_env.append("gpu_env" if self.env.cuda_version else "cpu_env")

        set_volatility = (
            platform_capabilities.PlatformCapabilities.get_instance().is_set_module_functions_volatility_from_manifest()
        )
        functions: dict[str, dict[str, Any]] = {}
        resolved_method_names: list[str] = []
        for function_name, signature in self.signatures.items():
            is_partitioned_function = bool(
                self.function_properties.get(function_name, {}).get(
                    model_meta_schema.FunctionProperties.PARTITIONED.value, False
                )
            )
            resolved_options = model_method_options.get_model_method_options_from_options(
                cast(model_types.ModelSaveOption, self.save_options),
                function_name,
                self.model_type,
                is_partitioned_function,
            )
            try:
                resolved_method_names.append(
                    sql_identifier.SqlIdentifier(
                        function_name, case_sensitive=bool(resolved_options.get("case_sensitive", False))
                    ).resolved()
                )
            except ValueError as err:
                raise ValueError(
                    f"Your target method {function_name} cannot be resolved as valid SQL identifier. "
                    "Try specifying `case_sensitive` as True."
                ) from err
            properties: dict[str, Any] = {"method": function_name}
            if is_partitioned_function:
                properties["is_partition"] = True
            if function_name == "explain":
                properties["is_explain"] = True
            if resolved_options.get("case_sensitive"):
                properties["case_sensitive"] = True
            max_batch_size = (
                self.save_options.get("method_options", {})
                .get(function_name, {})
                .get("max_batch_size", self.save_options.get("max_batch_size"))
            )
            if max_batch_size is not None:
                properties["max_batch_size"] = max_batch_size

            function_spec: dict[str, Any] = {
                "env": function_env,
                "type": resolved_options["function_type"],
                "handler": f"functions.{function_name}.infer",
                "signature": self._serialize_signature(function_name, signature),
                "properties": properties,
            }
            volatility = resolved_options.get("volatility")
            if set_volatility and volatility is not None:
                function_spec["volatility"] = volatility.name
            functions[function_name] = function_spec

        method_name_counter = collections.Counter(resolved_method_names)
        duplicate_method_names = [name for name, count in method_name_counter.items() if count > 1]
        if duplicate_method_names:
            raise ValueError(
                f"Found duplicate method named resolved as {', '.join(duplicate_method_names)} in the model. "
                "This might because you have methods with same letters but different cases. "
                "In this case, set case_sensitive as True for those methods to distinguish them."
            )

        spec: dict[str, Any] = {
            "version": _MODEL_EXTENSION_SPEC_VERSION,
            "model": {
                "name": self.name,
                "type": "USER_MODEL",
                "framework": self.model_type,
                "task": self.task.value,
                "client": {
                    "name": "snowflake-ml-python",
                    "version": snowml_version.VERSION,
                    "min_version": self.min_snowpark_ml_version,
                },
                "details": {
                    "models": self._v2_model_blobs(),
                },
            },
            "serving": {"env": serving_environments, "functions": functions},
        }
        if self.target_platforms:
            spec["model"]["target_platforms"] = [platform.value for platform in self.target_platforms]

        model_extension_spec.ModelExtensionSpecV2._validate(spec)

        with (model_dir / MODEL_METADATA_FILE).open("w", encoding="utf-8") as out:
            yaml.SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[method-assign]
            yaml.safe_dump(spec, stream=out, default_flow_style=False)
        self.wrote_v2_spec = True

    @staticmethod
    def _serving_environment(
        runtime: model_runtime.ModelRuntime,
        runtime_dict: model_meta_schema.ModelRuntimeDict,
        *,
        environment_type: str,
        include_pip: bool = True,
    ) -> dict[str, Any]:
        dependencies = runtime_dict["dependencies"]
        environment: dict[str, Any] = {
            "type": environment_type,
            "python_version": runtime.runtime_env.python_version,
            "snowpark_ml_version": runtime.runtime_env.snowpark_ml_version,
            "imports": [_workspace_relative_path(path) for path in runtime_dict["imports"]],
        }
        conda_path = dependencies.get("conda")
        if conda_path:
            environment["conda"] = _workspace_relative_path(conda_path)
        pip_path = dependencies.get("pip")
        if pip_path and include_pip:
            environment["pip"] = _workspace_relative_path(pip_path)
        if dependencies.get("artifact_repository_map"):
            environment["artifact_repository_map"] = dependencies["artifact_repository_map"]
        if runtime.runtime_env.cuda_version:
            environment["cuda_version"] = runtime.runtime_env.cuda_version
        if runtime_dict["resource_constraint"]:
            environment["resource_constraint"] = runtime_dict["resource_constraint"]
        return environment

    def _v2_model_blobs(self) -> dict[str, dict[str, Any]]:
        """Serialize blobs for the version 2.0 spec, including per-blob explainability."""
        blobs: dict[str, dict[str, Any]] = {}
        for model_name, blob in self.models.items():
            blob_dict: dict[str, Any] = {
                "model_type": blob.model_type.upper(),
                "handler_version": blob.handler_version,
                "path": blob.path,
                "artifacts": dict(blob.artifacts),
                "options": dict(blob.options),
            }
            if self.explain_algorithm:
                blob_dict["explainability"] = {
                    "algorithm": self.explain_algorithm.value,
                    "background_data": (f"models/explain_artifacts/{model_name}_background_data.pqt"),
                }
            blobs[model_name] = blob_dict
        return blobs

    def _save_packaged_env(self, model_dir_path: pathlib.Path) -> model_meta_schema.ModelEnvDict:
        """Write env files and record the packaged env dict for downstream consumers."""
        saved_env = self.env.save_as_dict(
            model_dir_path,
            default_channel_override=env_utils.SNOWFLAKE_CONDA_CHANNEL_URL,
            is_gpu=bool(self.env.cuda_version),
        )
        self._packaged_env_dict = saved_env
        return saved_env

    def _serialize_signature(self, func_name: str, sig: model_signature.ModelSignature) -> dict[str, Any]:
        """Serialize one method signature, optionally injecting ``sample_input_file_path``."""
        sig_dict = sig.to_dict()
        sample_path = self.sample_input_file_paths.get(func_name)
        if sample_path:
            sig_dict["sample_input_file_path"] = sample_path
        return sig_dict

    @staticmethod
    def _validate_model_metadata(loaded_meta: Any) -> model_meta_schema.ModelMetadataDict:
        if not isinstance(loaded_meta, dict):
            raise ValueError(f"Read ill-formatted model metadata, should be a dict, received {type(loaded_meta)}")

        original_loaded_meta_version = loaded_meta.get("version", None)
        if not original_loaded_meta_version:
            raise ValueError("Unable to get the version of the metadata file.")

        loaded_meta = migrator_plans.migrate_metadata(loaded_meta)

        loaded_meta_min_snowpark_ml_version = loaded_meta.get("min_snowpark_ml_version", None)
        if not loaded_meta_min_snowpark_ml_version or (
            version.parse(loaded_meta_min_snowpark_ml_version) > version.parse(snowml_version.VERSION)
        ):
            raise RuntimeError(
                f"The minimal version required to load the model is {loaded_meta_min_snowpark_ml_version}, "
                f"while current version of Snowpark ML library is {snowml_version.VERSION}."
            )
        validated = model_meta_schema.ModelMetadataDict(
            creation_timestamp=loaded_meta["creation_timestamp"],
            env=loaded_meta["env"],
            runtimes=loaded_meta.get("runtimes", None),
            metadata=loaded_meta.get("metadata", None),
            model_type=loaded_meta["model_type"],
            models=loaded_meta["models"],
            name=loaded_meta["name"],
            signatures=loaded_meta["signatures"],
            version=original_loaded_meta_version,
            min_snowpark_ml_version=loaded_meta_min_snowpark_ml_version,
            task=loaded_meta.get("task", model_types.Task.UNKNOWN.value),
            explainability=loaded_meta.get("explainability", None),
            function_properties=loaded_meta.get("function_properties", {}),
            method_options=loaded_meta.get("method_options", {}),
        )
        if loaded_meta.get("case_sensitive") is not None:
            validated["case_sensitive"] = bool(loaded_meta["case_sensitive"])
        return validated

    @classmethod
    def load(cls, model_dir_path: str) -> "ModelMetadata":
        """Load models for a directory. Model is initially loaded normally. If additional codes are included when
        packed, the code path is added to system path to be imported with highest priority.

        Args:
            model_dir_path: Path to the directory containing the model to be loaded.

        Returns:
            A model metadata object.
        """
        model_yaml_path = os.path.join(model_dir_path, MODEL_METADATA_FILE)
        with open(model_yaml_path, encoding="utf-8") as f:
            loaded_meta = yaml.safe_load(f.read())

        # Version 2.0 model.yaml is a model extension spec and is parsed by ModelMetadataV2.
        if isinstance(loaded_meta, dict) and str(loaded_meta.get("version")) == _MODEL_EXTENSION_SPEC_VERSION:
            return ModelMetadataV2.from_spec(model_dir_path, loaded_meta)

        model_dict = cls._validate_model_metadata(loaded_meta)

        signatures = {
            func_name: model_signature.ModelSignature.from_dict(sig)
            for func_name, sig in model_dict["signatures"].items()
        }
        sample_input_file_paths = {
            func_name: sig["sample_input_file_path"]
            for func_name, sig in model_dict["signatures"].items()
            if sig.get("sample_input_file_path")
        }
        models = {name: model_blob_meta.ModelBlobMeta(**blob_meta) for name, blob_meta in model_dict["models"].items()}
        packaged_env_dict = model_dict["env"]
        env = model_env.ModelEnv()
        env.load_from_dict(pathlib.Path(model_dir_path), packaged_env_dict)

        runtimes: dict[str, model_runtime.ModelRuntime] | None
        if model_dict.get("runtimes", None):
            runtimes = {
                name: model_runtime.ModelRuntime.load(pathlib.Path(model_dir_path), name, env, runtime_dict)
                for name, runtime_dict in model_dict["runtimes"].items()
            }
        else:
            runtimes = None

        explanation_algorithm_dict = model_dict.get("explainability", None)
        explanation_algorithm = None
        if explanation_algorithm_dict:
            explanation_algorithm = model_meta_schema.ModelExplainAlgorithm(explanation_algorithm_dict["algorithm"])

        return cls(
            name=model_dict["name"],
            model_type=model_dict["model_type"],
            env=env,
            runtimes=runtimes,
            signatures=signatures,
            metadata=model_dict.get("metadata", None),
            creation_timestamp=model_dict["creation_timestamp"],
            min_snowpark_ml_version=model_dict["min_snowpark_ml_version"],
            models=models,
            original_metadata_version=model_dict["version"],
            task=model_types.Task(model_dict.get("task", model_types.Task.UNKNOWN.value)),
            explain_algorithm=explanation_algorithm,
            function_properties=model_dict.get("function_properties", {}),
            method_options=model_dict.get("method_options", {}),
            case_sensitive=model_dict.get("case_sensitive"),
            sample_input_file_paths=sample_input_file_paths,
            packaged_env_dict=packaged_env_dict,
        )


class ModelMetadataV2(ModelMetadata):
    """Metadata loaded from a version 2.0 model extension spec.

    SQL function type stays on the function. Legacy metadata leaves it unset.
    """

    def __init__(self, *, function_types: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._function_types = function_types

    def function_type(self, function_name: str) -> str:
        """Return the SQL function type declared by a version 2.0 function.

        The type is required on every function and is validated before this metadata is built.
        A function present in ``signatures`` is present here.

        Args:
            function_name: Name of the model function.

        Returns:
            ``FUNCTION`` or ``TABLE_FUNCTION``.
        """
        return self._function_types[function_name]

    def save(self, model_dir_path: str) -> None:
        """Version 2.0 metadata is load-only.

        Args:
            model_dir_path: Unused. Present so the signature matches ``ModelMetadata.save``.

        Raises:
            RuntimeError: Always. A version 2.0 spec is not written back as legacy metadata.
        """
        raise RuntimeError("Saving a version 2.0 model specification is not supported.")

    @classmethod
    def from_spec(cls, model_dir_path: str, raw_spec: Mapping[str, Any]) -> "ModelMetadataV2":
        """Build metadata from a parsed version 2.0 model extension spec.

        Args:
            model_dir_path: Directory that contains ``model.yaml`` and env files.
            raw_spec: Parsed ``model.yaml`` document.

        Returns:
            Metadata for the version 2.0 spec.
        """
        parsed = model_meta_v2.parse(model_dir_path, raw_spec)
        return cls(
            name=parsed.name,
            model_type=parsed.model_type,
            env=parsed.env,
            runtimes=parsed.runtimes,
            signatures=parsed.signatures,
            metadata=parsed.metadata,
            min_snowpark_ml_version=parsed.min_snowpark_ml_version,
            models=parsed.models,
            original_metadata_version=parsed.original_metadata_version,
            task=parsed.task,
            explain_algorithm=parsed.explain_algorithm,
            function_properties=parsed.function_properties,
            method_options=parsed.method_options,
            case_sensitive=parsed.case_sensitive,
            sample_input_file_paths=parsed.sample_input_file_paths,
            packaged_env_dict=parsed.packaged_env_dict,
            function_types=parsed.function_types,
        )
