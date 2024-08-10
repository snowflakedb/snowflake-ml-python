import os
import pathlib
import sys
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from datetime import datetime
from types import ModuleType
from typing import Any, Dict, Generator, List, Optional, TypedDict

import cloudpickle
import yaml
from packaging import requirements, version
from typing_extensions import Required

from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import (
    _core_requirements,
    _packaging_requirements,
    model_blob_meta,
    model_meta_schema,
)
from snowflake.ml.model._packager.model_meta_migrator import migrator_plans
from snowflake.ml.model._packager.model_runtime import model_runtime

MODEL_METADATA_FILE = "model.yaml"
MODEL_CODE_DIR = "code"

_PACKAGING_CORE_DEPENDENCIES = [
    str(env_utils.get_package_spec_with_supported_ops_only(requirements.Requirement(r)))
    for r in _core_requirements.REQUIREMENTS
]  # Legacy Model only
_PACKAGING_REQUIREMENTS = [
    str(env_utils.get_package_spec_with_supported_ops_only(requirements.Requirement(r)))
    for r in _packaging_requirements.REQUIREMENTS
]  # New Model only
_SNOWFLAKE_PKG_NAME = "snowflake"
_SNOWFLAKE_ML_PKG_NAME = f"{_SNOWFLAKE_PKG_NAME}.ml"


@contextmanager
def create_model_metadata(
    *,
    model_dir_path: str,
    name: str,
    model_type: model_types.SupportedModelHandlerType,
    signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
    function_properties: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, str]] = None,
    code_paths: Optional[List[str]] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
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
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        **kwargs: Dict of attributes and values of the metadata. Used when loading from file.

    Raises:
        ValueError: Raised when the code path contains reserved file or directory.

    Yields:
        A model metadata object.
    """
    model_dir_path = os.path.normpath(model_dir_path)
    embed_local_ml_library = kwargs.pop("embed_local_ml_library", False)
    legacy_save = kwargs.pop("_legacy_save", False)
    if "relax_version" not in kwargs:
        warnings.warn(
            (
                "`relax_version` is not set and therefore defaulted to True. Dependency version constraints relaxed "
                "from ==x.y.z to >=x.y, <(x+1). To use specific dependency versions for compatibility, "
                "reproducibility, etc., set `options={'relax_version': False}` when logging the model."
            ),
            category=UserWarning,
            stacklevel=2,
        )
    relax_version = kwargs.pop("relax_version", True)

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

    env = _create_env_for_model_metadata(
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        python_version=python_version,
        embed_local_ml_library=embed_local_ml_library,
        legacy_save=legacy_save,
    )

    if embed_local_ml_library:
        env.snowpark_ml_version = f"{snowml_env.VERSION}+{file_utils.hash_directory(path_to_copy)}"

    model_meta = ModelMetadata(
        name=name,
        env=env,
        metadata=metadata,
        model_type=model_type,
        signatures=signatures,
        function_properties=function_properties,
    )

    code_dir_path = os.path.join(model_dir_path, MODEL_CODE_DIR)
    if (embed_local_ml_library and legacy_save) or code_paths:
        os.makedirs(code_dir_path, exist_ok=True)

    if embed_local_ml_library and legacy_save:
        snowml_path_in_code = os.path.join(code_dir_path, _SNOWFLAKE_PKG_NAME)
        os.makedirs(snowml_path_in_code, exist_ok=True)
        file_utils.copy_file_or_tree(path_to_copy, snowml_path_in_code)

    if code_paths:
        for code_path in code_paths:
            # This part is to prevent users from providing code following our naming and overwrite our code.
            if (
                os.path.isfile(code_path) and os.path.splitext(os.path.basename(code_path))[0] == _SNOWFLAKE_PKG_NAME
            ) or (os.path.isdir(code_path) and os.path.basename(code_path) == _SNOWFLAKE_PKG_NAME):
                raise ValueError("`snowflake` is a reserved name and you cannot contain that into code path.")
            file_utils.copy_file_or_tree(code_path, code_dir_path)

    try:
        imported_modules = []
        if ext_modules:
            registered_modules = cloudpickle.list_registry_pickle_by_value()
            for mod in ext_modules:
                if mod.__name__ not in registered_modules:
                    cloudpickle.register_pickle_by_value(mod)
                    imported_modules.append(mod)
        yield model_meta
        if relax_version:
            model_meta.env.relax_version()
        model_meta.save(model_dir_path)
    finally:
        for mod in imported_modules:
            cloudpickle.unregister_pickle_by_value(mod)


def _create_env_for_model_metadata(
    *,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    embed_local_ml_library: bool = False,
    legacy_save: bool = False,
) -> model_env.ModelEnv:
    env = model_env.ModelEnv()

    # Mypy doesn't like getter and setter have different types. See python/mypy #3004
    env.conda_dependencies = conda_dependencies  # type: ignore[assignment]
    env.pip_requirements = pip_requirements  # type: ignore[assignment]
    env.python_version = python_version  # type: ignore[assignment]
    env.snowpark_ml_version = snowml_env.VERSION

    requirements_to_add = _PACKAGING_CORE_DEPENDENCIES if legacy_save else _PACKAGING_REQUIREMENTS

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
        model_objective: Model objective like regression, classification etc.
    """

    def telemetry_metadata(self) -> ModelMetadataTelemetryDict:
        return ModelMetadataTelemetryDict(
            model_name=self.name,
            framework_type=self.model_type,
            number_of_functions=len(self.signatures.keys()),
        )

    def __init__(
        self,
        *,
        name: str,
        env: model_env.ModelEnv,
        model_type: model_types.SupportedModelHandlerType,
        runtimes: Optional[Dict[str, model_runtime.ModelRuntime]] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        function_properties: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        creation_timestamp: Optional[str] = None,
        min_snowpark_ml_version: Optional[str] = None,
        models: Optional[Dict[str, model_blob_meta.ModelBlobMeta]] = None,
        original_metadata_version: Optional[str] = model_meta_schema.MODEL_METADATA_VERSION,
        model_objective: Optional[model_meta_schema.ModelObjective] = model_meta_schema.ModelObjective.UNKNOWN,
        explain_algorithm: Optional[model_meta_schema.ModelExplainAlgorithm] = None,
    ) -> None:
        self.name = name
        self.signatures: Dict[str, model_signature.ModelSignature] = dict()
        if signatures:
            self.signatures = signatures
        self.function_properties = function_properties or {}
        self.metadata = metadata
        self.model_type = model_type
        self.env = env
        self.creation_timestamp = creation_timestamp if creation_timestamp else str(datetime.utcnow())
        self._min_snowpark_ml_version = version.parse(
            min_snowpark_ml_version
            if min_snowpark_ml_version
            else model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION
        )

        self.models: Dict[str, model_blob_meta.ModelBlobMeta] = dict()
        if models:
            self.models = models

        self._runtimes = runtimes

        self.original_metadata_version = original_metadata_version

        self.model_objective: model_meta_schema.ModelObjective = (
            model_objective or model_meta_schema.ModelObjective.UNKNOWN
        )
        self.explain_algorithm: Optional[model_meta_schema.ModelExplainAlgorithm] = explain_algorithm

    @property
    def min_snowpark_ml_version(self) -> str:
        return self._min_snowpark_ml_version.base_version

    @min_snowpark_ml_version.setter
    def min_snowpark_ml_version(self, min_snowpark_ml_version: str) -> None:
        parsed_min_snowpark_ml_version = version.parse(min_snowpark_ml_version)
        self._min_snowpark_ml_version = max(self._min_snowpark_ml_version, parsed_min_snowpark_ml_version)

    @property
    def runtimes(self) -> Dict[str, model_runtime.ModelRuntime]:
        if self._runtimes and "cpu" in self._runtimes:
            return self._runtimes
        runtimes = {
            "cpu": model_runtime.ModelRuntime("cpu", self.env),
        }
        if self.env.cuda_version:
            runtimes.update({"gpu": model_runtime.ModelRuntime("gpu", self.env, is_gpu=True)})
        return runtimes

    def save(self, model_dir_path: str) -> None:
        """Save the model metadata

        Raises:
            RuntimeError: Raised when the metadata is not ready to save

        Args:
            model_dir_path: Path to the directory containing the model to be loaded.
        """
        model_yaml_path = os.path.join(model_dir_path, MODEL_METADATA_FILE)

        if (not self.signatures) or (self.name not in self.models):
            raise RuntimeError("The meta data is not ready to save.")

        model_dict = model_meta_schema.ModelMetadataDict(
            {
                "creation_timestamp": self.creation_timestamp,
                "env": self.env.save_as_dict(
                    pathlib.Path(model_dir_path), default_channel_override=env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
                ),
                "runtimes": {
                    runtime_name: runtime.save(pathlib.Path(model_dir_path), default_channel_override="conda-forge")
                    for runtime_name, runtime in self.runtimes.items()
                },
                "metadata": self.metadata,
                "model_type": self.model_type,
                "models": {model_name: blob.to_dict() for model_name, blob in self.models.items()},
                "name": self.name,
                "signatures": {func_name: sig.to_dict() for func_name, sig in self.signatures.items()},
                "version": model_meta_schema.MODEL_METADATA_VERSION,
                "min_snowpark_ml_version": self.min_snowpark_ml_version,
                "model_objective": self.model_objective.value,
                "explainability": (
                    model_meta_schema.ExplainabilityMetadataDict(algorithm=self.explain_algorithm.value)
                    if self.explain_algorithm
                    else None
                ),
                "function_properties": self.function_properties,
            }
        )

        with open(model_yaml_path, "w", encoding="utf-8") as out:
            yaml.SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[method-assign]
            yaml.safe_dump(model_dict, stream=out, default_flow_style=False)

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
            version.parse(loaded_meta_min_snowpark_ml_version) > version.parse(snowml_env.VERSION)
        ):
            raise RuntimeError(
                f"The minimal version required to load the model is {loaded_meta_min_snowpark_ml_version},"
                f"while current version of Snowpark ML library is {snowml_env.VERSION}."
            )
        return model_meta_schema.ModelMetadataDict(
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
            model_objective=loaded_meta.get("model_objective", model_meta_schema.ModelObjective.UNKNOWN.value),
            explainability=loaded_meta.get("explainability", None),
            function_properties=loaded_meta.get("function_properties", {}),
        )

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

        model_dict = cls._validate_model_metadata(loaded_meta)

        signatures = {
            func_name: model_signature.ModelSignature.from_dict(sig)
            for func_name, sig in model_dict["signatures"].items()
        }
        models = {name: model_blob_meta.ModelBlobMeta(**blob_meta) for name, blob_meta in model_dict["models"].items()}
        env = model_env.ModelEnv()
        env.load_from_dict(pathlib.Path(model_dir_path), model_dict["env"])

        runtimes: Optional[Dict[str, model_runtime.ModelRuntime]]
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
            model_objective=model_meta_schema.ModelObjective(
                model_dict.get("model_objective", model_meta_schema.ModelObjective.UNKNOWN.value)
            ),
            explain_algorithm=explanation_algorithm,
            function_properties=model_dict.get("function_properties", {}),
        )
