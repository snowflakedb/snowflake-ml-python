import dataclasses
import importlib
import os
import sys
import warnings
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime
from types import ModuleType
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, cast

import cloudpickle
import yaml
from packaging import version

from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml.model import (
    _core_requirements,
    _env,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame

MODEL_METADATA_VERSION = 1
_BASIC_DEPENDENCIES = _core_requirements.REQUIREMENTS
_SNOWFLAKE_PKG_NAME = "snowflake"
_SNOWFLAKE_ML_PKG_NAME = f"{_SNOWFLAKE_PKG_NAME}.ml"
_DEFAULT_CUDA_VERSION = "11.7"

Dependency = namedtuple("Dependency", ["conda_name", "pip_name"])


@dataclasses.dataclass
class _ModelBlobMetadata:
    """Dataclass to store metadata of an individual model blob (sub-model) in the packed model.

    Attributes:
        name: The name to refer the sub-model.
        model_type: The type of the model and handler to use.
        path: Path to the picked model file. It is a relative path from the model blob directory.
        artifacts: Optional, used in custom model to show the mapping between artifact name and relative path
            from the model blob directory.
        options: Optional, used for some model specific metadata storage
    """

    name: str
    model_type: str
    path: str
    artifacts: Dict[str, str] = dataclasses.field(default_factory=dict)
    options: Dict[str, str] = dataclasses.field(default_factory=dict)


@contextmanager
def _create_model_metadata(
    *,
    model_dir_path: str,
    name: str,
    model_type: str,
    signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
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
    # Use the last one which is loaded first, that is mean, it is loaded from site-packages.
    # We could make sure that user does not overwrite our library with their code follow the same naming.
    snowml_path = list(importlib.import_module(_SNOWFLAKE_ML_PKG_NAME).__path__)[-1]
    if embed_local_ml_library:
        kwargs["local_ml_library_version"] = f"{snowml_env.VERSION}+{file_utils.hash_directory(snowml_path)}"

    model_meta = ModelMetadata(
        name=name,
        metadata=metadata,
        model_type=model_type,
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        python_version=python_version,
        signatures=signatures,
        **kwargs,
    )

    code_dir_path = os.path.join(model_dir_path, ModelMetadata.MODEL_CODE_DIR)
    if embed_local_ml_library or code_paths:
        os.makedirs(code_dir_path, exist_ok=True)

    if embed_local_ml_library:
        snowml_path_in_code = os.path.join(code_dir_path, _SNOWFLAKE_PKG_NAME)
        os.makedirs(snowml_path_in_code, exist_ok=True)
        file_utils.copy_file_or_tree(snowml_path, snowml_path_in_code)

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
        model_meta.save_model_metadata(model_dir_path)
    finally:
        for mod in imported_modules:
            cloudpickle.unregister_pickle_by_value(mod)


def _load_model_metadata(model_dir_path: str) -> "ModelMetadata":
    """Load models for a directory. Model is initially loaded normally. If additional codes are included when packed,
        the code path is added to system path to be imported with highest priority.

    Args:
        model_dir_path: Path to the directory containing the model to be loaded.

    Returns:
        A model metadata object.
    """
    model_dir_path = os.path.normpath(model_dir_path)
    meta = ModelMetadata.load_model_metadata(model_dir_path)
    return meta


def _load_code_path(model_dir_path: str) -> None:
    """Load custom code in the code path into memory.

    Args:
        model_dir_path: Path to the directory containing the model to be loaded.

    """
    model_dir_path = os.path.normpath(model_dir_path)
    code_path = os.path.join(model_dir_path, ModelMetadata.MODEL_CODE_DIR)
    if os.path.exists(code_path):
        if code_path in sys.path:
            sys.path.remove(code_path)
        sys.path.insert(0, code_path)
        module_names = file_utils.get_all_modules(code_path)
        # If the module_name starts with snowflake, then do not replace it.
        # When deploying, we would add them beforehand.
        # When in the local, they should not be added. We already prevent user from overwriting us.
        module_names = [
            module_name
            for module_name in module_names
            if not (module_name.startswith(f"{_SNOWFLAKE_PKG_NAME}.") or module_name == _SNOWFLAKE_PKG_NAME)
        ]
        for module_name in module_names:
            actual_module = sys.modules.pop(module_name, None)
            if actual_module is not None:
                sys.modules[module_name] = importlib.import_module(module_name)

        assert code_path in sys.path
        sys.path.remove(code_path)


class ModelMetadata:
    """Model metadata for Snowflake native model packaged model.

    Attributes:
        name: Name of the model.
        model_type: Type of the model.
        creation_timestamp: Unix timestamp when the model metadata is created.
        python_version: String 'major.minor.patchlevel' showing the python version where the model runs.
        cuda_version: CUDA version to be used, if None then the model cannot be deployed to instance with GPUs.
    """

    MANIFEST_FILE = "MANIFEST"
    ENV_DIR = "env"
    MODEL_CODE_DIR = "code"
    MODEL_METADATA_FILE = "model.yaml"

    def __init__(
        self,
        *,
        name: str,
        model_type: str,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        metadata: Optional[Dict[str, str]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        python_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model metadata. Anything indicates in kwargs has higher priority.

        Args:
            name: Name of the model.
            model_type: Type of the model.
            signatures: A dict mapping from target function name to input and output signatures.
            metadata: User provided key-value metadata of the model. Defaults to None.
            conda_dependencies: List of conda requirements for running the model. Defaults to None.
            pip_requirements: List of pip Python packages requirements for running the model. Defaults to None.
            python_version: A string of python version where model is run. Used for user override. If specified as None,
                current version would be captured. Defaults to None.
            **kwargs: Dict of attributes and values of the metadata. Used when loading from file.

        Raises:
            ValueError: Raised when the user provided version string is invalid.
        """
        self.name = name
        self._signatures = signatures
        self.metadata = metadata
        self.creation_timestamp = str(datetime.utcnow())
        self.model_type = model_type
        self._models: Dict[str, _ModelBlobMetadata] = dict()
        if python_version:
            try:
                self.python_version = str(version.parse(python_version))
                # We might have more check here later.
            except version.InvalidVersion:
                raise ValueError(f"{python_version} is not a valid Python version.")
        else:
            self.python_version = snowml_env.PYTHON_VERSION

        self._conda_dependencies = env_utils.validate_conda_dependency_string_list(
            conda_dependencies if conda_dependencies else []
        )
        self._pip_requirements = env_utils.validate_pip_requirement_string_list(
            pip_requirements if pip_requirements else []
        )
        if "local_ml_library_version" in kwargs:
            self._include_if_absent([Dependency(conda_name=dep, pip_name=dep) for dep in _BASIC_DEPENDENCIES])
        else:
            self._include_if_absent(
                [Dependency(conda_name=dep, pip_name=dep) for dep in _BASIC_DEPENDENCIES + [env_utils._SNOWML_PKG_NAME]]
            )
        self._cuda_version: Optional[str] = None

        self.__dict__.update(kwargs)

    @property
    def pip_requirements(self) -> List[str]:
        """List of pip Python packages requirements for running the model."""
        return list(sorted(map(str, self._pip_requirements)))

    @property
    def conda_dependencies(self) -> List[str]:
        """List of conda channel and dependencies from that to run the model"""
        return sorted(
            f"{chan}::{str(req)}" if chan else str(req)
            for chan, reqs in self._conda_dependencies.items()
            for req in reqs
        )

    def _include_if_absent(self, pkgs: List[Dependency]) -> None:
        conda_reqs_str, pip_reqs_str = tuple(zip(*pkgs))
        pip_reqs = env_utils.validate_pip_requirement_string_list(list(pip_reqs_str))
        conda_reqs = env_utils.validate_conda_dependency_string_list(list(conda_reqs_str))

        for conda_req, pip_req in zip(conda_reqs[env_utils.DEFAULT_CHANNEL_NAME], pip_reqs):
            req_to_add = env_utils.get_local_installed_version_of_pip_package(pip_req)
            req_to_add.name = conda_req.name
            for added_pip_req in self._pip_requirements:
                if added_pip_req.name == pip_req.name:
                    warnings.warn(
                        (
                            f"Basic dependency {conda_req} specified from PIP requirements."
                            + " This may prevent model deploying to Snowflake Warehouse."
                        ),
                        category=UserWarning,
                    )
            try:
                env_utils.append_conda_dependency(
                    self._conda_dependencies, (env_utils.DEFAULT_CHANNEL_NAME, req_to_add)
                )
            except env_utils.DuplicateDependencyError:
                pass
            except env_utils.DuplicateDependencyInMultipleChannelsError:
                warnings.warn(
                    (
                        f"Basic dependency {conda_req.name} specified from non-Snowflake channel."
                        + " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    category=UserWarning,
                )

    @property
    def cuda_version(self) -> Optional[str]:
        return self._cuda_version

    @cuda_version.setter
    def cuda_version(self, _cuda_version: str) -> None:
        if not isinstance(_cuda_version, str):
            raise ValueError("Cannot set CUDA version as a non-str object.")
        if self._cuda_version is None:
            self._cuda_version = _cuda_version
        else:
            if self._cuda_version != _cuda_version:
                raise ValueError(
                    f"Different CUDA version {self._cuda_version} and {_cuda_version} found in the same model!"
                )

    @property
    def signatures(self) -> Dict[str, model_signature.ModelSignature]:
        """Signatures of the model.

        Raises:
            RuntimeError: Raised when the metadata is not ready to save

        Returns:
            Model signatures.
        """
        if self._signatures is None:
            raise RuntimeError("The meta data is not ready to save.")
        return self._signatures

    @property
    def models(self) -> Dict[str, _ModelBlobMetadata]:
        """Dict showing the mapping from sub-models' name to corresponding model blob metadata."""
        return self._models

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary.

        Raises:
            RuntimeError: Raised when the metadata is not ready to save

        Returns:
            A dict containing the information of the model metadata.
        """
        if self._signatures is None:
            raise RuntimeError("The meta data is not ready to save.")
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        res["signatures"] = {func_name: sig.to_dict() for func_name, sig in self._signatures.items()}
        res["models"] = {name: dataclasses.asdict(blob_meta) for name, blob_meta in self._models.items()}
        res["pip_requirements"] = self.pip_requirements
        res["conda_dependencies"] = self.conda_dependencies
        res["cuda_version"] = self._cuda_version
        return res

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> "ModelMetadata":
        """Deserialize from a dictionary.

        Args:
            model_dict: The dict where metadata is stored.

        Returns:
            A model metadata object created from the given dict.
        """
        model_dict["signatures"] = {
            func_name: model_signature.ModelSignature.from_dict(sig)
            for func_name, sig in model_dict.pop("signatures").items()
        }
        model_dict["_models"] = {
            name: _ModelBlobMetadata(**blob_meta) for name, blob_meta in model_dict.pop("models").items()
        }
        model_dict["_cuda_version"] = model_dict.pop("cuda_version", None)
        return cls(**model_dict)

    def save_model_metadata(self, path: str) -> None:
        """Save the model metadata as a yaml file in the model directory.

        Args:
            path: The path of the directory to write a yaml file in it.
        """
        model_yaml_path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(model_yaml_path, "w", encoding="utf-8") as out:
            yaml.safe_dump({**self.to_dict(), "version": MODEL_METADATA_VERSION}, stream=out, default_flow_style=False)

        env_dir_path = os.path.join(path, ModelMetadata.ENV_DIR)
        os.makedirs(env_dir_path, exist_ok=True)

        _env.save_conda_env_file(env_dir_path, self._conda_dependencies, self.python_version)
        _env.save_requirements_file(env_dir_path, self._pip_requirements)

    @classmethod
    def load_model_metadata(cls, path: str) -> "ModelMetadata":
        """Load the model metadata from the model metadata yaml file in the model directory.

        Args:
            path: The path of the directory to read the metadata yaml file in it.

        Raises:
            NotImplementedError: raised when version is not found or unsupported in metadata file.

        Returns:
            Loaded model metadata object.
        """
        model_yaml_path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(model_yaml_path, encoding="utf-8") as f:
            loaded_meta = yaml.safe_load(f.read())

        loaded_meta_version = loaded_meta.pop("version", None)
        if not loaded_meta_version or loaded_meta_version != MODEL_METADATA_VERSION:
            raise NotImplementedError("Unknown or unsupported model metadata file found.")

        meta = ModelMetadata.from_dict(loaded_meta)
        env_dir_path = os.path.join(path, ModelMetadata.ENV_DIR)
        meta._conda_dependencies, python_version = _env.load_conda_env_file(
            os.path.join(env_dir_path, _env._CONDA_ENV_FILE_NAME)
        )
        if python_version:
            meta.python_version = python_version
        meta._pip_requirements = _env.load_requirements_file(os.path.join(env_dir_path, _env._REQUIREMENTS_FILE_NAME))
        return meta


def _is_callable(model: model_types.SupportedModelType, method_name: str) -> bool:
    return callable(getattr(model, method_name, None))


def _validate_signature(
    model: model_types.SupportedRequireSignatureModelType,
    model_meta: ModelMetadata,
    target_methods: Sequence[str],
    sample_input: Optional[model_types.SupportedDataType],
    get_prediction_fn: Callable[[str, model_types.SupportedLocalDataType], model_types.SupportedLocalDataType],
) -> ModelMetadata:
    if model_meta._signatures is not None:
        _validate_target_methods(model, list(model_meta.signatures.keys()))
        return model_meta

    # In this case sample_input should be available, because of the check in save_model.
    assert (
        sample_input is not None
    ), "Model signature and sample input are None at the same time. This should not happen with local model."
    model_meta._signatures = {}
    trunc_sample_input = model_signature._truncate_data(sample_input)
    if isinstance(sample_input, SnowparkDataFrame):
        # Added because of Any from missing stubs.
        trunc_sample_input = cast(SnowparkDataFrame, trunc_sample_input)
        local_sample_input = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(trunc_sample_input)
    else:
        local_sample_input = trunc_sample_input
    for target_method in target_methods:
        predictions_df = get_prediction_fn(target_method, local_sample_input)
        sig = model_signature.infer_signature(sample_input, predictions_df)
        model_meta._signatures[target_method] = sig
    return model_meta


def _get_target_methods(
    model: model_types.SupportedModelType,
    target_methods: Optional[Sequence[str]],
    default_target_methods: Sequence[str],
) -> Sequence[str]:
    if target_methods is None:
        target_methods = [method_name for method_name in default_target_methods if _is_callable(model, method_name)]

    _validate_target_methods(model, target_methods)
    return target_methods


def _validate_target_methods(model: model_types.SupportedModelType, target_methods: Sequence[str]) -> None:
    for method_name in target_methods:
        if method_name not in target_methods:
            raise ValueError(f"Target method {method_name} does not exists.")
        if not _is_callable(model, method_name):
            raise ValueError(f"Target method {method_name} is not callable.")
