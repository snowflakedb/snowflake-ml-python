import dataclasses
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Generator, List, Optional, Tuple

import cloudpickle
import yaml
from packaging import version

from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml.model import _env, model_signature

MANIFEST_VERSION = 1
MANIFEST_LANGUAGE = "python"
MANIFEST_KIND = "model"
MODEL_METADATA_VERSION = 1
_BASIC_DEPENDENCIES = [
    "pandas",
    "pyyaml",
    "typing-extensions",
    "cloudpickle",
    "anyio",
    "snowflake-snowpark-python",
]


@dataclasses.dataclass
class _ModelBlobMetadata:
    """Dataclass to store metadata of an individual model blob (sub-model) in the packed model.

    Attributes:
        name: The name to refer the sub-model.
        model_type: The type of the model and handler to use.
        path: Path to the picked model file. It is a relative path from the model blob directory.
        artifacts: Optional, used in custom model to show the mapping between artifact name and relative path
            from the model blob directory.
    """

    name: str
    model_type: str
    path: str
    artifacts: Dict[str, str] = dataclasses.field(default_factory=dict)


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

    Yields:
        A model metadata object.
    """
    model_dir_path = os.path.normpath(model_dir_path)

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
    if code_paths:
        code_dir_path = os.path.join(model_dir_path, ModelMetadata.MODEL_CODE_DIR)
        os.makedirs(code_dir_path, exist_ok=True)
        for code_path in code_paths:
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
        the code path is added to system path to be imported and overwrites those modules with the same name that has
        been imported.

    Args:
        model_dir_path: Path to the directory containing the model to be loaded.

    Returns:
        A model metadata object.
    """
    model_dir_path = os.path.normpath(model_dir_path)

    meta = ModelMetadata.load_model_metadata(model_dir_path)
    code_path = os.path.join(model_dir_path, ModelMetadata.MODEL_CODE_DIR)
    if os.path.exists(code_path):
        sys.path = [code_path] + sys.path
        modules = [
            p.stem
            for p in Path(code_path).rglob("*.py")
            if p.is_file() and p.name != "__init__.py" and p.name != "__main__.py"
        ]
        for module in modules:
            sys.modules.pop(module, None)
    return meta


class ModelMetadata:
    """Model metadata for Snowflake native model packaged model.

    Attributes:
        name: Name of the model.
        model_type: Type of the model.
        creation_timestamp: Unix timestamp when the model metadata is created.
        python_version: String 'major.minor.patchlevel' showing the python version where the model runs.
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
        self._metadata = metadata
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

        self._include_if_absent([(dep, dep) for dep in _BASIC_DEPENDENCIES])

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

    @property
    def metadata(self) -> Dict[str, str]:
        """User provided key-value metadata of the model."""
        return self._metadata if self._metadata else {}

    def _include_if_absent(self, pkgs: List[Tuple[str, str]]) -> None:
        conda_names, pip_names = tuple(zip(*pkgs))
        pip_reqs = env_utils.validate_pip_requirement_string_list(list(pip_names))

        for conda_name, pip_req in zip(conda_names, pip_reqs):
            req_to_add = env_utils.get_local_installed_version_of_pip_package(pip_req)
            req_to_add.name = conda_name
            for added_pip_req in self._pip_requirements:
                if added_pip_req.name == pip_req.name:
                    warnings.warn(
                        (
                            f"Basic dependency {conda_name} specified from PIP requirements."
                            + " This may prevent model deploying to Snowflake Warehouse."
                        ),
                        category=UserWarning,
                    )
            try:
                env_utils.append_conda_dependency(self._conda_dependencies, ("", req_to_add))
            except env_utils.DuplicateDependencyError:
                pass
            except env_utils.DuplicateDependencyInMultipleChannelsError:
                warnings.warn(
                    (
                        f"Basic dependency {conda_name} specified from non-Snowflake channel."
                        + " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    category=UserWarning,
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
        return cls(**model_dict)

    def save_manifest(self, path: str, conda_file_path: str, pip_file_path: str) -> None:
        """Save the model manifest file in the model directory.

        Args:
            path: The path of the directory to write a manifest file in it.
            conda_file_path: The path to conda env file.
            pip_file_path: The path to pip requirements file.
        """
        manifest_yaml_path = os.path.join(path, ModelMetadata.MANIFEST_FILE)
        manifest_data = {
            "metadata": self.metadata,
            "signatures": {method: sig.to_dict(as_sql_type=True) for method, sig in self.signatures.items()},
            "language": MANIFEST_LANGUAGE,
            "kind": MANIFEST_KIND,
            "env": {"conda": os.path.relpath(conda_file_path, path), "pip": os.path.relpath(pip_file_path, path)},
            "version": MANIFEST_VERSION,
        }
        with open(manifest_yaml_path, "w") as out:
            yaml.safe_dump(manifest_data, stream=out, default_flow_style=False)

    @classmethod
    def load_manifest(cls, path: str) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str]]:
        """Load the model metadata from the model metadata yaml file in the model directory.

        Args:
            path: The path of the directory to read the metadata yaml file in it.

        Raises:
            ValueError: raised when manifest file structure is incorrect.
            NotImplementedError: raised when version is not found or unsupported in manifest file.
            NotImplementedError: raised when language is not found or unsupported in manifest file.
            NotImplementedError: raised when kind is not found or unsupported in manifest file.

        Returns:
            A tuple of optional metadata, conda env file path and pip requirements file path.
        """
        manifest_yaml_path = os.path.join(path, ModelMetadata.MANIFEST_FILE)
        with open(manifest_yaml_path) as f:
            manifest_data = yaml.safe_load(f.read())

        if not isinstance(manifest_data, dict):
            raise ValueError("Incorrect manifest data found.")

        manifest_version = manifest_data.get("version", None)
        if not manifest_version or manifest_version != MANIFEST_VERSION:
            raise NotImplementedError("Unknown or unsupported manifest file found.")

        manifest_language = manifest_data.get("language", None)
        if not manifest_language or manifest_language != MANIFEST_LANGUAGE:
            raise NotImplementedError("Unknown or unsupported language found.")

        manifest_kind = manifest_data.get("kind", None)
        if not manifest_kind or manifest_kind != MANIFEST_KIND:
            raise NotImplementedError("Unknown or unsupported packaging kind found.")

        metadata = manifest_data.get("metadata", None)
        conda_file_path = manifest_data.get("env", dict()).get("conda", None)
        pip_file_path = manifest_data.get("env", dict()).get("pip", None)
        return metadata, conda_file_path, pip_file_path

    def save_model_metadata(self, path: str) -> None:
        """Save the model metadata as a yaml file in the model directory.

        Args:
            path: The path of the directory to write a yaml file in it.
        """
        model_yaml_path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(model_yaml_path, "w") as out:
            yaml.safe_dump({**self.to_dict(), "version": MODEL_METADATA_VERSION}, stream=out, default_flow_style=False)

        env_dir_path = os.path.join(path, ModelMetadata.ENV_DIR)
        os.makedirs(env_dir_path, exist_ok=True)

        conda_file_path = _env.save_conda_env_file(env_dir_path, self._conda_dependencies, self.python_version)
        pip_file_path = _env.save_requirements_file(env_dir_path, self._pip_requirements)

        self.save_manifest(path, conda_file_path=conda_file_path, pip_file_path=pip_file_path)

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
        metadata, conda_file_path, pip_file_path = ModelMetadata.load_manifest(path)

        env_dir_path = os.path.join(path, ModelMetadata.ENV_DIR)
        if not conda_file_path:
            conda_file_path = os.path.join(env_dir_path, _env._CONDA_ENV_FILE_NAME)

        if not pip_file_path:
            pip_file_path = os.path.join(env_dir_path, _env._REQUIREMENTS_FILE_NAME)

        model_yaml_path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(model_yaml_path) as f:
            loaded_mata = yaml.safe_load(f.read())

        loaded_mata_version = loaded_mata.get("version", None)
        if not loaded_mata_version or loaded_mata_version != MODEL_METADATA_VERSION:
            raise NotImplementedError("Unknown or unsupported model metadata file found.")

        meta = ModelMetadata.from_dict(loaded_mata)
        meta._metadata = metadata

        meta._conda_dependencies, python_version = _env.load_conda_env_file(os.path.join(path, conda_file_path))
        if python_version:
            meta.python_version = python_version
        meta._pip_requirements = _env.load_requirements_file(os.path.join(path, pip_file_path))
        return meta
