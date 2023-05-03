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
        target_method: Optional, used in some non-custom model to specify the method that the user would use to predict.
        artifacts: Optional, used in custom model to show the mapping between artifact name and relative path
            from the model blob directory.
    """

    name: str
    model_type: str
    path: str
    target_method: str = "predict"
    artifacts: Dict[str, str] = dataclasses.field(default_factory=dict)


@contextmanager
def _create_model_metadata(
    *,
    model_dir_path: str,
    name: str,
    model_type: str,
    signature: Optional[model_signature.ModelSignature] = None,
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
        signature: Schema of the model. If None, it will be inferred after the model meta is created. Defaults to None.
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
    model_meta = ModelMetadata(
        name=name,
        metadata=metadata,
        model_type=model_type,
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        python_version=python_version,
        signature=signature,
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
        model_meta.save(model_dir_path)
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
    meta = ModelMetadata.load(model_dir_path)
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

    TODO(SNOW-773948): Handle pip requirements here.

    Attributes:
        name: Name of the model.
        model_type: Type of the model.
        metadata: User provided key-value metadata of the model.
        creation_timestamp: Unix timestamp when the model metadata is created.
        version: A version number of the yaml schema.
        python_version: String 'major.minor.patchlevel' showing the python version where the model runs.
    """

    MODEL_CODE_DIR = "code"
    MODEL_METADATA_FILE = "model.yaml"

    def __init__(
        self,
        *,
        name: str,
        model_type: str,
        signature: Optional[model_signature.ModelSignature] = None,
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
            signature: Signature of the model.
            metadata: User provided key-value metadata of the model.. Defaults to None.
            conda_dependencies: List of conda requirements for running the model. Defaults to None.
            pip_requirements: List of pip Python packages requirements for running the model. Defaults to None.
            python_version: A string of python version where model is run. Used for user override. If specified as None,
                current version would be captured. Defaults to None.
            **kwargs: Dict of attributes and values of the metadata. Used when loading from file.

        Raises:
            ValueError: Raised when the user provided version string is invalid.
        """
        self.name = name
        self._signature = signature
        self.metadata = metadata
        self.creation_timestamp = str(datetime.utcnow())
        self.model_type = model_type
        self._models: Dict[str, _ModelBlobMetadata] = dict()
        self.version = MODEL_METADATA_VERSION
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
        return sorted(f"{chan}::{str(req)}" for chan, reqs in self._conda_dependencies.items() for req in reqs)

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
                env_utils.append_conda_dependency(self._conda_dependencies, ("defaults", req_to_add))
            except env_utils.DuplicateDependencyError:
                pass
            except env_utils.DuplicateDependencyInMultipleChannelsError:
                warnings.warn(
                    (
                        f"Basic dependency {conda_name} specified from non-defaults channel."
                        + " This may prevent model deploying to Snowflake Warehouse."
                    ),
                    category=UserWarning,
                )

    @property
    def signature(self) -> model_signature.ModelSignature:
        """Signature of the model.

        Raises:
            RuntimeError: Raised when the metadata is not ready to save

        Returns:
            Model signature.
        """
        if self._signature is None:
            raise RuntimeError("The meta data is not ready to save.")
        return self._signature

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
        if self._signature is None:
            raise RuntimeError("The meta data is not ready to save.")
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        res["signature"] = self._signature.to_dict()
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
        model_dict["signature"] = model_signature.ModelSignature.from_dict(model_dict.pop("signature"))
        model_dict["_models"] = {
            name: _ModelBlobMetadata(**blob_meta) for name, blob_meta in model_dict.pop("models").items()
        }
        return cls(**model_dict)

    def save(self, path: str) -> None:
        """Save the model metadata as a yaml file in the model directory.

        Args:
            path: The path of the directory to write a yaml file in it.
        """
        model_yaml_path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(model_yaml_path, "w") as out:
            yaml.safe_dump(self.to_dict(), stream=out, default_flow_style=False)

        _env.save_conda_env_file(path, self._conda_dependencies, self.python_version)
        _env.save_requirements_file(path, self._pip_requirements)

    @classmethod
    def load(cls, path: str) -> "ModelMetadata":
        """Load the model metadata from the model metadata yaml file in the model directory.

        Args:
            path: The path of the directory to read the metadata yaml file in it.

        Returns:
            Loaded model metadata object.
        """
        model_yaml_path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(model_yaml_path) as f:
            meta = cls.from_dict(yaml.safe_load(f.read()))

        meta._conda_dependencies, python_version = _env.load_conda_env_file(path)
        if python_version:
            meta.python_version = python_version
        meta._pip_requirements = _env.load_requirements_file(path)
        return meta
