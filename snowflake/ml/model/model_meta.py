import dataclasses
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Generator, List, Optional

import cloudpickle
import yaml

from snowflake.ml._internal import file_utils
from snowflake.ml.model import _env, schema

MODEL_METADATA_VERSION = 1


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
    schema: schema.Schema,
    metadata: Optional[Dict[str, str]] = None,
    code_paths: Optional[List[str]] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    pip_requirements: Optional[List[str]] = None,
    **kwargs: Any
) -> Generator["ModelMetadata", None, None]:
    """Create a generator for model metadata object. Use generator to ensure correct register and unregister for
        cloudpickle.

    Args:
        model_dir_path: Path to the directory containing the model to be packed.
        name: Name of the model.
        model_type: Type of the model.
        schema: Schema of the model.
        metadata: User provided key-value metadata of the model. Defaults to None.
        code_paths: List of paths to additional codes that needs to be packed with. Defaults to None.
        ext_modules: List of names of modules that need to be pickled with the model. Defaults to None.
        pip_requirements: List of pip Python packages requirements for running the model. Defaults to None.
        **kwargs: Dict of attributes and values of the metadata. Used when loading from file.

    Yields:
        A model metadata object.
    """
    model_meta = ModelMetadata(
        name=name, metadata=metadata, model_type=model_type, pip_requirements=pip_requirements, schema=schema, **kwargs
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
        _env.generate_env_files(model_dir_path, model_meta.pip_requirements)
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
    """

    MODEL_CODE_DIR = "code"
    MODEL_METADATA_FILE = "model.yaml"

    def __init__(
        self,
        *,
        name: str,
        model_type: str,
        schema: schema.Schema,
        metadata: Optional[Dict[str, str]] = None,
        pip_requirements: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the model metadata. Anything indicates in kwargs has higher priority.

        Args:
            name: Name of the model.
            model_type: Type of the model.
            schema: Schema of the model.
            metadata: User provided key-value metadata of the model.. Defaults to None.
            pip_requirements: List of pip Python packages requirements for running the model. Defaults to None.
            **kwargs: Dict of attributes and values of the metadata. Used when loading from file.
        """
        self.name = name
        self._schema = schema
        self.metadata = metadata
        self.creation_timestamp = str(datetime.utcnow())
        self.model_type = model_type
        self._pip_requirements = pip_requirements if pip_requirements else []
        self._models: Dict[str, _ModelBlobMetadata] = dict()
        self.version = MODEL_METADATA_VERSION
        self.__dict__.update(kwargs)

    @property
    def pip_requirements(self) -> List[str]:
        """List of pip Python packages requirements for running the model."""
        return self._pip_requirements

    def _include_if_absent(self, pkgs: List[str]) -> None:
        for pkg in pkgs:
            if not any(req.lower().startswith(pkg.lower()) for req in self._pip_requirements):
                self._pip_requirements.append(pkg)

    @property
    def schema(self) -> schema.Schema:
        """Schema of the model."""
        return self._schema

    @property
    def models(self) -> Dict[str, _ModelBlobMetadata]:
        """Dict showing the mapping from sub-models' name to corresponding model blob metadata."""
        return self._models

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary.

        Returns:
            A dict containing the information of the model metadata.
        """
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        res["schema"] = self._schema.to_dict()
        res["models"] = {name: dataclasses.asdict(blob_meta) for name, blob_meta in self._models.items()}
        return res

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> "ModelMetadata":
        """Deserialize from a dictionary.

        Args:
            model_dict: The dict where metadata is stored.

        Returns:
            A model metadata object created from the given dict.
        """
        model_dict["schema"] = schema.Schema.from_dict(model_dict.pop("schema"))
        model_dict["_models"] = {
            name: _ModelBlobMetadata(**blob_meta) for name, blob_meta in model_dict.pop("models").items()
        }
        return cls(**model_dict)

    def save(self, path: str) -> None:
        """Save the model metadata as a yaml file in the model directory.

        Args:
            path: The path of the directory to write a yaml file in it.
        """
        path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(path, "w") as out:
            yaml.safe_dump(self.to_dict(), stream=out, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "ModelMetadata":
        """Load the model metadata from the model metadata yaml file in the model directory.

        Args:
            path: The path of the directory to read the metadata yaml file in it.

        Returns:
            Loaded model metadata object.
        """
        path = os.path.join(path, ModelMetadata.MODEL_METADATA_FILE)
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f.read()))
