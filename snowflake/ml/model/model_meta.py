import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Generator, List, Optional

import cloudpickle
import yaml

from snowflake.ml.model import env, type_spec, util


@contextmanager
def create(
    *,
    model_dir_path: str,
    name: str,
    python_version: str,
    model_type: str,
    input_spec: type_spec.TypeSpec,
    output_spec: type_spec.TypeSpec,
    metadata: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    pip_requirements: Optional[List[str]] = None,
    **kwargs: Any
) -> Generator["ModelMetadata", None, None]:
    model_meta = ModelMetadata(
        name=name,
        python_version=python_version,
        metadata=metadata,
        model_type=model_type,
        pip_requirements=pip_requirements,
        input_spec=input_spec,
        output_spec=output_spec,
        **kwargs
    )
    if code_paths:
        code_dir_path = os.path.join(model_dir_path, "code")
        os.makedirs(code_dir_path, exist_ok=True)
        for code_path in code_paths:
            util.copy_file_or_tree(code_path, code_dir_path)
    try:
        imported_modules = []
        if ext_modules:
            registered_modules = cloudpickle.list_registry_pickle_by_value()
            for mod in ext_modules:
                if mod.__name__ not in registered_modules:
                    cloudpickle.register_pickle_by_value(mod)
                    imported_modules.append(mod)
        yield model_meta
        model_meta.save_yaml(os.path.join(model_dir_path, "model.yaml"))
        env.generate_env_files(model_dir_path, model_meta.pip_requirements)
    finally:
        for mod in imported_modules:
            cloudpickle.unregister_pickle_by_value(mod)


def load_model_metadata(model_dir_path: str) -> "ModelMetadata":
    meta = ModelMetadata.load(model_dir_path)
    # TODO: Move to common handling.
    code_path = os.path.join(model_dir_path, "code")
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

    TODO: Handle pip requirements here.
    """

    def __init__(
        self,
        *,
        name: str,
        model_type: str,
        python_version: str,
        input_spec: type_spec.TypeSpec,
        output_spec: type_spec.TypeSpec,
        metadata: Optional[Dict[str, Any]] = None,
        pip_requirements: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        self.name = name
        self._input_spec = input_spec
        self._output_spec = output_spec
        self.metadata = metadata
        self.utc_time_created = str(datetime.utcnow())
        self.model_type = model_type
        self.python_version = python_version
        self._pip_requirements = pip_requirements if pip_requirements else []
        self.models: Dict[str, Any] = dict()
        self.__dict__.update(kwargs)

    @property
    def pip_requirements(self) -> List[str]:
        return self._pip_requirements

    @property
    def input_spec(self) -> type_spec.TypeSpec:
        return self._input_spec

    @input_spec.setter
    def input_spec(self, v: type_spec.TypeSpec) -> None:
        self._input_spec = v

    @property
    def output_spec(self) -> type_spec.TypeSpec:
        return self._output_spec

    @output_spec.setter
    def output_spec(self, v: type_spec.TypeSpec) -> None:
        self._output_spec = v

    def to_dict(self) -> Dict[str, Any]:
        self._validate()
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        res["schema"] = {"input_spec": self._input_spec.to_dict(), "output_spec": self._output_spec.to_dict()}
        return res

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> "ModelMetadata":
        model_dict["input_spec"] = type_spec.TypeSpec.from_dict(model_dict["schema"]["input_spec"])
        model_dict["output_spec"] = type_spec.TypeSpec.from_dict(model_dict["schema"]["output_spec"])
        return cls(**model_dict)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as out:
            yaml.safe_dump(self.to_dict(), stream=out, default_flow_style=False)

    def _validate(self) -> None:
        assert self._input_spec is not None, "Input spec has to be set."
        assert self._output_spec is not None, "Output spec has to be set."
        assert self.model_type is not None, "Model type has to be set."

    @classmethod
    def load(cls, path: str) -> "ModelMetadata":
        path = os.path.join(path, "model.yaml")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f.read()))
