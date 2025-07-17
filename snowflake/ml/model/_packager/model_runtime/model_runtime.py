import copy
import pathlib
import warnings
from typing import Optional

from packaging import requirements

from snowflake.ml._internal import env_utils, file_utils
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.ml.model._packager.model_runtime import (
    _snowml_inference_alternative_requirements,
)

_SNOWML_INFERENCE_ALTERNATIVE_DEPENDENCIES = [
    str(env_utils.get_package_spec_with_supported_ops_only(requirements.Requirement(r)))
    for r in _snowml_inference_alternative_requirements.REQUIREMENTS
]

PACKAGES_NOT_ALLOWED_IN_WAREHOUSE = ["snowflake-connector-python", "pyarrow"]


class ModelRuntime:
    """Class to represent runtime in a model, which controls the runtime and version, imports and dependencies.

    Attributes:
        runtime_env: ModelEnv object representing the actual environment when deploying. The environment is based on
            the environment from the packaged model with additional dependencies required to deploy.
        imports: List of files to be imported in the created functions. At least packed model should be imported.
            If the required Snowpark ML library is not available in the server-side, we will automatically pack the
            local version as well as "snowflake-ml-python.zip" and added into the imports.
    """

    RUNTIME_DIR_REL_PATH = "runtimes"

    def __init__(
        self,
        name: str,
        env: model_env.ModelEnv,
        imports: Optional[list[str]] = None,
        is_warehouse: bool = False,
        is_gpu: bool = False,
        loading_from_file: bool = False,
    ) -> None:
        self.name = name
        self.runtime_env = copy.deepcopy(env)
        self.imports = imports or []
        self.is_gpu = is_gpu

        if loading_from_file:
            return

        snowml_pkg_spec = f"{env_utils.SNOWPARK_ML_PKG_NAME}=={self.runtime_env.snowpark_ml_version}"
        self.embed_local_ml_library = self.runtime_env._snowpark_ml_version.local

        additional_package = (
            _SNOWML_INFERENCE_ALTERNATIVE_DEPENDENCIES if self.embed_local_ml_library else [snowml_pkg_spec]
        )

        self.runtime_env.include_if_absent(
            [
                model_env.ModelDependency(requirement=dep, pip_name=requirements.Requirement(dep).name)
                for dep in additional_package
            ],
        )

        if is_warehouse and self.embed_local_ml_library:
            self.runtime_env.remove_if_present_conda(PACKAGES_NOT_ALLOWED_IN_WAREHOUSE)

        if is_gpu:
            self.runtime_env.generate_env_for_cuda()

    @property
    def runtime_rel_path(self) -> pathlib.PurePosixPath:
        return pathlib.PurePosixPath(ModelRuntime.RUNTIME_DIR_REL_PATH) / self.name

    def save(
        self, packager_path: pathlib.Path, default_channel_override: str = env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
    ) -> model_meta_schema.ModelRuntimeDict:
        runtime_base_path = packager_path / self.runtime_rel_path
        runtime_base_path.mkdir(parents=True, exist_ok=True)

        if getattr(self, "embed_local_ml_library", False):
            snowpark_ml_lib_path = runtime_base_path / "snowflake-ml-python.zip"
            file_utils.zip_python_package(str(snowpark_ml_lib_path), "snowflake.ml")
            snowpark_ml_lib_rel_path = pathlib.PurePosixPath(snowpark_ml_lib_path.relative_to(packager_path).as_posix())
            self.imports.append(str(snowpark_ml_lib_rel_path))

        self.runtime_env.conda_env_rel_path = self.runtime_rel_path / self.runtime_env.conda_env_rel_path
        self.runtime_env.pip_requirements_rel_path = self.runtime_rel_path / self.runtime_env.pip_requirements_rel_path

        env_dict = self.runtime_env.save_as_dict(
            packager_path, default_channel_override=default_channel_override, is_gpu=self.is_gpu
        )

        return model_meta_schema.ModelRuntimeDict(
            imports=list(map(str, self.imports)),
            dependencies=model_meta_schema.ModelRuntimeDependenciesDict(
                conda=env_dict["conda"],
                pip=env_dict["pip"],
                artifact_repository_map=(
                    env_dict["artifact_repository_map"] if env_dict.get("artifact_repository_map") is not None else {}
                ),
            ),
            resource_constraint=env_dict["resource_constraint"],
        )

    @staticmethod
    def load(
        packager_path: pathlib.Path,
        name: str,
        meta_env: model_env.ModelEnv,
        loaded_dict: model_meta_schema.ModelRuntimeDict,
    ) -> "ModelRuntime":
        env = model_env.ModelEnv()
        env.python_version = meta_env.python_version
        env.cuda_version = meta_env.cuda_version
        env.snowpark_ml_version = meta_env.snowpark_ml_version
        env.artifact_repository_map = meta_env.artifact_repository_map
        env.resource_constraint = meta_env.resource_constraint

        conda_env_rel_path = pathlib.PurePosixPath(loaded_dict["dependencies"]["conda"])
        pip_requirements_rel_path = pathlib.PurePosixPath(loaded_dict["dependencies"]["pip"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env.load_from_conda_file(packager_path / conda_env_rel_path)
            env.load_from_pip_file(packager_path / pip_requirements_rel_path)
        return ModelRuntime(name=name, env=env, imports=loaded_dict["imports"], loading_from_file=True)
