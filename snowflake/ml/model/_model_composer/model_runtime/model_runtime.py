import copy
import pathlib
from typing import List, Optional

from packaging import requirements

from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_runtime import _runtime_requirements
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_meta import model_meta as model_meta_api
from snowflake.snowpark import session

_UDF_INFERENCE_DEPENDENCIES = _runtime_requirements.REQUIREMENTS


class ModelRuntime:
    """Class to represent runtime in a model, which controls the runtime and version, imports and dependencies.

    Attributes:
        model_meta: Model Metadata.
        runtime_env: ModelEnv object representing the actual environment when deploying. The environment is based on
            the environment from the packaged model with additional dependencies required to deploy.
        imports: List of files to be imported in the created functions. At least packed model should be imported.
            If the required Snowpark ML library is not available in the server-side, we will automatically pack the
            local version as well as "snowflake-ml-python.zip" and added into the imports.
    """

    RUNTIME_DIR_REL_PATH = "runtimes"

    def __init__(
        self,
        session: session.Session,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        imports: Optional[List[pathlib.PurePosixPath]] = None,
    ) -> None:
        self.name = name
        self.model_meta = model_meta
        self.runtime_env = copy.deepcopy(self.model_meta.env)
        self.imports = imports or []

        snowml_pkg_spec = f"{env_utils.SNOWPARK_ML_PKG_NAME}=={self.runtime_env.snowpark_ml_version}"
        if self.runtime_env._snowpark_ml_version.local:
            self.embed_local_ml_library = True
        else:
            snowml_server_availability = (
                len(
                    env_utils.get_matched_package_versions_in_information_schema(
                        session=session,
                        reqs=[requirements.Requirement(snowml_pkg_spec)],
                        python_version=snowml_env.PYTHON_VERSION,
                    ).get(env_utils.SNOWPARK_ML_PKG_NAME, [])
                )
                >= 1
            )
            self.embed_local_ml_library = not snowml_server_availability

        if self.embed_local_ml_library:
            self.runtime_env.include_if_absent(
                [
                    model_env.ModelDependency(requirement=dep, pip_name=requirements.Requirement(dep).name)
                    for dep in _UDF_INFERENCE_DEPENDENCIES
                ],
            )
        else:
            self.runtime_env.include_if_absent(
                [
                    model_env.ModelDependency(requirement=dep, pip_name=requirements.Requirement(dep).name)
                    for dep in _UDF_INFERENCE_DEPENDENCIES + [snowml_pkg_spec]
                ],
            )

    def save(self, workspace_path: pathlib.Path) -> model_manifest_schema.ModelRuntimeDict:
        runtime_base_path = workspace_path / ModelRuntime.RUNTIME_DIR_REL_PATH / self.name
        runtime_base_path.mkdir(parents=True, exist_ok=True)

        if self.embed_local_ml_library:
            snowpark_ml_lib_path = runtime_base_path / "snowflake-ml-python.zip"
            file_utils.zip_python_package(str(snowpark_ml_lib_path), "snowflake.ml")
            snowpark_ml_lib_rel_path = pathlib.PurePosixPath(
                snowpark_ml_lib_path.relative_to(workspace_path).as_posix()
            )
            self.imports.append(snowpark_ml_lib_rel_path)

        env_dict = self.runtime_env.save_as_dict(runtime_base_path)
        return model_manifest_schema.ModelRuntimeDict(
            language="PYTHON",
            version=self.runtime_env.python_version,
            imports=list(map(str, self.imports)),
            dependencies=model_manifest_schema.ModelRuntimeDependenciesDict(
                conda=(runtime_base_path / env_dict["conda"]).relative_to(workspace_path).as_posix()
            ),
        )
