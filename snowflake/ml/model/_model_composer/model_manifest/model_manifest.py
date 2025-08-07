import collections
import logging
import pathlib
from typing import TYPE_CHECKING, Optional, cast

import yaml

from snowflake.ml._internal import env_utils
from snowflake.ml.data import data_source
from snowflake.ml.model import type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import (
    constants,
    function_generator,
    model_method,
)
from snowflake.ml.model._model_composer.model_user_file import model_user_file
from snowflake.ml.model._packager.model_meta import (
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._packager.model_runtime import model_runtime

if TYPE_CHECKING:
    from snowflake.ml.experiment._experiment_info import ExperimentInfo

logger = logging.getLogger(__name__)


class ModelManifest:
    """Class to construct MANIFEST.yml file for Model

    Attributes:
        workspace_path: A local path where model related files should be dumped to.
        runtimes: A list of ModelRuntime objects managing the runtimes and environment in the MODEL object.
        methods: A list of ModelMethod objects managing the method we registered to the MODEL object.
        user_files: A list of ModelUserFile objects managing extra files uploaded to the workspace.
    """

    MANIFEST_FILE_REL_PATH = "MANIFEST.yml"
    _DEFAULT_RUNTIME_NAME = "python_runtime"

    def __init__(self, workspace_path: pathlib.Path) -> None:
        self.workspace_path = workspace_path

    def save(
        self,
        model_meta: model_meta_api.ModelMetadata,
        model_rel_path: pathlib.PurePosixPath,
        user_files: Optional[dict[str, list[str]]] = None,
        options: Optional[type_hints.ModelSaveOption] = None,
        data_sources: Optional[list[data_source.DataSource]] = None,
        experiment_info: Optional["ExperimentInfo"] = None,
        target_platforms: Optional[list[type_hints.TargetPlatform]] = None,
    ) -> None:
        assert options is not None, "ModelParameterReconciler should have set options with relax_version"
        relax_version = options["relax_version"]

        runtime_to_use = model_runtime.ModelRuntime(
            name=self._DEFAULT_RUNTIME_NAME,
            env=model_meta.env,
            imports=[str(model_rel_path) + "/"],
            is_gpu=False,
            is_warehouse=True,
        )
        if relax_version:
            runtime_to_use.runtime_env.relax_version()
            logger.info("Relaxing version constraints for dependencies in the model.")
            logger.info(f"Conda dependencies: {runtime_to_use.runtime_env.conda_dependencies}")
            logger.info(f"Pip requirements: {runtime_to_use.runtime_env.pip_requirements}")
            logger.info(f"artifact_repository_map: {runtime_to_use.runtime_env.artifact_repository_map}")
            logger.info(f"resource_constraint: {runtime_to_use.runtime_env.resource_constraint}")
        runtime_dict = runtime_to_use.save(
            self.workspace_path, default_channel_override=env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
        )

        self.function_generator = function_generator.FunctionGenerator(model_dir_rel_path=model_rel_path)
        self.methods: list[model_method.ModelMethod] = []

        for target_method in model_meta.signatures.keys():
            method = model_method.ModelMethod(
                model_meta=model_meta,
                target_method=target_method,
                runtime_name=self._DEFAULT_RUNTIME_NAME,
                function_generator=self.function_generator,
                is_partitioned_function=model_meta.function_properties.get(target_method, {}).get(
                    model_meta_schema.FunctionProperties.PARTITIONED.value, False
                ),
                wide_input=len(model_meta.signatures[target_method].inputs) > constants.SNOWPARK_UDF_INPUT_COL_LIMIT,
                options=model_method.get_model_method_options_from_options(options, target_method),
            )

            self.methods.append(method)

        self.user_files: list[model_user_file.ModelUserFile] = []

        if user_files is not None:
            for subdirectory, paths in user_files.items():
                for path in paths:
                    self.user_files.append(
                        model_user_file.ModelUserFile(pathlib.PurePosixPath(subdirectory), pathlib.Path(path))
                    )

        method_name_counter = collections.Counter([method.method_name for method in self.methods])
        dup_method_names = [k for k, v in method_name_counter.items() if v > 1]
        if dup_method_names:
            raise ValueError(
                f"Found duplicate method named resolved as {', '.join(dup_method_names)} in the model. "
                "This might because you have methods with same letters but different cases. "
                "In this case, set case_sensitive as True for those methods to distinguish them."
            )

        dependencies = model_manifest_schema.ModelRuntimeDependenciesDict(conda=runtime_dict["dependencies"]["conda"])

        # We only want to include pip dependencies file if there are any pip requirements.
        if len(model_meta.env.pip_requirements) > 0:
            dependencies["pip"] = runtime_dict["dependencies"]["pip"]

        if model_meta.env.artifact_repository_map:
            dependencies["artifact_repository_map"] = runtime_dict["dependencies"]["artifact_repository_map"]

        runtime = model_manifest_schema.ModelRuntimeDict(
            language="PYTHON",
            version=runtime_to_use.runtime_env.python_version,
            imports=runtime_dict["imports"],
            dependencies=dependencies,
        )

        if runtime_dict["resource_constraint"]:
            runtime["resource_constraint"] = runtime_dict["resource_constraint"]

        manifest_dict = model_manifest_schema.ModelManifestDict(
            manifest_version=model_manifest_schema.MODEL_MANIFEST_VERSION,
            runtimes={self._DEFAULT_RUNTIME_NAME: runtime},
            methods=[
                method.save(
                    self.workspace_path,
                    options=function_generator.get_function_generate_options_from_options(
                        options, method.target_method
                    ),
                )
                for method in self.methods
            ],
        )

        if self.user_files:
            manifest_dict["user_files"] = [user_file.save(self.workspace_path) for user_file in self.user_files]

        lineage_sources = self._extract_lineage_info(data_sources, experiment_info)
        if lineage_sources:
            manifest_dict["lineage_sources"] = lineage_sources

        if target_platforms:
            manifest_dict["target_platforms"] = [platform.value for platform in target_platforms]

        with (self.workspace_path / ModelManifest.MANIFEST_FILE_REL_PATH).open("w", encoding="utf-8") as f:
            # Anchors are not supported in the server, avoid that.
            yaml.SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[method-assign]
            yaml.safe_dump(manifest_dict, f)

    def load(self) -> model_manifest_schema.ModelManifestDict:
        with (self.workspace_path / ModelManifest.MANIFEST_FILE_REL_PATH).open("r", encoding="utf-8") as f:
            raw_input = yaml.safe_load(f)
        if not isinstance(raw_input, dict):
            raise ValueError(f"Read ill-formatted model MANIFEST, should be a dict, received {type(raw_input)}")

        original_loaded_manifest_version = raw_input.get("manifest_version", None)
        if not original_loaded_manifest_version:
            raise ValueError("Unable to get the version of the MANIFEST file.")

        res = cast(model_manifest_schema.ModelManifestDict, raw_input)

        return res

    def _extract_lineage_info(
        self,
        data_sources: Optional[list[data_source.DataSource]],
        experiment_info: Optional["ExperimentInfo"],
    ) -> list[model_manifest_schema.LineageSourceDict]:
        result = []
        if data_sources:
            for source in data_sources:
                if isinstance(source, data_source.DatasetInfo):
                    result.append(
                        model_manifest_schema.LineageSourceDict(
                            type=model_manifest_schema.LineageSourceTypes.DATASET.value,
                            entity=source.fully_qualified_name,
                            version=source.version,
                        )
                    )
                elif isinstance(source, data_source.DataFrameInfo):
                    result.append(
                        model_manifest_schema.LineageSourceDict(
                            type=model_manifest_schema.LineageSourceTypes.QUERY.value, entity=source.sql
                        )
                    )
        if experiment_info:
            result.append(
                model_manifest_schema.LineageSourceDict(
                    type=model_manifest_schema.LineageSourceTypes.EXPERIMENT.value,
                    entity=experiment_info.fully_qualified_name,
                    version=experiment_info.run_name,
                )
            )
        return result
