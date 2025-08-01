import pathlib
import tempfile
import uuid
import warnings
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib import parse

from absl import logging
from packaging import requirements

from snowflake import snowpark
from snowflake.ml import version as snowml_version
from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml._internal.lineage import lineage_utils
from snowflake.ml.data import data_source
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._model_composer.model_manifest import model_manifest
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    from snowflake.ml.experiment._experiment_info import ExperimentInfo


class ModelComposer:
    """Top-level class to construct contents in a MODEL object in SQL.

    Attributes:
        session: The Snowpark Session.
        stage_path: A stage path representing the base directory where the content of a MODEL object will exist.
        workspace_path: A local path which is the exact mapping to the `stage_path`

        manifest: A ModelManifest object managing the MANIFEST file generation.
        packager: A ModelPackager object managing the (un)packaging of a Snowflake Native Model in the MODEL object.

        _packager_workspace_path: A local path created from packager where it will dump all files there and ModelModel
        will zip it. This would not be required if we make directory import work.
    """

    MODEL_DIR_REL_PATH = "model"

    def __init__(
        self,
        session: Session,
        stage_path: str,
        *,
        statement_params: Optional[dict[str, Any]] = None,
        save_location: Optional[str] = None,
    ) -> None:
        self.session = session
        self.stage_path: Union[pathlib.PurePosixPath, parse.ParseResult] = None  # type: ignore[assignment]
        if stage_path.startswith("snow://"):
            # The stage path is a snowflake internal stage path
            self.stage_path = parse.urlparse(stage_path)
        else:
            # The stage path is a user stage path
            self.stage_path = pathlib.PurePosixPath(stage_path)

        # Set up workspace based on save_location if provided, otherwise use temporary directory
        self.save_location = save_location
        if save_location:
            # Use the save_location directory directly
            self._workspace_path = pathlib.Path(save_location)
            self._workspace_path.mkdir(exist_ok=True)
            # ensure that the directory is empty
            if any(self._workspace_path.iterdir()):
                raise ValueError(f"The directory {self._workspace_path} is not empty.")
            self._workspace = None

            self._packager_workspace_path = self._workspace_path / ModelComposer.MODEL_DIR_REL_PATH
            self._packager_workspace_path.mkdir(exist_ok=True)
            self._packager_workspace = None
        else:
            # Use a temporary directory
            self._workspace = tempfile.TemporaryDirectory()
            self._workspace_path = pathlib.Path(self._workspace.name)

            self._packager_workspace_path = self._workspace_path / ModelComposer.MODEL_DIR_REL_PATH
            self._packager_workspace_path.mkdir(exist_ok=True)

        self.packager = model_packager.ModelPackager(local_dir_path=str(self.packager_workspace_path))
        self.manifest = model_manifest.ModelManifest(workspace_path=self.workspace_path)

        self.model_file_rel_path = f"model-{uuid.uuid4().hex}.zip"

        self._statement_params = statement_params

    def __del__(self) -> None:
        if self._workspace:
            self._workspace.cleanup()

    @property
    def workspace_path(self) -> pathlib.Path:
        return self._workspace_path

    @property
    def packager_workspace_path(self) -> pathlib.Path:
        return self._packager_workspace_path

    @property
    def model_stage_path(self) -> str:
        if isinstance(self.stage_path, parse.ParseResult):
            model_file_path = (pathlib.PosixPath(self.stage_path.path) / self.model_file_rel_path).as_posix()
            new_url = parse.ParseResult(
                scheme=self.stage_path.scheme,
                netloc=self.stage_path.netloc,
                path=str(model_file_path),
                params=self.stage_path.params,
                query=self.stage_path.query,
                fragment=self.stage_path.fragment,
            )
            return str(parse.urlunparse(new_url))
        else:
            assert isinstance(self.stage_path, pathlib.PurePosixPath)
            return (self.stage_path / self.model_file_rel_path).as_posix()

    @property
    def model_local_path(self) -> str:
        return str(self.workspace_path / self.model_file_rel_path)

    def save(
        self,
        *,
        name: str,
        model: model_types.SupportedModelType,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        metadata: Optional[dict[str, str]] = None,
        conda_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        resource_constraint: Optional[dict[str, str]] = None,
        target_platforms: Optional[list[model_types.TargetPlatform]] = None,
        python_version: Optional[str] = None,
        user_files: Optional[dict[str, list[str]]] = None,
        ext_modules: Optional[list[ModuleType]] = None,
        code_paths: Optional[list[str]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        experiment_info: Optional["ExperimentInfo"] = None,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> model_meta.ModelMetadata:
        # set enable_explainability=False if the model is not runnable in WH or the target platforms include SPCS
        conda_dep_dict = env_utils.validate_conda_dependency_string_list(
            conda_dependencies if conda_dependencies else []
        )

        enable_explainability = None

        if options:
            enable_explainability = options.get("enable_explainability", None)

        # skip everything if user said False explicitly
        if enable_explainability is None or enable_explainability is True:
            is_warehouse_runnable = (
                not conda_dep_dict
                or all(
                    chan == env_utils.DEFAULT_CHANNEL_NAME or chan == env_utils.SNOWFLAKE_CONDA_CHANNEL_URL
                    for chan in conda_dep_dict
                )
            ) and (not pip_requirements)

            only_spcs = (
                target_platforms
                and len(target_platforms) == 1
                and model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES in target_platforms
            )
            if only_spcs or (not is_warehouse_runnable):
                # if only SPCS and user asked for explainability we fail
                if enable_explainability is True:
                    raise ValueError(
                        "`enable_explainability` cannot be set to True when the model is not runnable in WH "
                        "or the target platforms include SPCS."
                    )
                elif not options:  # explicitly set flag to false in these cases if not specified
                    options = model_types.BaseModelSaveOption()
                    options["enable_explainability"] = False
            elif (
                target_platforms
                and len(target_platforms) > 1
                and model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES in target_platforms
            ):  # if both then only available for WH
                if enable_explainability is True:
                    warnings.warn(
                        ("Explain function will only be available for model deployed to warehouse."),
                        category=UserWarning,
                        stacklevel=2,
                    )

        if not options:
            options = model_types.BaseModelSaveOption()

        if not snowpark_utils.is_in_stored_procedure() and target_platforms != [  # type: ignore[no-untyped-call]
            model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES  # no information schema check for SPCS-only models
        ]:
            snowml_matched_versions = env_utils.get_matched_package_versions_in_information_schema(
                self.session,
                reqs=[requirements.Requirement(f"{env_utils.SNOWPARK_ML_PKG_NAME}=={snowml_version.VERSION}")],
                python_version=python_version or snowml_env.PYTHON_VERSION,
                statement_params=self._statement_params,
            ).get(env_utils.SNOWPARK_ML_PKG_NAME, [])

            if len(snowml_matched_versions) < 1 and options.get("embed_local_ml_library", False) is False:
                logging.info(
                    f"Local snowflake-ml-python library has version {snowml_version.VERSION},"
                    " which is not available in the Snowflake server, embedding local ML library automatically."
                )
                options["embed_local_ml_library"] = True

        model_metadata: model_meta.ModelMetadata = self.packager.save(
            name=name,
            model=model,
            signatures=signatures,
            sample_input_data=sample_input_data,
            metadata=metadata,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            artifact_repository_map=artifact_repository_map,
            resource_constraint=resource_constraint,
            target_platforms=target_platforms,
            python_version=python_version,
            ext_modules=ext_modules,
            code_paths=code_paths,
            task=task,
            options=options,
        )
        assert self.packager.meta is not None

        self.manifest.save(
            model_meta=self.packager.meta,
            model_rel_path=pathlib.PurePosixPath(ModelComposer.MODEL_DIR_REL_PATH),
            options=options,
            user_files=user_files,
            data_sources=self._get_data_sources(model, sample_input_data),
            experiment_info=experiment_info,
            target_platforms=target_platforms,
        )

        file_utils.upload_directory_to_stage(
            self.session,
            local_path=self.workspace_path,
            stage_path=self.stage_path,
            statement_params=self._statement_params,
        )
        return model_metadata

    @staticmethod
    def load(
        workspace_path: pathlib.Path,
        *,
        meta_only: bool = False,
        options: Optional[model_types.ModelLoadOption] = None,
    ) -> model_packager.ModelPackager:
        mp = model_packager.ModelPackager(str(workspace_path / ModelComposer.MODEL_DIR_REL_PATH))
        mp.load(meta_only=meta_only, options=options)
        return mp

    def _get_data_sources(
        self, model: model_types.SupportedModelType, sample_input_data: Optional[model_types.SupportedDataType] = None
    ) -> Optional[list[data_source.DataSource]]:
        data_sources = lineage_utils.get_data_sources(model)
        if not data_sources and sample_input_data is not None:
            data_sources = lineage_utils.get_data_sources(sample_input_data)
            if not data_sources and isinstance(sample_input_data, snowpark.DataFrame):
                data_sources = [data_source.DataFrameInfo(sample_input_data.queries["queries"][-1])]
        return data_sources
