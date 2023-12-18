import glob
import pathlib
import tempfile
import zipfile
from types import ModuleType
from typing import Any, Dict, List, Optional

from absl import logging
from packaging import requirements

from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._model_composer.model_manifest import model_manifest
from snowflake.ml.model._packager import model_packager
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils


class ModelComposer:
    """Top-level class to construct contents in a MODEL object in SQL.

    Attributes:
        session: The Snowpark Session.
        stage_path: A stage path representing the base directory where the content of a MODEL object will exist.
        workspace_path: A local path which is the exact mapping to the `stage_path`

        manifest: A ModelManifest object managing the MANIFEST file generation.
        packager: A ModelPackager object managing the (un)packaging of a Snowflake Native Model in the MODEL object.

        _packager_workspace_path: A local path created from packager where it will dump all files there and ModelModel
        will zip it. This would not required if we make directory import work.
    """

    MODEL_FILE_REL_PATH = "model.zip"
    MODEL_DIR_REL_PATH = "model"

    def __init__(
        self,
        session: Session,
        stage_path: str,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session = session
        self.stage_path = pathlib.PurePosixPath(stage_path)

        self._workspace = tempfile.TemporaryDirectory()
        self._packager_workspace = tempfile.TemporaryDirectory()

        self.packager = model_packager.ModelPackager(local_dir_path=str(self._packager_workspace_path))
        self.manifest = model_manifest.ModelManifest(workspace_path=self.workspace_path)

        self._statement_params = statement_params

    def __del__(self) -> None:
        self._workspace.cleanup()
        self._packager_workspace.cleanup()

    @property
    def workspace_path(self) -> pathlib.Path:
        return pathlib.Path(self._workspace.name)

    @property
    def _packager_workspace_path(self) -> pathlib.Path:
        return pathlib.Path(self._packager_workspace.name)

    @property
    def model_stage_path(self) -> str:
        return (self.stage_path / ModelComposer.MODEL_FILE_REL_PATH).as_posix()

    @property
    def model_local_path(self) -> str:
        return str(self.workspace_path / ModelComposer.MODEL_FILE_REL_PATH)

    def save(
        self,
        *,
        name: str,
        model: model_types.SupportedModelType,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input: Optional[model_types.SupportedDataType] = None,
        metadata: Optional[Dict[str, str]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        python_version: Optional[str] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        code_paths: Optional[List[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> None:
        if not options:
            options = model_types.BaseModelSaveOption()

        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            snowml_matched_versions = env_utils.get_matched_package_versions_in_snowflake_conda_channel(
                req=requirements.Requirement(f"snowflake-ml-python=={snowml_env.VERSION}")
            )

            if len(snowml_matched_versions) < 1 and options.get("embed_local_ml_library", False) is False:
                logging.info(
                    f"Local snowflake-ml-python library has version {snowml_env.VERSION},"
                    " which is not available in the Snowflake server, embedding local ML library automatically."
                )
                options["embed_local_ml_library"] = True

        self.packager.save(
            name=name,
            model=model,
            signatures=signatures,
            sample_input=sample_input,
            metadata=metadata,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            python_version=python_version,
            ext_modules=ext_modules,
            code_paths=code_paths,
            options=options,
        )

        assert self.packager.meta is not None

        if not options.get("_legacy_save", False):
            # Keep both loose files and zipped file.
            # TODO(SNOW-726678): Remove once import a directory is possible.
            file_utils.copytree(
                str(self._packager_workspace_path), str(self.workspace_path / ModelComposer.MODEL_DIR_REL_PATH)
            )

        file_utils.make_archive(self.model_local_path, str(self._packager_workspace_path))

        self.manifest.save(
            session=self.session,
            model_meta=self.packager.meta,
            model_file_rel_path=pathlib.PurePosixPath(ModelComposer.MODEL_FILE_REL_PATH),
            options=options,
        )

        file_utils.upload_directory_to_stage(
            self.session,
            local_path=self.workspace_path,
            stage_path=self.stage_path,
            statement_params=self._statement_params,
        )

    def load(
        self,
        *,
        meta_only: bool = False,
        options: Optional[model_types.ModelLoadOption] = None,
    ) -> None:
        file_utils.download_directory_from_stage(
            self.session,
            stage_path=self.stage_path,
            local_path=self.workspace_path,
            statement_params=self._statement_params,
        )

        # TODO (Server-side Model Rollout): Remove this section.
        model_zip_path = pathlib.Path(glob.glob(str(self.workspace_path / "*.zip"))[0])
        ModelComposer.MODEL_FILE_REL_PATH = str(model_zip_path.relative_to(self.workspace_path))

        with zipfile.ZipFile(self.model_local_path, mode="r", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.extractall(path=self._packager_workspace_path)
        self.packager.load(meta_only=meta_only, options=options)
