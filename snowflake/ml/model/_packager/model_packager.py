import logging
import os
from types import ModuleType
from typing import Optional

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_handler
from snowflake.ml.model._packager.model_meta import model_meta

logger = logging.getLogger(__name__)


class ModelPackager:
    """Top-level class to save/load and manage a Snowflake Native formatted model.
        It maintains the actual model blob files, environment required by model itself and signatures to do the
        inference with the model.

    Attributes:
        local_dir_path: A path to a local directory will files to dump and load.
        model: The model object to be saved / loaded from file.
        meta: The model metadata (ModelMetadata object) to be saved / loaded from file.
            model and meta will be set once save / load method is called.

    """

    MODEL_BLOBS_DIR = "models"

    def __init__(self, local_dir_path: str) -> None:
        self.local_dir_path = os.path.normpath(local_dir_path)
        self.model: Optional[model_types.SupportedModelType] = None
        self.meta: Optional[model_meta.ModelMetadata] = None

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
        ext_modules: Optional[list[ModuleType]] = None,
        code_paths: Optional[list[str]] = None,
        options: model_types.ModelSaveOption,
        task: model_types.Task = model_types.Task.UNKNOWN,
    ) -> model_meta.ModelMetadata:
        if (signatures is None) and (sample_input_data is None) and not model_handler.is_auto_signature_model(model):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Either of `signatures` or `sample_input_data` must be provided for this kind of model."
                ),
            )

        handler = model_handler.find_handler(model)
        if handler is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_TYPE,
                original_exception=TypeError(f"{type(model)} is not supported."),
            )
        with model_meta.create_model_metadata(
            model_dir_path=self.local_dir_path,
            name=name,
            model_type=handler.HANDLER_TYPE,
            metadata=metadata,
            code_paths=code_paths,
            signatures=signatures,
            ext_modules=ext_modules,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            artifact_repository_map=artifact_repository_map,
            resource_constraint=resource_constraint,
            python_version=python_version,
            task=task,
            target_platforms=target_platforms,
            **options,
        ) as meta:
            model_blobs_path = os.path.join(self.local_dir_path, ModelPackager.MODEL_BLOBS_DIR)
            os.makedirs(model_blobs_path, exist_ok=True)
            model = handler.cast_model(model)
            handler.save_model(
                name=name,
                model=model,
                model_meta=meta,
                model_blobs_dir_path=model_blobs_path,
                sample_input_data=sample_input_data,
                is_sub_model=False,
                **options,
            )
            if signatures is None:
                logger.info(f"Model signatures are auto inferred as:\n\n{meta.signatures}")

        self.model = model
        self.meta = meta
        return meta

    def load(
        self,
        *,
        meta_only: bool = False,
        as_custom_model: bool = False,
        options: Optional[model_types.ModelLoadOption] = None,
    ) -> model_meta.ModelMetadata:
        """Load the model into memory from directory. Used internal only.

        Args:
            meta_only: Flag to indicate that if only load metadata.
            as_custom_model: When set to True, It will try to convert the model as custom model after load.
            options: Model loading options.

        Raises:
            SnowflakeMLException: Raised if model is not native format.

        Returns:
            Metadata of loaded model.
        """

        self.meta = model_meta.ModelMetadata.load(self.local_dir_path)
        if meta_only:
            return self.meta

        model_meta.load_code_path(self.local_dir_path)

        handler = model_handler.load_handler(self.meta.model_type)
        if handler is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_TYPE,
                original_exception=TypeError(f"{self.meta.model_type} is not supported."),
            )
        model_blobs_path = os.path.join(self.local_dir_path, ModelPackager.MODEL_BLOBS_DIR)
        if options is None:
            options = {}

        handler.try_upgrade(self.meta.name, self.meta, model_blobs_path)
        m = handler.load_model(self.meta.name, self.meta, model_blobs_path, **options)

        if as_custom_model:
            background_data = handler.load_background_data(self.meta.name, model_blobs_path)
            m = handler.convert_as_custom_model(m, self.meta, background_data, **options)
            assert isinstance(m, custom_model.CustomModel)

        self.model = m
        return self.meta
