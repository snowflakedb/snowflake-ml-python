import inspect
import os
import pathlib
import sys
from typing import Optional, cast, final

import anyio
import cloudpickle
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import file_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_handler
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)


@final
class CustomModelHandler(_base.BaseModelHandler["custom_model.CustomModel"]):
    """Handler for custom model."""

    HANDLER_TYPE = "custom"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    @classmethod
    def can_handle(cls, model: model_types.SupportedModelType) -> TypeGuard["custom_model.CustomModel"]:
        return isinstance(model, custom_model.CustomModel)

    @classmethod
    def cast_model(cls, model: model_types.SupportedModelType) -> "custom_model.CustomModel":
        assert isinstance(model, custom_model.CustomModel)
        return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "custom_model.CustomModel",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.CustomModelSaveOption],
    ) -> None:
        assert isinstance(model, custom_model.CustomModel)
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for custom model.")

        def get_prediction(
            target_method_name: str, sample_input_data: model_types.SupportedLocalDataType
        ) -> model_types.SupportedLocalDataType:
            target_method = getattr(model, target_method_name, None)
            assert callable(target_method) and inspect.ismethod(target_method)
            target_method = target_method.__func__

            if not isinstance(sample_input_data, pd.DataFrame):
                sample_input_data = model_signature._convert_local_data_to_df(sample_input_data)

            if inspect.iscoroutinefunction(target_method):
                with anyio.from_thread.start_blocking_portal() as portal:
                    predictions_df = portal.call(target_method, model, sample_input_data)
            else:
                predictions_df = target_method(model, sample_input_data)
            return predictions_df

        for func_name in model._get_partitioned_methods():
            function_properties = model_meta.function_properties.get(func_name, {})
            function_properties[model_meta_schema.FunctionProperties.PARTITIONED.value] = True
            model_meta.function_properties[func_name] = function_properties

        if not is_sub_model:
            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=[method.__name__ for method in model._get_infer_methods()],
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        if model.context.artifacts:
            artifacts_path = os.path.join(model_blob_path, cls.MODEL_ARTIFACTS_DIR)
            os.makedirs(artifacts_path, exist_ok=True)
            for _name, uri in model.context.artifacts.items():
                file_utils.copy_file_or_tree(uri, artifacts_path)

        # Save sub-models
        if model.context.model_refs:
            for sub_name, model_ref in model.context.model_refs.items():
                handler = model_handler.find_handler(model_ref.model)
                if handler is None:
                    raise TypeError(
                        f"Model {sub_name} in model context is not a supported model type. See "
                        "https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/"
                        "bring-your-own-model-types for more details."
                    )
                sub_model = handler.cast_model(model_ref.model)
                handler.save_model(
                    name=sub_name,
                    model=sub_model,
                    model_meta=model_meta,
                    model_blobs_dir_path=model_blobs_dir_path,
                    is_sub_model=True,
                    **cast(model_types.BaseModelSaveOption, kwargs),
                )

        # Make sure that the module where the model is defined get pickled by value as well.
        cloudpickle.register_pickle_by_value(sys.modules[model.__module__])
        pickled_obj = (model.__class__, model.context)
        with open(os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR), "wb") as f:
            cloudpickle.dump(pickled_obj, f)
        # model meta will be saved by the context manager
        model_meta.models[name] = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            handler_version=cls.HANDLER_VERSION,
            function_properties=model_meta.function_properties,
            artifacts={
                name: pathlib.Path(
                    os.path.join(cls.MODEL_ARTIFACTS_DIR, os.path.basename(os.path.normpath(path=uri)))
                ).as_posix()
                for name, uri in model.context.artifacts.items()
            },
        )

        # For Custom we set only when user set it.
        cuda_version = kwargs.get("cuda_version", None)
        if cuda_version:
            model_meta.env.cuda_version = cuda_version

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.CustomModelLoadOption],
    ) -> "custom_model.CustomModel":
        model_blob_path = os.path.join(model_blobs_dir_path, name)

        model_blobs_metadata = model_meta.models

        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            picked_obj = cloudpickle.load(f)
        ModelClass, context = picked_obj

        assert issubclass(ModelClass, custom_model.CustomModel)
        assert isinstance(context, custom_model.ModelContext)

        artifacts_meta = model_blob_metadata.artifacts
        artifacts = {
            name: str(pathlib.PurePath(model_blob_path) / pathlib.PurePosixPath(rel_path))
            for name, rel_path in artifacts_meta.items()
        }
        models: dict[str, model_types.SupportedModelType] = dict()
        for sub_model_name, _ref in context.model_refs.items():
            model_type = model_meta.models[sub_model_name].model_type
            handler = model_handler.load_handler(model_type)
            assert handler
            handler.try_upgrade(
                name=sub_model_name,
                model_meta=model_meta,
                model_blobs_dir_path=model_blobs_dir_path,
            )
            sub_model = handler.load_model(
                name=sub_model_name,
                model_meta=model_meta,
                model_blobs_dir_path=model_blobs_dir_path,
                **cast(model_types.BaseModelLoadOption, kwargs),
            )
            models[sub_model_name] = sub_model
        reconstructed_context = custom_model.ModelContext(artifacts=artifacts, models=models)
        model = ModelClass(reconstructed_context)

        assert isinstance(model, custom_model.CustomModel)
        return model

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: custom_model.CustomModel,
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.CustomModelLoadOption],
    ) -> custom_model.CustomModel:
        return raw_model
