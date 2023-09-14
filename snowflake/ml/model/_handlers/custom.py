import inspect
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Dict, Optional

import anyio
import cloudpickle
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import file_utils, type_utils
from snowflake.ml.model import (
    _model_handler,
    _model_meta as model_meta_api,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._handlers import _base

if TYPE_CHECKING:
    from snowflake.ml.model import custom_model


class _CustomModelHandler(_base._ModelHandler["custom_model.CustomModel"]):
    """Handler for custom model."""

    handler_type = "custom"

    @staticmethod
    def can_handle(model: model_types.SupportedModelType) -> TypeGuard["custom_model.CustomModel"]:
        return bool(type_utils.LazyType("snowflake.ml.model.custom_model.CustomModel").isinstance(model))

    @staticmethod
    def cast_model(model: model_types.SupportedModelType) -> "custom_model.CustomModel":
        from snowflake.ml.model import custom_model

        assert isinstance(model, custom_model.CustomModel)
        return model

    @staticmethod
    def _save_model(
        name: str,
        model: "custom_model.CustomModel",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.CustomModelSaveOption],
    ) -> None:
        from snowflake.ml.model import custom_model

        assert isinstance(model, custom_model.CustomModel)

        def get_prediction(
            target_method_name: str, sample_input: model_types.SupportedLocalDataType
        ) -> model_types.SupportedLocalDataType:
            target_method = getattr(model, target_method_name, None)
            assert callable(target_method) and inspect.ismethod(target_method)
            target_method = target_method.__func__

            if not isinstance(sample_input, pd.DataFrame):
                sample_input = model_signature._convert_local_data_to_df(sample_input)

            if inspect.iscoroutinefunction(target_method):
                with anyio.start_blocking_portal() as portal:
                    predictions_df = portal.call(target_method, model, sample_input)
            else:
                predictions_df = target_method(model, sample_input)
            return predictions_df

        if not is_sub_model:
            model_meta = model_meta_api._validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=[method.__name__ for method in model._get_infer_methods()],
                sample_input=sample_input,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        if model.context.artifacts:
            artifacts_path = os.path.join(model_blob_path, _CustomModelHandler.MODEL_ARTIFACTS_DIR)
            os.makedirs(artifacts_path, exist_ok=True)
            for _name, uri in model.context.artifacts.items():
                file_utils.copy_file_or_tree(uri, artifacts_path)

        # Save sub-models
        if model.context.model_refs:
            for sub_name, model_ref in model.context.model_refs.items():
                handler = _model_handler._find_handler(model_ref.model)
                assert handler is not None
                sub_model = handler.cast_model(model_ref.model)
                handler._save_model(
                    name=sub_name,
                    model=sub_model,
                    model_meta=model_meta,
                    model_blobs_dir_path=model_blobs_dir_path,
                    is_sub_model=True,
                )

        # Make sure that the module where the model is defined get pickled by value as well.
        cloudpickle.register_pickle_by_value(sys.modules[model.__module__])
        picked_obj = (model.__class__, model.context)
        with open(os.path.join(model_blob_path, _CustomModelHandler.MODEL_BLOB_FILE), "wb") as f:
            cloudpickle.dump(picked_obj, f)
        model_meta.models[name] = model_meta_api._ModelBlobMetadata(
            name=name,
            model_type=_CustomModelHandler.handler_type,
            path=_CustomModelHandler.MODEL_BLOB_FILE,
            artifacts={
                name: pathlib.Path(
                    os.path.join(_CustomModelHandler.MODEL_ARTIFACTS_DIR, os.path.basename(os.path.normpath(path=uri)))
                ).as_posix()
                for name, uri in model.context.artifacts.items()
            },
        )

        # For Custom we set only when user set it.
        cuda_version = kwargs.get("cuda_version", None)
        if cuda_version:
            model_meta.cuda_version = cuda_version

    @staticmethod
    def _load_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "custom_model.CustomModel":
        from snowflake.ml.model import custom_model

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models

        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
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
        models: Dict[str, model_types.SupportedModelType] = dict()
        for sub_model_name, _ref in context.model_refs.items():
            model_type = model_meta.models[sub_model_name].model_type
            handler = _model_handler._load_handler(model_type)
            assert handler
            sub_model = handler._load_model(
                name=sub_model_name,
                model_meta=model_meta,
                model_blobs_dir_path=model_blobs_dir_path,
            )
            models[sub_model_name] = sub_model
        reconstructed_context = custom_model.ModelContext(artifacts=artifacts, models=models)
        model = ModelClass(reconstructed_context)

        assert isinstance(model, custom_model.CustomModel)
        return model
