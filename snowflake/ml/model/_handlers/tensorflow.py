import os
from typing import TYPE_CHECKING, Callable, Optional, Type, cast

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import (
    _model_meta as model_meta_api,
    custom_model,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._handlers import _base
from snowflake.ml.model._signatures import (
    numpy_handler,
    tensorflow_handler,
    utils as model_signature_utils,
)

if TYPE_CHECKING:
    import tensorflow


class _TensorFlowHandler(_base._ModelHandler["tensorflow.Module"]):
    """Handler for TensorFlow based model.

    Currently tensorflow.Module based classes are supported.
    """

    handler_type = "tensorflow"
    MODEL_BLOB_FILE = "model"
    DEFAULT_TARGET_METHODS = ["__call__"]

    @staticmethod
    def can_handle(
        model: model_types.SupportedModelType,
    ) -> TypeGuard["tensorflow.nn.Module"]:
        return type_utils.LazyType("tensorflow.Module").isinstance(model)

    @staticmethod
    def cast_model(
        model: model_types.SupportedModelType,
    ) -> "tensorflow.Module":
        import tensorflow

        assert isinstance(model, tensorflow.Module)

        return cast(tensorflow.Module, model)

    @staticmethod
    def _save_model(
        name: str,
        model: "tensorflow.Module",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.TensorflowSaveOptions],
    ) -> None:
        import tensorflow

        assert isinstance(model, tensorflow.Module)

        if isinstance(model, tensorflow.keras.Model):
            default_target_methods = ["predict"]
        else:
            default_target_methods = _TensorFlowHandler.DEFAULT_TARGET_METHODS

        if not is_sub_model:
            target_methods = model_meta_api._get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=default_target_methods,
            )

            def get_prediction(
                target_method_name: str, sample_input: "model_types.SupportedLocalDataType"
            ) -> model_types.SupportedLocalDataType:
                if not tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(sample_input):
                    sample_input = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                        model_signature._convert_local_data_to_df(sample_input)
                    )

                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                for tensor in sample_input:
                    tensorflow.stop_gradient(tensor)
                predictions_df = target_method(*sample_input)

                if isinstance(predictions_df, (tensorflow.Tensor, tensorflow.Variable, np.ndarray)):
                    predictions_df = [predictions_df]

                return predictions_df

            model_meta = model_meta_api._validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input=sample_input,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        if isinstance(model, tensorflow.keras.Model):
            tensorflow.keras.models.save_model(model, os.path.join(model_blob_path, _TensorFlowHandler.MODEL_BLOB_FILE))
        else:
            tensorflow.saved_model.save(model, os.path.join(model_blob_path, _TensorFlowHandler.MODEL_BLOB_FILE))

        base_meta = model_meta_api._ModelBlobMetadata(
            name=name, model_type=_TensorFlowHandler.handler_type, path=_TensorFlowHandler.MODEL_BLOB_FILE
        )
        model_meta.models[name] = base_meta
        model_meta._include_if_absent([model_meta_api.Dependency(conda_name="tensorflow", pip_name="tensorflow")])

        model_meta.cuda_version = kwargs.get("cuda_version", model_meta_api._DEFAULT_CUDA_VERSION)

    @staticmethod
    def _load_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "tensorflow.Module":
        import tensorflow

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models
        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        m = tensorflow.keras.models.load_model(os.path.join(model_blob_path, model_blob_filename), compile=False)
        if isinstance(m, tensorflow.keras.Model):
            return m
        return cast(tensorflow.Module, m)

    @staticmethod
    def _load_as_custom_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        """Create a custom model class wrap for unified interface when being deployed. The predict method will be
        re-targeted based on target_method metadata.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.
            kwargs: Options when loading the model.

        Returns:
            The model object as a custom model.
        """
        import tensorflow

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "tensorflow.Module",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "tensorflow.Module",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    if X.isnull().any(axis=None):
                        raise ValueError("Tensor cannot handle null values.")

                    t = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(X, signature.inputs)

                    for tensor in t:
                        tensorflow.stop_gradient(tensor)
                    res = getattr(raw_model, target_method)(*t)

                    if isinstance(res, (tensorflow.Tensor, tensorflow.Variable, np.ndarray)):
                        res = [res]

                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], np.ndarray):
                        # In case of running on CPU, it will return numpy array
                        df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(res)
                    else:
                        df = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(res)
                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _TensorFlowModel = type(
                "_TensorFlowModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _TensorFlowModel

        raw_model = _TensorFlowHandler()._load_model(name, model_meta, model_blobs_dir_path, **kwargs)
        _TensorFlowModel = _create_custom_model(raw_model, model_meta)
        tf_model = _TensorFlowModel(custom_model.ModelContext())

        return tf_model
