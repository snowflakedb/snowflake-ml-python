import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, cast, final

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
)
from snowflake.ml.model._signatures import (
    numpy_handler,
    tensorflow_handler,
    utils as model_signature_utils,
)

if TYPE_CHECKING:
    import tensorflow


@final
class TensorFlowHandler(_base.BaseModelHandler["tensorflow.Module"]):
    """Handler for TensorFlow based model.

    Currently tensorflow.Module based classes are supported.
    """

    HANDLER_TYPE = "tensorflow"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model"
    DEFAULT_TARGET_METHODS = ["__call__"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["tensorflow.nn.Module"]:
        return type_utils.LazyType("tensorflow.Module").isinstance(model)

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "tensorflow.Module":
        import tensorflow

        assert isinstance(model, tensorflow.Module)

        return cast(tensorflow.Module, model)

    @classmethod
    def save_model(
        cls,
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
            default_target_methods = cls.DEFAULT_TARGET_METHODS

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
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

            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input=sample_input,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        if isinstance(model, tensorflow.keras.Model):
            tensorflow.keras.models.save_model(model, os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR))
        else:
            tensorflow.saved_model.save(model, os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR))

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [model_env.ModelDependency(requirement="tensorflow", pip_name="tensorflow")], check_local_version=True
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", model_env.DEFAULT_CUDA_VERSION)

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "tensorflow.Module":
        import tensorflow

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        m = tensorflow.keras.models.load_model(os.path.join(model_blob_path, model_blob_filename), compile=False)
        if isinstance(m, tensorflow.keras.Model):
            return m
        return cast(tensorflow.Module, m)

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "tensorflow.Module",
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
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

        _TensorFlowModel = _create_custom_model(raw_model, model_meta)
        tf_model = _TensorFlowModel(custom_model.ModelContext())

        return tf_model
