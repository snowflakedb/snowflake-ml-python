import os
from typing import TYPE_CHECKING, Callable, Optional, cast, final

import pandas as pd
from packaging import version
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import (
    base_migrator,
    tensorflow_migrator_2023_12_01,
    tensorflow_migrator_2025_01_01,
)
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import (
    tensorflow_handler,
    utils as model_signature_utils,
)

if TYPE_CHECKING:
    import tensorflow


@final
class TensorFlowHandler(_base.BaseModelHandler["tensorflow.Module"]):
    """Handler for TensorFlow based model or keras v2 model.

    Currently tensorflow.Module based classes are supported.
    """

    HANDLER_TYPE = "tensorflow"
    HANDLER_VERSION = "2025-03-01"
    _MIN_SNOWPARK_ML_VERSION = "1.8.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {
        "2023-12-01": tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201,
        "2025-01-01": tensorflow_migrator_2025_01_01.TensorflowHandlerMigrator20250101,
    }

    MODEL_BLOB_FILE_OR_DIR = "model"
    DEFAULT_TARGET_METHODS = ["__call__"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["tensorflow.nn.Module"]:
        if not type_utils.LazyType("tensorflow.Module").isinstance(model):
            return False
        if type_utils.LazyType("keras.Model").isinstance(model):
            import keras

            return version.parse(keras.__version__) < version.parse("3.0.0")
        return True

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
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.TensorflowSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for Tensorflow model.")

        import tensorflow

        assert isinstance(model, tensorflow.Module)
        multiple_inputs = kwargs.get("multiple_inputs", False)

        is_keras_model = type_utils.LazyType("keras.Model").isinstance(model)
        is_tf_keras_model = type_utils.LazyType("tf_keras.Model").isinstance(model)
        # Tensorflow and keras model save format is different.
        # Keras v2 models are saved using keras api
        # Tensorflow models are saved using tensorflow api

        if is_keras_model or is_tf_keras_model:
            save_format = "keras_tf"
        else:
            save_format = "tf"

        if is_keras_model or is_tf_keras_model:
            default_target_methods = ["predict"]
        else:
            default_target_methods = cls.DEFAULT_TARGET_METHODS

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=default_target_methods,
            )

            if is_keras_model and len(target_methods) > 1:
                raise ValueError("Keras model can only have one target method.")

            def get_prediction(
                target_method_name: str, sample_input_data: "model_types.SupportedLocalDataType"
            ) -> model_types.SupportedLocalDataType:
                if multiple_inputs:
                    if not tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(sample_input_data):
                        sample_input_data = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                            model_signature._convert_local_data_to_df(sample_input_data)
                        )
                else:
                    if not tensorflow_handler.TensorflowTensorHandler.can_handle(sample_input_data):
                        sample_input_data = tensorflow_handler.TensorflowTensorHandler.convert_from_df(
                            model_signature._convert_local_data_to_df(sample_input_data)
                        )

                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                for tensor in sample_input_data:
                    tensorflow.stop_gradient(tensor)
                if multiple_inputs:
                    predictions_df = target_method(*sample_input_data)
                    if not isinstance(predictions_df, tuple):
                        predictions_df = [predictions_df]
                else:
                    predictions_df = target_method(sample_input_data)

                return predictions_df

            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        save_path = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
        if save_format == "keras_tf":
            model.save(save_path, save_format="tf")
        else:
            tensorflow.saved_model.save(
                model,
                save_path,
                options=tensorflow.saved_model.SaveOptions(experimental_custom_gradients=False),
            )

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.TensorflowModelBlobOptions(
                save_format=save_format, multiple_inputs=multiple_inputs
            ),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        dependencies = [
            model_env.ModelDependency(requirement="tensorflow", pip_name="tensorflow"),
        ]
        if is_keras_model:
            dependencies.append(model_env.ModelDependency(requirement="keras<=3", pip_name="keras"))
        elif is_tf_keras_model:
            dependencies.append(model_env.ModelDependency(requirement="tf-keras", pip_name="tf-keras"))

        model_meta.env.include_if_absent(
            dependencies,
            check_local_version=True,
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.TensorflowLoadOptions],
    ) -> "tensorflow.Module":
        import tensorflow

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_options = cast(model_meta_schema.TensorflowModelBlobOptions, model_blob_metadata.options)
        load_path = os.path.join(model_blob_path, model_blob_filename)
        save_format = model_blob_options.get("save_format", "keras_tf")
        if save_format == "keras_tf":
            if version.parse(tensorflow.keras.__version__) >= version.parse("3.0.0"):
                import tf_keras

                m = tf_keras.models.load_model(load_path)
            else:
                m = tensorflow.keras.models.load_model(load_path)
        else:
            m = tensorflow.saved_model.load(load_path)

        return cast(tensorflow.Module, m)

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "tensorflow.Module",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.TensorflowLoadOptions],
    ) -> custom_model.CustomModel:
        import tensorflow

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "tensorflow.Module",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            multiple_inputs = cast(
                model_meta_schema.TensorflowModelBlobOptions, model_meta.models[model_meta.name].options
            )["multiple_inputs"]

            def fn_factory(
                raw_model: "tensorflow.Module",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    if X.isnull().any(axis=None):
                        raise ValueError("Tensor cannot handle null values.")

                    if multiple_inputs:
                        t = tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(X, signature.inputs)

                        for tensor in t:
                            tensorflow.stop_gradient(tensor)
                        res = getattr(raw_model, target_method)(*t)

                        if not isinstance(res, tuple):
                            res = [res]
                    else:
                        t = tensorflow_handler.TensorflowTensorHandler.convert_from_df(X, signature.inputs)

                        tensorflow.stop_gradient(t)
                        res = getattr(raw_model, target_method)(t)

                    return model_signature_utils.rename_pandas_df(
                        model_signature._convert_local_data_to_df(res, ensure_serializable=True),
                        features=signature.outputs,
                    )

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
