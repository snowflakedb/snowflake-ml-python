import os
from typing import TYPE_CHECKING, Callable, Optional, cast, final

import cloudpickle
import numpy as np
import pandas as pd
from packaging import version
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
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    import keras


@final
class KerasHandler(_base.BaseModelHandler["keras.Model"]):
    """Handler for Keras v3 model.

    Currently keras.Model based classes are supported.
    """

    HANDLER_TYPE = "keras"
    HANDLER_VERSION = "2025-01-01"
    _MIN_SNOWPARK_ML_VERSION = "1.7.5"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model.keras"
    CUSTOM_OBJECT_SAVE_PATH = "custom_objects.pkl"
    DEFAULT_TARGET_METHODS = ["predict"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["keras.Model"]:
        if not type_utils.LazyType("keras.Model").isinstance(model):
            return False
        import keras

        return version.parse(keras.__version__) >= version.parse("3.0.0")

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "keras.Model":
        import keras

        assert isinstance(model, keras.Model)

        return cast(keras.Model, model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "keras.Model",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.TensorflowSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for Tensorflow model.")

        import keras

        assert isinstance(model, keras.Model)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str, sample_input_data: "model_types.SupportedLocalDataType"
            ) -> model_types.SupportedLocalDataType:
                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                predictions_df = target_method(sample_input_data)

                if (
                    type_utils.LazyType("tensorflow.Tensor").isinstance(predictions_df)
                    or type_utils.LazyType("tensorflow.Variable").isinstance(predictions_df)
                    or type_utils.LazyType("torch.Tensor").isinstance(predictions_df)
                ):
                    predictions_df = [predictions_df]

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
        model.save(save_path)

        custom_object_save_path = os.path.join(model_blob_path, cls.CUSTOM_OBJECT_SAVE_PATH)
        custom_objects = keras.saving.get_custom_objects()
        with open(custom_object_save_path, "wb") as f:
            cloudpickle.dump(custom_objects, f)

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        dependencies = [
            model_env.ModelDependency(requirement="keras>=3", pip_name="keras"),
        ]
        keras_backend = keras.backend.backend()
        if keras_backend == "tensorflow":
            dependencies.append(model_env.ModelDependency(requirement="tensorflow", pip_name="tensorflow"))
        elif keras_backend == "torch":
            dependencies.append(model_env.ModelDependency(requirement="pytorch", pip_name="torch"))
        elif keras_backend == "jax":
            dependencies.append(model_env.ModelDependency(requirement="jax", pip_name="jax"))
        else:
            raise ValueError(f"Unsupported backend {keras_backend}")

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
    ) -> "keras.Model":
        import keras

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path

        custom_object_save_path = os.path.join(model_blob_path, cls.CUSTOM_OBJECT_SAVE_PATH)
        with open(custom_object_save_path, "rb") as f:
            custom_objects = cloudpickle.load(f)
        load_path = os.path.join(model_blob_path, model_blob_filename)
        m = keras.models.load_model(load_path, custom_objects=custom_objects, safe_mode=False)

        return cast(keras.Model, m)

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "keras.Model",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.TensorflowLoadOptions],
    ) -> custom_model.CustomModel:

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "keras.Model",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "keras.Model",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                dtype_map = {spec.name: spec.as_dtype(force_numpy_dtype=True) for spec in signature.inputs}

                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    res = getattr(raw_model, target_method)(X.astype(dtype_map), verbose=0)

                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], np.ndarray):
                        # In case of multi-output estimators, predict_proba(), decision_function(), etc., functions
                        # return a list of ndarrays. We need to deal them separately
                        df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(res)
                    else:
                        df = pd.DataFrame(res)

                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _KerasModel = type(
                "_KerasModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _KerasModel

        _KerasModel = _create_custom_model(raw_model, model_meta)
        keras_model = _KerasModel(custom_model.ModelContext())

        return keras_model
