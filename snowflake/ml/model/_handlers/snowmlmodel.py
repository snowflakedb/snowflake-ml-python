import os
from typing import TYPE_CHECKING, Callable, Optional, Type, cast

import cloudpickle
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
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    from snowflake.ml.modeling.framework.base import BaseEstimator


class _SnowMLModelHandler(_base._ModelHandler["BaseEstimator"]):
    """Handler for SnowML based model.

    Currently snowflake.ml.modeling.framework.base.BaseEstimator
        and snowflake.ml.modeling.pipeline.Pipeline based classes are supported.
    """

    handler_type = "snowml"
    DEFAULT_TARGET_METHODS = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]
    is_auto_signature = True

    @staticmethod
    def can_handle(
        model: model_types.SupportedModelType,
    ) -> TypeGuard["BaseEstimator"]:
        return (
            type_utils.LazyType("snowflake.ml.modeling.framework.base.BaseEstimator").isinstance(model)
            # Pipeline is inherited from BaseEstimator, so no need to add one more check
        ) and any(
            (hasattr(model, method) and callable(getattr(model, method, None)))
            for method in _SnowMLModelHandler.DEFAULT_TARGET_METHODS
        )

    @staticmethod
    def cast_model(
        model: model_types.SupportedModelType,
    ) -> "BaseEstimator":
        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(model, BaseEstimator)
        # Pipeline is inherited from BaseEstimator, so no need to add one more check

        return cast("BaseEstimator", model)

    @staticmethod
    def _save_model(
        name: str,
        model: "BaseEstimator",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SNOWModelSaveOptions],
    ) -> None:
        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(model, BaseEstimator)
        # Pipeline is inherited from BaseEstimator, so no need to add one more check

        if not is_sub_model:
            # TODO(xjiang): get model signature from modeling.
            if model_meta._signatures is None and sample_input is None:
                assert hasattr(model, "model_signatures")
                model_meta._signatures = getattr(model, "model_signatures", {})
            else:
                target_methods = model_meta_api._get_target_methods(
                    model=model,
                    target_methods=kwargs.pop("target_methods", None),
                    default_target_methods=_SnowMLModelHandler.DEFAULT_TARGET_METHODS,
                )

                def get_prediction(
                    target_method_name: str, sample_input: model_types.SupportedLocalDataType
                ) -> model_types.SupportedLocalDataType:
                    if not isinstance(sample_input, (pd.DataFrame,)):
                        sample_input = model_signature._convert_local_data_to_df(sample_input)

                    target_method = getattr(model, target_method_name, None)
                    assert callable(target_method)
                    predictions_df = target_method(sample_input)
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
        with open(os.path.join(model_blob_path, _SnowMLModelHandler.MODEL_BLOB_FILE), "wb") as f:
            cloudpickle.dump(model, f)
        base_meta = model_meta_api._ModelBlobMetadata(
            name=name, model_type=_SnowMLModelHandler.handler_type, path=_SnowMLModelHandler.MODEL_BLOB_FILE
        )
        model_meta.models[name] = base_meta
        _include_if_absent_pkgs = []
        model_dependencies = model._get_dependencies()
        for dep in model_dependencies:
            pkg_name = dep.split("==")[0]
            _include_if_absent_pkgs.append(model_meta_api.Dependency(conda_name=pkg_name, pip_name=pkg_name))
        model_meta._include_if_absent(_include_if_absent_pkgs)

    @staticmethod
    def _load_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "BaseEstimator":
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models
        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = cloudpickle.load(f)

        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(m, BaseEstimator)
        return m

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
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "BaseEstimator",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "BaseEstimator",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    res = getattr(raw_model, target_method)(X)

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

            _SnowMLModel = type(
                "_SnowMLModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _SnowMLModel

        raw_model = _SnowMLModelHandler._load_model(name, model_meta, model_blobs_dir_path, **kwargs)
        _SnowMLModel = _create_custom_model(raw_model, model_meta)
        snowml_model = _SnowMLModel(custom_model.ModelContext())

        return snowml_model
