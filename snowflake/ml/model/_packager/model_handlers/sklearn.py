import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union, cast, final

import cloudpickle
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
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    import sklearn.base
    import sklearn.pipeline


@final
class SKLModelHandler(_base.BaseModelHandler[Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"]]):
    """Handler for scikit-learn based model.

    Currently sklearn.base.BaseEstimator and sklearn.pipeline.Pipeline based classes are supported.
    """

    HANDLER_TYPE = "sklearn"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    DEFAULT_TARGET_METHODS = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard[Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"]]:
        return (
            (
                type_utils.LazyType("sklearn.base.BaseEstimator").isinstance(model)
                or type_utils.LazyType("sklearn.pipeline.Pipeline").isinstance(model)
            )
            and (not type_utils.LazyType("xgboost.XGBModel").isinstance(model))  # XGBModel is actually a BaseEstimator
            and any(
                (hasattr(model, method) and callable(getattr(model, method, None)))
                for method in cls.DEFAULT_TARGET_METHODS
            )
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"]:
        import sklearn.base
        import sklearn.pipeline

        assert isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, sklearn.pipeline.Pipeline)

        return cast(Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"], model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"],
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SKLModelSaveOptions],
    ) -> None:
        import sklearn.base
        import sklearn.pipeline

        assert isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, sklearn.pipeline.Pipeline)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str, sample_input: model_types.SupportedLocalDataType
            ) -> model_types.SupportedLocalDataType:
                if not isinstance(sample_input, (pd.DataFrame, np.ndarray)):
                    sample_input = model_signature._convert_local_data_to_df(sample_input)

                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                predictions_df = target_method(sample_input)
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
        with open(os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR), "wb") as f:
            cloudpickle.dump(model, f)
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [model_env.ModelDependency(requirement="scikit-learn", pip_name="scikit-learn")], check_local_version=True
        )

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"]:
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = cloudpickle.load(f)

        import sklearn.base
        import sklearn.pipeline

        assert isinstance(m, sklearn.base.BaseEstimator) or isinstance(m, sklearn.pipeline.Pipeline)
        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"],
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"],
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"],
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

            _SKLModel = type(
                "_SKLModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _SKLModel

        _SKLModel = _create_custom_model(raw_model, model_meta)
        skl_model = _SKLModel(custom_model.ModelContext())

        return skl_model
