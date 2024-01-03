# mypy: disable-error-code="import"
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Type,
    Union,
    cast,
    final,
)

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
    model_meta_schema,
)
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    import xgboost


@final
class XGBModelHandler(_base.BaseModelHandler[Union["xgboost.Booster", "xgboost.XGBModel"]]):
    """Handler for XGBoost based model.

    Currently xgboost.XGBModel based classes are supported.
    """

    HANDLER_TYPE = "xgboost"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model.ubj"
    DEFAULT_TARGET_METHODS = ["apply", "predict", "predict_proba"]

    @classmethod
    def can_handle(
        cls, model: model_types.SupportedModelType
    ) -> TypeGuard[Union["xgboost.Booster", "xgboost.XGBModel"]]:
        return (
            type_utils.LazyType("xgboost.Booster").isinstance(model)
            or type_utils.LazyType("xgboost.XGBModel").isinstance(model)
        ) and any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> Union["xgboost.Booster", "xgboost.XGBModel"]:
        import xgboost

        assert isinstance(model, xgboost.Booster) or isinstance(model, xgboost.XGBModel)

        return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: Union["xgboost.Booster", "xgboost.XGBModel"],
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.XGBModelSaveOptions],
    ) -> None:
        import xgboost

        assert isinstance(model, xgboost.Booster) or isinstance(model, xgboost.XGBModel)

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

                if isinstance(model, xgboost.Booster):
                    sample_input = xgboost.DMatrix(sample_input)

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
        model.save_model(os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR))
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
            options=model_meta_schema.XgboostModelBlobOptions({"xgb_estimator_type": model.__class__.__name__}),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(requirement="scikit-learn", pip_name="scikit-learn"),
                model_env.ModelDependency(requirement="xgboost", pip_name="xgboost"),
            ],
            check_local_version=True,
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", model_env.DEFAULT_CUDA_VERSION)

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> Union["xgboost.Booster", "xgboost.XGBModel"]:
        import xgboost

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_options = cast(model_meta_schema.XgboostModelBlobOptions, model_blob_metadata.options)
        if "xgb_estimator_type" not in model_blob_options:
            raise ValueError("Missing field `xgb_estimator_type` in model blob metadata for type `xgboost`")

        xgb_estimator_type = model_blob_options["xgb_estimator_type"]
        if not hasattr(xgboost, xgb_estimator_type):
            raise ValueError("Type of XGB estimator is illegal.")
        m = getattr(xgboost, xgb_estimator_type)()
        m.load_model(os.path.join(model_blob_path, model_blob_filename))

        if kwargs.get("use_gpu", False):
            assert type(kwargs.get("use_gpu", False)) == bool
            gpu_params = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
            if isinstance(m, xgboost.Booster):
                m.set_param(gpu_params)
            elif isinstance(m, xgboost.XGBModel):
                m.set_params(**gpu_params)

        assert isinstance(m, xgboost.Booster) or isinstance(m, xgboost.XGBModel)
        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: Union["xgboost.Booster", "xgboost.XGBModel"],
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        import xgboost

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: Union["xgboost.Booster", "xgboost.XGBModel"],
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: Union["xgboost.Booster", "xgboost.XGBModel"],
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    if isinstance(raw_model, xgboost.Booster):
                        X = xgboost.DMatrix(X)

                    res = getattr(raw_model, target_method)(X)

                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], np.ndarray):
                        # In case of multi-output estimators, predict_proba(), decision_function(), etc., functions
                        # return a list of ndarrays. We need to deal them separately
                        df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(res)
                    else:
                        df = pd.DataFrame(res)

                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                return fn

            type_method_dict: Dict[str, Any] = {"_raw_model": raw_model}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _XGBModel = type(
                "_XGBModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _XGBModel

        _XGBModel = _create_custom_model(raw_model, model_meta)
        xgb_model = _XGBModel(custom_model.ModelContext())

        return xgb_model
