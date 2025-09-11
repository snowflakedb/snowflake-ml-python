# mypy: disable-error-code="import"
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast, final

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
from snowflake.ml.model._packager.model_task import model_task_utils
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
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model.ubj"
    DEFAULT_TARGET_METHODS = ["predict", "predict_proba"]
    EXPLAIN_TARGET_METHODS = ["predict", "predict_proba"]

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
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.XGBModelSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", True)

        import xgboost

        assert isinstance(model, xgboost.Booster) or isinstance(model, xgboost.XGBModel)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str, sample_input_data: model_types.SupportedLocalDataType
            ) -> model_types.SupportedLocalDataType:
                if not isinstance(sample_input_data, (pd.DataFrame, np.ndarray, xgboost.DMatrix)):
                    sample_input_data = model_signature._convert_local_data_to_df(sample_input_data)

                if isinstance(model, xgboost.Booster) and not isinstance(sample_input_data, xgboost.DMatrix):
                    sample_input_data = xgboost.DMatrix(sample_input_data)

                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                predictions_df = target_method(sample_input_data)
                return predictions_df

            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )
            model_task_and_output = model_task_utils.resolve_model_task_and_output_type(model, model_meta.task)
            model_meta.task = model_task_and_output.task
            if enable_explainability:
                model_meta = handlers_utils.add_explain_method_signature(
                    model_meta=model_meta,
                    explain_method="explain",
                    target_method="predict",
                    output_return_type=model_task_and_output.output_type,
                )
                model_meta.function_properties = {
                    "explain": {model_meta_schema.FunctionProperties.PARTITIONED.value: False}
                }

                explain_target_method = handlers_utils.get_explain_target_method(model_meta, cls.EXPLAIN_TARGET_METHODS)

                background_data = handlers_utils.get_explainability_supported_background(
                    sample_input_data, model_meta, explain_target_method
                )
                if background_data is not None:
                    handlers_utils.save_background_data(
                        model_blobs_dir_path, cls.EXPLAIN_ARTIFACTS_DIR, cls.BG_DATA_FILE_SUFFIX, name, background_data
                    )
                else:
                    warnings.warn(
                        "sample_input_data should be provided for better explainability results",
                        category=UserWarning,
                        stacklevel=1,
                    )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        model.save_model(os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR))
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.XgboostModelBlobOptions(
                {
                    "xgb_estimator_type": model.__class__.__name__,
                    "enable_categorical": getattr(model, "enable_categorical", False),
                }
            ),
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

        if enable_explainability:
            model_meta.env.include_if_absent([model_env.ModelDependency(requirement="shap>=0.46.0", pip_name="shap")])
            model_meta.explain_algorithm = model_meta_schema.ModelExplainAlgorithm.SHAP
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.XGBModelLoadOptions],
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
        m.enable_categorical = model_blob_options.get("enable_categorical", False)

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
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.XGBModelLoadOptions],
    ) -> custom_model.CustomModel:
        import xgboost

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: Union["xgboost.Booster", "xgboost.XGBModel"],
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: Union["xgboost.Booster", "xgboost.XGBModel"],
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    enable_categorical = False
                    for col, d_type in X.dtypes.items():
                        if pd.api.extensions.ExtensionDtype.is_dtype(d_type):
                            if pd.CategoricalDtype.is_dtype(d_type):
                                enable_categorical = True
                            elif isinstance(d_type, pd.StringDtype):
                                X[col] = X[col].astype("category")
                                enable_categorical = True
                            continue
                        if not np.issubdtype(d_type, np.number):
                            # categorical columns are converted to numpy's str dtype
                            X[col] = X[col].astype("category")
                            enable_categorical = True
                    if isinstance(raw_model, xgboost.Booster):
                        X = xgboost.DMatrix(X, enable_categorical=enable_categorical)

                    res = getattr(raw_model, target_method)(X)

                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], np.ndarray):
                        # In case of multi-output estimators, predict_proba(), decision_function(), etc., functions
                        # return a list of ndarrays. We need to deal them separately
                        df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(res)
                    else:
                        df = pd.DataFrame(res)

                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                @custom_model.inference_api
                def explain_fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    import shap

                    explainer = shap.TreeExplainer(raw_model)
                    df = handlers_utils.convert_explanations_to_2D_df(raw_model, explainer.shap_values(X))
                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                if target_method == "explain":
                    return explain_fn
                return fn

            type_method_dict: dict[str, Any] = {"_raw_model": raw_model}
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
