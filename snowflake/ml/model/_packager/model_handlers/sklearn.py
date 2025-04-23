import os
import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union, cast, final

import cloudpickle
import numpy as np
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import env, type_utils
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
    import sklearn.base
    import sklearn.pipeline


def _unpack_container_runtime_pipeline(model: "sklearn.pipeline.Pipeline") -> "sklearn.pipeline.Pipeline":
    new_steps = []
    for step_name, step in model.steps:
        new_reg = step
        if hasattr(step, "_sklearn_estimator") and step._sklearn_estimator is not None:
            # Unpack estimator to open source.
            new_reg = step._sklearn_estimator
        new_steps.append((step_name, new_reg))

    model.steps = new_steps
    return model


@final
class SKLModelHandler(_base.BaseModelHandler[Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"]]):
    """Handler for scikit-learn based model.

    Currently sklearn.base.BaseEstimator and sklearn.pipeline.Pipeline based classes are supported.
    """

    HANDLER_TYPE = "sklearn"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    DEFAULT_TARGET_METHODS = [
        "predict",
        "transform",
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "score_samples",
    ]
    EXPLAIN_TARGET_METHODS = ["predict", "predict_proba", "predict_log_proba"]

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
            and (
                not type_utils.LazyType("lightgbm.LGBMModel").isinstance(model)
            )  # LGBMModel is actually a BaseEstimator
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
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SKLModelSaveOptions],
    ) -> None:
        # setting None by default to distinguish if users did not set it
        enable_explainability = kwargs.get("enable_explainability", None)

        import sklearn.base
        import sklearn.pipeline

        assert isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, sklearn.pipeline.Pipeline)
        if enable_explainability:
            # if users set it explicitly but no sample_input_data then error out
            if sample_input_data is None:
                raise ValueError("Sample input data is required to enable explainability.")

        # If this is a pipeline and we are in the container runtime, check for distributed estimator.
        if env.IN_ML_RUNTIME and isinstance(model, sklearn.pipeline.Pipeline):
            model = _unpack_container_runtime_pipeline(model)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str,
                sample_input_data: model_types.SupportedLocalDataType,
            ) -> model_types.SupportedLocalDataType:
                if not isinstance(sample_input_data, (pd.DataFrame, np.ndarray)):
                    sample_input_data = model_signature._convert_local_data_to_df(sample_input_data)

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

            explain_target_method = handlers_utils.get_explain_target_method(model_meta, cls.EXPLAIN_TARGET_METHODS)

            background_data = handlers_utils.get_explainability_supported_background(
                sample_input_data, model_meta, explain_target_method
            )

            model_task_and_output_type = model_task_utils.resolve_model_task_and_output_type(model, model_meta.task)
            model_meta.task = model_task_and_output_type.task

            # if users did not ask then we enable if we have background data
            if enable_explainability is None:
                if background_data is None:
                    warnings.warn(
                        "sample_input_data should be provided to enable explainability by default",
                        category=UserWarning,
                        stacklevel=1,
                    )
                    enable_explainability = False
                elif model_meta.task == model_types.Task.UNKNOWN or explain_target_method is None:
                    enable_explainability = False
                else:
                    enable_explainability = True
            if enable_explainability:
                model_meta = handlers_utils.add_explain_method_signature(
                    model_meta=model_meta,
                    explain_method="explain",
                    target_method=explain_target_method,
                    output_return_type=model_task_and_output_type.output_type,
                )
                handlers_utils.save_background_data(
                    model_blobs_dir_path,
                    cls.EXPLAIN_ARTIFACTS_DIR,
                    cls.BG_DATA_FILE_SUFFIX,
                    name,
                    background_data,
                )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR), "wb") as f:
            cloudpickle.dump(model, f)
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        # if model instance is a pipeline, check the pipeline steps
        if isinstance(model, sklearn.pipeline.Pipeline):
            for _, pipeline_step in model.steps:
                if type_utils.LazyType("lightgbm.LGBMModel").isinstance(pipeline_step) or type_utils.LazyType(
                    "lightgbm.Booster"
                ).isinstance(pipeline_step):
                    model_meta.env.include_if_absent(
                        [
                            model_env.ModelDependency(requirement="lightgbm", pip_name="lightgbm"),
                        ],
                        check_local_version=True,
                    )
                elif type_utils.LazyType("xgboost.XGBModel").isinstance(pipeline_step) or type_utils.LazyType(
                    "xgboost.Booster"
                ).isinstance(pipeline_step):
                    model_meta.env.include_if_absent(
                        [
                            model_env.ModelDependency(requirement="xgboost", pip_name="xgboost"),
                        ],
                        check_local_version=True,
                    )
                elif type_utils.LazyType("catboost.CatBoost").isinstance(pipeline_step):
                    model_meta.env.include_if_absent(
                        [
                            model_env.ModelDependency(requirement="catboost", pip_name="catboost"),
                        ],
                        check_local_version=True,
                    )

        if enable_explainability:
            model_meta.env.include_if_absent([model_env.ModelDependency(requirement="shap", pip_name="shap")])
            model_meta.explain_algorithm = model_meta_schema.ModelExplainAlgorithm.SHAP

        model_meta.env.include_if_absent(
            [model_env.ModelDependency(requirement="scikit-learn", pip_name="scikit-learn")],
            check_local_version=True,
        )

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.SKLModelLoadOptions],
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
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.SKLModelLoadOptions],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"],
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"],
                signature: model_signature.ModelSignature,
                target_method: str,
                background_data: Optional[pd.DataFrame],
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

                @custom_model.inference_api
                def explain_fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    import shap

                    try:
                        explainer = shap.Explainer(raw_model, background_data)
                        df = handlers_utils.convert_explanations_to_2D_df(raw_model, explainer(X).values)
                    except TypeError:
                        try:
                            dtype_map = {spec.name: spec.as_dtype(force_numpy_dtype=True) for spec in signature.inputs}

                            if isinstance(X, pd.DataFrame):
                                X = X.astype(dtype_map, copy=False)
                            if hasattr(raw_model, "predict_proba"):
                                if isinstance(X, np.ndarray):
                                    explanations = shap.Explainer(
                                        raw_model.predict_proba, background_data.values  # type: ignore[union-attr]
                                    )(X).values
                                else:
                                    explanations = shap.Explainer(raw_model.predict_proba, background_data)(X).values
                            elif hasattr(raw_model, "predict"):
                                if isinstance(X, np.ndarray):
                                    explanations = shap.Explainer(
                                        raw_model.predict, background_data.values  # type: ignore[union-attr]
                                    )(X).values
                                else:
                                    explanations = shap.Explainer(raw_model.predict, background_data)(X).values
                            else:
                                raise ValueError("Missing any supported target method to explain.")
                            df = handlers_utils.convert_explanations_to_2D_df(raw_model, explanations)
                        except TypeError as e:
                            raise ValueError(f"Explanation for this model type not supported yet: {str(e)}")
                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                if target_method == "explain":
                    return explain_fn

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name, background_data)

            _SKLModel = type(
                "_SKLModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _SKLModel

        _SKLModel = _create_custom_model(raw_model, model_meta)
        skl_model = _SKLModel(custom_model.ModelContext())

        return skl_model
