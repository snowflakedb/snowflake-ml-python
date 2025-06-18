import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, cast, final

import cloudpickle
import numpy as np
import pandas as pd
import shap
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml._internal.exceptions import exceptions
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
    from snowflake.ml.modeling.framework.base import BaseEstimator


def _apply_transforms_up_to_last_step(
    model: "BaseEstimator",
    data: model_types.SupportedDataType,
) -> pd.DataFrame:
    """Apply all transformations in the snowml pipeline model up to the last step."""
    if type_utils.LazyType("snowflake.ml.modeling.pipeline.Pipeline").isinstance(model):
        for step_name, step in model.steps[:-1]:  # type: ignore[attr-defined]
            if not hasattr(step, "transform"):
                raise ValueError(f"Step '{step_name}' does not have a 'transform' method.")
            data = pd.DataFrame(step.transform(data))
    return data


@final
class SnowMLModelHandler(_base.BaseModelHandler["BaseEstimator"]):
    """Handler for SnowML based model.

    Currently snowflake.ml.modeling.framework.base.BaseEstimator
        and snowflake.ml.modeling.pipeline.Pipeline based classes are supported.
    """

    HANDLER_TYPE = "snowml"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    DEFAULT_TARGET_METHODS = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]
    EXPLAIN_TARGET_METHODS = ["predict_proba", "predict", "predict_log_proba"]

    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["BaseEstimator"]:
        return (
            type_utils.LazyType("snowflake.ml.modeling.framework.base.BaseEstimator").isinstance(model)
            # Pipeline is inherited from BaseEstimator, so no need to add one more check
        ) and any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "BaseEstimator":
        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(model, BaseEstimator)
        # Pipeline is inherited from BaseEstimator, so no need to add one more check

        return cast("BaseEstimator", model)

    @classmethod
    def _get_supported_object_for_explainability(
        cls,
        estimator: "BaseEstimator",
        background_data: Optional[model_types.SupportedDataType],
        enable_explainability: Optional[bool],
    ) -> Any:

        tree_methods = ["to_xgboost", "to_lightgbm", "to_sklearn"]
        non_tree_methods = ["to_sklearn"]
        for method_name in tree_methods:
            if hasattr(estimator, method_name):
                try:
                    result = getattr(estimator, method_name)()
                    return result
                except exceptions.SnowflakeMLException:
                    pass  # Do nothing and continue to the next method
        for method_name in non_tree_methods:
            if hasattr(estimator, method_name):
                try:
                    result = getattr(estimator, method_name)()
                    if enable_explainability is None and background_data is None:
                        return None  # cannot get explain without background data
                    elif enable_explainability and background_data is None:
                        raise ValueError(
                            "Provide `sample_input_data` to generate explanations for sklearn Snowpark ML models."
                        )
                    return result
                except exceptions.SnowflakeMLException:
                    pass  # Do nothing and continue to the next method
        return None

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "BaseEstimator",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SNOWModelSaveOptions],
    ) -> None:

        enable_explainability = kwargs.get("enable_explainability", None)

        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(model, BaseEstimator)
        # Pipeline is inherited from BaseEstimator, so no need to add one more check

        if not is_sub_model:
            if model_meta.signatures or sample_input_data is not None:
                warnings.warn(
                    "Providing model signature for Snowpark ML "
                    + "Modeling model is not required. Model signature will automatically be inferred during fitting. ",
                    UserWarning,
                    stacklevel=2,
                )
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
                    is_for_modeling_model=True,
                )
            else:
                assert hasattr(model, "model_signatures"), "Model does not have model signatures as expected."
                model_signature_dict = getattr(model, "model_signatures", {})
                optional_target_methods = kwargs.pop("target_methods", None)
                if not optional_target_methods:
                    model_meta.signatures = model_signature_dict
                else:
                    temp_model_signature_dict = {}
                    for method_name in optional_target_methods:
                        method_model_signature = model_signature_dict.get(method_name, None)
                        if method_model_signature is not None:
                            temp_model_signature_dict[method_name] = method_model_signature
                        else:
                            raise ValueError(f"Target method {method_name} does not exist in the model.")
                    model_meta.signatures = temp_model_signature_dict

        python_base_obj = cls._get_supported_object_for_explainability(model, sample_input_data, enable_explainability)
        explain_target_method = handlers_utils.get_explain_target_method(model_meta, cls.EXPLAIN_TARGET_METHODS)

        if enable_explainability:
            if explain_target_method is None:
                raise ValueError(
                    "The model must have one of the following methods to enable explainability: "
                    + ", ".join(cls.EXPLAIN_TARGET_METHODS)
                )
        if enable_explainability is None:
            if python_base_obj is None or explain_target_method is None:
                # set None to False so we don't include shap in the environment
                enable_explainability = False
            else:
                enable_explainability = True
        if enable_explainability:
            try:
                model_task_and_output_type = model_task_utils.resolve_model_task_and_output_type(
                    python_base_obj, model_meta.task
                )
                model_meta.task = model_task_and_output_type.task
                background_data = handlers_utils.get_explainability_supported_background(
                    sample_input_data, model_meta, explain_target_method
                )
                if type_utils.LazyType("snowflake.ml.modeling.pipeline.Pipeline").isinstance(model):
                    transformed_df = _apply_transforms_up_to_last_step(model, sample_input_data)
                    explain_fn = cls._build_explain_fn(model, background_data)
                    model_meta = handlers_utils.add_inferred_explain_method_signature(
                        model_meta=model_meta,
                        explain_method="explain",
                        target_method=explain_target_method,  # type: ignore[arg-type]
                        background_data=background_data,
                        explain_fn=explain_fn,
                        output_feature_names=transformed_df.columns,
                    )
                else:
                    model_meta = handlers_utils.add_explain_method_signature(
                        model_meta=model_meta,
                        explain_method="explain",
                        target_method=explain_target_method,
                        output_return_type=model_task_and_output_type.output_type,
                    )
                if background_data is not None:
                    handlers_utils.save_background_data(
                        model_blobs_dir_path,
                        cls.EXPLAIN_ARTIFACTS_DIR,
                        cls.BG_DATA_FILE_SUFFIX,
                        name,
                        background_data,
                    )
            except Exception:
                if kwargs.get("enable_explainability", None):
                    # user explicitly enabled explainability, so we should raise the error
                    raise ValueError(
                        "Explainability for this model is not supported. Please set `enable_explainability=False`"
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

        _include_if_absent_pkgs = []
        model_dependencies = model._get_dependencies()
        for dep in model_dependencies:
            pkg_name = dep.split("==")[0]
            _include_if_absent_pkgs.append(model_env.ModelDependency(requirement=pkg_name, pip_name=pkg_name))

        if enable_explainability:
            model_meta.env.include_if_absent([model_env.ModelDependency(requirement="shap>=0.46.0", pip_name="shap")])
            model_meta.explain_algorithm = model_meta_schema.ModelExplainAlgorithm.SHAP
        model_meta.env.include_if_absent(_include_if_absent_pkgs, check_local_version=True)

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.SNOWModelLoadOptions],
    ) -> "BaseEstimator":
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = cloudpickle.load(f)

        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(m, BaseEstimator)
        return m

    @classmethod
    def _build_explain_fn(
        cls, model: "BaseEstimator", background_data: model_types.SupportedDataType
    ) -> Callable[[model_types.SupportedDataType], pd.DataFrame]:

        predictor = model
        is_pipeline = type_utils.LazyType("snowflake.ml.modeling.pipeline.Pipeline").isinstance(model)
        if is_pipeline:
            background_data = _apply_transforms_up_to_last_step(model, background_data)
            predictor = model.steps[-1][1]  # type: ignore[attr-defined]

        def explain_fn(data: model_types.SupportedDataType) -> pd.DataFrame:
            data = _apply_transforms_up_to_last_step(model, data)
            tree_methods = ["to_xgboost", "to_lightgbm"]
            non_tree_methods = ["to_sklearn", None]  # None just uses the predictor directly
            for method_name in tree_methods:
                try:
                    base_model = getattr(predictor, method_name)()
                    explainer = shap.TreeExplainer(base_model)
                    return handlers_utils.convert_explanations_to_2D_df(model, explainer.shap_values(data))
                except exceptions.SnowflakeMLException:
                    pass  # Do nothing and continue to the next method
            for method_name in non_tree_methods:  # type: ignore[assignment]
                try:
                    base_model = getattr(predictor, method_name)() if method_name is not None else predictor
                    try:
                        explainer = shap.Explainer(base_model, masker=background_data)
                        return handlers_utils.convert_explanations_to_2D_df(base_model, explainer(data).values)
                    except TypeError:
                        for explain_target_method in cls.EXPLAIN_TARGET_METHODS:
                            if not hasattr(base_model, explain_target_method):
                                continue
                            explain_target_method_fn = getattr(base_model, explain_target_method)
                            if isinstance(data, np.ndarray):
                                explainer = shap.Explainer(
                                    explain_target_method_fn,
                                    background_data.values,  # type: ignore[union-attr]
                                )
                            else:
                                explainer = shap.Explainer(explain_target_method_fn, background_data)
                            return handlers_utils.convert_explanations_to_2D_df(base_model, explainer(data).values)
                except Exception:
                    pass  # Do nothing and continue to the next method
            raise ValueError("Explainability for this model is not supported.")

        return explain_fn

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "BaseEstimator",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.SNOWModelLoadOptions],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "BaseEstimator",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "BaseEstimator",
                signature: model_signature.ModelSignature,
                target_method: str,
                background_data: Optional[pd.DataFrame] = None,
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
                    fn = cls._build_explain_fn(raw_model, background_data)
                    return model_signature_utils.rename_pandas_df(fn(X), signature.outputs)

                if target_method == "explain":
                    return explain_fn

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name, background_data)

            _SnowMLModel = type(
                "_SnowMLModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _SnowMLModel

        _SnowMLModel = _create_custom_model(raw_model, model_meta)
        snowml_model = _SnowMLModel(custom_model.ModelContext())

        return snowml_model
