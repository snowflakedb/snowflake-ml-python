import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, cast, final

import cloudpickle
import numpy as np
import pandas as pd
from packaging import version
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import (
    _base,
    _utils as handlers_utils,
    model_objective_utils,
)
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    from snowflake.ml.modeling.framework.base import BaseEstimator


@final
class SnowMLModelHandler(_base.BaseModelHandler["BaseEstimator"]):
    """Handler for SnowML based model.

    Currently snowflake.ml.modeling.framework.base.BaseEstimator
        and snowflake.ml.modeling.pipeline.Pipeline based classes are supported.
    """

    HANDLER_TYPE = "snowml"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    DEFAULT_TARGET_METHODS = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]
    EXPLAIN_TARGET_METHODS = ["predict", "predict_proba", "predict_log_proba"]

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
    def _get_local_version_package(cls, pkg_name: str) -> Optional[version.Version]:
        from importlib import metadata as importlib_metadata

        from packaging import version

        local_version = None

        try:
            local_dist = importlib_metadata.distribution(pkg_name)
            local_version = version.parse(local_dist.version)
        except importlib_metadata.PackageNotFoundError:
            pass

        return local_version

    @classmethod
    def _can_support_xgb(cls, enable_explainability: Optional[bool]) -> bool:

        local_xgb_version = cls._get_local_version_package("xgboost")

        if local_xgb_version and local_xgb_version >= version.parse("2.1.0"):
            if enable_explainability:
                warnings.warn(
                    f"This version of xgboost {local_xgb_version} does not work with shap 0.42.1."
                    + "If you want model explanations, lower the xgboost version to <2.1.0.",
                    category=UserWarning,
                    stacklevel=1,
                )
            return False
        return True

    @classmethod
    def _get_supported_object_for_explainability(
        cls, estimator: "BaseEstimator", enable_explainability: Optional[bool]
    ) -> Any:
        from snowflake.ml.modeling import pipeline as snowml_pipeline

        # handle pipeline objects separately
        if isinstance(estimator, snowml_pipeline.Pipeline):  # type: ignore[attr-defined]
            return None

        methods = ["to_xgboost", "to_lightgbm", "to_sklearn"]
        for method_name in methods:
            if hasattr(estimator, method_name):
                try:
                    result = getattr(estimator, method_name)()
                    if method_name == "to_xgboost" and not cls._can_support_xgb(enable_explainability):
                        return None
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
            if model_meta.signatures:
                warnings.warn(
                    "Providing model signature for Snowpark ML "
                    + "Modeling model is not required. Model signature will automatically be inferred during fitting. ",
                    UserWarning,
                    stacklevel=2,
                )
            assert hasattr(model, "model_signatures"), "Model does not have model signatures as expected."
            model_signature_dict = getattr(model, "model_signatures", {})
            target_methods = kwargs.pop("target_methods", None)
            if not target_methods:
                model_meta.signatures = model_signature_dict
            else:
                temp_model_signature_dict = {}
                for method_name in target_methods:
                    method_model_signature = model_signature_dict.get(method_name, None)
                    if method_model_signature is not None:
                        temp_model_signature_dict[method_name] = method_model_signature
                    else:
                        raise ValueError(f"Target method {method_name} does not exist in the model.")
                model_meta.signatures = temp_model_signature_dict

        if enable_explainability or enable_explainability is None:
            python_base_obj = cls._get_supported_object_for_explainability(model, enable_explainability)
            if python_base_obj is None:
                if enable_explainability:  # if user set enable_explainability to True, throw error else silently skip
                    raise ValueError(
                        "Explain only supported for xgboost, lightgbm and sklearn (not pipeline) Snowpark ML models."
                    )
                # set None to False so we don't include shap in the environment
                enable_explainability = False
            else:
                model_task_and_output_type = model_objective_utils.get_model_task_and_output_type(python_base_obj)
                model_meta.task = model_task_and_output_type.task
                explain_target_method = handlers_utils.get_explain_target_method(model_meta, cls.EXPLAIN_TARGET_METHODS)
                model_meta = handlers_utils.add_explain_method_signature(
                    model_meta=model_meta,
                    explain_method="explain",
                    target_method=explain_target_method,
                    output_return_type=model_task_and_output_type.output_type,
                )
                enable_explainability = True

                background_data = handlers_utils.get_explainability_supported_background(
                    sample_input_data, model_meta, explain_target_method
                )
                if background_data is not None:
                    handlers_utils.save_background_data(
                        model_blobs_dir_path, cls.EXPLAIN_ARTIFACTS_DIR, cls.BG_DATA_FILE_SUFFIX, name, background_data
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
            if pkg_name != "xgboost":
                _include_if_absent_pkgs.append(model_env.ModelDependency(requirement=pkg_name, pip_name=pkg_name))
                continue

            local_xgb_version = cls._get_local_version_package("xgboost")
            if local_xgb_version and local_xgb_version >= version.parse("2.0.0") and enable_explainability:
                model_meta.env.include_if_absent(
                    [
                        model_env.ModelDependency(requirement="xgboost==2.0.*", pip_name="xgboost"),
                    ],
                    check_local_version=False,
                )
            else:
                model_meta.env.include_if_absent(
                    [
                        model_env.ModelDependency(requirement="xgboost", pip_name="xgboost"),
                    ],
                    check_local_version=True,
                )

        if enable_explainability:
            model_meta.env.include_if_absent([model_env.ModelDependency(requirement="shap", pip_name="shap")])
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
        ) -> Type[custom_model.CustomModel]:
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
                    import shap

                    methods = ["to_xgboost", "to_lightgbm", "to_sklearn"]
                    for method_name in methods:
                        try:
                            base_model = getattr(raw_model, method_name)()
                            explainer = shap.Explainer(base_model, masker=background_data)
                            df = handlers_utils.convert_explanations_to_2D_df(raw_model, explainer(X).values)
                            return model_signature_utils.rename_pandas_df(df, signature.outputs)
                        except exceptions.SnowflakeMLException:
                            pass  # Do nothing and continue to the next method
                    raise ValueError("The model must be an xgboost, lightgbm or sklearn (not pipeline) estimator.")

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
