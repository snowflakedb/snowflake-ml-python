import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, cast, final

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
    import catboost


@final
class CatBoostModelHandler(_base.BaseModelHandler["catboost.CatBoost"]):
    """Handler for CatBoost based model."""

    HANDLER_TYPE = "catboost"
    HANDLER_VERSION = "2024-03-21"
    _MIN_SNOWPARK_ML_VERSION = "1.3.1"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model.bin"
    DEFAULT_TARGET_METHODS = ["predict", "predict_proba"]

    @classmethod
    def get_model_objective_and_output_type(cls, model: "catboost.CatBoost") -> model_types.ModelObjective:
        import catboost

        if isinstance(model, catboost.CatBoostClassifier):
            num_classes = handlers_utils.get_num_classes_if_exists(model)
            if num_classes == 2:
                return model_types.ModelObjective.BINARY_CLASSIFICATION
            return model_types.ModelObjective.MULTI_CLASSIFICATION
        if isinstance(model, catboost.CatBoostRanker):
            return model_types.ModelObjective.RANKING
        if isinstance(model, catboost.CatBoostRegressor):
            return model_types.ModelObjective.REGRESSION
        # TODO: Find out model type from the generic Catboost Model
        return model_types.ModelObjective.UNKNOWN

    @classmethod
    def can_handle(cls, model: model_types.SupportedModelType) -> TypeGuard["catboost.CatBoost"]:
        return (type_utils.LazyType("catboost.CatBoost").isinstance(model)) and any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "catboost.CatBoost":
        import catboost

        assert isinstance(model, catboost.CatBoost)

        return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "catboost.CatBoost",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.CatBoostModelSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", True)

        import catboost

        assert isinstance(model, catboost.CatBoost)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str, sample_input_data: model_types.SupportedLocalDataType
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
            inferred_model_objective = cls.get_model_objective_and_output_type(model)
            model_meta.model_objective = handlers_utils.validate_model_objective(
                model_meta.model_objective, inferred_model_objective
            )
            model_objective = model_meta.model_objective
            if enable_explainability:
                output_type = model_signature.DataType.DOUBLE
                if model_objective == model_types.ModelObjective.MULTI_CLASSIFICATION:
                    output_type = model_signature.DataType.STRING
                model_meta = handlers_utils.add_explain_method_signature(
                    model_meta=model_meta,
                    explain_method="explain",
                    target_method="predict",
                    output_return_type=output_type,
                )
                model_meta.function_properties = {
                    "explain": {model_meta_schema.FunctionProperties.PARTITIONED.value: False}
                }

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        model_save_path = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)

        model.save_model(model_save_path)

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.CatBoostModelBlobOptions({"catboost_estimator_type": model.__class__.__name__}),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(requirement="catboost", pip_name="catboost"),
            ],
            check_local_version=True,
        )
        if enable_explainability:
            model_meta.env.include_if_absent([model_env.ModelDependency(requirement="shap", pip_name="shap")])
            model_meta.explain_algorithm = model_meta_schema.ModelExplainAlgorithm.SHAP
        model_meta.env.cuda_version = kwargs.get("cuda_version", model_env.DEFAULT_CUDA_VERSION)

        return None

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.CatBoostModelLoadOptions],
    ) -> "catboost.CatBoost":
        import catboost

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_file_path = os.path.join(model_blob_path, model_blob_filename)

        model_blob_options = cast(model_meta_schema.CatBoostModelBlobOptions, model_blob_metadata.options)
        if "catboost_estimator_type" not in model_blob_options:
            raise ValueError("Missing field `catboost_estimator_type` in model blob metadata for type `catboost`")

        catboost_estimator_type = model_blob_options["catboost_estimator_type"]
        if not hasattr(catboost, catboost_estimator_type):
            raise ValueError("Type of CatBoost estimator is not supported.")

        assert os.path.isfile(model_blob_file_path)  # saved model is a file
        model = getattr(catboost, catboost_estimator_type)()
        model.load_model(model_blob_file_path)
        assert isinstance(model, getattr(catboost, catboost_estimator_type))

        if kwargs.get("use_gpu", False):
            assert type(kwargs.get("use_gpu", False)) == bool
            gpu_params = {"task_type": "GPU"}
            model.__dict__.update(gpu_params)

        return model

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "catboost.CatBoost",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.CatBoostModelLoadOptions],
    ) -> custom_model.CustomModel:
        import catboost

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "catboost.CatBoost",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "catboost.CatBoost",
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

                @custom_model.inference_api
                def explain_fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    import shap

                    explainer = shap.TreeExplainer(raw_model)
                    df = handlers_utils.convert_explanations_to_2D_df(raw_model, explainer(X).values)
                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                if target_method == "explain":
                    return explain_fn

                return fn

            type_method_dict: Dict[str, Any] = {"_raw_model": raw_model}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _CatBoostModel = type(
                "_CatBoostModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _CatBoostModel

        _CatBoostModel = _create_custom_model(raw_model, model_meta)
        catboost_model = _CatBoostModel(custom_model.ModelContext())

        return catboost_model
