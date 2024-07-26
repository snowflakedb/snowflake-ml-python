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
    model_meta_schema,
)
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    import lightgbm


@final
class LGBMModelHandler(_base.BaseModelHandler[Union["lightgbm.Booster", "lightgbm.LGBMModel"]]):
    """Handler for LightGBM based model."""

    HANDLER_TYPE = "lightgbm"
    HANDLER_VERSION = "2024-03-19"
    _MIN_SNOWPARK_ML_VERSION = "1.3.1"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model.pkl"
    DEFAULT_TARGET_METHODS = ["predict", "predict_proba"]
    _BINARY_CLASSIFICATION_OBJECTIVES = ["binary"]
    _MULTI_CLASSIFICATION_OBJECTIVES = ["multiclass", "multiclassova"]
    _RANKING_OBJECTIVES = ["lambdarank", "rank_xendcg"]
    _REGRESSION_OBJECTIVES = [
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "poisson",
        "quantile",
        "tweedie",
        "mape",
        "gamma",
    ]

    @classmethod
    def get_model_objective(cls, model: Union["lightgbm.Booster", "lightgbm.LGBMModel"]) -> _base.ModelObjective:
        import lightgbm

        # does not account for cross-entropy and custom
        if isinstance(model, lightgbm.LGBMClassifier):
            num_classes = handlers_utils.get_num_classes_if_exists(model)
            if num_classes == 2:
                return _base.ModelObjective.BINARY_CLASSIFICATION
            return _base.ModelObjective.MULTI_CLASSIFICATION
        if isinstance(model, lightgbm.LGBMRanker):
            return _base.ModelObjective.RANKING
        if isinstance(model, lightgbm.LGBMRegressor):
            return _base.ModelObjective.REGRESSION
        model_objective = model.params["objective"]
        if model_objective in cls._BINARY_CLASSIFICATION_OBJECTIVES:
            return _base.ModelObjective.BINARY_CLASSIFICATION
        if model_objective in cls._MULTI_CLASSIFICATION_OBJECTIVES:
            return _base.ModelObjective.MULTI_CLASSIFICATION
        if model_objective in cls._RANKING_OBJECTIVES:
            return _base.ModelObjective.RANKING
        if model_objective in cls._REGRESSION_OBJECTIVES:
            return _base.ModelObjective.REGRESSION
        return _base.ModelObjective.UNKNOWN

    @classmethod
    def can_handle(
        cls, model: model_types.SupportedModelType
    ) -> TypeGuard[Union["lightgbm.Booster", "lightgbm.LGBMModel"]]:
        return (
            type_utils.LazyType("lightgbm.Booster").isinstance(model)
            or type_utils.LazyType("lightgbm.LGBMModel").isinstance(model)
        ) and any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> Union["lightgbm.Booster", "lightgbm.LGBMModel"]:
        import lightgbm

        assert isinstance(model, lightgbm.Booster) or isinstance(model, lightgbm.LGBMModel)

        return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: Union["lightgbm.Booster", "lightgbm.LGBMModel"],
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.LGBMModelSaveOptions],
    ) -> None:
        import lightgbm

        assert isinstance(model, lightgbm.Booster) or isinstance(model, lightgbm.LGBMModel)

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
            if kwargs.get("enable_explainability", False):
                output_type = model_signature.DataType.DOUBLE
                if cls.get_model_objective(model) in [
                    _base.ModelObjective.BINARY_CLASSIFICATION,
                    _base.ModelObjective.MULTI_CLASSIFICATION,
                ]:
                    output_type = model_signature.DataType.STRING
                model_meta = handlers_utils.add_explain_method_signature(
                    model_meta=model_meta,
                    explain_method="explain",
                    target_method="predict",
                    output_return_type=output_type,
                )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)

        model_save_path = os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR)
        with open(model_save_path, "wb") as f:
            cloudpickle.dump(model, f)

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
            options=model_meta_schema.LightGBMModelBlobOptions({"lightgbm_estimator_type": model.__class__.__name__}),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(requirement="lightgbm", pip_name="lightgbm"),
                model_env.ModelDependency(requirement="scikit-learn", pip_name="scikit-learn"),
            ],
            check_local_version=True,
        )
        if kwargs.get("enable_explainability", False):
            model_meta.env.include_if_absent(
                [model_env.ModelDependency(requirement="shap", pip_name="shap")],
                check_local_version=True,
            )

        return None

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.LGBMModelLoadOptions],
    ) -> Union["lightgbm.Booster", "lightgbm.LGBMModel"]:
        import lightgbm

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_file_path = os.path.join(model_blob_path, model_blob_filename)

        model_blob_options = cast(model_meta_schema.LightGBMModelBlobOptions, model_blob_metadata.options)
        if "lightgbm_estimator_type" not in model_blob_options:
            raise ValueError("Missing field `lightgbm_estimator_type` in model blob metadata for type `lightgbm`")

        lightgbm_estimator_type = model_blob_options["lightgbm_estimator_type"]
        if not hasattr(lightgbm, lightgbm_estimator_type):
            raise ValueError("Type of LightGBM estimator is not supported.")

        assert os.path.isfile(model_blob_file_path)  # saved model is a file
        with open(model_blob_file_path, "rb") as f:
            model = cloudpickle.load(f)
        assert isinstance(model, getattr(lightgbm, lightgbm_estimator_type))

        return model

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: Union["lightgbm.Booster", "lightgbm.XGBModel"],
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.LGBMModelLoadOptions],
    ) -> custom_model.CustomModel:
        import lightgbm

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: Union["lightgbm.Booster", "lightgbm.LGBMModel"],
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: Union["lightgbm.Booster", "lightgbm.LGBMModel"],
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

            _LightGBMModel = type(
                "_LightGBMModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _LightGBMModel

        _LightGBMModel = _create_custom_model(raw_model, model_meta)
        lightgbm_model = _LightGBMModel(custom_model.ModelContext())

        return lightgbm_model
