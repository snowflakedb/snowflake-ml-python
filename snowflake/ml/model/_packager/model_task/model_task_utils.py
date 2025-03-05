import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from snowflake.ml._internal import type_utils
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils

if TYPE_CHECKING:
    import catboost
    import lightgbm
    import sklearn
    import sklearn.pipeline
    import xgboost


@dataclass
class ModelTaskAndOutputType:
    task: type_hints.Task
    output_type: model_signature.DataType


def get_task_skl(model: Union["sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"]) -> type_hints.Task:
    from sklearn.base import is_classifier, is_regressor

    if type_utils.LazyType("sklearn.pipeline.Pipeline").isinstance(model):
        if hasattr(model, "predict_proba") or hasattr(model, "predict"):
            model = model.steps[-1][1]  # type: ignore[attr-defined]
            return _get_model_task(model)
        else:
            return type_hints.Task.UNKNOWN
    if is_regressor(model):
        return type_hints.Task.TABULAR_REGRESSION
    if is_classifier(model):
        classes_list = getattr(model, "classes_", [])
        num_classes = getattr(model, "n_classes_", None) or len(classes_list)
        if isinstance(num_classes, int):
            if num_classes > 2:
                return type_hints.Task.TABULAR_MULTI_CLASSIFICATION
            return type_hints.Task.TABULAR_BINARY_CLASSIFICATION
        return type_hints.Task.UNKNOWN
    return type_hints.Task.UNKNOWN


def get_model_task_catboost(model: "catboost.CatBoost") -> type_hints.Task:
    loss_function = None
    if type_utils.LazyType("catboost.CatBoost").isinstance(model):
        loss_function = model.get_all_params()["loss_function"]  # type: ignore[attr-defined]

    if (type_utils.LazyType("catboost.CatBoostClassifier").isinstance(model)) or model._is_classification_objective(
        loss_function
    ):
        num_classes = handlers_utils.get_num_classes_if_exists(model)
        if num_classes == 0:
            return type_hints.Task.UNKNOWN
        if num_classes <= 2:
            return type_hints.Task.TABULAR_BINARY_CLASSIFICATION
        return type_hints.Task.TABULAR_MULTI_CLASSIFICATION
    if (type_utils.LazyType("catboost.CatBoostRanker").isinstance(model)) or model._is_ranking_objective(loss_function):
        return type_hints.Task.TABULAR_RANKING
    if (type_utils.LazyType("catboost.CatBoostRegressor").isinstance(model)) or model._is_regression_objective(
        loss_function
    ):
        return type_hints.Task.TABULAR_REGRESSION

    return type_hints.Task.UNKNOWN


def get_model_task_lightgbm(model: Union["lightgbm.Booster", "lightgbm.LGBMModel"]) -> type_hints.Task:

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

    # does not account for cross-entropy and custom
    model_task = ""
    if type_utils.LazyType("lightgbm.Booster").isinstance(model):
        model_task = model.params["objective"]  # type: ignore[attr-defined]
    elif hasattr(model, "objective_"):
        model_task = model.objective_  # type: ignore[assignment]
    if model_task in _BINARY_CLASSIFICATION_OBJECTIVES:
        return type_hints.Task.TABULAR_BINARY_CLASSIFICATION
    if model_task in _MULTI_CLASSIFICATION_OBJECTIVES:
        return type_hints.Task.TABULAR_MULTI_CLASSIFICATION
    if model_task in _RANKING_OBJECTIVES:
        return type_hints.Task.TABULAR_RANKING
    if model_task in _REGRESSION_OBJECTIVES:
        return type_hints.Task.TABULAR_REGRESSION
    return type_hints.Task.UNKNOWN


def get_model_task_xgb(model: Union["xgboost.Booster", "xgboost.XGBModel"]) -> type_hints.Task:

    _BINARY_CLASSIFICATION_OBJECTIVE_PREFIX = ["binary:"]
    _MULTI_CLASSIFICATION_OBJECTIVE_PREFIX = ["multi:"]
    _RANKING_OBJECTIVE_PREFIX = ["rank:"]
    _REGRESSION_OBJECTIVE_PREFIX = ["reg:"]

    model_task = ""
    if type_utils.LazyType("xgboost.Booster").isinstance(model):
        model_params = json.loads(model.save_config())  # type: ignore[attr-defined]
        model_task = model_params.get("learner", {}).get("objective", "")
    else:
        if hasattr(model, "get_params"):
            model_task = model.get_params().get("objective", "")

    if isinstance(model_task, dict):
        model_task = model_task.get("name", "")
    for classification_objective in _BINARY_CLASSIFICATION_OBJECTIVE_PREFIX:
        if classification_objective in model_task:
            return type_hints.Task.TABULAR_BINARY_CLASSIFICATION
    for classification_objective in _MULTI_CLASSIFICATION_OBJECTIVE_PREFIX:
        if classification_objective in model_task:
            return type_hints.Task.TABULAR_MULTI_CLASSIFICATION
    for ranking_objective in _RANKING_OBJECTIVE_PREFIX:
        if ranking_objective in model_task:
            return type_hints.Task.TABULAR_RANKING
    for regression_objective in _REGRESSION_OBJECTIVE_PREFIX:
        if regression_objective in model_task:
            return type_hints.Task.TABULAR_REGRESSION
    return type_hints.Task.UNKNOWN


def _get_model_task(model: Any) -> type_hints.Task:
    if type_utils.LazyType("xgboost.Booster").isinstance(model) or type_utils.LazyType("xgboost.XGBModel").isinstance(
        model
    ):
        return get_model_task_xgb(model)

    if type_utils.LazyType("lightgbm.Booster").isinstance(model) or type_utils.LazyType(
        "lightgbm.LGBMModel"
    ).isinstance(model):
        return get_model_task_lightgbm(model)

    if type_utils.LazyType("catboost.CatBoost").isinstance(model):
        return get_model_task_catboost(model)

    if type_utils.LazyType("sklearn.base.BaseEstimator").isinstance(model) or type_utils.LazyType(
        "sklearn.pipeline.Pipeline"
    ).isinstance(model):
        return get_task_skl(model)
    raise ValueError(f"Model type {type(model)} is not supported")


def resolve_model_task_and_output_type(model: Any, passed_model_task: type_hints.Task) -> ModelTaskAndOutputType:
    inferred_task = _get_model_task(model)
    task = handlers_utils.validate_model_task(passed_model_task, inferred_task)
    output_type = model_signature.DataType.DOUBLE
    if task == type_hints.Task.TABULAR_MULTI_CLASSIFICATION:
        output_type = model_signature.DataType.STRING
    return ModelTaskAndOutputType(task=task, output_type=output_type)
