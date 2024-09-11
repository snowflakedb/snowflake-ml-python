import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._packager.model_handlers import _utils as handlers_utils

if TYPE_CHECKING:
    import lightgbm
    import xgboost


@dataclass
class ModelObjectiveAndOutputType:
    objective: type_hints.ModelObjective
    output_type: model_signature.DataType


def get_model_objective_lightgbm(model: Union["lightgbm.Booster", "lightgbm.LGBMModel"]) -> type_hints.ModelObjective:

    import lightgbm

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
    if isinstance(model, lightgbm.LGBMClassifier):
        num_classes = handlers_utils.get_num_classes_if_exists(model)
        if num_classes == 2:
            return type_hints.ModelObjective.BINARY_CLASSIFICATION
        return type_hints.ModelObjective.MULTI_CLASSIFICATION
    if isinstance(model, lightgbm.LGBMRanker):
        return type_hints.ModelObjective.RANKING
    if isinstance(model, lightgbm.LGBMRegressor):
        return type_hints.ModelObjective.REGRESSION
    model_objective = model.params["objective"]
    if model_objective in _BINARY_CLASSIFICATION_OBJECTIVES:
        return type_hints.ModelObjective.BINARY_CLASSIFICATION
    if model_objective in _MULTI_CLASSIFICATION_OBJECTIVES:
        return type_hints.ModelObjective.MULTI_CLASSIFICATION
    if model_objective in _RANKING_OBJECTIVES:
        return type_hints.ModelObjective.RANKING
    if model_objective in _REGRESSION_OBJECTIVES:
        return type_hints.ModelObjective.REGRESSION
    return type_hints.ModelObjective.UNKNOWN


def get_model_objective_xgb(model: Union["xgboost.Booster", "xgboost.XGBModel"]) -> type_hints.ModelObjective:

    import xgboost

    _BINARY_CLASSIFICATION_OBJECTIVE_PREFIX = ["binary:"]
    _MULTI_CLASSIFICATION_OBJECTIVE_PREFIX = ["multi:"]
    _RANKING_OBJECTIVE_PREFIX = ["rank:"]
    _REGRESSION_OBJECTIVE_PREFIX = ["reg:"]

    model_objective = ""
    if isinstance(model, xgboost.Booster):
        model_params = json.loads(model.save_config())
        model_objective = model_params.get("learner", {}).get("objective", "")
    else:
        if hasattr(model, "get_params"):
            model_objective = model.get_params().get("objective", "")

    if isinstance(model_objective, dict):
        model_objective = model_objective.get("name", "")
    for classification_objective in _BINARY_CLASSIFICATION_OBJECTIVE_PREFIX:
        if classification_objective in model_objective:
            return type_hints.ModelObjective.BINARY_CLASSIFICATION
    for classification_objective in _MULTI_CLASSIFICATION_OBJECTIVE_PREFIX:
        if classification_objective in model_objective:
            return type_hints.ModelObjective.MULTI_CLASSIFICATION
    for ranking_objective in _RANKING_OBJECTIVE_PREFIX:
        if ranking_objective in model_objective:
            return type_hints.ModelObjective.RANKING
    for regression_objective in _REGRESSION_OBJECTIVE_PREFIX:
        if regression_objective in model_objective:
            return type_hints.ModelObjective.REGRESSION
    return type_hints.ModelObjective.UNKNOWN


def get_model_objective_and_output_type(model: Any) -> ModelObjectiveAndOutputType:
    import xgboost

    if isinstance(model, xgboost.Booster) or isinstance(model, xgboost.XGBModel):
        model_objective = get_model_objective_xgb(model)
        output_type = model_signature.DataType.DOUBLE
        if model_objective == type_hints.ModelObjective.MULTI_CLASSIFICATION:
            output_type = model_signature.DataType.STRING
        return ModelObjectiveAndOutputType(objective=model_objective, output_type=output_type)

    import lightgbm

    if isinstance(model, lightgbm.Booster) or isinstance(model, lightgbm.LGBMModel):
        model_objective = get_model_objective_lightgbm(model)
        output_type = model_signature.DataType.DOUBLE
        if model_objective in [
            type_hints.ModelObjective.BINARY_CLASSIFICATION,
            type_hints.ModelObjective.MULTI_CLASSIFICATION,
        ]:
            output_type = model_signature.DataType.STRING
        return ModelObjectiveAndOutputType(objective=model_objective, output_type=output_type)

    raise ValueError(f"Model type {type(model)} is not supported")
