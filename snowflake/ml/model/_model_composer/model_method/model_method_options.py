from typing import TypedDict

from typing_extensions import NotRequired

from snowflake.ml.model import type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import utils
from snowflake.ml.model.volatility import Volatility


class ModelMethodOptions(TypedDict):
    """Options when creating model method.

    case_sensitive: Specify when the name of the method should be considered as case sensitive when registered to SQL.
    function_type: One of `ModelMethodFunctionTypes` specifying function type.
    volatility: One of `Volatility` enum values specifying function volatility.
    model_init_once: When True the init_once template is used (FUNCTION, TABLE_FUNCTION, or partitioned
        table function); when False the standard template is used. Populated by get_model_method_options_from_options()
        from the top-level save option.
    """

    case_sensitive: NotRequired[bool]
    function_type: NotRequired[str]
    volatility: NotRequired[Volatility]
    model_init_once: NotRequired[bool]


def get_model_method_options_from_options(
    options: type_hints.ModelSaveOption,
    target_method: str,
    model_type: str | None = None,
    is_partitioned_function: bool = False,
) -> ModelMethodOptions:
    default_function_type = model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value
    method_option = options.get("method_options", {}).get(target_method, {})
    global_case_sensitive = bool(options.get("case_sensitive", False))
    if "case_sensitive" in method_option:
        case_sensitive = bool(method_option["case_sensitive"])
    else:
        case_sensitive = global_case_sensitive
    if target_method == "explain":
        default_function_type = model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
        case_sensitive = utils.determine_explain_case_sensitive_from_method_options(
            options.get("method_options", {}),
            target_method,
            default=global_case_sensitive,
        )
    elif model_type == "prophet" or (model_type == "tabular_foundation_model" and is_partitioned_function):
        # Prophet needs the entire time series context, and a tabular foundation model served unbound
        # (in-context-learning) receives its context + query rows together as a partition. Both require
        # TABLE_FUNCTION. A tabular foundation model served bound (a pre-fitted estimator, scored row by
        # row) is a plain scalar FUNCTION instead -- it is not partitioned.
        default_function_type = model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
    global_function_type = options.get("function_type", default_function_type)
    function_type = method_option.get("function_type", global_function_type)
    if function_type not in [function_type.value for function_type in model_manifest_schema.ModelMethodFunctionTypes]:
        raise NotImplementedError(f"Function type {function_type} is not supported.")

    default_volatility = options.get("volatility")
    method_volatility = method_option.get("volatility")
    resolved_volatility = method_volatility or default_volatility

    # Only include volatility if explicitly provided in method options
    result: ModelMethodOptions = ModelMethodOptions(
        case_sensitive=case_sensitive,
        function_type=function_type,
    )
    if resolved_volatility:
        result["volatility"] = resolved_volatility

    model_init_once = options.get("model_init_once")
    if model_init_once is not None:
        result["model_init_once"] = model_init_once

    return result
