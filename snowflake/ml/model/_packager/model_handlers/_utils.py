from typing import Callable, Iterable, Optional, Sequence, cast

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame


def _is_callable(model: model_types.SupportedModelType, method_name: str) -> bool:
    return callable(getattr(model, method_name, None))


def validate_signature(
    model: model_types.SupportedRequireSignatureModelType,
    model_meta: model_meta.ModelMetadata,
    target_methods: Iterable[str],
    sample_input: Optional[model_types.SupportedDataType],
    get_prediction_fn: Callable[[str, model_types.SupportedLocalDataType], model_types.SupportedLocalDataType],
) -> model_meta.ModelMetadata:
    if model_meta.signatures:
        validate_target_methods(model, list(model_meta.signatures.keys()))
        return model_meta

    # In this case sample_input should be available, because of the check in save_model.
    assert (
        sample_input is not None
    ), "Model signature and sample input are None at the same time. This should not happen with local model."
    trunc_sample_input = model_signature._truncate_data(sample_input)
    if isinstance(sample_input, SnowparkDataFrame):
        # Added because of Any from missing stubs.
        trunc_sample_input = cast(SnowparkDataFrame, trunc_sample_input)
        local_sample_input = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(trunc_sample_input)
    else:
        local_sample_input = trunc_sample_input
    for target_method in target_methods:
        predictions_df = get_prediction_fn(target_method, local_sample_input)
        sig = model_signature.infer_signature(local_sample_input, predictions_df)
        model_meta.signatures[target_method] = sig
    return model_meta


def get_target_methods(
    model: model_types.SupportedModelType,
    target_methods: Optional[Sequence[str]],
    default_target_methods: Iterable[str],
) -> Sequence[str]:
    if target_methods is None:
        target_methods = [method_name for method_name in default_target_methods if _is_callable(model, method_name)]

    validate_target_methods(model, target_methods)
    return target_methods


def validate_target_methods(model: model_types.SupportedModelType, target_methods: Iterable[str]) -> None:
    for method_name in target_methods:
        if not _is_callable(model, method_name):
            raise ValueError(f"Target method {method_name} is not callable or does not exist in the model.")
