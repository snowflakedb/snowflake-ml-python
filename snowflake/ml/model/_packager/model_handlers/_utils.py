import json
from typing import Any, Callable, Iterable, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

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
    sample_input_data: Optional[model_types.SupportedDataType],
    get_prediction_fn: Callable[[str, model_types.SupportedLocalDataType], model_types.SupportedLocalDataType],
) -> model_meta.ModelMetadata:
    if model_meta.signatures:
        validate_target_methods(model, list(model_meta.signatures.keys()))
        return model_meta

    # In this case sample_input_data should be available, because of the check in save_model.
    assert (
        sample_input_data is not None
    ), "Model signature and sample input are None at the same time. This should not happen with local model."
    trunc_sample_input = model_signature._truncate_data(sample_input_data)
    if isinstance(sample_input_data, SnowparkDataFrame):
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


def add_explain_method_signature(
    model_meta: model_meta.ModelMetadata,
    explain_method: str,
    target_method: str,
    output_return_type: model_signature.DataType = model_signature.DataType.DOUBLE,
) -> model_meta.ModelMetadata:
    if target_method not in model_meta.signatures:
        raise ValueError(f"Signature for target method {target_method} is missing")
    inputs = model_meta.signatures[target_method].inputs
    model_meta.signatures[explain_method] = model_signature.ModelSignature(
        inputs=inputs,
        outputs=[
            model_signature.FeatureSpec(dtype=output_return_type, name=f"{spec.name}_explanation") for spec in inputs
        ],
    )
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


def get_num_classes_if_exists(model: model_types.SupportedModelType) -> int:
    num_classes = getattr(model, "classes_", [])
    return len(num_classes)


def convert_explanations_to_2D_df(
    model: model_types.SupportedModelType, explanations: npt.NDArray[Any]
) -> pd.DataFrame:
    if explanations.ndim != 3:
        return pd.DataFrame(explanations)

    if hasattr(model, "classes_"):
        classes_list = [cl for cl in model.classes_]  # type:ignore[union-attr]
        len_classes = len(classes_list)
        if explanations.shape[2] != len_classes:
            raise ValueError(f"Model has {len_classes} classes but explanations have {explanations.shape[2]}")
    else:
        classes_list = [i for i in range(explanations.shape[2])]
    exp_2d = []
    # TODO (SNOW-1549044): Optimize this
    for row in explanations:
        col_list = []
        for column in row:
            class_explanations = {}
            for cl, cl_exp in zip(classes_list, column):
                if isinstance(cl, (int, np.integer)):
                    cl = int(cl)
                class_explanations[cl] = cl_exp
            col_list.append(json.dumps(class_explanations))
        exp_2d.append(col_list)

    return pd.DataFrame(exp_2d)
