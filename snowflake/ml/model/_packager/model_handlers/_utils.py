import json
import os
import warnings
from typing import Any, Callable, Iterable, List, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from absl import logging

import snowflake.snowpark.dataframe as sp_df
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.ml.model._signatures import (
    core,
    snowpark_handler,
    utils as model_signature_utils,
)
from snowflake.snowpark import DataFrame as SnowparkDataFrame

EXPLAIN_BACKGROUND_DATA_ROWS_COUNT_LIMIT = 1000


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _is_callable(model: model_types.SupportedModelType, method_name: str) -> bool:
    return callable(getattr(model, method_name, None))


def get_truncated_sample_data(sample_input_data: model_types.SupportedDataType) -> model_types.SupportedLocalDataType:
    trunc_sample_input = model_signature._truncate_data(sample_input_data)
    local_sample_input: model_types.SupportedLocalDataType = None
    if isinstance(sample_input_data, SnowparkDataFrame):
        # Added because of Any from missing stubs.
        trunc_sample_input = cast(SnowparkDataFrame, trunc_sample_input)
        local_sample_input = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(trunc_sample_input)
    else:
        local_sample_input = trunc_sample_input
    return local_sample_input


def validate_signature(
    model: model_types.SupportedRequireSignatureModelType,
    model_meta: model_meta.ModelMetadata,
    target_methods: Iterable[str],
    sample_input_data: Optional[model_types.SupportedDataType],
    get_prediction_fn: Callable[[str, model_types.SupportedLocalDataType], model_types.SupportedLocalDataType],
) -> model_meta.ModelMetadata:
    if model_meta.signatures:
        validate_target_methods(model, list(model_meta.signatures.keys()))
        if sample_input_data is not None:
            local_sample_input = get_truncated_sample_data(sample_input_data)
            for target_method in model_meta.signatures.keys():

                model_signature_inst = model_meta.signatures.get(target_method)
                if model_signature_inst is not None:
                    # strict validation the input signature
                    model_signature._convert_and_validate_local_data(
                        local_sample_input, model_signature_inst._inputs, True
                    )
        return model_meta

    # In this case sample_input_data should be available, because of the check in save_model.
    assert (
        sample_input_data is not None
    ), "Model signature and sample input are None at the same time. This should not happen with local model."
    local_sample_input = get_truncated_sample_data(sample_input_data)
    for target_method in target_methods:
        predictions_df = get_prediction_fn(target_method, local_sample_input)
        sig = model_signature.infer_signature(local_sample_input, predictions_df)
        model_meta.signatures[target_method] = sig

    return model_meta


def get_input_signature(
    model_meta: model_meta.ModelMetadata, target_method: Optional[str]
) -> Sequence[core.BaseFeatureSpec]:
    if target_method is None or target_method not in model_meta.signatures:
        raise ValueError(f"Signature for target method {target_method} is missing or no method to explain.")
    input_sig = model_meta.signatures[target_method].inputs
    return input_sig


def add_explain_method_signature(
    model_meta: model_meta.ModelMetadata,
    explain_method: str,
    target_method: Optional[str],
    output_return_type: model_signature.DataType = model_signature.DataType.DOUBLE,
) -> model_meta.ModelMetadata:
    inputs = get_input_signature(model_meta, target_method)
    if model_meta.model_type == "snowml":
        output_feature_names = [identifier.concat_names([spec.name, "_explanation"]) for spec in inputs]
    else:
        output_feature_names = [f"{spec.name}_explanation" for spec in inputs]
    model_meta.signatures[explain_method] = model_signature.ModelSignature(
        inputs=inputs,
        outputs=[
            model_signature.FeatureSpec(dtype=output_return_type, name=output_name)
            for output_name in output_feature_names
        ],
    )
    return model_meta


def get_explainability_supported_background(
    sample_input_data: Optional[model_types.SupportedDataType],
    meta: model_meta.ModelMetadata,
    explain_target_method: Optional[str],
) -> pd.DataFrame:
    if sample_input_data is None:
        return None

    if isinstance(sample_input_data, pd.DataFrame):
        return sample_input_data
    if isinstance(sample_input_data, sp_df.DataFrame):
        return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sample_input_data)

    df = model_signature._convert_local_data_to_df(sample_input_data)
    input_signature_for_explain = get_input_signature(meta, explain_target_method)
    df_with_named_cols = model_signature_utils.rename_pandas_df(df, input_signature_for_explain)
    return df_with_named_cols


def get_target_methods(
    model: model_types.SupportedModelType,
    target_methods: Optional[Sequence[str]],
    default_target_methods: Iterable[str],
) -> Sequence[str]:
    if target_methods is None:
        target_methods = [method_name for method_name in default_target_methods if _is_callable(model, method_name)]

    validate_target_methods(model, target_methods)
    return target_methods


def save_background_data(
    model_blobs_dir_path: str,
    explain_artifact_dir: str,
    bg_data_file_suffix: str,
    model_name: str,
    background_data: pd.DataFrame,
) -> None:
    data_blob_path = os.path.join(model_blobs_dir_path, explain_artifact_dir)
    os.makedirs(data_blob_path, exist_ok=True)
    with open(os.path.join(data_blob_path, model_name + bg_data_file_suffix), "wb") as f:
        # saving only the truncated data
        trunc_background_data = background_data.head(
            min(len(background_data.index), EXPLAIN_BACKGROUND_DATA_ROWS_COUNT_LIMIT)
        )
        trunc_background_data.to_parquet(f)


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
        classes_list = [str(cl) for cl in model.classes_]  # type:ignore[union-attr]
        len_classes = len(classes_list)
        if explanations.shape[2] != len_classes:
            raise ValueError(f"Model has {len_classes} classes but explanations have {explanations.shape[2]}")
    else:
        classes_list = [str(i) for i in range(explanations.shape[2])]

    def row_to_dict(row: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Converts a single row to a dictionary."""
        # convert to object or numpy creates strings of fixed length
        return np.asarray(json.dumps(dict(zip(classes_list, row)), cls=NumpyEncoder), dtype=object)

    exp_2d = np.apply_along_axis(row_to_dict, -1, explanations)

    return pd.DataFrame(exp_2d)


def validate_model_task(passed_model_task: model_types.Task, inferred_model_task: model_types.Task) -> model_types.Task:
    if passed_model_task != model_types.Task.UNKNOWN and inferred_model_task != model_types.Task.UNKNOWN:
        if passed_model_task != inferred_model_task:
            warnings.warn(
                f"Inferred Task: {inferred_model_task.name} is used as task for this model "
                f"version and passed argument Task: {passed_model_task.name} is ignored",
                category=UserWarning,
                stacklevel=1,
            )
        return inferred_model_task
    elif inferred_model_task != model_types.Task.UNKNOWN:
        logging.info(f"Inferred Task: {inferred_model_task.name} is used as task for this model " f"version")
        return inferred_model_task
    return passed_model_task


def get_explain_target_method(
    model_metadata: model_meta.ModelMetadata, target_methods_list: List[str]
) -> Optional[str]:
    for method in model_metadata.signatures.keys():
        if method in target_methods_list:
            return method
    return None
