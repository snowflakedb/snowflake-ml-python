"""Capture a representative row of sample input data during model logging.

The captured row is written as a separate stage file next to ``model.yaml`` and
recorded on ``ModelMetadata.sample_input_file_paths`` so it can be uploaded with
the rest of the model artifacts.
"""

import datetime
import json
import logging
import math
import os
from functools import reduce
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from snowflake import snowpark
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_meta import model_meta as model_meta_api
from snowflake.ml.model._signatures import core, snowpark_handler

logger = logging.getLogger(__name__)

SAMPLE_INPUT_DATA_FILENAME = "sample_input_data.json"

_ROW_SCAN_LIMIT = 100

_LARGE_TENSOR_ELEMENT_THRESHOLD = 100

_MAX_STRING_LENGTH = 1000
_STRING_TRUNCATION_MARKER = "... (truncated)"


def persist_sample_input_data(
    *,
    sample_input_data: Optional[model_types.SupportedDataType],
    model_meta: model_meta_api.ModelMetadata,
    model_dir_path: str,
) -> None:
    """Capture a representative row of ``sample_input_data`` and write it to ``model_dir_path``.

    On success this writes ``<model_dir_path>/sample_input_data.json`` and
    populates ``model_meta.sample_input_file_paths`` with one entry per method
    whose input signature matches the captured schema.

    Args:
        sample_input_data: User-provided sample input passed to ``log_model``.
        model_meta: Model metadata with already-populated signatures.
        model_dir_path: Local directory in which the JSON file is written.
    """
    if sample_input_data is None or not model_meta.signatures:
        return

    try:
        local_df = _to_truncated_dataframe(sample_input_data)
        if local_df is None:
            return

        reference_method, reference_columns = _find_reference_method(local_df, model_meta.signatures)
        if reference_method is None:
            logger.info("Sample input data does not match any method's input signature; skipping capture.")
            return

        reference_signature = model_meta.signatures[reference_method]
        if _has_large_tensor_spec(reference_signature.inputs):
            logger.info(
                "Sample input shape is too large to capture (method '%s' exceeds the size threshold "
                "of %d elements).",
                reference_method,
                _LARGE_TENSOR_ELEMENT_THRESHOLD,
            )
            return

        aligned_df = local_df.copy()
        aligned_df.columns = reference_columns
        data_row = _select_representative_row(aligned_df)

        payload: dict[str, Any] = {
            "dataframe_split": {
                "index": [0],
                "columns": list(reference_columns),
                "data": [data_row],
            }
        }
        params_dict = _extract_default_params(reference_signature, reference_method)
        if params_dict:
            payload["params"] = params_dict

        json_str = json.dumps(payload)
        file_path = os.path.join(model_dir_path, SAMPLE_INPUT_DATA_FILENAME)
        with open(file_path, "w", encoding="utf-8") as out:
            out.write(json_str)

        for method in _find_matching_methods(reference_columns, model_meta.signatures):
            model_meta.sample_input_file_paths[method] = SAMPLE_INPUT_DATA_FILENAME
    except Exception as exc:
        logger.warning("Could not capture sample input data: %s", exc, exc_info=True)


def _to_truncated_dataframe(
    sample_input_data: model_types.SupportedDataType,
) -> Optional[pd.DataFrame]:
    """Convert any supported sample input format to a pandas DataFrame of up to ``_ROW_SCAN_LIMIT`` rows."""
    if isinstance(sample_input_data, snowpark.DataFrame):
        local_df: pd.DataFrame = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(
            sample_input_data.limit(_ROW_SCAN_LIMIT)
        )
    else:
        truncated = model_signature._truncate_data(sample_input_data, length=_ROW_SCAN_LIMIT)
        local_df = model_signature._convert_local_data_to_df(truncated)

    if local_df is None or local_df.empty:
        return None
    return local_df.reset_index(drop=True)


def _select_representative_row(df: pd.DataFrame) -> list[Any]:
    """Return the row from ``df`` with the fewest null values, as a JSON-serializable list."""
    null_counts_per_row = df.isna().sum(axis=1)
    best_idx = int(null_counts_per_row.idxmin())
    return [_to_json_serializable(value) for value in df.iloc[best_idx].tolist()]


def _static_element_count(spec: Any) -> int:
    """Count the fixed-shape scalar elements a spec contributes to a single captured row.

    A leaf spec contributes the product of its shape's fixed dimensions (a scalar counts as 1).
    A FeatureGroupSpec/ParamGroupSpec multiplies its own shape by the summed counts of its
    children. ``-1`` dimensions are variable-length (not bounded by the schema) and are ignored.

    Args:
        spec: A feature or param spec (FeatureSpec, FeatureGroupSpec, ParamSpec, or ParamGroupSpec).

    Returns:
        The number of fixed-shape elements the spec adds to one captured row.
    """
    shape = getattr(spec, "_shape", None) or ()
    element_count = reduce(lambda a, b: a * b, (dim for dim in shape if dim != -1), 1)
    child_specs = getattr(spec, "_specs", None)
    if child_specs:
        return element_count * sum(_static_element_count(child) for child in child_specs)
    return element_count


def _has_large_tensor_spec(specs: Sequence[Any]) -> bool:
    """Return True if any spec (input or param) exceeds the size threshold."""
    return any(_static_element_count(spec) > _LARGE_TENSOR_ELEMENT_THRESHOLD for spec in specs)


def _find_reference_method(
    df: pd.DataFrame,
    signatures: dict[str, model_signature.ModelSignature],
) -> tuple[Optional[str], list[str]]:
    """Pick a method whose input feature list aligns with the captured DataFrame."""
    # Preference order:
    #   1. A method whose top-level input names exactly match the DataFrame's
    #      columns (set equality, same count).
    #   2. The first method with the same number of top-level inputs.
    df_column_count = len(df.columns)
    df_columns_set = {str(c) for c in df.columns}
    fallback: tuple[Optional[str], list[str]] = (None, [])

    for method, sig in signatures.items():
        feature_names = _input_feature_names(sig)
        if feature_names is None:
            continue
        if len(feature_names) != df_column_count:
            continue
        if set(feature_names) == df_columns_set:
            return method, feature_names
        if fallback[0] is None:
            fallback = (method, feature_names)

    return fallback


def _find_matching_methods(
    reference_columns: list[str],
    signatures: dict[str, model_signature.ModelSignature],
) -> list[str]:
    """Return every method whose input signature has the same top-level feature schema."""
    matching: list[str] = []
    reference_set = set(reference_columns)
    reference_count = len(reference_columns)
    for method, sig in signatures.items():
        feature_names = _input_feature_names(sig)
        if feature_names is None:
            continue
        if len(feature_names) == reference_count and set(feature_names) == reference_set:
            matching.append(method)
    return matching


def _input_feature_names(sig: model_signature.ModelSignature) -> Optional[list[str]]:
    """Return ordered top-level input names (FeatureSpec or FeatureGroupSpec), or None if unrecognized."""
    feature_names: list[str] = []
    for spec in sig.inputs:
        if not isinstance(spec, (core.FeatureSpec, core.FeatureGroupSpec)):
            return None
        feature_names.append(spec.name)
    return feature_names


def _extract_default_params(sig: model_signature.ModelSignature, method_name: str) -> dict[str, Any]:
    """Collect each param's default as a ``{name: default_value}`` mapping for the captured file.

    Capture is all-or-nothing: if any param's default exceeds the size threshold, the whole params
    block is dropped and the server rebuilds omitted params from the same signature defaults.

    Args:
        sig: Signature of the reference method whose params are being captured.
        method_name: Name of the reference method, used only for the diagnostic log message.

    Returns:
        A mapping of param name to JSON-serializable default value, or ``{}`` when params are dropped.
    """
    if _has_large_tensor_spec(sig.params):
        logger.info(
            "Skipping params in sample input data capture for method '%s': a param default exceeds "
            "the size threshold of %d elements.",
            method_name,
            _LARGE_TENSOR_ELEMENT_THRESHOLD,
        )
        return {}
    params: dict[str, Any] = {}
    for spec in sig.params:
        try:
            params[spec.name] = _to_json_serializable(spec.default_value)
        except Exception as exc:
            logger.info("Skipping param '%s' in sample input data capture: %s", spec.name, exc)
            continue
    return params


def _to_json_serializable(value: Any) -> Any:
    """Convert numpy / pandas / datetime scalars into JSON-friendly Python primitives."""
    # Strings longer than _MAX_STRING_LENGTH are truncated with a marker.
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return _truncate_string(str(value))
    if isinstance(value, np.ndarray):
        return [_to_json_serializable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat(sep=" ")
    if isinstance(value, datetime.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, str):
        return _truncate_string(value)
    return value


def _truncate_string(value: str) -> str:
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return value[:_MAX_STRING_LENGTH] + _STRING_TRUNCATION_MARKER
