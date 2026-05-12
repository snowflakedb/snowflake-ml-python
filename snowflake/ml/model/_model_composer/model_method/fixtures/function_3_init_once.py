import functools
import inspect
import json
import os
import sys

import anyio
import numpy as np
import pandas as pd
from _snowflake import vectorized
from _snowflake import udf_init_once

from snowflake.ml.model._packager import model_packager


def _check_param_equality(series):
    """Check if all values in a series are equal, handling unhashable types."""
    try:
        # Fast path: use nunique directly for hashable types
        return series.nunique(dropna=False) <= 1
    except TypeError:
        # Slow path: serialize to JSON for unhashable types (lists, dicts)
        return series.apply(lambda val: json.dumps(val, sort_keys=True)).nunique() <= 1


def _is_null_param_value(val):
    """Check if a param value should use the default (None or NA scalar)."""
    if val is None:
        return True
    if isinstance(val, (list, np.ndarray, dict)):
        return False  # Containers are never considered "null", even if empty
    return pd.isna(val)


# User-defined parameters
MODEL_DIR_REL_PATH = "model"
TARGET_METHOD = "predict"
MAX_BATCH_SIZE = None

# Initialized to None so the table function can detect whether load_model() has run yet.
runner = None


@udf_init_once
def load_model():
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_dir_path = os.path.join(import_dir, MODEL_DIR_REL_PATH)

    # Load the model
    pk = model_packager.ModelPackager(model_dir_path)
    pk.load(as_custom_model=True)
    assert pk.model, "model is not loaded"
    assert pk.meta, "model metadata is not loaded"

    # Determine the actual runner
    model = pk.model
    meta = pk.meta
    func = getattr(model, TARGET_METHOD)

    global runner
    if inspect.iscoroutinefunction(func):
        runner = functools.partial(anyio.run, func)
    else:
        runner = functools.partial(func)

    # Determine preprocess parameters
    features = meta.signatures[TARGET_METHOD].inputs
    global input_cols
    global dtype_map
    input_cols = [feature.name for feature in features]
    dtype_map = {feature.name: feature.as_dtype() for feature in features}

    # Load inference parameters from method signature (if any)
    global param_cols
    global param_defaults
    param_cols = []
    param_defaults = {}
    if hasattr(meta.signatures[TARGET_METHOD], "params") and meta.signatures[TARGET_METHOD].params:
        for param_spec in meta.signatures[TARGET_METHOD].params:
            param_cols.append(param_spec.name)
            param_defaults[param_spec.name] = param_spec.default_value


# Actual table function
class infer:
    @vectorized(input=pd.DataFrame, max_batch_size=MAX_BATCH_SIZE, flatten_object_input=False)
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if runner is None:
            load_model()
        df.columns = input_cols + param_cols
        input_df = df[input_cols].astype(dtype=dtype_map)

        # Extract runtime param values, using defaults if None
        # Validate that all param values are equal (including nulls)
        method_params = {}
        for col in param_cols:
            if not _check_param_equality(df[col]):
                raise ValueError(f"All values for parameter '{col}' must be equal. Please provide a constant value.")
            val = df[col].iloc[0]
            if _is_null_param_value(val):
                method_params[col] = param_defaults[col]
            else:
                method_params[col] = val

        return runner(input_df, **method_params)
