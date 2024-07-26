import functools
import inspect
import os
import sys

import anyio
import pandas as pd
from _snowflake import vectorized

from snowflake.ml.model._packager import model_packager


# User-defined parameters
MODEL_DIR_REL_PATH = "model"
TARGET_METHOD = "predict"
MAX_BATCH_SIZE = None

# Retrieve the model
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
if inspect.iscoroutinefunction(func):
    runner = functools.partial(anyio.run, func)
else:
    runner = functools.partial(func)

# Determine preprocess parameters
features = meta.signatures[TARGET_METHOD].inputs
input_cols = [feature.name for feature in features]
dtype_map = {feature.name: feature.as_dtype() for feature in features}


# Actual table function
class infer:
    @vectorized(input=pd.DataFrame, max_batch_size=MAX_BATCH_SIZE)
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = input_cols
        input_df = df.astype(dtype=dtype_map)
        return runner(input_df[input_cols])
