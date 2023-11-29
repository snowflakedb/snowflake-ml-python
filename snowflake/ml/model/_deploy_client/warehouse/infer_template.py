_KEEP_ORDER_COL_NAME = "_ID"

_UDF_CODE_TEMPLATE = """
import fcntl
import functools
import inspect
import os
import sys
import threading
import zipfile
from types import TracebackType
from typing import Optional, Type

import anyio
import pandas as pd
from _snowflake import vectorized


class FileLock:
    def __enter__(self) -> None:
        self._lock = threading.Lock()
        self._lock.acquire()
        self._fd = open("/tmp/lockfile.LOCK", "w+")
        fcntl.lockf(self._fd, fcntl.LOCK_EX)

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]
    ) -> None:
        self._fd.close()
        self._lock.release()


# User-defined parameters
MODEL_FILE_NAME = "{model_stage_file_name}"
TARGET_METHOD = "{target_method}"
MAX_BATCH_SIZE = None


# Retrieve the model
IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]

model_dir_name = os.path.splitext(MODEL_FILE_NAME)[0]
zip_model_path = os.path.join(import_dir, MODEL_FILE_NAME)
extracted = "/tmp/models"
extracted_model_dir_path = os.path.join(extracted, model_dir_name)

with FileLock():
    if not os.path.isdir(extracted_model_dir_path):
        with zipfile.ZipFile(zip_model_path, "r") as myzip:
            myzip.extractall(extracted_model_dir_path)

sys.path.insert(0, os.path.join(extracted_model_dir_path, "{code_dir_name}"))

# Load the model
try:
    from snowflake.ml.model._packager import model_packager
    pk = model_packager.ModelPackager(extracted_model_dir_path)
    pk.load(as_custom_model=True)
    assert pk.model, "model is not loaded"
    assert pk.meta, "model metadata is not loaded"

    model = pk.model
    meta = pk.meta
except ImportError as e:
    if e.name and not e.name.startswith("snowflake.ml"):
        raise e
    # Support Legacy model
    from snowflake.ml.model import _model
    # Backward for <= 1.0.5
    if hasattr(_model, "_load_model_for_deploy"):
        model, meta = _model._load_model_for_deploy(extracted_model_dir_path)
    else:
        model, meta = _model._load(local_dir_path=extracted_model_dir_path, as_custom_model=True)

# Determine the actual runner
func = getattr(model, TARGET_METHOD)
if inspect.iscoroutinefunction(func):
    runner = functools.partial(anyio.run, func)
else:
    runner = functools.partial(func)

# Determine preprocess parameters
features = meta.signatures[TARGET_METHOD].inputs
input_cols = [feature.name for feature in features]
dtype_map = {{feature.name: feature.as_dtype() for feature in features}}


# Actual handler
@vectorized(input=pd.DataFrame, max_batch_size=MAX_BATCH_SIZE)
def infer(df: pd.DataFrame) -> dict:
    input_df = pd.json_normalize(df[0]).astype(dtype=dtype_map)
    predictions_df = runner(input_df[input_cols])

    if "{_KEEP_ORDER_COL_NAME}" in input_df.columns:
        predictions_df["{_KEEP_ORDER_COL_NAME}"] = input_df["{_KEEP_ORDER_COL_NAME}"]

    return predictions_df.to_dict("records")
"""
