_KEEP_ORDER_COL_NAME = "_ID"

_UDF_CODE_TEMPLATE = """
import pandas as pd
import numpy as np
import sys
from _snowflake import vectorized
import os
import fcntl
import threading
import zipfile
import anyio
import inspect

class FileLock:
   def __enter__(self):
      self._lock = threading.Lock()
      self._lock.acquire()
      self._fd = open('/tmp/lockfile.LOCK', 'w+')
      fcntl.lockf(self._fd, fcntl.LOCK_EX)

   def __exit__(self, type, value, traceback):
      self._fd.close()
      self._lock.release()

IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]

model_dir_name = os.path.splitext('{model_stage_file_name}')[0]
zip_model_path = os.path.join(import_dir, '{model_stage_file_name}')
extracted = '/tmp/models'
extracted_model_dir_path = os.path.join(extracted, model_dir_name)

with FileLock():
    if not os.path.isdir(extracted_model_dir_path):
        with zipfile.ZipFile(zip_model_path, 'r') as myzip:
            myzip.extractall(extracted_model_dir_path)

sys.path.insert(0, os.path.join(extracted_model_dir_path, "{code_dir_name}"))
from snowflake.ml.model import _model
# Backward for <= 1.0.5
if hasattr(_model, "_load_model_for_deploy"):
    model, meta = _model._load_model_for_deploy(extracted_model_dir_path)
else:
    model, meta = _model._load(local_dir_path=extracted_model_dir_path, as_custom_model=True)

features = meta.signatures["{target_method}"].inputs
input_cols = [feature.name for feature in features]
dtype_map = {{feature.name: feature.as_dtype() for feature in features}}

# TODO(halu): Wire `max_batch_size`.
# TODO(halu): Avoid per batch async detection branching.
@vectorized(input=pd.DataFrame, max_batch_size=10)
def infer(df):
    input_df = pd.json_normalize(df[0]).astype(dtype=dtype_map)
    if inspect.iscoroutinefunction(model.{target_method}):
        predictions_df = anyio.run(model.{target_method}, input_df[input_cols])
    else:
        predictions_df = model.{target_method}(input_df[input_cols])

    if "{_KEEP_ORDER_COL_NAME}" in input_df.columns:
        predictions_df["{_KEEP_ORDER_COL_NAME}"] = input_df["{_KEEP_ORDER_COL_NAME}"]

    return predictions_df.to_dict("records")
"""
