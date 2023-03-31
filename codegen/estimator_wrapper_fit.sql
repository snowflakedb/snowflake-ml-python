-- TODO: Create separate stored procedures for SkLearn and XGBoost training to reduce the sandbox image size.
CREATE OR REPLACE PROCEDURE estimator_wrapper_fit(
  query STRING,
  stage_transform_file_name STRING,
  stage_result_file_name STRING,
  input_cols ARRAY,
  label_cols ARRAY)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = (
  'snowflake-snowpark-python',
  'numpy==1.23.4',
  'pandas==1.4.4',
  'scikit-learn==1.1.3',
  'scipy==1.9.3',
  'xgboost==1.5.0'
)
HANDLER = 'run'
AS
$$
import joblib
import numpy as np
import os
import pandas
import sklearn
import tempfile
import xgboost

def run(session, sql_query, stage_transform_file_name, stage_result_file_name, input_cols, label_cols):
  df = session.sql(sql_query).to_pandas()

  local_transform_file = tempfile.NamedTemporaryFile(delete=True)
  local_transform_file_name = local_transform_file.name
  local_transform_file.close()

  session.file.get(stage_transform_file_name, local_transform_file_name)

  sklearn_transform = joblib.load(os.path.join(local_transform_file_name, os.listdir(local_transform_file_name)[0]))

  # Special handling for classifier-style transforms which
  # require two arguments to the fit() call.
  if label_cols == None:
    # Regular transform.
    sklearn_transform.fit(df[input_cols])
  else:
    # Classifier transform: Separate label column.
    # NOTE: assuming that label_col is not part of input_cols.
    label = df[label_cols]
    sklearn_transform.fit(X=df[input_cols], y=label)

  local_result_file = tempfile.NamedTemporaryFile(delete=True)
  local_result_file_name = local_result_file.name
  local_result_file.close()

  joblib_dump_files = joblib.dump(sklearn_transform, local_result_file_name)
  session.file.put(local_result_file_name, stage_result_file_name, auto_compress = False, overwrite = True)

  # Note: you can add something like  + "|" + str(df) to the return string
  # to pass debug information to the caller.
  return str(os.path.basename(joblib_dump_files[0]))
$$;
