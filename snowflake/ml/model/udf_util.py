import os
import tempfile
import types
from typing import List, Union

import snowflake.snowpark.types as st
from snowflake.ml.model.model_meta import ModelMetadata
from snowflake.snowpark import Session
from snowflake.snowpark._internal.utils import zip_file_or_directory_to_stream

_DTYPE_TYPE_MAPPING = {
    "float64": st.FloatType(),
    "float32": st.FloatType(),
    "int64": st.IntegerType(),
    "bool": st.BooleanType(),
    "int32": st.IntegerType(),
}

_UDF_CODE_TEMPLATE = """
import pandas as pd
import sys
from _snowflake import vectorized
from snowflake.ml.model.model import load_model
from snowflake.ml.model.type_spec import to_pd_series, from_pd_dataframe
import os
import fcntl
import threading
import zipfile

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
model_dir_name = '{model_dir_name}'
zip_model_path = os.path.join(import_dir, '{model_dir_name}.zip')
extracted = '/tmp/models'
extracted_model_dir_path = os.path.join(extracted, model_dir_name)

with FileLock():
    if not os.path.isdir(extracted_model_dir_path):
        with zipfile.ZipFile(zip_model_path, 'r') as myzip:
            myzip.extractall(extracted)
model, meta = load_model(extracted_model_dir_path)

#TODO: Wire `max_batch_size`
@vectorized(input=pd.DataFrame, max_batch_size=10)
def infer(df):
    in_data = from_pd_dataframe(meta.input_spec, df)
    predictions = model.predict(in_data)
    return to_pd_series(meta.output_spec, predictions)
"""


# TODO: Take care dedupe if already existed without code change.
def upload_snowml_to_tmp_stage(
    session: Session,
) -> str:
    """Upload model module of snowml to tmp stage.

    Args:
        session (Session): Snowpark session.

    Returns:
        str: Stage path.
    """
    # Only upload `model` module of snowml
    dir_path = os.path.normpath(os.path.join(__file__, ".."))
    # leading path to set up proper import path for the zipimport
    idx = dir_path.find("snowflake")
    leading_path = os.path.abspath(dir_path[:idx])
    tmp_stage = session.get_session_stage()
    filename = "snowml.zip"
    with zip_file_or_directory_to_stream(dir_path, leading_path, add_init_py=True) as input_stream:
        session._conn.upload_stream(
            input_stream=input_stream,
            stage_location=tmp_stage,
            dest_filename=filename,
            dest_prefix="",
            source_compression="DEFLATE",
            compress_data=False,
            overwrite=True,
            is_in_udf=True,
        )
    return f"{tmp_stage}/{filename}"


def deploy_to_warehouse(
    session: Session,
    *,
    model_dir_path: str,
    udf_name: str,
) -> None:
    """Deploy the model to warehouse as UDF.

    Args:
        session (Session): Snowpark session.
        model_dir_path (str): Path to model directory.
        udf_name (str): Name of the UDF.

    Raises:
        ValueError: Raise when incompatible model.
    """
    if not os.path.exists(model_dir_path):
        raise ValueError("Model config did not exist.")
    model_meta = ModelMetadata.load(model_dir_path)
    output_type = model_meta.output_spec.types()[0]
    input_types = model_meta.input_spec.types()
    snowpark_input_types = [_DTYPE_TYPE_MAPPING[dt] for dt in input_types]
    snowpark_output_type = _DTYPE_TYPE_MAPPING[output_type]
    model_dir_name = os.path.basename(model_dir_path)
    udf_code = _UDF_CODE_TEMPLATE.format(model_dir_name=model_dir_name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # TODO: Extract requirements from `requirements.txt`
        # `pyyaml`` and `typing_extensions` are always needed as long as snowml is not in conda.
        final_packages: List[Union[str, types.ModuleType]] = [
            "pandas",
            "pyyaml",
            "typing_extensions",
            "scikit-learn",
            "cloudpickle",
        ]
        f.write(udf_code)
        f.flush()
        print(f"Generated UDF file is persisted at: {f.name}")
        # TODO: Less hacky way to import `snowml`
        snowml_stage_path = upload_snowml_to_tmp_stage(session)
        session.udf.register_from_file(
            file_path=f.name,
            func_name="infer",
            name=f"{udf_name}",
            return_type=st.PandasSeriesType(snowpark_output_type),
            input_types=[st.PandasDataFrameType(snowpark_input_types)],
            replace=True,
            imports=[model_dir_path, snowml_stage_path],
            packages=final_packages,
        )
    print(f"{udf_name} is deployed to warehouse.")
