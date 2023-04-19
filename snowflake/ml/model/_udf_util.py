import os
import tempfile
from typing import Optional

from packaging.requirements import Requirement

from snowflake.ml._internal import file_utils as snowml_file_utils
from snowflake.ml.model import _env, _utils
from snowflake.snowpark import Session, types as st

_UDF_CODE_TEMPLATE = """
import pandas as pd
import numpy as np
import sys
from _snowflake import vectorized
from snowflake.ml.model.model import _load_model_for_deploy
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
model_dir_name = '{model_dir_name}'
zip_model_path = os.path.join(import_dir, '{model_dir_name}.zip')
extracted = '/tmp/models'
extracted_model_dir_path = os.path.join(extracted, model_dir_name)

with FileLock():
    if not os.path.isdir(extracted_model_dir_path):
        with zipfile.ZipFile(zip_model_path, 'r') as myzip:
            myzip.extractall(extracted)
model, meta = _load_model_for_deploy(extracted_model_dir_path)

# TODO(halu): Wire `max_batch_size`.
# TODO(halu): Avoid per batch async detection branching.
@vectorized(input=pd.DataFrame, max_batch_size=10)
def infer(df):
    input_cols = [spec.name for spec in meta.schema.inputs]
    input_df = pd.io.json.json_normalize(df[0])
    if inspect.iscoroutinefunction(model.predict):
        predictions_df = anyio.run(model.predict, input_df[input_cols])
    else:
        predictions_df = model.predict(input_df[input_cols])

    return predictions_df.to_dict("records")
"""


# TODO: Take care dedupe if already existed without code change.
def _upload_snowml_to_tmp_stage(
    session: Session,
) -> str:
    """Upload model module of snowml to tmp stage.

    Args:
        session: Snowpark session.

    Returns:
        The stage path to uploaded snowml.zip file.
    """
    # Only upload `model` module of snowml
    dir_path = os.path.normpath(os.path.join(__file__, "..", ".."))
    # leading path to set up proper import path for the zipimport
    idx = dir_path.rfind("snowflake")
    leading_path = os.path.abspath(dir_path[:idx])
    tmp_stage = session.get_session_stage()
    filename = "snowml.zip"
    with snowml_file_utils.zip_file_or_directory_to_stream(dir_path, leading_path) as input_stream:
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


def _deploy_to_warehouse(
    session: Session, *, model_dir_path: str, udf_name: str, relax_version: Optional[bool] = False
) -> None:
    """Deploy the model to warehouse as UDF.

    Args:
        session: Snowpark session.
        model_dir_path: Path to model directory.
        udf_name: Name of the UDF.
        relax_version: Whether or not relax the version restriction when fail to resolve dependencies.

    Raises:
        ValueError: Raised when incompatible model.
        RuntimeError: Raised when not all packages are available in snowflake conda channel.
    """
    if not os.path.exists(model_dir_path):
        raise ValueError("Model config did not exist.")
    model_dir_name = os.path.basename(model_dir_path)
    udf_code = _UDF_CODE_TEMPLATE.format(model_dir_name=model_dir_name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        with open(os.path.join(model_dir_path, _env._REQUIREMENTS_FILE_NAME)) as rf:
            final_packages = None
            deps = rf.readlines()
            _env._validate_dependencies(deps)
            if _utils._resolve_dependencies(deps, [_env._SNOWFLAKE_CONDA_CHANNEL_URL]) is None:
                if relax_version:
                    relaxed_deps = [Requirement(dep).name for dep in deps]
                    if _utils._resolve_dependencies(relaxed_deps, [_env._SNOWFLAKE_CONDA_CHANNEL_URL]) is not None:
                        final_packages = relaxed_deps
            else:
                final_packages = deps
            if final_packages is None:
                raise RuntimeError("Not all dependencies are resolvable in snowflake conda channel.")
        f.write(udf_code)
        f.flush()
        print(f"Generated UDF file is persisted at: {f.name}")
        # TODO: Less hacky way to import `snowml`
        snowml_stage_path = _upload_snowml_to_tmp_stage(session)
        session.udf.register_from_file(
            file_path=f.name,
            func_name="infer",
            name=f"{udf_name}",
            return_type=st.PandasSeriesType(st.MapType(st.StringType(), st.VariantType())),
            input_types=[st.PandasDataFrameType([st.MapType()])],
            replace=True,
            imports=[model_dir_path, snowml_stage_path],
            packages=list(final_packages),
        )
    print(f"{udf_name} is deployed to warehouse.")
