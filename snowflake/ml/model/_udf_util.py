import os
import tempfile
import warnings
from typing import IO, List, Optional, Tuple

from snowflake.ml._internal import env_utils, file_utils as snowml_file_utils
from snowflake.ml.model import model, model_meta, model_types
from snowflake.snowpark import session as snowpark_session, types as st

_SNOWFLAKE_CONDA_CHANNEL_URL = "https://repo.anaconda.com/pkgs/snowflake"

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
    input_cols = [spec.name for spec in meta.signature.inputs]
    input_df = pd.io.json.json_normalize(df[0])
    if inspect.iscoroutinefunction(model.predict):
        predictions_df = anyio.run(model.predict, input_df[input_cols])
    else:
        predictions_df = model.predict(input_df[input_cols])

    return predictions_df.to_dict("records")
"""


# TODO: Take care dedupe if already existed without code change.
def _upload_snowml_to_tmp_stage(
    session: snowpark_session.Session,
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
    session: snowpark_session.Session, *, model_dir_path: str, udf_name: str, relax_version: Optional[bool] = False
) -> Tuple[model_types.ModelType, model_meta.ModelMetadata]:
    """Deploy the model to warehouse as UDF.

    Args:
        session: Snowpark session.
        model_dir_path: Path to model directory.
        udf_name: Name of the UDF.
        relax_version: Whether or not relax the version restriction when fail to resolve dependencies.
            Defaults to False.

    Raises:
        ValueError: Raised when incompatible model.

    Returns:
        A Tuple of the model object and the metadata of the model deployed.
    """
    if not os.path.exists(model_dir_path):
        raise ValueError("Model config did not exist.")
    model_dir_name = os.path.basename(model_dir_path)
    m, meta = model.load_model(model_dir_path)
    final_packages = _get_model_final_packages(meta, session, relax_version)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        _write_UDF_py_file(f.file, model_dir_name)
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
    return m, meta


def _write_UDF_py_file(f: IO[str], model_dir_name: str) -> None:
    """Generate and write UDF python code into a file

    Args:
        f: File descriptor to write the python code.
        model_dir_name: Path to model directory.
    """
    udf_code = _UDF_CODE_TEMPLATE.format(model_dir_name=model_dir_name)
    f.write(udf_code)
    f.flush()


def _get_model_final_packages(
    meta: model_meta.ModelMetadata, session: snowpark_session.Session, relax_version: Optional[bool] = False
) -> List[str]:
    """Generate final packages list of dependency of a model to be deployed to warehouse.

    Args:
        meta: Model metadata to get dependency information.
        session: Snowpark connection session.
        relax_version: Whether or not relax the version restriction when fail to resolve dependencies.
            Defaults to False.

    Raises:
        RuntimeError: Raised when PIP requirements and dependencies from non-Snowflake anaconda channel found.
        RuntimeError: Raised when not all packages are available in snowflake conda channel.

    Returns:
        List of final packages string that is accepted by Snowpark register UDF call.
    """
    final_packages = None
    if (
        len(meta._conda_dependencies.keys()) > 1
        or list(meta._conda_dependencies.keys())[0] != "defaults"
        or meta.pip_requirements
    ):
        raise RuntimeError("PIP requirements and dependencies from non-Snowflake anaconda channel is not supported.")
    try:
        final_packages = env_utils.resolve_conda_environment(
            meta._conda_dependencies["defaults"], [_SNOWFLAKE_CONDA_CHANNEL_URL]
        )
        if final_packages is None and relax_version:
            final_packages = env_utils.resolve_conda_environment(
                list(map(env_utils.relax_requirement_version, meta._conda_dependencies["defaults"])),
                [_SNOWFLAKE_CONDA_CHANNEL_URL],
            )
    except ImportError:
        warnings.warn(
            "Cannot find conda resolver, use Snowflake information schema for best-effort dependency pre-check.",
            category=RuntimeWarning,
        )
        final_packages = env_utils.validate_requirements_in_snowflake_conda_channel(
            session=session, reqs=meta._conda_dependencies["defaults"]
        )
        if final_packages is None and relax_version:
            final_packages = env_utils.validate_requirements_in_snowflake_conda_channel(
                session=session,
                reqs=list(map(env_utils.relax_requirement_version, meta._conda_dependencies["defaults"])),
            )
    finally:
        if final_packages is None:
            raise RuntimeError(
                "The model's dependency cannot fit into Snowflake Warehouse. "
                + "Trying to set relax_version as True in the options."
            )
    return final_packages
