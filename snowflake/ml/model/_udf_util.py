import os
import tempfile
import warnings
from types import ModuleType
from typing import IO, List, Optional, Tuple, TypedDict, Union

from typing_extensions import Unpack

from snowflake.ml._internal import env_utils
from snowflake.ml.model import (
    _env as model_env,
    _model,
    _model_meta,
    type_hints as model_types,
)
from snowflake.snowpark import session as snowpark_session, types as st

_KEEP_ORDER_COL_NAME = "_ID"

_KEEP_ORDER_CODE_TEMPLATE = f'predictions_df["{_KEEP_ORDER_COL_NAME}"] = input_df["{_KEEP_ORDER_COL_NAME}"]'

_EXTRACT_LOCAL_MODEL_CODE = """
model_dir_name = '{model_dir_name}'
zip_model_path = os.path.join(import_dir, '{model_dir_name}.zip')
extracted = '/tmp/models'
extracted_model_dir_path = os.path.join(extracted, model_dir_name)

with FileLock():
    if not os.path.isdir(extracted_model_dir_path):
        with zipfile.ZipFile(zip_model_path, 'r') as myzip:
            myzip.extractall(extracted)
"""

_EXTRACT_STAGE_MODEL_CODE = """
model_dir_name = os.path.splitext('{model_stage_file_name}')[0]
zip_model_path = os.path.join(import_dir, '{model_stage_file_name}')
extracted = '/tmp/models'
extracted_model_dir_path = os.path.join(extracted, model_dir_name)

with FileLock():
    if not os.path.isdir(extracted_model_dir_path):
        with zipfile.ZipFile(zip_model_path, 'r') as myzip:
            myzip.extractall(extracted_model_dir_path)
"""

_SNOWML_IMPORT_CODE = """

snowml_filename = '{snowml_filename}'
snowml_path = import_dir + snowml_filename
snowml_extracted = '/tmp/' + snowml_filename
with FileLock():
    if not os.path.isdir(snowml_extracted):
        with zipfile.ZipFile(snowml_path, 'r') as myzip:
            myzip.extractall(snowml_extracted)
sys.path.insert(0, snowml_extracted)
"""

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

{snowml_import_code}

from snowflake.ml.model._model import _load_model_for_deploy

{extract_model_code}

model, meta = _load_model_for_deploy(extracted_model_dir_path)

# TODO(halu): Wire `max_batch_size`.
# TODO(halu): Avoid per batch async detection branching.
@vectorized(input=pd.DataFrame, max_batch_size=10)
def infer(df):
    input_cols = [spec.name for spec in meta.signatures["{target_method}"].inputs]
    input_df = pd.io.json.json_normalize(df[0])
    if inspect.iscoroutinefunction(model.{target_method}):
        predictions_df = anyio.run(model.{target_method}, input_df[input_cols])
    else:
        predictions_df = model.{target_method}(input_df[input_cols])

    {keep_order_code}

    return predictions_df.to_dict("records")
"""


def _deploy_to_warehouse(
    session: snowpark_session.Session,
    *,
    model_dir_path: Optional[str] = None,
    model_stage_file_path: Optional[str] = None,
    udf_name: str,
    target_method: str,
    **kwargs: Unpack[model_types.WarehouseDeployOptions],
) -> _model_meta.ModelMetadata:
    """Deploy the model to warehouse as UDF.

    Args:
        session: Snowpark session.
        model_dir_path: Path to model directory. Exclusive with model_stage_file_path.
        model_stage_file_path: Path to the stored model zip file in the stage. Exclusive with model_dir_path.
        udf_name: Name of the UDF.
        target_method: The name of the target method to be deployed.
        **kwargs: Options that control some features in generated udf code.

    Raises:
        ValueError: Raised when incompatible model.
        ValueError: Raised when target method does not exist in model.

    Returns:
        The metadata of the model deployed.
    """
    if model_dir_path:
        model_dir_path = os.path.normpath(model_dir_path)
        model_dir_name = os.path.basename(model_dir_path)
        extract_model_code = _EXTRACT_LOCAL_MODEL_CODE.format(model_dir_name=model_dir_name)
        meta = _model.load_model(model_dir_path=model_dir_path, meta_only=True)
    else:
        assert model_stage_file_path is not None, "Unreachable assertion error."
        model_stage_file_name = os.path.basename(model_stage_file_path)
        extract_model_code = _EXTRACT_STAGE_MODEL_CODE.format(model_stage_file_name=model_stage_file_name)
        meta = _model.load_model(session=session, model_stage_file_path=model_stage_file_path, meta_only=True)

    relax_version = kwargs.get("relax_version", False)

    if target_method not in meta.signatures.keys():
        raise ValueError(f"Target method {target_method} does not exist in model.")

    _snowml_wheel_path = kwargs.get("_snowml_wheel_path", None)

    final_packages = _get_model_final_packages(meta, session, relax_version=relax_version)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        _write_UDF_py_file(f.file, extract_model_code, target_method, **kwargs)
        print(f"Generated UDF file is persisted at: {f.name}")
        imports = (
            ([model_dir_path] if model_dir_path else [])
            + ([model_stage_file_path] if model_stage_file_path else [])
            + ([_snowml_wheel_path] if _snowml_wheel_path else [])
        )

        stage_location = kwargs.get("permanent_udf_stage_location", None)

        class _UDFParams(TypedDict):
            file_path: str
            func_name: str
            name: str
            input_types: List[st.DataType]
            return_type: st.DataType
            imports: List[Union[str, Tuple[str, str]]]
            packages: List[Union[str, ModuleType]]

        params = _UDFParams(
            file_path=f.name,
            func_name="infer",
            name=f"{udf_name}",
            return_type=st.PandasSeriesType(st.MapType(st.StringType(), st.VariantType())),
            input_types=[st.PandasDataFrameType([st.MapType()])],
            imports=list(imports),
            packages=list(final_packages),
        )
        if stage_location is None:  # Temporary UDF
            session.udf.register_from_file(**params, replace=True)
        else:  # Permanent UDF
            stage_location = stage_location.rstrip().rstrip("/")
            session.udf.register_from_file(**params, replace=False, is_permanent=True, stage_location=stage_location)

    print(f"{udf_name} is deployed to warehouse.")
    return meta


def _write_UDF_py_file(
    f: IO[str],
    extract_model_code: str,
    target_method: str,
    **kwargs: Unpack[model_types.WarehouseDeployOptions],
) -> None:
    """Generate and write UDF python code into a file

    Args:
        f: File descriptor to write the python code.
        extract_model_code: Code to extract the model.
        target_method: The name of the target method to be deployed.
        **kwargs: Options that control some features in generated udf code.
    """
    keep_order = kwargs.get("keep_order", True)
    snowml_wheel_path = kwargs.get("_snowml_wheel_path", None)
    if snowml_wheel_path:
        whl_filename = os.path.basename(snowml_wheel_path)
        snowml_import_code = _SNOWML_IMPORT_CODE.format(snowml_filename=whl_filename)

    udf_code = _UDF_CODE_TEMPLATE.format(
        extract_model_code=extract_model_code,
        keep_order_code=_KEEP_ORDER_CODE_TEMPLATE if keep_order else "",
        target_method=target_method,
        snowml_import_code=snowml_import_code if snowml_wheel_path else "",
    )
    f.write(udf_code)
    f.flush()


def _get_model_final_packages(
    meta: _model_meta.ModelMetadata, session: snowpark_session.Session, relax_version: Optional[bool] = False
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
        any(channel.lower() not in ["", "snowflake"] for channel in meta._conda_dependencies.keys())
        or meta.pip_requirements
    ):
        raise RuntimeError("PIP requirements and dependencies from non-Snowflake anaconda channel is not supported.")
    try:
        final_packages = env_utils.resolve_conda_environment(
            meta._conda_dependencies[""], [model_env._SNOWFLAKE_CONDA_CHANNEL_URL], python_version=meta.python_version
        )
        if final_packages is None and relax_version:
            final_packages = env_utils.resolve_conda_environment(
                list(map(env_utils.relax_requirement_version, meta._conda_dependencies[""])),
                [model_env._SNOWFLAKE_CONDA_CHANNEL_URL],
                python_version=meta.python_version,
            )
    except ImportError:
        warnings.warn(
            "Cannot find conda resolver, use Snowflake information schema for best-effort dependency pre-check.",
            category=RuntimeWarning,
        )
        final_packages = env_utils.validate_requirements_in_snowflake_conda_channel(
            session=session,
            reqs=meta._conda_dependencies[""],
            python_version=meta.python_version,
        )
        if final_packages is None and relax_version:
            final_packages = env_utils.validate_requirements_in_snowflake_conda_channel(
                session=session,
                reqs=list(map(env_utils.relax_requirement_version, meta._conda_dependencies[""])),
                python_version=meta.python_version,
            )

    finally:
        if final_packages is None:
            raise RuntimeError(
                "The model's dependency cannot fit into Snowflake Warehouse. "
                + "Trying to set relax_version as True in the options. Required packages are:\n"
                + '"'
                + " ".join(map(str, meta._conda_dependencies[""]))
                + '"'
            )
    return final_packages
