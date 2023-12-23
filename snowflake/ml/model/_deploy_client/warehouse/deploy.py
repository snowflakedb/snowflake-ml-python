import copy
import logging
import posixpath
import tempfile
import textwrap
from types import ModuleType
from typing import IO, List, Optional, Tuple, TypedDict, Union

from typing_extensions import Unpack

from snowflake.ml._internal import env_utils, file_utils
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._deploy_client.warehouse import infer_template
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.snowpark import session as snowpark_session, types as st

logger = logging.getLogger(__name__)


def _deploy_to_warehouse(
    session: snowpark_session.Session,
    *,
    model_stage_file_path: str,
    model_meta: model_meta.ModelMetadata,
    udf_name: str,
    target_method: str,
    **kwargs: Unpack[model_types.WarehouseDeployOptions],
) -> None:
    """Deploy the model to warehouse as UDF.

    Args:
        session: Snowpark session.
        model_stage_file_path: Path to the stored model zip file in the stage.
        model_meta: Model Metadata.
        udf_name: Name of the UDF.
        target_method: The name of the target method to be deployed.
        **kwargs: Options that control some features in generated udf code.

    Raises:
        SnowflakeMLException: Raised when model file name is unable to encoded using ASCII.
        SnowflakeMLException: Raised when incompatible model.
        SnowflakeMLException: Raised when target method does not exist in model.
        SnowflakeMLException: Raised when confronting invalid stage location.

    """
    # TODO(SNOW-862576): Should remove check on ASCII encoding after SNOW-862576 fixed.
    model_stage_file_name = posixpath.basename(model_stage_file_path)
    if not file_utils._able_ascii_encode(model_stage_file_name):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Model file name {model_stage_file_name} cannot be encoded using ASCII. Please rename."
            ),
        )

    relax_version = kwargs.get("relax_version", False)

    if target_method not in model_meta.signatures.keys():
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(f"Target method {target_method} does not exist in model."),
        )

    final_packages = _get_model_final_packages(model_meta, session, relax_version=relax_version)

    stage_location = kwargs.get("permanent_udf_stage_location", None)
    if stage_location:
        stage_location = posixpath.normpath(stage_location.strip())
        if not stage_location.startswith("@"):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid stage location {stage_location}."),
            )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        _write_UDF_py_file(f.file, model_stage_file_name=model_stage_file_name, target_method=target_method, **kwargs)
        logger.info(f"Generated UDF file is persisted at: {f.name}")

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
            name=udf_name,
            return_type=st.PandasSeriesType(st.MapType(st.StringType(), st.VariantType())),
            input_types=[st.PandasDataFrameType([st.MapType()])],
            imports=[model_stage_file_path],
            packages=list(final_packages),
        )
        if stage_location is None:  # Temporary UDF
            session.udf.register_from_file(**params, replace=True)
        else:  # Permanent UDF
            session.udf.register_from_file(
                **params,
                replace=kwargs.get("replace_udf", False),
                is_permanent=True,
                stage_location=stage_location,
            )

    logger.info(f"{udf_name} is deployed to warehouse.")


def _write_UDF_py_file(
    f: IO[str],
    model_stage_file_name: str,
    target_method: str,
    **kwargs: Unpack[model_types.WarehouseDeployOptions],
) -> None:
    """Generate and write UDF python code into a file

    Args:
        f: File descriptor to write the python code.
        model_stage_file_name: Model zip file name.
        target_method: The name of the target method to be deployed.
        **kwargs: Options that control some features in generated udf code.
    """
    udf_code = infer_template._UDF_CODE_TEMPLATE.format(
        model_stage_file_name=model_stage_file_name,
        _KEEP_ORDER_COL_NAME=infer_template._KEEP_ORDER_COL_NAME,
        target_method=target_method,
        code_dir_name=model_meta.MODEL_CODE_DIR,
    )
    f.write(udf_code)
    f.flush()


def _get_model_final_packages(
    meta: model_meta.ModelMetadata,
    session: snowpark_session.Session,
    relax_version: Optional[bool] = False,
) -> List[str]:
    """Generate final packages list of dependency of a model to be deployed to warehouse.

    Args:
        meta: Model metadata to get dependency information.
        session: Snowpark connection session.
        relax_version: Whether or not relax the version restriction when fail to resolve dependencies.
            Defaults to False.

    Raises:
        SnowflakeMLException: Raised when PIP requirements and dependencies from non-Snowflake anaconda channel found.
        SnowflakeMLException: Raised when not all packages are available in snowflake conda channel.

    Returns:
        List of final packages string that is accepted by Snowpark register UDF call.
    """

    if (
        any(channel.lower() not in [env_utils.DEFAULT_CHANNEL_NAME] for channel in meta.env._conda_dependencies.keys())
        or meta.env.pip_requirements
    ):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.DEPENDENCY_VERSION_ERROR,
            original_exception=RuntimeError(
                "PIP requirements and dependencies from non-Snowflake anaconda channel is not supported."
            ),
        )

    if relax_version:
        relaxed_env = copy.deepcopy(meta.env)
        relaxed_env.relax_version()
        required_packages = relaxed_env._conda_dependencies[env_utils.DEFAULT_CHANNEL_NAME]
    else:
        required_packages = meta.env._conda_dependencies[env_utils.DEFAULT_CHANNEL_NAME]

    package_availability_dict = env_utils.get_matched_package_versions_in_information_schema(
        session, required_packages, python_version=meta.env.python_version
    )
    no_version_available_packages = [
        req_name for req_name, ver_list in package_availability_dict.items() if len(ver_list) < 1
    ]
    unavailable_packages = [req.name for req in required_packages if req.name not in package_availability_dict]
    if no_version_available_packages or unavailable_packages:
        relax_version_info_str = "" if relax_version else "Try to set relax_version as True in the options. "
        required_package_str = " ".join(map(lambda x: f'"{x}"', required_packages))
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.DEPENDENCY_VERSION_ERROR,
            original_exception=RuntimeError(
                textwrap.dedent(
                    f"""
                The model's dependencies are not available in Snowflake Anaconda Channel. {relax_version_info_str}
                Required packages are: {required_package_str}
                Required Python version is: {meta.env.python_version}
                Packages that are not available are: {unavailable_packages}
                Packages that cannot meet your requirements are: {no_version_available_packages}
                Package availability information of those you requested is: {package_availability_dict}
                """
                ),
            ),
        )
    return list(sorted(map(str, required_packages)))
