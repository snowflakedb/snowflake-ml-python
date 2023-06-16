import os
import tempfile
import warnings
from types import ModuleType
from typing import IO, List, Optional, Tuple, TypedDict, Union

from typing_extensions import Unpack

from snowflake.ml._internal import env as snowml_env, env_utils, file_utils
from snowflake.ml.model import (
    _env as model_env,
    _model,
    _model_meta,
    type_hints as model_types,
)
from snowflake.ml.model._deploy_client.warehouse import infer_template
from snowflake.snowpark import session as snowpark_session, types as st


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
        ValueError: Raised when confronting invalid stage location.

    Returns:
        The metadata of the model deployed.
    """
    if model_dir_path:
        model_dir_path = os.path.normpath(model_dir_path)
        model_dir_name = os.path.basename(model_dir_path)
        extract_model_code = infer_template._EXTRACT_LOCAL_MODEL_CODE.format(model_dir_name=model_dir_name)
        meta = _model.load_model(model_dir_path=model_dir_path, meta_only=True)
    else:
        assert model_stage_file_path is not None, "Unreachable assertion error."
        model_stage_file_name = os.path.basename(model_stage_file_path)
        extract_model_code = infer_template._EXTRACT_STAGE_MODEL_CODE.format(
            model_stage_file_name=model_stage_file_name
        )
        meta = _model.load_model(session=session, model_stage_file_path=model_stage_file_path, meta_only=True)

    relax_version = kwargs.get("relax_version", False)

    if target_method not in meta.signatures.keys():
        raise ValueError(f"Target method {target_method} does not exist in model.")

    _use_local_snowml = kwargs.get("_use_local_snowml", False)

    final_packages = _get_model_final_packages(
        meta, session, relax_version=relax_version, _use_local_snowml=_use_local_snowml
    )

    stage_location = kwargs.get("permanent_udf_stage_location", None)
    if stage_location:
        stage_location = stage_location.strip().rstrip("/")
        if not stage_location.startswith("@"):
            raise ValueError(f"Invalid stage location {stage_location}.")

    _snowml_wheel_path = None
    if _use_local_snowml:
        _snowml_wheel_path = file_utils.upload_snowml(session, stage_location=stage_location)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        _write_UDF_py_file(f.file, extract_model_code, target_method, **kwargs)
        print(f"Generated UDF file is persisted at: {f.name}")
        imports = (
            ([model_dir_path] if model_dir_path else [])
            + ([model_stage_file_path] if model_stage_file_path else [])
            + ([_snowml_wheel_path] if _snowml_wheel_path else [])
        )

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
            session.udf.register_from_file(
                **params,
                replace=kwargs.get("replace_udf", False),
                is_permanent=True,
                stage_location=stage_location,
            )

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

    udf_code = infer_template._UDF_CODE_TEMPLATE.format(
        extract_model_code=extract_model_code,
        keep_order_code=infer_template._KEEP_ORDER_CODE_TEMPLATE if keep_order else "",
        target_method=target_method,
    )
    f.write(udf_code)
    f.flush()


def _get_model_final_packages(
    meta: _model_meta.ModelMetadata,
    session: snowpark_session.Session,
    relax_version: Optional[bool] = False,
    _use_local_snowml: Optional[bool] = False,
) -> List[str]:
    """Generate final packages list of dependency of a model to be deployed to warehouse.

    Args:
        meta: Model metadata to get dependency information.
        session: Snowpark connection session.
        relax_version: Whether or not relax the version restriction when fail to resolve dependencies.
            Defaults to False.
        _use_local_snowml: Flag to indicate if using local SnowML code as execution library

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

    deps = meta._conda_dependencies[""]
    if _use_local_snowml:
        local_snowml_version = snowml_env.VERSION
        snowml_dept = next((dep for dep in deps if dep.name == env_utils._SNOWML_PKG_NAME), None)
        if snowml_dept:
            if not snowml_dept.specifier.contains(local_snowml_version) and not relax_version:
                raise RuntimeError(
                    "Incompatible snowflake-ml-python-version is found. "
                    + f"Require {snowml_dept.specifier}, got {local_snowml_version}."
                )
            deps.remove(snowml_dept)

    try:
        final_packages = env_utils.resolve_conda_environment(
            deps, [model_env._SNOWFLAKE_CONDA_CHANNEL_URL], python_version=meta.python_version
        )
        if final_packages is None and relax_version:
            final_packages = env_utils.resolve_conda_environment(
                list(map(env_utils.relax_requirement_version, deps)),
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
            reqs=deps,
            python_version=meta.python_version,
        )
        if final_packages is None and relax_version:
            final_packages = env_utils.validate_requirements_in_snowflake_conda_channel(
                session=session,
                reqs=list(map(env_utils.relax_requirement_version, deps)),
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
