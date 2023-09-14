from typing import Any, Dict, Optional, TypedDict, Union, cast, overload

import pandas as pd
from typing_extensions import Required

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import (
    _model,
    deploy_platforms,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._deploy_client.snowservice import deploy as snowservice_deploy
from snowflake.ml.model._deploy_client.utils import constants as snowservice_constants
from snowflake.ml.model._deploy_client.warehouse import (
    deploy as warehouse_deploy,
    infer_template,
)
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session, functions as F


class Deployment(TypedDict):
    """Deployment information.

    Attributes:
        name: Name of the deployment.
        platform: Target platform to deploy the model.
        target_method: Target method name.
        signature: The signature of the model method.
        options: Additional options when deploying the model.
    """

    name: Required[str]
    platform: Required[deploy_platforms.TargetPlatform]
    target_method: Required[str]
    signature: model_signature.ModelSignature
    options: Required[model_types.DeployOptions]


@overload
def deploy(
    session: Session,
    *,
    name: str,
    platform: deploy_platforms.TargetPlatform,
    target_method: Optional[str],
    model_stage_file_path: str,
    options: Optional[model_types.DeployOptions],
) -> Optional[Deployment]:
    """Create a deployment from a model in a zip file in a stage and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed. Can be omitted if there is only 1 target method in
            the model.
        model_stage_file_path: Model file in the stage to be deployed. Must be a file with .zip extension.
        options: Additional options when deploying the model.
            Each target platform will have their own specifications of options.
    """
    ...


@overload
def deploy(
    session: Session,
    *,
    model_id: str,
    name: str,
    platform: deploy_platforms.TargetPlatform,
    target_method: Optional[str],
    model_stage_file_path: str,
    deployment_stage_path: str,
    options: Optional[model_types.DeployOptions],
) -> Optional[Deployment]:
    """Create a deployment from a model in a local directory and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        model_id: Internal model ID string.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed. Can be omitted if there is only 1 target method in
            the model.
        model_stage_file_path: Model file in the stage to be deployed. Must be a file with .zip extension.
        deployment_stage_path: Path to stage containing snowpark container service deployment artifacts.
        options: Additional options when deploying the model.
            Each target platform will have their own specifications of options.
    """
    ...


def deploy(
    session: Session,
    *,
    name: str,
    platform: deploy_platforms.TargetPlatform,
    model_stage_file_path: str,
    target_method: Optional[str] = None,
    deployment_stage_path: Optional[str] = None,
    model_id: Optional[str] = None,
    options: Optional[model_types.DeployOptions],
) -> Optional[Deployment]:
    """Create a deployment from a model and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        model_id: Internal model ID string.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed. Can be omitted if there is only 1 target method in
            the model.
        model_stage_file_path: Model file in the stage to be deployed. Exclusive with `model_dir_path`.
            Must be a file with .zip extension.
        deployment_stage_path: Path to stage containing deployment artifacts.
        options: Additional options when deploying the model.
            Each target platform will have their own specifications of options.

    Raises:
        SnowflakeMLException: Raised when target platform is unsupported.
        SnowflakeMLException: Raised when target method does not exist in model.

    Returns:
        The deployment information.
    """

    info = None

    if not options:
        options = {}

    meta = _model.load_model(session=session, model_stage_file_path=model_stage_file_path, meta_only=True)

    if target_method is None:
        if len(meta.signatures.keys()) == 1:
            target_method = list(meta.signatures.keys())[0]
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Only when the model has 1 target methods can target_method be omitted when deploying."
                ),
            )

    if platform == deploy_platforms.TargetPlatform.WAREHOUSE:
        warehouse_deploy._deploy_to_warehouse(
            session=session,
            model_stage_file_path=model_stage_file_path,
            model_meta=meta,
            udf_name=name,
            target_method=target_method,
            **options,
        )

    elif platform == deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES:
        options = cast(model_types.SnowparkContainerServiceDeployOptions, options)
        assert model_id, "Require 'model_id' for Snowpark container service deployment"
        assert model_stage_file_path, "Require 'model_stage_file_path' for Snowpark container service deployment"
        assert deployment_stage_path, "Require 'deployment_stage_path' for Snowpark container service deployment"
        if snowservice_constants.COMPUTE_POOL not in options:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Missing 'compute_pool' in options field for Snowpark container service deployment"
                ),
            )

        snowservice_deploy._deploy(
            session=session,
            model_id=model_id,
            model_meta=meta,
            service_func_name=name,
            model_zip_stage_path=model_stage_file_path,
            deployment_stage_path=deployment_stage_path,
            target_method=target_method,
            **options,
        )

    else:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_TYPE,
            original_exception=ValueError(f"Unsupported target Platform: {platform}"),
        )
    signature = meta.signatures.get(target_method, None)
    if not signature:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(f"Target method {target_method} does not exist in model."),
        )
    info = Deployment(name=name, platform=platform, target_method=target_method, signature=signature, options=options)
    return info


@overload
def predict(
    session: Session,
    *,
    deployment: Deployment,
    X: model_types.SupportedLocalDataType,
    statement_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Execute batch inference of a model remotely on local data. Can be any supported data type. Return a local
        Pandas Dataframe.

    Args:
        session: Snowpark Connection Session.
        deployment: The deployment info to use for predict.
        X: The input data.
        statement_params: Statement Parameters for telemetry.
    """
    ...


@overload
def predict(
    session: Session,
    *,
    deployment: Deployment,
    X: SnowparkDataFrame,
    statement_params: Optional[Dict[str, Any]] = None,
) -> SnowparkDataFrame:
    """Execute batch inference of a model remotely on a Snowpark DataFrame. Return a Snowpark DataFrame.

    Args:
        session: Snowpark Connection Session.
        deployment: The deployment info to use for predict.
        X: The input Snowpark dataframe.
        statement_params: Statement Parameters for telemetry.
    """
    ...


def predict(
    session: Session,
    *,
    deployment: Deployment,
    X: Union[model_types.SupportedDataType, SnowparkDataFrame],
    statement_params: Optional[Dict[str, Any]] = None,
) -> Union[pd.DataFrame, SnowparkDataFrame]:
    """Execute batch inference of a model remotely.

    Args:
        session: Snowpark Connection Session.
        deployment: The deployment info to use for predict.
        X: The input dataframe.
        statement_params: Statement Parameters for telemetry.

    Returns:
        The output dataframe.
    """

    # Get options
    INTERMEDIATE_OBJ_NAME = "tmp_result"
    sig = deployment["signature"]

    # Validate and prepare input
    if not isinstance(X, SnowparkDataFrame):
        keep_order = True
        output_with_input_features = False
        df = model_signature._convert_and_validate_local_data(X, sig.inputs)
        s_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(session, df, keep_order=keep_order)
    else:
        keep_order = False
        output_with_input_features = True
        model_signature._validate_snowpark_data(X, sig.inputs)
        s_df = X

    if statement_params:
        if s_df._statement_params is not None:
            s_df._statement_params.update(statement_params)
        else:
            s_df._statement_params = statement_params  # type: ignore[assignment]

    # Infer and get intermediate result
    input_cols = []
    for col_name in s_df.columns:
        literal_col_name = identifier.get_unescaped_names(col_name)
        input_cols.extend(
            [
                F.lit(literal_col_name),
                F.col(col_name),
            ]
        )

    # TODO[shchen]: SNOW-870032, For SnowService, external function name cannot be double quoted, else it results in
    # external function no found.
    udf_name = deployment["name"]
    output_obj = F.call_udf(udf_name, F.object_construct(*input_cols))

    if output_with_input_features:
        df_res = s_df.with_column(INTERMEDIATE_OBJ_NAME, output_obj)
    else:
        df_res = s_df.select(output_obj.alias(INTERMEDIATE_OBJ_NAME))

    if keep_order:
        df_res = df_res.order_by(
            F.col(INTERMEDIATE_OBJ_NAME)[infer_template._KEEP_ORDER_COL_NAME],
            ascending=True,
        )

    # Prepare the output
    output_cols = []
    for output_feature in sig.outputs:
        output_cols.append(F.col(INTERMEDIATE_OBJ_NAME)[output_feature.name].astype(output_feature.as_snowpark_type()))

    df_res = df_res.with_columns(
        [identifier.get_inferred_name(output_feature.name) for output_feature in sig.outputs],
        output_cols,
    ).drop(INTERMEDIATE_OBJ_NAME)

    # Get final result
    if not isinstance(X, SnowparkDataFrame):
        return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(df_res, features=sig.outputs)
    else:
        return df_res
