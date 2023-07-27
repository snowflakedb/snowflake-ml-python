import traceback
from enum import Enum
from typing import Optional, TypedDict, Union, cast, overload

import pandas as pd
from typing_extensions import Required

from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._deploy_client.snowservice import deploy as snowservice_deploy
from snowflake.ml.model._deploy_client.utils import constants as snowservice_constants
from snowflake.ml.model._deploy_client.warehouse import (
    deploy as warehouse_deploy,
    infer_template,
)
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session, functions as F


class TargetPlatform(Enum):
    WAREHOUSE = "warehouse"
    SNOWPARK_CONTAINER_SERVICE = "snowpark_container_service"

    def __repr__(self) -> str:
        """Construct a string format that works with the "ModelReference" in model_registry.py. Fundamentally,
        ModelReference uses the TargetPlatform enum type when constructing the "deploy" function through exec().
        Since "exec" in Python takes input as a string, we need to dynamically construct a full path so that the
        enum can be loaded successfully.

        Returns:
            A enum string representation.
        """

        return f"{__name__.split('.')[-1]}.{self.__class__.__name__}.{self.name}"


class Deployment(TypedDict):
    """Deployment information.

    Attributes:
        name: Name of the deployment.
        platform: Target platform to deploy the model.
        signature: The signature of the model method.
        options: Additional options when deploying the model.
    """

    name: Required[str]
    platform: Required[TargetPlatform]
    signature: model_signature.ModelSignature
    options: Required[model_types.DeployOptions]


@overload
def deploy(
    session: Session,
    *,
    name: str,
    platform: TargetPlatform,
    target_method: str,
    model_dir_path: str,
    options: Optional[model_types.DeployOptions],
) -> Optional[Deployment]:
    """Create a deployment from a model in a local directory and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed.
        model_dir_path: Directory of the model.
        options: Additional options when deploying the model.
            Each target platform will have their own specifications of options.
    """
    ...


@overload
def deploy(
    session: Session,
    *,
    name: str,
    platform: TargetPlatform,
    target_method: str,
    model_stage_file_path: str,
    options: Optional[model_types.DeployOptions],
) -> Optional[Deployment]:
    """Create a deployment from a model in a zip file in a stage and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed.
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
    platform: TargetPlatform,
    target_method: str,
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
        target_method: The name of the target method to be deployed.
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
    platform: TargetPlatform,
    target_method: str,
    model_dir_path: Optional[str] = None,
    model_stage_file_path: Optional[str] = None,
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
        target_method: The name of the target method to be deployed.
        model_dir_path: Directory of the model. Exclusive with `model_stage_dir_path`.
        model_stage_file_path: Model file in the stage to be deployed. Exclusive with `model_dir_path`.
            Must be a file with .zip extension.
        deployment_stage_path: Path to stage containing deployment artifacts.
        options: Additional options when deploying the model.
            Each target platform will have their own specifications of options.

    Raises:
        ValueError: Raised when target platform is unsupported.
        RuntimeError: Raised when running into errors when deploying to the warehouse.
        ValueError: Raised when target method does not exist in model.

    Returns:
        The deployment information.
    """
    if not ((model_stage_file_path is None) ^ (model_dir_path is None)):
        raise ValueError(
            "model_dir_path and model_stage_file_path both cannot be "
            + f"{'None' if model_stage_file_path is None else 'specified'} at the same time."
        )

    info = None

    if not options:
        options = {}

    if platform == TargetPlatform.WAREHOUSE:
        try:
            meta = warehouse_deploy._deploy_to_warehouse(
                session=session,
                model_dir_path=model_dir_path,
                model_stage_file_path=model_stage_file_path,
                udf_name=name,
                target_method=target_method,
                **options,
            )
        except Exception:
            raise RuntimeError("Error happened when deploying to the warehouse: " + traceback.format_exc())

    elif platform == TargetPlatform.SNOWPARK_CONTAINER_SERVICE:
        options = cast(model_types.SnowparkContainerServiceDeployOptions, options)
        assert model_id, "Require 'model_id' for Snowpark container service deployment"
        assert model_stage_file_path, "Require 'model_stage_file_path' for Snowpark container service deployment"
        assert deployment_stage_path, "Require 'deployment_stage_path' for Snowpark container service deployment"
        if snowservice_constants.COMPUTE_POOL not in options:
            raise ValueError("Missing 'compute_pool' in options field for Snowpark container service deployment")
        try:
            meta = snowservice_deploy._deploy(
                session=session,
                model_id=model_id,
                service_func_name=name,
                model_zip_stage_path=model_stage_file_path,
                deployment_stage_path=deployment_stage_path,
                **options,
            )
        except Exception:
            raise RuntimeError(f"Failed to deploy to Snowpark Container Service: {traceback.format_exc()}")

    else:
        raise ValueError(f"Unsupported target Platform: {platform}")
    signature = meta.signatures.get(target_method, None)
    if not signature:
        raise ValueError(f"Target method {target_method} does not exist in model.")
    info = Deployment(name=name, platform=platform, signature=signature, options=options)
    return info


@overload
def predict(session: Session, *, deployment: Deployment, X: model_types.SupportedLocalDataType) -> pd.DataFrame:
    """Execute batch inference of a model remotely on local data. Can be any supported data type. Return a local
        Pandas Dataframe.

    Args:
        session: Snowpark Connection Session.
        deployment: The deployment info to use for predict.
        X: The input data.
    """
    ...


@overload
def predict(session: Session, *, deployment: Deployment, X: SnowparkDataFrame) -> SnowparkDataFrame:
    """Execute batch inference of a model remotely on a Snowpark DataFrame. Return a Snowpark DataFrame.

    Args:
        session: Snowpark Connection Session.
        deployment: The deployment info to use for predict.
        X: The input Snowpark dataframe.

    """


def predict(
    session: Session, *, deployment: Deployment, X: Union[model_types.SupportedDataType, SnowparkDataFrame]
) -> Union[pd.DataFrame, SnowparkDataFrame]:
    """Execute batch inference of a model remotely.

    Args:
        session: Snowpark Connection Session.
        deployment: The deployment info to use for predict.
        X: The input dataframe.

    Raises:
        ValueError: Raised when the input is too large to use keep_order option.

    Returns:
        The output dataframe.
    """

    # Get options
    INTERMEDIATE_OBJ_NAME = "tmp_result"
    sig = deployment["signature"]
    keep_order = deployment["options"].get("keep_order", True)
    output_with_input_features = deployment["options"].get("output_with_input_features", False)
    platform = deployment["platform"]

    # Validate and prepare input
    if not isinstance(X, SnowparkDataFrame):
        df = model_signature._convert_and_validate_local_data(X, sig.inputs)
        s_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(session, df, keep_order=keep_order)
    else:
        model_signature._validate_snowpark_data(X, sig.inputs)
        s_df = X

        if keep_order:
            # ID is UINT64 type, this we should limit.
            if s_df.count() > 2**64:
                raise ValueError("Unable to keep order of a DataFrame with more than 2 ** 64 rows.")
            s_df = s_df.with_column(
                infer_template._KEEP_ORDER_COL_NAME,
                F.monotonically_increasing_id(),
            )

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
    udf_name = (
        deployment["name"]
        if platform == TargetPlatform.SNOWPARK_CONTAINER_SERVICE
        else identifier.get_inferred_name(deployment["name"])
    )
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
        if output_with_input_features:
            df_res = df_res.drop(infer_template._KEEP_ORDER_COL_NAME)

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
