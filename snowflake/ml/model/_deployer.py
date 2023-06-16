import json
import traceback
from enum import Enum
from typing import Optional, TypedDict, Union, overload

import numpy as np
import pandas as pd
from typing_extensions import Required

from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._deploy_client.warehouse import (
    deploy as warehouse_deploy,
    infer_template,
)
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session, functions as F


class TargetPlatform(Enum):
    WAREHOUSE = "warehouse"


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


def deploy(
    session: Session,
    *,
    name: str,
    platform: TargetPlatform,
    target_method: str,
    model_dir_path: Optional[str] = None,
    model_stage_file_path: Optional[str] = None,
    options: Optional[model_types.DeployOptions],
) -> Optional[Deployment]:
    """Create a deployment from a model and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed.
        model_dir_path: Directory of the model. Exclusive with `model_stage_dir_path`.
        model_stage_file_path: Model file in the stage to be deployed. Exclusive with `model_dir_path`.
            Must be a file with .zip extension.
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
    else:
        raise ValueError("Unsupported target Platform.")
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
        NotImplementedError: FeatureGroupSpec is not supported.

    Returns:
        The output dataframe.
    """

    # Get options
    INTERMEDIATE_OBJ_NAME = "tmp_result"
    sig = deployment["signature"]
    keep_order = deployment["options"].get("keep_order", True)
    output_with_input_features = deployment["options"].get("output_with_input_features", False)

    # Validate and prepare input
    if not isinstance(X, SnowparkDataFrame):
        df = model_signature._convert_and_validate_local_data(X, sig.inputs)
        s_df = session.create_dataframe(df)
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
                F.lit(literal_col_name),  # type:ignore[arg-type]
                F.col(col_name),
            ]
        )
    output_obj = F.call_udf(deployment["name"], F.object_construct(*input_cols))  # type:ignore[arg-type]
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
        [identifier.quote_name_without_upper_casing(output_feature.name) for output_feature in sig.outputs],
        output_cols,
    ).drop(INTERMEDIATE_OBJ_NAME)

    # Get final result
    if not isinstance(X, SnowparkDataFrame):
        dtype_map = {}
        for feature in sig.outputs:
            if isinstance(feature, model_signature.FeatureGroupSpec):
                raise NotImplementedError("FeatureGroupSpec is not supported.")
            assert isinstance(feature, model_signature.FeatureSpec), "Invalid feature kind."
            dtype_map[feature.name] = feature.as_dtype()
        df_local = df_res.to_pandas()
        # This is because Array and object will generate variant type and requires an additional loads to
        # get correct data otherwise it would be string.
        for col_name in [col_name for col_name, col_dtype in dtype_map.items() if col_dtype == np.object0]:
            df_local[col_name] = df_local[col_name].map(json.loads)
        df_local = df_local.astype(dtype=dtype_map)
        return pd.DataFrame(df_local)
    else:
        return df_res
