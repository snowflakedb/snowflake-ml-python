from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Union, cast, overload

import pandas as pd

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import (
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
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session, functions as F


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedNoSignatureRequirementsModelType,
    session: Session,
    stage_path: str,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> model_composer.ModelComposer:
    """Save a model that does not require a signature as model to a stage path.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        stage_path: Path to the stage where model will be saved.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedRequireSignatureModelType,
    session: Session,
    stage_path: str,
    signatures: Dict[str, model_signature.ModelSignature],
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> model_composer.ModelComposer:
    """Save a model that requires a external signature with user provided signatures as model to a stage path.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        stage_path: Path to the stage where model will be saved.
        signatures: Model data signatures for inputs and output for every target methods.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedRequireSignatureModelType,
    session: Session,
    stage_path: str,
    sample_input: model_types.SupportedDataType,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> model_composer.ModelComposer:
    """Save a model that requires a external signature as model to a stage path with signature inferred from a
      sample_input_data.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        stage_path: Path to the stage where model will be saved.
        sample_input: Sample input data to infer the model signatures from.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        options: Model specific kwargs.
    """
    ...


def save_model(
    *,
    name: str,
    model: model_types.SupportedModelType,
    session: Session,
    stage_path: str,
    signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
    sample_input: Optional[model_types.SupportedDataType] = None,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> model_composer.ModelComposer:
    """Save the model.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        stage_path: Path to the stage where model will be saved.
        signatures: Model data signatures for inputs and output for every target methods. If it is None, sample_input
            would be used to infer the signatures if it is a local (non-SnowML modeling model).
            If not None, sample_input should not be specified. Defaults to None.
        sample_input: Sample input data to infer the model signatures from. If it is None, signatures must be specified
            if it is a local (non-SnowML modeling model). If not None, signatures should not be specified.
            Defaults to None.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        options: Model specific kwargs.

    Returns:
        Model
    """
    if options is None:
        options = {}
    options["_legacy_save"] = True

    m = model_composer.ModelComposer(session=session, stage_path=stage_path)
    m.save(
        name=name,
        model=model,
        signatures=signatures,
        sample_input=sample_input,
        metadata=metadata,
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        python_version=python_version,
        ext_modules=ext_modules,
        code_paths=code_paths,
        options=options,
    )
    return m


@overload
def load_model(*, session: Session, stage_path: str) -> model_composer.ModelComposer:
    """Load the model into memory from a zip file in the stage.

    Args:
        session: Snowflake connection session.
        stage_path: Path to the stage where model will be loaded from.
    """
    ...


@overload
def load_model(*, session: Session, stage_path: str, meta_only: Literal[False]) -> model_composer.ModelComposer:
    """Load the model into memory from a zip file in the stage.

    Args:
        session: Snowflake connection session.
        stage_path: Path to the stage where model will be loaded from.
        meta_only: Flag to indicate that if only load metadata.
    """
    ...


@overload
def load_model(*, session: Session, stage_path: str, meta_only: Literal[True]) -> model_composer.ModelComposer:
    """Load the model into memory from a zip file in the stage with metadata only.

    Args:
        session: Snowflake connection session.
        stage_path: Path to the stage where model will be loaded from.
        meta_only: Flag to indicate that if only load metadata.
    """
    ...


def load_model(
    *,
    session: Session,
    stage_path: str,
    meta_only: bool = False,
) -> model_composer.ModelComposer:
    """Load the model into memory from directory or a zip file in the stage.

    Args:
        session: Snowflake connection session. Must be specified when specifying model_stage_file_path.
            Exclusive with model_dir_path.
        stage_path: Path to the stage where model will be loaded from.
        meta_only: Flag to indicate that if only load metadata.

    Returns:
        Loaded model.
    """
    m = model_composer.ModelComposer(session=session, stage_path=stage_path)
    m.load(meta_only=meta_only)
    return m


@overload
def deploy(
    session: Session,
    *,
    name: str,
    platform: deploy_platforms.TargetPlatform,
    target_method: Optional[str],
    stage_path: str,
    options: Optional[model_types.DeployOptions],
) -> Optional[model_types.Deployment]:
    """Create a deployment from a model in a zip file in a stage and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed. Can be omitted if there is only 1 target method in
            the model.
        stage_path: Path to the stage where model will be deployed.
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
    stage_path: str,
    deployment_stage_path: str,
    options: Optional[model_types.DeployOptions],
) -> Optional[model_types.Deployment]:
    """Create a deployment from a model in a local directory and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        model_id: Internal model ID string.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed. Can be omitted if there is only 1 target method in
            the model.
        stage_path: Path to the stage where model will be deployed.
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
    stage_path: str,
    target_method: Optional[str] = None,
    deployment_stage_path: Optional[str] = None,
    model_id: Optional[str] = None,
    options: Optional[model_types.DeployOptions],
) -> Optional[model_types.Deployment]:
    """Create a deployment from a model and deploy it to remote platform.

    Args:
        session: Snowpark Connection Session.
        model_id: Internal model ID string.
        name: Name of the deployment for the model.
        platform: Target platform to deploy the model.
        target_method: The name of the target method to be deployed. Can be omitted if there is only 1 target method in
            the model.
        stage_path: Path to the stage where model will be deployed.
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

    m = load_model(session=session, stage_path=stage_path, meta_only=True)
    assert m.packager.meta

    if target_method is None:
        if len(m.packager.meta.signatures.keys()) == 1:
            target_method = list(m.packager.meta.signatures.keys())[0]
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Only when the model has 1 target methods can target_method be omitted when deploying."
                ),
            )

    details: model_types.DeployDetails = {}
    if platform == deploy_platforms.TargetPlatform.WAREHOUSE:
        warehouse_deploy._deploy_to_warehouse(
            session=session,
            model_stage_file_path=m.model_stage_path,
            model_meta=m.packager.meta,
            udf_name=name,
            target_method=target_method,
            **options,
        )

    elif platform == deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES:
        options = cast(model_types.SnowparkContainerServiceDeployOptions, options)
        assert model_id, "Require 'model_id' for Snowpark container service deployment"
        assert m.model_stage_path, "Require 'model_stage_file_path' for Snowpark container service deployment"
        assert deployment_stage_path, "Require 'deployment_stage_path' for Snowpark container service deployment"
        if snowservice_constants.COMPUTE_POOL not in options:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Missing 'compute_pool' in options field for Snowpark container service deployment"
                ),
            )

        details = snowservice_deploy._deploy(
            session=session,
            model_id=model_id,
            model_meta=m.packager.meta,
            service_func_name=name,
            model_zip_stage_path=m.model_stage_path,
            deployment_stage_path=deployment_stage_path,
            target_method=target_method,
            **options,
        )

    else:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_TYPE,
            original_exception=ValueError(f"Unsupported target Platform: {platform}"),
        )
    signature = m.packager.meta.signatures.get(target_method, None)
    if not signature:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(f"Target method {target_method} does not exist in model."),
        )
    info = model_types.Deployment(
        name=name, platform=platform, target_method=target_method, signature=signature, options=options, details=details
    )
    return info


@overload
def predict(
    session: Session,
    *,
    deployment: model_types.Deployment,
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
    deployment: model_types.Deployment,
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
    deployment: model_types.Deployment,
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
    identifier_rule = model_signature.SnowparkIdentifierRule.INFERRED

    # Validate and prepare input
    if not isinstance(X, SnowparkDataFrame):
        keep_order = True
        output_with_input_features = False
        df = model_signature._convert_and_validate_local_data(X, sig.inputs)
        s_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(session, df, keep_order=keep_order)
    else:
        keep_order = False
        output_with_input_features = True
        identifier_rule = model_signature._validate_snowpark_data(X, sig.inputs)
        s_df = X

    if statement_params:
        if s_df._statement_params is not None:
            s_df._statement_params.update(statement_params)
        else:
            s_df._statement_params = statement_params  # type: ignore[assignment]

    original_cols = s_df.columns

    # Infer and get intermediate result
    input_cols = []
    for input_feature in sig.inputs:
        literal_col_name = input_feature.name
        col_name = identifier_rule.get_identifier_from_feature(input_feature.name)

        input_cols.extend(
            [
                F.lit(literal_col_name),
                F.col(col_name),
            ]
        )

    udf_name = deployment["name"]
    output_obj = F.call_udf(udf_name, F.object_construct_keep_null(*input_cols))
    df_res = s_df.with_column(INTERMEDIATE_OBJ_NAME, output_obj)

    if keep_order:
        df_res = df_res.order_by(
            F.col(infer_template._KEEP_ORDER_COL_NAME),
            ascending=True,
        )

    if not output_with_input_features:
        df_res = df_res.drop(*original_cols)

    # Prepare the output
    output_cols = []
    output_col_names = []
    for output_feature in sig.outputs:
        output_cols.append(F.col(INTERMEDIATE_OBJ_NAME)[output_feature.name].astype(output_feature.as_snowpark_type()))
        output_col_names.append(identifier_rule.get_identifier_from_feature(output_feature.name))

    df_res = df_res.with_columns(
        output_col_names,
        output_cols,
    ).drop(INTERMEDIATE_OBJ_NAME)

    # Get final result
    if not isinstance(X, SnowparkDataFrame):
        return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(df_res, features=sig.outputs)
    else:
        return df_res
