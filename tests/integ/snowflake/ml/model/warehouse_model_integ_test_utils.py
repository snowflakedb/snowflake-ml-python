import posixpath
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from snowflake.ml.model import (
    _api as model_api,
    deploy_platforms,
    type_hints as model_types,
)
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils


def base_test_case(
    db: db_manager.DBManager,
    run_id: str,
    full_qual_stage: str,
    name: str,
    model: model_types.SupportedModelType,
    sample_input: model_types.SupportedDataType,
    test_input: model_types.SupportedDataType,
    deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
    permanent_deploy: Optional[bool] = False,
    additional_dependencies: Optional[List[str]] = None,
) -> None:
    tmp_stage = db._session.get_session_stage()
    conda_dependencies = [
        test_env_utils.get_latest_package_version_spec_in_server(db._session, "snowflake-snowpark-python")
    ]
    if additional_dependencies:
        conda_dependencies.extend(additional_dependencies)

    if permanent_deploy:
        permanent_deploy_args = {"permanent_udf_stage_location": f"@{full_qual_stage}/"}
        perm_model_name = "perm"
    else:
        permanent_deploy_args = {}
        perm_model_name = "temp"

    actual_name = f"{name}_{perm_model_name}"

    model_api.save_model(
        name=actual_name,
        model=model,
        sample_input=sample_input,
        conda_dependencies=conda_dependencies,
        metadata={"author": "halu", "version": "1"},
        session=db._session,
        stage_path=posixpath.join(tmp_stage, f"{actual_name}_{run_id}"),
    )

    for target_method, (additional_deploy_options, check_func) in deploy_params.items():
        function_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            run_id, f"{actual_name}_{target_method}"
        )
        # This is to test the case for omitting target_method when deploying.
        if target_method == "":
            target_method_arg = None
        else:
            target_method_arg = target_method
        deploy_info = model_api.deploy(
            name=function_name,
            session=db._session,
            stage_path=posixpath.join(tmp_stage, f"{actual_name}_{run_id}"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            target_method=target_method_arg,
            options={
                **permanent_deploy_args,  # type: ignore[arg-type]
                **additional_deploy_options,
            },  # type: ignore[call-overload]
        )

        assert deploy_info is not None
        res = model_api.predict(session=db._session, deployment=deploy_info, X=test_input)

        check_func(res)

        if permanent_deploy:
            db.drop_function(function_name=function_name, args=["OBJECT"])


def check_sp_df_res(
    res_sp_df: SnowparkDataFrame,
    expected_pd_df: pd.DataFrame,
    *,
    check_dtype: bool = True,
    check_index_type: Union[bool, Literal["equiv"]] = "equiv",
    check_column_type: Union[bool, Literal["equiv"]] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
) -> None:
    res_pd_df = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(res_sp_df)

    def totuple(a: Union[npt.ArrayLike, Tuple[object], object]) -> Union[Tuple[object], object]:
        try:
            return tuple(totuple(i) for i in a)  # type: ignore[union-attr]
        except TypeError:
            return a

    for df in [res_pd_df, expected_pd_df]:
        for col in df.columns:
            if isinstance(df[col][0], list):
                df[col] = df[col].apply(tuple)
            elif isinstance(df[col][0], np.ndarray):
                df[col] = df[col].apply(totuple)

    pd.testing.assert_frame_equal(
        res_pd_df.sort_values(by=res_pd_df.columns.tolist()).reset_index(drop=True),
        expected_pd_df.sort_values(by=expected_pd_df.columns.tolist()).reset_index(drop=True),
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=check_frame_type,
        check_names=check_names,
    )
