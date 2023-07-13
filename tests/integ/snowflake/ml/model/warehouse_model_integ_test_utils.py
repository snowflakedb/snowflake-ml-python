#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import os
import posixpath
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

from snowflake.ml.model import _deployer, _model as model_api, type_hints as model_types
from snowflake.snowpark import DataFrame as SnowparkDataFrame
from tests.integ.snowflake.ml.test_utils import db_manager


def base_test_case(
    db: db_manager.DBManager,
    run_id: str,
    full_qual_stage: str,
    name: str,
    model: model_types.SupportedModelType,
    sample_input: model_types.SupportedDataType,
    test_input: model_types.SupportedDataType,
    deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
    model_in_stage: Optional[bool] = False,
    permanent_deploy: Optional[bool] = False,
    test_released_library: Optional[bool] = False,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        version_args: Dict[str, Any] = {}
        tmp_stage = db._session.get_session_stage()
        if test_released_library:
            actual_name = f"{name}_v_released"
        else:
            actual_name = f"{name}_v_current"
            version_args["options"] = {"embed_local_ml_library": True}
        if model_in_stage:
            actual_name = f"{actual_name}_remote"
            location_args = {
                "session": db._session,
                "model_stage_file_path": posixpath.join(tmp_stage, f"{actual_name}_{run_id}.zip"),
            }
        else:
            actual_name = f"{actual_name}_local"
            location_args = {"model_dir_path": os.path.join(tmpdir, actual_name)}

        model_api.save_model(  # type:ignore[call-overload]
            name=actual_name,
            model=model,
            sample_input=sample_input,
            metadata={"author": "halu", "version": "1"},
            **location_args,
            **version_args,
        )

        for target_method, (additional_deploy_options, check_func) in deploy_params.items():
            deploy_version_args = {}
            if test_released_library:
                deploy_version_args = {"disable_local_conda_resolver": True}
            if permanent_deploy:
                permanent_deploy_args = {"permanent_udf_stage_location": f"@{full_qual_stage}/"}
            else:
                permanent_deploy_args = {}
            if "session" not in location_args:
                location_args.update(session=db._session)
            function_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                run_id, f"{actual_name}_{target_method}"
            )
            deploy_info = _deployer.deploy(
                name=function_name,
                **location_args,
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method=target_method,
                options={
                    "relax_version": True,
                    **permanent_deploy_args,  # type: ignore[arg-type]
                    **additional_deploy_options,
                    **deploy_version_args,
                },  # type: ignore[call-overload]
            )

            assert deploy_info is not None
            res = _deployer.predict(session=db._session, deployment=deploy_info, X=test_input)

            check_func(res)

            if permanent_deploy:
                db.drop_function(function_name=function_name, args=["OBJECT"])
