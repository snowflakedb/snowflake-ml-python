#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import os
import posixpath
import tempfile
import unittest
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
from packaging import version

from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    deploy_platforms,
    type_hints as model_types,
)
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
    model_in_stage: Optional[bool] = False,
    permanent_deploy: Optional[bool] = False,
    test_released_version: Optional[str] = None,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        version_args: Dict[str, Any] = {}
        tmp_stage = db._session.get_session_stage()
        conda_dependencies = [
            test_env_utils.get_latest_package_versions_in_server(db._session, "snowflake-snowpark-python")
        ]
        # We only test when the test is added before the current version available in the server.
        snowml_req_str = test_env_utils.get_latest_package_versions_in_server(db._session, "snowflake-ml-python")

        if test_released_version:
            if version.parse(test_released_version) <= version.parse(snowml_req_str.split("==")[-1]):
                actual_name = f"{name}_v_released"
                conda_dependencies.append(snowml_req_str)
            else:
                raise unittest.SkipTest(
                    f"Skip test on released version {test_released_version} which has not been available yet."
                )
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
            conda_dependencies=conda_dependencies,
            metadata={"author": "halu", "version": "1"},
            **location_args,
            **version_args,
        )

        for target_method, (additional_deploy_options, check_func) in deploy_params.items():
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
                platform=deploy_platforms.TargetPlatform.WAREHOUSE,
                target_method=target_method,
                options={
                    "relax_version": test_env_utils.is_in_pip_env(),
                    **permanent_deploy_args,  # type: ignore[arg-type]
                    **additional_deploy_options,
                },  # type: ignore[call-overload]
            )

            assert deploy_info is not None
            res = _deployer.predict(session=db._session, deployment=deploy_info, X=test_input)

            check_func(res)

            if permanent_deploy:
                db.drop_function(function_name=function_name, args=["OBJECT"])
