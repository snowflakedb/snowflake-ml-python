#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import os
import tempfile
import uuid

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    custom_model,
    type_hints as model_types,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class TestModelBadCaseInteg(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        # To create different UDF names among different runs
        self._db_manager = db_manager.DBManager(self._session)

        self._db_manager.cleanup_schemas()
        self._db_manager.cleanup_stages()
        self._db_manager.cleanup_user_functions()

        # To create different UDF names among different runs
        self.run_id = uuid.uuid4().hex
        self._test_schema_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "model_deployment_bad_case_test_schema"
        )
        self._db_manager.create_schema(self._test_schema_name)
        self._db_manager.use_schema(self._test_schema_name)

        self.deploy_stage_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "deployment_stage"
        )
        self.full_qual_stage = self._db_manager.create_stage(
            self.deploy_stage_name, schema_name=self._test_schema_name, sse_encrypted=False
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_stage(self.deploy_stage_name, schema_name=self._test_schema_name)
        self._db_manager.drop_schema(self._test_schema_name)
        self._session.close()

    def test_bad_model_deploy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            model_api.save_model(
                name="custom_bad_model",
                model_dir_path=os.path.join(tmpdir, "custom_bad_model"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["invalidnumpy==1.22.4"],
                options={"embed_local_ml_library": True},
            )
            function_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                self.run_id, "custom_bad_model"
            )
            with self.assertRaises(RuntimeError):
                _ = _deployer.deploy(
                    session=self._session,
                    name=function_name,
                    model_dir_path=os.path.join(tmpdir, "custom_bad_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions({"relax_version": False}),
                )

    def test_custom_demo_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = np.random.randint(100, size=(10000, 3))
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            model_api.save_model(
                name="custom_demo_model",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
                options={"embed_local_ml_library": True},
            )
            function_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                self.run_id, "custom_demo_model"
            )
            with self.assertRaises(RuntimeError):
                deploy_info = _deployer.deploy(
                    session=self._session,
                    name=function_name,
                    model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions(
                        {
                            "relax_version": True,
                            "permanent_udf_stage_location": f"{self.full_qual_stage}/",
                            # Test stage location validation
                        }
                    ),
                )

            deploy_info = _deployer.deploy(
                session=self._session,
                name=function_name,
                model_dir_path=os.path.join(tmpdir, "custom_demo_model", ""),  # Test sanitizing user path input.
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "permanent_udf_stage_location": f"@{self.full_qual_stage}/",
                    }
                ),
            )
            assert deploy_info is not None
            res = _deployer.predict(session=self._session, deployment=deploy_info, X=pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(arr[:, 0], columns=["output"]),
            )

            with self.assertRaises(RuntimeError):
                deploy_info = _deployer.deploy(
                    session=self._session,
                    name=function_name,
                    model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions(
                        {
                            "relax_version": True,
                            "permanent_udf_stage_location": f"@{self.full_qual_stage}/",
                        }
                    ),
                )

            self._db_manager.drop_function(function_name=function_name, args=["OBJECT"])

            deploy_info = _deployer.deploy(
                session=self._session,
                name=function_name,
                model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "permanent_udf_stage_location": f"@{self.full_qual_stage}/",
                        "replace_udf": True,
                    }
                ),
            )

            self._db_manager.drop_function(function_name=function_name, args=["OBJECT"])


if __name__ == "__main__":
    absltest.main()
