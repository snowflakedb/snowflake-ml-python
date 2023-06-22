#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import os
import sys
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd
from absl import flags
from absl.testing import absltest

from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    custom_model,
    type_hints as model_types,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session

flags.FLAGS(sys.argv)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


def _create_stage(session: Session, stage_qual_name: str) -> None:
    sql = f"CREATE STAGE {stage_qual_name}"
    session.sql(sql).collect()


def _drop_function(session: Session, func_name: str) -> None:
    sql = f"DROP FUNCTION {func_name}(OBJECT)"
    session.sql(sql).collect()


def _drop_stage(session: Session, stage_qual_name: str) -> None:
    sql = f"DROP STAGE {stage_qual_name}"
    session.sql(sql).collect()


class TestModelBadCaseInteg(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        # To create different UDF names among different runs
        self.run_id = str(uuid4()).replace("-", "_")

        db = self._session.get_current_database()
        schema = self._session.get_current_schema()
        self.stage_qual_name = f"{db}.{schema}.SNOWML_MODEL_TEST_STAGE_{self.run_id.upper()}"
        _create_stage(session=self._session, stage_qual_name=self.stage_qual_name)

    @classmethod
    def tearDownClass(self) -> None:
        _drop_stage(session=self._session, stage_qual_name=self.stage_qual_name)
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

            with self.assertRaises(RuntimeError):
                _ = _deployer.deploy(
                    session=self._session,
                    name=f"custom_bad_model_{self.run_id}",
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

            with self.assertRaises(RuntimeError):
                deploy_info = _deployer.deploy(
                    session=self._session,
                    name=f"custom_demo_model_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions(
                        {
                            "relax_version": True,
                            "permanent_udf_stage_location": f"{self.stage_qual_name}/",
                            # Test stage location validation
                        }
                    ),
                )

            deploy_info = _deployer.deploy(
                session=self._session,
                name=f"custom_demo_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model", ""),  # Test sanitizing user path input.
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "permanent_udf_stage_location": f"@{self.stage_qual_name}/",
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
                    name=f"custom_demo_model_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions(
                        {
                            "relax_version": True,
                            "permanent_udf_stage_location": f"@{self.stage_qual_name}/",
                        }
                    ),
                )

            _drop_function(self._session, f"custom_demo_model_{self.run_id}")

            deploy_info = _deployer.deploy(
                session=self._session,
                name=f"custom_demo_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "permanent_udf_stage_location": f"@{self.stage_qual_name}/",
                        "replace_udf": True,
                    }
                ),
            )
