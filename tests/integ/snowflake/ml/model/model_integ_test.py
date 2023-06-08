#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import asyncio
import json
import os
import sys
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd
import xgboost
from absl import flags
from absl.testing import absltest
from sklearn import datasets, ensemble, linear_model, model_selection, multioutput

from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    custom_model,
    type_hints as model_types,
)
from snowflake.ml.modeling.lightgbm import LGBMRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session

flags.FLAGS(sys.argv)


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class DemoModelSPQuote(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({'"output"': input['"c1"']})


class DemoModelArray(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input.values.tolist()})


class AsyncComposeModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    async def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        res1 = await self.context.model_ref("m1").predict.async_run(input)
        res_sum = res1["output"] + self.context.model_ref("m2").predict(input)["output"]
        return pd.DataFrame({"output": res_sum / 2})


class DemoModelWithArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(context.path("bias")) as f:
            v = int(f.read())
        self.bias = v

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": (input["c1"] + self.bias) > 12})


def _create_stage(session: Session, stage_qual_name: str) -> None:
    sql = f"CREATE STAGE {stage_qual_name}"
    session.sql(sql).collect()


def _drop_function(session: Session, func_name: str) -> None:
    sql = f"DROP FUNCTION {func_name}(OBJECT)"
    session.sql(sql).collect()


def _drop_stage(session: Session, stage_qual_name: str) -> None:
    sql = f"DROP STAGE {stage_qual_name}"
    session.sql(sql).collect()


class TestModelInteg(absltest.TestCase):
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

    def test_async_model_composition(self) -> None:
        async def _test(self: "TestModelInteg") -> None:
            arr = np.random.randint(100, size=(10000, 3))
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            clf = DemoModel(custom_model.ModelContext())
            model_context = custom_model.ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            with tempfile.TemporaryDirectory() as tmpdir:
                model_api.save_model(
                    name="async_model_composition",
                    model_dir_path=os.path.join(tmpdir, "async_model_composition"),
                    model=acm,
                    sample_input=pd_df,
                    metadata={"author": "halu", "version": "1"},
                )
                deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
                deploy_info = deployer.create_deployment(
                    name=f"async_model_composition_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "async_model_composition"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
                )

                assert deploy_info is not None
                res = deployer.predict(deploy_info["name"], pd_df)

                pd.testing.assert_frame_equal(
                    res,
                    pd.DataFrame(arr[:, 0], columns=["output"], dtype=float),
                )

                self.assertTrue(deploy_info in deployer.list_deployments())
                self.assertEqual(deploy_info, deployer.get_deployment(f"async_model_composition_{self.run_id}"))

        asyncio.get_event_loop().run_until_complete(_test(self))

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
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            with self.assertRaises(RuntimeError):
                _ = deployer.create_deployment(
                    name=f"custom_bad_model_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "custom_bad_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions({"relax_version": False, "_use_local_snowml": True}),
                )

            with self.assertRaises(ValueError):
                _ = deployer.predict(f"custom_bad_model_{self.run_id}", pd_df)

    def test_custom_demo_model_sp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = [[1, 2, 3], [4, 2, 5]]
            sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
            model_api.save_model(
                name="custom_demo_model_sp0",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp0"),
                model=lm,
                sample_input=sp_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_sp0_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp0"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], sp_df)

            pd.testing.assert_frame_equal(
                res.to_pandas(),
                pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
            )

    def test_custom_demo_model_sp_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModelSPQuote(custom_model.ModelContext())
            arr = [[1, 2, 3], [4, 2, 5]]
            sp_df = self._session.create_dataframe(arr, schema=['"""c1"""', '"""c2"""', '"""c3"""'])
            pd_df = pd.DataFrame(arr, columns=['"c1"', '"c2"', '"c3"'])
            model_api.save_model(
                name="custom_demo_model_sp_good",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp_good"),
                model=lm,
                sample_input=sp_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_sp_good_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp_good"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame([1, 4], columns=['"output"'], dtype=np.int8),
            )

    def test_custom_demo_model_sp_mix_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = [[1, 2, 3], [4, 2, 5]]
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
            model_api.save_model(
                name="custom_demo_model_sp1",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp1"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_sp1_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp1"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], sp_df)

            pd.testing.assert_frame_equal(
                res.to_pandas(),
                pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
            )

    def test_custom_demo_model_sp_mix_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = [[1, 2, 3], [4, 2, 5]]
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
            model_api.save_model(
                name="custom_demo_model_sp2",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp2"),
                model=lm,
                sample_input=sp_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_sp2_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_sp2"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
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
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model", ""),  # Test sanitizing user path input.
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "_use_local_snowml": True,
                        "permanent_udf_stage_location": f"@{self.stage_qual_name}/",
                    }
                ),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(arr[:, 0], columns=["output"]),
            )

            self.assertTrue(deploy_info in deployer.list_deployments())
            self.assertEqual(deploy_info, deployer.get_deployment(f"custom_demo_model_{self.run_id}"))

            with self.assertRaises(RuntimeError):
                deploy_info = deployer.create_deployment(
                    name=f"custom_demo_model_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method="predict",
                    options=model_types.WarehouseDeployOptions(
                        {
                            "relax_version": True,
                            "_use_local_snowml": True,
                            "permanent_udf_stage_location": f"@{self.stage_qual_name}/",
                        }
                    ),
                )

            _drop_function(self._session, f"custom_demo_model_{self.run_id}")

    def test_custom_demo_model_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModelArray(custom_model.ModelContext())
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            model_api.save_model(
                name="custom_demo_model_array",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_array"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_array_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_array"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]}),
            )

    def test_custom_demo_model_str(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            pd_df = pd.DataFrame(
                [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"]
            )
            model_api.save_model(
                name="custom_demo_model_str",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_str"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_str_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_str"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(data={"output": ["Yogiri", "Artia"]}),
            )

    def test_custom_demo_model_array_sp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModelArray(custom_model.ModelContext())
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            sp_df = self._session.create_dataframe(pd_df)
            model_api.save_model(
                name="custom_demo_model_array_sp",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_array_sp"),
                model=lm,
                sample_input=sp_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_array_sp_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_array_sp"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], sp_df)

            pd.testing.assert_frame_equal(
                res.to_pandas().applymap(json.loads),
                pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]}),
            )

    def test_custom_demo_model_str_sp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            pd_df = pd.DataFrame(
                [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"]
            )
            sp_df = self._session.create_dataframe(pd_df)
            model_api.save_model(
                name="custom_demo_model_str_sp",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_str_sp"),
                model=lm,
                sample_input=sp_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_str_sp_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_str_sp"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], sp_df)

            pd.testing.assert_frame_equal(
                res.to_pandas(),
                pd.DataFrame(data={"output": ["Yogiri", "Artia"]}),
            )

    def test_custom_demo_model_array_str(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModelArray(custom_model.ModelContext())
            pd_df = pd.DataFrame(
                [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"]
            )
            model_api.save_model(
                name="custom_demo_model_array_str",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_array_str"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_array_str_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_array_str"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(data={"output": [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]]}),
            )

    def test_custom_demo_model_with_input_no_keep_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = np.random.randint(100, size=(10000, 3))
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            model_api.save_model(
                name="custom_demo_model_with_input_no_keep_order",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_with_input_no_keep_order"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_with_input_no_keep_order_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_with_input_no_keep_order"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "_use_local_snowml": True,
                        "output_with_input_features": True,
                        "keep_order": False,
                    }
                ),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)
            pd.testing.assert_series_equal(res["output"], res["c1"], check_dtype=False, check_names=False)

    def test_custom_demo_model_with_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = np.random.randint(100, size=(10000, 3))
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            model_api.save_model(
                name="custom_demo_model_with_input",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_with_input"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_demo_model_with_input_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model_with_input"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions(
                    {
                        "relax_version": True,
                        "_use_local_snowml": True,
                        "output_with_input_features": True,
                    }
                ),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df)
            pd.testing.assert_series_equal(res["output"], res["c1"], check_dtype=False, check_names=False)
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(
                    np.concatenate([arr, np.expand_dims(arr[:, 0], axis=1)], axis=1),
                    columns=["c1", "c2", "c3", "output"],
                ),
                check_dtype=False,
            )

    def test_custom_model_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            model_api.save_model(
                name="custom_model_with_artifacts",
                model_dir_path=os.path.join(tmpdir, "custom_model_with_artifacts"),
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_model_with_artifacts_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_model_with_artifacts"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df[["c3", "c1", "c2"]])

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame([False, True], columns=["output"]),
        )

    def test_custom_demo_model_in_stage(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = np.random.randint(100, size=(10000, 3))
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        tmp_stage = self._session.get_session_stage()
        model_path_in_stage = f"{tmp_stage}/custom_demo_model_in_stage_{self.run_id}.zip"
        model_api.save_model(
            name="custom_demo_model_in_stage",
            session=self._session,
            model_stage_file_path=model_path_in_stage,
            model=lm,
            sample_input=pd_df,
            metadata={"author": "halu", "version": "1"},
        )

        loaded_model, _ = model_api.load_model(session=self._session, model_stage_file_path=model_path_in_stage)
        assert isinstance(loaded_model, DemoModel)

        local_loaded_res = loaded_model.predict(pd_df)
        pd.testing.assert_frame_equal(
            local_loaded_res,
            pd.DataFrame(arr[:, 0], columns=["output"]),
        )

        deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
        deploy_info = deployer.create_deployment(
            name=f"custom_demo_model_{self.run_id}",
            model_stage_file_path=model_path_in_stage,
            platform=_deployer.TargetPlatform.WAREHOUSE,
            target_method="predict",
            options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
        )
        assert deploy_info is not None
        res = deployer.predict(deploy_info["name"], pd_df)

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(arr[:, 0], columns=["output"]),
        )

        self.assertTrue(deploy_info in deployer.list_deployments())
        self.assertEqual(deploy_info, deployer.get_deployment(f"custom_demo_model_{self.run_id}"))

    def test_custom_model_with_artifacts_in_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            tmp_stage = self._session.get_session_stage()
            model_path_in_stage = f"{tmp_stage}/custom_model_with_artifacts_in_stage_{self.run_id}.zip"
            model_api.save_model(
                name="custom_model_with_artifacts",
                session=self._session,
                model_stage_file_path=model_path_in_stage,
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
            )

            loaded_model, _ = model_api.load_model(session=self._session, model_stage_file_path=model_path_in_stage)
            assert isinstance(loaded_model, DemoModelWithArtifacts)

            local_loaded_res = loaded_model.predict(pd_df)
            pd.testing.assert_frame_equal(
                local_loaded_res,
                pd.DataFrame([False, True], columns=["output"]),
            )

            deployer = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            deploy_info = deployer.create_deployment(
                name=f"custom_model_with_artifacts{self.run_id}",
                model_stage_file_path=model_path_in_stage,
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert deploy_info is not None
            res = deployer.predict(deploy_info["name"], pd_df[["c3", "c1", "c2"]])

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame([False, True], columns=["output"]),
        )

    def test_skl_model_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="skl_model",
                model_dir_path=os.path.join(tmpdir, "skl_model"),
                model=regr,
                sample_input=iris_X,
                metadata={"author": "halu", "version": "1"},
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"skl_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )

            assert di is not None
            res = dc.predict(di["name"], iris_X)
            np.testing.assert_allclose(res["output_feature_0"].values, regr.predict(iris_X))

    def test_skl_model_proba_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        model = ensemble.RandomForestClassifier(random_state=42)
        model.fit(iris_X[:10], iris_y[:10])
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="skl_model_proba",
                model_dir_path=os.path.join(tmpdir, "skl_model_proba"),
                model=model,
                sample_input=iris_X,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["scikit-learn"],
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di_predict = dc.create_deployment(
                name=f"skl_model_predict_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_model_proba"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert di_predict is not None
            res = dc.predict(di_predict["name"], iris_X[:10])
            np.testing.assert_allclose(res["output_feature_0"].values, model.predict(iris_X[:10]))

            di_predict_proba = dc.create_deployment(
                name=f"skl_model_predict_proba_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_model_proba"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict_proba",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert di_predict_proba is not None
            res = dc.predict(di_predict_proba["name"], iris_X[:10])
            np.testing.assert_allclose(res.values, model.predict_proba(iris_X[:10]))

    def test_skl_multiple_output_model_proba_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        model.fit(iris_X[:10], dual_target[:10])
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="skl_multiple_output_model_proba",
                model_dir_path=os.path.join(tmpdir, "skl_multiple_output_model_proba"),
                model=model,
                sample_input=iris_X,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["scikit-learn"],
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di_predict = dc.create_deployment(
                name=f"skl_multiple_output_model_predict_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_multiple_output_model_proba"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert di_predict is not None
            res = dc.predict(di_predict["name"], iris_X[-10:])
            np.testing.assert_allclose(res.values, model.predict(iris_X[-10:]))

            di_predict_proba = dc.create_deployment(
                name=f"skl_multiple_output_model_predict_proba_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_multiple_output_model_proba"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict_proba",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert di_predict_proba is not None
            res = dc.predict(di_predict_proba["name"], iris_X[-10:])
            np.testing.assert_allclose(res.values, np.hstack(model.predict_proba(iris_X[-10:])))

    def test_xgb(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="xgb_model",
                model_dir_path=os.path.join(tmpdir, "xgb_model"),
                model=regressor,
                sample_input=cal_X_test,
                metadata={"author": "halu", "version": "1"},
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di_predict = dc.create_deployment(
                name=f"xgb_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "xgb_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert di_predict is not None
            res = dc.predict(di_predict["name"], cal_X_test)
            np.testing.assert_allclose(res.values, np.expand_dims(regressor.predict(cal_X_test), axis=1))

    def test_xgb_sp(self) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_data_sp_df = self._session.create_dataframe(cal_data.frame)
        cal_data_sp_df_train, cal_data_sp_df_test = tuple(cal_data_sp_df.random_split([0.25, 0.75], seed=2568))
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        cal_data_pd_df_train = cal_data_sp_df_train.to_pandas()
        regressor.fit(cal_data_pd_df_train.drop(columns=["target"]), cal_data_pd_df_train["target"])
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertWarns(UserWarning):
                model_api.save_model(
                    name="xgb_model",
                    model_dir_path=os.path.join(tmpdir, "xgb_model"),
                    model=regressor,
                    sample_input=cal_data_sp_df_train.drop('"target"'),
                    metadata={"author": "halu", "version": "1"},
                )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di_predict = dc.create_deployment(
                name=f"xgb_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "xgb_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )
            assert di_predict is not None
            cal_data_sp_df_test_X = cal_data_sp_df_test.drop('"target"')
            res = dc.predict(di_predict["name"], cal_data_sp_df_test_X)
            np.testing.assert_allclose(
                res.to_pandas().values, np.expand_dims(regressor.predict(cal_data_sp_df_test_X.to_pandas()), axis=1)
            )

    def test_snowml_model_deploy_snowml_sklearn(self) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        # no sample input because snowml can infer the model signature itself
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="snowml_model",
                model_dir_path=os.path.join(tmpdir, "snowml_model"),
                model=regr,
                metadata={"author": "xjiang", "version": "1"},
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"snowml_model{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "snowml_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )

            assert di is not None
            res = dc.predict(di["name"], test_features)
            np.testing.assert_allclose(res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values)

    def test_snowml_model_deploy_xgboost(self) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        # no sample input because snowml can infer the model signature itself
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="snowml_model",
                model_dir_path=os.path.join(tmpdir, "snowml_model"),
                model=regr,
                metadata={"author": "xjiang", "version": "1"},
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"snowml_model{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "snowml_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )

            assert di is not None
            res = dc.predict(di["name"], test_features)
            np.testing.assert_allclose(res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values)

    def test_snowml_model_deploy_lightgbm(self) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        # no sample input because snowml can infer the model signature itself
        with tempfile.TemporaryDirectory() as tmpdir:
            model_api.save_model(
                name="snowml_model",
                model_dir_path=os.path.join(tmpdir, "snowml_model"),
                model=regr,
                metadata={"author": "xjiang", "version": "1"},
            )
            dc = _deployer.Deployer(self._session, _deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"snowml_model{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "snowml_model"),
                platform=_deployer.TargetPlatform.WAREHOUSE,
                target_method="predict",
                options=model_types.WarehouseDeployOptions({"relax_version": True, "_use_local_snowml": True}),
            )

            assert di is not None
            res = dc.predict(di["name"], test_features)
            np.testing.assert_allclose(res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values)


if __name__ == "__main__":
    absltest.main()
