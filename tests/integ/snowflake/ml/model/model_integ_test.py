#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import asyncio
import os
import random
import string
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, ensemble, linear_model, multioutput

from snowflake.ml.model import (
    custom_model,
    deployer,
    model as model_api,
    model_signature,
)
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


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


class TestModelInteg(absltest.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        # To create different UDF names among different runs
        self.run_id = "".join(random.choices(string.ascii_lowercase, k=8))

    def tearDown(self) -> None:
        self._session.close()

    def test_async_model_composition(self) -> None:
        async def _test(self: "TestModelInteg") -> None:
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            clf = DemoModel(custom_model.ModelContext())
            model_context = custom_model.ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            p2 = await acm.predict(d)
            s = model_signature.infer_signature(d, p2)
            with tempfile.TemporaryDirectory() as tmpdir:
                model_api.save_model(
                    name="async_model_composition",
                    model_dir_path=os.path.join(tmpdir, "async_model_composition"),
                    model=acm,
                    signature=s,
                    metadata={"author": "halu", "version": "1"},
                )
                dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
                di = dc.create_deployment(
                    name=f"async_model_composition_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "async_model_composition"),
                    platform=deployer.TargetPlatform.WAREHOUSE,
                    options={"relax_version": True},
                )

                assert di is not None
                res = dc.predict(di.name, d)

                pd.testing.assert_frame_equal(
                    res,
                    pd.DataFrame([1.0, 4.0], columns=["output"]),
                )

                self.assertTrue(di in dc.list_deployments())
                self.assertEqual(di, dc.get_deployment(f"async_model_composition_{self.run_id}"))

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_bad_model_deploy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = model_signature.infer_signature(d, lm.predict(d))
            model_api.save_model(
                name="custom_bad_model",
                model_dir_path=os.path.join(tmpdir, "custom_bad_model"),
                model=lm,
                signature=s,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["numpy==1.22.4"],
            )

            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            with self.assertRaises(RuntimeError):
                _ = dc.create_deployment(
                    name=f"custom_bad_model_{self.run_id}",
                    model_dir_path=os.path.join(tmpdir, "custom_bad_model"),
                    platform=deployer.TargetPlatform.WAREHOUSE,
                    options={"relax_version": False},
                )

            with self.assertRaises(ValueError):
                _ = dc.predict(f"custom_bad_model_{self.run_id}", d)

    def test_custom_demo_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = DemoModel(custom_model.ModelContext())
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = model_signature.infer_signature(d, lm.predict(d))
            model_api.save_model(
                name="custom_demo_model",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                model=lm,
                signature=s,
                metadata={"author": "halu", "version": "1"},
            )

            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"custom_demo_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_demo_model"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": True},
            )
            assert di is not None
            res = dc.predict(di.name, d)

            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame([1, 4], columns=["output"]),
            )

            self.assertTrue(di in dc.list_deployments())
            self.assertEqual(di, dc.get_deployment(f"custom_demo_model_{self.run_id}"))

    def test_custom_model_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = model_signature.infer_signature(d, lm.predict(d))
            model_api.save_model(
                name="custom_model_with_artifacts",
                model_dir_path=os.path.join(tmpdir, "custom_model_with_artifacts"),
                model=lm,
                signature=s,
                metadata={"author": "halu", "version": "1"},
            )

            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"custom_model_with_artifacts_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "custom_model_with_artifacts"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": True},
            )
            assert di is not None
            res = dc.predict(di.name, d)

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame([False, True], columns=["output"]),
        )

    def test_skl_model_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = model_signature.infer_signature(iris_X_df, regr.predict(iris_X_df))
            model_api.save_model(
                name="skl_model",
                model_dir_path=os.path.join(tmpdir, "skl_model"),
                model=regr,
                signature=s,
                metadata={"author": "halu", "version": "1"},
            )
            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"skl_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_model"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": True},
            )

            assert di is not None
            res = dc.predict(di.name, iris_X_df[:1])
            np.testing.assert_allclose(np.array([[-0.08254936]]), res.values)

    def test_skl_model_proba_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        model = ensemble.RandomForestClassifier(random_state=42)
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df, iris_y)
        with tempfile.TemporaryDirectory() as tmpdir:
            s = model_signature.infer_signature(iris_X_df, model.predict_proba(iris_X_df))
            model_api.save_model(
                name="skl_model_proba",
                model_dir_path=os.path.join(tmpdir, "skl_model_proba"),
                model=model,
                signature=s,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["scikit-learn"],
                target_method="predict_proba",
            )
            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"skl_model_proba_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_model_proba"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": True},
            )
            assert di is not None
            res = dc.predict(di.name, iris_X_df[:10])
            np.testing.assert_allclose(res.values, model.predict_proba(iris_X_df[:10]))

    def test_skl_multiple_output_model_proba_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df[:-10], dual_target[:-10])
        with tempfile.TemporaryDirectory() as tmpdir:
            s = model_signature.infer_signature(iris_X_df, model.predict_proba(iris_X_df))
            model_api.save_model(
                name="skl_multiple_output_model_proba",
                model_dir_path=os.path.join(tmpdir, "skl_multiple_output_model_proba"),
                model=model,
                signature=s,
                metadata={"author": "halu", "version": "1"},
                conda_dependencies=["scikit-learn"],
                target_method="predict_proba",
            )
            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"skl_multiple_output_model_proba_{self.run_id}",
                model_dir_path=os.path.join(tmpdir, "skl_multiple_output_model_proba"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": True},
            )
            assert di is not None
            res = dc.predict(di.name, iris_X_df[-10:])
            np.testing.assert_allclose(res.values, np.hstack(model.predict_proba(iris_X_df[-10:])))


if __name__ == "__main__":
    absltest.main()
