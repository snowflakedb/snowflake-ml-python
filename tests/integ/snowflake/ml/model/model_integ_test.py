#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import asyncio
import os
import random
import string

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from snowflake.ml.model import custom_model, deployer, model as model_api, schema
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class AsyncComposeModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

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

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


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
            s = schema.infer_schema(d, p2)
            tmpdir = self.create_tempdir()
            model_api.save_model(
                name="async_model_composition",
                model_dir_path=os.path.join(tmpdir.full_path, "async_model_composition"),
                model=acm,
                schema=s,
                metadata={"author": "halu", "version": "1"},
            )
            dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
            di = dc.create_deployment(
                name=f"async_model_composition_{self.run_id}",
                model_dir_path=os.path.join(tmpdir.full_path, "async_model_composition"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": True},
            )
            res = dc.predict(di.name, d)

            pd.testing.assert_series_equal(
                res["OUTPUT"].astype(str).astype(int),
                pd.Series([1, 4]),
                check_dtype=False,
                check_names=False,
                check_index=False,
            )

            self.assertTrue(di in dc.list_deployments())
            self.assertEqual(di, dc.get_deployment(f"async_model_composition_{self.run_id}"))

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_bad_model_deploy(self) -> None:
        tmpdir = self.create_tempdir()
        lm = DemoModel(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        s = schema.infer_schema(d, lm.predict(d))
        model_api.save_model(
            name="custom_bad_model",
            model_dir_path=os.path.join(tmpdir.full_path, "custom_bad_model"),
            model=lm,
            schema=s,
            metadata={"author": "halu", "version": "1"},
            pip_requirements=["numpy==1.22.4"],
        )

        dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
        with self.assertRaises(RuntimeError):
            _ = dc.create_deployment(
                name=f"custom_bad_model_{self.run_id}",
                model_dir_path=os.path.join(tmpdir.full_path, "custom_bad_model"),
                platform=deployer.TargetPlatform.WAREHOUSE,
                options={"relax_version": False},
            )

        with self.assertRaises(ValueError):
            _ = dc.predict(f"custom_bad_model_{self.run_id}", d)

    def test_custom_demo_model(self) -> None:
        tmpdir = self.create_tempdir()
        lm = DemoModel(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        s = schema.infer_schema(d, lm.predict(d))
        model_api.save_model(
            name="custom_demo_model",
            model_dir_path=os.path.join(tmpdir.full_path, "custom_demo_model"),
            model=lm,
            schema=s,
            metadata={"author": "halu", "version": "1"},
        )

        dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
        di = dc.create_deployment(
            name=f"custom_demo_model_{self.run_id}",
            model_dir_path=os.path.join(tmpdir.full_path, "custom_demo_model"),
            platform=deployer.TargetPlatform.WAREHOUSE,
            options={"relax_version": True},
        )
        res = dc.predict(di.name, d)

        pd.testing.assert_series_equal(
            res["OUTPUT"].astype(str).astype(int),
            pd.Series([1, 4]),
            check_dtype=False,
            check_names=False,
            check_index=False,
        )

        self.assertTrue(di in dc.list_deployments())
        self.assertEqual(di, dc.get_deployment(f"custom_demo_model_{self.run_id}"))

    def test_custom_model_with_artifacts(self) -> None:
        tmpdir = self.create_tempdir()
        with open(os.path.join(tmpdir.full_path, "bias"), "w") as f:
            f.write("10")
        lm = DemoModelWithArtifacts(
            custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir.full_path, "bias")})
        )
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        s = schema.infer_schema(d, lm.predict(d))
        model_api.save_model(
            name="custom_model_with_artifacts",
            model_dir_path=os.path.join(tmpdir.full_path, "custom_model_with_artifacts"),
            model=lm,
            schema=s,
            metadata={"author": "halu", "version": "1"},
        )

        dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
        di = dc.create_deployment(
            name=f"custom_model_with_artifacts_{self.run_id}",
            model_dir_path=os.path.join(tmpdir.full_path, "custom_model_with_artifacts"),
            platform=deployer.TargetPlatform.WAREHOUSE,
            options={"relax_version": True},
        )
        res = dc.predict(di.name, d)

        pd.testing.assert_series_equal(
            res["OUTPUT"].astype(str).astype(int),
            pd.Series([11, 14]),
            check_dtype=False,
            check_names=False,
            check_index=False,
        )

    def test_skl_model_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)
        tmpdir = self.create_tempdir()
        s = schema.infer_schema(iris_X_df, regr.predict(iris_X_df))
        model_api.save_model(
            name="skl_model",
            model_dir_path=os.path.join(tmpdir.full_path, "skl_model"),
            model=regr,
            schema=s,
            metadata={"author": "halu", "version": "1"},
        )
        dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
        di = dc.create_deployment(
            name=f"skl_model_{self.run_id}",
            model_dir_path=os.path.join(tmpdir.full_path, "skl_model"),
            platform=deployer.TargetPlatform.WAREHOUSE,
            options={"relax_version": True},
        )
        res = dc.predict(di.name, iris_X_df[:1])
        self.assertTrue(np.allclose(np.array([-0.08254936]), res.astype(str).astype(float).values))

    def test_skl_model_proba_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        model = RandomForestClassifier(random_state=42)
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df, iris_y)
        tmpdir = self.create_tempdir()
        s = schema.infer_schema(iris_X_df, model.predict_proba(iris_X_df))
        model_api.save_model(
            name="skl_model_proba",
            model_dir_path=os.path.join(tmpdir.full_path, "skl_model_proba"),
            model=model,
            schema=s,
            metadata={"author": "halu", "version": "1"},
            pip_requirements=["scikit-learn"],
            target_method="predict_proba",
        )
        dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
        di = dc.create_deployment(
            name=f"skl_model_proba_{self.run_id}",
            model_dir_path=os.path.join(tmpdir.full_path, "skl_model_proba"),
            platform=deployer.TargetPlatform.WAREHOUSE,
            options={"relax_version": True},
        )
        res = dc.predict(di.name, iris_X_df[:10])
        self.assertTrue(np.allclose(res.astype(str).astype(float).values, model.predict_proba(iris_X_df[:10])))

    def test_skl_multiple_output_model_proba_deploy(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df[:-10], dual_target[:-10])
        tmpdir = self.create_tempdir()
        s = schema.infer_schema(iris_X_df, model.predict_proba(iris_X_df))
        model_api.save_model(
            name="skl_multiple_output_model_proba",
            model_dir_path=os.path.join(tmpdir.full_path, "skl_multiple_output_model_proba"),
            model=model,
            schema=s,
            metadata={"author": "halu", "version": "1"},
            pip_requirements=["scikit-learn"],
            target_method="predict_proba",
        )
        dc = deployer.Deployer(self._session, deployer.LocalDeploymentManager())
        di = dc.create_deployment(
            name=f"skl_multiple_output_model_proba_{self.run_id}",
            model_dir_path=os.path.join(tmpdir.full_path, "skl_multiple_output_model_proba"),
            platform=deployer.TargetPlatform.WAREHOUSE,
            options={"relax_version": True},
        )
        res = dc.predict(di.name, iris_X_df[-10:])
        self.assertTrue(
            np.allclose(res.astype(str).astype(float).values, np.hstack(model.predict_proba(iris_X_df[-10:])))
        )


if __name__ == "__main__":
    absltest.main()
