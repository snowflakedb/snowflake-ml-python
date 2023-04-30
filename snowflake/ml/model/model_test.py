import asyncio
import os

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, ensemble, linear_model, multioutput

from snowflake.ml.model import custom_model, model as model_api, schema


class DemoModelWithManyArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(os.path.join(context.path("bias"), "bias1")) as f:
            v1 = int(f.read())
        with open(os.path.join(context.path("bias"), "bias2")) as f:
            v2 = int(f.read())
        self.bias = v1 + v2

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


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
        return pd.DataFrame(res_sum / 2)


class DemoModelWithArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(context.path("bias")) as f:
            v = int(f.read())
        self.bias = v

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class ModelTest(absltest.TestCase):
    def test_custom_model_with_multiple_artifacts(self) -> None:
        tmpdir = self.create_tempdir()
        os.mkdir(os.path.join(tmpdir.full_path, "bias"))
        with open(os.path.join(tmpdir.full_path, "bias", "bias1"), "w") as f:
            f.write("25")
        with open(os.path.join(tmpdir.full_path, "bias", "bias2"), "w") as f:
            f.write("68")
        lm = DemoModelWithManyArtifacts(
            custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir.full_path, "bias")})
        )
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        s = schema.infer_schema(d, lm.predict(d))
        model_api.save_model(
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=lm,
            schema=s,
            metadata={"author": "halu", "version": "1"},
        )
        m: custom_model.CustomModel
        m, meta = model_api.load_model(os.path.join(tmpdir.full_path, "model1"))
        res = m.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([94, 97]))))
        m_UDF, meta = model_api._load_model_for_deploy(os.path.join(tmpdir.full_path, "model1"))
        res = m_UDF.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([94, 97]))))
        self.assertEqual(meta.metadata["author"] if meta.metadata else None, "halu")

    def test_async_model_composition(self) -> None:
        async def _test(self: "ModelTest") -> None:
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
            p1 = clf.predict(d)
            p2 = await acm.predict(d)
            s = schema.infer_schema(d, p2)
            tmpdir = self.create_tempdir()
            model_api.save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir.full_path, "model1"),
                model=acm,
                schema=s,
                metadata={"author": "halu", "version": "1"},
            )
            lm: custom_model.CustomModel
            lm, _ = model_api.load_model(os.path.join(tmpdir.full_path, "model1"))
            p3 = await lm.predict(d)
            m_UDF, _ = model_api._load_model_for_deploy(os.path.join(tmpdir.full_path, "model1"))
            p4 = await m_UDF.predict(d)
            self.assertTrue(np.allclose(p1, p2))
            self.assertTrue(np.allclose(p2, p3))
            self.assertTrue(np.allclose(p2, p4))

        asyncio.get_event_loop().run_until_complete(_test(self))

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
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=lm,
            schema=s,
            metadata={"author": "halu", "version": "1"},
        )

        m: custom_model.CustomModel
        m, meta = model_api.load_model(os.path.join(tmpdir.full_path, "model1"))
        res = m.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([11, 14]))))

        # test re-init when loading the model
        with open(os.path.join(tmpdir.full_path, "model1", "models", "model1", "artifacts", "bias"), "w") as f:
            f.write("20")

        m_UDF, meta = model_api._load_model_for_deploy(os.path.join(tmpdir.full_path, "model1"))
        res = m_UDF.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([21, 24]))))
        self.assertEqual(meta.metadata["author"] if meta.metadata else None, "halu")

    def test_skl_multiple_output_proba(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        model.fit(iris_X_df[:-10], dual_target[:-10])
        tmpdir = self.create_tempdir()
        s = schema.infer_schema(iris_X_df, model.predict_proba(iris_X_df))
        model_api.save_model(
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=model,
            schema=s,
            metadata={"author": "halu", "version": "1"},
            pip_requirements=["scikit-learn"],
            target_method="predict_proba",
        )

        m: multioutput.MultiOutputClassifier
        m, _ = model_api.load_model(os.path.join(tmpdir.full_path, "model1"))
        self.assertTrue(
            np.allclose(np.hstack(model.predict_proba(iris_X_df[-10:])), np.hstack(m.predict_proba(iris_X_df[-10:])))
        )
        m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir.full_path, "model1"))
        self.assertTrue(np.allclose(np.hstack(model.predict_proba(iris_X_df[-10:])), m_udf.predict(iris_X_df[-10:])))

    def test_skl(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        iris_X_df = pd.DataFrame(iris_X, columns=["c1", "c2", "c3", "c4"])
        regr.fit(iris_X_df, iris_y)
        tmpdir = self.create_tempdir()
        s = schema.infer_schema(iris_X_df, regr.predict(iris_X_df))
        model_api.save_model(
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=regr,
            schema=s,
            metadata={"author": "halu", "version": "1"},
        )
        m: linear_model.LinearRegression
        m, _ = model_api.load_model(os.path.join(tmpdir.full_path, "model1"))
        self.assertTrue(np.allclose(np.array([-0.08254936]), m.predict(iris_X_df[:1])))
        m_udf, _ = model_api._load_model_for_deploy(os.path.join(tmpdir.full_path, "model1"))
        self.assertTrue(np.allclose(np.array([-0.08254936]), m_udf.predict(iris_X_df[:1])))


if __name__ == "__main__":
    absltest.main()
