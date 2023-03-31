import asyncio
import os

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, linear_model, svm

from snowflake.ml.model.custom_model import CustomModel, ModelContext
from snowflake.ml.model.model import load_model, save_model
from snowflake.ml.model.type_spec import (
    DataType,
    NumpyNdarray,
    PandasDataFrame,
    PandasSeries,
)


class DemoModel(CustomModel):
    def __init__(self, context: ModelContext) -> None:
        super().__init__(context)

    @CustomModel.api(
        input_spec=PandasDataFrame(dtypes=DataType.long, dim=3, cols=["c1", "c2", "c3"]),
        output_spec=PandasSeries(dtype=DataType.long, cols=["oc1"]),
    )
    def predict(self, input: pd.DataFrame) -> pd.Series:
        return input["c1"]


class AsyncComposeModel(CustomModel):
    def __init__(self, context: ModelContext) -> None:
        super().__init__(context)

    @CustomModel.api(
        input_spec=NumpyNdarray(dtype=DataType.float, dim=64), output_spec=NumpyNdarray(dtype=DataType.float, dim=1)
    )
    async def predict(self, input: np.ndarray) -> np.ndarray:
        res_sum = await self.context.model_ref("m1").predict.async_run(input) + self.context.model_ref("m2").predict(
            input
        )
        return res_sum / 2


class ModelTest(absltest.TestCase):
    def test_custom(self) -> None:
        tmpdir = self.create_tempdir()
        lm = DemoModel(ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        lm.predict(d)
        save_model(
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=lm,
            metadata={"author": "halu", "version": 1},
            pip_requirements=["scikit-learn"],
        )

        m, meta = load_model(os.path.join(tmpdir.full_path, "model1"))
        res = m.predict(d)
        self.assertTrue(np.allclose(res, pd.Series(np.array([1, 4]))))
        self.assertEqual(meta.metadata["author"], "halu")

    def test_skl(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        regr = linear_model.LinearRegression()
        regr.fit(iris_X, iris_y)
        tmpdir = self.create_tempdir()
        save_model(
            name="model1",
            model_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=regr,
            metadata={"author": "halu", "version": 1},
            pip_requirements=["scikit-learn"],
            sample_data=iris_X,
        )
        m, _ = load_model(os.path.join(tmpdir.full_path, "model1"))
        assert np.allclose(np.array([-0.08254936]), m.predict(iris_X[:1]))

    def test_async_model_composition(self) -> None:
        async def _test(self: "ModelTest") -> None:
            digits = datasets.load_digits()
            clf = svm.SVC(gamma=0.001, C=100.0)
            clf.fit(digits.data[:-10], digits.target[:-10])
            model_context = ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            p1 = clf.predict(digits.data[-10:])
            p2 = await acm.predict(digits.data[-10:])
            tmpdir = self.create_tempdir()
            save_model(
                name="model1",
                model_dir_path=os.path.join(tmpdir.full_path, "model1"),
                model=acm,
                metadata={"author": "halu", "version": 1},
                pip_requirements=["scikit-learn"],
            )
            lm, _ = load_model(os.path.join(tmpdir.full_path, "model1"))
            p3 = await lm.predict(digits.data[-10:])
            assert np.allclose(p1, p2)
            assert np.allclose(p2, p3)

        asyncio.get_event_loop().run_until_complete(_test(self))


if __name__ == "__main__":
    absltest.main()
