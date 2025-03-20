import asyncio

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, svm

from snowflake.ml.model import custom_model


class ModelTest(absltest.TestCase):
    def test_bad_custom_model_predict_type_incorrect(self) -> None:
        class BadModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input):  # type: ignore[no-untyped-def]
                return pd.DataFrame(input)

        arr = np.array([[2, 5], [6, 8]])
        d = pd.DataFrame(arr, columns=["c1", "c2"])
        with self.assertRaises(TypeError):
            _ = BadModel(custom_model.ModelContext())

        class GoodModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(input)

        good_model = GoodModel(custom_model.ModelContext())
        _ = good_model.predict(d)
        self.assertListEqual(list(good_model._get_infer_methods()), [GoodModel.predict])

        @custom_model.inference_api
        def bad_predict_1(self, input):  # type: ignore[no-untyped-def]
            return pd.DataFrame(input)

        with self.assertRaises(TypeError):
            good_model.predict = bad_predict_1  # type: ignore[assignment]

        with self.assertRaises(TypeError):

            @custom_model.inference_api  # type: ignore[arg-type]
            def bad_predict_2(input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(input)

            bad_predict_2 = bad_predict_2.__get__(good_model, type(good_model))

            good_model.predict = bad_predict_2  # type: ignore[assignment]

        class AnotherBadModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> int:
                return 42

        with self.assertRaises(TypeError):
            _ = AnotherBadModel(custom_model.ModelContext())

        class BadAsyncModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            async def predict(self, input: int) -> pd.DataFrame:
                await asyncio.sleep(0.1)
                return pd.DataFrame([input])

        async def _test(self: "ModelTest") -> None:
            with self.assertRaises(TypeError):
                bad_async_model = BadAsyncModel(custom_model.ModelContext())
                _ = await bad_async_model.predict(d)

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_custom_model_type(self) -> None:
        class DemoModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": input["c1"]})

        lm = DemoModel(custom_model.ModelContext())
        self.assertListEqual(list(lm._get_infer_methods()), [DemoModel.predict])
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        res = lm.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([1, 4]))))

        class AnotherDemoModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict_c1(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": input["c1"]})

            @custom_model.inference_api
            def predict_c2(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": input["c2"]})

        lm_2 = AnotherDemoModel(custom_model.ModelContext())
        self.assertListEqual(
            list(lm_2._get_infer_methods()),
            [AnotherDemoModel.predict_c1, AnotherDemoModel.predict_c2],
        )
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        res = lm_2.predict_c1(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([1, 4]))))
        res = lm_2.predict_c2(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([2, 2]))))

    def test_custom_async_model_composition_type(self) -> None:
        class AsyncComposeModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)
                self.m1 = self.context["m1"]

            @custom_model.inference_api
            async def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                res_sum = await self.m1.predict.async_run(input) + self.context.model_ref(  # type: ignore[union-attr]
                    "m2"
                ).predict(input)
                return pd.DataFrame({"output": res_sum / 2})

        async def _test(self: "ModelTest") -> None:
            digits = datasets.load_digits()
            clf = svm.SVC(gamma=0.001, C=100.0)
            clf.fit(digits.data[:-10], digits.target[:-10])
            model_context = custom_model.ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            self.assertListEqual(list(acm._get_infer_methods()), [AsyncComposeModel.predict])
            digits_df = pd.DataFrame(digits.data, columns=[f"col_{i}" for i in range(digits.data.shape[1])])
            p1 = clf.predict(digits_df[-10:])
            p2 = await acm.predict(digits_df[-10:])
            self.assertTrue(np.allclose(p1, p2["output"]))

        asyncio.get_event_loop().run_until_complete(_test(self))


if __name__ == "__main__":
    absltest.main()
