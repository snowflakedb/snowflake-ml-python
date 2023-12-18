import asyncio
import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils


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
        with open(context.path("bias"), encoding="utf-8") as f:
            v = int(f.read())
        self.bias = v

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": (input["c1"] + self.bias) > 12})


class TestRegistryCustomModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_async_model_composition(
        self,
    ) -> None:
        async def _test(self: "TestRegistryCustomModelInteg") -> None:
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
            self._test_registry_model(
                model=acm,
                sample_input=pd_df,
                prediction_assert_fns={
                    "predict": (
                        pd_df,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(arr[:, 0], columns=["output"], dtype=float),
                        ),
                    ),
                },
            )

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_custom_demo_model_sp(
        self,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [4, 2, 5, 4]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input=sp_df,
            prediction_assert_fns={
                "predict": (sp_df, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_sp_quote(
        self,
    ) -> None:
        lm = DemoModelSPQuote(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self._session.create_dataframe(arr, schema=['"""c1"""', '"""c2"""', '"""c3"""'])
        pd_df = pd.DataFrame(arr, columns=['"c1"', '"c2"', '"c3"'])
        self._test_registry_model(
            model=lm,
            sample_input=sp_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=['"output"'], dtype=np.int8),
                    ),
                )
            },
        )

    def test_custom_demo_model_sp_mix_1(
        self,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.concat([pd_df, pd_df[["c1"]].rename(columns={"c1": "output"})], axis=1)
        self._test_registry_model(
            model=lm,
            sample_input=pd_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                )
            },
        )

    def test_custom_demo_model_sp_mix_2(
        self,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        self._test_registry_model(
            model=lm,
            sample_input=sp_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
                    ),
                )
            },
        )

    def test_custom_demo_model_array(
        self,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]}),
                    ),
                )
            },
        )

    def test_custom_demo_model_str(
        self,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": ["Yogiri", "Artia"]}),
                    ),
                )
            },
        )

    def test_custom_demo_model_array_sp(
        self,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(pd_df)
        y_df_expected = pd.concat([pd_df, pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]})], axis=1)
        self._test_registry_model(
            model=lm,
            sample_input=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                )
            },
        )

    def test_custom_demo_model_str_sp(
        self,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(pd_df)
        y_df_expected = pd.concat([pd_df, pd.DataFrame(data={"output": ["Yogiri", "Artia"]})], axis=1)
        self._test_registry_model(
            model=lm,
            sample_input=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                )
            },
        )

    def test_custom_demo_model_array_str(
        self,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]]}),
                    ),
                )
            },
        )

    def test_custom_model_with_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            self._test_registry_model(
                model=lm,
                sample_input=pd_df,
                prediction_assert_fns={
                    "predict": (
                        pd_df,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame([False, True], columns=["output"]),
                        ),
                    )
                },
            )

    def test_custom_model_bool_sp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            sp_df = self._session.create_dataframe(pd_df)
            y_df_expected = pd.concat([pd_df, pd.DataFrame([False, True], columns=["output"])], axis=1)
            self._test_registry_model(
                model=lm,
                sample_input=sp_df,
                prediction_assert_fns={
                    "predict": (
                        sp_df,
                        lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                    )
                },
            )


if __name__ == "__main__":
    absltest.main()
