import asyncio
import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model
from snowflake.snowpark._internal import utils as snowpark_utils
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
        self.m1 = self.context["m1"]

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


class DemoModelWithPdSeriesOutPut(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.Series:
        return input["c1"]


class TestRegistryCustomModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_async_model_composition(
        self,
        registry_test_fn: str,
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
            getattr(self, registry_test_fn)(
                model=acm,
                sample_input_data=pd_df,
                prediction_assert_fns={
                    "predict": (
                        pd_df,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(arr[:, 0], columns=["output"], dtype=pd.Float64Dtype()),
                        ),
                    ),
                },
            )

        asyncio.get_event_loop().run_until_complete(_test(self))

    @registry_model_test_base.RegistryModelTestBase.sproc_test(test_owners_rights=False)
    def test_large_input(self) -> None:
        arr = np.random.randint(100, size=(1_000_000, 3))
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        clf = DemoModel(custom_model.ModelContext())
        self._test_registry_model(
            model=clf,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(arr[:, 0], columns=["output"], dtype=pd.Int64Dtype()),
                        check_dtype=False,
                    ),
                ),
            },
            options={"embed_local_ml_library": True},
        )

    def test_custom_demo_model_sp(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [4, 2, 5, 4]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (sp_df, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_decimal(self) -> None:
        import decimal

        lm = DemoModel(custom_model.ModelContext())
        arr = [[decimal.Decimal(1.2), 2.3, 3.4], [decimal.Decimal(4.6), 2.7, 5.5]]
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1.2, 2.3, 3.4, 1.2], [4.6, 2.7, 5.5, 4.6]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (sp_df, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_sp_one_query(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)
        sp_df.write.save_as_table(table_name, mode="errorifexists", table_type="temporary")
        sp_df_2 = self.session.table(table_name)
        assert len(sp_df_2.queries["queries"]) == 1, sp_df_2.queries
        assert len(sp_df_2.queries["post_actions"]) == 0, sp_df_2.queries
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [4, 2, 5, 4]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df_2,
            prediction_assert_fns={
                "predict": (sp_df_2, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_none(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [None, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, None], columns=["output"], dtype=pd.Int64Dtype()),
                        check_dtype=False,
                    ),
                )
            },
        )

    def test_custom_demo_model_none_multi_block(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2]] * 5000 + [[None, 2]] * 5000
        pd_df = pd.DataFrame(arr, columns=["c1", "c2"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1] * 5000 + [None] * 5000, columns=["output"], dtype=pd.Float64Dtype()),
                    ),
                )
            },
        )

    def test_custom_demo_model_none_sp(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [None, 2, 5]]
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [None, 2, 5, None]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (sp_df, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_none_sp_mix1(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [None, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [None, 2, 5, None]], columns=["c1", "c2", "c3", "output"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (sp_df, lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False))
            },
        )

    def test_custom_demo_model_none_sp_mix2(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [None, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, None], columns=["output"], dtype=pd.Int64Dtype()),
                        check_dtype=False,
                    ),
                )
            },
        )

    def test_custom_demo_model_sp_quote(self) -> None:
        lm = DemoModelSPQuote(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self.session.create_dataframe(arr, schema=['"""c1"""', '"""c2"""', '"""c3"""'])
        pd_df = pd.DataFrame(arr, columns=['"c1"', '"c2"', '"c3"'])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=['"output"'], dtype=pd.Int8Dtype()),
                    ),
                )
            },
        )

    def test_custom_demo_model_sp_mix_1(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.concat([pd_df, pd_df[["c1"]].rename(columns={"c1": "output"})], axis=1)
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                )
            },
        )

    def test_custom_demo_model_sp_mix_2(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=["output"], dtype=pd.Int8Dtype()),
                    ),
                )
            },
        )

    def test_custom_demo_model_array(self) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
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

    @absltest.skip("Skip until we support pd.Series as output")
    def test_custom_demo_model_pd_series(self) -> None:
        lm = DemoModelWithPdSeriesOutPut(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_series_equal(
                        res["output"],
                        pd.Series([1, 4], name="output"),
                    ),
                )
            },
        )

    def test_custom_demo_model_str(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
            prediction_assert_fns={
                "predict": (
                    pd_df,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": ["Yogiri", "Artia"]}, dtype=pd.StringDtype()),
                    ),
                )
            },
        )

    def test_custom_demo_model_str_sp_none(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([[None, "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(pd_df)
        y_df_expected = pd.DataFrame(
            [[None, "Civia", "Echo", None], ["Artia", "Doris", "Rosalyn", "Artia"]],
            columns=["c1", "c2", "c3", "output"],
        )
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                )
            },
        )

    def test_custom_demo_model_array_sp(self) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(pd_df)
        y_df_expected = pd.concat([pd_df, pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]})], axis=1)
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                )
            },
        )

    def test_custom_demo_model_str_sp(self) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        sp_df = self.session.create_dataframe(pd_df)
        y_df_expected = pd.concat([pd_df, pd.DataFrame(data={"output": ["Yogiri", "Artia"]})], axis=1)
        self._test_registry_model(
            model=lm,
            sample_input_data=sp_df,
            prediction_assert_fns={
                "predict": (
                    sp_df,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                )
            },
        )

    def test_custom_demo_model_array_str(self) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self._test_registry_model(
            model=lm,
            sample_input_data=pd_df,
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

    def test_custom_model_with_artifacts(self) -> None:
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
                sample_input_data=pd_df,
                prediction_assert_fns={
                    "predict": (
                        pd_df,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame([False, True], columns=["output"], dtype=pd.BooleanDtype()),
                        ),
                    )
                },
            )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
        key_name=["bias", "models", "artifacts"],
    )
    def test_custom_model_with_model_ref(self, registry_test_fn: str, key_name: str) -> None:
        class DemoModelWithXGB(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(self.context[key_name].predict(input), columns=["output"])

        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        data = load_breast_cancer()
        X = data.data
        y = data.target

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize the classifier
        model = XGBClassifier(
            objective="binary:logistic",  # For binary classification
            eval_metric="logloss",  # Evaluation metric
            use_label_encoder=False,  # Disable the use of label encoder to avoid warnings
            random_state=42,  # For reproducibility
        )

        # Train the model
        model.fit(X_train, y_train)

        output = pd.DataFrame(model.predict(X_test), columns=["output"])

        kwargs = {key_name: model}
        lm = DemoModelWithXGB(custom_model.ModelContext(**kwargs))

        getattr(self, registry_test_fn)(
            model=lm,
            sample_input_data=X_test,
            prediction_assert_fns={
                "predict": (
                    X_test,
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        output,
                        check_dtype=False,
                    ),
                )
            },
        )

    @parameterized.product(  # type: ignore[misc]
        key_name=["bias", "models", "artifacts"],
    )
    def test_custom_model_with_kwargs(self, key_name: str) -> None:
        class DemoModelWithKwargs(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)
                with open(context[key_name], encoding="utf-8") as f:
                    v = int(f.read())
                self.bias = v

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": (input["c1"] + self.bias) > 12})

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, key_name), "w", encoding="utf-8") as f:
                f.write("10")
            kwargs = {key_name: os.path.join(tmpdir, key_name)}
            lm = DemoModelWithKwargs(custom_model.ModelContext(**kwargs))
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            self._test_registry_model(
                model=lm,
                sample_input_data=pd_df,
                prediction_assert_fns={
                    "predict": (
                        pd_df,
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame([False, True], columns=["output"]),
                            check_dtype=False,
                        ),
                    )
                },
            )

    def test_custom_model_bool_sp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            sp_df = self.session.create_dataframe(pd_df)
            y_df_expected = pd.concat([pd_df, pd.DataFrame([False, True], columns=["output"])], axis=1)
            self._test_registry_model(
                model=lm,
                sample_input_data=sp_df,
                prediction_assert_fns={
                    "predict": (
                        sp_df,
                        lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                    )
                },
            )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_table_function_column_names(
        self,
        registry_test_fn: str,
    ) -> None:
        class RowOrderModel(custom_model.CustomModel):
            @custom_model.inference_api
            def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "ascending": list(range(len(X))),
                        "output": X["feature_0"],
                    }
                )

        input_df = pd.DataFrame({"feature_0": [1]})
        model = RowOrderModel(custom_model.ModelContext())
        expected_output_df = model.predict(input_df)
        getattr(self, registry_test_fn)(
            model=RowOrderModel(custom_model.ModelContext()),
            sample_input_data=input_df,
            prediction_assert_fns={
                "predict": (
                    input_df,
                    lambda res: pd.testing.assert_index_equal(res.columns, expected_output_df.columns),
                ),
            },
            options={"function_type": "TABLE_FUNCTION"},
        )


if __name__ == "__main__":
    absltest.main()
