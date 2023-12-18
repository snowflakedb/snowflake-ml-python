import asyncio
import os
import tempfile
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model, type_hints as model_types
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import dataframe_utils, db_manager


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


class TestWarehouseCustomModelInteg(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.cleanup_schemas()
        self._db_manager.cleanup_stages()
        self._db_manager.cleanup_user_functions()

        # To create different UDF names among different runs
        self.run_id = uuid.uuid4().hex
        self._test_schema_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "model_deployment_custom_model_test_schema"
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

    def base_test_case(
        self,
        name: str,
        model: model_types.SupportedModelType,
        sample_input: model_types.SupportedDataType,
        test_input: model_types.SupportedDataType,
        deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        warehouse_model_integ_test_utils.base_test_case(
            self._db_manager,
            run_id=self.run_id,
            full_qual_stage=self.full_qual_stage,
            name=name,
            model=model,
            sample_input=sample_input,
            test_input=test_input,
            deploy_params=deploy_params,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_async_model_composition(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        async def _test(self: "TestWarehouseCustomModelInteg") -> None:
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
            self.base_test_case(
                name="async_model_composition",
                model=acm,
                sample_input=pd_df,
                test_input=pd_df,
                deploy_params={
                    "": (
                        {},
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(arr[:, 0], columns=["output"], dtype=float),
                        ),
                    ),
                },
                permanent_deploy=permanent_deploy,
            )

        asyncio.get_event_loop().run_until_complete(_test(self))

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [4, 2, 5, 4]], columns=["c1", "c2", "c3", "output"])
        self.base_test_case(
            name="custom_demo_model_sp0",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp_quote(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModelSPQuote(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self._session.create_dataframe(arr, schema=['"""c1"""', '"""c2"""', '"""c3"""'])
        pd_df = pd.DataFrame(arr, columns=['"c1"', '"c2"', '"c3"'])
        self.base_test_case(
            name="custom_demo_model_sp_quote",
            model=lm,
            sample_input=sp_df,
            test_input=pd_df,
            deploy_params={
                "": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=['"output"'], dtype=np.int8),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp_quote_norm_1(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModelSPQuote(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=['"""c1"""', '"""c2"""', '"""c3"""'])
        sp_df_1 = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.concat([pd_df, pd_df[["c1"]].rename(columns={"c1": "output"})], axis=1)
        self.base_test_case(
            name="custom_demo_model_sp_quote",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df_1,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp_mix_1(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.concat([pd_df, pd_df[["c1"]].rename(columns={"c1": "output"})], axis=1)
        self.base_test_case(
            name="custom_demo_model_sp1",
            model=lm,
            sample_input=pd_df,
            test_input=sp_df,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp_mix_1_norm(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=["c1", "c2", "c3"])
        y_df_expected = pd.concat(
            [pd_df.rename(columns=str.upper), pd_df[["c1"]].rename(columns={"c1": "OUTPUT"})], axis=1
        )
        self.base_test_case(
            name="custom_demo_model_sp1",
            model=lm,
            sample_input=pd_df,
            test_input=sp_df,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp_mix_2(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        self.base_test_case(
            name="custom_demo_model_sp2",
            model=lm,
            sample_input=sp_df,
            test_input=pd_df,
            deploy_params={
                "": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_sp_mix_2_norm(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        sp_df_1 = self._session.create_dataframe(arr, schema=["c1", "c2", "c3"])
        pd_df = pd.DataFrame(arr, columns=["C1", "C2", "C3"])
        y_df_expected = pd.concat([pd_df, pd_df[["C1"]].rename(columns={"C1": "OUTPUT"})], axis=1)
        self.base_test_case(
            name="custom_demo_model_sp2",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df_1,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_array(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        self.base_test_case(
            name="custom_demo_model_array",
            model=lm,
            sample_input=pd_df,
            test_input=pd_df,
            deploy_params={
                "": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]}),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_str(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self.base_test_case(
            name="custom_demo_model_str",
            model=lm,
            sample_input=pd_df,
            test_input=pd_df,
            deploy_params={
                "": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": ["Yogiri", "Artia"]}),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_array_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(pd_df)
        y_df_expected = pd.concat([pd_df, pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]})], axis=1)
        self.base_test_case(
            name="custom_demo_model_array_sp",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                )
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_str_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(pd_df)
        y_df_expected = pd.concat([pd_df, pd.DataFrame(data={"output": ["Yogiri", "Artia"]})], axis=1)
        self.base_test_case(
            name="custom_demo_model_str_sp",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df,
            deploy_params={
                "": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected),
                )
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_demo_model_array_str(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        self.base_test_case(
            name="custom_demo_model_array_str",
            model=lm,
            sample_input=pd_df,
            test_input=pd_df,
            deploy_params={
                "": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]]}),
                    ),
                )
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_model_with_artifacts(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            self.base_test_case(
                name="custom_model_with_artifacts",
                model=lm,
                sample_input=pd_df,
                test_input=pd_df,
                deploy_params={
                    "": (
                        {},
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame([False, True], columns=["output"]),
                        ),
                    )
                },
                permanent_deploy=permanent_deploy,
            )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_custom_model_bool_sp(
        self,
        permanent_deploy: Optional[bool] = False,
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
            self.base_test_case(
                name="custom_model_bool_sp",
                model=lm,
                sample_input=sp_df,
                test_input=sp_df,
                deploy_params={
                    "": (
                        {},
                        lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                    )
                },
                permanent_deploy=permanent_deploy,
            )


if __name__ == "__main__":
    absltest.main()
