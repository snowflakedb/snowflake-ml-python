#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import xgboost
from absl import flags
from absl.testing import absltest, parameterized
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
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session

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


class TestModelInteg(parameterized.TestCase):
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

    def base_test_case(
        self,
        name: str,
        model: model_types.SupportedModelType,
        sample_input: model_types.SupportedDataType,
        test_input: model_types.SupportedDataType,
        deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            version_args: Dict[str, Any]
            tmp_stage = self._session.get_session_stage()
            actual_name = f"{name}_v_current"
            version_args = {"options": {"embed_local_ml_library": True}}
            if model_in_stage:
                actual_name = f"{actual_name}_remote"
                location_args = {
                    "session": self._session,
                    "model_stage_file_path": os.path.join(tmp_stage, f"{actual_name}_{self.run_id}.zip"),
                }
            else:
                actual_name = f"{actual_name}_local"
                location_args = {"model_dir_path": os.path.join(tmpdir, actual_name)}

            model_api.save_model(  # type:ignore[call-overload]
                name=actual_name,
                model=model,
                sample_input=sample_input,
                metadata={"author": "halu", "version": "1"},
                **location_args,
                **version_args,
            )
            for target_method, (additional_deploy_options, check_func) in deploy_params.items():
                if permanent_deploy:
                    permanent_deploy_args = {"permanent_udf_stage_location": f"@{self.stage_qual_name}/"}
                else:
                    permanent_deploy_args = {}
                if "session" not in location_args:
                    location_args.update(session=self._session)
                deploy_info = _deployer.deploy(
                    name=f"{actual_name}_{target_method}_{self.run_id}",
                    **location_args,
                    platform=_deployer.TargetPlatform.WAREHOUSE,
                    target_method=target_method,
                    options={
                        "relax_version": True,
                        **permanent_deploy_args,  # type: ignore[arg-type]
                        **additional_deploy_options,
                    },  # type: ignore[call-overload]
                )

                assert deploy_info is not None
                res = _deployer.predict(session=self._session, deployment=deploy_info, X=test_input)

                check_func(res)

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_async_model_composition(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
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
            self.base_test_case(
                name="async_model_composition",
                model=acm,
                sample_input=pd_df,
                test_input=pd_df,
                deploy_params={
                    "predict": (
                        {},
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame(arr[:, 0], columns=["output"], dtype=float),
                        ),
                    ),
                },
                model_in_stage=model_in_stage,
                permanent_deploy=permanent_deploy,
            )

        asyncio.get_event_loop().run_until_complete(_test(self))

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        self.base_test_case(
            name="custom_demo_model_sp0",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res.to_pandas(),
                        pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_sp_quote(
        self,
        model_in_stage: Optional[bool] = False,
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
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=['"output"'], dtype=np.int8),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_sp_mix_1(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = [[1, 2, 3], [4, 2, 5]]
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(arr, schema=['"c1"', '"c2"', '"c3"'])
        self.base_test_case(
            name="custom_demo_model_sp1",
            model=lm,
            sample_input=pd_df,
            test_input=sp_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res.to_pandas(),
                        pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_sp_mix_2(
        self,
        model_in_stage: Optional[bool] = False,
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
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame([1, 4], columns=["output"], dtype=np.int8),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_array(
        self,
        model_in_stage: Optional[bool] = False,
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
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]}),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_str(
        self,
        model_in_stage: Optional[bool] = False,
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
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": ["Yogiri", "Artia"]}),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_array_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModelArray(custom_model.ModelContext())
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(pd_df)
        self.base_test_case(
            name="custom_demo_model_array_sp",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res.to_pandas().applymap(json.loads),
                        pd.DataFrame(data={"output": [[1, 2, 3], [4, 2, 5]]}),
                    ),
                )
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_str_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        pd_df = pd.DataFrame([["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]], columns=["c1", "c2", "c3"])
        sp_df = self._session.create_dataframe(pd_df)
        self.base_test_case(
            name="custom_demo_model_str_sp",
            model=lm,
            sample_input=sp_df,
            test_input=sp_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res.to_pandas(),
                        pd.DataFrame(data={"output": ["Yogiri", "Artia"]}),
                    ),
                )
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_array_str(
        self,
        model_in_stage: Optional[bool] = False,
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
                "predict": (
                    {},
                    lambda res: pd.testing.assert_frame_equal(
                        res,
                        pd.DataFrame(data={"output": [["Yogiri", "Civia", "Echo"], ["Artia", "Doris", "Rosalyn"]]}),
                    ),
                )
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_with_input_no_keep_order(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = np.random.randint(100, size=(10000, 3))
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        self.base_test_case(
            name="custom_demo_model_with_input_no_keep_order",
            model=lm,
            sample_input=pd_df,
            test_input=pd_df,
            deploy_params={
                "predict": (
                    {"output_with_input_features": True, "keep_order": False},
                    lambda res: pd.testing.assert_series_equal(
                        res["output"], res["c1"], check_dtype=False, check_names=False
                    ),
                )
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_demo_model_with_input(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        lm = DemoModel(custom_model.ModelContext())
        arr = np.random.randint(100, size=(10000, 3))
        pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

        def check_res(res: pd.DataFrame) -> Any:
            pd.testing.assert_series_equal(res["output"], res["c1"], check_dtype=False, check_names=False)
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(
                    np.concatenate([arr, np.expand_dims(arr[:, 0], axis=1)], axis=1),
                    columns=["c1", "c2", "c3", "output"],
                ),
                check_dtype=False,
            )

        self.base_test_case(
            name="custom_demo_model_with_input",
            model=lm,
            sample_input=pd_df,
            test_input=pd_df,
            deploy_params={"predict": ({"output_with_input_features": True}, check_res)},
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_model_with_artifacts(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w") as f:
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
                    "predict": (
                        {},
                        lambda res: pd.testing.assert_frame_equal(
                            res,
                            pd.DataFrame([False, True], columns=["output"]),
                        ),
                    )
                },
                model_in_stage=model_in_stage,
                permanent_deploy=permanent_deploy,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_custom_model_bool_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            pd_df = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            sp_df = self._session.create_dataframe(pd_df)
            self.base_test_case(
                name="custom_model_bool_sp",
                model=lm,
                sample_input=sp_df,
                test_input=sp_df,
                deploy_params={
                    "predict": (
                        {},
                        lambda res: pd.testing.assert_frame_equal(
                            res.to_pandas(),
                            pd.DataFrame([False, True], columns=["output"]),
                        ),
                    )
                },
                model_in_stage=model_in_stage,
                permanent_deploy=permanent_deploy,
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_skl_model_deploy(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        self.base_test_case(
            name="skl_model",
            model=regr,
            sample_input=iris_X,
            test_input=iris_X,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(res["output_feature_0"].values, regr.predict(iris_X)),
                ),
            },
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_skl_model_proba_deploy(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        model = ensemble.RandomForestClassifier(random_state=42)
        model.fit(iris_X[:10], iris_y[:10])
        self.base_test_case(
            name="skl_model_proba_deploy",
            model=model,
            sample_input=iris_X,
            test_input=iris_X[:10],
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(res["output_feature_0"].values, model.predict(iris_X[:10])),
                ),
                "predict_proba": (
                    {},
                    lambda res: np.testing.assert_allclose(res.values, model.predict_proba(iris_X[:10])),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_skl_multiple_output_model_proba_deploy(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        target2 = np.random.randint(0, 6, size=iris_y.shape)
        dual_target = np.vstack([iris_y, target2]).T
        model = multioutput.MultiOutputClassifier(ensemble.RandomForestClassifier(random_state=42))
        model.fit(iris_X[:10], dual_target[:10])
        self.base_test_case(
            name="skl_multiple_output_model_proba",
            model=model,
            sample_input=iris_X,
            test_input=iris_X[-10:],
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(res.values, model.predict(iris_X[-10:])),
                ),
                "predict_proba": (
                    {},
                    lambda res: np.testing.assert_allclose(res.values, np.hstack(model.predict_proba(iris_X[-10:]))),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_xgb(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)
        self.base_test_case(
            name="xgb_model",
            model=regressor,
            sample_input=cal_X_test,
            test_input=cal_X_test,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(regressor.predict(cal_X_test), axis=1)
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_xgb_sp(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_data_sp_df = self._session.create_dataframe(cal_data.frame)
        cal_data_sp_df_train, cal_data_sp_df_test = tuple(cal_data_sp_df.random_split([0.25, 0.75], seed=2568))
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        cal_data_pd_df_train = cal_data_sp_df_train.to_pandas()
        regressor.fit(cal_data_pd_df_train.drop(columns=["target"]), cal_data_pd_df_train["target"])
        cal_data_sp_df_test_X = cal_data_sp_df_test.drop('"target"')
        self.base_test_case(
            name="xgb_model_sp",
            model=regressor,
            sample_input=cal_data_sp_df_train.drop('"target"'),
            test_input=cal_data_sp_df_test_X,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res.to_pandas().values,
                        np.expand_dims(regressor.predict(cal_data_sp_df_test_X.to_pandas()), axis=1),
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @pytest.mark.pip_incompatible
    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_snowml_model_deploy_snowml_sklearn(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        self.base_test_case(
            name="snowml_model_sklearn",
            model=regr,
            sample_input=None,
            test_input=test_features,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @pytest.mark.pip_incompatible
    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_snowml_model_deploy_xgboost(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self.base_test_case(
            name="snowml_model_xgb",
            model=regr,
            sample_input=None,
            test_input=test_features,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )

    @pytest.mark.pip_incompatible
    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True},
        {"model_in_stage": False, "permanent_deploy": False},
    )
    def test_snowml_model_deploy_lightgbm(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self.base_test_case(
            name="snowml_model_lightgbm",
            model=regr,
            sample_input=None,
            test_input=test_features,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
        )


if __name__ == "__main__":
    absltest.main()
