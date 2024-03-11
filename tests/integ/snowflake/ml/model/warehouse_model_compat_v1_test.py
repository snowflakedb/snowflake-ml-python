import posixpath
import unittest
import uuid
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from absl.testing import absltest
from packaging import version
from sklearn import datasets

from snowflake.ml._internal import env
from snowflake.ml.model import _api as model_api, deploy_platforms
from snowflake.snowpark import session
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


@unittest.skipIf(
    version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
    "Skip compat test for Python higher than 3.11 since we previously does not support it.",
)
class TestWarehouseCustomModelCompat(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        self.run_id = uuid.uuid4().hex
        self.session_stage = self.session.get_session_stage()
        self.model_stage_path = posixpath.join(self.session_stage, self.run_id)
        self.model_stage_file_path = posixpath.join(self.session_stage, self.run_id, f"{self.run_id}.zip")

    def _log_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import pandas as pd

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
                custom_model,
            )

            class DemoModel(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                    return pd.DataFrame({"output": input["c1"]})

            lm = DemoModel(custom_model.ModelContext())
            pd_df = pd.DataFrame([[1, 2, 3], [4, 2, 5]], columns=["c1", "c2", "c3"])

            model_api.save_model(
                name=run_id,
                model=lm,
                sample_input=pd_df,
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_model_factory, version_range=">=1.0.8,<=1.0.11"  # type: ignore[misc, arg-type]
    )
    def test_deploy_custom_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        model_api.predict(
            self.session, deployment=deploy_info, X=pd.DataFrame([[1, 2, 3], [4, 2, 5]], columns=["c1", "c2", "c3"])
        )

    def _log_model_multiple_components_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import os
            import tempfile

            import pandas as pd

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
                custom_model,
            )

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
                    with open(context.path("bias"), encoding="utf-8") as f:
                        v = int(f.read())
                    self.bias = v

                @custom_model.inference_api
                def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                    return pd.DataFrame({"output": (input["c1"] + self.bias) > 12})

            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                    f.write("10")
                lm_1 = DemoModelWithArtifacts(
                    custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
                )
                lm_2 = DemoModel(custom_model.ModelContext())
                model_context = custom_model.ModelContext(
                    models={
                        "m1": lm_1,
                        "m2": lm_2,
                    }
                )
                acm = AsyncComposeModel(model_context)
                pd_df = pd.DataFrame([[1, 2, 3], [4, 2, 5]], columns=["c1", "c2", "c3"])

                model_api.save_model(
                    name=run_id,
                    model=acm,
                    sample_input=pd_df,
                    metadata={"author": "halu", "version": "1"},
                    session=session,
                    model_stage_file_path=model_stage_file_path,
                )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_model_multiple_components_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.8,<=1.0.11",
    )
    def test_deploy_custom_model_multiple_components_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        model_api.predict(
            self.session, deployment=deploy_info, X=pd.DataFrame([[1, 2, 3], [4, 2, 5]], columns=["c1", "c2", "c3"])
        )

    def _log_sklearn_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            from sklearn import datasets, linear_model

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            iris_X, iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
            # LogisticRegression is for classfication task, such as iris
            regr = linear_model.LogisticRegression()
            regr.fit(iris_X, iris_y)

            model_api.save_model(
                name=run_id,
                model=regr,
                sample_input=iris_X,
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_sklearn_model_factory, version_range=">=1.0.6,<=1.0.11"  # type: ignore[misc, arg-type]
    )
    def test_deploy_sklearn_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        iris_X, _ = datasets.load_iris(return_X_y=True, as_frame=True)
        model_api.predict(self.session, deployment=deploy_info, X=iris_X)

    def _log_xgboost_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import xgboost
            from sklearn import datasets, model_selection

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            cal_data = datasets.load_breast_cancer(as_frame=True)
            cal_X = cal_data.data
            cal_y = cal_data.target
            cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
            regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
            regressor.fit(cal_X_train, cal_y_train)

            model_api.save_model(
                name=run_id,
                model=regressor,
                sample_input=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_xgboost_model_factory, version_range=">=1.0.6,<=1.0.11"  # type: ignore[misc, arg-type]
    )
    def test_deploy_xgboost_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        model_api.predict(self.session, deployment=deploy_info, X=cal_X)

    def _log_xgboost_booster_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import xgboost
            from sklearn import datasets, model_selection

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            cal_data = datasets.load_breast_cancer(as_frame=True)
            cal_X = cal_data.data
            cal_y = cal_data.target
            cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
            params = dict(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, objective="binary:logistic")
            regressor = xgboost.train(params, xgboost.DMatrix(data=cal_X_train, label=cal_y_train))

            model_api.save_model(
                name=run_id,
                model=regressor,
                sample_input=cal_X_test,
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_xgboost_booster_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.6,<=1.0.11",
    )
    def test_deploy_xgboost_booster_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        model_api.predict(self.session, deployment=deploy_info, X=cal_X)

    def _log_snowml_sklearn_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            from sklearn import datasets

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )
            from snowflake.ml.modeling.linear_model import (  # type: ignore[attr-defined]
                LogisticRegression,
            )

            iris_X = datasets.load_iris(as_frame=True).frame
            iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

            INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
            LABEL_COLUMNS = "TARGET"
            OUTPUT_COLUMNS = "PREDICTED_TARGET"
            regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
            test_features = iris_X
            regr.fit(test_features)

            model_api.save_model(
                name=run_id,
                model=regr,
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_snowml_sklearn_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.8,<=1.0.11",
    )
    def test_deploy_snowml_sklearn_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        model_api.predict(self.session, deployment=deploy_info, X=iris_X)

    def _log_snowml_xgboost_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            from sklearn import datasets

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )
            from snowflake.ml.modeling.xgboost import (  # type: ignore[attr-defined]
                XGBRegressor,
            )

            iris_X = datasets.load_iris(as_frame=True).frame
            iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

            INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
            LABEL_COLUMNS = "TARGET"
            OUTPUT_COLUMNS = "PREDICTED_TARGET"
            regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
            test_features = iris_X
            regr.fit(test_features)

            model_api.save_model(
                name=run_id,
                model=regr,
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_snowml_xgboost_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.8,<=1.0.11",
    )
    def test_deploy_snowml_xgboost_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        model_api.predict(self.session, deployment=deploy_info, X=iris_X)

    def _log_pytorch_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import numpy as np
            import torch

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            class TorchModel(torch.nn.Module):
                def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
                    super().__init__()
                    self.model = torch.nn.Sequential(
                        torch.nn.Linear(n_input, n_hidden, dtype=dtype),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, n_out, dtype=dtype),
                        torch.nn.Sigmoid(),
                    )

                def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                    return self.model(tensor)  # type: ignore[no-any-return]

            n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
            x = np.random.rand(batch_size, n_input)
            dtype = torch.float32
            data_x = torch.from_numpy(x).to(dtype=dtype)
            data_y = (torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)

            model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
            loss_function = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            for _epoch in range(100):
                pred_y = model.forward(data_x)
                loss = loss_function(pred_y, data_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model_api.save_model(
                name=run_id,
                model=model,
                sample_input=[data_x],
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_pytorch_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.6,<=1.0.11",
        additional_packages=["pytorch"],
    )
    def test_deploy_pytorch_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "forward"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="forward",
            options={},
        )
        assert deploy_info

        n_input, batch_size = 10, 100
        x = np.random.rand(batch_size, n_input)
        dtype = torch.float32
        data_x = torch.from_numpy(x).to(dtype=dtype)

        model_api.predict(self.session, deployment=deploy_info, X=[data_x])

    def _log_torchscript_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import numpy as np
            import torch

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            class TorchModel(torch.nn.Module):
                def __init__(self, n_input: int, n_hidden: int, n_out: int, dtype: torch.dtype = torch.float32) -> None:
                    super().__init__()
                    self.model = torch.nn.Sequential(
                        torch.nn.Linear(n_input, n_hidden, dtype=dtype),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, n_out, dtype=dtype),
                        torch.nn.Sigmoid(),
                    )

                def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                    return self.model(tensor)  # type: ignore[no-any-return]

            n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
            x = np.random.rand(batch_size, n_input)
            dtype = torch.float32
            data_x = torch.from_numpy(x).to(dtype=dtype)
            data_y = (torch.rand(size=(batch_size, 1)) < 0.5).to(dtype=dtype)

            model = TorchModel(n_input, n_hidden, n_out, dtype=dtype)
            loss_function = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            for _epoch in range(100):
                pred_y = model.forward(data_x)
                loss = loss_function(pred_y, data_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model_script = torch.jit.script(model)  # type:ignore[attr-defined]

            model_api.save_model(
                name=run_id,
                model=model_script,
                sample_input=[data_x],
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_torchscript_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.6,<=1.0.11",
        additional_packages=["pytorch"],
    )
    def test_deploy_torchscript_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "forward"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="forward",
            options={},
        )
        assert deploy_info

        n_input, batch_size = 10, 100
        x = np.random.rand(batch_size, n_input)
        dtype = torch.float32
        data_x = torch.from_numpy(x).to(dtype=dtype)

        model_api.predict(self.session, deployment=deploy_info, X=[data_x])

    def _log_tensorflow_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            from typing import Optional

            import tensorflow as tf

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            class SimpleModule(tf.Module):
                def __init__(self, name: Optional[str] = None) -> None:
                    super().__init__(name=name)
                    self.a_variable = tf.Variable(5.0, name="train_me")
                    self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

                @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])  # type: ignore[misc]
                def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
                    return self.a_variable * tensor + self.non_trainable_variable

            model = SimpleModule(name="simple")
            data_x = tf.constant([[5.0], [10.0]])

            model_api.save_model(
                name=run_id,
                model=model,
                sample_input=[data_x],
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_tensorflow_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.6,<=1.0.11",
        additional_packages=["tensorflow"],
    )
    def test_deploy_tensorflow_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "__call__"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="__call__",
            options={},
        )
        assert deploy_info

        data_x = tf.constant([[5.0], [10.0]])

        model_api.predict(self.session, deployment=deploy_info, X=[data_x])

    def _log_keras_model_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str, str], None], Tuple[str, str]]:
        def log_model(session: session.Session, run_id: str, model_stage_file_path: str) -> None:
            import numpy as np
            import tensorflow as tf

            from snowflake.ml.model import (  # type: ignore[attr-defined]
                _model as model_api,
            )

            class KerasModel(tf.keras.Model):
                def __init__(self, n_hidden: int, n_out: int) -> None:
                    super().__init__()
                    self.fc_1 = tf.keras.layers.Dense(n_hidden, activation="relu")
                    self.fc_2 = tf.keras.layers.Dense(n_out, activation="sigmoid")

                def call(self, tensor: tf.Tensor) -> tf.Tensor:
                    input = tensor
                    x = self.fc_1(input)
                    x = self.fc_2(x)
                    return x

            dtype = tf.float32
            n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01
            x = np.random.rand(batch_size, n_input)
            data_x = tf.convert_to_tensor(x, dtype=dtype)
            raw_data_y = tf.random.uniform((batch_size, 1))
            raw_data_y = tf.where(raw_data_y > 0.5, tf.ones_like(raw_data_y), tf.zeros_like(raw_data_y))
            data_y = tf.cast(raw_data_y, dtype=dtype)

            model = KerasModel(n_hidden, n_out)
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.MeanSquaredError()
            )
            model.fit(data_x, data_y, batch_size=batch_size, epochs=100)

            model_api.save_model(
                name=run_id,
                model=model,
                sample_input=[data_x],
                metadata={"author": "halu", "version": "1"},
                session=session,
                model_stage_file_path=model_stage_file_path,
            )

        return log_model, (self.run_id, self.model_stage_file_path)

    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_log_keras_model_factory,  # type: ignore[misc, arg-type]
        version_range=">=1.0.6,<=1.0.11",
        additional_packages=["tensorflow"],
    )
    def test_deploy_keras_model_compat_v1(self) -> None:
        deploy_info = model_api.deploy(
            self.session,
            name=db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "predict"),
            platform=deploy_platforms.TargetPlatform.WAREHOUSE,
            stage_path=self.model_stage_path,
            target_method="predict",
            options={},
        )
        assert deploy_info

        dtype = tf.float32
        n_input, batch_size = 10, 100
        x = np.random.rand(batch_size, n_input)
        data_x = tf.convert_to_tensor(x, dtype=dtype)

        model_api.predict(self.session, deployment=deploy_info, X=[data_x])


if __name__ == "__main__":
    absltest.main()
