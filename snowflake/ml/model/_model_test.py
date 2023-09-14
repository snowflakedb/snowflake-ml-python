import importlib
import os
import sys
import tempfile
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import linear_model

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import _model as model_api, custom_model, model_signature
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.test_utils import exception_utils, mock_session
from snowflake.snowpark import FileOperation, Session


class DemoModelWithManyArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(os.path.join(context.path("bias"), "bias1"), encoding="utf-8") as f:
            v1 = int(f.read())
        with open(os.path.join(context.path("bias"), "bias2"), encoding="utf-8") as f:
            v2 = int(f.read())
        self.bias = v1 + v2

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


PY_SRC = """\
def get_name():
    return __name__
def get_file():
    return __file__
"""


class ModelLoadHygieneTest(absltest.TestCase):
    def test_model_load_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as src_path:
                fake_mod_dirpath = os.path.join(src_path, "fake", "fake_module")
                os.makedirs(fake_mod_dirpath)

                py_file_path = os.path.join(fake_mod_dirpath, "p.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                sys.path.insert(0, src_path)

                from fake.fake_module import p

                self.assertEqual(p.__file__, py_file_path)

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                model_api._save(
                    name="model1",
                    local_dir_path=os.path.join(workspace, "model1"),
                    model=lm,
                    sample_input=d,
                    metadata={"author": "halu", "version": "1"},
                    code_paths=[os.path.join(src_path, "fake")],
                )

                _ = model_api._load(local_dir_path=os.path.join(workspace, "model1"))
                from fake.fake_module import p

                self.assertEqual(p.__file__, os.path.join(workspace, "model1", "code", "fake", "fake_module", "p.py"))

                importlib.reload(p)
                self.assertEqual(p.__file__, py_file_path)
                sys.path.remove(src_path)

    def test_model_save_validation(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as src_path:
                fake_mod_dirpath = os.path.join(src_path, "snowflake", "fake_module")
                os.makedirs(fake_mod_dirpath)

                py_file_path = os.path.join(fake_mod_dirpath, "p.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                with self.assertRaises(ValueError):
                    model_api._save(
                        name="model1",
                        local_dir_path=os.path.join(workspace, "model1"),
                        model=lm,
                        sample_input=d,
                        metadata={"author": "halu", "version": "1"},
                        code_paths=[os.path.join(src_path, "snowflake")],
                    )

            with tempfile.TemporaryDirectory() as src_path:
                py_file_path = os.path.join(src_path, "snowflake.py")
                with open(py_file_path, "w", encoding="utf-8") as f:
                    f.write(PY_SRC)
                    f.flush()

                lm = DemoModel(context=custom_model.ModelContext(models={}, artifacts={}))
                arr = np.array([[1, 2, 3], [4, 2, 5]])
                d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
                with self.assertRaises(ValueError):
                    model_api._save(
                        name="model1",
                        local_dir_path=os.path.join(workspace, "model1"),
                        model=lm,
                        sample_input=d,
                        metadata={"author": "halu", "version": "1"},
                        code_paths=[py_file_path],
                    )


class ModelInterfaceTest(absltest.TestCase):
    def test_save_interface(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        stage_path = '@"db"."schema"."stage"/model.zip'

        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Signatures and sample_input both cannot be specified at the same time.",
        ):
            model_api.save_model(  # type:ignore[call-overload]
                name="model1",
                session=c_session,
                model_stage_file_path=stage_path,
                model=linear_model.LinearRegression(),
                sample_input=d,
                signatures={"predict": model_signature.ModelSignature(inputs=[], outputs=[])},
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Signatures and sample_input both cannot be None at the same time for this kind of model.",
        ):
            model_api.save_model(
                name="model1",
                session=c_session,
                model_stage_file_path=stage_path,
                model=linear_model.LinearRegression(),
            )

        with mock.patch.object(model_api, "_save", return_value=None) as mock_save:
            with mock.patch.object(FileOperation, "put_stream", return_value=None) as mock_put_stream:
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=[""]
                ):
                    model_api.save_model(
                        name="model1",
                        session=c_session,
                        model_stage_file_path=stage_path,
                        model=LinearRegression(),
                    )
            mock_save.assert_called_once()

        with mock.patch.object(model_api, "_save", return_value=None) as mock_save:
            with mock.patch.object(FileOperation, "put_stream", return_value=None) as mock_put_stream:
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=[""]
                ):
                    model_api.save_model(
                        name="model1",
                        session=c_session,
                        model_stage_file_path=stage_path,
                        model=LinearRegression(),
                    )

            mock_save.assert_called_once()

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Provided model path in the stage [^\\s]* must be a path to a zip file.",
        ):
            model_api.save_model(
                name="model1",
                model=linear_model.LinearRegression(),
                session=c_session,
                model_stage_file_path='@"db"."schema"."stage"/model',
                sample_input=d,
            )

        with mock.patch.object(model_api, "_save", return_value=None):
            with mock.patch.object(FileOperation, "put_stream", return_value=None):
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=None
                ):
                    with self.assertLogs(level="INFO") as cm:
                        model_api.save_model(
                            name="model1",
                            model=linear_model.LinearRegression(),
                            session=c_session,
                            model_stage_file_path=stage_path,
                            sample_input=d,
                        )
                        self.assertListEqual(
                            cm.output,
                            [
                                (
                                    f"INFO:absl:Local snowflake-ml-python library has version {snowml_env.VERSION},"
                                    " which is not available in the Snowflake server, embedding local ML "
                                    "library automatically."
                                )
                            ],
                        )

        with mock.patch.object(model_api, "_save", return_value=None):
            with mock.patch.object(FileOperation, "put_stream", return_value=None) as mock_put_stream:
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=[""]
                ):
                    model_api.save_model(
                        name="model1",
                        model=linear_model.LinearRegression(),
                        session=c_session,
                        model_stage_file_path=stage_path,
                        sample_input=d,
                    )
            mock_put_stream.assert_called_once_with(mock.ANY, stage_path, auto_compress=False, overwrite=False)

        with mock.patch.object(model_api, "_save", return_value=None):
            with mock.patch.object(FileOperation, "put_stream", return_value=None) as mock_put_stream:
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=[""]
                ):
                    model_api.save_model(
                        name="model1",
                        model=linear_model.LinearRegression(),
                        session=c_session,
                        model_stage_file_path=stage_path,
                        sample_input=d,
                        options={"allow_overwritten_stage_file": True},
                    )
            mock_put_stream.assert_called_once_with(mock.ANY, stage_path, auto_compress=False, overwrite=True)

    def test_load_interface(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Provided model path in the stage [^\\s]* must be a path to a zip file.",
        ):
            model_api.load_model(session=c_session, model_stage_file_path='@"db"."schema"."stage"/model')


class ModelTest(absltest.TestCase):
    def test_bad_save_model(self) -> None:
        tmpdir = self.create_tempdir()
        os.mkdir(os.path.join(tmpdir.full_path, "bias"))
        with open(os.path.join(tmpdir.full_path, "bias", "bias1"), "w", encoding="utf-8") as f:
            f.write("25")
        with open(os.path.join(tmpdir.full_path, "bias", "bias2"), "w", encoding="utf-8") as f:
            f.write("68")
        lm = DemoModelWithManyArtifacts(
            custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir.full_path, "bias")})
        )
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        s = {"predict": model_signature.infer_signature(d, lm.predict(d))}

        with self.assertRaises(ValueError):
            model_api._save(
                name="model1",
                local_dir_path=os.path.join(tmpdir.full_path, "model1"),
                model=lm,
                signatures={**s, "another_predict": s["predict"]},
                metadata={"author": "halu", "version": "1"},
            )

        model_api._save(
            name="model1",
            local_dir_path=os.path.join(tmpdir.full_path, "model1"),
            model=lm,
            signatures=s,
            metadata={"author": "halu", "version": "1"},
            python_version="3.5.2",
        )

        _ = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"), meta_only=True)

        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
            m, meta = model_api._load(local_dir_path=os.path.join(tmpdir, "model1"))


if __name__ == "__main__":
    absltest.main()
