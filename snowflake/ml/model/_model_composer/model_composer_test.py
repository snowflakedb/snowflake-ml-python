import os
import pathlib
import tempfile
from typing import Any, cast
from unittest import mock
from urllib import parse

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import linear_model

from snowflake.ml._internal import env_utils, file_utils
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._packager import model_packager
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Session


class ModelInterfaceTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"use_save_location": False}},
        {"params": {"use_save_location": True}},
    )
    def test_save_interface(self, params: dict[str, Any]) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        stage_path = '@"db"."schema"."stage"'
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

        mock_pk = mock.MagicMock()
        mock_pk.meta = mock.MagicMock()
        mock_pk.meta.signatures = mock.MagicMock()
        if params["use_save_location"]:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = None
        m = model_composer.ModelComposer(
            session=c_session,
            stage_path=stage_path,
            save_location=temp_dir,
        )

        with open(os.path.join(m.packager_workspace_path, "model.yaml"), "w", encoding="utf-8") as f:
            f.write("")
        m.packager = mock_pk
        with mock.patch.object(m.packager, "save", return_value=mock_pk.meta) as mock_save:
            with mock.patch.object(m.manifest, "save") as mock_manifest_save:
                with mock.patch.object(
                    file_utils, "upload_directory_to_stage", return_value=None
                ) as mock_upload_directory_to_stage:
                    with mock.patch.object(
                        env_utils,
                        "get_matched_package_versions_in_information_schema",
                        return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
                    ):
                        m.save(
                            name="model1",
                            model=LinearRegression(),
                        )
                mock_save.assert_called_once()
                mock_manifest_save.assert_called_once()
                mock_upload_directory_to_stage.assert_called_once_with(
                    c_session,
                    local_path=mock.ANY,
                    stage_path=pathlib.PurePosixPath(stage_path),
                    statement_params=None,
                )

        m = model_composer.ModelComposer(session=c_session, stage_path=stage_path)
        m.packager = mock_pk
        with open(os.path.join(m.packager_workspace_path, "model.yaml"), "w", encoding="utf-8") as f:
            f.write("")
        with mock.patch.object(m.packager, "save", return_value=mock_pk.meta) as mock_save:
            with mock.patch.object(m.manifest, "save") as mock_manifest_save:
                with mock.patch.object(
                    file_utils, "upload_directory_to_stage", return_value=None
                ) as mock_upload_directory_to_stage:
                    with mock.patch.object(
                        env_utils,
                        "get_matched_package_versions_in_information_schema",
                        return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
                    ):
                        m.save(
                            name="model1",
                            model=linear_model.LinearRegression(),
                            sample_input_data=d,
                            task=model_types.Task.TABULAR_REGRESSION,
                        )

                    mock_upload_directory_to_stage.assert_called_once_with(
                        c_session,
                        local_path=mock.ANY,
                        stage_path=pathlib.PurePosixPath(stage_path),
                        statement_params=None,
                    )
                mock_save.assert_called_once()
                mock_manifest_save.assert_called_once()

        # test for live and commit model
        # the snow url is passed to ModelComposer
        snow_stage_path = "snow://model/a_model_name/versions/a_version_name"
        m = model_composer.ModelComposer(session=c_session, stage_path=snow_stage_path)
        m.packager = mock_pk
        with open(os.path.join(m.packager_workspace_path, "model.yaml"), "w", encoding="utf-8") as f:
            f.write("")
        with mock.patch.object(m.packager, "save", return_value=mock_pk.meta) as mock_save:
            with mock.patch.object(m.manifest, "save") as mock_manifest_save:
                with mock.patch.object(
                    file_utils, "upload_directory_to_stage", return_value=None
                ) as mock_upload_directory_to_stage:
                    with mock.patch.object(
                        env_utils,
                        "get_matched_package_versions_in_information_schema",
                        return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
                    ):
                        m.save(
                            name="model1",
                            model=linear_model.LinearRegression(),
                            sample_input_data=d,
                            task=model_types.Task.TABULAR_REGRESSION,
                        )

                    mock_upload_directory_to_stage.assert_called_once_with(
                        c_session,
                        local_path=mock.ANY,
                        stage_path=parse.urlparse(snow_stage_path),
                        statement_params=None,
                    )
                mock_save.assert_called_once()
                mock_manifest_save.assert_called_once()

    @parameterized.parameters(  # type: ignore[misc]
        {"disable_explainability": True, "target_platforms": [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]},
        {
            "disable_explainability": False,
            "target_platforms": [
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                model_types.TargetPlatform.WAREHOUSE,
            ],
        },
        {"disable_explainability": False, "target_platforms": []},
        {
            "disable_explainability": True,
            "conda_dependencies": ["python-package1==1.0.0", "conda-forge::python-package2==1.1.0"],
        },
        {
            "disable_explainability": False,
            "conda_dependencies": [
                "python-package1==1.0.0",
                "https://repo.anaconda.com/pkgs/snowflake::python-package2",
            ],
        },
        {"disable_explainability": True, "pip_requirements": ["python-package==1.0.0"]},
        {"disable_explainability": False, "pip_requirements": None},
    )
    def test_save_enable_explainability(self, disable_explainability: bool, **kwargs: Any) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        stage_path = '@"db"."schema"."stage"'

        mock_pk = mock.MagicMock()
        mock_pk.meta = mock.MagicMock()
        mock_pk.meta.signatures = mock.MagicMock()
        m = model_composer.ModelComposer(session=c_session, stage_path=stage_path)

        with open(os.path.join(m._packager_workspace_path, "model.yaml"), "w", encoding="utf-8") as f:
            f.write("")
        m.packager = mock_pk

        with mock.patch.object(m.packager, "save", return_value=mock_pk.meta) as mock_save:
            with mock.patch.object(m.manifest, "save"):
                with mock.patch.object(file_utils, "upload_directory_to_stage", return_value=None):
                    with mock.patch.object(
                        env_utils,
                        "get_matched_package_versions_in_information_schema",
                        return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
                    ):
                        m.save(
                            name="model1",
                            model=linear_model.LinearRegression(),
                            **kwargs,
                        )
                mock_save.assert_called_once()
                _, called_kwargs = mock_save.call_args
                self.assertIn("options", called_kwargs)
                if (
                    disable_explainability
                ):  # set to false if the model is not runnable in WH or the target platforms is only SPCS
                    self.assertEqual(
                        called_kwargs["options"], called_kwargs["options"] | {"enable_explainability": False}
                    )
                else:
                    # else options should be empty since user did not pass anything
                    # and explainability does not need to be explicitly disabled
                    self.assertNotIn("enable_explainability", called_kwargs["options"])

        if disable_explainability:
            with self.assertRaisesRegex(
                ValueError,
                "`enable_explainability` cannot be set to True when the model is not runnable in WH "
                "or the target platforms include SPCS.",
            ):
                with mock.patch.object(m.packager, "save", return_value=mock_pk.meta):
                    with mock.patch.object(m.manifest, "save"):
                        with mock.patch.object(
                            file_utils, "upload_directory_to_stage", return_value=None
                        ), mock.patch.object(file_utils, "copytree", return_value="/model"):
                            with mock.patch.object(
                                env_utils,
                                "get_matched_package_versions_in_information_schema",
                                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
                            ):
                                m.save(
                                    name="model1",
                                    model=linear_model.LinearRegression(),
                                    options={"enable_explainability": True},
                                    **kwargs,
                                )

    def test_load(self) -> None:
        m_options = model_types.PyTorchLoadOptions(use_gpu=False)
        with mock.patch.object(model_packager.ModelPackager, "load") as mock_load:
            model_composer.ModelComposer.load(pathlib.Path("workspace"), meta_only=True, options=m_options)
            mock_load.assert_called_once_with(meta_only=True, options=m_options)


if __name__ == "__main__":
    absltest.main()
