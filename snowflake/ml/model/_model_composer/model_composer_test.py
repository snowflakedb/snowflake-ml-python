import os
import pathlib
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import linear_model

from snowflake.ml._internal import env_utils, file_utils
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Session


class ModelInterfaceTest(absltest.TestCase):
    def test_save_interface(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        stage_path = '@"db"."schema"."stage"'
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

        mock_pk = mock.MagicMock()
        mock_pk.meta = mock.MagicMock()
        mock_pk.meta.signatures = mock.MagicMock()
        m = model_composer.ModelComposer(session=c_session, stage_path=stage_path)

        with open(os.path.join(m._packager_workspace_path, "model.yaml"), "w", encoding="utf-8") as f:
            f.write("")
        m.packager = mock_pk
        with mock.patch.object(m.packager, "save") as mock_save:
            with mock.patch.object(m.manifest, "save") as mock_manifest_save:
                with mock.patch.object(
                    file_utils, "upload_directory_to_stage", return_value=None
                ) as mock_upload_directory_to_stage:
                    with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                        m.save(
                            name="model1",
                            model=LinearRegression(),
                        )
                mock_save.assert_called_once()
                mock_manifest_save.assert_called_once()
                mock_upload_directory_to_stage.assert_called_once_with(
                    c_session, local_path=mock.ANY, stage_path=pathlib.PurePosixPath(stage_path), statement_params=None
                )

        m = model_composer.ModelComposer(session=c_session, stage_path=stage_path)
        m.packager = mock_pk
        with open(os.path.join(m._packager_workspace_path, "model.yaml"), "w", encoding="utf-8") as f:
            f.write("")
        with mock.patch.object(m.packager, "save") as mock_save:
            with mock.patch.object(m.manifest, "save") as mock_manifest_save:
                with mock.patch.object(
                    file_utils, "upload_directory_to_stage", return_value=None
                ) as mock_upload_directory_to_stage:
                    with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                        m.save(
                            name="model1",
                            model=linear_model.LinearRegression(),
                            sample_input=d,
                        )
                mock_save.assert_called_once()
                mock_manifest_save.assert_called_once()
                mock_upload_directory_to_stage.assert_called_once_with(
                    c_session, local_path=mock.ANY, stage_path=pathlib.PurePosixPath(stage_path), statement_params=None
                )


if __name__ == "__main__":
    absltest.main()
