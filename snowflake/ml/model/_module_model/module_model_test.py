from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import linear_model

from snowflake.ml._internal import env_utils
from snowflake.ml.model._module_model import module_model
from snowflake.ml.modeling.linear_model import (  # type:ignore[attr-defined]
    LinearRegression,
)
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import FileOperation, Session


class ModuleInterfaceTest(absltest.TestCase):
    def test_save_interface(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        stage_path = '@"db"."schema"."stage"'
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])

        mock_pk = mock.MagicMock()
        mock_pk.meta = mock.MagicMock()
        mock_pk.meta.signatures = mock.MagicMock()
        m = module_model.ModuleModel(session=c_session, stage_path=stage_path)
        with mock.patch.object(m.packager, "save") as mock_save:
            with mock.patch.object(FileOperation, "put", return_value=None) as mock_put_stream:
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=[""]
                ):
                    m.save(
                        name="model1",
                        model=LinearRegression(),
                    )
            mock_save.assert_called_once()

        m = module_model.ModuleModel(session=c_session, stage_path=stage_path)
        with mock.patch.object(m.packager, "save") as mock_save:
            with mock.patch.object(FileOperation, "put", return_value=None) as mock_put_stream:
                with mock.patch.object(
                    env_utils, "validate_requirements_in_snowflake_conda_channel", return_value=[""]
                ):
                    m.save(
                        name="model1",
                        model=linear_model.LinearRegression(),
                        sample_input=d,
                    )
            mock_put_stream.assert_called_once_with(mock.ANY, stage_path, auto_compress=False, overwrite=False)


if __name__ == "__main__":
    absltest.main()
