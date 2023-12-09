from typing import cast

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Session


class ModelImplTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)
        self.m_model = model_impl.Model.create(
            self.c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
        )

    def test_property(self) -> None:
        self.assertEqual(self.m_model.name, "MODEL")
        self.assertEqual(self.m_model.fully_qualified_name, 'TEMP."test".MODEL')

    def test_version(self) -> None:
        mv = model_version_impl.ModelVersion(
            self.m_model._model_ops,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
        )
        self.assertDictEqual(self.m_model.version("V1").__dict__, mv.__dict__)


if __name__ == "__main__":
    absltest.main()
