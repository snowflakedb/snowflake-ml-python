import datetime
from typing import List, cast

from absl.testing import absltest

from snowflake import connector, snowpark
from snowflake.ml._internal.utils import identifier, table_manager
from snowflake.ml.registry import _artifact_manager, artifact
from snowflake.ml.test_utils import mock_data_frame, mock_session

_DATABASE_NAME = identifier.get_inferred_name("_SYSTEM_MODEL_REGISTRY")
_SCHEMA_NAME = identifier.get_inferred_name("_SYSTEM_MODEL_REGISTRY_SCHEMA")
_TABLE_NAME = identifier.get_inferred_name("_SYSTEM_REGISTRY_ARTIFACTS")
_FULLY_QUALIFIED_TABLE_NAME = table_manager.get_fully_qualified_table_name(_DATABASE_NAME, _SCHEMA_NAME, _TABLE_NAME)


class ArtifactTest(absltest.TestCase):
    """Testing Artifact table related functions."""

    def setUp(self) -> None:
        """Creates Snowpark environments for testing."""
        self._session = mock_session.MockSession(conn=None, test_case=self)

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

    def _get_show_tables_success(
        self, name: str, database_name: str = _DATABASE_NAME, schema_name: str = _SCHEMA_NAME
    ) -> List[snowpark.Row]:
        """Helper method that returns a DataFrame that looks like the response of from a successful listing of
        tables."""
        return [
            snowpark.Row(
                created_on=datetime.datetime(2022, 11, 4, 17, 1, 30, 153000),
                name=name,
                database_name=database_name,
                schema_name=schema_name,
                kind="TABLE",
                comment="",
                cluster_by="",
                rows=0,
                bytes=0,
                owner="OWNER_ROLE",
                retention_time=1,
                change_tracking="OFF",
                is_external="N",
                enable_schema_evolution="N",
            )
        ]

    def _get_select_artifact(self) -> List[snowpark.Row]:
        """Helper method that returns a DataFrame that looks like the response of from a successful listing of
        tables."""
        return [
            snowpark.Row(
                id="FAKE_ID",
                type=artifact.ArtifactType.TESTTYPE,
                creation_time=datetime.datetime(2022, 11, 4, 17, 1, 30, 153000),
                creation_role="OWNER_ROLE",
                artifact_spec={},
            )
        ]

    def test_if_artifact_exists(self) -> None:
        for mock_df_collect, expected_res in [
            (self._get_select_artifact(), True),
            ([], False),
        ]:
            with self.subTest():
                artifact_name = "FAKE_ID"
                artifact_version = "FAKE_VERSION"
                expected_df = mock_data_frame.MockDataFrame()
                expected_df.add_operation("filter")
                expected_df.add_operation("filter")
                expected_df.add_collect_result(cast(List[snowpark.Row], mock_df_collect))
                self._session.add_mock_sql(query=f"SELECT * FROM {_FULLY_QUALIFIED_TABLE_NAME}", result=expected_df)
                self.assertEqual(
                    _artifact_manager.ArtifactManager(
                        session=cast(snowpark.Session, self._session),
                        database_name=_DATABASE_NAME,
                        schema_name=_SCHEMA_NAME,
                    ).exists(
                        artifact_name,
                        artifact_version,
                    ),
                    expected_res,
                )

    def test_add_artifact(self) -> None:
        artifact_id = "FAKE_ID"
        artifact_name = "FAKE_NAME"
        artifact_version = "FAKE_VERSION"
        art_obj = artifact.Artifact(type=artifact.ArtifactType.TESTTYPE, spec='{"description": "mock description"}')

        # Mock the insertion call
        self._session.add_operation("get_current_role", result="current_role")
        insert_query = (
            f"INSERT INTO {_FULLY_QUALIFIED_TABLE_NAME}"
            " ( ARTIFACT_SPEC,CREATION_ROLE,CREATION_TIME,ID,NAME,TYPE,VERSION )"
            " SELECT"
            " '{\"description\": \"mock description\"}','current_role',CURRENT_TIMESTAMP(),"
            "'FAKE_ID','FAKE_NAME', 'TESTTYPE', 'FAKE_VERSION' "
        )
        self._session.add_mock_sql(
            query=insert_query,
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )
        _artifact_manager.ArtifactManager(
            session=cast(snowpark.Session, self._session),
            database_name=_DATABASE_NAME,
            schema_name=_SCHEMA_NAME,
        ).add(
            artifact=art_obj,
            artifact_id=artifact_id,
            artifact_name=artifact_name,
            artifact_version=artifact_version,
        )

    def test_delete_artifact(self) -> None:
        for error_if_not_exist in [True, False]:
            with self.subTest():
                if error_if_not_exist:
                    artifact_name = "FAKE_NAME"
                    artifact_version = "FAKE_VERSION"
                    expected_df = mock_data_frame.MockDataFrame()
                    expected_df.add_operation("filter")
                    expected_df.add_operation("filter")
                    expected_df.add_collect_result([])
                    self._session.add_mock_sql(query=f"SELECT * FROM {_FULLY_QUALIFIED_TABLE_NAME}", result=expected_df)
                    with self.assertRaises(connector.DataError):
                        _artifact_manager.ArtifactManager(
                            session=cast(snowpark.Session, self._session),
                            database_name=_DATABASE_NAME,
                            schema_name=_SCHEMA_NAME,
                        ).delete(
                            artifact_name,
                            artifact_version,
                            True,
                        )
                else:
                    expected_df = mock_data_frame.MockDataFrame()
                    expected_df.add_operation("filter")
                    expected_df.add_operation("filter")
                    expected_df.add_collect_result([])
                    self._session.add_mock_sql(query=f"SELECT * FROM {_FULLY_QUALIFIED_TABLE_NAME}", result=expected_df)
                    _artifact_manager.ArtifactManager(
                        session=cast(snowpark.Session, self._session),
                        database_name=_DATABASE_NAME,
                        schema_name=_SCHEMA_NAME,
                    ).delete(
                        artifact_name,
                        artifact_version,
                    )


if __name__ == "__main__":
    absltest.main()
