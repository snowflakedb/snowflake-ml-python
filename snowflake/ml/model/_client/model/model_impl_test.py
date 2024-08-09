from typing import cast
from unittest import mock

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.model._client.ops import model_ops, service_ops
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Row, Session


class ModelImplTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)
        self.m_model = model_impl.Model._ref(
            model_ops.ModelOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ),
            service_ops=service_ops.ServiceOperator(
                self.c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
        )

    def test_property(self) -> None:
        self.assertEqual(self.m_model.name, "MODEL")
        self.assertEqual(self.m_model.fully_qualified_name, 'TEMP."test".MODEL')

    def test_version(self) -> None:
        with mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]):
            m_mv = model_version_impl.ModelVersion._ref(
                self.m_model._model_ops,
                service_ops=self.m_model._service_ops,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
            )
            with mock.patch.object(
                self.m_model._model_ops, "validate_existence", return_value=True
            ) as mock_validate_existence, mock.patch.object(
                self.m_model._model_ops, "get_version_by_alias", return_value=None
            ) as mock_get_version_by_alias:
                mv = self.m_model.version("v1")
                self.assertEqual(mv, m_mv)
                mock_validate_existence.assert_called_once_with(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=mock.ANY,
                )
                mock_get_version_by_alias.assert_called_once_with(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    alias_name=sql_identifier.SqlIdentifier("V1"),
                    statement_params=mock.ANY,
                )

    def test_version_with_alias(self) -> None:
        with mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]):
            m_mv = model_version_impl.ModelVersion._ref(
                self.m_model._model_ops,
                service_ops=self.m_model._service_ops,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
            )

            with mock.patch.object(
                self.m_model._model_ops, "get_version_by_alias", return_value="V1"
            ) as mock_get_version_by_alias:
                mv = self.m_model.version("A1")
                self.assertEqual(mv, m_mv)
                mock_get_version_by_alias.assert_called_once_with(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    alias_name=sql_identifier.SqlIdentifier("A1"),
                    statement_params=mock.ANY,
                )

    def test_version_not_exist(self) -> None:
        with mock.patch.object(
            self.m_model._model_ops, "validate_existence", return_value=False
        ) as mock_validate_existence, mock.patch.object(
            self.m_model._model_ops, "get_version_by_alias", return_value=None
        ):
            with self.assertRaisesRegex(
                ValueError, 'Unable to find version or alias with name V1 in model TEMP."test"'
            ):
                self.m_model.version("v1")
            mock_validate_existence.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=mock.ANY,
            )

    def test_versions(self) -> None:
        with mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]):
            m_mv_1 = model_version_impl.ModelVersion._ref(
                self.m_model._model_ops,
                service_ops=self.m_model._service_ops,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
            )
            m_mv_2 = model_version_impl.ModelVersion._ref(
                self.m_model._model_ops,
                service_ops=self.m_model._service_ops,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            )
            with mock.patch.object(
                self.m_model._model_ops,
                "list_models_or_versions",
                return_value=[
                    sql_identifier.SqlIdentifier("V1"),
                    sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                ],
            ) as mock_list_models_or_versions:
                mv_list = self.m_model.versions()
                self.assertListEqual(mv_list, [m_mv_1, m_mv_2])
                mock_list_models_or_versions.assert_called_once_with(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    statement_params=mock.ANY,
                )

    def test_show_versions(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=True,
            ),
            Row(
                create_on="06/01",
                name="V1",
                comment="This is a comment",
                model_name="MODEL",
                is_default_version=False,
            ),
        ]
        with mock.patch.object(
            self.m_model._model_ops,
            "show_models_or_versions",
            return_value=m_list_res,
        ) as mock_show_models_or_versions:
            mv_info = self.m_model.show_versions()
            pd.testing.assert_frame_equal(mv_info, pd.DataFrame([row.as_dict() for row in m_list_res]))
            mock_show_models_or_versions.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_description_getter(self) -> None:
        with mock.patch.object(
            self.m_model._model_ops, "get_comment", return_value="this is a comment"
        ) as mock_get_comment:
            self.assertEqual("this is a comment", self.m_model.description)
            mock_get_comment.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_description_setter(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "set_comment") as mock_set_comment:
            self.m_model.description = "this is a comment"
            mock_set_comment.assert_called_once_with(
                comment="this is a comment",
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_comment_getter(self) -> None:
        with mock.patch.object(
            self.m_model._model_ops, "get_comment", return_value="this is a comment"
        ) as mock_get_comment:
            self.assertEqual("this is a comment", self.m_model.comment)
            mock_get_comment.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_comment_setter(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "set_comment") as mock_set_comment:
            self.m_model.comment = "this is a comment"
            mock_set_comment.assert_called_once_with(
                comment="this is a comment",
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_default_getter(self) -> None:
        with mock.patch.object(
            self.m_model._model_ops,
            "get_default_version",
            return_value=sql_identifier.SqlIdentifier("V1", case_sensitive=True),
        ) as mock_get_default_version, mock.patch.object(
            self.m_model._model_ops, "validate_existence", return_value=True
        ), mock.patch.object(
            model_version_impl.ModelVersion, "_get_functions", return_value=[]
        ), mock.patch.object(
            self.m_model._model_ops, "get_version_by_alias", return_value=None
        ):
            self.assertEqual("V1", self.m_model.default.version_name)
            mock_get_default_version.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_system_aliases(self) -> None:
        m_list_res = [
            Row(
                create_on="06/01",
                name="v1",
                comment="This is a comment",
                model_name="MODEL",
                aliases=["FIRST", "DEFAULT"],
            ),
            Row(
                create_on="06/01",
                name="v2",
                comment="This is a comment",
                model_name="MODEL",
                aliases=["LAST"],
            ),
        ]
        with mock.patch.object(
            self.m_model._model_ops._model_client,
            "show_versions",
            return_value=m_list_res,
        ), mock.patch.object(self.m_model._model_ops, "validate_existence", return_value=True), mock.patch.object(
            model_version_impl.ModelVersion, "_get_functions", return_value=[]
        ):
            self.assertEqual(self.m_model.first().version_name, '"v1"')
            self.assertEqual(self.m_model.last().version_name, '"v2"')

    def test_default_setter(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "set_default_version") as mock_set_default_version:
            self.m_model.default = "V1"  # type: ignore[assignment]
            mock_set_default_version.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                statement_params=mock.ANY,
            )

        with mock.patch.object(
            self.m_model._model_ops, "set_default_version"
        ) as mock_set_default_version, mock.patch.object(
            model_version_impl.ModelVersion, "_get_functions", return_value=[]
        ):
            mv = model_version_impl.ModelVersion._ref(
                self.m_model._model_ops,
                service_ops=self.m_model._service_ops,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V2"),
            )
            self.m_model.default = mv
            mock_set_default_version.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V2"),
                statement_params=mock.ANY,
            )

    def test_delete_version(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "delete_model_or_version") as mock_delete_model_or_version:
            self.m_model.delete_version(version_name="V2")
            mock_delete_model_or_version.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V2"),
                statement_params=mock.ANY,
            )

    def test_show_tags(self) -> None:
        m_res = {'DB."schema".MYTAG': "tag content", 'MYDB.SCHEMA."my_another_tag"': "1"}
        with mock.patch.object(self.m_model._model_ops, "show_tags", return_value=m_res) as mock_show_tags:
            res = self.m_model.show_tags()
            self.assertDictEqual(res, m_res)
            mock_show_tags.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                statement_params=mock.ANY,
            )

    def test_get_tag_1(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "get_tag_value", return_value="tag content") as mock_get_tag:
            res = self.m_model.get_tag(tag_name="MYTAG")
            self.assertEqual(res, "tag content")
            mock_get_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=None,
                tag_schema_name=None,
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=mock.ANY,
            )

    def test_get_tag_2(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "get_tag_value", return_value="tag content") as mock_get_tag:
            res = self.m_model.get_tag(tag_name='"schema".MYTAG')
            self.assertEqual(res, "tag content")
            mock_get_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=None,
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=mock.ANY,
            )

    def test_get_tag_3(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "get_tag_value", return_value="tag content") as mock_get_tag:
            res = self.m_model.get_tag(tag_name='DB."schema".MYTAG')
            self.assertEqual(res, "tag content")
            mock_get_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=mock.ANY,
            )

    def test_set_tag_1(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "set_tag") as mock_set_tag:
            self.m_model.set_tag(tag_name="MYTAG", tag_value="tag content")
            mock_set_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=None,
                tag_schema_name=None,
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                tag_value="tag content",
                statement_params=mock.ANY,
            )

    def test_set_tag_2(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "set_tag") as mock_set_tag:
            self.m_model.set_tag(tag_name='"schema".MYTAG', tag_value="tag content")
            mock_set_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=None,
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                tag_value="tag content",
                statement_params=mock.ANY,
            )

    def test_set_tag_3(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "set_tag") as mock_set_tag:
            self.m_model.set_tag(tag_name='DB."schema".MYTAG', tag_value="tag content")
            mock_set_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                tag_value="tag content",
                statement_params=mock.ANY,
            )

    def test_unset_tag_1(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "unset_tag") as mock_unset_tag:
            self.m_model.unset_tag(tag_name="MYTAG")
            mock_unset_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=None,
                tag_schema_name=None,
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=mock.ANY,
            )

    def test_unset_tag_2(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "unset_tag") as mock_unset_tag:
            self.m_model.unset_tag(tag_name='"schema".MYTAG')
            mock_unset_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=None,
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=mock.ANY,
            )

    def test_unset_tag_3(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "unset_tag") as mock_unset_tag:
            self.m_model.unset_tag(tag_name='DB."schema".MYTAG')
            mock_unset_tag.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                tag_database_name=sql_identifier.SqlIdentifier("DB"),
                tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                tag_name=sql_identifier.SqlIdentifier("MYTAG"),
                statement_params=mock.ANY,
            )

    def test_rename(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "rename") as mock_rename:
            self.m_model.rename(model_name="MODEL2")
            mock_rename.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                new_model_db=None,
                new_model_schema=None,
                new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
                statement_params=mock.ANY,
            )

    def test_rename_fully_qualified_name(self) -> None:
        with mock.patch.object(self.m_model._model_ops, "rename") as mock_rename:
            self.m_model.rename(model_name='TEMP."test".MODEL2')
            mock_rename.assert_called_once_with(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                new_model_db=sql_identifier.SqlIdentifier("TEMP"),
                new_model_schema=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
                statement_params=mock.ANY,
            )


if __name__ == "__main__":
    absltest.main()
