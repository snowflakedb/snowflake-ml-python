from typing import List, cast

from absl.testing import absltest

from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.utils import pkg_version_utils
from snowflake.snowpark import row, session


class PackageVersionUtilsTest(absltest.TestCase):
    def test_happy_case(self) -> None:
        query = """SELECT *
                FROM information_schema.packages
                WHERE package_name = 'xgboost'
                AND version = '1.7.3';"""
        sql_result = [row.Row("xgboost", "1.7.3", "python")]

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        # Test
        pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

        # Test subsequent calls are served through cache.
        pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

    def test_unsupported_version(self) -> None:
        query = """SELECT *
                FROM information_schema.packages
                WHERE package_name = 'xgboost'
                AND version = '1.0.0';"""
        sql_result: List[row.Row] = [row.Row()]

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        # Test
        with self.assertRaises(RuntimeError):
            pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost==1.0.0"], session=c_session
            )

        # Test subsequent calls are served through cache.
        with self.assertRaises(RuntimeError):
            pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost==1.0.0"], session=c_session
            )

    def test_invalid_package_name(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(session.Session, m_session)
        with self.assertRaises(RuntimeError):
            pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost"], session=c_session
            )


if __name__ == "__main__":
    absltest.main()
