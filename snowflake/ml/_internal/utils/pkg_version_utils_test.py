import sys
from typing import cast

from absl.testing import absltest

from snowflake.ml._internal.utils import pkg_version_utils
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import session

_RUNTIME_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


class PackageVersionUtilsTest(absltest.TestCase):
    def test_happy_case(self) -> None:
        query = """SELECT *
                FROM information_schema.packages
                WHERE package_name = 'xgboost'
                AND version = '1.7.3'"""

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(
            query=query,
            result=mock_data_frame.MockDataFrame(count_result=1, columns=["PACKAGE_NAME", "VERSION", "LANGUAGE"]),
        )
        c_session = cast(session.Session, m_session)

        # Test
        pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

        # Test subsequent calls are served through cache.
        pkg_version_utils.validate_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

    def test_happy_case_with_runtime_version_column(self) -> None:
        query = """SELECT *
                FROM information_schema.packages
                WHERE package_name = 'xgboost'
                AND version = '1.7.3'"""

        m_session = mock_session.MockSession(conn=None, test_case=self)
        mock_df = mock_data_frame.MockDataFrame(columns=["PACKAGE_NAME", "VERSION", "LANGUAGE", "RUNTIME_VERSION"])
        mock_df = mock_df.add_mock_filter(
            expr=f"RUNTIME_VERSION = {_RUNTIME_VERSION}", result=mock_data_frame.MockDataFrame(count_result=1)
        )
        m_session.add_mock_sql(query=query, result=mock_df)
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
                AND version = '1.0.0'"""

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(
            query=query,
            result=mock_data_frame.MockDataFrame(count_result=0, columns=["PACKAGE_NAME", "VERSION", "LANGUAGE"]),
        )
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

    def test_unsupported_version_with_runtime_version_column(self) -> None:
        query = """SELECT *
                FROM information_schema.packages
                WHERE package_name = 'xgboost'
                AND version = '1.0.0'"""

        m_session = mock_session.MockSession(conn=None, test_case=self)
        mock_df = mock_data_frame.MockDataFrame(columns=["PACKAGE_NAME", "VERSION", "LANGUAGE", "RUNTIME_VERSION"])
        mock_df.add_mock_filter(
            expr=f"RUNTIME_VERSION = {_RUNTIME_VERSION}", result=mock_data_frame.MockDataFrame(count_result=0)
        )
        m_session.add_mock_sql(query=query, result=mock_df)
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
