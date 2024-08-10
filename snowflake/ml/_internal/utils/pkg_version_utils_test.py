import sys
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import pkg_version_utils
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, session

_RUNTIME_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


class PackageVersionUtilsTest(absltest.TestCase):
    @mock.patch.dict(pkg_version_utils.cache, {})
    def test_happy_case_async(self) -> None:
        pkg_name = "xgboost"
        major_version, minor_version, micro_version = 1, 7, 3
        query = f"""
        SELECT PACKAGE_NAME, VERSION, LANGUAGE
            FROM (
                SELECT *,
                SUBSTRING(VERSION, LEN(VERSION) - CHARINDEX('.', REVERSE(VERSION)) + 2, LEN(VERSION)) as micro_version
                FROM information_schema.packages
                WHERE package_name = '{pkg_name}'
                AND version LIKE '{major_version}.{minor_version}.%'
                ORDER BY abs({micro_version}-micro_version), -micro_version
            )
        """

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(
            query=query,
            result=mock_data_frame.MockDataFrame(
                collect_result=mock_data_frame.MockAsyncJob(
                    result=[Row(PACKAGE_NAME="xgboost", VERSION="1.7.3", LANGUAGE="python")]
                ),
                columns=["PACKAGE_NAME", "VERSION", "LANGUAGE"],
                collect_block=False,
            ),
        )
        c_session = cast(session.Session, m_session)

        # Test
        pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

        # Test subsequent calls are served through cache.
        pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

    @mock.patch("snowflake.ml._internal.utils.pkg_version_utils.snowpark_utils.is_in_stored_procedure")
    def test_happy_case(self, mock_is_in_stored_procedure: mock.Mock) -> None:
        mock_is_in_stored_procedure.return_value = True

        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(session.Session, m_session)

        # Test
        valid_pkg_versions = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )
        self.assertEqual(valid_pkg_versions, ["xgboost==1.7.3"])

    @mock.patch.dict(pkg_version_utils.cache, {})
    def test_happy_case_with_runtime_version_column_async(self) -> None:
        pkg_name = "xgboost"
        major_version, minor_version, micro_version = 1, 7, 3
        query = f"""
        SELECT PACKAGE_NAME, VERSION, LANGUAGE
            FROM (
                SELECT *,
                SUBSTRING(VERSION, LEN(VERSION) - CHARINDEX('.', REVERSE(VERSION)) + 2, LEN(VERSION)) as micro_version
                FROM information_schema.packages
                WHERE package_name = '{pkg_name}'
                AND version LIKE '{major_version}.{minor_version}.%'
                ORDER BY abs({micro_version}-micro_version), -micro_version
            )
        """

        m_session = mock_session.MockSession(conn=None, test_case=self)
        mock_df = mock_data_frame.MockDataFrame(
            columns=["PACKAGE_NAME", "VERSION", "LANGUAGE", "RUNTIME_VERSION"], collect_block=False
        )
        mock_df = mock_df.add_mock_filter(
            expr=f"RUNTIME_VERSION = {_RUNTIME_VERSION}",
            result=mock_data_frame.MockDataFrame(
                collect_result=mock_data_frame.MockAsyncJob(
                    result=[
                        Row(
                            PACKAGE_NAME="xgboost",
                            VERSION="1.7.3",
                            LANGUAGE="python",
                            RUNTIME_VERSION=_RUNTIME_VERSION,
                        )
                    ]
                ),
                columns=["PACKAGE_NAME", "VERSION", "LANGUAGE", "RUNTIME_VERSION"],
                collect_block=False,
            ),
        )

        m_session.add_mock_sql(query=query, result=mock_df)
        c_session = cast(session.Session, m_session)

        # Test
        pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

        # Test subsequent calls are served through cache.
        pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=["xgboost==1.7.3"], session=c_session
        )

    def test_unsupported_version_async(self) -> None:
        pkg_name = "xgboost"
        major_version, minor_version, micro_version = 1, 0, 0
        query = f"""
        SELECT PACKAGE_NAME, VERSION, LANGUAGE
            FROM (
                SELECT *,
                SUBSTRING(VERSION, LEN(VERSION) - CHARINDEX('.', REVERSE(VERSION)) + 2, LEN(VERSION)) as micro_version
                FROM information_schema.packages
                WHERE package_name = '{pkg_name}'
                AND version LIKE '{major_version}.{minor_version}.%'
                ORDER BY abs({micro_version}-micro_version), -micro_version
            )
        """

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(
            query=query,
            result=mock_data_frame.MockDataFrame(
                collect_result=mock_data_frame.MockAsyncJob(result=[]),
                columns=["PACKAGE_NAME", "VERSION", "LANGUAGE"],
                collect_block=False,
            ),
        )
        c_session = cast(session.Session, m_session)

        # Test
        with self.assertRaises(RuntimeError):
            pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost==1.0.0"], session=c_session
            )

        # Test subsequent calls are served through cache.
        with self.assertRaises(RuntimeError):
            pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost==1.0.0"], session=c_session
            )

    def test_unsupported_version_with_runtime_version_column_async(self) -> None:
        query = """SELECT PACKAGE_NAME, VERSION, LANGUAGE
            FROM (
                SELECT *,
                SUBSTRING(VERSION, LEN(VERSION) - CHARINDEX('.', REVERSE(VERSION)) + 2, LEN(VERSION)) as micro_version
                FROM information_schema.packages
                WHERE package_name = 'xgboost'
                AND version LIKE '1.0.%'
                ORDER BY abs(0-micro_version), -micro_version
            )"""

        m_session = mock_session.MockSession(conn=None, test_case=self)
        mock_df = mock_data_frame.MockDataFrame(
            columns=["PACKAGE_NAME", "VERSION", "LANGUAGE", "RUNTIME_VERSION"], collect_block=False
        )
        mock_df.add_mock_filter(
            expr=f"RUNTIME_VERSION = {_RUNTIME_VERSION}", result=mock_data_frame.MockDataFrame(count_result=0)
        )
        m_session.add_mock_sql(query=query, result=mock_df)
        c_session = cast(session.Session, m_session)

        # Test
        with self.assertRaises(RuntimeError):
            pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost==1.0.0"], session=c_session
            )

        # Test subsequent calls are served through cache.
        with self.assertRaises(RuntimeError):
            pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost==1.0.0"], session=c_session
            )

    def test_invalid_package_name(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(session.Session, m_session)
        with self.assertRaises(RuntimeError):
            pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=["xgboost"], session=c_session
            )


if __name__ == "__main__":
    absltest.main()
