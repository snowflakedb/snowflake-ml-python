from typing import Optional, Union, cast

from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session, exceptions as sp_exceptions


class SnowflakeEnvTest(parameterized.TestCase):
    def test_current_snowflake_version_1(self) -> None:
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "SELECT CURRENT_VERSION() AS CURRENT_VERSION"
        sql_result = [Row(CURRENT_VERSION="8.0.0")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = snowflake_env.get_current_snowflake_version(cast(Session, session))
        self.assertEqual(actual_result, version.parse("8.0.0"))

    def test_current_snowflake_version_2(self) -> None:
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "SELECT CURRENT_VERSION() AS CURRENT_VERSION"
        sql_result = [Row(CURRENT_VERSION="8.0.0 1234567890ab")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = snowflake_env.get_current_snowflake_version(cast(Session, session))
        self.assertEqual(actual_result, version.parse("8.0.0+1234567890ab"))

    def test_get_regions(self) -> None:
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "SHOW REGIONS"
        sql_result = [
            Row(
                snowflake_region="AWS_US_WEST_1",
                cloud="aws",
                region="us-west-1",
                display_name="US West (Oregon)",
            ),
            Row(
                region_group="PUBLIC",
                snowflake_region="AWS_US_WEST_2",
                cloud="aws",
                region="us-west-2",
                display_name="US West (Oregon)",
            ),
            Row(
                region_group="PUBLIC",
                snowflake_region="AZURE_EASTUS2",
                cloud="azure",
                region="eastus2",
                display_name="East US 2 (Virginia)",
            ),
            Row(
                region_group="PUBLIC",
                snowflake_region="GCP_EUROPE_WEST2",
                cloud="gcp",
                region="europe-west2",
                display_name="Europe West 2 (London)",
            ),
        ]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = snowflake_env.get_regions(cast(Session, session))
        self.assertDictEqual(
            {
                "AWS_US_WEST_1": snowflake_env.SnowflakeRegion(
                    snowflake_region="AWS_US_WEST_1",
                    cloud=snowflake_env.SnowflakeCloudType.AWS,
                    region="us-west-1",
                    display_name="US West (Oregon)",
                ),
                "PUBLIC.AWS_US_WEST_2": snowflake_env.SnowflakeRegion(
                    region_group="PUBLIC",
                    snowflake_region="AWS_US_WEST_2",
                    cloud=snowflake_env.SnowflakeCloudType.AWS,
                    region="us-west-2",
                    display_name="US West (Oregon)",
                ),
                "PUBLIC.AZURE_EASTUS2": snowflake_env.SnowflakeRegion(
                    region_group="PUBLIC",
                    snowflake_region="AZURE_EASTUS2",
                    cloud=snowflake_env.SnowflakeCloudType.AZURE,
                    region="eastus2",
                    display_name="East US 2 (Virginia)",
                ),
                "PUBLIC.GCP_EUROPE_WEST2": snowflake_env.SnowflakeRegion(
                    region_group="PUBLIC",
                    snowflake_region="GCP_EUROPE_WEST2",
                    cloud=snowflake_env.SnowflakeCloudType.GCP,
                    region="europe-west2",
                    display_name="Europe West 2 (London)",
                ),
            },
            actual_result,
        )

    def test_get_current_region_id(self) -> None:
        session = mock_session.MockSession(conn=None, test_case=self)
        query = "SELECT CURRENT_REGION() AS CURRENT_REGION"
        sql_result = [Row(CURRENT_REGION="PUBLIC.AWS_US_WEST_2")]
        session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        actual_result = snowflake_env.get_current_region_id(cast(Session, session))
        self.assertEqual(actual_result, "PUBLIC.AWS_US_WEST_2")

    @parameterized.parameters(  # type: ignore[misc]
        ("AWS_US_WEST_2", "aws"),
        ("PUBLIC.AWS_US_WEST_2", "aws"),
        ("AZURE_EASTUS2", "azure"),
        ("GCP_EUROPE_WEST2", "gcp"),
        ("PREPROD6", "aws", "aws"),
        ("PUBLIC.PREPROD6", "aws", "aws"),
        ("PREPROD6", ValueError("enum not found")),
        ("PUBLIC.PREPROD6", ValueError("enum not found")),
    )
    def test_get_current_cloud_no_show_regions(
        self, region: str, expected: Union[str, Exception], default_cloud: Optional[str] = None
    ) -> None:
        session = mock_session.MockSession(conn=None, test_case=self)

        # Configure CURRENT_REGION() query
        query1 = "SELECT CURRENT_REGION() AS CURRENT_REGION"
        sql_result1 = [Row(CURRENT_REGION=region)]
        session.add_mock_sql(query=query1, result=mock_data_frame.MockDataFrame(sql_result1))

        # Configure SHOW REGIONS query
        query2 = "SHOW REGIONS"
        sql_result2 = sp_exceptions.SnowparkSQLException("Unsupported statement type 'SHOW DEPLOYMENT_LOCATION'")
        session.add_mock_sql(query=query2, result=mock_data_frame.MockDataFrame(sql_result2))

        default = snowflake_env.SnowflakeCloudType(default_cloud) if default_cloud is not None else None
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                snowflake_env.get_current_cloud(cast(Session, session), default=default)
        else:
            actual_result = snowflake_env.get_current_cloud(cast(Session, session), default=default)
            self.assertEqual(actual_result, snowflake_env.SnowflakeCloudType(expected))


if __name__ == "__main__":
    absltest.main()
