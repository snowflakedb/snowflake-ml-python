from typing import cast

from absl.testing import absltest
from packaging import version

from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


class SnowflakeEnvTest(absltest.TestCase):
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


if __name__ == "__main__":
    absltest.main()
