import enum
from typing import Any, Optional, TypedDict, cast

from packaging import version
from typing_extensions import NotRequired, Required

from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import exceptions as sp_exceptions, session


def get_current_snowflake_version(
    sess: session.Session, *, statement_params: Optional[dict[str, Any]] = None
) -> version.Version:
    """Get Snowflake Version as a version.Version object follow PEP way of versioning, that is to say:
        "7.44.2 b202312132139364eb71238" to <Version('7.44.2+b202312132139364eb71238')>

    Args:
        sess: Snowpark Session.
        statement_params: Statement params. Defaults to None.

    Returns:
        The version of Snowflake Version.
    """
    res = (
        query_result_checker.SqlResultValidator(
            sess, "SELECT CURRENT_VERSION() AS CURRENT_VERSION", statement_params=statement_params
        )
        .has_dimensions(expected_rows=1, expected_cols=1)
        .validate()[0]
    )

    version_str = res.CURRENT_VERSION
    assert isinstance(version_str, str)

    version_str = "+".join(version_str.split())
    return version.parse(version_str)


class SnowflakeCloudType(enum.Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

    @classmethod
    def from_value(cls, value: str) -> "SnowflakeCloudType":
        assert value
        for k in cls:
            if k.value == value.lower():
                return k
        else:
            raise ValueError(f"'{cls.__name__}' enum not found for '{value}'")


class SnowflakeRegion(TypedDict):
    region_group: NotRequired[str]
    snowflake_region: Required[str]
    cloud: Required[SnowflakeCloudType]
    region: Required[str]
    display_name: Required[str]


def get_regions(
    sess: session.Session, *, statement_params: Optional[dict[str, Any]] = None
) -> dict[str, SnowflakeRegion]:
    res = (
        query_result_checker.SqlResultValidator(sess, "SHOW REGIONS", statement_params=statement_params)
        .has_column("snowflake_region")
        .has_column("cloud")
        .has_column("region")
        .has_column("display_name")
        .validate()
    )
    res_dict = {}
    for r in res:
        if hasattr(r, "region_group") and r.region_group:
            key = f"{r.region_group}.{r.snowflake_region}"
            res_dict[key] = SnowflakeRegion(
                region_group=r.region_group,
                snowflake_region=r.snowflake_region,
                cloud=SnowflakeCloudType.from_value(r.cloud),
                region=r.region,
                display_name=r.display_name,
            )
        else:
            key = r.snowflake_region
            res_dict[key] = SnowflakeRegion(
                snowflake_region=r.snowflake_region,
                cloud=SnowflakeCloudType.from_value(r.cloud),
                region=r.region,
                display_name=r.display_name,
            )

    return res_dict


def get_current_region_id(sess: session.Session, *, statement_params: Optional[dict[str, Any]] = None) -> str:
    res = (
        query_result_checker.SqlResultValidator(
            sess, "SELECT CURRENT_REGION() AS CURRENT_REGION", statement_params=statement_params
        )
        .has_dimensions(expected_rows=1, expected_cols=1)
        .validate()[0]
    )

    return cast(str, res.CURRENT_REGION)


def get_current_cloud(
    sess: session.Session,
    default: Optional[SnowflakeCloudType] = None,
    *,
    statement_params: Optional[dict[str, Any]] = None,
) -> SnowflakeCloudType:
    region_id = get_current_region_id(sess, statement_params=statement_params)
    try:
        region = get_regions(sess, statement_params=statement_params)[region_id]
        return region["cloud"]
    except sp_exceptions.SnowparkSQLException:
        # SHOW REGIONS not available, try to infer cloud from region name
        region_name = region_id.split(".", 1)[-1]  # Drop region group if any, e.g. PUBLIC
        cloud_name_maybe = region_name.split("_", 1)[0]  # Extract cloud name, e.g. AWS_US_WEST -> AWS
        try:
            return SnowflakeCloudType.from_value(cloud_name_maybe)
        except ValueError:
            if default:
                return default
            raise
