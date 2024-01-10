import enum
from typing import Any, Dict, Optional, TypedDict, cast

from packaging import version
from typing_extensions import Required

from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import session


def get_current_snowflake_version(
    sess: session.Session, *, statement_params: Optional[Dict[str, Any]] = None
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
    region_group: Required[str]
    snowflake_region: Required[str]
    cloud: Required[SnowflakeCloudType]
    region: Required[str]
    display_name: Required[str]


def get_regions(
    sess: session.Session, *, statement_params: Optional[Dict[str, Any]] = None
) -> Dict[str, SnowflakeRegion]:
    res = (
        query_result_checker.SqlResultValidator(sess, "SHOW REGIONS", statement_params=statement_params)
        .has_column("region_group")
        .has_column("snowflake_region")
        .has_column("cloud")
        .has_column("region")
        .has_column("display_name")
        .validate()
    )
    return {
        f"{r.region_group}.{r.snowflake_region}": SnowflakeRegion(
            region_group=r.region_group,
            snowflake_region=r.snowflake_region,
            cloud=SnowflakeCloudType.from_value(r.cloud),
            region=r.region,
            display_name=r.display_name,
        )
        for r in res
    }


def get_current_region_id(sess: session.Session, *, statement_params: Optional[Dict[str, Any]] = None) -> str:
    res = (
        query_result_checker.SqlResultValidator(
            sess, "SELECT CURRENT_REGION() AS CURRENT_REGION", statement_params=statement_params
        )
        .has_dimensions(expected_rows=1, expected_cols=1)
        .validate()[0]
    )

    return cast(str, res.CURRENT_REGION)
