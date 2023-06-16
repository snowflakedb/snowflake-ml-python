#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Dict, Tuple

from typing_extensions import TypedDict

from snowflake import snowpark
from snowflake.ml._internal import telemetry

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"


class BucketConfig(TypedDict):
    """ "Options for bucketizing the data."""

    min: int
    max: int
    size: int


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
def compare_udfs_outputs(
    base_udf_name: str, test_udf_name: str, input_data_df: snowpark.DataFrame, bucket_config: BucketConfig = None
) -> snowpark.DataFrame:
    """Compare outputs of 2 UDFs. Outputs are bucketized the based on bucketConfig.
    This is useful when someone retrain a Model and deploy as UDF to compare against earlier UDF as ground truth.
    NOTE: Only supports UDFs with single Column output.

    Args:
        base_udf_name: used as control ground truth UDF.
        test_udf_name: output of this UDF is compared against that of `base_udf`.
        input_data_df: Input data used for computing metric.
        bucket_config: must have the kv as {"min":xx, "max":xx, "size"}, keys in lowercase; it's width_bucket
            Sqloperator's config, using https://docs.snowflake.com/en/sql-reference/functions/width_bucket.

    Returns:
        snowpark.DataFrame.
        "BASEUDF" is base_udf's bucketized output, "TESTUDF" is test_udf's bucketized output,
    """
    if bucket_config:
        assert len(bucket_config) == 3
        assert "min" in bucket_config and "max" in bucket_config and "size" in bucket_config

    argStr = ",".join(input_data_df.columns)
    query1Str = _get_udf_query_str("BASEUDF", f"{base_udf_name}({argStr})", input_data_df, bucket_config)
    query2Str = _get_udf_query_str("TESTUDF", f"{test_udf_name}({argStr})", input_data_df, bucket_config)

    if bucket_config:
        finalStr = (
            "select A.bucket, BASEUDF, TESTUDF \n from ({}) as A \n join ({}) as B \n on A.bucket=B.bucket".format(
                query1Str, query2Str
            )
        )
    else:  # don't bucket at all
        finalStr = "select {},{} \n from ({})".format(query1Str, query2Str, input_data_df.queries["queries"][0])

    assert input_data_df._session is not None
    return input_data_df._session.sql(finalStr)


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
def get_basic_stats(df: snowpark.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Get basic stats of 2 Columns
    Note this isn't public API. Only support min, max, stddev, HLL--cardinality estimate

    Args:
        df: input Snowpark Dataframe, must have 2 and only 2 columns

    Returns:
        2 Dict for 2 columns' stats
    """
    projStr = ""
    stats = ["MIN", "MAX", "STDDEV", "HLL"]
    assert len(df.columns) == 2
    for colName in df.columns:
        for stat in stats:
            projStr += f"{stat}({colName}) as {colName}_{stat},"
    finalStr = "select {} \n from ({})".format(projStr[:-1], df.queries["queries"][0])
    assert df._session is not None
    resDf = df._session.sql(finalStr).to_pandas()
    d1 = {}
    col1 = df.columns[0]
    d2 = {}
    col2 = df.columns[1]
    for stat in stats:
        d1[stat] = resDf.iloc[0][f"{col1}_{stat}"]
        d2[stat] = resDf.iloc[0][f"{col2}_{stat}"]
    return d1, d2


def _get_udf_query_str(name: str, col: str, df: snowpark.DataFrame, bucket_config: BucketConfig = None) -> str:
    if bucket_config:
        return "select count(1) as {}, width_bucket({}, {}, {}, {}) bucket from ({}) group by bucket".format(
            name, col, bucket_config["min"], bucket_config["max"], bucket_config["size"], df.queries["queries"][0]
        )
    else:  # don't bucket at all
        return f"{col} as {name}"
