from typing import Dict, Optional, Tuple

from typing_extensions import TypedDict

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.snowpark import functions

_PROJECT = "MLOps"
_SUBPROJECT = "Monitor"


class BucketConfig(TypedDict):
    """ "Options for bucketizing the data."""

    min: int
    max: int
    size: int


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
@snowpark._internal.utils.private_preview(version="1.0.10")  # TODO: update versions when release
def compare_udfs_outputs(
    base_udf_name: str,
    test_udf_name: str,
    input_data_df: snowpark.DataFrame,
    bucket_config: Optional[BucketConfig] = None,
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
@snowpark._internal.utils.private_preview(version="1.0.10")  # TODO: update versions when release
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


@telemetry.send_api_usage_telemetry(
    project=_PROJECT,
    subproject=_SUBPROJECT,
)
@snowpark._internal.utils.private_preview(version="1.0.10")  # TODO: update versions when release
def jensenshannon(df1: snowpark.DataFrame, colname1: str, df2: snowpark.DataFrame, colname2: str) -> float:
    """
    Similar to scipy implementation:
    https://github.com/scipy/scipy/blob/e4dec2c5993faa381bb4f76dce551d0d79734f8f/scipy/spatial/distance.py#L1174
    It's server solution, all computing being in Snowflake warehouse, so will be significantly faster than client.

    Args:
        df1: 1st Snowpark Dataframe;
        colname1: the col to be selected in df1
        df2: 2nd Snowpark Dataframe;
        colname2: the col to be selected in df2
            Supported data Tyte: any data type that Snowflake supports, including VARIANT, OBJECT...etc.

    Returns:
        a jensenshannon value
    """
    df1 = df1.select(colname1)
    df1 = (
        df1.group_by(colname1)
        .agg(functions.count(colname1).alias("c1"))
        .select(functions.col(colname1).alias("d1"), "c1")
    )
    df2 = df2.select(colname2)
    df2 = (
        df2.group_by(colname2)
        .agg(functions.count(colname2).alias("c2"))
        .select(functions.col(colname2).alias("d2"), "c2")
    )

    dfsum = df1.select("c1").agg(functions.sum("c1").alias("SUM1"))
    sum1 = dfsum.collect()[0].as_dict()["SUM1"]
    dfsum = df2.select("c2").agg(functions.sum("c2").alias("SUM2"))
    sum2 = dfsum.collect()[0].as_dict()["SUM2"]

    df1 = df1.select("d1", functions.sql_expr("c1 / " + str(sum1)).alias("p"))
    minp = df1.select(functions.min("P").alias("MINP")).collect()[0].as_dict()["MINP"]
    df2 = df2.select("d2", functions.sql_expr("c2 / " + str(sum2)).alias("q"))
    minq = df2.select(functions.min("Q").alias("MINQ")).collect()[0].as_dict()["MINQ"]

    DECAY_FACTOR = 0.5
    df = df1.join(df2, df1.d1 == df2.d2, "fullouter").select(
        "d1",
        "d2",
        functions.sql_expr(
            """
            CASE
                WHEN p is NULL THEN {}*{}
                ELSE p
            END
            """.format(
                minp, DECAY_FACTOR
            )
        ).alias("p"),
        functions.sql_expr(
            """
            CASE
                WHEN q is NULL THEN {}*{}
                ELSE q
            END
            """.format(
                minq, DECAY_FACTOR
            )
        ).alias("q"),
    )

    df = df.select("p", "q", functions.sql_expr("(p+q)/2.0").alias("m"))
    df = df.select(
        functions.sql_expr(
            """
            CASE
                WHEN p > 0 AND m > 0 THEN p * LOG(2, p/m)
                ELSE 0
            END
            """
        ).alias("left"),
        functions.sql_expr(
            """
            CASE
                WHEN q > 0 AND m > 0 THEN q * LOG(2, q/m)
                ELSE 0
            END
            """
        ).alias("right"),
    )
    resdf = df.select(functions.sql_expr("sqrt((sum(left) + sum(right)) / 2.0)").alias("JS"))
    return float(resdf.collect()[0].as_dict()["JS"])


def _get_udf_query_str(
    name: str, col: str, df: snowpark.DataFrame, bucket_config: Optional[BucketConfig] = None
) -> str:
    if bucket_config:
        return "select count(1) as {}, width_bucket({}, {}, {}, {}) bucket from ({}) group by bucket".format(
            name, col, bucket_config["min"], bucket_config["max"], bucket_config["size"], df.queries["queries"][0]
        )
    else:  # don't bucket at all
        return f"{col} as {name}"
