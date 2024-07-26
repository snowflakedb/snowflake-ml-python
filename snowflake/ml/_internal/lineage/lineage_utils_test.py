from typing import Any, List, Optional, Union, get_args

from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml._internal.lineage import lineage_utils
from snowflake.ml.data import data_source
from snowflake.ml.utils import connection_params
from snowflake.snowpark import functions as F


class LineageUtilsTest(parameterized.TestCase):
    class TestSourcedObject:
        def __init__(self, sources: Optional[List[data_source.DataSource]]) -> None:
            setattr(self, lineage_utils._DATA_SOURCES_ATTR, sources)

    def setUp(self) -> None:
        connection_parameters = connection_params.SnowflakeLoginOptions()
        self.session = snowpark.Session.builder.configs(connection_parameters).create()

        self.datasource = data_source.DatasetInfo("db.schema.my_ds", "v1", "snow://dataset/my_ds/versions/v1")
        self.df = lineage_utils.patch_dataframe(
            self.session.sql(
                "SELECT SEQ4() AS ID"
                ", UNIFORM(1, 10, RANDOM(42)) AS VALUE"
                ", [1,2] AS ARRAY"
                " FROM TABLE(GENERATOR(ROWCOUNT => 10))"
            ),
            [self.datasource],
            inplace=True,
        )

    @parameterized.parameters(  # type: ignore[misc]
        (
            [],
            None,
        ),
        (
            [TestSourcedObject(None)],
            None,
        ),
        (
            [TestSourcedObject(None), TestSourcedObject(None)],
            None,
        ),
        (
            [TestSourcedObject([])],
            [],
        ),
        (
            [TestSourcedObject([data_source.DatasetInfo("foo", "v1", "foo_url")])],
            [data_source.DatasetInfo("foo", "v1", "foo_url")],
        ),
        (
            [
                TestSourcedObject([data_source.DatasetInfo("foo", "v1", "foo_url")]),
                TestSourcedObject([data_source.DatasetInfo("foo", "v2", "foo_url")]),
            ],
            [data_source.DatasetInfo("foo", "v1", "foo_url"), data_source.DatasetInfo("foo", "v2", "foo_url")],
        ),
        # FIXME: Enable this test case once dedupe support added
        # (
        #     [
        #         TestSourcedObject([data_source.DatasetInfo("foo", "v1", "foo_url")]),
        #         TestSourcedObject([data_source.DatasetInfo("foo", "v1", "foo_url")]),
        #         TestSourcedObject([data_source.DatasetInfo("foo", "v2", "foo_url")]),
        #     ],
        #     [data_source.DatasetInfo("foo", "v1", "foo_url"), data_source.DatasetInfo("foo", "v2", "foo_url")],
        # ),
    )
    def test_get_data_sources(
        self, args: List[TestSourcedObject], expected: Optional[List[data_source.DataSource]]
    ) -> None:
        self.assertEqual(expected, lineage_utils.get_data_sources(*args))

    @parameterized.product(  # type: ignore[misc]
        data_sources=[
            None,
            [],
            [data_source.DatasetInfo("foo", "v1", "foo_url")],
            [data_source.DatasetInfo("foo", "v1", "foo_url"), data_source.DatasetInfo("foo", "v1", "foo_url")],
        ],
        inplace=[True, False],
    )
    def test_patch_dataframe(self, data_sources: List[data_source.DataSource], inplace: bool) -> None:
        df = self.session.sql("SELECT 1")
        out_df = lineage_utils.patch_dataframe(df, data_sources=data_sources, inplace=inplace)
        self.validate_dataframe(out_df, data_sources)

    @parameterized.parameters(  # type: ignore[misc]
        ("select", F.col("id")),
        ("select_expr", "abs(value)"),
        ("drop", F.col("id")),
        ("filter", F.col("value") > 5),
        ("sort", F.col("value"), F.col("id")),
        ("alias", "foo"),
        ("agg", (F.sum(F.col("value")))),
        ("rollup", F.col("value")),
        ("group_by", F.col("value")),
        ("group_by_grouping_sets", snowpark.GroupingSets(F.col("value"))),
        ("cube", F.col("value")),
        ("distinct",),
        ("drop_duplicates", "id"),
        ("pivot", F.col("value"), list(range(1, 11))),
        ("unpivot", "foo", "bar", "value"),
        ("limit", 1),
        ("with_column", "new_id", F.seq8()),
        ("with_columns", ["id1", "id2"], [F.uniform(1, 5, 42), F.uniform(6, 10, 42)]),
        ("flatten", F.col("array")),
        ("sample", 0.5),
        ("describe", "value"),
        ("rename", F.col("value"), "my_val"),
        ("with_column_renamed", F.col("id"), "new_id"),
        ("cache_result",),
        ("random_split", [0.8, 0.2]),
        ("join_table_function", F.flatten(F.col("array"))),
        ("__copy__"),
    )
    def test_dataframe_func(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        func = getattr(self.df, func_name)
        out_df = func(*args, **kwargs)
        if isinstance(out_df, list):
            for df in out_df:
                self.validate_dataframe(df, [self.datasource])
        else:
            self.validate_dataframe(out_df, [self.datasource])

    @parameterized.parameters(  # type: ignore[misc]
        ("union",),
        ("union_all",),
        ("union_by_name",),
        ("union_all_by_name",),
        ("intersect",),
        ("except_",),
        ("natural_join",),
        ("join",),
        ("cross_join",),
    )
    def test_combine_dataframe_func(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        other_df = self.session.sql(
            "SELECT SEQ4() AS ID"
            ", UNIFORM(1, 10, RANDOM(0)) AS VALUE"
            ", [3,4] AS ARRAY"
            " FROM TABLE(GENERATOR(ROWCOUNT => 10))"
        )
        func = getattr(self.df, func_name)
        out_df = func(other_df, *args, **kwargs)
        self.validate_dataframe(out_df, [self.datasource])

    def test_dataframe_pipeline(self) -> None:
        out_df = self.df.with_column("new_id", F.seq8()).alias("foo").drop(F.col("new_id")).limit(10)
        self.validate_dataframe(out_df, [self.datasource])

    @parameterized.parameters(  # type: ignore[misc]
        ("select", F.col("id")),
        ("select_expr", "abs(value)"),
        ("drop", F.col("id")),
        ("filter", F.col("value") > 5),
        ("sort", F.col("value"), F.col("id")),
        ("alias", "foo"),
        ("agg", (F.sum(F.col("value")))),
        ("rollup", F.col("value")),
        ("group_by", F.col("value")),
        ("group_by_grouping_sets", snowpark.GroupingSets(F.col("value"))),
        ("cube", F.col("value")),
        ("distinct",),
        ("drop_duplicates", "id"),
        ("pivot", F.col("value"), list(range(1, 11))),
        ("unpivot", "foo", "bar", "value"),
        ("limit", 1),
        ("with_column", "new_id", F.seq8()),
        ("with_columns", ["id1", "id2"], [F.uniform(1, 5, 42), F.uniform(6, 10, 42)]),
        ("flatten", F.col("array")),
        ("sample", 0.5),
        ("describe", "value"),
        ("rename", F.col("value"), "my_val"),
        ("with_column_renamed", F.col("id"), "new_id"),
        ("cache_result",),
        ("random_split", [0.8, 0.2]),
        ("join_table_function", F.flatten(F.col("array"))),
        ("__copy__"),
    )
    def test_vanilla_dataframe_func(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        df = self.session.sql(
            "SELECT SEQ4() AS ID"
            ", UNIFORM(1, 10, RANDOM(42)) AS VALUE"
            ", [1,2] AS ARRAY"
            " FROM TABLE(GENERATOR(ROWCOUNT => 10))"
        )
        func = getattr(df, func_name)
        out_df = func(*args, **kwargs)
        if isinstance(out_df, list):
            for _df in out_df:
                self.validate_dataframe(_df, None)
        else:
            self.validate_dataframe(out_df, None)

    def validate_dataframe(self, df: Any, data_sources: Optional[List[data_source.DataSource]]) -> None:
        self.assertIsInstance(df, get_args(Union[snowpark.DataFrame, snowpark.RelationalGroupedDataFrame]))
        self.assertEqual(data_sources, lineage_utils.get_data_sources(df))


if __name__ == "__main__":
    absltest.main()
