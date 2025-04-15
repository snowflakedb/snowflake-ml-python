import json
from typing import Any, Optional

import dataset_metadata
from absl.testing import absltest, parameterized

from snowflake.ml.feature_store import entity, feature_view
from snowflake.snowpark import dataframe


class MockDataFrame(dataframe.DataFrame):
    def __init__(self, query: str, columns: list[str]) -> None:
        self._query = query
        self._columns = [c.upper() for c in columns]

    @property
    def queries(self) -> dict[str, list[str]]:
        return {"queries": [self._query]}

    @property
    def columns(self) -> list[str]:
        return self._columns


def _create_feature_view(name: str, columns: list[str]) -> feature_view.FeatureView:
    df = MockDataFrame("test query", columns)
    return feature_view.FeatureView(
        name,
        [entity.Entity("e", columns[:1])],
        feature_df=df,
    )


def _create_metadata(props: Any) -> dataset_metadata.DatasetMetadata:
    return dataset_metadata.DatasetMetadata(
        source_query="test",
        owner="test",
        properties=props,
    )


class DatasetMetadataTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        {"input_props": None},
        {"input_props": dataset_metadata.FeatureStoreMetadata("query", {"conn": "test"}, ["feat1", "feat2"])},
        {"input_props": dataset_metadata.FeatureStoreMetadata("query", ["feat1", "feat2"], spine_timestamp_col="ts")},
        {
            "input_props": dataset_metadata.FeatureStoreMetadata(
                "query",
                [
                    _create_feature_view("fv1", ["col1", "col2"]).to_json(),
                    _create_feature_view("fv2", ["col1", "col3", "col4"]).slice(["col4"]).to_json(),
                ],
                spine_timestamp_col="ts",
            )
        },
    )
    def test_json_convert(self, input_props: Any) -> None:
        expected = input_props

        source = _create_metadata(input_props)
        serialized = source.to_json()
        actual = dataset_metadata.DatasetMetadata.from_json(serialized).properties

        self.assertEqual(expected, actual)

    @parameterized.parameters(  # type: ignore[misc]
        {"input_props": {"custom": "value"}},
    )
    def test_json_convert_negative(self, input_props: Any) -> None:
        source = _create_metadata(input_props)
        with self.assertRaises(ValueError):
            source.to_json()

    @parameterized.parameters(  # type: ignore[misc]
        '{"source_query": "test_source", "owner": "test"}',
        '{"source_query": "test_source", "owner": "test", "exclude_cols": ["col1"]}',
        '{"source_query": "test_source", "owner": "test", "label_cols": ["col1"]}',
        '{"source_query": "test_source", "owner": "test", "properties": null}',
        '{"source_query": "test_source", "owner": "test", "properties": { } }',
        '{"source_query": "test_source", "owner": "test", "properties": null}',
        """
        {
            "source_query": "test_source",
            "owner": "test",
            "$proptype$": "FeatureStoreMetadata",
            "properties": {
                "spine_query": "test query",
                "serialized_feature_views": []
            }
        }
        """,
        """
        {
            "source_query": "test_source",
            "owner": "test",
            "$proptype$": "FeatureStoreMetadata",
            "properties": {
                "spine_query": "test query",
                "serialized_feature_views": [
                    "{\\\"_name\\\": \\\"fv1\\\"}"
                ],
                "spine_timestamp_col": "ts_col"
            }
        }
        """,
    )
    def test_deserialize(self, json_str: str) -> None:
        actual = dataset_metadata.DatasetMetadata.from_json(json_str)
        self.assertIsNotNone(actual)

        json_dict = json.loads(json_str)
        actual2 = dataset_metadata.DatasetMetadata.from_json(json_dict)
        self.assertIsNotNone(actual2)

    @parameterized.parameters(  # type: ignore[misc]
        None,
        "",
        "{}",
        '{"unrelated": "value"}',
        '{"source_query": "test_source", "owner": "test", "properties": {"prop1": "val1"} }',
        """
        {
            "source_query": "test_source",
            "owner": "test",
            "$proptype$": "unrecognized",
            "properties": {"prop1": "val1"}
        }
        """,
        """
        {
            "source_query": "test_source",
            "owner": "test",
            "$proptype$": "dict",
            "properties": {"prop1": "val1"}
        }
        """,
        """
        {
            "source_query": "test_source",
            "owner": "test",
            "$proptype$": "FeatureStoreMetadata",
            "properties": {"prop1": "val1"}
        }
        """,
        """
        {
            "source_query": "test_source",
            "owner": "test",
            "$proptype$": "FeatureStoreMetadata"
        }
        """,
        # FIXME: These test cases currently fail due to lack of type enforcement
        # '{"source_query": "test_source", "owner": "test"}',
        # '{"source_query": "test_source", "owner": "test", "exclude_cols": "col1"}',
        # '{"source_query": "test_source", "owner": "test", "label_cols": "col1"}',
        # '{"source_query": "test_source", "owner": "test", "properties": "value"}',
    )
    def test_deserialize_negative(self, json_str: Optional[str]) -> None:
        with self.assertRaises(ValueError):
            dataset_metadata.DatasetMetadata.from_json(json_str)

        json_dict = json.loads(json_str) if json_str else None
        with self.assertRaises(ValueError):
            dataset_metadata.DatasetMetadata.from_json(json_dict)


if __name__ == "__main__":
    absltest.main()
