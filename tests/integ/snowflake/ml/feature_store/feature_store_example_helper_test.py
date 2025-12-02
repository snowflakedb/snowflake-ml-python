from absl.testing import absltest
from common_utils import compare_dataframe, create_random_schema
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    CreationMode,
    FeatureStore,
)
from snowflake.ml.feature_store.examples.example_helper import ExampleHelper


class FeatureStoreExampleHelperTest(FeatureStoreIntegTestBase):
    def test_example_helper(self) -> None:
        current_schema = create_random_schema(self._session, "FS_EXAMPLE_HELP_TEST", database=self.test_db)
        default_warehouse = self._session.get_current_warehouse()
        fs = FeatureStore(
            self._session,
            self.test_db,
            current_schema,
            default_warehouse=default_warehouse,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        helper = ExampleHelper(self._session, self.test_db, current_schema)
        all_examples = helper.list_examples()
        self.assertIsNotNone(all_examples)
        expected_examples = [
            "new_york_taxi_features",
            "wine_quality_features",
            "airline_features",
        ]
        compare_dataframe(
            actual_df=all_examples.drop("desc", "label_cols", "model_category").to_pandas(),  # type: ignore[union-attr]
            target_data={
                "NAME": expected_examples,
            },
            sort_cols=["NAME"],
        )

        for example in expected_examples:
            loaded_tables = helper.load_example(example)
            self.assertGreater(len(loaded_tables), 0)
            self.assertEqual(helper.get_current_schema(), current_schema)
            self.assertGreater(len(helper.get_label_cols()), 0)
            self.assertIsNotNone(helper.get_excluded_cols())
            # assert entities
            all_entities = helper.load_entities()
            self.assertGreater(len(all_entities), 0)
            for e in all_entities:
                fs.register_entity(e)
            self.assertGreater(fs.list_entities().count(), 0)
            # assert feature view
            all_fvs = helper.load_draft_feature_views()
            self.assertGreater(len(all_fvs), 0)
            for fv in all_fvs:
                fs.register_feature_view(fv, version="1.0")
            self.assertGreater(fs.list_feature_views().count(), 0)


if __name__ == "__main__":
    absltest.main()
