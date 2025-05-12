from absl.testing import absltest

from snowflake.ml._internal.utils import identifier
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import common_test_base, model_factory

MODEL_NAME = "TEST_MODEL"
NEW_MODEL_NAME = "NEW_MODEL"
VERSION_NAME = "V1"
ADDED_VERSION_NAME = "V2"


class RegistryInSprocTest(registry_model_test_base.RegistryModelTestBase):
    @common_test_base.CommonTestBase.sproc_test(
        test_owners_rights=False, additional_packages=["inflection", "scikit-learn==1.5.1"]
    )
    def test_workflow(self) -> None:
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self.mv_1 = self.registry.log_model(
            model=model,
            model_name=MODEL_NAME,
            version_name=VERSION_NAME,
            sample_input_data=test_features,
            options={"embed_local_ml_library": True},
        )

        self.mv_2 = self.registry.log_model(
            model=model,
            model_name=MODEL_NAME,
            version_name=ADDED_VERSION_NAME,
            sample_input_data=test_features,
            options={"embed_local_ml_library": True},
        )

        self.model = self.registry.get_model(MODEL_NAME)
        self.assertLen(self.registry.show_models(), 1)
        self.assertEqual(self.model.versions(), [self.mv_1, self.mv_2])
        self.assertLen(self.model.show_versions(), 2)

        description = "test description"
        self.mv_1.description = description
        self.assertEqual(self.mv_1.description, description)

        self.mv_1.set_metric("a", 1)
        expected_metrics = {"a": 2, "b": 1.0, "c": True}
        for k, v in expected_metrics.items():
            self.mv_1.set_metric(k, v)

        self.assertEqual(self.mv_1.get_metric("a"), expected_metrics["a"])
        self.assertDictEqual(self.mv_1.show_metrics(), expected_metrics)

        expected_metrics.pop("b")
        self.mv_1.delete_metric("b")
        self.assertDictEqual(self.mv_1.show_metrics(), expected_metrics)
        with self.assertRaises(KeyError):
            self.mv_1.get_metric("b")

        description = "test description"
        self.model.description = description
        self.assertEqual(self.model.description, description)

        self.assertEqual(self.model.default.version_name, VERSION_NAME)

        self.model.default = ADDED_VERSION_NAME
        self.assertEqual(self.model.default.version_name, ADDED_VERSION_NAME)

        self.model.delete_version(VERSION_NAME)
        self.assertLen(self.model.show_versions(), 1)

        self._tag_name1 = "MYTAG"
        self._tag_name2 = '"live_version"'

        self.session.sql(f"CREATE TAG {self._tag_name1}").collect()
        self.session.sql(f"CREATE TAG {self._tag_name2}").collect()

        fq_tag_name1 = identifier.get_schema_level_object_identifier(self._test_db, self._test_schema, self._tag_name1)
        fq_tag_name2 = identifier.get_schema_level_object_identifier(self._test_db, self._test_schema, self._tag_name2)

        self.assertDictEqual({}, self.model.show_tags())
        self.assertIsNone(self.model.get_tag(self._tag_name1))
        self.model.set_tag(self._tag_name1, "val1")
        self.assertEqual(
            "val1",
            self.model.get_tag(fq_tag_name1),
        )
        self.assertDictEqual(
            {fq_tag_name1: "val1"},
            self.model.show_tags(),
        )
        self.model.set_tag(fq_tag_name2, "v2")
        self.assertEqual("v2", self.model.get_tag(self._tag_name2))
        self.assertDictEqual(
            {
                fq_tag_name1: "val1",
                fq_tag_name2: "v2",
            },
            self.model.show_tags(),
        )
        self.model.unset_tag(fq_tag_name2)
        self.assertDictEqual(
            {fq_tag_name1: "val1"},
            self.model.show_tags(),
        )
        self.model.unset_tag(self._tag_name1)
        self.assertDictEqual({}, self.model.show_tags())

        self.model.rename(NEW_MODEL_NAME)
        self.assertEqual(self.model.name, NEW_MODEL_NAME)
        self.registry.delete_model(NEW_MODEL_NAME)

        self.assertLen(self.registry.show_models(), 0)


if __name__ == "__main__":
    absltest.main()
