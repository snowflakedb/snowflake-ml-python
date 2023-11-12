import datetime

from absl.testing import absltest

from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.ml.model._packager.model_meta_migrator import migrator_plans


class ModelMetaMigratorTest(absltest.TestCase):
    def test_registered_handler(self) -> None:
        all_source_versions = set()
        all_target_versions = set()
        for source_version, migrator_plan in migrator_plans.MODEL_META_MIGRATOR_PLANS.items():
            self.assertNotEqual(
                model_meta_schema.MODEL_METADATA_VERSION,
                source_version,
                "There shouldn't be a migrator whose source version is current handler version.",
            )
            self.assertEqual(
                source_version,
                migrator_plan.source_version,
                "There shouldn't be a migrator whose source version does not equal to the key in the plans.",
            )
            if source_version == "1":
                # Legacy check
                self.assertEqual(migrator_plan.target_version, "2023-12-01")
            else:
                self.assertLess(
                    datetime.datetime.strptime(migrator_plan.source_version, "%Y-%m-%d"),
                    datetime.datetime.strptime(migrator_plan.target_version, "%Y-%m-%d"),
                    "Migrator should not be able to downgrade.",
                )
            if migrator_plan.target_version != model_meta_schema.MODEL_METADATA_VERSION:
                self.assertIn(
                    migrator_plan.target_version,
                    migrator_plans.MODEL_META_MIGRATOR_PLANS.keys(),
                    (
                        "There shouldn't be a migrator whose target version "
                        "is not current version and has not a migrator plan"
                    ),
                )
            all_source_versions.add(migrator_plan.source_version)
            all_target_versions.add(migrator_plan.target_version)
        self.assertEqual(len(all_source_versions), len(all_target_versions), "The migrator plan is not monotonic.")


if __name__ == "__main__":
    absltest.main()
