import datetime

from absl.testing import absltest

from snowflake.ml._internal import env as snowml_env
from snowflake.ml.model._packager import model_handler
from snowflake.ml.test_utils import test_env_utils


class ModelHandlerTest(absltest.TestCase):
    def test_registered_handler(self) -> None:
        model_handler._register_handlers()
        self.assertGreater(len(model_handler._MODEL_HANDLER_REGISTRY), 0, "No model handlers are registered.")
        for handler_name, handler in model_handler._MODEL_HANDLER_REGISTRY.items():
            with self.subTest(f"Testing Handler for {handler_name}"):
                # Validate name
                self.assertEqual(handler_name, handler.HANDLER_TYPE)
                # Validate version
                datetime.datetime.strptime(handler.HANDLER_VERSION, "%Y-%m-%d")
                # Validate min snowpark ml version
                if handler._MIN_SNOWPARK_ML_VERSION != snowml_env.VERSION:
                    self.assertIn(
                        handler._MIN_SNOWPARK_ML_VERSION,
                        test_env_utils.get_snowpark_ml_released_versions(),
                        "The min Snowpark ML version is not released or not current.",
                    )
                all_source_versions = set()
                all_target_versions = set()
                for source_version, migrator_plan in handler._HANDLER_MIGRATOR_PLANS.items():
                    self.assertNotEqual(
                        handler.HANDLER_VERSION,
                        source_version,
                        "There shouldn't be a migrator whose source version is current handler version.",
                    )
                    self.assertEqual(
                        source_version,
                        migrator_plan.source_version,
                        "There shouldn't be a migrator whose source version does not equal to the key in the plans.",
                    )
                    self.assertLess(
                        datetime.datetime.strptime(migrator_plan.source_version, "%Y-%m-%d"),
                        datetime.datetime.strptime(migrator_plan.target_version, "%Y-%m-%d"),
                        "Migrator should not be able to downgrade.",
                    )
                    if migrator_plan.target_version != handler.HANDLER_VERSION:
                        self.assertIn(
                            migrator_plan.target_version,
                            handler._HANDLER_MIGRATOR_PLANS.keys(),
                            (
                                "There shouldn't be a migrator whose target version "
                                "is not current version and has not a migrator plan"
                            ),
                        )
                    all_source_versions.add(migrator_plan.source_version)
                    all_target_versions.add(migrator_plan.target_version)
                self.assertEqual(
                    len(all_source_versions), len(all_target_versions), "The migrator plan is not monotonic."
                )


if __name__ == "__main__":
    absltest.main()
