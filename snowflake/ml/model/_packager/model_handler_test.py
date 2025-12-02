import datetime
from unittest import mock

from absl.testing import absltest

from snowflake.ml import version as snowml_version
from snowflake.ml.model._packager import model_handler
from snowflake.ml.model._packager.model_handlers import _base
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
                if handler._MIN_SNOWPARK_ML_VERSION != snowml_version.VERSION:
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

    def test_handler_can_handle_error_logging(self) -> None:
        """Test that errors in can_handle are caught and logged."""

        # Create a mock handler that raises an error in can_handle
        # Just used for testing, so ignoring all type errors
        class FailingTestHandler(_base.BaseModelHandler):  # type: ignore[type-arg]
            HANDLER_TYPE = "failing_test_handler"  # type: ignore[assignment]
            HANDLER_VERSION = "2024-01-01"
            _MIN_SNOWPARK_ML_VERSION = snowml_version.VERSION
            _HANDLER_MIGRATOR_PLANS = {}
            IS_AUTO_SIGNATURE = False

            @classmethod
            def can_handle(cls, model):  # type: ignore[no-untyped-def]
                raise RuntimeError("Test error in can_handle")

            @classmethod
            def save_model(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            @classmethod
            def load_model(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            @classmethod
            def convert_as_custom_model(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

        # Register the failing handler
        original_registry = model_handler._MODEL_HANDLER_REGISTRY.copy()
        model_handler._MODEL_HANDLER_REGISTRY[
            "failing_test_handler"
        ] = FailingTestHandler  # type: ignore[type-abstract]

        try:
            # Mock the logger to capture log messages
            with mock.patch("snowflake.ml.model._packager.model_handler.logger") as mock_logger:
                # Call find_handler with a test model
                test_model = "dummy_model"
                model_handler.find_handler(test_model)

                # Verify that logger.error was called with the expected message
                mock_logger.error.assert_called()
                call_args = mock_logger.error.call_args
                error_msg = call_args[0][0]

                # Check that the error message contains expected information
                self.assertIn("Error in", error_msg)
                self.assertIn("FailingTestHandler", error_msg)
                self.assertIn("can_handle", error_msg)
                self.assertIn("str", error_msg)  # type(model) is str

                # Verify exc_info=True was passed to log the traceback
                self.assertTrue(call_args[1].get("exc_info"))
        finally:
            # Restore original registry
            model_handler._MODEL_HANDLER_REGISTRY = original_registry


if __name__ == "__main__":
    absltest.main()
