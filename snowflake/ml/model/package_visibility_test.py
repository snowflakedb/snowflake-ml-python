from types import ModuleType

from absl.testing import absltest

from snowflake.ml import model
from snowflake.ml.model import custom_model, model_signature, type_hints


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def test_class_visible(self) -> None:
        self.assertIsInstance(model.Model, type)
        self.assertIsInstance(model.ModelVersion, type)
        self.assertIsInstance(model.HuggingFacePipelineModel, type)

    def test_module_visible(self) -> None:
        self.assertIsInstance(custom_model, ModuleType)
        self.assertIsInstance(model_signature, ModuleType)
        self.assertIsInstance(type_hints, ModuleType)


if __name__ == "__main__":
    absltest.main()
