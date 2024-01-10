from types import ModuleType

from absl.testing import absltest

from snowflake.ml import model
from snowflake.ml.model import (
    _api,
    custom_model,
    deploy_platforms,
    model_signature,
    type_hints,
)


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def test_class_visible(self) -> None:
        self.assertIsInstance(model.Model, type)
        self.assertIsInstance(model.ModelVersion, type)
        self.assertIsInstance(model.HuggingFacePipelineModel, type)
        self.assertIsInstance(model.LLM, type)
        self.assertIsInstance(model.LLMOptions, type)

    def test_module_visible(self) -> None:
        self.assertIsInstance(_api, ModuleType)
        self.assertIsInstance(custom_model, ModuleType)
        self.assertIsInstance(model_signature, ModuleType)
        self.assertIsInstance(deploy_platforms, ModuleType)
        self.assertIsInstance(type_hints, ModuleType)


if __name__ == "__main__":
    absltest.main()
