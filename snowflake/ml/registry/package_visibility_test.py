from types import ModuleType

from absl.testing import absltest

from snowflake.ml import registry
from snowflake.ml.registry import artifact, model_registry


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def test_class_visible(self) -> None:
        self.assertIsInstance(registry.Registry, type)

    def test_module_visible(self) -> None:
        self.assertIsInstance(model_registry, ModuleType)
        self.assertIsInstance(artifact, ModuleType)


if __name__ == "__main__":
    absltest.main()
