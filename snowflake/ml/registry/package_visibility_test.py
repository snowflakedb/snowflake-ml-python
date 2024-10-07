from absl.testing import absltest

from snowflake.ml import registry


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def test_class_visible(self) -> None:
        self.assertIsInstance(registry.Registry, type)


if __name__ == "__main__":
    absltest.main()
