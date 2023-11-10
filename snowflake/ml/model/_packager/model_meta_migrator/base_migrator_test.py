from typing import Any, Dict

from absl.testing import absltest

from snowflake.ml._internal import migrator_utils
from snowflake.ml.model._packager.model_meta_migrator import base_migrator


class MetaMigrator_1(base_migrator.BaseModelMetaMigrator):
    source_version = "version_0"
    target_version = "version_1"

    @staticmethod
    def upgrade(original_meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return original_meta_dict


class MetaMigrator_2(base_migrator.BaseModelMetaMigrator):
    source_version = "version_1"
    target_version = "version_2"

    @staticmethod
    def upgrade(original_meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise migrator_utils.UnableToUpgradeError(last_supported_version="1.0.9")


class BaseMigratorTest(absltest.TestCase):
    def test_model_meta_dependencies_no_packages(self) -> None:
        bad_meta: Dict[str, Any] = {}
        migrator_1 = MetaMigrator_1()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Unknown or unsupported model metadata file with version .* found.",
        ):
            migrator_1.try_upgrade(bad_meta)

        good_meta = {"version": "version_0"}

        self.assertDictEqual(good_meta, migrator_1.try_upgrade(good_meta))
        self.assertIsNot(good_meta, migrator_1.try_upgrade(good_meta))

        migrator_2 = MetaMigrator_2()
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Can not upgrade your model metadata from version version_1 to version_2."
                "The latest version support the original version of Snowpark ML library is 1.0.9."
            ),
        ):
            migrator_2.try_upgrade({"version": "version_1"})


if __name__ == "__main__":
    absltest.main()
