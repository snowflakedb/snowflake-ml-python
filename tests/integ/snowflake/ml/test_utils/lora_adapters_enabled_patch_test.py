from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities
from tests.integ.snowflake.ml.test_utils import lora_adapters_enabled_patch


class LoraAdaptersEnabledPatchTest(absltest.TestCase):
    def test_enable_is_idempotent(self) -> None:
        lora_adapters_enabled_patch.enable()
        lora_adapters_enabled_patch.enable()
        self.assertTrue(platform_capabilities.PlatformCapabilities(features={}).is_lora_adapters_enabled())

    def test_enable_preserves_method_signature(self) -> None:
        lora_adapters_enabled_patch.enable()
        with self.assertRaises(TypeError):
            platform_capabilities.PlatformCapabilities(features={}).is_lora_adapters_enabled(unexpected=True)


if __name__ == "__main__":
    absltest.main()
