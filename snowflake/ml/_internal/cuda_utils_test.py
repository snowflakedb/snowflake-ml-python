from absl.testing import absltest
from packaging import version

from snowflake.ml._internal import cuda_utils


class CudaUtilsTest(absltest.TestCase):
    def test_validate_torch_config(self) -> None:
        # Check all torch, cuda versions
        for cfg in cuda_utils._TORCH_CUDA_COMPAT_CONFIGS:
            _ = version.Version(cfg.torch)
            for c in cfg.cudas:
                self.assertLessEqual(version.Version(c), version.Version(cuda_utils._SPCS_CUDA_VERSION))

    def test_is_torch_cuda_compatible(self) -> None:
        self.assertFalse(cuda_utils.is_torch_cuda_compatible("1.13", "12.1"))
        self.assertFalse(cuda_utils.is_torch_cuda_compatible("2.0.1", "12.1"))
        self.assertTrue(cuda_utils.is_torch_cuda_compatible("2.0.1.dev123", "11.7"))

    def test_get_latest_cuda_for_torch(self) -> None:
        self.assertEqual(cuda_utils.get_latest_cuda_for_torch("2.0.1"), "11.8")
        self.assertEqual(cuda_utils.get_latest_cuda_for_torch("10.0.1"), None)


if __name__ == "__main__":
    absltest.main()
