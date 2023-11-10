from absl.testing import absltest


class GPUTest(absltest.TestCase):
    def test_gpu(self):
        import torch

        self.assertEqual(torch.cuda.is_available(), True)


if __name__ == "__main__":
    absltest.main()
