import numpy as np
import pandas as pd
import torch
from absl.testing import absltest

from snowflake.ml.model._signatures import core, pytorch_handler, utils


class SeqOfPyTorchTensorHandlerTest(absltest.TestCase):
    def test_validate_list_of_pytorch_tensor(self) -> None:
        lt1 = [np.array([1, 4]), np.array([2, 3])]
        self.assertFalse(pytorch_handler.SeqOfPyTorchTensorHandler.can_handle(lt1))

        lt2 = [np.array([1, 4]), torch.Tensor([2, 3])]
        self.assertFalse(pytorch_handler.SeqOfPyTorchTensorHandler.can_handle(lt2))

        lt3 = [torch.Tensor([1, 4]), torch.Tensor([2, 3])]
        self.assertTrue(pytorch_handler.SeqOfPyTorchTensorHandler.can_handle(lt3))

    def test_validate_torch_tensor(self) -> None:
        t = [torch.Tensor([])]
        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            pytorch_handler.SeqOfPyTorchTensorHandler.validate(t)

        t = [torch.Tensor(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            pytorch_handler.SeqOfPyTorchTensorHandler.validate(t)

        t = [torch.Tensor([1, 2]), torch.Tensor(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            pytorch_handler.SeqOfPyTorchTensorHandler.validate(t)

    def test_trunc_torch_tensor(self) -> None:
        t = [torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))]

        for ts in pytorch_handler.SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))]

        for ts in pytorch_handler.SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)), ts
            )

        t = [torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for ts in pytorch_handler.SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [
            torch.Tensor([1]),
            torch.Tensor([1] * (pytorch_handler.SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for ts in pytorch_handler.SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1]), ts
            )

    def test_infer_schema_torch_tensor(self) -> None:
        t1 = [torch.IntTensor([1, 2, 3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t1, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT32)],
        )

        t2 = [torch.LongTensor([1, 2, 3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t2, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64)],
        )

        t3 = [torch.ShortTensor([1, 2, 3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t3, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT16)],
        )

        t4 = [torch.CharTensor([1, 2, 3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t4, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT8)],
        )

        t5 = [torch.ByteTensor([1, 2, 3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t5, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT8)],
        )

        t6 = [torch.BoolTensor([False, True])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t6, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.BOOL)],
        )

        t7 = [torch.FloatTensor([1.2, 3.4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t7, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.FLOAT)],
        )

        t8 = [torch.DoubleTensor([1.2, 3.4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t8, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.DOUBLE)],
        )

        t9 = [torch.LongTensor([[1, 2], [3, 4]])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t9, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64, shape=(2,)),
            ],
        )

        t10 = [torch.LongTensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t10, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64, shape=(2, 2))],
        )

        t11 = [torch.LongTensor([1, 2, 3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t11, role="output"),
            [core.FeatureSpec("output_feature_0", core.DataType.INT64)],
        )

        t12 = [torch.LongTensor([1, 2]), torch.LongTensor([3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t12, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64),
                core.FeatureSpec("output_feature_1", core.DataType.INT64),
            ],
        )

        t13 = [torch.FloatTensor([1.2, 2.4]), torch.LongTensor([3, 4])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t13, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.FLOAT),
                core.FeatureSpec("output_feature_1", core.DataType.INT64),
            ],
        )

        t14 = [torch.LongTensor([[1, 1], [2, 2]]), torch.LongTensor([[3, 3], [4, 4]])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t14, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64, shape=(2,)),
                core.FeatureSpec("output_feature_1", core.DataType.INT64, shape=(2,)),
            ],
        )

        t15 = [torch.LongTensor([[1, 1], [2, 2]]), torch.DoubleTensor([[1.5, 6.8], [2.9, 9.2]])]
        self.assertListEqual(
            pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t15, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64, shape=(2,)),
                core.FeatureSpec("output_feature_1", core.DataType.DOUBLE, shape=(2,)),
            ],
        )

    def test_convert_to_df_torch_tensor(self) -> None:
        t1 = [torch.LongTensor([1, 2, 3, 4])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t1),
            pd.DataFrame([1, 2, 3, 4]),
        )

        t2 = [torch.DoubleTensor([1, 2, 3, 4])]
        t2[0].requires_grad = True
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t2),
            pd.DataFrame([1, 2, 3, 4], dtype=np.double),
        )

        t3 = [torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t3),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]}),
        )

        t4 = [torch.LongTensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t4),
            pd.DataFrame(data={0: [np.array([[1, 1], [2, 2]]), np.array([[3, 3], [4, 4]])]}),
        )

        t5 = [torch.LongTensor([1, 2]), torch.LongTensor([3, 4])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t5),
            pd.DataFrame([[1, 3], [2, 4]]),
        )

        t6 = [torch.DoubleTensor([1.2, 2.4]), torch.LongTensor([3, 4])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t6),
            pd.DataFrame([[1.2, 3], [2.4, 4]]),
        )

        t7 = [torch.LongTensor([[1, 1], [2, 2]]), torch.LongTensor([[3, 3], [4, 4]])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t7),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([3, 3]), np.array([4, 4])]}),
        )

        t8 = [torch.LongTensor([[1, 1], [2, 2]]), torch.DoubleTensor([[1.5, 6.8], [2.9, 9.2]])]
        pd.testing.assert_frame_equal(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t8),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([1.5, 6.8]), np.array([2.9, 9.2])]}),
        )

    def test_convert_from_df_torch_tensor(self) -> None:
        t1 = [torch.LongTensor([1, 2, 3, 4])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t1)
            )
        ):
            torch.testing.assert_close(t, t1[idx])  # type:ignore[attr-defined]

        t2 = [torch.DoubleTensor([1, 2, 3, 4])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t2)
            )
        ):
            torch.testing.assert_close(t, t2[idx])  # type:ignore[attr-defined]

        t3 = [torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t3)
            )
        ):
            torch.testing.assert_close(t, t3[idx])  # type:ignore[attr-defined]

        t4 = [torch.LongTensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t4)
            )
        ):
            torch.testing.assert_close(t, t4[idx])  # type:ignore[attr-defined]

        t5 = [torch.LongTensor([1, 2]), torch.LongTensor([3, 4])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t5)
            )
        ):
            torch.testing.assert_close(t, t5[idx])  # type:ignore[attr-defined]

        t6 = [torch.DoubleTensor([1.2, 2.4]), torch.LongTensor([3, 4])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t6)
            )
        ):
            torch.testing.assert_close(t, t6[idx])  # type:ignore[attr-defined]

        t7 = [torch.LongTensor([[1, 1], [2, 2]]), torch.LongTensor([[3, 3], [4, 4]])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t7)
            )
        ):
            torch.testing.assert_close(t, t7[idx])  # type:ignore[attr-defined]

        t8 = [torch.LongTensor([[1, 1], [2, 2]]), torch.DoubleTensor([[1.5, 6.8], [2.9, 9.2]])]
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t8)
            )
        ):
            torch.testing.assert_close(t, t8[idx])  # type:ignore[attr-defined]

        t9 = [torch.IntTensor([1, 2, 3, 4])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t9, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t9), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t9[idx])  # type:ignore[attr-defined]

        t10 = [torch.tensor([1.2, 3.4])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t10, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t10), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t10[idx])  # type:ignore[attr-defined]

        t11 = [torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t11, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t11), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t11[idx])  # type:ignore[attr-defined]

        t12 = [torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t12, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t12), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t12[idx])  # type:ignore[attr-defined]

        t13 = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t13, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t13), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t13[idx])  # type:ignore[attr-defined]

        t14 = [torch.tensor([1.2, 2.4]), torch.tensor([3, 4])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t14, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t14), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t14[idx])  # type:ignore[attr-defined]

        t15 = [torch.tensor([[1, 1], [2, 2]]), torch.tensor([[3, 3], [4, 4]])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t15, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t15), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t15[idx])  # type:ignore[attr-defined]

        t16 = [torch.tensor([[1, 1], [2, 2]]), torch.tensor([[1.5, 6.8], [2.9, 9.2]])]
        fts = pytorch_handler.SeqOfPyTorchTensorHandler.infer_signature(t16, role="input")
        for idx, t in enumerate(
            pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                utils.rename_pandas_df(pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(t16), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t16[idx])  # type:ignore[attr-defined]


if __name__ == "__main__":
    absltest.main()
