import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import builtins_handler, core
from snowflake.ml.test_utils import exception_utils


class ListOfBuiltinsHandlerTest(absltest.TestCase):
    def test_validate_list_builtins(self) -> None:
        lt6 = ["Hello", [2, 3]]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Inconsistent type of object found in data"
        ):
            builtins_handler.ListOfBuiltinHandler.validate(lt6)  # type:ignore[arg-type]

        lt7 = [[1], [2, 3]]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Ill-shaped list data"
        ):
            builtins_handler.ListOfBuiltinHandler.validate(lt7)

        lt8 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        self.assertFalse(builtins_handler.ListOfBuiltinHandler.can_handle(lt8))

    def test_infer_signature_list_builtins(self) -> None:
        lt1 = [1, 2, 3, 4]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt1, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64)],
        )

        lt2 = ["a", "b", "c", "d"]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt2, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.STRING)],
        )

        lt3 = [ele.encode() for ele in lt2]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt3, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.BYTES)],
        )

        lt4 = [[1, 2], [3, 4]]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt4, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64),
                core.FeatureSpec("input_feature_1", core.DataType.INT64),
            ],
        )

        lt5 = [[1, 2.0], [3, 4]]  # This is not encouraged and will have type error, but we support it.
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt5, role="input"),  # type:ignore[arg-type]
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64),
                core.FeatureSpec("input_feature_1", core.DataType.DOUBLE),
            ],
        )

        lt6 = [[[1, 1], [2, 2]], [[3, 3], [4, 4]]]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt6, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64, shape=(2,)),
                core.FeatureSpec("input_feature_1", core.DataType.INT64, shape=(2,)),
            ],
        )


if __name__ == "__main__":
    absltest.main()
