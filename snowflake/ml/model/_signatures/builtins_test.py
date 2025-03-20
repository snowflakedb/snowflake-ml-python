import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import builtins_handler, core
from snowflake.ml.test_utils import exception_utils


class ListOfBuiltinsHandlerTest(absltest.TestCase):
    def test_can_handle_list_builtins(self) -> None:
        lt1 = [(2, 3), [2, 3]]
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt1))

        lt2 = (2, 3)
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt2))

        lt3 = ([3, 3], 3)
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt3))

        lt4 = ({"a": 1}, 3)
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt4))

        lt5 = [({"a": 1}, 3)]
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt5))

        lt6 = "abcd"
        self.assertFalse(builtins_handler.ListOfBuiltinHandler.can_handle(lt6))

        lt7 = ["abcd", "abcd"]
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt7))

        lt8 = [("ab", "ab"), "ab"]
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt8))

        lt9 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        self.assertFalse(builtins_handler.ListOfBuiltinHandler.can_handle(lt9))

        lt10 = [{"answer": "A.I.Channel", "coordinates": [(0, 0)], "cells": ["A.I.Channel"], "aggregator": "NONE"}]
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt10))

        lt11 = [
            [
                {"entity": "I-MISC", "score": 0.11474538, "index": 1, "word": "my", "start": 0, "end": 2},
                {"entity": "I-MISC", "score": 0.11474538, "index": 2, "word": "name", "start": 3, "end": 7},
                {"entity": "I-MISC", "score": 0.11474538, "index": 10, "word": "in", "start": 28, "end": 30},
                {"entity": "I-MISC", "score": 0.11474538, "index": 16, "word": "##apa", "start": 39, "end": 42},
                {"entity": "I-MISC", "score": 0.11474538, "index": 17, "word": "##n", "start": 42, "end": 43},
            ]
        ]
        self.assertTrue(builtins_handler.ListOfBuiltinHandler.can_handle(lt11))

    def test_validate_list_builtins(self) -> None:
        lt1 = ["Hello", [2, 3]]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Inconsistent type of object found in data"
        ):
            builtins_handler.ListOfBuiltinHandler.validate(lt1)  # type:ignore[arg-type]

        lt2 = [[1], [2, 3]]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Ill-shaped list data"
        ):
            builtins_handler.ListOfBuiltinHandler.validate(lt2)

        lt3 = [("ab", "ab"), "ab"]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Inconsistent type of object found in data"
        ):
            builtins_handler.ListOfBuiltinHandler.validate(lt3)

        lt4 = [{"a": 1}, {"a": 1}]
        builtins_handler.ListOfBuiltinHandler.validate(lt4)  # type:ignore[arg-type]

        lt5 = [[{"a": 1}], [{"a": 1}]]
        builtins_handler.ListOfBuiltinHandler.validate(lt5)  # type:ignore[arg-type]

        lt6 = [({"a": 1}, 3)]
        builtins_handler.ListOfBuiltinHandler.validate(lt6)  # type:ignore[arg-type]

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

        lt7 = [[{"a": 1, "b": 2}], [{"a": 3, "b": 4}]]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt7, role="input"),  # type:ignore[arg-type]
            [
                core.FeatureGroupSpec(
                    "input_feature_0",
                    [core.FeatureSpec("a", core.DataType.INT64), core.FeatureSpec("b", core.DataType.INT64)],
                )
            ],
        )

        lt8 = [[[{"a": 1, "b": 2}]], [[{"a": 3, "b": 4}]]]
        self.assertListEqual(
            builtins_handler.ListOfBuiltinHandler.infer_signature(lt8, role="input"),  # type:ignore[arg-type]
            [
                core.FeatureGroupSpec(
                    "input_feature_0",
                    [core.FeatureSpec("a", core.DataType.INT64), core.FeatureSpec("b", core.DataType.INT64)],
                    shape=(-1,),
                )
            ],
        )


if __name__ == "__main__":
    absltest.main()
