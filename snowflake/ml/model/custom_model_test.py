import asyncio
import datetime

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, svm

from snowflake.ml.model import custom_model


class ModelTest(absltest.TestCase):
    def test_bad_custom_model_predict_type_incorrect(self) -> None:
        class BadModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input):  # type: ignore[no-untyped-def]
                return pd.DataFrame(input)

        arr = np.array([[2, 5], [6, 8]])
        d = pd.DataFrame(arr, columns=["c1", "c2"])
        with self.assertRaises(TypeError):
            _ = BadModel(custom_model.ModelContext())

        class GoodModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(input)

        good_model = GoodModel(custom_model.ModelContext())
        _ = good_model.predict(d)
        self.assertListEqual(list(good_model._get_infer_methods()), [GoodModel.predict])

        @custom_model.inference_api
        def bad_predict_1(self, input):  # type: ignore[no-untyped-def]
            return pd.DataFrame(input)

        with self.assertRaises(TypeError):
            good_model.predict = bad_predict_1  # type: ignore[method-assign,assignment]

        with self.assertRaises(TypeError):

            @custom_model.inference_api  # type: ignore[arg-type]
            def bad_predict_2(input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(input)

            bad_predict_2 = bad_predict_2.__get__(good_model, type(good_model))

            good_model.predict = bad_predict_2  # type: ignore[method-assign,assignment]

        class AnotherBadModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> int:
                return 42

        with self.assertRaises(TypeError):
            _ = AnotherBadModel(custom_model.ModelContext())

        class BadAsyncModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            async def predict(self, input: int) -> pd.DataFrame:
                await asyncio.sleep(0.1)
                return pd.DataFrame([input])

        async def _test(self: "ModelTest") -> None:
            with self.assertRaises(TypeError):
                bad_async_model = BadAsyncModel(custom_model.ModelContext())
                _ = await bad_async_model.predict(d)

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_custom_model_with_keyword_args(self) -> None:
        """Test that keyword-only arguments with defaults are allowed."""

        class ModelWithKwargs(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame, *, param1: int = 10, param2: str = "default") -> pd.DataFrame:
                return pd.DataFrame({"output": input["c1"] * param1, "param2": param2})

        model = ModelWithKwargs(custom_model.ModelContext())
        arr = np.array([[1, 2], [3, 4]])
        d = pd.DataFrame(arr, columns=["c1", "c2"])

        # Test with default kwargs
        res = model.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([10, 30]))))

        # Test with custom kwargs
        res = model.predict(d, param1=5)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([5, 15]))))

    def test_custom_model_keyword_args_validation(self) -> None:
        """Test that additional parameters must be keyword-only with defaults."""

        # Positional argument after input should fail
        with self.assertRaises(TypeError) as cm:

            class BadModelNoKeyword(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(self, input: pd.DataFrame, extra_param: int = 5) -> pd.DataFrame:
                    return pd.DataFrame(input)

            _ = BadModelNoKeyword(custom_model.ModelContext())

        self.assertIn("keyword-only", str(cm.exception))

        # Keyword-only argument without default should fail
        with self.assertRaises(TypeError) as cm:

            class BadModelNoDefault(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(self, input: pd.DataFrame, *, required_param: int) -> pd.DataFrame:
                    return pd.DataFrame(input)

            _ = BadModelNoDefault(custom_model.ModelContext())

        self.assertIn("default value", str(cm.exception))

        # Keyword-only argument without type annotation should fail
        with self.assertRaises(TypeError) as cm:

            class BadModelNoTypeAnnotation(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(  # type: ignore[no-untyped-def]
                    self, input: pd.DataFrame, *, required_param=5
                ) -> pd.DataFrame:
                    return pd.DataFrame(input)

            _ = BadModelNoTypeAnnotation(custom_model.ModelContext())

        self.assertIn("type annotation", str(cm.exception))

    def test_get_method_parameters(self) -> None:
        """Test that get_method_parameters extracts keyword-only parameters correctly."""

        class ModelWithParams(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame, *, threshold: float = 0.5, name: str = "test") -> pd.DataFrame:
                return pd.DataFrame(input)

        model = ModelWithParams(custom_model.ModelContext())

        # Get the method and extract parameters
        for method in model._get_infer_methods():
            params = custom_model.get_method_parameters(method)
            self.assertEqual(len(params), 2)
            self.assertEqual(params[0][0], "threshold")
            self.assertEqual(params[0][1], float)
            self.assertEqual(params[0][2], 0.5)
            self.assertEqual(params[1][0], "name")
            self.assertEqual(params[1][1], str)
            self.assertEqual(params[1][2], "test")

    def test_custom_model_type(self) -> None:
        class DemoModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": input["c1"]})

        lm = DemoModel(custom_model.ModelContext())
        self.assertListEqual(list(lm._get_infer_methods()), [DemoModel.predict])
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        res = lm.predict(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([1, 4]))))

        class AnotherDemoModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict_c1(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": input["c1"]})

            @custom_model.inference_api
            def predict_c2(self, input: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"output": input["c2"]})

        lm_2 = AnotherDemoModel(custom_model.ModelContext())
        self.assertListEqual(
            list(lm_2._get_infer_methods()),
            [AnotherDemoModel.predict_c1, AnotherDemoModel.predict_c2],
        )
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        res = lm_2.predict_c1(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([1, 4]))))
        res = lm_2.predict_c2(d)
        self.assertTrue(np.allclose(res["output"], pd.Series(np.array([2, 2]))))

    def test_custom_async_model_composition_type(self) -> None:
        class AsyncComposeModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)
                self.m1 = self.context["m1"]

            @custom_model.inference_api
            async def predict(self, input: pd.DataFrame) -> pd.DataFrame:
                res_sum = await self.m1.predict.async_run(input) + self.context.model_ref(  # type: ignore[union-attr]
                    "m2"
                ).predict(input)
                return pd.DataFrame({"output": res_sum / 2})

        async def _test(self: "ModelTest") -> None:
            digits = datasets.load_digits()
            clf = svm.SVC(gamma=0.001, C=100.0)
            clf.fit(digits.data[:-10], digits.target[:-10])
            model_context = custom_model.ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            self.assertListEqual(list(acm._get_infer_methods()), [AsyncComposeModel.predict])
            digits_df = pd.DataFrame(digits.data, columns=[f"col_{i}" for i in range(digits.data.shape[1])])
            p1 = clf.predict(digits_df[-10:])
            p2 = await acm.predict(digits_df[-10:])
            self.assertTrue(np.allclose(p1, p2["output"]))

        asyncio.get_event_loop().run_until_complete(_test(self))


class IsSupportedAnnotationTest(absltest.TestCase):
    """Test cases for _is_supported_annotation function."""

    def test_supported_scalar_types(self) -> None:
        """Test that all supported scalar types are accepted."""
        supported_scalars = [int, float, str, bool, bytes, datetime.datetime]
        for scalar_type in supported_scalars:
            with self.subTest(scalar_type=scalar_type):
                self.assertTrue(custom_model._is_supported_annotation(scalar_type))

    def test_bare_list_type_rejected(self) -> None:
        """Test that bare list type is rejected (element type required for serialization)."""
        self.assertFalse(custom_model._is_supported_annotation(list))

    def test_list_of_scalars(self) -> None:
        """Test that list of supported scalar types are accepted."""
        self.assertTrue(custom_model._is_supported_annotation(list[int]))
        self.assertTrue(custom_model._is_supported_annotation(list[float]))
        self.assertTrue(custom_model._is_supported_annotation(list[str]))
        self.assertTrue(custom_model._is_supported_annotation(list[bool]))
        self.assertTrue(custom_model._is_supported_annotation(list[bytes]))
        self.assertTrue(custom_model._is_supported_annotation(list[datetime.datetime]))

    def test_nested_lists(self) -> None:
        """Test that nested lists are accepted."""
        self.assertTrue(custom_model._is_supported_annotation(list[list[int]]))
        self.assertTrue(custom_model._is_supported_annotation(list[list[str]]))
        self.assertTrue(custom_model._is_supported_annotation(list[list[list[float]]]))

    def test_deeply_nested_lists(self) -> None:
        """Test that deeply nested lists are accepted."""
        self.assertTrue(custom_model._is_supported_annotation(list[list[list[list[int]]]]))

    def test_unsupported_scalar_types(self) -> None:
        """Test that unsupported scalar types are rejected."""

        class CustomClass:
            pass

        self.assertFalse(custom_model._is_supported_annotation(CustomClass))
        self.assertFalse(custom_model._is_supported_annotation(type(None)))

    def test_unsupported_container_types(self) -> None:
        """Test that unsupported container types are rejected."""
        self.assertFalse(custom_model._is_supported_annotation(dict))
        self.assertFalse(custom_model._is_supported_annotation(dict[str, int]))
        self.assertFalse(custom_model._is_supported_annotation(tuple))
        self.assertFalse(custom_model._is_supported_annotation(tuple[int, str]))
        self.assertFalse(custom_model._is_supported_annotation(set))
        self.assertFalse(custom_model._is_supported_annotation(set[int]))

    def test_list_of_unsupported_inner_type(self) -> None:
        """Test that list of unsupported inner types are rejected."""

        class CustomClass:
            pass

        self.assertFalse(custom_model._is_supported_annotation(list[CustomClass]))
        self.assertFalse(custom_model._is_supported_annotation(list[dict]))  # type: ignore[type-arg]
        self.assertFalse(custom_model._is_supported_annotation(list[dict[str, int]]))

    def test_nested_list_with_unsupported_innermost_type(self) -> None:
        """Test that nested lists with unsupported innermost types are rejected."""

        class CustomClass:
            pass

        self.assertFalse(custom_model._is_supported_annotation(list[list[CustomClass]]))
        self.assertFalse(custom_model._is_supported_annotation(list[list[list[dict]]]))  # type: ignore[type-arg]


class ValidateParameterListTypesTest(absltest.TestCase):
    """Test cases for _validate_parameter with list types."""

    def test_custom_model_with_list_parameter(self) -> None:
        """Test that list parameters are allowed in custom models."""

        class ModelWithListParam(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(
                self, input: pd.DataFrame, *, stop_words: list[str] = ["stop", "words"]  # noqa: B006
            ) -> pd.DataFrame:
                return pd.DataFrame(input)

        # Should not raise
        model = ModelWithListParam(custom_model.ModelContext())
        self.assertIsNotNone(model)

    def test_custom_model_with_nested_list_parameter(self) -> None:
        """Test that nested list parameters are allowed."""

        class ModelWithNestedListParam(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(
                self, input: pd.DataFrame, *, matrix: list[list[int]] = [[1, 2], [3, 4]]  # noqa: B006
            ) -> pd.DataFrame:
                return pd.DataFrame(input)

        # Should not raise
        model = ModelWithNestedListParam(custom_model.ModelContext())
        self.assertIsNotNone(model)

    def test_custom_model_with_deeply_nested_list_parameter(self) -> None:
        """Test that deeply nested list parameters are allowed."""

        class ModelWithDeeplyNestedListParam(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(
                self, input: pd.DataFrame, *, tensor: list[list[list[float]]] = [[[1.0]]]  # noqa: B006
            ) -> pd.DataFrame:
                return pd.DataFrame(input)

        # Should not raise
        model = ModelWithDeeplyNestedListParam(custom_model.ModelContext())
        self.assertIsNotNone(model)

    def test_custom_model_with_unsupported_list_inner_type_raises(self) -> None:
        """Test that list with unsupported inner type raises TypeError."""

        with self.assertRaises(TypeError) as cm:

            class ModelWithBadListParam(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(
                    self,
                    input: pd.DataFrame,
                    *,
                    bad_param: list[dict] = [{}],  # type: ignore[type-arg]  # noqa: B006
                ) -> pd.DataFrame:
                    return pd.DataFrame(input)

            _ = ModelWithBadListParam(custom_model.ModelContext())

        self.assertIn("unsupported type annotation", str(cm.exception))

    def test_custom_model_with_dict_parameter_raises(self) -> None:
        """Test that dict parameters raise TypeError."""

        with self.assertRaises(TypeError) as cm:

            class ModelWithDictParam(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(self, input: pd.DataFrame, *, config: dict[str, int] = {}) -> pd.DataFrame:  # noqa: B006
                    return pd.DataFrame(input)

            _ = ModelWithDictParam(custom_model.ModelContext())

        self.assertIn("unsupported type annotation", str(cm.exception))

    def test_custom_model_with_bare_list_parameter_raises(self) -> None:
        """Test that bare list parameter raises TypeError early with clear message."""

        with self.assertRaises(TypeError) as cm:

            class ModelWithBareListParam(custom_model.CustomModel):
                def __init__(self, context: custom_model.ModelContext) -> None:
                    super().__init__(context)

                @custom_model.inference_api
                def predict(
                    self,
                    input: pd.DataFrame,
                    *,
                    items: list = [],  # type: ignore[type-arg]  # noqa: B006
                ) -> pd.DataFrame:
                    return pd.DataFrame(input)

            _ = ModelWithBareListParam(custom_model.ModelContext())

        self.assertIn("unsupported type annotation", str(cm.exception))
        self.assertIn("bare 'list' without element type is not supported", str(cm.exception))

    def test_get_method_parameters_with_list_type(self) -> None:
        """Test that get_method_parameters works with list types."""

        class ModelWithListParams(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(
                self,
                input: pd.DataFrame,
                *,
                tokens: list[str] = ["a", "b"],  # noqa: B006
                weights: list[float] = [0.5, 0.5],  # noqa: B006
            ) -> pd.DataFrame:
                return pd.DataFrame(input)

        model = ModelWithListParams(custom_model.ModelContext())

        for method in model._get_infer_methods():
            params = custom_model.get_method_parameters(method)
            self.assertEqual(len(params), 2)
            self.assertEqual(params[0][0], "tokens")
            self.assertEqual(params[0][1], list[str])
            self.assertEqual(params[0][2], ["a", "b"])
            self.assertEqual(params[1][0], "weights")
            self.assertEqual(params[1][1], list[float])
            self.assertEqual(params[1][2], [0.5, 0.5])


if __name__ == "__main__":
    absltest.main()
