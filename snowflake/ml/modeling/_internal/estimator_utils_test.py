from typing import Any, Optional

from absl.testing import absltest, parameterized
from sklearn.linear_model import LinearRegression as SkLinearRegression

from snowflake import snowpark
from snowflake.ml._internal.exceptions.exceptions import SnowflakeMLException
from snowflake.ml.modeling._internal.estimator_utils import (
    gather_dependencies,
    original_estimator_has_callable,
    transform_snowml_obj_to_sklearn_obj,
    validate_sklearn_args,
)
from snowflake.ml.modeling.framework.base import BaseTransformer


class TestEstimator(BaseTransformer):
    def __init__(
        self,
        estimator: Any,
        dependencies: Optional[list[str]] = None,
        drop_input_cols: Optional[bool] = False,
        file_names: Optional[list[str]] = None,
        custom_states: Optional[list[str]] = None,
        sample_weight_col: Optional[str] = None,
    ) -> None:
        super().__init__(
            drop_input_cols=drop_input_cols,
            file_names=file_names,
            custom_states=custom_states,
            sample_weight_col=sample_weight_col,
        )
        self._sklearn_object = estimator
        self._deps = dependencies

    def _create_sklearn_object(self) -> Any:
        return self._sklearn_object

    def _create_unfitted_sklearn_object(self) -> Any:
        return self._sklearn_object

    def _get_dependencies(self) -> list[str]:
        return self._deps or []

    def _fit(self, dataset: snowpark.DataFrame) -> "BaseTransformer":
        return self


class EstimatorsUtilsTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            ({"fit_intercept": [True, True, False]}, True),
            ({"fake_param": [None, None, False]}, True),
            ({"fake_param": [True, False, False]}, False),
        ]
    )  # type: ignore[misc]
    def test_validate_sklearn_args(self, param_dict: dict[str, tuple[Any, Any, bool]], is_valid: bool) -> None:
        if not is_valid:
            with self.assertRaises(SnowflakeMLException):
                validate_sklearn_args(param_dict, SkLinearRegression)
        else:
            validate_sklearn_args(param_dict, SkLinearRegression)

    def test_transform_snowml_obj_to_sklearn_obj(self) -> None:
        sk_estimator = SkLinearRegression()
        estimator = TestEstimator(estimator=sk_estimator)
        with self.subTest("Test transformer directly"):
            sk_transform = transform_snowml_obj_to_sklearn_obj(estimator)
            self.assertEqual(sk_transform, sk_estimator)

        with self.subTest("Test transformer in list"):
            sk_transform = transform_snowml_obj_to_sklearn_obj([estimator])
            self.assertEqual(sk_transform[0], sk_estimator)

        with self.subTest("Test transformer in tuple"):
            sk_transform = transform_snowml_obj_to_sklearn_obj((estimator,))
            self.assertEqual(sk_transform[0], sk_estimator)

    def test_original_estimator_has_callable(self) -> None:
        sk_estimator = SkLinearRegression()
        estimator = TestEstimator(estimator=sk_estimator)
        check = original_estimator_has_callable("fit")
        self.assertTrue(check(estimator))

    def test_gather_dependencies(self) -> None:
        sk_estimator = SkLinearRegression()
        estimator_1 = TestEstimator(estimator=sk_estimator, dependencies=["dep-1"])
        estimator_2 = TestEstimator(estimator=sk_estimator, dependencies=["dep-2", "dep-3"])

        deps = gather_dependencies([estimator_1, estimator_2])
        self.assertEqual(deps, {"dep-1", "dep-2", "dep-3"})


if __name__ == "__main__":
    absltest.main()
