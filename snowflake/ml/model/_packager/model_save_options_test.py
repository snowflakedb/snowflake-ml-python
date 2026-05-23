import re
from typing import Any, NotRequired, TypedDict, get_args

from absl.testing import absltest

from snowflake.ml._internal.exceptions import error_codes
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._packager import model_save_options
from snowflake.ml.test_utils import exception_utils


class ModelSaveOptionsTest(absltest.TestCase):
    def test_typed_dict_annotation_keys_returns_keys_from_single_typed_dict(self) -> None:
        keys = model_save_options._typed_dict_annotation_keys(model_types.ModelMethodSaveOptions)
        self.assertEqual(
            keys,
            frozenset({"case_sensitive", "function_type", "max_batch_size", "volatility"}),
        )

    def test_typed_dict_annotation_keys_includes_keys_from_typed_dict_bases(self) -> None:
        class _ParentOption(TypedDict):
            parent_key: NotRequired[str]

        class _ChildOption(_ParentOption):
            child_key: NotRequired[str]

        keys = model_save_options._typed_dict_annotation_keys(_ChildOption)
        self.assertEqual(keys, frozenset({"parent_key", "child_key"}))

    def test_typed_dict_annotation_keys_sklearn_includes_base_and_handler_keys(self) -> None:
        all_keys = model_save_options._typed_dict_annotation_keys(model_types.SKLModelSaveOptions)
        base_keys = model_save_options._typed_dict_annotation_keys(model_types.BaseModelSaveOption)
        self.assertTrue(base_keys.issubset(all_keys))
        self.assertIn("target_methods", all_keys - base_keys)

    def test_validate_method_options_skips_when_absent(self) -> None:
        model_save_options._validate_method_options(options={}, include_internal_option_keys=True)

    def test_validate_method_options_accepts_valid_options(self) -> None:
        model_save_options._validate_method_options(
            options={
                "method_options": {
                    "predict": model_types.ModelMethodSaveOptions(function_type="FUNCTION"),
                }
            },
            include_internal_option_keys=True,
        )

    def test_validate_method_options_rejects_non_dict_method_options(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=TypeError,
            expected_regex=r"method_options must be a dictionary",
        ):
            invalid_options: dict[str, Any] = {"method_options": []}
            model_save_options._validate_method_options(
                options=invalid_options,
                include_internal_option_keys=True,
            )

    def test_validate_method_options_rejects_non_dict_per_method_entry(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=TypeError,
            expected_regex=r"each method_options value must be a dictionary",
        ):
            invalid_options: dict[str, Any] = {"method_options": {"predict": "not-a-dict"}}
            model_save_options._validate_method_options(
                options=invalid_options,
                include_internal_option_keys=True,
            )

    def test_validate_method_options_rejects_unknown_keys_and_lists_allowed_keys(self) -> None:
        allowed = ", ".join(sorted(model_save_options._MODEL_METHOD_SAVE_OPTION_KEYS))
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=(
                r"unrecognized method_options keys.*method 'predict'.*not_a_method_option.*"
                rf"Supported per-method keys are {re.escape(allowed)}"
            ),
        ):
            model_save_options._validate_method_options(
                options={"method_options": {"predict": {"not_a_method_option": True}}},
                include_internal_option_keys=True,
            )

    def test_validate_model_save_option_keys_accepts_empty_options(self) -> None:
        model_save_options.validate_model_save_option_keys(handler_type="sklearn", options={})

    def test_validate_model_save_option_keys_accepts_base_and_handler_specific_keys(self) -> None:
        model_save_options.validate_model_save_option_keys(
            handler_type="sklearn",
            options={
                "relax_version": True,
                "target_methods": ["predict"],
            },
        )

    def test_validate_model_save_option_keys_rejects_unknown_top_level_key(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"unrecognized option keys.*not_a_real_save_option",
        ):
            model_save_options.validate_model_save_option_keys(
                handler_type="sklearn",
                options={"not_a_real_save_option": True},
            )

    def test_validate_model_save_option_keys_rejects_cross_framework_handler_key(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"unrecognized option keys.*model_uri",
        ):
            model_save_options.validate_model_save_option_keys(
                handler_type="sklearn",
                options={"model_uri": "runs:/foo/bar"},
            )

    def test_validate_model_save_option_keys_rejects_invalid_method_options(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"unrecognized method_options keys.*not_a_method_option",
        ):
            model_save_options.validate_model_save_option_keys(
                handler_type="sklearn",
                options={"method_options": {"predict": {"not_a_method_option": True}}},
            )

    def test_handler_option_dict_matches_supported_handler_types(self) -> None:
        expected = set(get_args(model_types.SupportedModelHandlerType))
        actual = set(model_save_options._HANDLER_OPTION_TYPED_DICT)
        self.assertEqual(
            actual,
            expected,
            msg=(
                "model_save_options._HANDLER_OPTION_TYPED_DICT keys must match "
                "SupportedModelHandlerType; update both when adding a handler."
            ),
        )


if __name__ == "__main__":
    absltest.main()
