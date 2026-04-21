"""Tests for param_ops shared module.

Uses SimpleNamespace to create mock spec objects -- no dependency on core.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
from absl.testing import absltest

from snowflake.ml.model._client.ops import param_ops


def _make_leaf_spec(name: str, shape: Any = None, numpy_type: Any = np.object_) -> SimpleNamespace:
    return SimpleNamespace(name=name, dtype=SimpleNamespace(_numpy_type=numpy_type), shape=shape)


def _make_group_spec(
    name: str,
    specs: list[SimpleNamespace],
    shape: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(name=name, specs=specs, shape=shape, dtype=SimpleNamespace(_numpy_type=np.object_))


class IsGroupSpecTest(absltest.TestCase):
    def test_leaf_spec(self) -> None:
        spec = _make_leaf_spec("temperature")
        self.assertFalse(param_ops.is_group_spec(spec))

    def test_group_spec(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        self.assertTrue(param_ops.is_group_spec(spec))

    def test_group_with_empty_specs(self) -> None:
        spec = _make_group_spec("config", [])
        self.assertTrue(param_ops.is_group_spec(spec))

    def test_no_specs_attribute(self) -> None:
        spec = SimpleNamespace(name="x")
        self.assertFalse(param_ops.is_group_spec(spec))

    def test_specs_not_a_list(self) -> None:
        spec = SimpleNamespace(name="x", specs="not_a_list")
        self.assertFalse(param_ops.is_group_spec(spec))


class CoerceParamValueTest(absltest.TestCase):
    def test_int_to_float_for_float32(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float32)
        self.assertEqual(param_ops.coerce_param_value(spec, 1), 1.0)
        self.assertIsInstance(param_ops.coerce_param_value(spec, 1), float)

    def test_int_to_float_for_float64(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        self.assertEqual(param_ops.coerce_param_value(spec, 1), 1.0)

    def test_no_coerce_for_int_dtype(self) -> None:
        spec = _make_leaf_spec("top_k", numpy_type=np.int64)
        self.assertEqual(param_ops.coerce_param_value(spec, 5), 5)
        self.assertIsInstance(param_ops.coerce_param_value(spec, 5), int)

    def test_no_coerce_for_string_dtype(self) -> None:
        spec = _make_leaf_spec("mode", numpy_type=np.object_)
        self.assertEqual(param_ops.coerce_param_value(spec, "greedy"), "greedy")

    def test_none_passthrough(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        self.assertIsNone(param_ops.coerce_param_value(spec, None))

    def test_float_stays_float(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        self.assertEqual(param_ops.coerce_param_value(spec, 0.5), 0.5)

    def test_nested_list_coercion(self) -> None:
        spec = _make_leaf_spec("weights", numpy_type=np.float32)
        result = param_ops.coerce_param_value(spec, [1, 2, 3])
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_bool_not_coerced(self) -> None:
        spec = _make_leaf_spec("flag", numpy_type=np.float64)
        self.assertIs(param_ops.coerce_param_value(spec, True), True)

    def test_numpy_integer_coerced(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        self.assertIsInstance(param_ops.coerce_param_value(spec, np.int64(1)), float)


class ValidateLeafParamValueTest(absltest.TestCase):
    def test_valid_scalar(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        param_ops.validate_leaf_param_value(spec, 0.5, "temperature")

    def test_none_is_valid(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        param_ops.validate_leaf_param_value(spec, None, "temperature")

    def test_scalar_gets_array_raises(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, [0.5, 0.6], "temperature")
        self.assertIn("expected scalar value", str(ctx.exception))

    def test_array_correct_shape(self) -> None:
        spec = _make_leaf_spec("weights", numpy_type=np.float64, shape=(3,))
        param_ops.validate_leaf_param_value(spec, [0.1, 0.2, 0.3], "weights")

    def test_array_wrong_length(self) -> None:
        spec = _make_leaf_spec("weights", numpy_type=np.float64, shape=(3,))
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, [0.1, 0.2], "weights")
        self.assertIn("dimension 0", str(ctx.exception))

    def test_array_variable_length(self) -> None:
        spec = _make_leaf_spec("weights", numpy_type=np.float64, shape=(-1,))
        param_ops.validate_leaf_param_value(spec, [0.1, 0.2, 0.3, 0.4], "weights")

    def test_2d_array_correct(self) -> None:
        spec = _make_leaf_spec("matrix", numpy_type=np.float64, shape=(2, 3))
        param_ops.validate_leaf_param_value(spec, [[1, 2, 3], [4, 5, 6]], "matrix")

    def test_2d_array_wrong_ndim(self) -> None:
        spec = _make_leaf_spec("matrix", numpy_type=np.float64, shape=(2, 3))
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, [1, 2, 3], "matrix")
        self.assertIn("1-dimensional", str(ctx.exception))

    def test_incompatible_type(self) -> None:
        spec = _make_leaf_spec("count", numpy_type=np.int64)
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, "not_a_number", "count")
        self.assertIn("not compatible", str(ctx.exception))

    def test_int_value_for_float_dtype(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        param_ops.validate_leaf_param_value(spec, 1, "temperature")

    def test_string_for_int_dtype_raises(self) -> None:
        spec = _make_leaf_spec("count", numpy_type=np.int64)
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, "hello", "count")
        self.assertIn("Expected int", str(ctx.exception))

    def test_string_for_float_dtype_raises(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, "hello", "temperature")
        self.assertIn("Expected int or float", str(ctx.exception))

    def test_bool_for_int_dtype_raises(self) -> None:
        spec = _make_leaf_spec("count", numpy_type=np.int64)
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_leaf_param_value(spec, True, "count")
        self.assertIn("Expected int", str(ctx.exception))


class DeepMergeParamGroupTest(absltest.TestCase):
    def test_override_scalar(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature"), _make_leaf_spec("top_k")])
        result = param_ops.deep_merge_param_group(spec, {"temperature": 0.7, "top_k": 50}, {"temperature": 0.9})
        self.assertEqual(result, {"temperature": 0.9, "top_k": 50})

    def test_defaults_only(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        result = param_ops.deep_merge_param_group(spec, {"temperature": 0.7}, {})
        self.assertEqual(result, {"temperature": 0.7})

    def test_override_only(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        result = param_ops.deep_merge_param_group(spec, {}, {"temperature": 0.9})
        self.assertEqual(result, {"temperature": 0.9})

    def test_case_insensitive_keys(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        result = param_ops.deep_merge_param_group(spec, {"temperature": 0.7}, {"TEMPERATURE": 0.9})
        self.assertEqual(result, {"temperature": 0.9})

    def test_nested_group_merge(self) -> None:
        inner_spec = _make_group_spec("sampling", [_make_leaf_spec("temperature"), _make_leaf_spec("top_k")])
        outer_spec = _make_group_spec("config", [inner_spec, _make_leaf_spec("max_tokens")])
        default = {"sampling": {"temperature": 0.7, "top_k": 50}, "max_tokens": 100}
        override = {"sampling": {"temperature": 0.9}}
        result = param_ops.deep_merge_param_group(outer_spec, default, override)
        self.assertEqual(result, {"sampling": {"temperature": 0.9, "top_k": 50}, "max_tokens": 100})

    def test_non_dict_override_for_group_replaces(self) -> None:
        inner_spec = _make_group_spec("sampling", [_make_leaf_spec("temperature")])
        outer_spec = _make_group_spec("config", [inner_spec])
        result = param_ops.deep_merge_param_group(
            outer_spec, {"sampling": {"temperature": 0.7}}, {"sampling": "use_defaults"}
        )
        self.assertEqual(result, {"sampling": "use_defaults"})

    def test_missing_default_key(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature"), _make_leaf_spec("top_k")])
        result = param_ops.deep_merge_param_group(spec, {}, {"temperature": 0.9})
        self.assertEqual(result, {"temperature": 0.9, "top_k": None})


class ValidateParamGroupDictTest(absltest.TestCase):
    def test_valid_dict(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        param_ops._validate_param_group_dict(spec, {"temperature": 0.7}, "config")

    def test_not_a_dict(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        with self.assertRaises(ValueError) as ctx:
            param_ops._validate_param_group_dict(spec, "not_a_dict", "config")
        self.assertIn("expected a dict", str(ctx.exception))

    def test_non_string_key(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        with self.assertRaises(TypeError) as ctx:
            param_ops._validate_param_group_dict(spec, {123: 0.7}, "config")
        self.assertIn("non-string key: 123 (type: int)", str(ctx.exception))

    def test_duplicate_case_insensitive_key(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        with self.assertRaises(ValueError) as ctx:
            param_ops._validate_param_group_dict(spec, {"temperature": 0.7, "TEMPERATURE": 0.9}, "config")
        self.assertIn("duplicate case-insensitive key", str(ctx.exception))

    def test_unknown_key(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        with self.assertRaises(ValueError) as ctx:
            param_ops._validate_param_group_dict(spec, {"unknown_key": 1}, "config")
        self.assertIn("Unknown key", str(ctx.exception))

    def test_leaf_validated_inline(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature", numpy_type=np.float64)])
        param_ops._validate_param_group_dict(spec, {"temperature": 0.7}, "config")

    def test_leaf_none_value_valid(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature", numpy_type=np.float64)])
        param_ops._validate_param_group_dict(spec, {"temperature": None}, "config")

    def test_nested_group_recurses(self) -> None:
        inner = _make_group_spec("sampling", [_make_leaf_spec("temperature", numpy_type=np.float64)])
        outer = _make_group_spec("config", [inner])
        param_ops._validate_param_group_dict(outer, {"sampling": {"temperature": 0.7}}, "config")

    def test_invalid_leaf_type_raises(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("count", numpy_type=np.int64)])
        with self.assertRaises(ValueError) as ctx:
            param_ops._validate_param_group_dict(spec, {"count": "not_int"}, "config")
        self.assertIn("Expected int", str(ctx.exception))


class ValidateParamGroupValueTest(absltest.TestCase):
    def test_none_passthrough(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        param_ops.validate_param_group_value(spec, None, "config")

    def test_unshaped_delegates_to_dict(self) -> None:
        spec = _make_group_spec("config", [_make_leaf_spec("temperature")])
        param_ops.validate_param_group_value(spec, {"temperature": 0.7}, "config")

    def test_shaped_list_validation(self) -> None:
        spec = _make_group_spec("items", [_make_leaf_spec("value")], shape=(2,))
        param_ops.validate_param_group_value(spec, [{"value": 1}, {"value": 2}], "items")

    def test_shaped_wrong_type(self) -> None:
        spec = _make_group_spec("items", [_make_leaf_spec("value")], shape=(2,))
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_param_group_value(spec, {"value": 1}, "items")
        self.assertIn("expected a list", str(ctx.exception))

    def test_shaped_wrong_length(self) -> None:
        spec = _make_group_spec("items", [_make_leaf_spec("value")], shape=(2,))
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_param_group_value(spec, [{"value": 1}], "items")
        self.assertIn("expected length 2", str(ctx.exception))

    def test_shaped_variable_length(self) -> None:
        spec = _make_group_spec("items", [_make_leaf_spec("value")], shape=(-1,))
        param_ops.validate_param_group_value(spec, [{"value": 1}, {"value": 2}, {"value": 3}], "items")

    def test_shaped_multidimensional(self) -> None:
        spec = _make_group_spec("matrix", [_make_leaf_spec("v")], shape=(2, 3))
        value = [[{"v": 1}, {"v": 2}, {"v": 3}], [{"v": 4}, {"v": 5}, {"v": 6}]]
        param_ops.validate_param_group_value(spec, value, "matrix")


class ValidateParamsTest(absltest.TestCase):
    def test_no_params_no_specs(self) -> None:
        param_ops.validate_params(None, None)

    def test_no_params_with_specs(self) -> None:
        param_ops.validate_params(None, [_make_leaf_spec("temperature")])

    def test_params_but_no_specs(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_params({"temperature": 0.7}, None)
        self.assertIn("does not accept any parameters", str(ctx.exception))

    def test_params_but_empty_specs(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_params({"temperature": 0.7}, [])
        self.assertIn("does not accept any parameters", str(ctx.exception))

    def test_unknown_param(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_params({"unknown": 1}, [_make_leaf_spec("temperature")])
        self.assertIn("Unknown parameter", str(ctx.exception))

    def test_duplicate_case_insensitive_param(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_params({"temperature": 0.7, "TEMPERATURE": 0.9}, [_make_leaf_spec("temperature")])
        self.assertIn("Duplicate parameter", str(ctx.exception))

    def test_case_insensitive_matching(self) -> None:
        param_ops.validate_params({"TEMPERATURE": 0.7}, [_make_leaf_spec("temperature", numpy_type=np.float64)])

    def test_scalar_validated(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        param_ops.validate_params({"temperature": 0.7}, [spec])

    def test_scalar_invalid_type_raises(self) -> None:
        spec = _make_leaf_spec("count", numpy_type=np.int64)
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_params({"count": "not_a_number"}, [spec])
        self.assertIn("Expected int", str(ctx.exception))

    def test_none_value_valid(self) -> None:
        spec = _make_leaf_spec("temperature", numpy_type=np.float64)
        param_ops.validate_params({"temperature": None}, [spec])

    def test_group_param_validated(self) -> None:
        group = _make_group_spec("config", [_make_leaf_spec("temperature", numpy_type=np.float64)])
        param_ops.validate_params({"config": {"temperature": 0.7}}, [group])

    def test_group_param_unknown_key_rejected(self) -> None:
        group = _make_group_spec("config", [_make_leaf_spec("temperature")])
        with self.assertRaises(ValueError) as ctx:
            param_ops.validate_params({"config": {"bad_key": 1}}, [group])
        self.assertIn("Unknown key", str(ctx.exception))


class ResolveParamsTest(absltest.TestCase):
    def test_defaults_only(self) -> None:
        specs = [SimpleNamespace(name="temperature", default_value=0.7, dtype=SimpleNamespace(_numpy_type=np.float64))]
        result = param_ops.resolve_params(None, specs)
        self.assertEqual(result, {"temperature": 0.7})

    def test_override_scalar(self) -> None:
        specs = [SimpleNamespace(name="temperature", default_value=0.7, dtype=SimpleNamespace(_numpy_type=np.float64))]
        result = param_ops.resolve_params({"temperature": 0.9}, specs)
        self.assertEqual(result, {"temperature": 0.9})

    def test_case_insensitive_override(self) -> None:
        specs = [SimpleNamespace(name="temperature", default_value=0.7, dtype=SimpleNamespace(_numpy_type=np.float64))]
        result = param_ops.resolve_params({"TEMPERATURE": 0.9}, specs)
        self.assertEqual(result, {"temperature": 0.9})

    def test_int_default_coerced_to_float(self) -> None:
        spec = SimpleNamespace(name="temperature", default_value=1, dtype=SimpleNamespace(_numpy_type=np.float64))
        result = param_ops.resolve_params(None, [spec])
        self.assertEqual(result, {"temperature": 1.0})
        self.assertIsInstance(result["temperature"], float)

    def test_int_override_coerced_to_float(self) -> None:
        spec = SimpleNamespace(name="temperature", default_value=0.7, dtype=SimpleNamespace(_numpy_type=np.float64))
        result = param_ops.resolve_params({"temperature": 1}, [spec])
        self.assertEqual(result, {"temperature": 1.0})
        self.assertIsInstance(result["temperature"], float)

    def test_no_coerce_for_int_dtype(self) -> None:
        spec = SimpleNamespace(name="top_k", default_value=50, dtype=SimpleNamespace(_numpy_type=np.int64))
        result = param_ops.resolve_params(None, [spec])
        self.assertEqual(result, {"top_k": 50})
        self.assertIsInstance(result["top_k"], int)

    def test_coerce_not_called_for_group(self) -> None:
        group = _make_group_spec("config", [_make_leaf_spec("temperature")])
        group.default_value = {"temperature": 0.7}
        result = param_ops.resolve_params(None, [group])
        self.assertEqual(result, {"config": {"temperature": 0.7}})

    def test_deep_merge_for_group_override(self) -> None:
        inner = _make_group_spec("config", [_make_leaf_spec("temperature"), _make_leaf_spec("top_k")])
        inner.default_value = {"temperature": 0.7, "top_k": 50}
        result = param_ops.resolve_params({"config": {"temperature": 0.9}}, [inner])
        self.assertEqual(result, {"config": {"temperature": 0.9, "top_k": 50}})

    def test_no_default_value(self) -> None:
        spec = SimpleNamespace(name="temperature", dtype=SimpleNamespace(_numpy_type=np.float64))
        result = param_ops.resolve_params(None, [spec])
        self.assertEqual(result, {})

    def test_empty_params(self) -> None:
        specs = [SimpleNamespace(name="temperature", default_value=0.7, dtype=SimpleNamespace(_numpy_type=np.float64))]
        result = param_ops.resolve_params({}, specs)
        self.assertEqual(result, {"temperature": 0.7})


if __name__ == "__main__":
    absltest.main()
