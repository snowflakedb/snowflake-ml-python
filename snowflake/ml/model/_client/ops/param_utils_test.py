import datetime

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.ops import param_utils
from snowflake.ml.model._signatures import core


class FormatParamValueForSqlTest(absltest.TestCase):
    """Tests for format_param_value_for_sql function."""

    def test_format_none_value(self) -> None:
        """Test that None is formatted as 'NULL' for SQL compatibility."""
        self.assertEqual(param_utils.format_param_value_for_sql(None), "NULL")

    def test_format_boolean_values(self) -> None:
        """Test that boolean values are formatted as lowercase SQL booleans."""
        self.assertEqual(param_utils.format_param_value_for_sql(True), "true")
        self.assertEqual(param_utils.format_param_value_for_sql(False), "false")

    def test_format_string_values(self) -> None:
        """Test that string values are formatted as single-quoted SQL literals."""
        self.assertEqual(param_utils.format_param_value_for_sql("hello"), "'hello'")
        self.assertEqual(param_utils.format_param_value_for_sql("default"), "'default'")
        # Test escaping single quotes
        self.assertEqual(param_utils.format_param_value_for_sql("it's"), "'it\\'s'")

    def test_format_bytes_values(self) -> None:
        """Test that bytes values are formatted as SQL hex literals."""
        self.assertEqual(param_utils.format_param_value_for_sql(b"hello"), "X'68656c6c6f'")
        self.assertEqual(param_utils.format_param_value_for_sql(b""), "X''")

    def test_format_datetime_values(self) -> None:
        """Test that datetime values are formatted as SQL timestamp literals."""
        dt = datetime.datetime(2024, 1, 1, 12, 0, 0)
        self.assertEqual(param_utils.format_param_value_for_sql(dt), "'2024-01-01 12:00:00'::TIMESTAMP_NTZ")

    def test_format_date_values(self) -> None:
        """Test that date values are formatted as SQL date literals."""
        d = datetime.date(2024, 1, 1)
        self.assertEqual(param_utils.format_param_value_for_sql(d), "'2024-01-01'::DATE")

    def test_format_list_values(self) -> None:
        """Test that list values are formatted using Python's str() representation.

        This uses single quotes for strings, which SQL interprets as string literals.
        (JSON's double quotes would be interpreted as SQL identifiers.)
        """
        self.assertEqual(param_utils.format_param_value_for_sql([]), "[]")
        self.assertEqual(param_utils.format_param_value_for_sql([1, 2, 3]), "[1, 2, 3]")
        self.assertEqual(param_utils.format_param_value_for_sql(["a", "b"]), "['a', 'b']")
        self.assertEqual(param_utils.format_param_value_for_sql([1.5, 2.5]), "[1.5, 2.5]")

    def test_format_numeric_values(self) -> None:
        """Test that numeric values are formatted via str()."""
        self.assertEqual(param_utils.format_param_value_for_sql(0.5), "0.5")
        self.assertEqual(param_utils.format_param_value_for_sql(100), "100")
        self.assertEqual(param_utils.format_param_value_for_sql(-42), "-42")
        self.assertEqual(param_utils.format_param_value_for_sql(3.14159), "3.14159")

    def test_format_dict_values(self) -> None:
        """Test that dict values are formatted as OBJECT_CONSTRUCT_KEEP_NULL."""
        self.assertEqual(
            param_utils.format_param_value_for_sql({"temperature": 0.5}),
            "OBJECT_CONSTRUCT_KEEP_NULL('temperature', 0.5)",
        )
        self.assertEqual(
            param_utils.format_param_value_for_sql({"key": "value"}),
            "OBJECT_CONSTRUCT_KEEP_NULL('key', 'value')",
        )
        self.assertEqual(
            param_utils.format_param_value_for_sql({}),
            "OBJECT_CONSTRUCT_KEEP_NULL()",
        )

    def test_format_dict_with_none_value(self) -> None:
        """Test that dict values with None are preserved as NULL."""
        self.assertEqual(
            param_utils.format_param_value_for_sql({"key": None}),
            "OBJECT_CONSTRUCT_KEEP_NULL('key', NULL)",
        )

    def test_format_dict_with_mixed_types(self) -> None:
        """Test dict with heterogeneous value types."""
        result = param_utils.format_param_value_for_sql({"temp": 0.5, "name": "gpt", "on": True})
        self.assertEqual(
            result,
            "OBJECT_CONSTRUCT_KEEP_NULL('temp', 0.5, 'name', 'gpt', 'on', true)",
        )

    def test_format_nested_dict(self) -> None:
        """Test that nested dicts produce nested OBJECT_CONSTRUCT_KEEP_NULL."""
        result = param_utils.format_param_value_for_sql({"config": {"lr": 0.01}})
        self.assertEqual(
            result,
            "OBJECT_CONSTRUCT_KEEP_NULL('config', OBJECT_CONSTRUCT_KEEP_NULL('lr', 0.01))",
        )

    def test_format_float_for_table_function(self) -> None:
        """Test that float values get explicit ::FLOAT cast for table function invocation."""
        self.assertEqual(param_utils.format_param_value_for_table_function_sql(0.5), "0.5::FLOAT")
        self.assertEqual(param_utils.format_param_value_for_table_function_sql(3.14159), "3.14159::FLOAT")
        self.assertEqual(param_utils.format_param_value_for_table_function_sql(100), "100")
        self.assertEqual(param_utils.format_param_value_for_table_function_sql("hello"), "'hello'")

    def test_format_dict_for_table_function(self) -> None:
        """Test that dict values are formatted correctly for table functions."""
        result = param_utils.format_param_value_for_table_function_sql({"temp": 0.5})
        self.assertEqual(result, "OBJECT_CONSTRUCT_KEEP_NULL('temp', 0.5)")


class ValidateParamsTest(absltest.TestCase):
    """Tests for validate_params function."""

    def test_no_params_no_signature(self) -> None:
        """Test with no params and no signature - should pass."""
        param_utils.validate_params(None, None)
        param_utils.validate_params({}, None)
        param_utils.validate_params(None, [])
        param_utils.validate_params({}, [])

    def test_params_provided_but_no_signature(self) -> None:
        """Test error when params provided but signature has no params."""
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"temperature": 0.5}, None)
        self.assertIn("does not accept any parameters", str(ctx.exception))

        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"temperature": 0.5}, [])
        self.assertIn("does not accept any parameters", str(ctx.exception))

    def test_no_params_with_signature(self) -> None:
        """Test with no params but valid signature - should pass."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        param_utils.validate_params(None, signature_params)
        param_utils.validate_params({}, signature_params)

    def test_valid_params(self) -> None:
        """Test with valid params matching signature."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        param_utils.validate_params({"temperature": 0.5}, signature_params)
        param_utils.validate_params({"max_tokens": 50}, signature_params)
        param_utils.validate_params({"temperature": 0.5, "max_tokens": 50}, signature_params)

    def test_case_insensitive_params(self) -> None:
        """Test that param names are matched case-insensitively."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        param_utils.validate_params({"TEMPERATURE": 0.5}, signature_params)
        param_utils.validate_params({"Temperature": 0.5}, signature_params)
        param_utils.validate_params({"tEmPerAtUrE": 0.5}, signature_params)

    def test_unknown_params(self) -> None:
        """Test error when unknown params are provided."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"unknown_param": 0.5}, signature_params)
        self.assertIn("Unknown parameter(s)", str(ctx.exception))
        self.assertIn("unknown_param", str(ctx.exception))

    def test_duplicate_params_different_cases(self) -> None:
        """Test error when duplicate params with different cases are provided."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"temperature": 0.5, "TEMPERATURE": 0.8}, signature_params)
        self.assertIn("Duplicate parameter(s)", str(ctx.exception))
        self.assertIn("case-insensitive", str(ctx.exception))

    def test_invalid_param_type(self) -> None:
        """Test error when param value has invalid type."""
        signature_params = [
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"max_tokens": "not_an_int"}, signature_params)
        self.assertIn("not compatible with dtype", str(ctx.exception))

    def test_rejects_string_for_int_param(self) -> None:
        """Test that numeric string values are rejected for INT params."""
        signature_params = [
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"max_tokens": "10"}, signature_params)
        self.assertIn("not compatible with dtype", str(ctx.exception))

    def test_rejects_string_for_float_param(self) -> None:
        """Test that numeric string values are rejected for FLOAT params."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.DOUBLE, default_value=0.7),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"temperature": "0.7"}, signature_params)
        self.assertIn("not compatible with dtype", str(ctx.exception))

    def test_rejects_float_for_int_param(self) -> None:
        """Test that float values are rejected for INT params."""
        signature_params = [
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"max_tokens": 0.1}, signature_params)
        self.assertIn("not compatible with dtype", str(ctx.exception))

    def test_valid_param_group_spec(self) -> None:
        """Test validation passes for valid dict params matching ParamGroupSpec."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[
                    core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
                    core.ParamSpec(name="top_k", dtype=core.DataType.INT32, default_value=50),
                ],
            ),
        ]
        param_utils.validate_params({"config": {"temperature": 0.5}}, signature_params)
        param_utils.validate_params({"config": {"temperature": 0.5, "top_k": 10}}, signature_params)

    def test_param_group_spec_none_value(self) -> None:
        """Test that None is valid for a ParamGroupSpec param."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        param_utils.validate_params({"config": None}, signature_params)

    def test_param_group_spec_non_dict_value(self) -> None:
        """Test error when a non-dict value is provided for a ParamGroupSpec param."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"config": "not_a_dict"}, signature_params)
        self.assertIn("expected a dict", str(ctx.exception))

    def test_param_group_spec_unknown_key(self) -> None:
        """Test error when dict contains unknown keys."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"config": {"lr": 0.01, "unknown": 42}}, signature_params)
        self.assertIn("Unknown key(s)", str(ctx.exception))
        self.assertIn("unknown", str(ctx.exception))

    def test_param_group_spec_invalid_child_type(self) -> None:
        """Test error when a child value has an invalid type."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"config": {"lr": "not_a_float"}}, signature_params)
        self.assertIn("not compatible with dtype", str(ctx.exception))

    def test_param_group_spec_nested_validation(self) -> None:
        """Test recursive validation of nested ParamGroupSpec (up to 3 levels)."""
        signature_params = [
            core.ParamGroupSpec(
                name="training",
                specs=[
                    core.ParamSpec(name="epochs", dtype=core.DataType.INT32, default_value=10),
                    core.ParamGroupSpec(
                        name="optimizer",
                        specs=[
                            core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01),
                            core.ParamSpec(name="momentum", dtype=core.DataType.FLOAT, default_value=0.9),
                            core.ParamGroupSpec(
                                name="schedule",
                                specs=[
                                    core.ParamSpec(name="warmup_steps", dtype=core.DataType.INT32, default_value=100),
                                    core.ParamSpec(name="decay_rate", dtype=core.DataType.FLOAT, default_value=0.99),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
        # Valid 2-level nested dict
        param_utils.validate_params({"training": {"optimizer": {"lr": 0.001}}}, signature_params)
        # Valid 3-level nested dict
        param_utils.validate_params({"training": {"optimizer": {"schedule": {"warmup_steps": 200}}}}, signature_params)
        # Full override at all levels
        param_utils.validate_params(
            {
                "training": {
                    "epochs": 5,
                    "optimizer": {
                        "lr": 0.002,
                        "schedule": {"warmup_steps": 50, "decay_rate": 0.95},
                    },
                }
            },
            signature_params,
        )
        # Invalid key at level 2
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"training": {"optimizer": {"bad_key": 1}}}, signature_params)
        self.assertIn("Unknown key(s)", str(ctx.exception))
        self.assertIn("training.optimizer", str(ctx.exception))
        # Invalid key at level 3
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"training": {"optimizer": {"schedule": {"bad_key": 1}}}}, signature_params)
        self.assertIn("Unknown key(s)", str(ctx.exception))
        self.assertIn("training.optimizer.schedule", str(ctx.exception))

    def test_param_group_spec_non_string_keys(self) -> None:
        """Test error when dict has non-string keys."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"config": {1: "value"}}, signature_params)
        self.assertIn("non-string key", str(ctx.exception))

    def test_param_group_spec_duplicate_case_insensitive_keys(self) -> None:
        """Test error when nested dict has keys that collide case-insensitively."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_params({"config": {"lr": 0.1, "LR": 0.2}}, signature_params)
        self.assertIn("duplicate case-insensitive key", str(ctx.exception))


class ResolveParamsTest(absltest.TestCase):
    """Tests for resolve_params function."""

    def test_no_params_uses_defaults(self) -> None:
        """Test that defaults are used when no params provided."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        result = param_utils.resolve_params(None, signature_params)

        self.assertEqual(len(result), 2)
        # SqlIdentifier returns uppercase by default
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["TEMPERATURE"], 0.7)
        self.assertEqual(result_dict["MAX_TOKENS"], 100)

    def test_empty_params_uses_defaults(self) -> None:
        """Test that defaults are used when empty params provided."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        result = param_utils.resolve_params({}, signature_params)

        self.assertEqual(len(result), 1)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["TEMPERATURE"], 0.7)

    def test_override_defaults(self) -> None:
        """Test that provided params override defaults."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        result = param_utils.resolve_params({"temperature": 0.5}, signature_params)

        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["TEMPERATURE"], 0.5)  # Overridden
        self.assertEqual(result_dict["MAX_TOKENS"], 100)  # Default

    def test_case_insensitive_override(self) -> None:
        """Test that overrides work case-insensitively but use canonical name."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        result = param_utils.resolve_params({"TEMPERATURE": 0.5}, signature_params)

        # Result should use canonical name from signature (SqlIdentifier uppercases by default)
        self.assertEqual(len(result), 1)
        name, value = result[0]
        self.assertEqual(str(name), "TEMPERATURE")
        self.assertEqual(value, 0.5)

    def test_returns_sql_identifiers(self) -> None:
        """Test that result contains SqlIdentifier objects."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        result = param_utils.resolve_params(None, signature_params)

        self.assertEqual(len(result), 1)
        name, value = result[0]
        self.assertIsInstance(name, sql_identifier.SqlIdentifier)

    def test_resolve_coerces_int_to_float_for_override(self) -> None:
        """Test that int override values are coerced to float for DOUBLE params."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.DOUBLE, default_value=0.7),
        ]
        result = param_utils.resolve_params({"temperature": 1}, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertIsInstance(result_dict["TEMPERATURE"], float)
        self.assertEqual(result_dict["TEMPERATURE"], 1.0)

    def test_resolve_coerces_int_default_to_float(self) -> None:
        """Test that int default values are coerced to float for DOUBLE params."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.DOUBLE, default_value=1),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertIsInstance(result_dict["TEMPERATURE"], float)
        self.assertEqual(result_dict["TEMPERATURE"], 1.0)

    def test_resolve_coerces_int_array_to_float(self) -> None:
        """Test that int elements in arrays are coerced to float for FLOAT params."""
        signature_params = [
            core.ParamSpec(name="weights", dtype=core.DataType.FLOAT, default_value=[1, 2, 3], shape=(-1,)),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["WEIGHTS"], [1.0, 2.0, 3.0])
        for v in result_dict["WEIGHTS"]:
            self.assertIsInstance(v, float)

    def test_resolve_coerces_int_tuple_to_float(self) -> None:
        """Test that int elements in tuples are coerced to float for FLOAT params."""
        signature_params = [
            core.ParamSpec(name="weights", dtype=core.DataType.FLOAT, default_value=(1, 2, 3), shape=(-1,)),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["WEIGHTS"], (1.0, 2.0, 3.0))
        for v in result_dict["WEIGHTS"]:
            self.assertIsInstance(v, float)

    def test_resolve_coerces_nested_int_arrays_to_float(self) -> None:
        """Test that int elements in nested lists and mixed list/tuple structures are coerced to float."""
        signature_params = [
            core.ParamSpec(name="matrix", dtype=core.DataType.FLOAT, default_value=[[1, 2], [3, 4]], shape=(-1, -1)),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["MATRIX"], [[1.0, 2.0], [3.0, 4.0]])
        for row in result_dict["MATRIX"]:
            for v in row:
                self.assertIsInstance(v, float)

        signature_params = [
            core.ParamSpec(name="matrix", dtype=core.DataType.FLOAT, default_value=[(1, 2), (3, 4)], shape=(-1, -1)),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["MATRIX"], [(1.0, 2.0), (3.0, 4.0)])
        for row in result_dict["MATRIX"]:
            for v in row:
                self.assertIsInstance(v, float)

    def test_resolve_preserves_correct_types(self) -> None:
        """Test that already-correct types are preserved without modification."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.DOUBLE, default_value=0.7),
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertIsInstance(result_dict["TEMPERATURE"], float)
        self.assertEqual(result_dict["TEMPERATURE"], 0.7)
        self.assertIsInstance(result_dict["MAX_TOKENS"], int)
        self.assertEqual(result_dict["MAX_TOKENS"], 100)

    def test_param_group_spec_defaults(self) -> None:
        """Test that ParamGroupSpec defaults are resolved as dicts."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[
                    core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01),
                    core.ParamSpec(name="momentum", dtype=core.DataType.FLOAT, default_value=0.9),
                ],
            ),
        ]
        result = param_utils.resolve_params(None, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["CONFIG"], {"lr": 0.01, "momentum": 0.9})

    def test_param_group_spec_partial_override(self) -> None:
        """Test that partial dict overrides are deep-merged with defaults."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[
                    core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
                    core.ParamSpec(name="top_k", dtype=core.DataType.INT32, default_value=50),
                ],
            ),
        ]
        result = param_utils.resolve_params({"config": {"temperature": 0.5}}, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["CONFIG"], {"temperature": 0.5, "top_k": 50})

    def test_param_group_spec_full_override(self) -> None:
        """Test that a full dict override replaces all defaults."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[
                    core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
                    core.ParamSpec(name="top_k", dtype=core.DataType.INT32, default_value=50),
                ],
            ),
        ]
        result = param_utils.resolve_params({"config": {"temperature": 0.1, "top_k": 10}}, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["CONFIG"], {"temperature": 0.1, "top_k": 10})

    def test_param_group_spec_none_override(self) -> None:
        """Test that None override replaces the entire group default."""
        signature_params = [
            core.ParamGroupSpec(
                name="config",
                specs=[core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)],
            ),
        ]
        result = param_utils.resolve_params({"config": None}, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertIsNone(result_dict["CONFIG"])

    def test_param_group_spec_deep_merge_nested(self) -> None:
        """Test deep merge with nested ParamGroupSpec (up to 3 levels)."""
        signature_params = [
            core.ParamGroupSpec(
                name="training",
                specs=[
                    core.ParamSpec(name="epochs", dtype=core.DataType.INT32, default_value=10),
                    core.ParamGroupSpec(
                        name="optimizer",
                        specs=[
                            core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01),
                            core.ParamSpec(name="momentum", dtype=core.DataType.FLOAT, default_value=0.9),
                            core.ParamGroupSpec(
                                name="schedule",
                                specs=[
                                    core.ParamSpec(name="warmup_steps", dtype=core.DataType.INT32, default_value=100),
                                    core.ParamSpec(name="decay_rate", dtype=core.DataType.FLOAT, default_value=0.99),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
        # Override only a level-2 leaf, keeping everything else default
        result = param_utils.resolve_params({"training": {"optimizer": {"lr": 0.001}}}, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(
            result_dict["TRAINING"],
            {
                "epochs": 10,
                "optimizer": {"lr": 0.001, "momentum": 0.9, "schedule": {"warmup_steps": 100, "decay_rate": 0.99}},
            },
        )
        # Override only the deepest field, keeping all other defaults
        result = param_utils.resolve_params(
            {"training": {"optimizer": {"schedule": {"warmup_steps": 200}}}}, signature_params
        )
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(
            result_dict["TRAINING"],
            {
                "epochs": 10,
                "optimizer": {
                    "lr": 0.01,
                    "momentum": 0.9,
                    "schedule": {"warmup_steps": 200, "decay_rate": 0.99},
                },
            },
        )

    def test_param_group_spec_mixed_with_scalar_params(self) -> None:
        """Test resolution with both ParamGroupSpec and scalar ParamSpec."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
            core.ParamGroupSpec(
                name="config",
                specs=[
                    core.ParamSpec(name="top_k", dtype=core.DataType.INT32, default_value=50),
                ],
            ),
        ]
        result = param_utils.resolve_params({"temperature": 0.5, "config": {"top_k": 10}}, signature_params)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["TEMPERATURE"], 0.5)
        self.assertEqual(result_dict["CONFIG"], {"top_k": 10})


class ValidateAndResolveParamsTest(absltest.TestCase):
    """Tests for validate_and_resolve_params function."""

    def test_no_signature_returns_none(self) -> None:
        """Test that None is returned when no signature params."""
        result = param_utils.validate_and_resolve_params(None, None)
        self.assertIsNone(result)

        result = param_utils.validate_and_resolve_params(None, [])
        self.assertIsNone(result)

    def test_validates_before_resolving(self) -> None:
        """Test that validation happens before resolution."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        with self.assertRaises(exceptions.SnowflakeMLException) as ctx:
            param_utils.validate_and_resolve_params({"unknown": 0.5}, signature_params)
        self.assertIn("Unknown parameter(s)", str(ctx.exception))

    def test_valid_params_are_resolved(self) -> None:
        """Test that valid params are validated and resolved."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
            core.ParamSpec(name="max_tokens", dtype=core.DataType.INT32, default_value=100),
        ]
        result = param_utils.validate_and_resolve_params({"temperature": 0.5}, signature_params)

        self.assertIsNotNone(result)
        assert result is not None  # For type narrowing
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["TEMPERATURE"], 0.5)
        self.assertEqual(result_dict["MAX_TOKENS"], 100)

    def test_no_params_with_signature(self) -> None:
        """Test with no params but valid signature."""
        signature_params = [
            core.ParamSpec(name="temperature", dtype=core.DataType.FLOAT, default_value=0.7),
        ]
        result = param_utils.validate_and_resolve_params(None, signature_params)

        self.assertIsNotNone(result)
        assert result is not None  # For type narrowing
        self.assertEqual(len(result), 1)
        result_dict = {str(name): value for name, value in result}
        self.assertEqual(result_dict["TEMPERATURE"], 0.7)


if __name__ == "__main__":
    absltest.main()
