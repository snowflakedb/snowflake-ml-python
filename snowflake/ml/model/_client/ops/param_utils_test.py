from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.ops import param_utils
from snowflake.ml.model._signatures import core


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
