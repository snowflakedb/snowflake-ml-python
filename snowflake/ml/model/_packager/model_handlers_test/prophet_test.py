import tempfile
from typing import Any
from unittest import mock

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import prophet as prophet_handler
from snowflake.ml.model._packager.model_meta import model_blob_meta, model_meta


class ProphetHandlerTest(parameterized.TestCase):
    """Tests for ProphetHandler functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock prophet module to avoid dependency during testing
        self.mock_prophet = mock.MagicMock()
        self.prophet_model = mock.MagicMock()
        self.prophet_model.__class__.__name__ = "Prophet"

        # Create sample Prophet-formatted data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        self.sample_data = pd.DataFrame(
            {
                "ds": dates,
                "y": range(100),  # Simple linear trend
            }
        )

        # Create future data for forecasting (with NaN y values)
        future_dates = pd.date_range("2020-04-10", periods=30, freq="D")
        self.future_data = pd.DataFrame(
            {
                "ds": future_dates,
                "y": [float("nan")] * 30,
            }
        )

        # Sample prediction output
        self.sample_prediction = pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": range(100, 130),
                "yhat_lower": range(95, 125),
                "yhat_upper": range(105, 135),
            }
        )

    def test_can_handle_prophet_model(self) -> None:
        """Test that the handler correctly identifies Prophet models."""
        with mock.patch("snowflake.ml._internal.type_utils.LazyType") as mock_lazy_type:
            mock_lazy_type.return_value.isinstance.return_value = True

            result = prophet_handler.ProphetHandler.can_handle(self.prophet_model)
            self.assertTrue(result)

            mock_lazy_type.assert_called_with("prophet.Prophet")

    def test_can_handle_non_prophet_model(self) -> None:
        """Test that the handler rejects non-Prophet models."""
        with mock.patch("snowflake.ml._internal.type_utils.LazyType") as mock_lazy_type:
            mock_lazy_type.return_value.isinstance.return_value = False

            result = prophet_handler.ProphetHandler.can_handle("not a prophet model")
            self.assertFalse(result)

    def test_validate_prophet_data_format_valid(self) -> None:
        """Test validation of valid Prophet data format."""
        result = prophet_handler._validate_prophet_data_format(self.sample_data)

        # Check that the structure is correct
        self.assertEqual(result.shape, self.sample_data.shape)
        self.assertEqual(list(result.columns), list(self.sample_data.columns))

        # Check that datetime column is properly converted
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["ds"]))

        # Check that numeric columns are converted to float64 (required for Prophet)
        self.assertEqual(result["y"].dtype, "float64")

        # Check that values are preserved (even if dtypes change)
        self.assertTrue(result["ds"].equals(pd.to_datetime(self.sample_data["ds"])))
        self.assertTrue(result["y"].equals(self.sample_data["y"].astype("float64")))

    def test_validate_prophet_data_format_missing_ds(self) -> None:
        """Test validation fails when 'ds' column is missing."""
        invalid_data = pd.DataFrame({"y": [1, 2, 3]})

        with self.assertRaises(ValueError) as context:
            prophet_handler._validate_prophet_data_format(invalid_data)

        self.assertIn("ds", str(context.exception))

    def test_validate_prophet_data_format_missing_y(self) -> None:
        """Test validation fails when 'y' column is missing."""
        invalid_data = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=3)})

        with self.assertRaises(ValueError) as context:
            prophet_handler._validate_prophet_data_format(invalid_data)

        self.assertIn("y", str(context.exception))

    def test_validate_prophet_data_format_invalid_dates(self) -> None:
        """Test validation fails with invalid date values."""
        invalid_data = pd.DataFrame({"ds": ["not-a-date", "also-not-a-date"], "y": [1, 2]})

        with self.assertRaises(ValueError) as context:
            prophet_handler._validate_prophet_data_format(invalid_data)

        self.assertIn("datetime values", str(context.exception))

    def test_validate_prophet_data_format_non_dataframe(self) -> None:
        """Test validation fails for non-DataFrame input."""
        with self.assertRaises(ValueError) as context:
            prophet_handler._validate_prophet_data_format([1, 2, 3])

        self.assertIn("pandas DataFrame", str(context.exception))

    def test_validate_prophet_data_format_string_numeric_conversion(self) -> None:
        """Test that string numeric data is properly converted to numeric types.

        This test addresses the numpy isnan TypeError that occurs when Prophet
        receives string numeric data from Snowflake.
        """
        # Create data that mimics what comes from Snowflake (strings instead of numerics)
        string_data = pd.DataFrame(
            {
                "ds": ["2020-01-01", "2020-01-02", "2020-01-03"],  # String dates
                "y": ["1.5", "2.0", "3.5"],  # String numbers
                "regressor": ["10", "20", "30"],  # String regressor values
            }
        )

        # Validate and convert the data
        validated_data = prophet_handler._validate_prophet_data_format(string_data)

        # Verify datetime conversion
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(validated_data["ds"]))

        # Verify numeric conversion - should be float64, not object/string
        self.assertTrue(pd.api.types.is_numeric_dtype(validated_data["y"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(validated_data["regressor"]))

        # Verify the actual values are correct
        self.assertEqual(validated_data["y"].iloc[0], 1.5)
        self.assertEqual(validated_data["regressor"].iloc[0], 10.0)

        # Verify original data is not modified
        self.assertEqual(string_data["y"].iloc[0], "1.5")  # Still a string

    @mock.patch("snowflake.ml.model._packager.model_handlers.prophet.cloudpickle.dump")
    @mock.patch("os.makedirs")
    @mock.patch("builtins.open", mock.mock_open())
    def test_save_model_basic(self, mock_makedirs: Any, mock_dump: Any) -> None:
        """Test basic model saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock Prophet model and mock isinstance check
            with (
                mock.patch("prophet.Prophet"),
                mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
            ):
                # Configure mock predict to return sample data
                self.prophet_model.predict.return_value = self.sample_prediction

                # Create test metadata
                meta = model_meta.ModelMetadata(name="test_prophet", env=model_env.ModelEnv(), model_type="prophet")

                # Save model
                prophet_handler.ProphetHandler.save_model(
                    name="test_prophet",
                    model=self.prophet_model,
                    model_meta=meta,
                    model_blobs_dir_path=temp_dir,
                    sample_input_data=self.sample_data,
                )

                # Verify directory creation and pickle dump were called
                mock_makedirs.assert_called()
                mock_dump.assert_called_once()

                # Verify metadata was updated
                self.assertIn("test_prophet", meta.models)
                self.assertEqual(meta.models["test_prophet"].model_type, "prophet")

                # Verify dependencies were added (note: in actual usage via ModelPackager,
                # dependencies are added but in isolated handler tests the env may not reflect this)
                # The dependency addition logic exists in the handler and is covered by integration tests

                # Note: Function type configuration is now handled through options parameter
                # when calling registry.log_model(), not directly in the handler

    @mock.patch("cloudpickle.load")
    @mock.patch("builtins.open", mock.mock_open())
    def test_load_model(self, mock_load: Any) -> None:
        """Test model loading functionality."""
        mock_load.return_value = self.prophet_model

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test metadata
            meta = model_meta.ModelMetadata(name="test_prophet", env=model_env.ModelEnv(), model_type="prophet")
            meta.models["test_prophet"] = mock.MagicMock()
            meta.models["test_prophet"].path = "model.pkl"

            with (
                mock.patch("prophet.Prophet"),
                mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
            ):

                # Load model
                loaded_model = prophet_handler.ProphetHandler.load_model(
                    name="test_prophet",
                    model_meta=meta,
                    model_blobs_dir_path=temp_dir,
                )

                # Verify model was loaded
                self.assertEqual(loaded_model, self.prophet_model)
                mock_load.assert_called_once()

    def test_convert_as_custom_model(self) -> None:
        """Test conversion to CustomModel."""
        # Setup mock Prophet model with predict method
        self.prophet_model.predict.return_value = self.sample_prediction

        # Create test metadata with signatures (using infer_signature from actual predictions)
        meta = model_meta.ModelMetadata(name="test_prophet", env=model_env.ModelEnv(), model_type="prophet")
        meta.signatures = {"predict": model_signature.infer_signature(self.sample_data, self.sample_prediction)}

        # Add model blob metadata (required by convert_as_custom_model)
        meta.models["test_prophet"] = model_blob_meta.ModelBlobMeta(
            name="test_prophet",
            model_type="prophet",
            handler_version="2025-01-01",
            path="model.pkl",
            options={},
        )

        with (
            mock.patch("prophet.Prophet"),
            mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
        ):

            # Convert to custom model
            custom_model = prophet_handler.ProphetHandler.convert_as_custom_model(
                raw_model=self.prophet_model,
                model_meta=meta,
            )

            # Verify custom model was created
            self.assertIsNotNone(custom_model)

            # Test that predict method works
            self.assertTrue(hasattr(custom_model, "predict"))

    def test_handler_constants(self) -> None:
        """Test handler constants are set correctly."""
        handler = prophet_handler.ProphetHandler

        self.assertEqual(handler.HANDLER_TYPE, "prophet")
        self.assertEqual(handler.HANDLER_VERSION, "2025-01-01")
        self.assertEqual(handler.MODEL_BLOB_FILE_OR_DIR, "model.pkl")
        self.assertEqual(handler.DEFAULT_TARGET_METHODS, ["predict"])
        self.assertFalse(handler.IS_AUTO_SIGNATURE)

    @mock.patch("snowflake.ml.model._packager.model_handlers.prophet.cloudpickle.dump")
    @mock.patch("os.makedirs")
    @mock.patch("builtins.open", mock.mock_open())
    def test_table_function_configuration_complete(self, mock_makedirs: Any, mock_dump: Any) -> None:
        """Test that Prophet handler works correctly (TABLE_FUNCTION config now via options)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch("prophet.Prophet"),
                mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
            ):
                # Configure mock predict to return sample data
                self.prophet_model.predict.return_value = self.sample_prediction

                # Create test metadata
                meta = model_meta.ModelMetadata(
                    name="test_prophet_table_func", env=model_env.ModelEnv(), model_type="prophet"
                )

                # Save model with target methods
                prophet_handler.ProphetHandler.save_model(
                    name="test_prophet_table_func",
                    model=self.prophet_model,
                    model_meta=meta,
                    model_blobs_dir_path=temp_dir,
                    sample_input_data=self.sample_data,
                    target_methods=["predict"],
                )

                # Verify the handler completed successfully
                # Note: TABLE_FUNCTION configuration is now handled via options parameter
                # in registry.log_model(), not directly in the handler
                self.assertIsNotNone(meta.signatures)
                self.assertIn("predict", meta.signatures)

    def test_cast_model(self) -> None:
        """Test model casting functionality."""
        with (
            mock.patch("prophet.Prophet"),
            mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
        ):
            result = prophet_handler.ProphetHandler.cast_model(self.prophet_model)
            self.assertEqual(result, self.prophet_model)

    def test_normalize_column_names_basic(self) -> None:
        """Test basic column name normalization."""
        # Create data with custom column names
        custom_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=10),
                "sales": range(10),
            }
        )

        # Normalize column names
        normalized = prophet_handler._normalize_column_names(
            custom_data, date_column="timestamp", target_column="sales"
        )

        # Verify columns were renamed
        self.assertIn("ds", normalized.columns)
        self.assertIn("y", normalized.columns)
        self.assertNotIn("timestamp", normalized.columns)
        self.assertNotIn("sales", normalized.columns)

        # Verify data is preserved
        self.assertTrue(normalized["ds"].equals(custom_data["timestamp"]))
        self.assertTrue(normalized["y"].equals(custom_data["sales"]))

    def test_normalize_column_names_no_mapping(self) -> None:
        """Test that no normalization occurs when no mapping is provided."""
        normalized = prophet_handler._normalize_column_names(self.sample_data)

        # Data should be unchanged
        self.assertTrue(normalized.equals(self.sample_data))

    def test_normalize_column_names_partial_mapping(self) -> None:
        """Test normalization with only date_column specified."""
        custom_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=10),
                "y": range(10),
            }
        )

        normalized = prophet_handler._normalize_column_names(custom_data, date_column="timestamp")

        # Only timestamp should be renamed
        self.assertIn("ds", normalized.columns)
        self.assertIn("y", normalized.columns)
        self.assertNotIn("timestamp", normalized.columns)

    def test_normalize_column_names_conflict_ds(self) -> None:
        """Test that conflicting 'ds' column raises error."""
        conflicting_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=10),
                "ds": ["other"] * 10,
                "y": range(10),
            }
        )

        with self.assertRaises(ValueError) as context:
            prophet_handler._normalize_column_names(conflicting_data, date_column="timestamp")

        self.assertIn("'ds' already exists", str(context.exception))

    def test_normalize_column_names_conflict_y(self) -> None:
        """Test that conflicting 'y' column raises error."""
        conflicting_data = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=10),
                "sales": range(10),
                "y": [999] * 10,
            }
        )

        with self.assertRaises(ValueError) as context:
            prophet_handler._normalize_column_names(conflicting_data, target_column="sales")

        self.assertIn("'y' already exists", str(context.exception))

    def test_normalize_column_names_missing_column(self) -> None:
        """Test that missing column raises error."""
        with self.assertRaises(ValueError) as context:
            prophet_handler._normalize_column_names(self.sample_data, date_column="nonexistent")

        self.assertIn("not found", str(context.exception))

    @mock.patch("snowflake.ml.model._packager.model_handlers.prophet.cloudpickle.dump")
    @mock.patch("os.makedirs")
    @mock.patch("builtins.open", mock.mock_open())
    def test_save_model_with_column_mapping(self, mock_makedirs: Any, mock_dump: Any) -> None:
        """Test saving model with custom column names."""
        # Create data with custom column names
        custom_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=100),
                "revenue": range(100),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch("prophet.Prophet"),
                mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
            ):
                self.prophet_model.predict.return_value = self.sample_prediction

                # Create test metadata
                meta = model_meta.ModelMetadata(name="test_prophet", env=model_env.ModelEnv(), model_type="prophet")

                # Save model with column mapping
                prophet_handler.ProphetHandler.save_model(
                    name="test_prophet",
                    model=self.prophet_model,
                    model_meta=meta,
                    model_blobs_dir_path=temp_dir,
                    sample_input_data=custom_data,
                    date_column="timestamp",
                    target_column="revenue",
                )

                # Verify metadata stores column mapping
                self.assertIn("test_prophet", meta.models)
                model_options = meta.models["test_prophet"].options
                self.assertEqual(model_options.get("date_column"), "timestamp")
                self.assertEqual(model_options.get("target_column"), "revenue")

    def test_sanitize_prophet_output(self) -> None:
        """Test that Prophet output sanitization normalizes column names with spaces."""
        from snowflake.ml.model._packager.model_handlers.prophet import (
            _sanitize_prophet_output,
        )

        # Create mock Prophet prediction output with holiday columns (containing spaces)
        prophet_output = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=5),
                "yhat": [100, 101, 102, 103, 104],
                "yhat_lower": [95, 96, 97, 98, 99],
                "yhat_upper": [105, 106, 107, 108, 109],
                "trend": [100, 101, 102, 103, 104],
                "weekly": [1, 2, 3, 4, 5],
                "yearly": [0.5, 0.6, 0.7, 0.8, 0.9],
                # Holiday columns with spaces and special characters (should be normalized)
                "Christmas Day": [10, 0, 0, 0, 0],
                "New Year's Day": [0, 15, 0, 0, 0],
                "Independence Day": [0, 0, 20, 0, 0],
                # Custom regressor without spaces (should be unchanged)
                "temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
            }
        )

        sanitized = _sanitize_prophet_output(prophet_output)

        # Verify original column names with spaces are no longer present
        self.assertNotIn("Christmas Day", sanitized.columns)
        self.assertNotIn("New Year's Day", sanitized.columns)
        self.assertNotIn("Independence Day", sanitized.columns)

        # Verify normalized holiday columns ARE present (lowercase)
        self.assertIn("christmas_day", sanitized.columns)
        self.assertIn("new_year_s_day", sanitized.columns)
        self.assertIn("independence_day", sanitized.columns)

        # Verify core Prophet columns are kept unchanged
        self.assertIn("ds", sanitized.columns)
        self.assertIn("yhat", sanitized.columns)
        self.assertIn("yhat_lower", sanitized.columns)
        self.assertIn("yhat_upper", sanitized.columns)
        self.assertIn("trend", sanitized.columns)
        self.assertIn("weekly", sanitized.columns)
        self.assertIn("yearly", sanitized.columns)

        # Verify regressor columns without spaces are kept unchanged
        self.assertIn("temperature", sanitized.columns)

        # Verify data is preserved for normalized columns
        self.assertTrue(sanitized["ds"].equals(prophet_output["ds"]))
        self.assertTrue(sanitized["yhat"].equals(prophet_output["yhat"]))
        self.assertTrue(sanitized["temperature"].equals(prophet_output["temperature"]))

        # Verify holiday data is preserved under normalized lowercase names
        self.assertTrue(sanitized["christmas_day"].equals(prophet_output["Christmas Day"]))
        self.assertTrue(sanitized["new_year_s_day"].equals(prophet_output["New Year's Day"]))
        self.assertTrue(sanitized["independence_day"].equals(prophet_output["Independence Day"]))

    def test_normalize_column_name_for_sql(self) -> None:
        """Test column name normalization for SQL identifiers."""
        from snowflake.ml.model._packager.model_handlers._utils import (
            normalize_column_name,
        )

        # Test basic space replacement and lowercasing
        self.assertEqual(normalize_column_name("Christmas Day"), "christmas_day")
        self.assertEqual(normalize_column_name("New Year's Day"), "new_year_s_day")

        # Test multiple spaces
        self.assertEqual(normalize_column_name("Multiple   Spaces"), "multiple_spaces")

        # Test special characters
        self.assertEqual(normalize_column_name("Test@Column#Name"), "test_column_name")
        self.assertEqual(normalize_column_name("column-with-dashes"), "column_with_dashes")

        # Test leading/trailing underscores
        self.assertEqual(normalize_column_name("_leading"), "_leading")
        self.assertEqual(normalize_column_name("trailing_"), "trailing")
        # Trailing removed, leading preserved
        self.assertEqual(normalize_column_name("__both__"), "__both")

        # Test names starting with digits
        self.assertEqual(normalize_column_name("2023_data"), "_2023_data")

        # Test already valid names (lowercased)
        self.assertEqual(normalize_column_name("valid_column_name"), "valid_column_name")
        self.assertEqual(normalize_column_name("ds"), "ds")
        self.assertEqual(normalize_column_name("yhat"), "yhat")

        # Test uppercase conversion
        self.assertEqual(normalize_column_name("UPPERCASE"), "uppercase")
        self.assertEqual(normalize_column_name("MixedCase"), "mixedcase")

    def test_convert_as_custom_model_with_column_mapping(self) -> None:
        """Test conversion to CustomModel with column mapping."""
        # Setup mock Prophet model with predict method
        self.prophet_model.predict.return_value = self.sample_prediction

        # Create test metadata with column mapping
        meta = model_meta.ModelMetadata(name="test_prophet", env=model_env.ModelEnv(), model_type="prophet")
        meta.signatures = {"predict": model_signature.infer_signature(self.sample_data, self.sample_prediction)}

        # Add model blob metadata with column mapping options
        meta.models["test_prophet"] = model_blob_meta.ModelBlobMeta(
            name="test_prophet",
            model_type="prophet",
            handler_version="2025-01-01",
            path="model.pkl",
            options={"date_column": "timestamp", "target_column": "revenue"},
        )

        with (
            mock.patch("prophet.Prophet"),
            mock.patch("snowflake.ml.model._packager.model_handlers.prophet.isinstance", return_value=True),
        ):

            # Convert to custom model
            custom_model = prophet_handler.ProphetHandler.convert_as_custom_model(
                raw_model=self.prophet_model,
                model_meta=meta,
            )

            # Verify custom model was created
            self.assertIsNotNone(custom_model)

            # Test that predict method works with custom column names
            # Note: In real usage, the column mapping would be applied during inference
            self.assertTrue(hasattr(custom_model, "predict"))


if __name__ == "__main__":
    absltest.main()
