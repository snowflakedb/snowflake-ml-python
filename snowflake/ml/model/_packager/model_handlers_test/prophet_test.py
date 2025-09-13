# mypy: disable-error-code="import"
import tempfile
from typing import TYPE_CHECKING
from unittest import mock

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers import prophet as prophet_handler
from snowflake.ml.model._packager.model_meta import model_meta


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
        pd.testing.assert_frame_equal(result, self.sample_data)

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

    def test_create_prophet_signature(self) -> None:
        """Test creation of Prophet model signature."""
        signature = prophet_handler._create_prophet_signature(self.sample_data)

        # Check input signature
        self.assertEqual(len(signature.inputs), 2)
        self.assertEqual(signature.inputs[0].name, "ds")
        self.assertEqual(signature.inputs[0].dtype, model_signature.DataType.TIMESTAMP)
        self.assertEqual(signature.inputs[1].name, "y")
        self.assertEqual(signature.inputs[1].dtype, model_signature.DataType.DOUBLE)

        # Check output signature
        self.assertEqual(len(signature.outputs), 4)
        output_names = [spec.name for spec in signature.outputs]
        self.assertIn("ds", output_names)
        self.assertIn("yhat", output_names)
        self.assertIn("yhat_lower", output_names)
        self.assertIn("yhat_upper", output_names)

    def test_create_prophet_signature_with_regressors(self) -> None:
        """Test signature creation with additional regressors."""
        data_with_regressors = self.sample_data.copy()
        data_with_regressors["holiday"] = [0] * 100
        data_with_regressors["temp"] = range(20, 120)

        signature = prophet_handler._create_prophet_signature(data_with_regressors)

        # Should have 4 inputs: ds, y, holiday, temp
        self.assertEqual(len(signature.inputs), 4)
        input_names = [spec.name for spec in signature.inputs]
        self.assertIn("ds", input_names)
        self.assertIn("y", input_names)
        self.assertIn("holiday", input_names)
        self.assertIn("temp", input_names)

    def test_create_prophet_components_signature(self) -> None:
        """Test creation of Prophet components signature."""
        signature = prophet_handler._create_prophet_components_signature(self.sample_data)

        # Check output signature has trend and seasonality components
        self.assertEqual(len(signature.outputs), 4)
        output_names = [spec.name for spec in signature.outputs]
        self.assertIn("ds", output_names)
        self.assertIn("trend", output_names)
        self.assertIn("weekly", output_names)
        self.assertIn("yearly", output_names)

    @mock.patch("cloudpickle.dump")
    @mock.patch("os.makedirs")
    @mock.patch("builtins.open", mock.mock_open())
    def test_save_model_basic(self, mock_makedirs, mock_dump) -> None:
        """Test basic model saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock Prophet model
            with mock.patch("prophet.Prophet") as mock_prophet_class:
                mock_prophet_class.return_value = self.prophet_model

                # Create test metadata
                meta = model_meta.ModelMetadata()

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

                # Verify dependencies were added
                dependency_names = [dep.requirement for dep in meta.env.dependencies]
                self.assertIn("prophet", dependency_names)
                self.assertIn("pandas", dependency_names)

                # Verify function properties are set to TABLE_FUNCTION
                self.assertIsNotNone(meta.function_properties)
                for method in ["predict", "predict_components"]:
                    if method in meta.function_properties:
                        self.assertEqual(meta.function_properties[method]["function_type"], "TABLE_FUNCTION")

    @mock.patch("cloudpickle.load")
    @mock.patch("builtins.open", mock.mock_open())
    def test_load_model(self, mock_load) -> None:
        """Test model loading functionality."""
        mock_load.return_value = self.prophet_model

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test metadata
            meta = model_meta.ModelMetadata()
            meta.models["test_prophet"] = mock.MagicMock()
            meta.models["test_prophet"].path = "model.pkl"

            with mock.patch("prophet.Prophet") as mock_prophet_class:
                mock_prophet_class.return_value = self.prophet_model

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

        # Create test metadata with signatures
        meta = model_meta.ModelMetadata()
        meta.signatures = {"predict": prophet_handler._create_prophet_signature(self.sample_data)}

        with mock.patch("prophet.Prophet") as mock_prophet_class:
            mock_prophet_class.return_value = self.prophet_model

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
        self.assertEqual(handler.DEFAULT_TARGET_METHODS, ["predict", "predict_components"])
        self.assertFalse(handler.IS_AUTO_SIGNATURE)

    def test_table_function_configuration_complete(self) -> None:
        """Test that Prophet models are correctly configured as TABLE_FUNCTION."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("prophet.Prophet") as mock_prophet_class:
                mock_prophet_class.return_value = self.prophet_model

                # Create test metadata
                meta = model_meta.ModelMetadata()

                # Save model with both target methods
                prophet_handler.ProphetHandler.save_model(
                    name="test_prophet_table_func",
                    model=self.prophet_model,
                    model_meta=meta,
                    model_blobs_dir_path=temp_dir,
                    sample_input_data=self.sample_data,
                    target_methods=["predict", "predict_components"],
                )

                # Verify function properties are set correctly
                self.assertIsNotNone(meta.function_properties)

                # Both methods should be TABLE_FUNCTION
                self.assertIn("predict", meta.function_properties)
                self.assertIn("predict_components", meta.function_properties)

                for method in ["predict", "predict_components"]:
                    self.assertEqual(
                        meta.function_properties[method]["function_type"],
                        "TABLE_FUNCTION",
                        f"Method '{method}' should use TABLE_FUNCTION",
                    )

                # Verify this is different from typical ML models
                # (XGBoost, sklearn would use regular "FUNCTION" or not set function_type)

    def test_cast_model(self) -> None:
        """Test model casting functionality."""
        with mock.patch("prophet.Prophet") as mock_prophet_class:
            mock_prophet_class.return_value = self.prophet_model
            type(self.prophet_model).__name__ = "Prophet"

            # Mock isinstance to return True for Prophet
            with mock.patch("builtins.isinstance", return_value=True):
                result = prophet_handler.ProphetHandler.cast_model(self.prophet_model)
                self.assertEqual(result, self.prophet_model)


if __name__ == "__main__":
    absltest.main()
