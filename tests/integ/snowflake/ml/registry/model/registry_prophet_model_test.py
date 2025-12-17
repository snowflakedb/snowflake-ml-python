import numpy as np
import pandas as pd
import prophet
from absl.testing import absltest

from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistryProphetModelInteg(registry_model_test_base.RegistryModelTestBase):
    """Integration tests for Prophet time series forecasting models."""

    @staticmethod
    def _create_sample_time_series_data(
        start_date: str = "2020-01-01",
        periods: int = 100,
        freq: str = "D",
        include_regressors: bool = False,
    ) -> pd.DataFrame:
        """Create sample time series data in Prophet format."""
        dates = pd.date_range(start_date, periods=periods, freq=freq)

        # Create synthetic data with trend and seasonality
        trend = np.linspace(100, 200, periods)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
        noise = np.random.normal(0, 2, periods)  # Reduced noise for stable tests
        values = trend + seasonal + noise

        data = pd.DataFrame({"ds": dates, "y": values})

        if include_regressors:
            # Add holiday indicator (weekends)
            data["holiday"] = (data["ds"].dt.dayofweek >= 5).astype(int)
            # Add temperature regressor
            data["temperature"] = 20 + 5 * np.sin(2 * np.pi * np.arange(periods) / 365.25)

        return data

    @staticmethod
    def _create_future_data(
        last_date: str,
        periods: int = 30,
        freq: str = "D",
        include_regressors: bool = False,
    ) -> pd.DataFrame:
        """Create future data for forecasting."""
        last_dt = pd.to_datetime(last_date)
        future_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=periods, freq=freq)

        future_data = pd.DataFrame(
            {"ds": future_dates, "y": [float("nan")] * periods}  # NaN indicates periods to forecast
        )

        if include_regressors:
            # Provide future regressor values
            future_data["holiday"] = (future_data["ds"].dt.dayofweek >= 5).astype(int)
            future_data["temperature"] = [22.0] * periods  # Sample future temperatures

        return future_data

    def test_prophet_basic_model(self) -> None:
        """Test basic Prophet model without regressors."""
        # Create training data
        training_data = self._create_sample_time_series_data(start_date="2020-01-01", periods=365, freq="D")

        # Train Prophet model
        model = prophet.Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.8,  # Narrower intervals for more stable tests
        )
        model.fit(training_data)

        # Create future data for testing
        future_data = self._create_future_data(last_date="2020-12-31", periods=30, freq="D")

        # Expected outputs for assertions
        # Prophet predict already includes component information

        def assert_predict(result: pd.DataFrame) -> None:
            """Assert predict method returns correct format."""
            # Check required columns are present
            required_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            for col in required_cols:
                self.assertIn(col, result.columns, f"Missing column: {col}")

            # Check data types and shapes
            self.assertEqual(len(result), len(future_data))
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["ds"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(result["yhat"]))

            # Check forecast values are reasonable (not NaN/infinite)
            self.assertFalse(result["yhat"].isna().any())
            self.assertTrue(np.isfinite(result["yhat"]).all())

            # Check uncertainty intervals are valid
            self.assertTrue((result["yhat_lower"] <= result["yhat"]).all())
            self.assertTrue((result["yhat"] <= result["yhat_upper"]).all())

        self._test_registry_model(
            model=model,
            sample_input_data=training_data,
            prediction_assert_fns={
                "predict": (future_data, assert_predict),
            },
            function_type_assert={
                "predict": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
            },
            additional_dependencies=[f"prophet=={prophet.__version__}"],
            # Note: TABLE_FUNCTION is automatically configured for Prophet models
        )

    def test_prophet_model_with_regressors(self) -> None:
        """Test Prophet model with additional regressors."""
        # Create training data with regressors
        training_data = self._create_sample_time_series_data(
            start_date="2020-01-01", periods=365, freq="D", include_regressors=True
        )

        # Train Prophet model with regressors
        model = prophet.Prophet(interval_width=0.8)
        model.add_regressor("holiday")
        model.add_regressor("temperature")
        model.fit(training_data)

        # Create future data with regressor values
        future_data = self._create_future_data(last_date="2020-12-31", periods=30, freq="D", include_regressors=True)

        def assert_predict_with_regressors(result: pd.DataFrame) -> None:
            """Assert predict method works with regressors."""
            required_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            for col in required_cols:
                self.assertIn(col, result.columns)

            # Validate forecasts
            self.assertEqual(len(result), len(future_data))
            self.assertFalse(result["yhat"].isna().any())
            self.assertTrue(np.isfinite(result["yhat"]).all())

            # Check uncertainty intervals
            self.assertTrue((result["yhat_lower"] <= result["yhat"]).all())
            self.assertTrue((result["yhat"] <= result["yhat_upper"]).all())

        self._test_registry_model(
            model=model,
            sample_input_data=training_data,
            prediction_assert_fns={
                "predict": (future_data, assert_predict_with_regressors),
            },
            function_type_assert={
                "predict": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
            },
            additional_dependencies=[f"prophet=={prophet.__version__}"],
            # Note: TABLE_FUNCTION is automatically configured for Prophet models
        )

    def test_prophet_weekly_model(self) -> None:
        """Test Prophet model with weekly frequency."""
        # Create weekly training data
        training_data = self._create_sample_time_series_data(
            start_date="2020-01-06", periods=52, freq="W-MON"  # Weekly on Mondays
        )

        # Train Prophet model optimized for weekly data
        model = prophet.Prophet(
            weekly_seasonality=False,  # Disable weekly seasonality for weekly data
            yearly_seasonality=True,
            interval_width=0.8,
        )
        model.fit(training_data)

        # Create future weekly data
        future_data = self._create_future_data(last_date="2020-12-28", periods=8, freq="W-MON")  # 8 weeks ahead

        def assert_weekly_predict(result: pd.DataFrame) -> None:
            """Assert weekly predictions are valid."""
            required_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            for col in required_cols:
                self.assertIn(col, result.columns)

            # Check weekly frequency is maintained
            self.assertEqual(len(result), 8)  # 8 weeks

            # Validate all forecasts are on Mondays
            self.assertTrue((result["ds"].dt.dayofweek == 0).all())  # Monday = 0

            # Check forecast quality
            self.assertFalse(result["yhat"].isna().any())
            self.assertTrue(np.isfinite(result["yhat"]).all())

        self._test_registry_model(
            model=model,
            sample_input_data=training_data,
            prediction_assert_fns={
                "predict": (future_data, assert_weekly_predict),
            },
            function_type_assert={
                "predict": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
            },
            additional_dependencies=[f"prophet=={prophet.__version__}"],
            # Note: TABLE_FUNCTION is automatically configured for Prophet models
        )

    def test_prophet_model_with_holidays(self) -> None:
        """Test Prophet model with built-in holidays."""
        # Create training data
        training_data = self._create_sample_time_series_data(start_date="2020-01-01", periods=365, freq="D")

        # Train Prophet model with US holidays
        model = prophet.Prophet(
            interval_width=0.8,
        )
        # Add US holidays
        model.add_country_holidays(country_name="US")
        model.fit(training_data)

        # Create future data
        future_data = self._create_future_data(last_date="2020-12-31", periods=30, freq="D")

        def assert_holiday_predict(result: pd.DataFrame) -> None:
            """Assert predictions work with holiday effects."""
            required_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            for col in required_cols:
                self.assertIn(col, result.columns)

            # Standard validation
            self.assertEqual(len(result), len(future_data))
            self.assertFalse(result["yhat"].isna().any())
            self.assertTrue(np.isfinite(result["yhat"]).all())

        self._test_registry_model(
            model=model,
            sample_input_data=training_data,
            prediction_assert_fns={
                "predict": (future_data, assert_holiday_predict),
            },
            function_type_assert={
                "predict": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
            },
            additional_dependencies=[f"prophet=={prophet.__version__}"],
            # Note: TABLE_FUNCTION is automatically configured for Prophet models
        )

    def test_prophet_model_edge_cases(self) -> None:
        """Test Prophet model with edge cases and validation."""
        # Create training data
        training_data = self._create_sample_time_series_data(start_date="2020-01-01", periods=100, freq="D")

        # Train Prophet model
        model = prophet.Prophet(interval_width=0.8)
        model.fit(training_data)

        # Test 1: Single day forecast
        single_day_future = self._create_future_data(last_date="2020-04-09", periods=1, freq="D")

        # Test 2: Longer forecast horizon
        long_future = self._create_future_data(last_date="2020-04-09", periods=90, freq="D")

        def assert_single_day(result: pd.DataFrame) -> None:
            """Assert single day forecast works."""
            self.assertEqual(len(result), 1)
            self.assertIn("yhat", result.columns)
            self.assertFalse(result["yhat"].isna().any())

        def assert_long_forecast(result: pd.DataFrame) -> None:
            """Assert long-term forecast works."""
            self.assertEqual(len(result), 90)
            self.assertIn("yhat", result.columns)
            self.assertFalse(result["yhat"].isna().any())

            # Check that forecasts extend properly into the future
            self.assertTrue((result["ds"] > pd.to_datetime("2020-04-09")).all())

        self._test_registry_model(
            model=model,
            sample_input_data=training_data,
            prediction_assert_fns={
                "predict": (single_day_future, assert_single_day),
            },
            function_type_assert={
                "predict": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
            },
            additional_dependencies=[f"prophet=={prophet.__version__}"],
            additional_version_suffix="single_day",
            # Note: TABLE_FUNCTION is automatically configured for Prophet models
        )

        # Test long forecast in separate model to avoid conflicts
        self._test_registry_model(
            model=model,
            sample_input_data=training_data,
            prediction_assert_fns={
                "predict": (long_future, assert_long_forecast),
            },
            function_type_assert={
                "predict": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
            },
            additional_dependencies=[f"prophet=={prophet.__version__}"],
            additional_version_suffix="long_forecast",
            # Note: TABLE_FUNCTION is automatically configured for Prophet models
        )

    def test_prophet_data_validation(self) -> None:
        """Test that Prophet handler correctly validates input data format."""
        # This test verifies the data validation without going through full registration
        from snowflake.ml.model._packager.model_handlers.prophet import (
            _validate_prophet_data_format,
        )

        # Valid data should pass validation and be converted to proper types
        valid_data = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=10), "y": range(10)})
        result = _validate_prophet_data_format(valid_data)

        # Check that the structure is correct
        self.assertEqual(result.shape, valid_data.shape)
        self.assertEqual(list(result.columns), list(valid_data.columns))

        # Check that datetime column is properly converted
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["ds"]))

        # Check that numeric columns are converted to float64 (required for Prophet)
        self.assertEqual(result["y"].dtype, "float64")

        # Check that values are preserved (even if dtypes change for Prophet compatibility)
        self.assertTrue(result["ds"].equals(pd.to_datetime(valid_data["ds"])))
        self.assertTrue(result["y"].equals(pd.Series(range(10), dtype="float64")))

        # Invalid data should raise errors
        with self.assertRaises(ValueError):
            _validate_prophet_data_format(pd.DataFrame({"y": [1, 2, 3]}))  # Missing 'ds'

        with self.assertRaises(ValueError):
            _validate_prophet_data_format(pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=3)}))  # Missing 'y'

        with self.assertRaises(ValueError):
            _validate_prophet_data_format([1, 2, 3])  # Not a DataFrame


if __name__ == "__main__":
    absltest.main()
