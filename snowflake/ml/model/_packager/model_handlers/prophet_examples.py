"""
Examples demonstrating how to use Prophet models with Snowflake ML Model Registry.

This module provides comprehensive examples showing how to:
1. Train Prophet models
2. Register them in the Snowflake Model Registry
3. Use them for forecasting with proper future date handling

Prophet models require specific data format and forecasting approaches.
"""

from typing import Optional

import pandas as pd

# Import the required Snowflake ML components
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session


def create_sample_time_series_data(start_date: str = "2020-01-01", periods: int = 365, freq: str = "D") -> pd.DataFrame:
    """Create sample time series data in Prophet format.

    Args:
        start_date: Start date for the time series
        periods: Number of time periods to generate
        freq: Frequency string (D=daily, W=weekly, M=monthly, etc.)

    Returns:
        DataFrame with 'ds' (date) and 'y' (value) columns
    """
    dates = pd.date_range(start_date, periods=periods, freq=freq)

    # Create synthetic data with trend and seasonality
    import numpy as np

    trend = np.linspace(100, 200, periods)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    noise = np.random.normal(0, 5, periods)

    values = trend + seasonal + noise

    return pd.DataFrame({"ds": dates, "y": values})


def create_future_forecast_data(
    last_date: str, forecast_periods: int = 30, freq: str = "D", include_regressors: Optional[dict] = None
) -> pd.DataFrame:
    """Create future data for forecasting with Prophet.

    This is the key pattern for Prophet forecasting: provide future dates
    with NaN values for 'y' column to indicate periods to forecast.

    Args:
        last_date: The last date in your training data
        forecast_periods: Number of periods to forecast
        freq: Frequency string matching your training data
        include_regressors: Optional dict of regressor columns with future values

    Returns:
        DataFrame with future dates and NaN values for forecasting
    """
    # Create future dates starting after the last training date
    last_dt = pd.to_datetime(last_date)
    future_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=forecast_periods, freq=freq)

    # Create future dataframe with NaN y values
    future_df = pd.DataFrame(
        {"ds": future_dates, "y": [float("nan")] * forecast_periods}  # NaN indicates periods to forecast
    )

    # Add any additional regressors if provided
    if include_regressors:
        for col_name, values in include_regressors.items():
            if len(values) != forecast_periods:
                raise ValueError(f"Regressor '{col_name}' must have {forecast_periods} values")
            future_df[col_name] = values

    return future_df


def example_basic_prophet_usage():
    """Example: Basic Prophet model training and registration."""

    # Step 1: Import Prophet
    import prophet

    # Step 2: Create sample training data
    training_data = create_sample_time_series_data(start_date="2020-01-01", periods=365, freq="D")

    print("Training data format (Prophet requires 'ds' and 'y' columns):")
    print(training_data.head())
    print(f"Data shape: {training_data.shape}")

    # Step 3: Train Prophet model
    model = prophet.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(training_data)

    # Step 4: Create future data for forecasting (KEY CONCEPT)
    # For forecasting, provide dates with NaN 'y' values
    future_data = create_future_forecast_data(
        last_date="2020-12-31", forecast_periods=30, freq="D"  # Last date in training data  # Forecast 30 days ahead
    )

    print("\nFuture data format for forecasting:")
    print(future_data.head())
    print("Notice: 'y' column has NaN values - this tells Prophet to forecast these periods")

    # Step 5: Test the model works
    forecast = model.predict(future_data)
    print(f"\nForecast output shape: {forecast.shape}")
    print("Key forecast columns: ds, yhat, yhat_lower, yhat_upper")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())

    return model, training_data, future_data


def example_register_prophet_model(session: Session):
    """Example: Register Prophet model in Snowflake ML Registry."""

    # Train the model
    model, training_data, future_data = example_basic_prophet_usage()

    # Create registry
    registry = Registry(session=session)

    # Register the Prophet model
    model_version = registry.log_model(
        model=model,
        model_name="sales_forecast_prophet",
        version_name="v1",
        sample_input_data=training_data,  # Provide sample data for signature inference
        comment="Prophet model for daily sales forecasting",
        # Prophet-specific options
        options={
            "target_methods": ["predict", "predict_components"],  # Available methods
            "relax_version": True,  # Allow flexible dependency versions
            # Note: Prophet methods automatically use TABLE_FUNCTION type
            # because they require entire time series context
        },
    )

    print(f"Prophet model registered: {model_version.fully_qualified_model_name}")
    print("Note: Prophet methods automatically configured as TABLE_FUNCTION")
    print("Use TABLE syntax in Snowflake SQL: SELECT * FROM TABLE(model!predict(...))")
    return model_version


def example_prophet_with_regressors():
    """Example: Prophet model with additional regressors."""

    import prophet

    # Create training data with additional regressors
    training_data = create_sample_time_series_data(periods=365)

    # Add holiday indicator and temperature data
    training_data["holiday"] = 0  # 0=normal day, 1=holiday
    training_data.loc[training_data["ds"].dt.dayofweek >= 5, "holiday"] = 1  # weekends
    training_data["temperature"] = 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365.25)

    print("Training data with regressors:")
    print(training_data.head())

    # Train Prophet with additional regressors
    model = prophet.Prophet()
    model.add_regressor("holiday")
    model.add_regressor("temperature")
    model.fit(training_data)

    # Create future data with regressor values
    future_data = create_future_forecast_data(
        last_date="2020-12-31",
        forecast_periods=30,
        include_regressors={
            "holiday": [1, 0, 0, 1, 0, 0, 0] * 4 + [1, 0],  # Sample holiday pattern
            "temperature": [25.0] * 30,  # Sample future temperatures
        },
    )

    print("\nFuture data with regressors:")
    print(future_data.head())

    # Test prediction
    forecast = model.predict(future_data)
    print(f"\nForecast with regressors completed: {forecast.shape}")

    return model, training_data, future_data


def example_prophet_model_inference(session: Session, model_name: str):
    """Example: Using registered Prophet model for inference."""

    # Get the registered model
    registry = Registry(session=session)
    model_version = registry.get_model(model_name).default

    # Create future data for forecasting
    # This is what users need to provide for inference
    inference_data = create_future_forecast_data(
        last_date="2021-12-31",  # Extend beyond original training period
        forecast_periods=60,  # Forecast 2 months ahead
        freq="D",
    )

    print("Inference data format:")
    print(inference_data.head())
    print("\nKey points for Prophet inference:")
    print("1. 'ds' column: dates you want forecasts for")
    print("2. 'y' column: should be NaN for future periods to forecast")
    print("3. Additional regressors: provide future values if model was trained with them")

    # Perform inference
    predictions = model_version.run(inference_data, function_name="predict")  # Use predict method

    print(f"\nPredictions received: {predictions.shape}")
    print("Prophet forecast output includes:")
    print("- yhat: point forecast")
    print("- yhat_lower: lower bound of prediction interval")
    print("- yhat_upper: upper bound of prediction interval")

    # Get trend components (if available)
    try:
        components = model_version.run(inference_data, function_name="predict_components")
        print(f"\nTrend components available: {components.shape}")
        print("Components typically include: trend, weekly, yearly seasonality")
    except Exception as e:
        print(f"Components not available: {e}")

    return predictions


# User Guide Documentation
PROPHET_USER_GUIDE = """
# Prophet Model Usage Guide for Snowflake ML Registry

## Overview
Prophet is Facebook's time series forecasting library that excels at handling:
- Missing data points
- Trend changes
- Seasonal patterns (daily, weekly, yearly)
- Holiday effects
- Additional regressors

## ⚠️ CRITICAL: Prophet Uses TABLE_FUNCTION Architecture

**Prophet models are fundamentally different from traditional ML models:**

### Traditional ML (XGBoost, sklearn): Row-by-Row Processing
```sql
-- Each row processed independently
SELECT xgb_model!predict(feature1, feature2) FROM input_table;
```

### Prophet: Batch Time Series Processing
```sql
-- Entire time series processed together (TABLE_FUNCTION)
SELECT * FROM TABLE(prophet_model!predict(
    SELECT ds, y FROM future_periods_table
));
```

**Why Prophet Needs TABLE_FUNCTION:**
- Time series context is essential
- Seasonality patterns require full dataset
- Forecasts depend on temporal relationships
- Cannot meaningfully predict single time points in isolation

## Key Data Format Requirements

### Training Data
Prophet requires pandas DataFrame with specific columns:
```python
training_df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365),  # Date column (required)
    'y': your_time_series_values,                    # Value column (required)
    'regressor1': additional_features,               # Optional regressors
})
```

### Forecasting Data (IMPORTANT!)
For forecasting, provide future dates with NaN values for 'y':
```python
future_df = pd.DataFrame({
    'ds': pd.date_range('2021-01-01', periods=30),  # Future dates
    'y': [float('nan')] * 30,                       # NaN = forecast these periods
    'regressor1': future_regressor_values,          # Future regressor values if used
})
```

## Registration Process

1. **Train your Prophet model:**
```python
import prophet
model = prophet.Prophet()
model.fit(training_data)
```

2. **Register in Snowflake ML:**
```python
from snowflake.ml.registry import Registry
registry = Registry(session=session)

model_version = registry.log_model(
    model=model,
    model_name="my_prophet_model",
    sample_input_data=training_data,  # Required for signature inference
    options={
        "target_methods": ["predict", "predict_components"]
    }
)
```

3. **Use for inference:**
```python
# Create future data with NaN y values
future_data = pd.DataFrame({
    'ds': pd.date_range('2021-01-01', periods=30),
    'y': [float('nan')] * 30
})

# Get forecasts using Python API
predictions = model_version.run(future_data, function_name="predict")

# Or use directly in Snowflake SQL (TABLE_FUNCTION syntax):
# SELECT * FROM TABLE(my_prophet_model!predict(
#     SELECT
#         date_column as ds,
#         NULL as y  -- NULL/NaN indicates forecast periods
#     FROM my_future_dates_table
#     ORDER BY ds
# ));
```

## Available Methods

- **predict**: Main forecasting method
  - Returns: ds, yhat, yhat_lower, yhat_upper

- **predict_components**: Trend decomposition
  - Returns: ds, trend, weekly, yearly, holidays (if applicable)

## Best Practices

1. **Date Range Planning**: Always ensure your future dates extend beyond training data
2. **Regressor Handling**: If your model uses regressors, provide future values for all of them
3. **Frequency Consistency**: Match the frequency of your future data to training data
4. **Validation**: Test your model with sample future data before registration

## Common Pitfalls

- Forgetting to set 'y' values to NaN for forecasting periods
- Mismatched date frequencies between training and inference data
- Missing future values for regressors when model was trained with them
- Providing past dates instead of future dates for forecasting

## Example Complete Workflow

See the example functions in this module for complete working examples:
- `example_basic_prophet_usage()`: Basic Prophet training and forecasting
- `example_prophet_with_regressors()`: Using additional features
- `example_register_prophet_model()`: Full registration workflow
- `example_prophet_model_inference()`: Using registered models
"""


if __name__ == "__main__":
    print("Prophet Examples for Snowflake ML")
    print("=" * 50)

    # Run basic example
    print("\n1. Basic Prophet Usage:")
    model, training_data, future_data = example_basic_prophet_usage()

    print("\n2. Prophet with Regressors:")
    import numpy as np

    model_reg, training_reg, future_reg = example_prophet_with_regressors()

    print("\n" + "=" * 50)
    print("For full documentation, see PROPHET_USER_GUIDE in this module")
    print("For Snowflake integration, ensure you have an active Snowpark session")
