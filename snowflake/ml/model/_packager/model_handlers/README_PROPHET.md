# Prophet Model Support in Snowflake ML Registry

This document describes the complete implementation of Prophet model support in the Snowflake ML Model Registry, including usage examples and implementation details.

## Overview

Facebook Prophet is a powerful time series forecasting library that automatically handles:
- Missing data and outliers
- Trend changes and seasonality
- Holiday effects
- Additional regressors

The Snowflake ML Registry now provides native support for Prophet models through the `ProphetHandler`, enabling seamless registration, versioning, and deployment of time series forecasting models.

## Key Features

### ✅ Full Prophet Support
- Native registration of Prophet models
- Automatic signature inference for time series data
- Support for both `predict()` and `predict_components()` methods
- Proper serialization/deserialization using cloudpickle
- Dependency management for Prophet environment

### ✅ Time Series Specific Handling
- Validates Prophet's required data format (`ds` and `y` columns)
- Handles future date forecasting with NaN values
- Supports additional regressors and holiday effects
- Preserves Prophet model configuration during save/load

### ✅ Consistent Integration
- Follows the same patterns as other supported model types (XGBoost, scikit-learn, etc.)
- Auto-discovery through the existing handler registry system
- Full compatibility with Snowflake's deployment platforms

## Data Format Requirements

### Training Data Format
Prophet requires pandas DataFrame with specific columns:

```python
import pandas as pd

# Required format for Prophet
training_data = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365, freq='D'),  # Date column (required)
    'y': your_time_series_values,                             # Value column (required)
    'holiday': holiday_indicators,                            # Optional regressors
    'temperature': temperature_values,                        # Optional regressors
})
```

### Forecasting Data Format (Critical!)
**For forecasting, provide future dates with NaN values for 'y' column:**

```python
# Create future data for forecasting - this is the key pattern!
future_data = pd.DataFrame({
    'ds': pd.date_range('2021-01-01', periods=30, freq='D'),  # Future dates
    'y': [float('nan')] * 30,                                 # NaN = forecast these periods
    'holiday': future_holiday_values,                         # Future regressor values if used
    'temperature': future_temperature_values,                 # Future regressor values if used
})
```

## Complete Usage Example

### 1. Train Prophet Model

```python
import pandas as pd
import prophet

# Create sample time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
values = range(365)  # Your actual time series values

training_data = pd.DataFrame({
    'ds': dates,
    'y': values
})

# Train Prophet model
model = prophet.Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)
model.fit(training_data)
```

### 2. Register in Snowflake ML Registry

```python
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

# Create session and registry
session = Session.builder.configs(your_connection_params).create()
registry = Registry(session=session)

# Register Prophet model
model_version = registry.log_model(
    model=model,
    model_name="sales_forecast_prophet",
    version_name="v1",
    sample_input_data=training_data,  # Required for signature inference
    comment="Prophet model for daily sales forecasting",
    
    # Prophet-specific options
    options={
        "target_methods": ["predict", "predict_components"],
        "relax_version": True,  # Allow flexible dependency versions
    }
)

print(f"Prophet model registered: {model_version.fully_qualified_model_name}")
```

### 3. Use for Forecasting

```python
# Create future data for forecasting (30 days ahead)
future_data = pd.DataFrame({
    'ds': pd.date_range('2021-01-01', periods=30, freq='D'),
    'y': [float('nan')] * 30  # NaN values indicate periods to forecast
})

# Get the registered model
model_ref = registry.get_model("sales_forecast_prophet").default

# Generate forecasts
forecasts = model_ref.run(future_data, function_name="predict")

print("Forecast results:")
print(forecasts[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Get trend components (optional)
components = model_ref.run(future_data, function_name="predict_components")
print("Trend components:")
print(components[['ds', 'trend', 'weekly', 'yearly']].head())
```

## Advanced Usage: Prophet with Regressors

```python
import numpy as np

# Training data with additional regressors
training_data = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365, freq='D'),
    'y': your_values,
    'holiday': holiday_indicators,      # Binary holiday flag
    'temperature': temperature_data,    # Continuous regressor
    'promotion': promotion_effects      # Another regressor
})

# Train Prophet with regressors
model = prophet.Prophet()
model.add_regressor('holiday')
model.add_regressor('temperature') 
model.add_regressor('promotion')
model.fit(training_data)

# Register model (same as above)
model_version = registry.log_model(model=model, ...)

# For forecasting with regressors, provide future values
future_data = pd.DataFrame({
    'ds': pd.date_range('2021-01-01', periods=30, freq='D'),
    'y': [float('nan')] * 30,
    'holiday': future_holiday_values,      # Must provide future regressor values
    'temperature': future_temperature,     # Must provide future regressor values  
    'promotion': future_promotion_effects  # Must provide future regressor values
})

# Generate forecasts
forecasts = model_ref.run(future_data, function_name="predict")
```

## Available Methods

### `predict` Method
- **Purpose**: Main forecasting method
- **Function Type**: `TABLE_FUNCTION` (processes entire time series at once)
- **Input**: DataFrame with `ds` (dates) and `y` (NaN for future periods)
- **Output**: DataFrame with forecast columns:
  - `ds`: Dates
  - `yhat`: Point forecasts
  - `yhat_lower`: Lower bound of prediction interval
  - `yhat_upper`: Upper bound of prediction interval

### `predict_components` Method
- **Purpose**: Trend decomposition and analysis
- **Function Type**: `TABLE_FUNCTION` (processes entire time series at once)
- **Input**: Same as `predict`
- **Output**: DataFrame with trend components:
  - `ds`: Dates
  - `trend`: Overall trend component
  - `weekly`: Weekly seasonality
  - `yearly`: Yearly seasonality
  - Additional components if holidays/regressors are used

## ⚠️ **Critical Architecture Difference: TABLE_FUNCTION vs FUNCTION**

### **Prophet Models: TABLE_FUNCTION Required**
Prophet models **MUST** use `TABLE_FUNCTION` because:
- **Time Series Context**: Prophet needs to see the entire sequence of dates
- **Batch Processing**: Forecasts are generated for multiple time periods together
- **Seasonality Detection**: Requires historical patterns across the full dataset

```sql
-- Prophet inference in Snowflake (TABLE_FUNCTION)
SELECT * FROM TABLE(my_prophet_model!predict(
  SELECT ds, y FROM future_dates_table  -- Entire table of future dates
));
```

### **Traditional ML Models: Regular FUNCTION**
XGBoost, sklearn models use regular `FUNCTION`:
- **Row-by-Row**: Each prediction is independent
- **Single Input**: One feature vector → One prediction

```sql
-- Traditional ML inference in Snowflake (FUNCTION)
SELECT my_xgb_model!predict(feature1, feature2, feature3) FROM data_table;
```

### **Why This Matters**
❌ **Won't Work**: Trying to call Prophet row-by-row
```python
# This breaks Prophet's assumptions
for row in data:
    prediction = prophet_model.predict([row])  # Missing time series context!
```

✅ **Correct**: Prophet with full time series
```python
# Prophet needs entire time series context
predictions = prophet_model.predict(entire_future_dataframe)
```

## Implementation Details

### Handler Architecture
The Prophet support is implemented through `ProphetHandler` which:

1. **Auto-Discovery**: Automatically registered via the existing handler registry system
2. **Type Detection**: Uses `type_utils.LazyType("prophet.Prophet").isinstance(model)` to identify Prophet models
3. **Serialization**: Uses `cloudpickle` for reliable model persistence
4. **Signature Inference**: Creates time series-specific signatures
5. **Dependency Management**: Automatically adds Prophet, pandas, and numpy dependencies

### File Structure
```
snowflake/ml/model/_packager/model_handlers/
├── prophet.py                    # Main ProphetHandler implementation
├── prophet_examples.py           # Comprehensive usage examples
└── README_PROPHET.md            # This documentation

snowflake/ml/model/_packager/model_handlers_test/
└── prophet_test.py              # Complete test suite

snowflake/ml/model/
└── type_hints.py                # Updated with Prophet types
```

### Key Implementation Features

#### Data Validation
```python
def _validate_prophet_data_format(data):
    """Validates Prophet's required 'ds' and 'y' columns"""
    # Checks for required columns
    # Validates date format in 'ds' column
    # Allows NaN values in 'y' for forecasting
```

#### Signature Creation
```python
def _create_prophet_signature(sample_data):
    """Creates time series-specific model signatures"""
    # Input: ds (TIMESTAMP), y (DOUBLE), regressors (DOUBLE)
    # Output: ds, yhat, yhat_lower, yhat_upper
```

#### Custom Model Wrapper
The handler creates a custom model wrapper that:
- Validates input data format during inference
- Calls the appropriate Prophet method (`predict` or `predict_components`)
- Returns properly formatted output matching the signature

## Best Practices

### 1. Data Preparation
- Ensure consistent date frequency between training and forecasting
- Handle missing values in training data appropriately
- Provide all regressor values for future periods if model uses them

### 2. Model Configuration
- Use appropriate seasonality settings for your data frequency
- Consider adding holidays relevant to your domain
- Test different uncertainty interval settings

### 3. Forecasting Strategy
- Always use NaN values in 'y' column for periods you want to forecast
- Don't forecast too far beyond your training data period
- Validate forecasts against holdout data before deployment

### 4. Registry Usage
- Provide representative sample data for signature inference
- Use meaningful model names and version names
- Add descriptive comments for model documentation

## Troubleshooting

### Common Issues

1. **"Prophet models require 'ds' and 'y' columns"**
   - Solution: Ensure your DataFrame has exactly these column names
   - Check column names with `df.columns.tolist()`

2. **"'ds' column must contain valid datetime values"**
   - Solution: Convert to datetime with `pd.to_datetime(df['ds'])`
   - Ensure consistent date format

3. **Missing regressor values for future periods**
   - Solution: Provide future values for all regressors used in training
   - Example: If model has 'holiday' regressor, future_data must include 'holiday' column

4. **Inconsistent date frequencies**
   - Solution: Match frequency between training and forecasting data
   - Use same `freq` parameter in `pd.date_range()`

5. **"Function not found" or SQL execution errors**
   - **Cause**: Trying to use Prophet with regular FUNCTION syntax instead of TABLE_FUNCTION
   - **Solution**: Use TABLE syntax in Snowflake SQL:
   ```sql
   -- ❌ Wrong (regular function syntax)
   SELECT prophet_model!predict(ds, y) FROM data;
   
   -- ✅ Correct (table function syntax) 
   SELECT * FROM TABLE(prophet_model!predict(
       SELECT ds, y FROM data ORDER BY ds
   ));
   ```

6. **Poor forecast quality or unexpected results**
   - **Cause**: Prophet receiving incomplete or unordered time series data
   - **Solution**: Ensure your input data includes the full date sequence:
   ```python
   # ✅ Provide complete date sequence
   future_data = pd.DataFrame({
       'ds': pd.date_range('2024-01-01', periods=30, freq='D'),  # Complete sequence
       'y': [float('nan')] * 30
   })
   ```

### Performance Tips

- Use daily frequency for most business forecasting
- Consider weekly/monthly aggregation for long-term forecasts
- Prophet works best with at least several months of training data
- Be mindful of computational cost for very high-frequency data

## Migration from Custom Models

If you previously used Prophet through custom model wrappers, migration is straightforward:

### Before (Custom Model)
```python
# Previous approach using custom model
class ProphetWrapper(CustomModel):
    def __init__(self, context):
        super().__init__(context)
        self.model = context.model_ref("prophet_model.pkl")
    
    def predict(self, X):
        return self.model.predict(X)

# Register as custom model
registry.log_model(ProphetWrapper, ...)
```

### After (Native Support)
```python
# New approach with native support
import prophet

model = prophet.Prophet()
model.fit(training_data)

# Direct registration - much simpler!
registry.log_model(model, model_name="prophet_model", ...)
```

## Conclusion

The native Prophet support in Snowflake ML Registry provides a seamless experience for time series forecasting workflows. The implementation handles all Prophet-specific requirements while maintaining consistency with other supported model types.

For additional examples and advanced usage patterns, see `prophet_examples.py` in this directory.
