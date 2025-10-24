import logging
import os
from typing import TYPE_CHECKING, Callable, Optional, cast, final

import cloudpickle
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
)
from snowflake.ml.model._signatures import utils as model_signature_utils

if TYPE_CHECKING:
    import prophet

logger = logging.getLogger(__name__)


def _normalize_column_names(
    data: pd.DataFrame,
    date_column: Optional[str] = None,
    target_column: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize user column names to Prophet's required 'ds' and 'y' format.

    Args:
        data: Input DataFrame with user's column names
        date_column: Name of the date column to map to 'ds'
        target_column: Name of the target column to map to 'y'

    Returns:
        DataFrame with columns renamed to Prophet format

    Raises:
        ValueError: If specified columns don't exist or if there are naming conflicts
    """
    if date_column is None and target_column is None:
        return data

    data = data.copy()

    if date_column is not None:
        if date_column not in data.columns:
            raise ValueError(
                f"Specified date_column '{date_column}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}"
            )
        if date_column != "ds":
            # Check if 'ds' already exists as a different column
            if "ds" in data.columns:
                raise ValueError(
                    f"Cannot rename '{date_column}' to 'ds' because 'ds' already exists in the DataFrame. "
                    f"Please either: (1) rename or remove the existing 'ds' column, or "
                    f"(2) if 'ds' is your date column, set date_column='ds' instead."
                )
            data = data.rename(columns={date_column: "ds"})

    if target_column is not None:
        if target_column not in data.columns:
            raise ValueError(
                f"Specified target_column '{target_column}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}"
            )
        if target_column != "y":
            # Check if 'y' already exists as a different column
            if "y" in data.columns:
                raise ValueError(
                    f"Cannot rename '{target_column}' to 'y' because 'y' already exists in the DataFrame. "
                    f"Please either: (1) rename or remove the existing 'y' column, or "
                    f"(2) if 'y' is your target column, set target_column='y' instead."
                )
            data = data.rename(columns={target_column: "y"})

    return data


def _sanitize_prophet_output(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sanitize Prophet prediction output to have SQL-safe column names.

    Prophet may include holiday columns with names containing spaces (e.g., "Christmas Day")
    which cannot be used as unquoted SQL identifiers in Snowflake. This function normalizes all
    column names to be valid unquoted SQL identifiers by replacing spaces with underscores and
    removing special characters.

    Args:
        predictions: Raw prediction DataFrame from Prophet

    Returns:
        DataFrame with normalized SQL-safe column names

    Raises:
        ValueError: If predictions DataFrame is empty, has no columns, or is missing required
            columns 'ds' and 'yhat'
    """
    # Check if predictions is empty or has no columns
    if predictions is None or len(predictions.columns) == 0:
        raise ValueError(
            f"Prophet predictions DataFrame is empty or has no columns. "
            f"DataFrame shape: {predictions.shape if predictions is not None else 'None'}, "
            f"Type: {type(predictions)}"
        )

    if "ds" not in predictions.columns or "yhat" not in predictions.columns:
        raise ValueError(
            f"Prophet predictions missing required columns 'ds' and 'yhat'. "
            f"Available columns: {list(predictions.columns)}"
        )

    # Normalize all column names to be SQL-safe
    normalized_columns = {col: handlers_utils.normalize_column_name(col) for col in predictions.columns}

    # Check for conflicts after normalization
    normalized_values = list(normalized_columns.values())
    if len(normalized_values) != len(set(normalized_values)):
        # Find duplicates
        seen = set()
        duplicates = []
        for val in normalized_values:
            if val in seen:
                duplicates.append(val)
            seen.add(val)

        logger.warning(
            f"Column name normalization resulted in duplicates: {duplicates}. "
            f"Original columns: {[k for k, v in normalized_columns.items() if v in duplicates]}"
        )

    # Rename columns
    sanitized_predictions = predictions.rename(columns=normalized_columns)

    return sanitized_predictions


def _validate_prophet_data_format(data: model_types.SupportedDataType) -> pd.DataFrame:
    """Validate that input data follows Prophet's required format with 'ds' and 'y' columns.

    Args:
        data: Input data to validate

    Returns:
        DataFrame with validated Prophet format and proper data types

    Raises:
        ValueError: If data doesn't meet Prophet requirements
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Prophet models require pandas DataFrame input with 'ds' and 'y' columns")

    if "ds" not in data.columns:
        raise ValueError(
            "Prophet models require a 'ds' column containing dates. "
            "If your date column has a different name, use the 'date_column' parameter "
            "when saving the model to map it to 'ds'."
        )

    if "y" not in data.columns:
        # Allow 'y' column with NaN values for future predictions
        raise ValueError(
            "Prophet models require a 'y' column containing values (can be NaN for future periods). "
            "If your target column has a different name, use the 'target_column' parameter "
            "when saving the model to map it to 'y'."
        )

    validated_data = data.copy()

    # Convert datetime column - this handles string timestamps from Snowflake
    try:
        validated_data["ds"] = pd.to_datetime(validated_data["ds"])
    except Exception as e:
        raise ValueError(f"'ds' column must contain valid datetime values: {e}")

    # Convert numeric columns to proper float types
    for col in validated_data.columns:
        if col != "ds":
            try:
                # Convert to numeric, coercing errors to NaN
                original_col = validated_data[col]
                validated_data[col] = pd.to_numeric(validated_data[col], errors="coerce")

                # Force explicit dtype conversion to ensure numpy operations work
                validated_data[col] = validated_data[col].astype("float64")

                logger.debug(f"Converted column '{col}' from {original_col.dtype} to {validated_data[col].dtype}")

            except Exception as e:
                # If conversion fails completely, provide detailed error
                raise ValueError(
                    f"Column '{col}' contains data that cannot be converted to numeric: {e}. "
                    f"Original dtype: {validated_data[col].dtype}, sample values: {validated_data[col].head().tolist()}"
                )

    return validated_data


@final
class ProphetHandler(_base.BaseModelHandler["prophet.Prophet"]):
    """Handler for prophet time series forecasting models."""

    HANDLER_TYPE = "prophet"
    HANDLER_VERSION = "2025-01-01"
    _MIN_SNOWPARK_ML_VERSION = "1.8.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model.pkl"
    DEFAULT_TARGET_METHODS = ["predict"]

    # Prophet models require sample data to infer signatures because the data may contain regressors.
    IS_AUTO_SIGNATURE = False

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["prophet.Prophet"]:
        """Check if this handler can process the given model.

        Args:
            model: The model object to check

        Returns:
            True if this is a Prophet model, False otherwise
        """
        return type_utils.LazyType("prophet.Prophet").isinstance(model) and any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "prophet.Prophet":
        """Cast the model to Prophet type.

        Args:
            model: The model object

        Returns:
            The model cast as Prophet
        """
        import prophet

        assert isinstance(model, prophet.Prophet)
        return cast("prophet.Prophet", model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "prophet.Prophet",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.ProphetSaveOptions],
    ) -> None:
        """Save Prophet model and metadata.

        Args:
            name: Name of the model
            model: The Prophet model object
            model_meta: The model metadata
            model_blobs_dir_path: Directory to save model files
            sample_input_data: Sample input data for signature inference
            is_sub_model: Whether this is a sub-model
            **kwargs: Additional save options including date_column and target_column for column mapping

        Raises:
            ValueError: If sample_input_data is not a pandas DataFrame or if column mapping fails
        """
        import prophet

        assert isinstance(model, prophet.Prophet)

        date_column = kwargs.pop("date_column", None)
        target_column = kwargs.pop("target_column", None)

        if not is_sub_model:
            # Validate sample input data if provided
            if sample_input_data is not None:
                if isinstance(sample_input_data, pd.DataFrame):
                    # Normalize for validation purposes
                    normalized_sample = _normalize_column_names(
                        sample_input_data.copy(),
                        date_column=date_column,
                        target_column=target_column,
                    )
                    _validate_prophet_data_format(normalized_sample)
                else:
                    raise ValueError("Prophet models require pandas DataFrame sample input data")

            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str,
                sample_input_data: model_types.SupportedLocalDataType,
            ) -> model_types.SupportedLocalDataType:
                """Generate predictions for signature inference."""
                if not isinstance(sample_input_data, pd.DataFrame):
                    raise ValueError("Prophet requires pandas DataFrame input")

                normalized_data = _normalize_column_names(
                    sample_input_data,
                    date_column=date_column,
                    target_column=target_column,
                )
                validated_data = _validate_prophet_data_format(normalized_data)

                target_method = getattr(model, target_method_name, None)
                if not callable(target_method):
                    raise ValueError(f"Method {target_method_name} not found on Prophet model")

                if target_method_name == "predict":
                    # Use the input data as the future dataframe for prediction
                    try:
                        predictions = target_method(validated_data)
                        predictions = _sanitize_prophet_output(predictions)
                        return predictions
                    except Exception as e:
                        if "numpy._core.numeric" in str(e):
                            raise RuntimeError(
                                f"Prophet model logging failed due to NumPy compatibility issue: {e}. "
                                f"Try using compatible NumPy versions (e.g., 1.24.x or 1.26.x) in pip_requirements "
                                f"with relax_version=False."
                            ) from e
                        else:
                            raise
                elif target_method_name == "predict_components":
                    predictions = target_method(validated_data)
                    predictions = _sanitize_prophet_output(predictions)
                else:
                    raise ValueError(f"Unsupported target method: {target_method_name}")

                return predictions

            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )

            model_meta.task = model_types.Task.UNKNOWN  # Prophet is forecasting, which isn't in standard tasks

        # Save the Prophet model using cloudpickle
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)

        with open(os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR), "wb") as f:
            cloudpickle.dump(model, f)

        # Create model blob metadata with column mapping options
        from snowflake.ml.model._packager.model_meta import model_meta_schema

        options: model_meta_schema.ProphetModelBlobOptions = {}
        if date_column is not None:
            options["date_column"] = date_column
        if target_column is not None:
            options["target_column"] = target_column

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=options if options else {},
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        # Add Prophet dependencies
        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(requirement="prophet", pip_name="prophet"),
                model_env.ModelDependency(requirement="pandas", pip_name="pandas"),
                model_env.ModelDependency(requirement="numpy", pip_name="numpy"),
            ],
            check_local_version=True,
        )

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ProphetLoadOptions],
    ) -> "prophet.Prophet":
        """Load Prophet model from storage.

        Args:
            name: Name of the model
            model_meta: The model metadata
            model_blobs_dir_path: Directory containing model files
            **kwargs: Additional load options

        Returns:
            The loaded Prophet model
        """
        import prophet

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path

        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            model = cloudpickle.load(f)

        assert isinstance(model, prophet.Prophet)
        return model

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "prophet.Prophet",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.ProphetLoadOptions],
    ) -> custom_model.CustomModel:
        """Convert Prophet model to CustomModel for unified inference interface.

        Args:
            raw_model: The original Prophet model
            model_meta: The model metadata
            background_data: Background data for explanations (not used for Prophet)
            **kwargs: Additional options

        Returns:
            CustomModel wrapper for the Prophet model
        """
        from snowflake.ml.model import custom_model

        model_blob_meta = next(iter(model_meta.models.values()))
        date_column: Optional[str] = cast(Optional[str], model_blob_meta.options.get("date_column", None))
        target_column: Optional[str] = cast(Optional[str], model_blob_meta.options.get("target_column", None))

        def _create_custom_model(
            raw_model: "prophet.Prophet",
            model_meta: model_meta_api.ModelMetadata,
            date_column: Optional[str],
            target_column: Optional[str],
        ) -> type[custom_model.CustomModel]:
            """Create custom model class for Prophet."""

            def fn_factory(
                raw_model: "prophet.Prophet",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                """Factory function to create method implementations."""

                @custom_model.inference_api
                def predict_fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    """Predict method for Prophet forecasting.

                    For forecasting, users should provide a DataFrame with:
                    - 'ds' column (or custom date column name): dates for which to generate forecasts
                    - 'y' column (or custom target column name): can be NaN for future periods to forecast
                    - Additional regressor columns if the model was trained with them

                    Args:
                        self: The custom model instance
                        X: Input DataFrame with dates and optional target values for forecasting

                    Returns:
                        DataFrame containing Prophet predictions with columns like ds, yhat, yhat_lower, etc.

                    Raises:
                        ValueError: If column normalization fails or method is unsupported
                        RuntimeError: If NumPy compatibility issues are detected during validation or prediction
                    """
                    try:
                        normalized_data = _normalize_column_names(
                            X, date_column=date_column, target_column=target_column
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to normalize column names: {e}") from e

                    # Validate input format with runtime error handling
                    try:
                        validated_data = _validate_prophet_data_format(normalized_data)
                    except Exception as e:
                        if "numpy._core.numeric" in str(e):
                            raise RuntimeError(
                                f"Prophet input validation failed in Snowflake runtime due to "
                                f"NumPy compatibility: {e}. Redeploy model with compatible dependency versions "
                                f"(e.g., NumPy 1.24.x or 1.26.x, Prophet 1.1.x) in pip_requirements "
                                f"with relax_version=False."
                            ) from e
                        else:
                            raise

                    # Generate predictions using Prophet with runtime error handling
                    if target_method == "predict":
                        try:
                            predictions = raw_model.predict(validated_data)
                            # Sanitize output to remove columns with problematic names
                            predictions = _sanitize_prophet_output(predictions)
                        except Exception as e:
                            if "numpy._core.numeric" in str(e) or "np.float_" in str(e):
                                raise RuntimeError(
                                    f"Prophet prediction failed in Snowflake runtime due to NumPy compatibility: {e}. "
                                    f"This indicates Prophet's internal NumPy operations are incompatible. "
                                    f"Redeploy with compatible dependency versions in pip_requirements."
                                ) from e
                            else:
                                raise
                    else:
                        raise ValueError(f"Unsupported method: {target_method}")

                    # Prophet returns many columns, but we only want the ones in our signature
                    # Filter to only the columns we expect based on our signature
                    expected_columns = [spec.name for spec in signature.outputs]
                    available_columns = [col for col in expected_columns if col in predictions.columns]

                    # Fill missing columns with zeros to match the expected signature
                    filtered_predictions = predictions[available_columns].copy()

                    # Add missing columns with zeros if they're not present
                    for col_name in expected_columns:
                        if col_name not in filtered_predictions.columns:
                            # Add missing seasonal component columns with zeros
                            if col_name in ["weekly", "yearly", "daily"]:
                                filtered_predictions[col_name] = 0.0
                            else:
                                # For required columns like ds, yhat, etc., this would be an error
                                raise ValueError(f"Required column '{col_name}' missing from Prophet output")

                    # Reorder columns to match signature order
                    filtered_predictions = filtered_predictions[expected_columns]

                    # Ensure the output matches the expected signature
                    return model_signature_utils.rename_pandas_df(filtered_predictions, signature.outputs)

                return predict_fn

            # Create method dictionary for the custom model class
            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            # Create the custom model class
            _ProphetModel = type(
                "_ProphetModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _ProphetModel

        _ProphetModel = _create_custom_model(raw_model, model_meta, date_column, target_column)
        prophet_model = _ProphetModel(custom_model.ModelContext())

        return prophet_model
