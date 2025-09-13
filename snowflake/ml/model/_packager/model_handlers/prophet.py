# mypy: disable-error-code="import"
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


def _validate_prophet_data_format(data: model_types.SupportedDataType) -> pd.DataFrame:
    """Validate that input data follows Prophet's required format with 'ds' and 'y' columns.

    Args:
        data: Input data to validate

    Returns:
        DataFrame with validated Prophet format

    Raises:
        ValueError: If data doesn't meet Prophet requirements
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Prophet models require pandas DataFrame input with 'ds' and 'y' columns")

    if "ds" not in data.columns:
        raise ValueError("Prophet models require a 'ds' column containing dates")

    if "y" not in data.columns:
        # Allow 'y' column with NaN values for future predictions
        raise ValueError("Prophet models require a 'y' column containing values (can be NaN for future periods)")

    # Validate date column
    try:
        pd.to_datetime(data["ds"])
    except Exception as e:
        raise ValueError(f"'ds' column must contain valid datetime values: {e}")

    return data


def _create_prophet_signature(sample_data: pd.DataFrame) -> model_signature.ModelSignature:
    """Create model signature specific to Prophet time series format.

    Args:
        sample_data: Sample Prophet-formatted data

    Returns:
        ModelSignature for Prophet model
    """
    # Prophet requires specific input format
    inputs = [
        model_signature.FeatureSpec(name="ds", dtype=model_signature.DataType.TIMESTAMP),
        model_signature.FeatureSpec(name="y", dtype=model_signature.DataType.DOUBLE),
    ]

    # Add any additional regressors found in the data
    for col in sample_data.columns:
        if col not in ["ds", "y"]:
            inputs.append(model_signature.FeatureSpec(name=col, dtype=model_signature.DataType.DOUBLE))

    # Prophet forecast output format
    outputs = [
        model_signature.FeatureSpec(name="ds", dtype=model_signature.DataType.TIMESTAMP),
        model_signature.FeatureSpec(name="yhat", dtype=model_signature.DataType.DOUBLE),
        model_signature.FeatureSpec(name="yhat_lower", dtype=model_signature.DataType.DOUBLE),
        model_signature.FeatureSpec(name="yhat_upper", dtype=model_signature.DataType.DOUBLE),
    ]

    return model_signature.ModelSignature(inputs=inputs, outputs=outputs)


def _create_prophet_components_signature(sample_data: pd.DataFrame) -> model_signature.ModelSignature:
    """Create model signature for Prophet's predict_components method.

    Args:
        sample_data: Sample Prophet-formatted data

    Returns:
        ModelSignature for Prophet components
    """
    # Same inputs as predict
    inputs = [
        model_signature.FeatureSpec(name="ds", dtype=model_signature.DataType.TIMESTAMP),
        model_signature.FeatureSpec(name="y", dtype=model_signature.DataType.DOUBLE),
    ]

    # Add any additional regressors
    for col in sample_data.columns:
        if col not in ["ds", "y"]:
            inputs.append(model_signature.FeatureSpec(name=col, dtype=model_signature.DataType.DOUBLE))

    # Prophet components output (basic components - actual components depend on model config)
    outputs = [
        model_signature.FeatureSpec(name="ds", dtype=model_signature.DataType.TIMESTAMP),
        model_signature.FeatureSpec(name="trend", dtype=model_signature.DataType.DOUBLE),
        model_signature.FeatureSpec(name="weekly", dtype=model_signature.DataType.DOUBLE),
        model_signature.FeatureSpec(name="yearly", dtype=model_signature.DataType.DOUBLE),
    ]

    return model_signature.ModelSignature(inputs=inputs, outputs=outputs)


@final
class ProphetHandler(_base.BaseModelHandler["prophet.Prophet"]):
    """Handler for Facebook Prophet time series forecasting models.

    Prophet is a time series forecasting library that handles missing data,
    trend changes, and seasonality automatically. It requires input data with
    'ds' (datestamp) and 'y' (value) columns.
    """

    HANDLER_TYPE = "prophet"
    HANDLER_VERSION = "2025-01-01"
    _MIN_SNOWPARK_ML_VERSION = "1.8.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model.pkl"
    DEFAULT_TARGET_METHODS = ["predict", "predict_components"]

    # Prophet models can automatically infer signature from data format
    IS_AUTO_SIGNATURE = False  # We'll handle signature creation manually

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
        return type_utils.LazyType("prophet.Prophet").isinstance(model)

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
            **kwargs: Additional save options
        """
        import prophet

        assert isinstance(model, prophet.Prophet)

        if not is_sub_model:
            # Validate sample input data format
            if sample_input_data is not None:
                if isinstance(sample_input_data, pd.DataFrame):
                    sample_input_data = _validate_prophet_data_format(sample_input_data)
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

                # Validate Prophet data format
                validated_data = _validate_prophet_data_format(sample_input_data)

                target_method = getattr(model, target_method_name, None)
                if not callable(target_method):
                    raise ValueError(f"Method {target_method_name} not found on Prophet model")

                # For signature inference, we need to create future dataframe
                if target_method_name == "predict":
                    # Use the input data as the future dataframe for prediction
                    predictions = target_method(validated_data)
                elif target_method_name == "predict_components":
                    predictions = target_method(validated_data)
                else:
                    raise ValueError(f"Unsupported target method: {target_method_name}")

                return predictions

            # Create signatures manually for Prophet-specific format
            signatures = {}
            if sample_input_data is not None:
                for method_name in target_methods:
                    if method_name == "predict":
                        signatures[method_name] = _create_prophet_signature(sample_input_data)
                    elif method_name == "predict_components":
                        signatures[method_name] = _create_prophet_components_signature(sample_input_data)

            # Update model_meta with signatures
            if signatures:
                if model_meta.signatures is None:
                    model_meta.signatures = {}
                model_meta.signatures.update(signatures)

            # Set function properties for Prophet methods - they must be TABLE_FUNCTION
            # Prophet requires entire time series context, cannot work row-by-row
            if model_meta.function_properties is None:
                model_meta.function_properties = {}

            for method_name in target_methods:
                model_meta.function_properties[method_name] = {
                    "function_type": "TABLE_FUNCTION"  # Prophet must process entire table at once
                }

            # Determine task type (always forecasting for Prophet)
            model_meta.task = model_types.Task.UNKNOWN  # Prophet is forecasting, which isn't in standard tasks

        # Save the Prophet model using cloudpickle
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)

        with open(os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR), "wb") as f:
            cloudpickle.dump(model, f)

        # Create model blob metadata
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
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

        def _create_custom_model(
            raw_model: "prophet.Prophet",
            model_meta: model_meta_api.ModelMetadata,
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
                    - 'ds' column: dates for which to generate forecasts
                    - 'y' column: can be NaN for future periods to forecast
                    - Additional regressor columns if the model was trained with them
                    """
                    # Validate input format
                    validated_data = _validate_prophet_data_format(X)

                    # Generate predictions using Prophet
                    if target_method == "predict":
                        predictions = raw_model.predict(validated_data)
                    elif target_method == "predict_components":
                        predictions = raw_model.predict_components(validated_data)
                    else:
                        raise ValueError(f"Unsupported method: {target_method}")

                    # Ensure the output matches the expected signature
                    return model_signature_utils.rename_pandas_df(predictions, signature.outputs)

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

        _ProphetModel = _create_custom_model(raw_model, model_meta)
        prophet_model = _ProphetModel(custom_model.ModelContext())

        return prophet_model
