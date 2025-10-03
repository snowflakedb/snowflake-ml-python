"""Volatility definitions for model functions."""

from enum import Enum, auto


class Volatility(Enum):
    """Volatility levels for model functions.

    Attributes:
        VOLATILE: Function results may change between calls with the same arguments.
            Use this for functions that depend on external data or have non-deterministic behavior.
        IMMUTABLE: Function results are guaranteed to be the same for the same arguments.
            Use this for pure functions that always return the same output for the same input.
    """

    VOLATILE = auto()
    IMMUTABLE = auto()


DEFAULT_VOLATILITY_BY_MODEL_TYPE = {
    "catboost": Volatility.IMMUTABLE,
    "custom": Volatility.VOLATILE,
    "huggingface_pipeline": Volatility.IMMUTABLE,
    "keras": Volatility.IMMUTABLE,
    "lightgbm": Volatility.IMMUTABLE,
    "mlflow": Volatility.IMMUTABLE,
    "pytorch": Volatility.IMMUTABLE,
    "sentence_transformers": Volatility.IMMUTABLE,
    "sklearn": Volatility.IMMUTABLE,
    "snowml": Volatility.IMMUTABLE,
    "tensorflow": Volatility.IMMUTABLE,
    "torchscript": Volatility.IMMUTABLE,
    "xgboost": Volatility.IMMUTABLE,
}
