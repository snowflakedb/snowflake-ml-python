from abc import ABC, abstractmethod
from typing import Optional

from snowflake.ml.model import _model_meta, type_hints as model_types


class _ModelHandler(ABC):
    """Provides handling for a given type of model defined by `type` class property."""

    handler_type = "_base"
    MODEL_BLOB_FILE = "model.pkl"
    MODEL_ARTIFACTS_DIR = "artifacts"
    DEFAULT_TARGET_METHODS = ["predict"]

    @staticmethod
    @abstractmethod
    def can_handle(model: model_types.ModelType) -> bool:
        """Whether this handler could support the type of the `model`.

        Args:
            model: The model object.
        """
        ...

    @staticmethod
    @abstractmethod
    def _save_model(
        name: str,
        model: model_types.ModelType,
        model_meta: _model_meta.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
    ) -> None:
        """Save the model.

        Args:
            name: Name of the model.
            model: The model object.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the model.
            sample_input: Sample input to infer the signatures from.
        """
        ...

    @staticmethod
    @abstractmethod
    def _load_model(
        name: str, model_meta: _model_meta.ModelMetadata, model_blobs_dir_path: str
    ) -> model_types.ModelType:
        """Load the model into memory.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.
        """
        ...
