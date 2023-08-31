from abc import ABC, abstractmethod
from typing import Generic, Optional

from typing_extensions import TypeGuard, Unpack

from snowflake.ml.model import _model_meta, type_hints as model_types


class _ModelHandler(ABC, Generic[model_types._ModelType]):
    """
    Provides handling for a given type of model defined by `type` class property.

    handler_type: The string type that identify the handler. Should be unique in the library.
    MODEL_BLOB_FILE: Relative path of the model blob file in the model subdir.
    MODEL_ARTIFACTS_DIR: Relative path of the model artifacts dir in the model subdir.
    DEFAULT_TARGET_METHODS: Default target methods to be logged if not specified in this kind of model.
    is_auto_signature: Set to True if the model could get model signature automatically and do not require user
        inputting sample data or model signature.
    """

    handler_type = "_base"
    MODEL_BLOB_FILE = "model.pkl"
    MODEL_ARTIFACTS_DIR = "artifacts"
    DEFAULT_TARGET_METHODS = ["predict"]
    is_auto_signature = False

    @staticmethod
    @abstractmethod
    def can_handle(model: model_types.SupportedDataType) -> TypeGuard[model_types._ModelType]:
        """Whether this handler could support the type of the `model`.

        Args:
            model: The model object.
        """
        ...

    @staticmethod
    @abstractmethod
    def cast_model(model: model_types.SupportedModelType) -> model_types._ModelType:
        """Cast the model from Union type into the type that handler could handle.

        Args:
            model: The model object.
        """
        ...

    @staticmethod
    @abstractmethod
    def _save_model(
        name: str,
        model: model_types._ModelType,
        model_meta: _model_meta.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.BaseModelSaveOption],
    ) -> None:
        """Save the model.

        Args:
            name: Name of the model.
            model: The model object.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the model.
            sample_input: Sample input to infer the signatures from.
            is_sub_model: Flag to show if it is a sub model, a sub model does not need signature.
            kwargs: Additional saving options.
        """
        ...

    @staticmethod
    @abstractmethod
    def _load_model(
        name: str,
        model_meta: _model_meta.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> model_types._ModelType:
        """Load the model into memory.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.
            kwargs: Options when loading the model.
        """
        ...
