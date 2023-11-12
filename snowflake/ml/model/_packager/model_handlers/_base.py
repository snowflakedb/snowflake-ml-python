from abc import abstractmethod
from typing import Dict, Generic, Optional, Protocol, Type, final

from typing_extensions import TypeGuard, Unpack

from snowflake.ml.model import custom_model, type_hints as model_types
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import model_meta


class _BaseModelHandlerProtocol(Protocol[model_types._ModelType]):
    HANDLER_TYPE: model_types.SupportedModelHandlerType
    HANDLER_VERSION: str
    _MIN_SNOWPARK_ML_VERSION: str
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]]

    @classmethod
    @abstractmethod
    def can_handle(cls, model: model_types.SupportedDataType) -> TypeGuard[model_types._ModelType]:
        """Whether this handler could support the type of the `model`.

        Args:
            model: The model object.

        Raises:
            NotImplementedError: Not Implemented
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def cast_model(cls, model: model_types.SupportedModelType) -> model_types._ModelType:
        """Cast the model from Union type into the type that handler could handle.

        Args:
            model: The model object.

        Raises:
            NotImplementedError: Not Implemented
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def save_model(
        cls,
        name: str,
        model: model_types._ModelType,
        model_meta: model_meta.ModelMetadata,
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

        Raises:
            NotImplementedError: Not Implemented
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> model_types._ModelType:
        """Load the model into memory.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.
            kwargs: Options when loading the model.

        Raises:
            NotImplementedError: Not Implemented
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_as_custom_model(
        cls,
        raw_model: model_types._ModelType,
        model_meta: model_meta.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        """Create a custom model class wrap for unified interface when being deployed. The predict method will be
        re-targeted based on target_method metadata.

        Args:
            raw_model: original model object,
            model_meta: The model metadata.
            kwargs: Options when converting the model.

        Raises:
            NotImplementedError: Not Implemented
        """
        raise NotImplementedError


class BaseModelHandler(Generic[model_types._ModelType], _BaseModelHandlerProtocol[model_types._ModelType]):
    """
    Provides handling for a given type of model defined by `HANDLER_TYPE` class property.

    HANDLER_TYPE: The string type that identify the handler. Should be unique in the library.
    HANDLER_VERSION: The version of the handler.
    _MIN_SNOWPARK_ML_VERSION: The minimal version of Snowpark ML library to use the current handler.
    _HANDLER_MIGRATOR_PLANS: Dict holding handler migrator plans.

    MODELE_BLOB_FILE_OR_DIR: Relative path of the model blob file in the model subdir. Default to "model.pkl".
    MODEL_ARTIFACTS_DIR: Relative path of the model artifacts dir in the model subdir. Default to "artifacts"
    DEFAULT_TARGET_METHODS: Default target methods to be logged if not specified in this kind of model. Default to
        ["predict"]
    IS_AUTO_SIGNATURE: Set to True if the model could get model signature automatically and do not require user
        inputting sample data or model signature. Default to False.
    """

    MODELE_BLOB_FILE_OR_DIR = "model.pkl"
    MODEL_ARTIFACTS_DIR = "artifacts"
    DEFAULT_TARGET_METHODS = ["predict"]
    IS_AUTO_SIGNATURE = False

    @classmethod
    @final
    def try_upgrade(cls, name: str, model_meta: model_meta.ModelMetadata, model_blobs_dir_path: str) -> None:
        """Try upgrade the stored model to adapt latest handler

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.

        Raises:
            RuntimeError: Raised when there is no corresponding migrator available.
        """
        while model_meta.models[name].handler_version != cls.HANDLER_VERSION:
            if model_meta.models[name].handler_version not in cls._HANDLER_MIGRATOR_PLANS.keys():
                raise RuntimeError(
                    f"Can not find migrator to migrate model {name} from {model_meta.models[name].handler_version}"
                    f" to version {cls.HANDLER_VERSION}."
                )
            migrator = cls._HANDLER_MIGRATOR_PLANS[model_meta.models[name].handler_version]()
            migrator.try_upgrade(
                name=name,
                model_meta=model_meta,
                model_blobs_dir_path=model_blobs_dir_path,
            )
