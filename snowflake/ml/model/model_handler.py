import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import cloudpickle

from snowflake.ml.model import model_meta, type_spec, type_util, util

ModelType = Any

MODEL_HANDLER_REGISTRY: Dict[str, Type["ModelHandler"]] = dict()


def register_handler(cls: Type["ModelHandler"]) -> Type["ModelHandler"]:
    MODEL_HANDLER_REGISTRY[cls.type] = cls
    return cls


class BaseModelContainer:
    """Common base class enforcing an interface to be used for autogen inference."""

    # TODO(halu): Type constraint
    def predict(self, X: Any) -> Any:
        pass


class ModelHandler(ABC):
    """Provides handling for a given type of model defined by `type` class property."""

    type = "_base"

    @staticmethod
    @abstractmethod
    def can_handle(model: ModelType) -> bool:
        """Whether this handler could support the tyep of the `model`.

        Args:
            model (ModelType): The model object.
        """
        ...

    @staticmethod
    @abstractmethod
    def infer_schema(model: ModelType, X: Any) -> Tuple[type_spec.TypeSpec, type_spec.TypeSpec]:
        """Infer both input and output schema type based on `model` and sampled input data.

        Args:
            model (ModelType): The model object.
            X (Any): Sample input data.
        """
        ...

    @staticmethod
    @abstractmethod
    def _save_model(
        name: str, model: ModelType, model_meta: model_meta.ModelMetadata, model_dir_path: str, **kwargs: Any
    ) -> None:
        """Save the model.

        Args:
            name (str): Name of the model.
            model (ModelType): The model object.
            model_meta (model_meta.ModelMetadata): The model metadata.
            model_dir_path (str): Directory path to the model.
            kwargs: Additional keyword args.
        """
        ...

    @staticmethod
    @abstractmethod
    def _load_model(name: str, model_meta: model_meta.ModelMetadata, model_dir_path: str) -> ModelType:
        """Load the model into memory.

        Args:
            name (str): Name of the model.
            model_meta (model_meta.ModelMetadata): The model metadata.
            model_dir_path (str): Directory path to the model.
        """
        ...

    @staticmethod
    @abstractmethod
    def _load_model_container(model_meta: model_meta.ModelMetadata, model_dir_path: str) -> BaseModelContainer:
        """Load the model into memory within a container.

        Args:
            model_meta (model_meta.ModelMetadata): The model metadata.
            model_dir_path (str): Directory path to the model.
        """
        ...


def find_handler(model: ModelType) -> Optional[Type[ModelHandler]]:
    for handler in MODEL_HANDLER_REGISTRY.values():
        if handler.can_handle(model):
            return handler
    return None


def load_handler(target_model_type: str) -> Optional[Type[ModelHandler]]:
    for model_type, handler in MODEL_HANDLER_REGISTRY.items():
        if target_model_type == model_type:
            return handler
    return None


@register_handler
class SKLModelHandler(ModelHandler):
    type = "sklearn"

    @staticmethod
    def can_handle(model: ModelType) -> bool:
        return (
            type_util.LazyType("sklearn.base.BaseEstimator").isinstance(model)
            or type_util.LazyType("sklearn.pipeline.Pipeline").isinstance(model)
        ) and hasattr(model, "predict")

    @staticmethod
    def infer_schema(model: ModelType, X: Any) -> Tuple[type_spec.TypeSpec, type_spec.TypeSpec]:
        input_spec = type_spec.infer_spec(X)
        y = model.predict(X)
        output_spec = type_spec.infer_spec(y)
        return (input_spec, output_spec)

    @staticmethod
    def _save_model(
        name: str, model: ModelType, model_meta: model_meta.ModelMetadata, model_dir_path: str, **kwargs: Any
    ) -> None:
        model_blob_path = os.path.join(model_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, "model.pkl"), "wb") as f:
            cloudpickle.dump(model, f)
        model_meta.models[name] = {
            "name": name,
            "type": SKLModelHandler.type,
            "path": model_blob_path,
        }

    @staticmethod
    def _load_model(name: str, model_meta: model_meta.ModelMetadata, model_dir_path: str) -> ModelType:
        model_blob_dir_path = os.path.join(model_dir_path, name)
        with open(os.path.join(model_blob_dir_path, "model.pkl"), "rb") as f:
            m = cloudpickle.load(f)
        return m


@register_handler
class CustomModelHandler(ModelHandler):
    type = "custom"

    @staticmethod
    def can_handle(model: ModelType) -> bool:
        return bool(type_util.LazyType("snowflake.ml.model.custom_model.CustomModel").isinstance(model))

    @staticmethod
    def infer_schema(model: ModelType, X: Any) -> Tuple[type_spec.TypeSpec, type_spec.TypeSpec]:
        return model._input_spec, model._output_spec

    @staticmethod
    def _save_model(
        name: str, model: ModelType, model_meta: model_meta.ModelMetadata, model_dir_path: str, **kwargs: Any
    ) -> None:
        model._validate()
        model_blob_path = os.path.join(model_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        if model.context.artifacts:
            artifacts_path = os.path.join(model_blob_path, "artifacts")
            os.makedirs(artifacts_path, exist_ok=True)
            for _name, uri in model.context.artifacts.items():
                util.copy_file_or_tree(uri, artifacts_path)

        # Save submodels
        if model.context.model_refs:
            for sub_name, model_ref in model.context.model_refs.items():
                handler = find_handler(model_ref.model)
                assert handler is not None
                handler._save_model(sub_name, model_ref.model, model_meta, model_dir_path)

        with open(os.path.join(model_blob_path, "model.pkl"), "wb") as f:
            cloudpickle.dump(model, f)
        # TODO(halu): dataclass instead of pure dict.
        model_meta.models[name] = {
            "name": name,
            "type": CustomModelHandler.type,
            "path": model_blob_path,
            "artifacts": model.context.artifacts.copy(),
        }

    @staticmethod
    def _load_model(name: str, model_meta: model_meta.ModelMetadata, model_dir_path: str) -> ModelType:
        from snowflake.ml.model.custom_model import ModelContext

        model_blob_dir_path = os.path.join(model_dir_path, name)
        with open(os.path.join(model_blob_dir_path, "model.pkl"), "rb") as f:
            m = cloudpickle.load(f)
        artifacts_meta = model_meta.models[name]["artifacts"]
        artifacts = {
            name: os.path.join(model_blob_dir_path, "artifacts", rel_path) for name, rel_path in artifacts_meta.items()
        }
        models = dict()
        for sub_model_name, _ref in m.context.model_refs.items():
            model_type = model_meta.models[sub_model_name]["type"]
            sub_model = MODEL_HANDLER_REGISTRY[model_type]._load_model(
                name=sub_model_name,
                model_meta=model_meta,
                model_dir_path=model_dir_path,
            )
            models[sub_model_name] = sub_model
        ctx = ModelContext(artifacts=artifacts, models=models)
        m.context = ctx
        return m
