import importlib
import os
import pkgutil
from types import ModuleType
from typing import Dict, Optional, Type

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._handlers import _base

_HANDLERS_BASE = "_handlers"
_MODEL_HANDLER_REGISTRY: Dict[str, Type[_base._ModelHandler[model_types.SupportedModelType]]] = dict()


def _register_handlers() -> None:
    """
    Scan all Python modules in _HANDLERS_BASE directory and register every found non-base ModelHandler
    automatically.
    """
    model_module = importlib.import_module("snowflake.ml.model")
    model_path = model_module.__path__

    for _, name, _ in pkgutil.iter_modules(
        map(lambda x: os.path.join(x, _HANDLERS_BASE), model_path), "snowflake.ml.model._handlers."
    ):
        handler_module = importlib.import_module(name)
        if type(handler_module) == ModuleType:
            for c in dir(handler_module):
                k_class = getattr(handler_module, c)
                if (
                    isinstance(k_class, type)
                    and k_class is not _base._ModelHandler
                    and issubclass(k_class, _base._ModelHandler)
                ):
                    _MODEL_HANDLER_REGISTRY[k_class.handler_type] = k_class


def _find_handler(
    model: model_types.SupportedModelType,
) -> Optional[Type[_base._ModelHandler[model_types.SupportedModelType]]]:
    retried = False
    while True:
        for handler in _MODEL_HANDLER_REGISTRY.values():
            if handler.can_handle(model):
                return handler
        if retried:
            return None
        else:
            _register_handlers()
            retried = True


def _load_handler(target_model_type: str) -> Optional[Type[_base._ModelHandler[model_types.SupportedModelType]]]:
    retried = False
    while True:
        for model_type, handler in _MODEL_HANDLER_REGISTRY.items():
            if target_model_type == model_type:
                return handler
        if retried:
            return None
        else:
            _register_handlers()
            retried = True
