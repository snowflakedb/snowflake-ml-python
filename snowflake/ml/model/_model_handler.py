import functools
import importlib
import os
import pkgutil
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._handlers import _base

_HANDLERS_BASE = "_handlers"
_MODEL_HANDLER_REGISTRY: Dict[str, Type[_base._ModelHandler[model_types.SupportedModelType]]] = dict()
_IS_HANDLER_LOADED = False


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


F = TypeVar("F", bound=Callable[..., Any])


def ensure_handlers_registration(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _IS_HANDLER_LOADED
        if not _IS_HANDLER_LOADED:
            _register_handlers()
            _IS_HANDLER_LOADED = True

        return fn(*args, **kwargs)

    return cast(F, wrapper)


@ensure_handlers_registration
def _find_handler(
    model: model_types.SupportedModelType,
) -> Optional[Type[_base._ModelHandler[model_types.SupportedModelType]]]:
    for handler in _MODEL_HANDLER_REGISTRY.values():
        if handler.can_handle(model):
            return handler
    return None


@ensure_handlers_registration
def _load_handler(target_model_type: str) -> Optional[Type[_base._ModelHandler[model_types.SupportedModelType]]]:
    for model_type, handler in _MODEL_HANDLER_REGISTRY.items():
        if target_model_type == model_type:
            return handler
    return None


@ensure_handlers_registration
def is_auto_signature_model(model: model_types.SupportedModelType) -> bool:
    for handler in _MODEL_HANDLER_REGISTRY.values():
        if handler.can_handle(model):
            return handler.is_auto_signature
    return False
