import functools
import importlib
import pkgutil
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar, cast

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._packager.model_handlers import _base

_HANDLERS_BASE = "snowflake.ml.model._packager.model_handlers"
_MODEL_HANDLER_REGISTRY: dict[str, type[_base.BaseModelHandler[model_types.SupportedModelType]]] = dict()
_IS_HANDLER_LOADED = False


def _register_handlers() -> None:
    """
    Scan all Python modules in _HANDLERS_BASE directory and register every found non-base ModelHandler
    automatically.
    """
    model_module = importlib.import_module(_HANDLERS_BASE)
    model_path = model_module.__path__

    for _, name, _ in pkgutil.iter_modules(model_path, f"{_HANDLERS_BASE}."):
        if name.startswith("_"):
            continue
        handler_module = importlib.import_module(name)
        if isinstance(handler_module, ModuleType):
            for c in dir(handler_module):
                k_class = getattr(handler_module, c)
                if (
                    isinstance(k_class, type)
                    and k_class is not _base.BaseModelHandler
                    and issubclass(k_class, _base.BaseModelHandler)
                ):
                    _MODEL_HANDLER_REGISTRY[k_class.HANDLER_TYPE] = k_class


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
def find_handler(
    model: model_types.SupportedModelType,
) -> Optional[type[_base.BaseModelHandler[model_types.SupportedModelType]]]:
    for handler in _MODEL_HANDLER_REGISTRY.values():
        if handler.can_handle(model):
            return handler
    return None


@ensure_handlers_registration
def load_handler(
    target_model_type: model_types.SupportedModelHandlerType,
) -> Optional[type[_base.BaseModelHandler[model_types.SupportedModelType]]]:
    for model_type, handler in _MODEL_HANDLER_REGISTRY.items():
        if target_model_type == model_type:
            return handler
    return None


@ensure_handlers_registration
def is_auto_signature_model(model: model_types.SupportedModelType) -> bool:
    for handler in _MODEL_HANDLER_REGISTRY.values():
        if handler.can_handle(model):
            return handler.IS_AUTO_SIGNATURE
    return False
