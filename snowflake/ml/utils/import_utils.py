import importlib
from typing import Any, Tuple


class MissingOptionalDependency:
    """A class to replace missing dependencies.

    The only thing this class is supposed to do is raise a ImportError when __getattr__ is called.
    This will be triggered whenever module.member is going to be called.
    """

    def __init__(self, dep_name: str) -> None:
        self._dep_name = dep_name

    def __getattr__(self, item: str) -> None:
        raise ImportError(f"Unable to import {self._dep_name}.")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(f"Unable to import {self._dep_name}.")


def import_or_get_dummy(target: str) -> Tuple[Any, bool]:
    """Try to import the the given target or return a dummy object.

    If the import target (package/module/symbol) is available, the target will be returned. If it is not available,
    a dummy target will be returned, which will raise an ImportError whenever it is actually used.

    Args:
        target: A string representing the target which needs to be imported. It should be a list of symbol name
            joined by dot. Some valid examples:
                - <some_package>
                - <some_module>
                - <some_package>.<some_module>
                - <some_module>.<some_symbol>

    Returns:
        A 2-tuple consists of the imported target and a boolean of whether the target is available.
    """
    # First try to import the target as a module
    try:
        res = importlib.import_module(target)
        return (res, True)
    except ImportError:
        pass

    # Try to import the target as a symbol
    try:
        res = _try_import_symbol(target)
        return (res, True)
    except ImportError:
        return (MissingOptionalDependency(target), False)


def _try_import_symbol(target: str) -> Any:
    module_name, class_name = target.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (AttributeError, ImportError):
        raise ImportError
