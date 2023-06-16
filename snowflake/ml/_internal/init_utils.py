import importlib
import inspect
import pkgutil
from types import FunctionType
from typing import Dict


def fetch_classes_from_modules_in_pkg_dir(pkg_dir: str, pkg_name: str) -> Dict[str, type]:
    """Finds classes defined all the python modules in the given package directory.

    Args:
        pkg_dir: Path of the package directory.
        pkg_name: Package name. Example, "snowflake.ml.modeling.preprocessing".

    Returns:
        A dict with class_name as key and class object as value.
    """
    # iterate through the modules in the current package
    exportable_classes = {}
    for module_info in pkgutil.iter_modules([pkg_dir]):
        if module_info.ispkg:
            continue

        # import the module and iterate through its attributes
        try:
            module = importlib.import_module(f"{pkg_name}.{module_info.name}")
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                # if attr is a class and is not an imported class
                if inspect.isclass(attribute) and attribute.__module__.startswith(pkg_name):
                    exportable_classes[attribute_name] = attribute
        except ModuleNotFoundError:
            # Ignore not found modules. import statements will fail
            # and provide better error messages.
            pass
    return exportable_classes


def fetch_functions_from_modules_in_pkg_dir(pkg_dir: str, pkg_name: str) -> Dict[str, FunctionType]:
    """Finds functions defined all the python modules in the given package directory.

    Args:
        pkg_dir: Path of the package directory.
        pkg_name: Package name. Example, "snowflake.ml.modeling.preprocessing".

    Returns:
        A dict with function_name as key and function as value.
    """
    # iterate through the modules in the current package
    exportable_functions = {}
    for module_info in pkgutil.iter_modules([pkg_dir]):
        if module_info.ispkg:
            continue

        # import the module and iterate through its attributes
        try:
            module = importlib.import_module(f"{pkg_name}.{module_info.name}")
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                # if attr is a function and is not an imported function
                if inspect.isfunction(attribute) and attribute.__module__.startswith(pkg_name):
                    exportable_functions[attribute_name] = attribute
        except ModuleNotFoundError:
            # Ignore not found modules. import statements will fail
            # and provide better error messages.
            pass
    return exportable_functions
