import importlib
import inspect
from typing import Dict, List, Optional, Tuple

from snowflake.ml.snowpark_pandas.modules import MODULE_MAPPING, MODULES

TYPE_CLASS = "CLASS"
TYPE_FUNCTION = "FUNCTION"


def import_modules() -> Dict[str, Dict[str, List[Tuple[str, Optional[str]]]]]:
    # { module_name -> { type -> [(class_name/function_name, native_module_name)] } }
    modules: Dict[str, Dict[str, List[Tuple[str, Optional[str]]]]] = {}
    for module_info in MODULES:
        try:
            module = importlib.import_module(module_info.module_name)
            class_names = []
            function_names = []
            for estimator in inspect.getmembers(module):
                if (
                    inspect.isclass(estimator[1])
                    # Not an imported class
                    and estimator[1].__module__.startswith(module_info.module_name)
                ):
                    class_names.append(estimator[0])

                if module_info.has_functions:
                    if (
                        inspect.isfunction(estimator[1])
                        # Not an imported class
                        and estimator[1].__module__.startswith(module_info.module_name)
                    ):
                        function_names.append(estimator[0])

            modules[module_info.module_name] = {}
            modules[module_info.module_name][TYPE_CLASS] = [
                (v, MODULE_MAPPING[module_info.module_name]) if v in module_info.native_list else (v, None)
                for v in class_names
                if (
                    not v.startswith("_")
                    and (
                        v not in module_info.exclude_list
                        if len(module_info.exclude_list) > 0
                        else v in module_info.include_list
                        if len(module_info.include_list) > 0
                        else True
                    )
                )
            ]

            modules[module_info.module_name][TYPE_FUNCTION] = [
                (v, MODULE_MAPPING[module_info.module_name]) if v in module_info.native_list else (v, None)
                for v in function_names
                if (
                    not v.startswith("_")
                    and (
                        v not in module_info.exclude_list
                        if len(module_info.exclude_list) > 0
                        else v in module_info.include_list
                        if len(module_info.include_list) > 0
                        else True
                    )
                )
            ]
        except ImportError:
            pass
    return modules


def import_native_modules() -> Dict[str, Dict[str, List[str]]]:
    # { module_name -> { type -> [class_name/function_name] } }
    modules: Dict[str, Dict[str, List[str]]] = {}
    for module_info in MODULES:
        if module_info.module_name in MODULE_MAPPING.keys():
            native_module_name = MODULE_MAPPING[module_info.module_name]
            try:
                module = importlib.import_module(native_module_name)
                class_names = []
                function_names = []
                for estimator in inspect.getmembers(module):
                    if (
                        inspect.isclass(estimator[1])
                        # Not an imported class
                        and estimator[1].__module__.startswith(native_module_name)
                    ):
                        class_names.append(estimator[0])
                    if module_info.has_functions:
                        if (
                            inspect.isfunction(estimator[1])
                            # Not an imported class
                            and estimator[1].__module__.startswith(native_module_name)
                        ):
                            function_names.append(estimator[0])
                modules[native_module_name] = {}
                modules[native_module_name][TYPE_CLASS] = [
                    v
                    for v in class_names
                    if (
                        not v.startswith("_")
                        and (
                            v not in module_info.exclude_list
                            if len(module_info.exclude_list) > 0
                            else v in module_info.native_list
                            if len(module_info.native_list) > 0
                            else True
                        )
                    )
                ]
                modules[native_module_name][TYPE_FUNCTION] = [
                    v
                    for v in function_names
                    if (
                        not v.startswith("_")
                        and (
                            v not in module_info.exclude_list
                            if len(module_info.exclude_list) > 0
                            else v in module_info.native_list
                            if len(module_info.native_list) > 0
                            else True
                        )
                    )
                ]
            except ImportError:
                pass
    return modules
