import importlib
import os
import sys
from typing import Any


def run_reflected_func(module_path: str, func_name: str, *args: Any, **kwargs: Any) -> Any:
    project_root = os.path.dirname(__file__)
    if project_root not in sys.path:
        sys.path.append(project_root)
    try:
        mod = importlib.import_module(module_path)
        run_func = getattr(mod, func_name)
        return run_func(*args, **kwargs)
    except Exception as e:
        raise ModuleNotFoundError(f"[ERROR] Failed to load '{func_name}' from {module_path}: {e}")
