from typing import Any, Union


def flatten_nested_params(params: Union[list[Any], dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    flat_params = {}
    items = params.items() if isinstance(params, dict) else enumerate(params)
    for key, value in items:
        key = str(key).replace(".", "_")  # Replace dots in keys to avoid collisions involving nested keys
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (dict, list)):
            flat_params.update(flatten_nested_params(value, new_prefix))
        else:
            flat_params[new_prefix] = value
    return flat_params
