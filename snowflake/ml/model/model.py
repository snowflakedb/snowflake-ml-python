import os
from types import ModuleType
from typing import Any, Dict, List, Optional

from snowflake.ml.model import env, model_handler, model_meta

ModelType = Any


def save_model(
    *,
    name: str,
    model_dir_path: str,
    model: Any,
    metadata: Optional[Dict[str, Any]] = None,
    pip_requirements: Optional[List[str]] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    sample_data: Optional[Any] = None,
) -> None:
    """Save the model under `dir_path`.

    Args:
        name (str): Name of the model.
        model_dir_path (str): Directory to save the model.
        model (Any): Model object.
        metadata (Dict[str, Any]): _description_
        pip_requirements: List of PIP package specs.
        code_paths: Directory of code to import.
        ext_modules (_type_, optional): _description_. Defaults to None.
        sample_data: Sample input data for schema inference.

    Raises:
        TypeError: Raise if model type is not supported.
    """
    handler = model_handler.find_handler(model)
    if handler is None:
        raise TypeError(f"{type(model)} is not supported.")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    input_spec, output_spec = handler.infer_schema(model, sample_data)
    with model_meta.create(
        model_dir_path=model_dir_path,
        name=name,
        python_version=env.PYTHON_VERSION,
        model_type=handler.type,
        metadata=metadata,
        code_paths=code_paths,
        input_spec=input_spec,
        output_spec=output_spec,
        ext_modules=ext_modules,
        pip_requirements=pip_requirements,
    ) as meta:
        model_blob_path = os.path.join(model_dir_path, "models")
        os.makedirs(model_blob_path, exist_ok=True)
        handler._save_model(
            name, model, meta, model_blob_path, ext_modules=ext_modules, code_paths=code_paths, sample_data=sample_data
        )


# TODO: Allows path to be stage path.
def load_model(model_dir_path: str) -> ModelType:
    """Load the model into memory from directory.

    Args:
        model_dir_path (str): Directory containing the model.

    Raises:
        TypeError: Raise if model is not native format.

    Returns:
        ModelType: _description_
    """
    meta = model_meta.load_model_metadata(model_dir_path)
    handler = model_handler.load_handler(meta.model_type)
    if handler is None:
        raise TypeError(f"{meta.model_type} is not supported.")
    model_blob_path = os.path.join(model_dir_path, "models")
    m = handler._load_model(meta.name, meta, model_blob_path)
    return m, meta
