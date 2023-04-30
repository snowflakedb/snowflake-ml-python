import os
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

from snowflake.ml.model import (
    custom_model,
    model_handler,
    model_meta,
    model_types,
    schema,
)

MODEL_BLOBS_DIR = "models"


def save_model(
    *,
    name: str,
    model_dir_path: str,
    model: Any,
    schema: schema.Schema,
    metadata: Optional[Dict[str, str]] = None,
    pip_requirements: Optional[List[str]] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    **kwargs: Any,
) -> model_meta.ModelMetadata:
    """Save the model under `dir_path`.

    Args:
        name: Name of the model.
        model_dir_path: Directory to save the model.
        model: Model object.
        schema: Model data schema for inputs and output.
        metadata: Model metadata.
        pip_requirements: List of PIP package specs.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        **kwargs: Model specific kwargs.

    Returns:
        Model metadata.

    Raises:
        TypeError: Raised if model type is not supported.
    """
    handler = model_handler._find_handler(model)
    if handler is None:
        raise TypeError(f"{type(model)} is not supported.")
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    with model_meta._create_model_metadata(
        model_dir_path=model_dir_path,
        name=name,
        model_type=handler.handler_type,
        metadata=metadata,
        code_paths=code_paths,
        schema=schema,
        ext_modules=ext_modules,
        pip_requirements=pip_requirements,
    ) as meta:
        model_blobs_path = os.path.join(model_dir_path, MODEL_BLOBS_DIR)
        os.makedirs(model_blobs_path, exist_ok=True)
        handler._save_model(
            name,
            model,
            meta,
            model_blobs_path,
            ext_modules=ext_modules,
            code_paths=code_paths,
            **kwargs,
        )
    return meta


# TODO(SNOW-786570): Allows path to be stage path.
def load_model(model_dir_path: str) -> Tuple[model_types.ModelType, model_meta.ModelMetadata]:
    """Load the model into memory from directory.

    Args:
        model_dir_path: Directory containing the model.

    Raises:
        TypeError: Raised if model is not native format.

    Returns:
        A tuple containing the model object and the model metadata.
    """
    meta = model_meta._load_model_metadata(model_dir_path)
    handler = model_handler._load_handler(meta.model_type)
    if handler is None:
        raise TypeError(f"{meta.model_type} is not supported.")
    model_blobs_path = os.path.join(model_dir_path, MODEL_BLOBS_DIR)
    m = handler._load_model(meta.name, meta, model_blobs_path)
    return m, meta


def _load_model_for_deploy(model_dir_path: str) -> Tuple[custom_model.CustomModel, model_meta.ModelMetadata]:
    """Load the model into memory from directory. Internal used when deploying only.
    It will try to use _load_as_custom_model method in the handler if provided, otherwise, it will use _load_model.

    Args:
        model_dir_path: Directory containing the model.

    Raises:
        TypeError: Raised if model is not native format.

    Returns:
        A tuple containing the model object as a custom model and the model metadata.
    """
    meta = model_meta._load_model_metadata(model_dir_path)
    handler = model_handler._load_handler(meta.model_type)
    if handler is None:
        raise TypeError(f"{meta.model_type} is not supported.")
    model_blobs_path = os.path.join(model_dir_path, MODEL_BLOBS_DIR)
    load_func = getattr(handler, "_load_as_custom_model", None)
    if not callable(load_func):
        load_func = handler._load_model
    m = load_func(meta.name, meta, model_blobs_path)

    assert isinstance(m, custom_model.CustomModel)

    return m, meta
