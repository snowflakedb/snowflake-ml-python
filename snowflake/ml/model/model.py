import os
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, overload

from snowflake.ml.model import (
    custom_model,
    model_handler,
    model_meta,
    model_signature,
    model_types,
)

MODEL_BLOBS_DIR = "models"


@overload
def save_model(
    *,
    name: str,
    model_dir_path: str,
    model: Any,
    signature: model_signature.ModelSignature,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    **kwargs: Any,
) -> model_meta.ModelMetadata:
    """Save the model under `dir_path`.

    Args:
        name: Name of the model.
        model_dir_path: Directory to save the model.
        model: Model object.
        signature: Model data signature for inputs and output.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        **kwargs: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model_dir_path: str,
    model: Any,
    sample_input: Any,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    **kwargs: Any,
) -> model_meta.ModelMetadata:
    """Save the model under `dir_path`.

    Args:
        name: Name of the model.
        model_dir_path: Directory to save the model.
        model: Model object.
        sample_input: Sample input data to infer the model signature from.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        **kwargs: Model specific kwargs.
    """
    ...


def save_model(
    *,
    name: str,
    model_dir_path: str,
    model: Any,
    signature: Optional[model_signature.ModelSignature] = None,
    sample_input: Optional[Any] = None,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    **kwargs: Any,
) -> model_meta.ModelMetadata:
    """Save the model under `dir_path`.

    Args:
        name: Name of the model.
        model_dir_path: Directory to save the model.
        model: Model object.
        signature: Model data signature for inputs and output. If it is None, sample_input would be used to infer the
            signature. If not None, sample_input should not be specified. Defaults to None.
        sample_input: Sample input data to infer the model signature from. If it is None, signature must be specified.
            If not None, signature should not be specified. Defaults to None.
        metadata: Model metadata.
        conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to specify
            a dependency. It is a recommended way to specify your dependencies using conda. When channel is not
            specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel would be
            replaced with the Snowflake Anaconda channel.
        pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is pip
            requirements.
        python_version: A string of python version where model is run. Used for user override. If specified as None,
            current version would be captured. Defaults to None.
        code_paths: Directory of code to import.
        ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
        **kwargs: Model specific kwargs.

    Returns:
        Model metadata.

    Raises:
        ValueError: Raised when the signature and sample_input specified at the same time.
        TypeError: Raised if model type is not supported.
    """

    if not ((signature is None) ^ (sample_input is None)):
        raise ValueError(
            "Signature and sample_input both cannot be "
            + f"{'None' if signature is None else 'specified'} at the same time."
        )

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
        signature=signature,
        ext_modules=ext_modules,
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        python_version=python_version,
    ) as meta:
        model_blobs_path = os.path.join(model_dir_path, MODEL_BLOBS_DIR)
        os.makedirs(model_blobs_path, exist_ok=True)
        handler._save_model(
            name,
            model,
            meta,
            model_blobs_path,
            sample_input=sample_input,
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
