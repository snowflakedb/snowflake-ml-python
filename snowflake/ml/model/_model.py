import os
import posixpath
import tempfile
import warnings
from types import ModuleType
from typing import Dict, List, Literal, Optional, Tuple, Union, overload

from snowflake.ml._internal import file_utils, type_utils
from snowflake.ml.model import (
    _env,
    _model_handler,
    _model_meta,
    custom_model,
    model_signature,
    type_hints as model_types,
)
from snowflake.snowpark import FileOperation, Session

MODEL_BLOBS_DIR = "models"


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedNoSignatureRequirementsModelType,
    model_dir_path: str,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save a model that does not require a signature under `dir_path`.

    Args:
        name: Name of the model.
        model: Model object.
        model_dir_path: Directory to save the model.
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
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedRequireSignatureModelType,
    model_dir_path: str,
    signatures: Dict[str, model_signature.ModelSignature],
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save a model that requires a external signature with user provided signatures under `dir_path`.

    Args:
        name: Name of the model.
        model: Model object.
        model_dir_path: Directory to save the model.
        signatures: Model data signatures for inputs and output for every target methods.
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
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedRequireSignatureModelType,
    model_dir_path: str,
    sample_input: model_types.SupportedDataType,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save a model that requires a external signature under `dir_path` with signature
        inferred from a sample_input_data.

    Args:
        name: Name of the model.
        model: Model object.
        model_dir_path: Directory to save the model.
        sample_input: Sample input data to infer the model signatures from.
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
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedNoSignatureRequirementsModelType,
    session: Session,
    model_stage_file_path: str,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save a model that does not require a signature to a zip file whose path is the provided stage file path.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        model_stage_file_path: Path to the file in Snowflake stage where the function should put the saved model.
            Must be a file with .zip extension.
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
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedRequireSignatureModelType,
    session: Session,
    model_stage_file_path: str,
    signatures: Dict[str, model_signature.ModelSignature],
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save a model that requires a external signature with user provided signatures
         to a zip file whose path is the provided stage file path.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        model_stage_file_path: Path to the file in Snowflake stage where the function should put the saved model.
            Must be a file with .zip extension.
        signatures: Model data signatures for inputs and output for every target methods.
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
        options: Model specific kwargs.
    """
    ...


@overload
def save_model(
    *,
    name: str,
    model: model_types.SupportedRequireSignatureModelType,
    session: Session,
    model_stage_file_path: str,
    sample_input: model_types.SupportedDataType,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save a model that requires a external signature to a zip file whose path is the
    provided stage file path with signature inferred from a sample_input_data.

    Args:
        name: Name of the model.
        model: Model object.
        session: Snowpark connection session.
        model_stage_file_path: Path to the file in Snowflake stage where the function should put the saved model.
            Must be a file with .zip extension.
        sample_input: Sample input data to infer the model signatures from.
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
        options: Model specific kwargs.
    """
    ...


def save_model(
    *,
    name: str,
    model: model_types.SupportedModelType,
    session: Optional[Session] = None,
    model_stage_file_path: Optional[str] = None,
    model_dir_path: Optional[str] = None,
    signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
    sample_input: Optional[model_types.SupportedDataType] = None,
    metadata: Optional[Dict[str, str]] = None,
    conda_dependencies: Optional[List[str]] = None,
    pip_requirements: Optional[List[str]] = None,
    python_version: Optional[str] = None,
    ext_modules: Optional[List[ModuleType]] = None,
    code_paths: Optional[List[str]] = None,
    options: Optional[model_types.ModelSaveOption] = None,
) -> _model_meta.ModelMetadata:
    """Save the model.

    Args:
        name: Name of the model.
        model: Model object.
        model_dir_path: Directory to save the model. Exclusive with `session` and `model_stage_file_path`.
        session: Snowpark connection session. Needs to present with `model_stage_file_path`.
            Exclusive with `model_dir_path`.
        model_stage_file_path: Path to the file in Snowflake stage where the function should put the saved model.
            Needs to present with `session`. Exclusive with `model_dir_path`. Must be a file with .zip extension.
        signatures: Model data signatures for inputs and output for every target methods. If it is None, sample_input
            would be used to infer the signatures if it is a local (non-SnowML modeling model).
            If not None, sample_input should not be specified. Defaults to None.
        sample_input: Sample input data to infer the model signatures from. If it is None, signatures must be specified
            if it is a local (non-SnowML modeling model). If not None, signatures should not be specified.
            Defaults to None.
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
        options: Model specific kwargs.

    Returns:
        Model metadata.

    Raises:
        ValueError: Raised when the session and model_stage_file_path not specified or not be None at the same time.
        ValueError: Raised when the model_stage_file_path and model_dir_path specified at the same time.
        ValueError: Raised when the signatures and sample_input specified at the same time, or not presented when
            specifying local model.
        ValueError: Raised when provided model directory is not a directory.
        ValueError: Raised when provided model stage path is not a zip file.
    """
    if (session is None) ^ (model_stage_file_path is None):
        raise ValueError(
            "Session and model_stage_file_path must be "
            + f"{'None' if session is None else 'specified'} at the same time."
        )

    if not ((model_stage_file_path is None) ^ (model_dir_path is None)):
        raise ValueError(
            "model_dir_path and model_stage_file_path both cannot be "
            + f"{'None' if model_stage_file_path is None else 'specified'} at the same time."
        )

    if (
        (signatures is None)
        and (sample_input is None)
        and not (
            type_utils.LazyType("snowflake.ml.modeling.framework.base.BaseEstimator").isinstance(model)
            or type_utils.LazyType("mlflow.pyfunc.PyFuncModel").isinstance(model)
        )
    ) or ((signatures is not None) and (sample_input is not None)):
        raise ValueError(
            "Signatures and sample_input both cannot be "
            + f"{'None for local model' if signatures is None else 'specified'} at the same time."
        )

    if not options:
        options = model_types.BaseModelSaveOption()

    if model_dir_path:
        if os.path.exists(model_dir_path):
            if not os.path.isdir(model_dir_path):
                raise ValueError(f"Provided model directory {model_dir_path} is not a directory.")
            if os.listdir(model_dir_path):
                warnings.warn(
                    f"Provided model directory {model_dir_path} is not an empty directory. Files might be overwritten.",
                    category=UserWarning,
                )
        else:
            os.makedirs(model_dir_path)
        return _save(
            name=name,
            model=model,
            local_dir_path=model_dir_path,
            signatures=signatures,
            sample_input=sample_input,
            metadata=metadata,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            python_version=python_version,
            ext_modules=ext_modules,
            code_paths=code_paths,
            options=options,
        )

    assert session and model_stage_file_path
    if posixpath.splitext(model_stage_file_path)[1] != ".zip":
        raise ValueError(f"Provided model path in the stage {model_stage_file_path} must be a path to a zip file.")

    with tempfile.TemporaryDirectory() as temp_local_model_dir_path:
        meta = _save(
            name=name,
            model=model,
            local_dir_path=temp_local_model_dir_path,
            signatures=signatures,
            sample_input=sample_input,
            metadata=metadata,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            python_version=python_version,
            ext_modules=ext_modules,
            code_paths=code_paths,
            options=options,
        )
        with file_utils.zip_file_or_directory_to_stream(
            temp_local_model_dir_path, leading_path=temp_local_model_dir_path
        ) as zf:
            assert session and model_stage_file_path
            fo = FileOperation(session=session)
            fo.put_stream(
                zf,
                model_stage_file_path,
                auto_compress=False,
                overwrite=options.get("allow_overwritten_stage_file", False),
            )
        return meta


def _save(
    *,
    name: str,
    model: model_types.SupportedModelType,
    local_dir_path: str,
    signatures: Optional[Dict[str, model_signature.ModelSignature]],
    sample_input: Optional[model_types.SupportedDataType],
    metadata: Optional[Dict[str, str]],
    conda_dependencies: Optional[List[str]],
    pip_requirements: Optional[List[str]],
    python_version: Optional[str],
    ext_modules: Optional[List[ModuleType]],
    code_paths: Optional[List[str]],
    options: model_types.ModelSaveOption,
) -> _model_meta.ModelMetadata:
    local_dir_path = os.path.normpath(local_dir_path)

    handler = _model_handler._find_handler(model)
    if handler is None:
        raise TypeError(f"{type(model)} is not supported.")
    with _model_meta._create_model_metadata(
        model_dir_path=local_dir_path,
        name=name,
        model_type=handler.handler_type,
        metadata=metadata,
        code_paths=code_paths,
        signatures=signatures,
        ext_modules=ext_modules,
        conda_dependencies=conda_dependencies,
        pip_requirements=pip_requirements,
        python_version=python_version,
        **options,
    ) as meta:
        model_blobs_path = os.path.join(local_dir_path, MODEL_BLOBS_DIR)
        os.makedirs(model_blobs_path, exist_ok=True)
        model = handler.cast_model(model)
        handler._save_model(
            name=name,
            model=model,
            model_meta=meta,
            model_blobs_dir_path=model_blobs_path,
            sample_input=sample_input,
            is_sub_model=False,
            **options,
        )

    return meta


@overload
def load_model(
    *, model_dir_path: str, meta_only: Optional[Literal[False]] = None
) -> Tuple[model_types.SupportedModelType, _model_meta.ModelMetadata]:
    """Load the model into memory from directory.

    Args:
        model_dir_path: Directory containing the model.
        meta_only: Flag to indicate that if only load metadata.
    """
    ...


@overload
def load_model(*, model_dir_path: str, meta_only: Literal[True]) -> _model_meta.ModelMetadata:
    """Load the model into memory from directory with metadata only.

    Args:
        model_dir_path: Directory containing the model.
        meta_only: Flag to indicate that if only load metadata.
    """
    ...


@overload
def load_model(
    *, session: Session, model_stage_file_path: str, meta_only: Optional[Literal[False]] = None
) -> Tuple[model_types.SupportedModelType, _model_meta.ModelMetadata]:
    """Load the model into memory from a zip file in the stage.

    Args:
        session: Snowflake connection session.
        model_stage_file_path: The path to zipped model file in the stage. Must be a file with .zip extension.
        meta_only: Flag to indicate that if only load metadata.
    """
    ...


@overload
def load_model(*, session: Session, model_stage_file_path: str, meta_only: Literal[True]) -> _model_meta.ModelMetadata:
    """Load the model into memory from a zip file in the stage with metadata only.

    Args:
        session: Snowflake connection session.
        model_stage_file_path: The path to zipped model file in the stage. Must be a file with .zip extension.
        meta_only: Flag to indicate that if only load metadata.
    """
    ...


def load_model(
    *,
    session: Optional[Session] = None,
    model_stage_file_path: Optional[str] = None,
    model_dir_path: Optional[str] = None,
    meta_only: Optional[bool] = None,
) -> Union[_model_meta.ModelMetadata, Tuple[model_types.SupportedModelType, _model_meta.ModelMetadata]]:
    """Load the model into memory from directory or a zip file in the stage.

    Args:
        session: Snowflake connection session. Must be specified when specifying model_stage_file_path.
            Exclusive with model_dir_path.
        model_stage_file_path: The path to zipped model file in the stage. Must be specified when specifying session.
            Exclusive with model_dir_path. Must be a file with .zip extension.
        model_dir_path: Directory containing the model. Exclusive with session and model_stage_file_path.
        meta_only: Flag to indicate that if only load metadata.

    Raises:
        ValueError: Raised when the session and model_stage_file_path not specified or not be None at the same time.
        ValueError: Raised when the model_stage_file_path and model_dir_path specified at the same time.
        ValueError: Raised if model directory does not exist.
        ValueError: Raised if model directory is not a directory.
        ValueError: Raised if model provided in the stage is not a zip file.

    Returns:
        A tuple containing the model object and the model metadata.
    """
    if (session is None) ^ (model_stage_file_path is None):
        raise ValueError(
            "Session and model_stage_file_path must be "
            + f"{'None' if session is None else 'specified'} at the same time."
        )

    if not ((model_stage_file_path is None) ^ (model_dir_path is None)):
        raise ValueError(
            "model_dir_path and model_stage_file_path both cannot be "
            + f"{'None' if model_stage_file_path is None else 'specified'} at the same time."
        )

    if model_dir_path:
        if not os.path.exists(model_dir_path):
            raise ValueError(f"Provided model directory {model_dir_path} does not exist.")
        if not os.path.isdir(model_dir_path):
            raise ValueError(f"Provided model directory {model_dir_path} is not a directory.")

        return _load(local_dir_path=model_dir_path, meta_only=meta_only)

    assert session and model_stage_file_path
    if posixpath.splitext(model_stage_file_path)[1] != ".zip":
        raise ValueError(f"Provided model path in the stage {model_stage_file_path} must be a path to a zip file.")

    fo = FileOperation(session=session)
    zf = fo.get_stream(model_stage_file_path)
    with file_utils.unzip_stream_in_temp_dir(stream=zf) as temp_local_model_dir_path:
        return _load(local_dir_path=temp_local_model_dir_path, meta_only=meta_only)


def _load(
    *,
    local_dir_path: str,
    meta_only: Optional[bool] = None,
) -> Union[_model_meta.ModelMetadata, Tuple[model_types.SupportedModelType, _model_meta.ModelMetadata]]:
    local_dir_path = os.path.normpath(local_dir_path)
    meta = _model_meta._load_model_metadata(local_dir_path)
    if meta_only:
        return meta

    _env.validate_py_runtime_version(meta.python_version)

    handler = _model_handler._load_handler(meta.model_type)
    if handler is None:
        raise TypeError(f"{meta.model_type} is not supported.")
    model_blobs_path = os.path.join(local_dir_path, MODEL_BLOBS_DIR)
    m = handler._load_model(meta.name, meta, model_blobs_path)
    return m, meta


def _load_model_for_deploy(model_dir_path: str) -> Tuple[custom_model.CustomModel, _model_meta.ModelMetadata]:
    """Load the model into memory from directory. Internal used when deploying only.
    It will try to use _load_as_custom_model method in the handler if provided, otherwise, it will use _load_model.

    Args:
        model_dir_path: Directory containing the model.

    Raises:
        TypeError: Raised if model is not native format.

    Returns:
        A tuple containing the model object as a custom model and the model metadata.
    """
    model_dir_path = os.path.normpath(model_dir_path)

    meta = _model_meta._load_model_metadata(model_dir_path)
    handler = _model_handler._load_handler(meta.model_type)
    if handler is None:
        raise TypeError(f"{meta.model_type} is not supported.")
    model_blobs_path = os.path.join(model_dir_path, MODEL_BLOBS_DIR)
    load_func = getattr(handler, "_load_as_custom_model", None)
    if not callable(load_func):
        load_func = handler._load_model
    m = load_func(meta.name, meta, model_blobs_path)

    assert isinstance(m, custom_model.CustomModel)

    return m, meta
