import contextlib
import logging
import os
import pathlib
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Callable, cast, final

import numpy as np
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import file_utils, type_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import utils as model_signature_utils

if TYPE_CHECKING:
    import mlflow

logger = logging.getLogger(__name__)


def _to_native_python(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization.

    MLflow's schema enforcement can inject numpy scalar types (e.g. numpy.int64)
    into lists and dicts. These aren't JSON-serializable and must be converted
    before returning results from a Snowflake UDF.

    Args:
        obj: The object to convert. Can be a numpy scalar, ndarray, list, dict, or any other type.

    Returns:
        The object with all numpy types replaced by their native Python equivalents.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_native_python(x) for x in obj]
    if isinstance(obj, list):
        return [_to_native_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_native_python(v) for k, v in obj.items()}
    return obj


def _ensure_serializable_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure object-dtype columns in a DataFrame contain JSON-serializable values."""
    for col in df.columns:
        if df[col].dtype == np.dtype("O"):
            df[col] = df[col].map(_to_native_python)
    return df


def _get_native_model(model: "mlflow.pyfunc.PyFuncModel") -> Any:
    """Return the flavor's native model, unwrapping MLflow's PyFunc wrapper when one is present.

    Flavor ``log_model`` functions expect the native model (for example the scikit-learn estimator),
    not the wrapper that ``PyFuncModel`` holds. Serializers that only understand flavor-native types,
    such as skops, reject the wrapper outright.

    Args:
        model: The loaded MLflow PyFunc model.

    Returns:
        The native model when the wrapper exposes one, otherwise the wrapper itself.
    """
    try:
        return model.get_raw_model()
    except (AttributeError, NotImplementedError):
        return model._model_impl


def _get_sklearn_serialization_kwargs(model: "mlflow.pyfunc.PyFuncModel") -> dict[str, Any]:
    """Return the scikit-learn serialization settings the model was originally saved with.

    A later save that omits them uses MLflow's current default backend. That default has changed
    across versions, and a switched backend needs a serializer the model's environment does not
    declare.

    Args:
        model: The loaded MLflow PyFunc model.

    Returns:
        Keyword arguments to forward to ``mlflow.sklearn.save_model``.
    """
    import mlflow

    flavor_config = model.metadata.flavors.get(mlflow.sklearn.FLAVOR_NAME, {})
    return {
        key: flavor_config[key]
        for key in ("serialization_format", "skops_trusted_types")
        if flavor_config.get(key) is not None
    }


def _resave_mlflow_model(
    model: "mlflow.pyfunc.PyFuncModel",
    dest_root: str,
) -> tuple["mlflow.pyfunc.PyFuncModel", str]:
    """Write a model that has no tracking run into a local directory.

    ``mlflow.*.save_model()`` models have no run id, so there is no artifact URI to download. The
    flavor's ``save_model`` writes a directory this handler can copy, without opening a tracking
    store. Scikit-learn models are saved from the native estimator, with the serialization settings
    the model was originally saved with.

    Args:
        model: An mlflow.pyfunc.PyFuncModel whose metadata has no tracking run.
        dest_root: Directory that will contain the re-saved model. The caller keeps it alive until
            the artifacts have been copied.

    Returns:
        The reloaded model and the directory it was saved to (``dest_root/model``).

    Raises:
        ValueError: If the model's flavor loader module cannot be determined or imported.
    """
    import importlib

    import mlflow

    pyfunc_flavor = model.metadata.flavors.get(mlflow.pyfunc.FLAVOR_NAME)
    if pyfunc_flavor is None:
        raise ValueError("Cannot re-save an MLflow model without a PyFunc flavor.")

    loader_module_name = pyfunc_flavor.get("loader_module")
    if loader_module_name is None:
        raise ValueError(
            "Cannot determine the MLflow flavor loader module from the model metadata. "
            "The model's pyfunc flavor is missing the 'loader_module' field."
        )

    loader_module = importlib.import_module(loader_module_name)
    save_path = os.path.join(dest_root, "model")
    logger.warning(
        "Model was created with save_model() and lacks tracking metadata. Auto re-saving the model",
    )
    if loader_module_name == "mlflow.pyfunc.model":
        python_model = getattr(model._model_impl, "python_model", model._model_impl)
        if python_model is None:
            raise ValueError("The PyFunc model has no python_model attribute.")
        mlflow.pyfunc.save_model(
            path=save_path,
            python_model=python_model,
            signature=model.metadata.signature,
        )
    elif hasattr(loader_module, "save_model"):
        save_model_kwargs = _get_sklearn_serialization_kwargs(model) if loader_module_name == "mlflow.sklearn" else {}
        loader_module.save_model(
            _get_native_model(model),
            path=save_path,
            signature=model.metadata.signature,
            **save_model_kwargs,
        )
    else:
        raise ValueError(f"The MLflow flavor module '{loader_module_name}' does not have a save_model function.")

    return mlflow.pyfunc.load_model(save_path), save_path


def _parse_mlflow_env(model_uri: str, env: model_env.ModelEnv) -> model_env.ModelEnv:
    """Parse MLFlow env file and modify model env in model meta based on MLFlow env.

    Args:
        model_uri: Model uri where the env file could be downloaded
        env: ModelEnv object to be modified

    Raises:
        ValueError: Raised when cannot download MLFlow model dependencies file.

    Returns:
        Modified model env.
    """
    import mlflow

    try:
        conda_env_file_path = mlflow.pyfunc.get_model_dependencies(model_uri, format="conda")
    except (mlflow.MlflowException, OSError):
        raise ValueError("Cannot load MLFlow model dependencies.")

    if not os.path.exists(conda_env_file_path):
        raise ValueError("Cannot load MLFlow model dependencies.")

    env.load_from_conda_file(pathlib.Path(conda_env_file_path))

    return env


@final
class MLFlowHandler(_base.BaseModelHandler["mlflow.pyfunc.PyFuncModel"]):
    """Handler for MLFlow based model.

    Currently mlflow.pyfunc.PyFuncModel based classes are supported.
    """

    HANDLER_TYPE = "mlflow"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model"
    _DEFAULT_TARGET_METHOD = "predict"
    DEFAULT_TARGET_METHODS = [_DEFAULT_TARGET_METHOD]
    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["mlflow.pyfunc.PyFuncModel"]:
        return type_utils.LazyType("mlflow.pyfunc.PyFuncModel").isinstance(model)

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "mlflow.pyfunc.PyFuncModel":
        import mlflow

        assert isinstance(model, mlflow.pyfunc.PyFuncModel)

        return cast(mlflow.pyfunc.PyFuncModel, model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "mlflow.pyfunc.PyFuncModel",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: model_types.SupportedDataType | None = None,
        is_sub_model: bool | None = False,
        **kwargs: Unpack[model_types.MLFlowSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for MLFlow model.")

        import mlflow

        assert isinstance(model, mlflow.pyfunc.PyFuncModel)

        user_model_uri = kwargs.get("model_uri")
        # A logged model exposes a resolvable artifact URI. A save_model() object does not:
        # get_model_info() raises on MLflow 2.x, or reports run_id=None on 3.x. When the caller
        # also passed model_uri, that directory is the artifact source and is copied as-is.
        resolved_model_uri = None
        try:
            model_info = model.metadata.get_model_info()
            if model_info.run_id is not None:
                resolved_model_uri = model_info.model_uri
        except AttributeError:
            pass

        # Signature, flavors, and metadata live on the in-memory model whether or not it has a run.
        mlflow_signature = model.metadata.signature
        flavors = model.metadata.flavors
        mlflow_model_metadata = model.metadata.metadata

        # A re-saved directory has to stay on disk through dependency parsing and the artifact copy.
        with contextlib.ExitStack() as artifact_stack:
            artifact_uri = user_model_uri or resolved_model_uri
            if artifact_uri is None:
                resave_root = artifact_stack.enter_context(tempfile.TemporaryDirectory())
                _, artifact_uri = _resave_mlflow_model(model, resave_root)
            # Re-saving writes a new environment. Prefer the caller's original model directory when
            # they supplied one, so the packaged dependencies stay the ones they saved.
            deps_uri = user_model_uri or artifact_uri

            pyfunc_flavor_info = flavors.get(mlflow.pyfunc.FLAVOR_NAME, None)
            if pyfunc_flavor_info is None:
                raise ValueError("Cannot save MLFlow model that does not have PyFunc flavor.")

            # Port MLFlow signature
            if not is_sub_model:
                if model_meta.signatures:
                    handlers_utils.validate_target_methods(model, list(model_meta.signatures.keys()))
                else:
                    handlers_utils.validate_target_methods(model, cls.DEFAULT_TARGET_METHODS)
                    model_meta.signatures = {
                        cls._DEFAULT_TARGET_METHOD: model_signature.ModelSignature.from_mlflow_sig(mlflow_signature)
                    }

            # Port MLFlow metadata
            if mlflow_model_metadata and not kwargs.get("ignore_mlflow_metadata", False):
                if not model_meta.metadata:
                    model_meta.metadata = {}
                model_meta.metadata.update(mlflow_model_metadata)

            # Port MLFlow dependencies
            if kwargs.get("ignore_mlflow_dependencies", False):
                model_meta.env.include_if_absent(
                    [model_env.ModelDependency(requirement="mlflow", pip_name="mlflow")], check_local_version=True
                )
            else:
                model_meta.env = _parse_mlflow_env(deps_uri, model_meta.env)

            # setuptools removed the bundled `pkg_resources` module in 82.0.0 (81.x only deprecated
            # it). Loading an MLflow model pulls in opentelemetry (via mlflow's tracing subsystem),
            # whose older releases still do `from pkg_resources import ...` at import time, so a
            # resolved environment with setuptools >= 82 fails to load the model. Pin below that
            # boundary to keep it available.
            model_meta.env.include_if_absent(
                [model_env.ModelDependency(requirement="setuptools<82", pip_name="setuptools")],
                check_local_version=False,
            )

            # NOTE: This pip_requirements + artifact_repository_map validation logic mirrors
            # ModelParameterReconciler._validate_pip_requirements_warehouse_compatibility in
            # snowflake/ml/registry/_manager/model_parameter_reconciler.py — keep them in sync.
            # This handles the MLflow-specific case where pip requirements are parsed from the
            # MLflow model's conda.yaml and not explicitly provided by the user.
            if model_meta.env.pip_requirements and not model_meta.env.artifact_repository_map:
                if not model_meta.env.target_platforms:
                    warnings.warn(
                        "MLflow model has pip requirements that require `artifact_repository_map` to run in a "
                        "Snowflake Warehouse. Without `artifact_repository_map`, this model will only be runnable "
                        "in Snowpark Container Services. See "
                        "https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/container. "
                        f"Detected pip requirements: {model_meta.env.pip_requirements}",
                        category=UserWarning,
                        stacklevel=2,
                    )
                elif model_meta.env.targets_warehouse:
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ARGUMENT,
                        original_exception=ValueError(
                            "MLflow model has pip requirements that require `artifact_repository_map` to run in a "
                            "Snowflake Warehouse. Either provide an `artifact_repository_map` or set "
                            '`target_platforms=["SNOWPARK_CONTAINER_SERVICES"]`. '
                            "See https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/container. "
                            f"Detected pip requirements: {model_meta.env.pip_requirements}."
                        ),
                    )

            model_blob_path = os.path.join(model_blobs_dir_path, name)

            os.makedirs(model_blob_path, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    local_path = mlflow.artifacts.download_artifacts(artifact_uri, dst_path=tmpdir)
                except (mlflow.MlflowException, OSError):
                    raise ValueError("Cannot load MLFlow model artifacts.")

                file_utils.copy_file_or_tree(local_path, os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR))

                # Persist the relative dir name that load_model uses to rebuild the on-disk path.
                # Derive it from the downloaded dir, not model_info.artifact_path, which is relative
                # in MLflow 2.x but an absolute path in 3.x.
                artifact_path = os.path.basename(os.path.abspath(local_path))

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.MLFlowModelBlobOptions({"artifact_path": artifact_path}),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.MLFlowLoadOptions],
    ) -> "mlflow.pyfunc.PyFuncModel":
        import mlflow

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_options = cast(model_meta_schema.MLFlowModelBlobOptions, model_blob_metadata.options)
        if "artifact_path" not in model_blob_options:
            raise ValueError("Missing field `artifact_path` in model blob metadata for type `mlflow`")

        model_artifact_path = model_blob_options["artifact_path"]
        model_blob_filename = model_blob_metadata.path

        # Load directly from the on-disk artifacts; no MLflow tracking backend is used.
        return mlflow.pyfunc.load_model(os.path.join(model_blob_path, model_blob_filename, model_artifact_path))

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "mlflow.pyfunc.PyFuncModel",
        model_meta: model_meta_api.ModelMetadata,
        background_data: pd.DataFrame | None = None,
        **kwargs: Unpack[model_types.MLFlowLoadOptions],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "mlflow.pyfunc.PyFuncModel",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "mlflow.pyfunc.PyFuncModel",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model._internal_inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame, **method_kwargs: Any) -> pd.DataFrame:
                    # Convert pd.StringDtype columns to object dtype for MLflow compatibility.
                    # MLflow's internal schema validation doesn't handle pd.StringDtype correctly
                    # and fails with "Can not safely convert string to <U0" errors.
                    for col in X.columns:
                        if isinstance(X[col].dtype, pd.StringDtype):
                            X[col] = X[col].astype(object)

                    # Drop None-valued kwargs so snowml's "omit -> default" maps onto MLflow's
                    # defined "key absent -> ParamSchema default" path. MLflow has no runtime
                    # semantic for None in a typed ParamSchema; an explicit None is preserved by
                    # the validator's short-circuit and reaches the wrapped model in a state its
                    # declared dtype does not permit.
                    forwarded = {k: v for k, v in method_kwargs.items() if v is not None}
                    if forwarded:
                        res = raw_model.predict(X, params=forwarded)
                    else:
                        res = raw_model.predict(X)
                    result_df = model_signature_utils.rename_pandas_df(
                        model_signature._convert_local_data_to_df(res), features=signature.outputs
                    )
                    return _ensure_serializable_objects(result_df)

                return fn

            type_method_dict: dict[str, Any] = {"_allows_kwargs": True}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _MLFlowModel = type(
                "_MLFlowModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _MLFlowModel

        _MLFlowModel = _create_custom_model(raw_model, model_meta)
        mlflow_model = _MLFlowModel(custom_model.ModelContext())

        return mlflow_model
