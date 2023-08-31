import itertools
import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Callable, Optional, Type, cast

import pandas as pd
import yaml
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import env_utils, file_utils, type_utils
from snowflake.ml.model import (
    _model_meta as model_meta_api,
    custom_model,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._handlers import _base
from snowflake.ml.model._signatures import utils as model_signature_utils

if TYPE_CHECKING:
    import mlflow


def _parse_mlflow_env(model_uri: str, model_meta: model_meta_api.ModelMetadata) -> model_meta_api.ModelMetadata:
    """Parse MLFlow env file and modify model meta based on MLFlow env.

    Args:
        model_uri: Model uri where the env file could be downloaded
        model_meta: model meta to be modified

    Raises:
        ValueError: Raised when cannot download MLFlow model dependencies file.

    Returns:
        Modified model metadata.
    """
    import mlflow

    try:
        conda_env_file_path = mlflow.pyfunc.get_model_dependencies(model_uri, format="conda")

        with open(conda_env_file_path, encoding="utf-8") as f:
            env = yaml.safe_load(stream=f)
    except (mlflow.MlflowException, OSError):
        raise ValueError("Cannot load MLFlow model dependencies.")

    assert isinstance(env, dict)

    mlflow_conda_deps = []
    mlflow_pip_deps = []
    mlflow_python_version = None

    mlflow_conda_channels = env.get("channels", [])

    for dep in env["dependencies"]:
        if isinstance(dep, str):
            ver = env_utils.parse_python_version_string(dep)
            # ver is None: not python, ver is "": python w/o specifier, ver is str: python w/ specifier
            if ver is not None:
                if ver:
                    mlflow_python_version = ver
            else:
                mlflow_conda_deps.append(dep)
        elif isinstance(dep, dict) and "pip" in dep:
            mlflow_pip_deps.extend(dep["pip"])

    if mlflow_python_version:
        model_meta.python_version = mlflow_python_version

    mlflow_conda_deps_dict = env_utils.validate_conda_dependency_string_list(mlflow_conda_deps)
    mlflow_pip_deps_list = env_utils.validate_pip_requirement_string_list(mlflow_pip_deps)

    for mlflow_channel, mlflow_channel_dependencies in mlflow_conda_deps_dict.items():
        if mlflow_channel != env_utils.DEFAULT_CHANNEL_NAME:
            warnings.warn(
                (
                    "Found dependencies from MLflow specified from non-Snowflake channel."
                    + " This may prevent model deploying to Snowflake Warehouse."
                ),
                category=UserWarning,
            )
        for mlflow_channel_dependency in mlflow_channel_dependencies:
            try:
                env_utils.append_conda_dependency(
                    model_meta._conda_dependencies, (mlflow_channel, mlflow_channel_dependency)
                )
            except env_utils.DuplicateDependencyError:
                pass
            except env_utils.DuplicateDependencyInMultipleChannelsError:
                warnings.warn(
                    (
                        f"Dependency {mlflow_channel_dependency.name} appeared in multiple channels."
                        + " This may be unintentional."
                    ),
                    category=UserWarning,
                )

    if mlflow_conda_channels:
        warnings.warn(
            (
                "Found conda channels specified from MLflow."
                + " This may prevent model deploying to Snowflake Warehouse."
            ),
            category=UserWarning,
        )
        for channel_name in mlflow_conda_channels:
            model_meta._conda_dependencies[channel_name] = []

    if mlflow_pip_deps_list:
        warnings.warn(
            (
                "Found dependencies from MLflow specified as pip requirements."
                + " This may prevent model deploying to Snowflake Warehouse."
            ),
            category=UserWarning,
        )
        for mlflow_pip_dependency in mlflow_pip_deps_list:
            if any(
                mlflow_channel_dependency.name == mlflow_pip_dependency.name
                for mlflow_channel_dependency in itertools.chain(*mlflow_conda_deps_dict.values())
            ):
                continue
            env_utils.append_requirement_list(model_meta._pip_requirements, mlflow_pip_dependency)

    return model_meta


class _MLFlowHandler(_base._ModelHandler["mlflow.pyfunc.PyFuncModel"]):
    """Handler for MLFlow based model.

    Currently mlflow.pyfunc.PyFuncModel based classes are supported.
    """

    handler_type = "mlflow"
    MODEL_BLOB_FILE = "model"
    _DEFAULT_TARGET_METHOD = "predict"
    DEFAULT_TARGET_METHODS = [_DEFAULT_TARGET_METHOD]
    is_auto_signature = True

    @staticmethod
    def can_handle(
        model: model_types.SupportedModelType,
    ) -> TypeGuard["mlflow.pyfunc.PyFuncModel"]:
        return type_utils.LazyType("mlflow.pyfunc.PyFuncModel").isinstance(model)

    @staticmethod
    def cast_model(
        model: model_types.SupportedModelType,
    ) -> "mlflow.pyfunc.PyFuncModel":
        import mlflow

        assert isinstance(model, mlflow.pyfunc.PyFuncModel)

        return cast(mlflow.pyfunc.PyFuncModel, model)

    @staticmethod
    def _save_model(
        name: str,
        model: "mlflow.pyfunc.PyFuncModel",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.MLFlowSaveOptions],
    ) -> None:
        import mlflow

        assert isinstance(model, mlflow.pyfunc.PyFuncModel)

        model_info = model.metadata.get_model_info()
        model_uri = kwargs.get("model_uri", model_info.model_uri)

        pyfunc_flavor_info = model_info.flavors.get(mlflow.pyfunc.FLAVOR_NAME, None)
        if pyfunc_flavor_info is None:
            raise ValueError("Cannot save MLFlow model that does not have PyFunc flavor.")

        # Port MLFlow signature
        if not is_sub_model:
            if model_meta._signatures is not None:
                model_meta_api._validate_target_methods(model, list(model_meta.signatures.keys()))
            else:
                model_meta_api._validate_target_methods(model, _MLFlowHandler.DEFAULT_TARGET_METHODS)
                model_meta._signatures = {
                    _MLFlowHandler._DEFAULT_TARGET_METHOD: model_signature.ModelSignature.from_mlflow_sig(
                        model_info.signature
                    )
                }

        # Port MLFlow metadata
        mlflow_model_metadata = model_info.metadata
        if mlflow_model_metadata and not kwargs.get("ignore_mlflow_metadata", False):
            if not model_meta.metadata:
                model_meta.metadata = {}
            model_meta.metadata.update(mlflow_model_metadata)

        # Port MLFlow dependencies
        if kwargs.get("ignore_mlflow_dependencies", False):
            model_meta._include_if_absent([model_meta_api.Dependency(conda_name="mlflow", pip_name="mlflow")])
        else:
            model_meta = _parse_mlflow_env(model_uri, model_meta)

        model_blob_path = os.path.join(model_blobs_dir_path, name)

        os.makedirs(model_blob_path, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmpdir)
            except (mlflow.MlflowException, OSError):
                raise ValueError("Cannot load MLFlow model artifacts.")

            file_utils.copy_file_or_tree(local_path, os.path.join(model_blob_path, _MLFlowHandler.MODEL_BLOB_FILE))

        base_meta = model_meta_api._ModelBlobMetadata(
            name=name,
            model_type=_MLFlowHandler.handler_type,
            path=_MLFlowHandler.MODEL_BLOB_FILE,
            options={"artifact_path": model_info.artifact_path},
        )
        model_meta.models[name] = base_meta

    @staticmethod
    def _load_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "mlflow.pyfunc.PyFuncModel":
        import mlflow

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models
        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
        model_blob_metadata = model_blobs_metadata[name]

        model_blob_options = model_blob_metadata.options

        model_artifact_path = model_blob_options.get("artifact_path", None)
        if model_artifact_path is None:
            raise ValueError("Cannot find a place to load the MLFlow model.")

        model_blob_filename = model_blob_metadata.path

        # This is to make sure the loaded model can be saved again.
        with mlflow.start_run() as run:
            mlflow.log_artifacts(
                os.path.join(model_blob_path, model_blob_filename, model_artifact_path),
                artifact_path=model_artifact_path,
            )
            m = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/{model_artifact_path}")
            m.metadata.run_id = run.info.run_id
        return m

    @staticmethod
    def _load_as_custom_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        """Create a custom model class wrap for unified interface when being deployed. The predict method will be
        re-targeted based on target_method metadata.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.
            kwargs: Options when loading the model.

        Returns:
            The model object as a custom model.
        """
        import mlflow

        from snowflake.ml.model import custom_model

        # We need to redirect the mlruns folder to a writable location in the sandbox.
        tmpdir = tempfile.TemporaryDirectory(dir="/tmp")
        mlflow.set_tracking_uri(f"file://{tmpdir}")

        def _create_custom_model(
            raw_model: "mlflow.pyfunc.PyFuncModel",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "mlflow.pyfunc.PyFuncModel",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    res = raw_model.predict(X)
                    return model_signature_utils.rename_pandas_df(
                        model_signature._convert_local_data_to_df(res), features=signature.outputs
                    )

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _MLFlowModel = type(
                "_MLFlowModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _MLFlowModel

        raw_model = _MLFlowHandler._load_model(name, model_meta, model_blobs_dir_path, **kwargs)
        _MLFlowModel = _create_custom_model(raw_model, model_meta)
        mlflow_model = _MLFlowModel(custom_model.ModelContext())

        return mlflow_model
