import os
import pathlib
import tempfile
from typing import TYPE_CHECKING, Callable, Optional, cast, final

import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import file_utils, type_utils
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
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    import mlflow


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
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.MLFlowSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for MLFlow model.")

        import mlflow

        assert isinstance(model, mlflow.pyfunc.PyFuncModel)

        model_info = model.metadata.get_model_info()
        model_uri = kwargs.get("model_uri", model_info.model_uri)

        pyfunc_flavor_info = model_info.flavors.get(mlflow.pyfunc.FLAVOR_NAME, None)
        if pyfunc_flavor_info is None:
            raise ValueError("Cannot save MLFlow model that does not have PyFunc flavor.")

        # Port MLFlow signature
        if not is_sub_model:
            if model_meta.signatures:
                handlers_utils.validate_target_methods(model, list(model_meta.signatures.keys()))
            else:
                handlers_utils.validate_target_methods(model, cls.DEFAULT_TARGET_METHODS)
                model_meta.signatures = {
                    cls._DEFAULT_TARGET_METHOD: model_signature.ModelSignature.from_mlflow_sig(model_info.signature)
                }

        # Port MLFlow metadata
        mlflow_model_metadata = model_info.metadata
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
            model_meta.env = _parse_mlflow_env(model_uri, model_meta.env)

        model_blob_path = os.path.join(model_blobs_dir_path, name)

        os.makedirs(model_blob_path, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=tmpdir)
            except (mlflow.MlflowException, OSError):
                raise ValueError("Cannot load MLFlow model artifacts.")

            file_utils.copy_file_or_tree(local_path, os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR))

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.MLFlowModelBlobOptions({"artifact_path": model_info.artifact_path}),
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

        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            return mlflow.pyfunc.load_model(os.path.join(model_blob_path, model_blob_filename, model_artifact_path))

        # This is to make sure the loaded model can be saved again.
        with mlflow.start_run() as run:
            mlflow.log_artifacts(
                os.path.join(model_blob_path, model_blob_filename, model_artifact_path),
                artifact_path=model_artifact_path,
            )
            m = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/{model_artifact_path}")
            m.metadata.run_id = run.info.run_id
        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "mlflow.pyfunc.PyFuncModel",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
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

        _MLFlowModel = _create_custom_model(raw_model, model_meta)
        mlflow_model = _MLFlowModel(custom_model.ModelContext())

        return mlflow_model
