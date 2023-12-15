import os
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, cast, final

import cloudpickle
import numpy as np
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
)
from snowflake.ml.model._signatures import numpy_handler, utils as model_signature_utils

if TYPE_CHECKING:
    from snowflake.ml.modeling.framework.base import BaseEstimator


@final
class SnowMLModelHandler(_base.BaseModelHandler["BaseEstimator"]):
    """Handler for SnowML based model.

    Currently snowflake.ml.modeling.framework.base.BaseEstimator
        and snowflake.ml.modeling.pipeline.Pipeline based classes are supported.
    """

    HANDLER_TYPE = "snowml"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    DEFAULT_TARGET_METHODS = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]
    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["BaseEstimator"]:
        return (
            type_utils.LazyType("snowflake.ml.modeling.framework.base.BaseEstimator").isinstance(model)
            # Pipeline is inherited from BaseEstimator, so no need to add one more check
        ) and any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "BaseEstimator":
        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(model, BaseEstimator)
        # Pipeline is inherited from BaseEstimator, so no need to add one more check

        return cast("BaseEstimator", model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "BaseEstimator",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SNOWModelSaveOptions],
    ) -> None:
        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(model, BaseEstimator)
        # Pipeline is inherited from BaseEstimator, so no need to add one more check

        if not is_sub_model:
            if sample_input is not None or model_meta.signatures:
                warnings.warn(
                    "Inferring model signature from sample input or providing model signature for Snowpark ML "
                    + "Modeling model is not required. Model signature will automatically be inferred during fitting. ",
                    UserWarning,
                    stacklevel=2,
                )
            assert hasattr(model, "model_signatures"), "Model does not have model signatures as expected."
            model_meta.signatures = getattr(model, "model_signatures", {})

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR), "wb") as f:
            cloudpickle.dump(model, f)
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        _include_if_absent_pkgs = []
        model_dependencies = model._get_dependencies()
        for dep in model_dependencies:
            pkg_name = dep.split("==")[0]
            _include_if_absent_pkgs.append(model_env.ModelDependency(requirement=pkg_name, pip_name=pkg_name))
        model_meta.env.include_if_absent(_include_if_absent_pkgs, check_local_version=True)

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "BaseEstimator":
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = cloudpickle.load(f)

        from snowflake.ml.modeling.framework.base import BaseEstimator

        assert isinstance(m, BaseEstimator)
        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "BaseEstimator",
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "BaseEstimator",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "BaseEstimator",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    res = getattr(raw_model, target_method)(X)

                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], np.ndarray):
                        # In case of multi-output estimators, predict_proba(), decision_function(), etc., functions
                        # return a list of ndarrays. We need to deal them separately
                        df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(res)
                    else:
                        df = pd.DataFrame(res)

                    return model_signature_utils.rename_pandas_df(df, signature.outputs)

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _SnowMLModel = type(
                "_SnowMLModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _SnowMLModel

        _SnowMLModel = _create_custom_model(raw_model, model_meta)
        snowml_model = _SnowMLModel(custom_model.ModelContext())

        return snowml_model
