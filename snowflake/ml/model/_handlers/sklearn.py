import os
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_meta as model_meta_api, model_types
from snowflake.ml.model._handlers import _base


class _SKLModelHandler(_base._ModelHandler):
    """Handler for scikit-learn based model.

    Currently sklearn.base.BaseEstimator and sklearn.pipeline.Pipeline based classes are supported.
    """

    handler_type = "sklearn"

    @staticmethod
    def can_handle(model: model_types.ModelType) -> bool:
        return (
            type_utils.LazyType("sklearn.base.BaseEstimator").isinstance(model)
            or type_utils.LazyType("sklearn.pipeline.Pipeline").isinstance(model)
        ) and hasattr(model, "predict")

    @staticmethod
    def _save_model(
        name: str,
        model: model_types.ModelType,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Any,
    ) -> None:
        import sklearn.base
        import sklearn.pipeline

        assert isinstance(model, sklearn.base.BaseEstimator) or isinstance(model, sklearn.pipeline.Pipeline)

        target_method = kwargs.pop("target_method", _SKLModelHandler.DEFAULT_TARGET_METHOD)
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, _SKLModelHandler.MODEL_BLOB_FILE), "wb") as f:
            cloudpickle.dump(model, f)
        base_meta = model_meta_api._ModelBlobMetadata(
            name=name,
            model_type=_SKLModelHandler.handler_type,
            path=_SKLModelHandler.MODEL_BLOB_FILE,
            target_method=target_method,
        )
        model_meta.models[name] = base_meta
        model_meta._include_if_absent(["scikit-learn"])

    @staticmethod
    def _load_model(
        name: str, model_meta: model_meta_api.ModelMetadata, model_blobs_dir_path: str
    ) -> model_types.ModelType:
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models
        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = cloudpickle.load(f)
        return m

    @staticmethod
    def _load_as_custom_model(
        name: str, model_meta: model_meta_api.ModelMetadata, model_blobs_dir_path: str
    ) -> custom_model.CustomModel:
        """Create a custom model class wrap for unified interface when being deployed. The predict method will be
        re-targeted based on target_method metadata.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.

        Returns:
            The model object as a custom model.
        """
        raw_m = _SKLModelHandler._load_model(name, model_meta, model_blobs_dir_path)
        target_method = model_meta.models[name].target_method
        output_col_names = [spec.name for spec in model_meta.schema.outputs]

        class SKLModel(custom_model.CustomModel):
            def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                func = getattr(raw_m, target_method)
                predicts_as_numpy = func(X)
                if (
                    isinstance(predicts_as_numpy, list)
                    and len(predicts_as_numpy) > 0
                    and isinstance(predicts_as_numpy[0], np.ndarray)
                ):
                    # In case of multi-output estimators, predict_proba(), decision_function(), etc., functions return
                    # a list of ndarrays. We need to concatenate them.
                    predicts_as_numpy = np.concatenate(predicts_as_numpy, axis=1)
                return pd.DataFrame(predicts_as_numpy, columns=output_col_names)

        return SKLModel(custom_model.ModelContext())
