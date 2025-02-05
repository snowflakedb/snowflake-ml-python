from typing import cast

from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_meta as model_meta_api,
    model_meta_schema,
)


class TensorflowHandlerMigrator20231201(base_migrator.BaseModelHandlerMigrator):
    source_version = "2023-12-01"
    target_version = "2025-01-01"

    @staticmethod
    def upgrade(name: str, model_meta: model_meta_api.ModelMetadata, model_blobs_dir_path: str) -> None:

        model_blob_metadata = model_meta.models[name]
        model_blob_options = cast(model_meta_schema.TensorflowModelBlobOptions, model_blob_metadata.options)
        # To migrate code <= 1.7.0, default to keras model
        is_old_model = "save_format" not in model_blob_options and "is_keras_model" not in model_blob_options
        # To migrate code form 1.7.1, default to False.
        is_keras_model = model_blob_options.get("is_keras_model", False)
        # To migrate code from 1.7.2, default to tf, has options keras, keras_tf, cloudpickle, tf
        #
        # if is_keras_model or is_tf_keras_model:
        #     if is_keras_functional_or_sequential_model:
        #         save_format = "keras"
        #     elif keras_version.major == 2 or is_tf_keras_model:
        #         save_format = "keras_tf"
        #     else:
        #         save_format = "cloudpickle"
        # else:
        #     save_format = "tf"
        #
        save_format = model_blob_options.get("save_format", "tf")

        if save_format == "keras" or is_keras_model or is_old_model:
            save_format = "keras_tf"
        elif save_format == "cloudpickle":
            # Given the old logic, this could only happen if the original model is a keras model, and keras is 3.x
            # However, in this case, keras.Model does not extends from tensorflow.Module
            # So actually TensorflowHandler will not be triggered, we could safely error this out.
            raise NotImplementedError(
                "Unable to upgrade keras 3.x model saved by old handler. This is not supposed to happen"
            )

        model_blob_options["save_format"] = save_format
        model_meta.models[name].options = model_blob_options
