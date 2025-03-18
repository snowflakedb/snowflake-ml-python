from typing import cast

from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_meta as model_meta_api,
    model_meta_schema,
)


class PyTorchHandlerMigrator20231201(base_migrator.BaseModelHandlerMigrator):
    source_version = "2023-12-01"
    target_version = "2025-03-01"

    @staticmethod
    def upgrade(name: str, model_meta: model_meta_api.ModelMetadata, model_blobs_dir_path: str) -> None:

        model_blob_metadata = model_meta.models[name]
        model_blob_options = cast(model_meta_schema.PyTorchModelBlobOptions, model_blob_metadata.options)
        model_blob_options["multiple_inputs"] = True
        model_meta.models[name].options = model_blob_options
