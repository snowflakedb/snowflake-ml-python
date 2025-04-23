from typing import Any

from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.ml.model._packager.model_meta_migrator import base_migrator, migrator_v1

MODEL_META_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelMetaMigrator]] = {
    "1": migrator_v1.MetaMigrator_v1,
}


def migrate_metadata(loaded_meta: dict[str, Any]) -> dict[str, Any]:
    loaded_meta_version = str(loaded_meta.get("version", None))
    while loaded_meta_version != model_meta_schema.MODEL_METADATA_VERSION:
        if loaded_meta_version not in MODEL_META_MIGRATOR_PLANS.keys():
            raise RuntimeError(
                f"Can not find migrator to migrate model metadata from {loaded_meta_version}"
                f" to version {model_meta_schema.MODEL_METADATA_VERSION}."
            )
        migrator = MODEL_META_MIGRATOR_PLANS[loaded_meta_version]()
        loaded_meta = migrator.try_upgrade(original_meta_dict=loaded_meta)
        loaded_meta_version = str(loaded_meta["version"])

    return loaded_meta
