# Schema for the thin MANIFEST.yml envelope used by model specification version 2.0.
# Unlike the version 1.0 manifest, this envelope carries no runtime, method, or user-data
# content; all model semantics live in the extension spec it points to.
from typing import TypedDict

from typing_extensions import NotRequired, Required

from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema

MODEL_MANIFEST_VERSION_V2 = "2.0"
DEFAULT_MODEL_EXTENSION_SPEC_PATH = "model/model.yaml"


class ModelManifestV2Dict(TypedDict):
    """Thin MANIFEST.yml for extension-spec-backed model versions."""

    manifest_version: Required[str]
    model_extension_spec: Required[str]
    lineage_sources: NotRequired[list[model_manifest_schema.LineageSourceDict]]
