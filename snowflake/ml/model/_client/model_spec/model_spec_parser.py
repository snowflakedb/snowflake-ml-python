"""Parser for registry model specification documents."""

from collections.abc import Mapping
from typing import Any

from snowflake.ml.model._client.model_spec import (
    legacy_model_spec,
    model_extension_spec,
    model_spec,
)
from snowflake.ml.model._packager.model_meta import model_meta


def parse_model_spec(raw: Any) -> model_spec.ModelSpec:
    """Parse a registry model specification without changing its schema.

    Version 2.0 model extension specs bypass all legacy migration. Other
    documents continue through the existing legacy validator and migrator.

    Args:
        raw: Parsed ``model_spec`` value returned by registry ``SHOW VERSIONS``.

    Returns:
        A read-only model specification adapter.

    Raises:
        ValueError: If the input or a version 2.0 envelope is malformed.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("model specification: document must be a mapping.")

    spec_version = raw.get("version")
    if str(spec_version) == "2.0":
        return model_extension_spec.ModelExtensionSpecV2(raw)

    # Any other version is a legacy packager spec and goes through the existing validator.
    validated = model_meta.ModelMetadata._validate_model_metadata(raw)
    return legacy_model_spec.LegacyModelSpec(validated)
