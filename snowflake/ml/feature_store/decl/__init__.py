"""Public API for the declarative feature store library.

Import the key functions and types from this package:

    from snowflake.ml.feature_store.decl import (
        load_specs,
        validate_specs,
        generate_plan,
        fetch_applied_state,
        SpecBatch,
        AppliedState,
        Plan,
        PlanOptions,
        ValidationResult,
        PlanOp,
        AppliedObject,
        SpecLoadError,
        ValidationError,
        DependencyError,
        StateDriftError,
        FSBaseType,
        FeatureViewKind,
        SourceType,
        FeatureAggregationMethod,
        OpKind,
        TYPE_ALIASES,
        normalize_type,
        FSColumn,
        SpecBase,
        Entity,
        StreamingSource,
        BatchSource,
        SourceRef,
        UDF,
        Feature,
        FeatureView,
        FeatureViewRef,
        FeatureGroup,
    )
"""

try:
    from snowflake.ml.feature_store.decl.api import (
        discover_project,
        fetch_applied_state,
        generate_plan,
        load_manifest,
        load_project,
        load_specs,
        resolve_datasource_columns,
        resolve_target,
        validate_specs,
    )
except ModuleNotFoundError:  # lower-layer BUILD targets omit api.py
    pass
from snowflake.ml.feature_store.decl.enums import (
    TYPE_ALIASES,
    FeatureAggregationMethod,
    FeatureViewKind,
    FSBaseType,
    OpKind,
    SourceType,
    normalize_type,
)
from snowflake.ml.feature_store.decl.errors import (
    DependencyError,
    SpecLoadError,
    StateDriftError,
    ValidationError,
)
from snowflake.ml.feature_store.decl.manifest import (
    FSManifest,
    FSTarget,
    FSTemplating,
    InvalidManifestError,
    ManifestConfigurationError,
    ManifestNotFoundError,
    TargetContext,
)
from snowflake.ml.feature_store.decl.project_paths import FSProjectPaths
from snowflake.ml.feature_store.decl.spec_models import (
    UDF,
    Backfill,
    BatchFeatureView,
    BatchSource,
    Entity,
    Feature,
    FeatureGroup,
    FeatureView,
    FeatureViewRef,
    FSColumn,
    RealtimeFeatureView,
    SourceRef,
    SpecBase,
    StorageConfig,
    StreamingFeatureView,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    Plan,
    PlanOp,
    PlanOptions,
    SpecBatch,
    ValidationResult,
)

__all__ = [
    # api.py
    "load_specs",
    "validate_specs",
    "generate_plan",
    "fetch_applied_state",
    "resolve_datasource_columns",
    "load_manifest",
    "discover_project",
    "resolve_target",
    "load_project",
    # enums.py
    "FSBaseType",
    "FeatureViewKind",
    "SourceType",
    "FeatureAggregationMethod",
    "OpKind",
    "TYPE_ALIASES",
    "normalize_type",
    # errors.py
    "SpecLoadError",
    "ValidationError",
    "DependencyError",
    "StateDriftError",
    # manifest.py
    "FSManifest",
    "FSTarget",
    "FSTemplating",
    "TargetContext",
    "ManifestNotFoundError",
    "InvalidManifestError",
    "ManifestConfigurationError",
    # project_paths.py
    "FSProjectPaths",
    # spec_models.py
    "FSColumn",
    "SpecBase",
    "Entity",
    "StreamingSource",
    "BatchSource",
    "SourceRef",
    "UDF",
    "Feature",
    "FeatureView",
    "StreamingFeatureView",
    "BatchFeatureView",
    "RealtimeFeatureView",
    "FeatureViewRef",
    "FeatureGroup",
    "Backfill",
    "StorageConfig",
    # types.py
    "AppliedObject",
    "AppliedState",
    "PlanOp",
    "Plan",
    "ValidationResult",
    "PlanOptions",
    "SpecBatch",
]
