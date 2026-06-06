"""Metadata manager for Feature Store internal metadata table.

This module provides a centralized class for managing the internal metadata table
used by Feature Store to store configuration that doesn't fit in Snowflake object
properties (tags, comments, etc.).

Currently used for:
- Feature specifications for tiled feature views
- Feature descriptions for tiled feature views (since tile columns differ from output columns)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from snowflake.ml.feature_store.aggregation import AggregationSpec

if TYPE_CHECKING:
    from snowflake.snowpark import Session


# Table and column names
_METADATA_TABLE_NAME = "_FEATURE_STORE_METADATA"
_METADATA_TABLE_COMMENT = (
    "Internal metadata table for Feature Store. " "DO NOT modify directly - used for Feature Store internal operations."
)


class MetadataObjectType(str, Enum):
    """Types of objects that can have metadata stored."""

    FEATURE_VIEW = "FEATURE_VIEW"
    STREAM_SOURCE = "STREAM_SOURCE"
    FEATURE_GROUP = "FEATURE_GROUP"


class MetadataType(str, Enum):
    """Types of metadata that can be stored."""

    FEATURE_SPECS = "FEATURE_SPECS"
    FEATURE_DESCS = "FEATURE_DESCS"
    ROLLUP_CONFIG = "ROLLUP_CONFIG"
    STREAM_SOURCE_CONFIG = "STREAM_SOURCE_CONFIG"
    STREAM_CONFIG = "STREAM_CONFIG"
    FEATURE_GROUP_CONFIG = "FEATURE_GROUP_CONFIG"
    # Persisted configuration for realtime feature views (RTFV). Holds
    # everything ``get_feature_view`` needs to faithfully reconstruct the
    # original :class:`RealtimeConfig`: compute_fn source + name, source FV
    # references (as ``FvSourceRef`` rows), request source schema, output
    # schema, and the captured output_columns list.
    REALTIME_CONFIG = "REALTIME_CONFIG"
    FEATURE_VIEW_METADATA = "FEATURE_VIEW_METADATA"


@dataclass
class FeatureViewMetadataConfig:
    """General-purpose metadata for a registered feature view.

    Stored under MetadataType.FEATURE_VIEW_METADATA. Currently tracks the
    snowml package version that authored the feature view, used to select
    between legacy and new tile/merge behavior.
    """

    authoring_pkg_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"authoring_pkg_version": self.authoring_pkg_version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureViewMetadataConfig:
        """Create from dictionary."""
        return cls(authoring_pkg_version=data["authoring_pkg_version"])


@dataclass
class AggregationMetadata:
    """Aggregation configuration for tiled feature views."""

    feature_granularity: str
    features: list[AggregationSpec]
    feature_aggregation_method: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: dict[str, Any] = {
            "feature_granularity": self.feature_granularity,
            "features": [f.to_dict() for f in self.features],
        }
        if self.feature_aggregation_method is not None:
            d["feature_aggregation_method"] = self.feature_aggregation_method
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregationMetadata:
        """Create from dictionary."""
        return cls(
            feature_granularity=data["feature_granularity"],
            features=[AggregationSpec.from_dict(f) for f in data["features"]],
            feature_aggregation_method=data.get("feature_aggregation_method"),
        )


# Allowed values for ``StreamingMetadata.backfill_state``.
# ``RUNNING`` is written client-side at registration after task graph DDL succeeds.
# ``COMPLETED``/``FAILED`` are written server-side by the finalizer task body.
BACKFILL_STATE_RUNNING = "RUNNING"
BACKFILL_STATE_COMPLETED = "COMPLETED"
BACKFILL_STATE_FAILED = "FAILED"
_VALID_BACKFILL_STATES = frozenset({BACKFILL_STATE_RUNNING, BACKFILL_STATE_COMPLETED, BACKFILL_STATE_FAILED})


@dataclass
class StreamingMetadata:
    """Streaming configuration metadata for streaming feature views.

    Stored in the metadata table (not the tag) to avoid 256-char tag limit.
    """

    stream_source_name: str
    transformation_fn_name: str
    transformation_fn_source: Optional[str] = None  # Full source code of the UDF
    backfill_start_time: Optional[str] = None  # ISO format, or None if no filter
    backfill_root_task_name: Optional[str] = None  # Root task of the backfill task graph
    backfill_finalize_task_name: Optional[str] = None  # Finalizer task of the backfill task graph
    backfill_proc_name: Optional[str] = None  # Stored procedure that the root task CALLs
    backfill_udtf_name: Optional[str] = None  # Per-FV permanent UDTF the proc invokes
    backfill_udtf_signature: Optional[str] = None  # Argument-type signature for DROP FUNCTION
    # ``RUNNING`` while the backfill task graph is in flight, ``COMPLETED`` /
    # ``FAILED`` once the finalizer has observed terminal state. ``None`` for
    # streaming FVs registered before this field existed.
    backfill_state: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: dict[str, Any] = {
            "stream_source_name": self.stream_source_name,
            "transformation_fn_name": self.transformation_fn_name,
        }
        if self.transformation_fn_source is not None:
            d["transformation_fn_source"] = self.transformation_fn_source
        if self.backfill_start_time is not None:
            d["backfill_start_time"] = self.backfill_start_time
        if self.backfill_root_task_name is not None:
            d["backfill_root_task_name"] = self.backfill_root_task_name
        if self.backfill_finalize_task_name is not None:
            d["backfill_finalize_task_name"] = self.backfill_finalize_task_name
        if self.backfill_proc_name is not None:
            d["backfill_proc_name"] = self.backfill_proc_name
        if self.backfill_udtf_name is not None:
            d["backfill_udtf_name"] = self.backfill_udtf_name
        if self.backfill_udtf_signature is not None:
            d["backfill_udtf_signature"] = self.backfill_udtf_signature
        if self.backfill_state is not None:
            d["backfill_state"] = self.backfill_state
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamingMetadata:
        """Create from dictionary.

        Unknown keys (e.g. legacy ``backfill_query_id`` from rows written by
        earlier versions) are silently ignored. Unrecognized ``backfill_state``
        values are dropped to ``None`` rather than crashing forward-loading.

        Args:
            data: JSON-decoded streaming metadata blob.

        Returns:
            ``StreamingMetadata`` instance.
        """
        raw_backfill_state = data.get("backfill_state")
        if raw_backfill_state is not None and raw_backfill_state not in _VALID_BACKFILL_STATES:
            raw_backfill_state = None
        return cls(
            stream_source_name=data["stream_source_name"],
            transformation_fn_name=data["transformation_fn_name"],
            transformation_fn_source=data.get("transformation_fn_source"),
            backfill_start_time=data.get("backfill_start_time"),
            backfill_root_task_name=data.get("backfill_root_task_name"),
            backfill_finalize_task_name=data.get("backfill_finalize_task_name"),
            backfill_proc_name=data.get("backfill_proc_name"),
            backfill_udtf_name=data.get("backfill_udtf_name"),
            backfill_udtf_signature=data.get("backfill_udtf_signature"),
            backfill_state=raw_backfill_state,
        )


@dataclass
class FvSourceRef:
    """One source FV reference for any consumer that lists source FVs.

    Originally named ``FeatureGroupSourceRef`` and used only by
    :class:`FeatureGroup`; the realtime feature view (RTFV) authoring path
    reuses the same shape for its persisted ``sources`` list. The shape is
    deliberately consumer-agnostic — only ``(fv_name, fv_version)`` plus the
    optional slice/alias projection — so the same JSON shape round-trips
    through every consumer's metadata column.

    Captures everything :meth:`FeatureStore.get_feature_group` and
    :meth:`FeatureStore.get_feature_view` (for RTFVs) need to reconstruct
    the original feature item:

    - ``fv_name`` / ``fv_version``: identity of the upstream FeatureView.
    - ``slice_columns``: ``None`` for a full FeatureView, or the list of
      selected feature names (in slice order) for a FeatureViewSlice.
    - ``alias``: ``None`` if no :meth:`with_name` was applied; otherwise
      the alias string (``""`` is a valid alias meaning "no prefix"). RTFV
      sources never carry an alias; the field is preserved for FG.
    """

    fv_name: str
    fv_version: str
    slice_columns: Optional[list[str]] = None
    alias: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        d: dict[str, Any] = {"fv_name": self.fv_name, "fv_version": self.fv_version}
        if self.slice_columns is not None:
            d["slice_columns"] = list(self.slice_columns)
        if self.alias is not None:
            d["alias"] = self.alias
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FvSourceRef:
        """Build from a dict produced by :meth:`to_dict`."""
        return cls(
            fv_name=data["fv_name"],
            fv_version=data["fv_version"],
            slice_columns=list(data["slice_columns"]) if "slice_columns" in data else None,
            alias=data.get("alias"),
        )


# Backward-compatibility alias. Internal name was ``FeatureGroupSourceRef``
# before RTFV reuse generalized it; existing callers (FG path, tests) keep
# working with no rename ripple.
FeatureGroupSourceRef = FvSourceRef


@dataclass
class RealtimeConfigMetadata:
    """Persisted configuration for a realtime feature view (RTFV).

    Holds everything ``FeatureStore.get_feature_view`` needs to reconstruct
    an equivalent :class:`RealtimeConfig` without re-parsing the OFT spec
    JSON. Keyed by ``(FEATURE_VIEW, name, version)`` in the metadata table,
    same shape as :class:`FeatureGroupMetadata`.

    Attributes:
        name: Resolved (case-canonical) RTFV name.
        version: RTFV version (case-preserved).
        desc: User-supplied description.
        compute_fn_name: ``__name__`` of the ``compute_fn`` callable.
        compute_fn_source: Plain-text source of the ``compute_fn``.
        sources: Upstream FV references in source order (excluding the
            optional :class:`RequestSource`, which is persisted separately
            via ``request_schema_json``).
        request_schema_json: JSON-encoded :class:`StructType` for the
            :class:`RequestSource`. Stored as JSON rather than a structured
            list so the round trip preserves Snowpark type metadata exactly.
            ``None`` when the RTFV was registered without a RequestSource.
        output_schema_json: JSON-encoded :class:`StructType` for the
            realtime output schema.
        output_columns: Resolved output column names in source-emission
            order. Captured at registration time so listing / read APIs
            don't have to re-derive them.
        entity_names: Resolved (case-canonical) names of the RTFV's
            declared entities, in declaration order and de-duplicated.
            Captured at registration time so ``list_feature_views`` can
            render the RTFV row without re-fetching each upstream FV.
    """

    name: str
    version: str
    desc: str
    compute_fn_name: str
    compute_fn_source: str
    sources: list[FvSourceRef]
    request_schema_json: Optional[str]
    output_schema_json: str
    output_columns: list[str]
    entity_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "name": self.name,
            "version": self.version,
            "desc": self.desc,
            "compute_fn_name": self.compute_fn_name,
            "compute_fn_source": self.compute_fn_source,
            "sources": [s.to_dict() for s in self.sources],
            "request_schema_json": self.request_schema_json,
            "output_schema_json": self.output_schema_json,
            "output_columns": list(self.output_columns),
            "entity_names": list(self.entity_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RealtimeConfigMetadata:
        """Build from a dict produced by :meth:`to_dict`."""
        return cls(
            name=data["name"],
            version=data["version"],
            desc=data.get("desc", ""),
            compute_fn_name=data["compute_fn_name"],
            compute_fn_source=data["compute_fn_source"],
            sources=[FvSourceRef.from_dict(s) for s in data.get("sources", [])],
            request_schema_json=data.get("request_schema_json"),
            output_schema_json=data["output_schema_json"],
            output_columns=list(data.get("output_columns") or []),
            entity_names=list(data.get("entity_names") or []),
        )


@dataclass
class FeatureGroupMetadata:
    """Persisted configuration for a registered :class:`FeatureGroup`, keyed by ``(name, version)``.

    ``output_columns`` is captured at register time and persisted so
    ``list_feature_groups`` does not have to rehydrate source FVs; ``Optional``
    so legacy rows written before the field existed decode as ``None``.
    """

    name: str
    version: str
    desc: str
    auto_prefix: bool
    sources: list[FeatureGroupSourceRef]
    output_columns: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "desc": self.desc,
            "auto_prefix": self.auto_prefix,
            "sources": [s.to_dict() for s in self.sources],
        }
        if self.output_columns is not None:
            d["output_columns"] = list(self.output_columns)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureGroupMetadata:
        """Build from a dict produced by :meth:`to_dict`."""
        raw_output_columns = data.get("output_columns")
        return cls(
            name=data["name"],
            version=data["version"],
            desc=data.get("desc", ""),
            auto_prefix=bool(data["auto_prefix"]),
            sources=[FeatureGroupSourceRef.from_dict(s) for s in data.get("sources", [])],
            output_columns=list(raw_output_columns) if raw_output_columns is not None else None,
        )


class FeatureStoreMetadataManager:
    """Manages the internal metadata table for Feature Store objects.

    This class encapsulates all operations on the _FEATURE_STORE_METADATA table,
    providing typed APIs for reading and writing different types of metadata.

    The metadata table schema:
        - OBJECT_TYPE: Type of object (e.g., 'FEATURE_VIEW')
        - OBJECT_NAME: Name of the object
        - VERSION: Version of the object (nullable for non-versioned objects)
        - METADATA_TYPE: Type of metadata (e.g., 'FEATURE_SPECS', 'FEATURE_DESCS')
        - METADATA: VARIANT column containing the actual metadata as JSON
        - CREATED_AT: Timestamp when the entry was created
        - UPDATED_AT: Timestamp when the entry was last updated
    """

    def __init__(
        self,
        session: Session,
        schema_path: str,
        fs_object_tag_path: str,
        telemetry_stmp: dict[str, Any],
    ) -> None:
        """Initialize the metadata manager.

        Args:
            session: Snowpark session.
            schema_path: Fully qualified schema path (e.g., "DB.SCHEMA").
            fs_object_tag_path: Fully qualified path to the feature store object tag.
            telemetry_stmp: Telemetry statement parameters.
        """
        self._session = session
        self._schema_path = schema_path
        self._fs_object_tag_path = fs_object_tag_path
        self._table_path = f"{schema_path}.{_METADATA_TABLE_NAME}"
        self._telemetry_stmp = telemetry_stmp
        self._table_exists: Optional[bool] = None

    def ensure_table_exists(self) -> None:
        """Create the metadata table if it doesn't exist.

        This method is idempotent and safe to call multiple times.
        The table is tagged as a feature store object and has a description
        indicating it's for internal use only.
        """
        if self._table_exists:
            return

        # Create the table
        self._session.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_path} (
                OBJECT_TYPE VARCHAR(50) NOT NULL,
                OBJECT_NAME VARCHAR(256) NOT NULL,
                VERSION VARCHAR(128) NOT NULL,
                METADATA_TYPE VARCHAR(50) NOT NULL,
                METADATA VARIANT NOT NULL,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (OBJECT_TYPE, OBJECT_NAME, VERSION, METADATA_TYPE)
            )
            COMMENT = '{_METADATA_TABLE_COMMENT}'
            """
        ).collect(statement_params=self._telemetry_stmp)

        # Add feature store object tag to identify this as an internal FS object
        # Import here to avoid circular dependency
        import snowflake.ml.version as snowml_version
        from snowflake.ml.feature_store.feature_store import (
            _FeatureStoreObjInfo,
            _FeatureStoreObjTypes,
        )

        obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.INTERNAL_METADATA_TABLE, snowml_version.VERSION)
        self._session.sql(
            f"""
            ALTER TABLE {self._table_path}
            SET TAG {self._fs_object_tag_path} = '{obj_info.to_json()}'
            """
        ).collect(statement_params=self._telemetry_stmp)

        self._table_exists = True

    # =========================================================================
    # Feature Specs
    # =========================================================================

    def save_feature_specs(
        self,
        fv_name: str,
        version: str,
        metadata: AggregationMetadata,
    ) -> None:
        """Save feature specifications for a tiled feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
            metadata: Aggregation metadata to save.
        """
        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.FEATURE_SPECS,
            metadata=metadata.to_dict(),
        )

    def get_feature_specs(
        self,
        fv_name: str,
        version: str,
    ) -> Optional[AggregationMetadata]:
        """Get feature specifications for a tiled feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.

        Returns:
            AggregationMetadata if found, None otherwise.
        """
        data = self._get_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.FEATURE_SPECS,
        )
        if data is None:
            return None
        return AggregationMetadata.from_dict(data)

    # =========================================================================
    # Feature Descriptions
    # =========================================================================

    def save_feature_descs(
        self,
        fv_name: str,
        version: str,
        descs: dict[str, str],
    ) -> None:
        """Save feature descriptions for a tiled feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
            descs: Dictionary mapping output column names to descriptions.
        """
        if not descs:
            return  # Don't save empty descriptions

        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.FEATURE_DESCS,
            metadata=descs,
        )

    def save_feature_view_metadata(
        self,
        fv_name: str,
        version: str,
        specs: AggregationMetadata,
        descs: Optional[dict[str, str]] = None,
        fv_metadata_config: Optional[FeatureViewMetadataConfig] = None,
    ) -> None:
        """Save all metadata for a tiled feature view atomically.

        This method saves feature specs, descriptions, and general FV metadata
        in a single INSERT statement for atomicity during creation.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
            specs: Aggregation metadata (required).
            descs: Optional dictionary of feature descriptions.
            fv_metadata_config: Optional general-purpose FV metadata (e.g. authoring version).
        """
        self.ensure_table_exists()

        normalized_name = fv_name.strip('"')
        specs_json = json.dumps(specs.to_dict())

        # Build SELECT statements for atomic insert (PARSE_JSON can't be used in VALUES)
        selects = [
            f"SELECT '{MetadataObjectType.FEATURE_VIEW.value}', '{normalized_name}', "
            f"'{version}', '{MetadataType.FEATURE_SPECS.value}', PARSE_JSON($${specs_json}$$)"
        ]

        if descs:
            descs_json = json.dumps(descs)
            selects.append(
                f"SELECT '{MetadataObjectType.FEATURE_VIEW.value}', '{normalized_name}', "
                f"'{version}', '{MetadataType.FEATURE_DESCS.value}', PARSE_JSON($${descs_json}$$)"
            )

        if fv_metadata_config is not None:
            config_json = json.dumps(fv_metadata_config.to_dict())
            selects.append(
                f"SELECT '{MetadataObjectType.FEATURE_VIEW.value}', '{normalized_name}', "
                f"'{version}', '{MetadataType.FEATURE_VIEW_METADATA.value}', PARSE_JSON($${config_json}$$)"
            )

        union_query = " UNION ALL ".join(selects)

        self._session.sql(
            f"""
            INSERT INTO {self._table_path}
            (OBJECT_TYPE, OBJECT_NAME, VERSION, METADATA_TYPE, METADATA)
            {union_query}
            """
        ).collect(statement_params=self._telemetry_stmp)

    def get_feature_descs(
        self,
        fv_name: str,
        version: str,
    ) -> Optional[dict[str, str]]:
        """Get feature descriptions for a tiled feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.

        Returns:
            Dictionary of feature descriptions if found, None otherwise.
        """
        data = self._get_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.FEATURE_DESCS,
        )
        return data

    # =========================================================================
    # Feature View Metadata Config
    # =========================================================================

    def get_feature_view_metadata_config(
        self,
        fv_name: str,
        version: str,
    ) -> Optional[FeatureViewMetadataConfig]:
        """Get general-purpose metadata for a feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.

        Returns:
            FeatureViewMetadataConfig if found, None otherwise.
        """
        data = self._get_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.FEATURE_VIEW_METADATA,
        )
        if data is None:
            return None
        return FeatureViewMetadataConfig.from_dict(data)

    # =========================================================================
    # Rollup Config
    # =========================================================================

    def save_rollup_metadata(
        self,
        fv_name: str,
        version: str,
        metadata: dict[str, Any],
    ) -> None:
        """Save rollup configuration metadata for a rollup feature view.

        This stores parent tile table, join keys, mapping query, and optional
        mapping_valid_from_col / mapping_valid_to_col needed for PIT-correct ASOF JOIN
        at training time.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
            metadata: Dictionary from RollupMetadata.to_dict().
        """
        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.ROLLUP_CONFIG,
            metadata=metadata,
        )

    def get_rollup_metadata(
        self,
        fv_name: str,
        version: str,
    ) -> Optional[dict[str, Any]]:
        """Get rollup configuration metadata for a feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.

        Returns:
            Dictionary suitable for RollupMetadata.from_dict() if found, None otherwise.
        """
        return self._get_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.ROLLUP_CONFIG,
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def delete_feature_view_metadata(
        self,
        fv_name: str,
        version: str,
    ) -> None:
        """Delete all metadata entries for a feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
        """
        # Check if table exists before trying to delete
        if not self._check_table_exists():
            return

        normalized_name = fv_name.strip('"')

        self._session.sql(
            f"""
            DELETE FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.FEATURE_VIEW.value}'
            AND OBJECT_NAME = '{normalized_name}'
            AND VERSION = '{version}'
            """
        ).collect(statement_params=self._telemetry_stmp)

    # =========================================================================
    # Streaming Feature View Metadata
    # =========================================================================

    def save_streaming_metadata(
        self,
        fv_name: str,
        version: str,
        metadata: StreamingMetadata,
    ) -> None:
        """Save streaming metadata for a streaming feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
            metadata: Streaming metadata to save.
        """
        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.STREAM_CONFIG,
            metadata=metadata.to_dict(),
        )

    def get_streaming_metadata(
        self,
        fv_name: str,
        version: str,
    ) -> Optional[StreamingMetadata]:
        """Get streaming metadata for a streaming feature view.

        Args:
            fv_name: Feature view name.
            version: Feature view version.

        Returns:
            StreamingMetadata if found, None otherwise.
        """
        data = self._get_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=fv_name,
            version=version,
            metadata_type=MetadataType.STREAM_CONFIG,
        )
        if data is None:
            return None
        return StreamingMetadata.from_dict(data)

    # =========================================================================
    # Feature Group Metadata
    # =========================================================================

    def save_feature_group_metadata(self, metadata: FeatureGroupMetadata) -> None:
        """Persist :class:`FeatureGroupMetadata` for a registered FeatureGroup.

        Args:
            metadata: The metadata to upsert.
        """
        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.FEATURE_GROUP,
            object_name=metadata.name,
            version=metadata.version,
            metadata_type=MetadataType.FEATURE_GROUP_CONFIG,
            metadata=metadata.to_dict(),
        )

    def get_feature_group_metadata(self, name: str, version: str) -> Optional[FeatureGroupMetadata]:
        """Read :class:`FeatureGroupMetadata` for a registered FeatureGroup.

        Args:
            name: FeatureGroup name (resolved identifier, no quoting).
            version: FeatureGroup version (case-preserved string).

        Returns:
            The metadata, or ``None`` if no row exists.
        """
        data = self._get_metadata(
            object_type=MetadataObjectType.FEATURE_GROUP,
            object_name=name,
            version=version,
            metadata_type=MetadataType.FEATURE_GROUP_CONFIG,
        )
        if data is None:
            return None
        return FeatureGroupMetadata.from_dict(data)

    def list_feature_group_metadata(self) -> list[FeatureGroupMetadata]:
        """Return every persisted :class:`FeatureGroupMetadata` row.

        Used by :meth:`FeatureStore.list_feature_groups` to surface the
        ``version`` column without N+1 metadata fetches.

        Returns:
            All FG metadata rows in unspecified order; empty list when the
            metadata table doesn't exist or holds no FG rows.
        """
        if not self._check_table_exists():
            return []

        result = self._session.sql(
            f"""
            SELECT METADATA
            FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.FEATURE_GROUP.value}'
            AND METADATA_TYPE = '{MetadataType.FEATURE_GROUP_CONFIG.value}'
            """
        ).collect(statement_params=self._telemetry_stmp)

        out: list[FeatureGroupMetadata] = []
        for row in result:
            metadata_value = row["METADATA"]
            if isinstance(metadata_value, str):
                data = json.loads(metadata_value)
            else:
                data = dict(metadata_value)
            out.append(FeatureGroupMetadata.from_dict(data))
        return out

    def delete_feature_group_metadata(self, name: str, version: str) -> None:
        """Delete the metadata row for a FeatureGroup ``(name, version)``, if present.

        Args:
            name: FeatureGroup name (resolved identifier, no quoting).
            version: FeatureGroup version (case-preserved string).
        """
        if not self._check_table_exists():
            return

        normalized_name = name.strip('"')
        self._session.sql(
            f"""
            DELETE FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.FEATURE_GROUP.value}'
            AND OBJECT_NAME = '{normalized_name}'
            AND VERSION = '{version}'
            AND METADATA_TYPE = '{MetadataType.FEATURE_GROUP_CONFIG.value}'
            """
        ).collect(statement_params=self._telemetry_stmp)

    # =========================================================================
    # Realtime Feature View Configuration
    # =========================================================================

    def save_realtime_config(self, metadata: RealtimeConfigMetadata) -> None:
        """Persist a :class:`RealtimeConfigMetadata` row for a registered RTFV.

        Keyed by ``(FEATURE_VIEW, name, version)`` — RTFV metadata coexists
        alongside other FV metadata types (specs/descs/rollup/stream) under
        the same primary key.

        Args:
            metadata: The metadata to upsert.
        """
        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=metadata.name,
            version=metadata.version,
            metadata_type=MetadataType.REALTIME_CONFIG,
            metadata=metadata.to_dict(),
        )

    def get_realtime_config(self, name: str, version: str) -> Optional[RealtimeConfigMetadata]:
        """Read :class:`RealtimeConfigMetadata` for an RTFV ``(name, version)``.

        Args:
            name: RTFV name (resolved identifier, no quoting).
            version: RTFV version (case-preserved string).

        Returns:
            The metadata, or ``None`` if no row exists.
        """
        data = self._get_metadata(
            object_type=MetadataObjectType.FEATURE_VIEW,
            object_name=name,
            version=version,
            metadata_type=MetadataType.REALTIME_CONFIG,
        )
        if data is None:
            return None
        return RealtimeConfigMetadata.from_dict(data)

    def list_realtime_config_metadata(self) -> list[RealtimeConfigMetadata]:
        """Return every persisted :class:`RealtimeConfigMetadata` row.

        Used by :meth:`FeatureStore.list_feature_views` to enumerate RTFV
        rows without per-row metadata fetches. The OFT backing the RTFV is
        listed separately so transient OFT drops surface as ``owner=None``
        in the returned listing (same shape as FG).

        Returns:
            All RTFV metadata rows in unspecified order; empty list when
            the metadata table doesn't exist or holds no RTFV rows.
        """
        if not self._check_table_exists():
            return []

        result = self._session.sql(
            f"""
            SELECT METADATA
            FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.FEATURE_VIEW.value}'
            AND METADATA_TYPE = '{MetadataType.REALTIME_CONFIG.value}'
            """
        ).collect(statement_params=self._telemetry_stmp)

        out: list[RealtimeConfigMetadata] = []
        for row in result:
            metadata_value = row["METADATA"]
            if isinstance(metadata_value, str):
                data = json.loads(metadata_value)
            else:
                data = dict(metadata_value)
            out.append(RealtimeConfigMetadata.from_dict(data))
        return out

    def delete_realtime_config(self, name: str, version: str) -> None:
        """Delete the :class:`RealtimeConfigMetadata` row for an RTFV ``(name, version)``.

        No-op when the metadata table is missing or the row isn't there.

        Args:
            name: RTFV name (resolved identifier, no quoting).
            version: RTFV version (case-preserved string).
        """
        if not self._check_table_exists():
            return

        normalized_name = name.strip('"')
        self._session.sql(
            f"""
            DELETE FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.FEATURE_VIEW.value}'
            AND OBJECT_NAME = '{normalized_name}'
            AND VERSION = '{version}'
            AND METADATA_TYPE = '{MetadataType.REALTIME_CONFIG.value}'
            """
        ).collect(statement_params=self._telemetry_stmp)

    # =========================================================================
    # Stream Source
    # =========================================================================

    def save_stream_source(self, name: str, metadata: dict[str, Any]) -> None:
        """Save a stream source configuration.

        Args:
            name: Stream source name (resolved identifier).
            metadata: Dictionary containing schema, desc, owner, and ref_count.
        """
        self.ensure_table_exists()
        self._upsert_metadata(
            object_type=MetadataObjectType.STREAM_SOURCE,
            object_name=name,
            version="",
            metadata_type=MetadataType.STREAM_SOURCE_CONFIG,
            metadata=metadata,
        )

    def get_stream_source_metadata(self, name: str) -> Optional[dict[str, Any]]:
        """Get a stream source configuration by name.

        Args:
            name: Stream source name (resolved identifier).

        Returns:
            Metadata dictionary if found, None otherwise.
        """
        return self._get_metadata(
            object_type=MetadataObjectType.STREAM_SOURCE,
            object_name=name,
            version="",
            metadata_type=MetadataType.STREAM_SOURCE_CONFIG,
        )

    def list_stream_source_metadata(self) -> list[dict[str, Any]]:
        """List all stream source configurations.

        Returns:
            List of metadata dictionaries for all registered stream sources.
        """
        if not self._check_table_exists():
            return []

        result = self._session.sql(
            f"""
            SELECT METADATA
            FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.STREAM_SOURCE.value}'
            AND METADATA_TYPE = '{MetadataType.STREAM_SOURCE_CONFIG.value}'
            ORDER BY OBJECT_NAME
            """
        ).collect(statement_params=self._telemetry_stmp)

        sources = []
        for row in result:
            metadata_value = row["METADATA"]
            if isinstance(metadata_value, str):
                sources.append(json.loads(metadata_value))
            else:
                sources.append(dict(metadata_value))
        return sources

    def delete_stream_source_metadata(self, name: str) -> None:
        """Delete a stream source configuration.

        Args:
            name: Stream source name (resolved identifier).
        """
        if not self._check_table_exists():
            return

        normalized_name = name.strip('"')

        self._session.sql(
            f"""
            DELETE FROM {self._table_path}
            WHERE OBJECT_TYPE = '{MetadataObjectType.STREAM_SOURCE.value}'
            AND OBJECT_NAME = '{normalized_name}'
            """
        ).collect(statement_params=self._telemetry_stmp)

    def stream_source_exists(self, name: str) -> bool:
        """Check if a stream source exists.

        Args:
            name: Stream source name (resolved identifier).

        Returns:
            True if the stream source exists, False otherwise.
        """
        return self.get_stream_source_metadata(name) is not None

    def get_stream_source_ref_count(self, name: str) -> int:
        """Get the reference count for a stream source.

        Args:
            name: Stream source name (resolved identifier).

        Returns:
            Number of active references. 0 if stream source not found.
        """
        metadata = self.get_stream_source_metadata(name)
        if metadata is None:
            return 0
        return int(metadata.get("ref_count", 0))

    def increment_stream_source_ref_count(self, name: str) -> None:
        """Increment the reference count of a stream source.

        Called when a FeatureView that uses this stream source is registered.

        Args:
            name: Stream source name (resolved identifier).
        """
        metadata = self.get_stream_source_metadata(name)
        if metadata is None:
            return

        metadata["ref_count"] = int(metadata.get("ref_count", 0)) + 1
        self._upsert_metadata(
            object_type=MetadataObjectType.STREAM_SOURCE,
            object_name=name,
            version="",
            metadata_type=MetadataType.STREAM_SOURCE_CONFIG,
            metadata=metadata,
        )

    def decrement_stream_source_ref_count(self, name: str) -> None:
        """Decrement the reference count of a stream source (clamped to 0).

        Called when a FeatureView that uses this stream source is deleted.

        Args:
            name: Stream source name (resolved identifier).
        """
        metadata = self.get_stream_source_metadata(name)
        if metadata is None:
            return

        current = int(metadata.get("ref_count", 0))
        metadata["ref_count"] = max(0, current - 1)
        self._upsert_metadata(
            object_type=MetadataObjectType.STREAM_SOURCE,
            object_name=name,
            version="",
            metadata_type=MetadataType.STREAM_SOURCE_CONFIG,
            metadata=metadata,
        )

    def update_stream_source_desc(self, name: str, desc: str) -> None:
        """Update the description of an existing stream source.

        Args:
            name: Stream source name (resolved identifier).
            desc: New description to set.
        """
        existing = self.get_stream_source_metadata(name)
        if existing is None:
            return

        existing["desc"] = desc
        self._upsert_metadata(
            object_type=MetadataObjectType.STREAM_SOURCE,
            object_name=name,
            version="",
            metadata_type=MetadataType.STREAM_SOURCE_CONFIG,
            metadata=existing,
        )

    @property
    def table_path(self) -> str:
        """Fully qualified path to the metadata table."""
        return self._table_path

    def table_exists(self) -> bool:
        """Check if the metadata table exists."""
        return self._check_table_exists()

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _check_table_exists(self) -> bool:
        """Check if the metadata table exists."""
        if self._table_exists is not None:
            return self._table_exists

        result = self._session.sql(
            f"""
            SELECT COUNT(*) as cnt
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{self._schema_path.split('.')[1]}'
            AND TABLE_NAME = '{_METADATA_TABLE_NAME}'
            """
        ).collect(statement_params=self._telemetry_stmp)

        self._table_exists = result[0]["CNT"] > 0
        return self._table_exists

    def _upsert_metadata(
        self,
        object_type: MetadataObjectType,
        object_name: str,
        version: str,
        metadata_type: MetadataType,
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a metadata entry."""
        metadata_json = json.dumps(metadata)
        # Strip surrounding quotes if present; callers pass resolved() which handles casing.
        normalized_name = object_name.strip('"')

        self._session.sql(
            f"""
            MERGE INTO {self._table_path} AS target
            USING (
                SELECT
                    '{object_type.value}' AS OBJECT_TYPE,
                    '{normalized_name}' AS OBJECT_NAME,
                    '{version}' AS VERSION,
                    '{metadata_type.value}' AS METADATA_TYPE,
                    PARSE_JSON($${metadata_json}$$) AS METADATA
            ) AS source
            ON target.OBJECT_TYPE = source.OBJECT_TYPE
            AND target.OBJECT_NAME = source.OBJECT_NAME
            AND target.VERSION = source.VERSION
            AND target.METADATA_TYPE = source.METADATA_TYPE
            WHEN MATCHED THEN UPDATE SET
                METADATA = source.METADATA,
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                OBJECT_TYPE, OBJECT_NAME, VERSION, METADATA_TYPE, METADATA
            ) VALUES (
                source.OBJECT_TYPE, source.OBJECT_NAME,
                source.VERSION, source.METADATA_TYPE, source.METADATA
            )
            """
        ).collect(statement_params=self._telemetry_stmp)

    def _get_metadata(
        self,
        object_type: MetadataObjectType,
        object_name: str,
        version: str,
        metadata_type: MetadataType,
    ) -> Optional[dict[str, Any]]:
        """Get a metadata entry."""
        # Check if table exists before querying
        if not self._check_table_exists():
            return None

        # Strip surrounding quotes if present; callers pass resolved() which handles casing.
        normalized_name = object_name.strip('"')

        result = self._session.sql(
            f"""
            SELECT METADATA
            FROM {self._table_path}
            WHERE OBJECT_TYPE = '{object_type.value}'
            AND OBJECT_NAME = '{normalized_name}'
            AND VERSION = '{version}'
            AND METADATA_TYPE = '{metadata_type.value}'
            """
        ).collect(statement_params=self._telemetry_stmp)

        if not result:
            return None

        metadata_value = result[0]["METADATA"]
        # Handle both string and dict responses from Snowflake
        if isinstance(metadata_value, str):
            result_dict: dict[str, Any] = json.loads(metadata_value)
            return result_dict
        return dict(metadata_value)
