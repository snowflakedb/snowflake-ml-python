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
from dataclasses import dataclass
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


class MetadataType(str, Enum):
    """Types of metadata that can be stored."""

    FEATURE_SPECS = "FEATURE_SPECS"
    FEATURE_DESCS = "FEATURE_DESCS"
    STREAM_SOURCE_CONFIG = "STREAM_SOURCE_CONFIG"


@dataclass
class AggregationMetadata:
    """Aggregation configuration for tiled feature views."""

    feature_granularity: str
    features: list[AggregationSpec]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_granularity": self.feature_granularity,
            "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregationMetadata:
        """Create from dictionary."""
        return cls(
            feature_granularity=data["feature_granularity"],
            features=[AggregationSpec.from_dict(f) for f in data["features"]],
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
    ) -> None:
        """Save all metadata for a tiled feature view atomically.

        This method saves both feature specs and descriptions in a single
        INSERT statement for atomicity during creation.

        Args:
            fv_name: Feature view name.
            version: Feature view version.
            specs: Aggregation metadata (required).
            descs: Optional dictionary of feature descriptions.
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
