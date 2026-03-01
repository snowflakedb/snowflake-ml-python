"""FeatureViewSpecBuilder — constructs validated JSON payloads for OFT creation specification.

This is the single entry point for internal SDK code to build feature view specs.

Usage::

    from snowflake.ml.feature_store.spec.builder import FeatureViewSpecBuilder
    from snowflake.ml.feature_store.spec.enums import *

    spec_obj = (
        FeatureViewSpecBuilder(
            FeatureViewKind.StreamingFeatureView,
            database="DB", schema="SCH", name="my_fv", version="v1",
        )
        .set_sources([stream_source])
        .set_udf(name=..., engine=..., output_columns=[...], function_definition=...)
        .set_features([agg_spec_1, agg_spec_2])
        .set_properties(entity_columns=[...], ...)
        .set_offline_configs([...])
        .build()
    )
    spec_dict = spec_obj.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from snowflake.ml.feature_store.aggregation import (
    AggregationSpec,
    AggregationType,
    interval_to_seconds,
)
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.ml.feature_store.spec.enums import (
    FeatureAggregationMethod,
    FeatureViewKind,
    SourceType,
    StoreType,
    TableType,
)
from snowflake.ml.feature_store.spec.models import (
    UDF,
    Feature,
    FeatureViewSpec,
    FSColumn,
    Metadata,
    OfflineTableConfig,
    Source,
    Spec,
    _columns_from_struct_type,
    _make_fs_column,
)
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.ml.version import VERSION
from snowflake.snowpark.types import DataType, StructType

if TYPE_CHECKING:
    from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewSlice


@dataclass(frozen=True)
class SnowflakeTableInfo:
    """Describes a Snowflake table used for offline storage in a feature view.

    Passed to :meth:`FeatureViewSpecBuilder.set_offline_configs` to describe
    each offline table.  The builder converts the Snowpark ``StructType`` to
    internal column representations automatically.

    Attributes:
        table_type: Kind of offline table
            (``UDF_TRANSFORMED``, ``TILED``, or ``BATCH_SOURCE``).
        database: Database name.
        schema: Schema name.
        table: Physical table name.
        columns: Table schema as a Snowpark ``StructType``.
    """

    table_type: TableType
    database: str
    schema: str
    table: str
    columns: StructType


@dataclass(frozen=True)
class BatchSource:
    """Schema of the source table/query for a batch feature view.

    Wraps the StructType from the feature_df used to define the batch FV.
    Converted to an internal ``Source`` with ``source_type=BATCH`` during
    ``set_sources()``.  The builder uses the columns on the BATCH source
    to resolve feature column types during ``build()``.

    Attributes:
        schema: Snowpark StructType from the feature_df.
    """

    schema: StructType


# Aggregation functions whose output type is predetermined regardless of source type.
# Functions not listed here (SUM, MIN, MAX, LAST_N, etc.) preserve the source type.
_AGG_PREDETERMINED_OUTPUT: dict[AggregationType, FSColumn] = {
    # COUNT / APPROX_COUNT_DISTINCT always produce an integer.
    AggregationType.COUNT: FSColumn(name="", type="DecimalType", precision=18, scale=0),
    AggregationType.APPROX_COUNT_DISTINCT: FSColumn(name="", type="DecimalType", precision=18, scale=0),
    # AVG / STD / VAR / APPROX_PERCENTILE always produce a float.
    AggregationType.AVG: FSColumn(name="", type="FloatType"),
    AggregationType.STD: FSColumn(name="", type="FloatType"),
    AggregationType.VAR: FSColumn(name="", type="FloatType"),
    AggregationType.APPROX_PERCENTILE: FSColumn(name="", type="FloatType"),
}

# Type alias for the polymorphic source input
SourceInput = Union[StreamSource, RequestSource, "FeatureView", "FeatureViewSlice", BatchSource]


class FeatureViewSpecBuilder:
    """Builds a validated FeatureView spec payload for the Go backend.

    All methods use the ``set_*`` convention. Accepts raw inputs — Snowpark types,
    StreamSource objects, user-facing Feature objects. Constructs spec-internal
    Pydantic models internally. Validates cross-model rules at ``build()`` time.
    """

    # Schema constants — bumped when the spec format changes
    _SPEC_FORMAT_VERSION = "1"
    _INTERNAL_DATA_VERSION = "1"

    def __init__(
        self,
        kind: FeatureViewKind,
        *,
        database: str,
        schema: str,
        name: str,
        version: str,
    ) -> None:
        """Initialize the builder with feature view identity and kind.

        Args:
            kind: The kind of feature view.
            database: Database name.
            schema: Schema name.
            name: Feature view name.
            version: Feature view version.
        """
        self._kind = kind
        self._database = database
        self._schema = schema
        self._name = name
        self._version = version

        # State — set by builder methods, consumed at build()
        self._offline_configs: list[OfflineTableConfig] = []
        self._sources: list[Source] = []
        self._agg_specs: list[AggregationSpec] = []
        self._udf: Optional[UDF] = None

        # Properties
        self._entity_columns: list[str] = []
        self._timestamp_field: Optional[str] = None
        self._granularity_sec: Optional[int] = None
        self._agg_method: Optional[FeatureAggregationMethod] = None
        self._target_lag_sec: Optional[int] = None

    # -----------------------------------------------------------------------
    # set_* methods
    # -----------------------------------------------------------------------

    def set_offline_configs(
        self,
        configs: list[SnowflakeTableInfo],
    ) -> FeatureViewSpecBuilder:
        """Set offline table configurations.

        Args:
            configs: List of :class:`SnowflakeTableInfo` describing each
                offline table.  The Snowpark ``StructType`` on each entry is
                converted to internal ``FSColumn`` representations automatically.

        Returns:
            self for method chaining.
        """
        self._offline_configs = []
        for cfg in configs:
            self._offline_configs.append(
                OfflineTableConfig(
                    store_type=StoreType.SNOWFLAKE,
                    table_type=cfg.table_type,
                    database=cfg.database,
                    schema_=cfg.schema,
                    table=cfg.table,
                    columns=_columns_from_struct_type(cfg.columns),
                )
            )
        return self

    def set_properties(
        self,
        *,
        entity_columns: list[str],
        timestamp_field: Optional[str] = None,
        granularity: Optional[str] = None,
        agg_method: Optional[FeatureAggregationMethod] = None,
        target_lag: Optional[str] = None,
    ) -> FeatureViewSpecBuilder:
        """Set core spec properties.

        Interval strings (e.g., ``'1h'``, ``'30s'``) are converted to integer
        seconds internally.

        Args:
            entity_columns: Ordered list of entity/join-key column names.
            timestamp_field: Optional timestamp column name.
            granularity: Optional tile interval (e.g., ``"1h"``). Converted to seconds.
            agg_method: Optional aggregation method.
            target_lag: Optional target lag (e.g., ``"30s"``). Converted to seconds.

        Returns:
            self for method chaining.
        """
        self._entity_columns = entity_columns
        self._timestamp_field = timestamp_field
        self._granularity_sec = interval_to_seconds(granularity) if granularity else None
        self._agg_method = agg_method
        self._target_lag_sec = interval_to_seconds(target_lag) if target_lag else None
        return self

    def set_sources(
        self,
        sources: list[SourceInput],
    ) -> FeatureViewSpecBuilder:
        """Set data sources.

        Args:
            sources: List of source inputs (``StreamSource``, ``RequestSource``,
                ``FeatureView``, ``FeatureViewSlice``, or ``BatchSource``).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If an unsupported source type is encountered.
        """
        # Lazy imports to avoid circular dependencies
        from snowflake.ml.feature_store.feature_view import (
            FeatureView,
            FeatureViewSlice,
        )

        self._sources = []

        for src in sources:
            if isinstance(src, BatchSource):
                self._sources.append(self._convert_batch_source(src))
            elif isinstance(src, StreamSource):
                self._sources.append(self._convert_stream_source(src))
            elif isinstance(src, RequestSource):
                self._sources.append(self._convert_request_source(src))
            elif isinstance(src, FeatureViewSlice):
                self._sources.append(self._convert_feature_view_slice(src))
            elif isinstance(src, FeatureView):
                self._sources.append(self._convert_feature_view(src))
            else:
                raise ValueError(f"Unsupported source type: {type(src).__name__}")
        return self

    def set_udf(
        self,
        *,
        name: str,
        engine: str,
        output_columns: list[tuple[str, DataType]],
        function_definition: str,
    ) -> FeatureViewSpecBuilder:
        """Set the UDF transform.

        The ``function_definition`` is stored as plain text. SQL ``$$`` quoting
        is handled during serialization.

        Args:
            name: UDF name.
            engine: Execution engine (e.g., ``"python"``).
            output_columns: List of (name, Snowpark DataType) pairs.
            function_definition: The UDF source code (plain text).

        Returns:
            self for method chaining.
        """
        fs_columns = [_make_fs_column(n, dt) for n, dt in output_columns]
        self._udf = UDF(
            name=name,
            engine=engine,
            output_columns=fs_columns,
            function_definition=function_definition,
        )
        return self

    def set_features(
        self,
        features: list[AggregationSpec],
    ) -> FeatureViewSpecBuilder:
        """Set features from AggregationSpec objects.

        Column types are resolved from sources/UDF outputs at ``build()`` time.

        Args:
            features: List of :class:`AggregationSpec` instances describing
                each feature's aggregation function, source column, window, etc.

        Returns:
            self for method chaining.
        """
        self._agg_specs = features
        return self

    # -----------------------------------------------------------------------
    # build
    # -----------------------------------------------------------------------

    def build(self) -> FeatureViewSpec:
        """Validate all rules and construct the typed spec model.

        Returns:
            FeatureViewSpec instance.  Call ``.to_dict()`` or
            ``.to_json()`` to serialize.

        Raises:
            ValueError: If validation fails.  # noqa: DAR402
        """
        # 1. Cross-model validation
        self._validate()

        # 2. Resolve user features → spec features
        spec_features = self._resolve_features()

        # 3. Auto-populate metadata
        metadata = Metadata(
            database=self._database,
            schema_=self._schema,
            name=self._name,
            version=self._version,
            spec_format_version=self._SPEC_FORMAT_VERSION,
            internal_data_version=self._INTERNAL_DATA_VERSION,
            client_version=VERSION,
        )

        # 4. Construct spec — filter out BATCH sources (builder-internal only)
        external_sources = [s for s in self._sources if s.source_type != SourceType.BATCH]
        spec = Spec(
            ordered_entity_column_names=self._entity_columns,
            sources=external_sources,
            features=spec_features,
            timestamp_field=self._timestamp_field,
            feature_granularity_sec=self._granularity_sec,
            feature_aggregation_method=self._agg_method,
            udf=self._udf,
            target_lag_sec=self._target_lag_sec,
        )

        # 5. Construct and return root
        return FeatureViewSpec(
            kind=self._kind,
            metadata=metadata,
            offline_configs=self._offline_configs,
            spec=spec,
            online_store_type=StoreType.POSTGRES,
        )

    # -----------------------------------------------------------------------
    # Source conversion helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _convert_batch_source(src: BatchSource) -> Source:
        """Convert a BatchSource to a spec Source with source_type=BATCH."""
        return Source(
            name="batch",
            source_type=SourceType.BATCH,
            columns=_columns_from_struct_type(src.schema),
        )

    @staticmethod
    def _convert_stream_source(src: StreamSource) -> Source:
        """Convert a StreamSource to a spec Source."""
        return Source(
            name=src.name.resolved(),
            source_type=SourceType.STREAM,
            columns=_columns_from_struct_type(src.schema),
        )

    @staticmethod
    def _convert_request_source(src: RequestSource) -> Source:
        """Convert a RequestSource to a spec Source."""
        return Source(
            name="request",
            source_type=SourceType.REQUEST,
            columns=_columns_from_struct_type(src.schema),
        )

    @staticmethod
    def _columns_from_feature_view(fv: FeatureView) -> list[FSColumn]:
        """Extract feature columns from a FeatureView's output schema."""
        feature_name_set = {fn.resolved() for fn in fv.feature_names}
        return [_make_fs_column(f.name, f.datatype) for f in fv.output_schema.fields if f.name in feature_name_set]

    @staticmethod
    def _convert_feature_view(fv: FeatureView) -> Source:
        """Convert a FeatureView to a spec Source with source_type=FEATURES."""
        return Source(
            name=fv.name.resolved(),
            source_type=SourceType.FEATURES,
            columns=FeatureViewSpecBuilder._columns_from_feature_view(fv),
            source_version=str(fv.version) if fv.version else None,
        )

    @staticmethod
    def _convert_feature_view_slice(fvs: FeatureViewSlice) -> Source:
        """Convert a FeatureViewSlice to a spec Source with selected_features."""
        fv = fvs.feature_view_ref
        return Source(
            name=fv.name.resolved(),
            source_type=SourceType.FEATURES,
            columns=FeatureViewSpecBuilder._columns_from_feature_view(fv),
            source_version=str(fv.version) if fv.version else None,
            selected_features=[n.resolved() for n in fvs.names],
        )

    # -----------------------------------------------------------------------
    # Feature resolution
    # -----------------------------------------------------------------------

    def _find_offline_config(self, table_type: TableType) -> Optional[OfflineTableConfig]:
        """Return the first offline config matching *table_type*, or ``None``."""
        for cfg in self._offline_configs:
            if cfg.table_type == table_type:
                return cfg
        return None

    def _find_batch_source(self) -> Optional[Source]:
        """Return the first BATCH source, or ``None``."""
        for src in self._sources:
            if src.source_type == SourceType.BATCH:
                return src
        return None

    def _get_feature_input_columns(self) -> list[FSColumn]:
        """Return the pipeline-stage columns that features aggregate over.

        Which columns serve as input depends on the FV kind:
          - **Streaming**: UDF_TRANSFORMED offline config columns.
          - **Batch tiled**: BATCH source columns (from ``BatchSource``
            passed via ``set_sources``).
          - **Batch non-tiled**: BATCH_SOURCE offline config columns.
          - **Realtime**: not supported (no aggregation features).

        Returns:
            List of FSColumn instances from the appropriate pipeline stage.

        Raises:
            ValueError: If the required column pool is not available.
        """
        if self._kind == FeatureViewKind.StreamingFeatureView:
            config = self._find_offline_config(TableType.UDF_TRANSFORMED)
            if config is None:
                raise ValueError("Streaming FV with features requires a " "UDF_TRANSFORMED offline config")
            return config.columns

        if self._kind == FeatureViewKind.BatchFeatureView:
            if self._agg_method is not None:
                # Tiled batch: features aggregate over the FV source table
                batch_src = self._find_batch_source()
                if batch_src is None or not batch_src.columns:
                    raise ValueError(
                        "Batch tiled FV requires the FV source schema " "(pass a BatchSource in set_sources)"
                    )
                return batch_src.columns
            # Non-tiled batch: input = output = BATCH_SOURCE
            config = self._find_offline_config(TableType.BATCH_SOURCE)
            if config is None:
                raise ValueError("Batch FV with features requires a " "BATCH_SOURCE offline config")
            return config.columns

        raise ValueError(f"{self._kind.value} does not support aggregation features")

    def _resolve_features(self) -> list[Feature]:
        """Convert AggregationSpecs to spec-internal Feature models.

        Column types are resolved from a pipeline-specific column pool
        determined by :meth:`_get_feature_input_columns`.

        Returns:
            List of spec-internal Feature models.

        Raises:
            ValueError: If a feature references a column not in the resolution pool.
        """
        if not self._agg_specs:
            return []

        resolve_columns = self._get_feature_input_columns()
        column_map: dict[str, FSColumn] = {col.name: col for col in resolve_columns}

        spec_features = []
        for agg_spec in self._agg_specs:
            source_col = column_map.get(agg_spec.source_column)
            if source_col is None:
                raise ValueError(
                    f"Column '{agg_spec.source_column}' not found in "
                    f"resolution pool. Available columns: "
                    f"{list(column_map.keys())}"
                )

            # Determine output column type based on aggregation function.
            predetermined = _AGG_PREDETERMINED_OUTPUT.get(agg_spec.function)
            if predetermined is not None:
                output_col = FSColumn(
                    name=agg_spec.output_column,
                    type=predetermined.type,
                    precision=predetermined.precision,
                    scale=predetermined.scale,
                )
            else:
                # SUM, MIN, MAX, LAST_N, etc. preserve source column type
                # including precision/scale/length/timezone.
                output_col = FSColumn(
                    name=agg_spec.output_column,
                    type=source_col.type,
                    precision=source_col.precision,
                    scale=source_col.scale,
                    length=source_col.length,
                    timezone=source_col.timezone,
                )

            # interval_to_seconds() returns -1 as a sentinel for lifetime
            # windows.  Convert non-positive values to None so they are
            # omitted (omitempty).
            raw_window = agg_spec.get_window_seconds()
            window_sec = raw_window if raw_window > 0 else None

            raw_offset = interval_to_seconds(agg_spec.offset) if agg_spec.offset != "0" else 0
            offset_sec = raw_offset if raw_offset > 0 else None

            spec_features.append(
                Feature(
                    source_column=source_col,
                    output_column=output_col,
                    function=agg_spec.function.value,
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    function_params=(agg_spec.params if agg_spec.params else None),
                )
            )
        return spec_features

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate(self) -> None:
        """Cross-model validation. Raises ValueError on rule violations."""
        kind = self._kind
        table_types = [c.table_type for c in self._offline_configs]
        source_types = [s.source_type for s in self._sources]

        if kind == FeatureViewKind.StreamingFeatureView:
            self._validate_streaming(table_types, source_types)
        elif kind == FeatureViewKind.BatchFeatureView:
            self._validate_batch(table_types, source_types)
        elif kind == FeatureViewKind.RealtimeFeatureView:
            self._validate_realtime(table_types, source_types)

        # Source field validation (applies to all kinds)
        self._validate_source_fields()

    def _validate_streaming(self, table_types: list[TableType], source_types: list[SourceType]) -> None:
        """Validate rules specific to StreamingFeatureView."""
        # UDFTransformed always required
        if TableType.UDF_TRANSFORMED not in table_types:
            raise ValueError("StreamingFeatureView requires a UDFTransformed offline config")

        if self._agg_method is not None:
            # tiles or continuous → 2 configs: UDFTransformed + Tiled
            if len(self._offline_configs) != 2:
                raise ValueError(
                    f"StreamingFeatureView with agg_method={self._agg_method.value} "
                    f"requires 2 offline configs (UDFTransformed + Tiled), "
                    f"got {len(self._offline_configs)}"
                )
            if TableType.TILED not in table_types:
                raise ValueError(
                    f"StreamingFeatureView with agg_method={self._agg_method.value} " f"requires a Tiled offline config"
                )
            if self._granularity_sec is None:
                raise ValueError(f"granularity_sec is required when agg_method is " f"{self._agg_method.value}")
        else:
            # No aggregation → 1 config: UDFTransformed only
            if len(self._offline_configs) != 1:
                raise ValueError(
                    "StreamingFeatureView with no agg_method requires 1 offline "
                    f"config (UDFTransformed), got {len(self._offline_configs)}"
                )
            if self._granularity_sec is not None:
                raise ValueError("granularity_sec must be None when agg_method is None")

        # Exactly 1 Stream source
        if len(self._sources) != 1 or source_types != [SourceType.STREAM]:
            raise ValueError("StreamingFeatureView requires exactly 1 Stream source")

        # UDF required
        if self._udf is None:
            raise ValueError("StreamingFeatureView requires a UDF")

        # Timestamp required
        if self._timestamp_field is None:
            raise ValueError("StreamingFeatureView requires a timestamp_field")

    def _validate_batch(self, table_types: list[TableType], source_types: list[SourceType]) -> None:
        """Validate rules specific to BatchFeatureView."""
        # Exactly 1 BatchSource offline config
        if len(self._offline_configs) != 1 or table_types != [TableType.BATCH_SOURCE]:
            raise ValueError("BatchFeatureView requires exactly 1 BatchSource offline config")

        # No UDF
        if self._udf is not None:
            raise ValueError("BatchFeatureView must not have a UDF")

        # agg_method: tiles or None
        if self._agg_method is not None and self._agg_method != FeatureAggregationMethod.TILES:
            raise ValueError(f"BatchFeatureView agg_method must be 'tiles' or None, " f"got {self._agg_method.value}")

        # Tiled vs non-tiled: different granularity + source rules
        if self._agg_method == FeatureAggregationMethod.TILES:
            if self._granularity_sec is None:
                raise ValueError("granularity_sec is required when agg_method is tiles")
            # Tiled: requires exactly 1 Batch source (raw events schema)
            if len(self._sources) != 1 or source_types != [SourceType.BATCH]:
                raise ValueError("Tiled BatchFeatureView requires exactly 1 Batch source")
        else:
            if self._granularity_sec is not None:
                raise ValueError("granularity_sec must be None when agg_method is None")
            # Non-tiled: no sources needed (features == offline table columns)
            if self._sources:
                raise ValueError("Non-tiled BatchFeatureView must not have sources")

    def _validate_realtime(self, table_types: list[TableType], source_types: list[SourceType]) -> None:
        """Validate rules specific to RealtimeFeatureView."""
        # No offline configs
        if len(self._offline_configs) != 0:
            raise ValueError("RealtimeFeatureView must not have offline configs")

        # Exactly 1 Request source, 0+ Features sources, no Stream/Batch
        request_count = source_types.count(SourceType.REQUEST)
        if request_count != 1:
            raise ValueError(f"RealtimeFeatureView requires exactly 1 Request source, " f"got {request_count}")
        # No Stream sources allowed
        if SourceType.STREAM in source_types:
            raise ValueError("RealtimeFeatureView must not have Stream sources")
        # No Batch sources allowed
        if SourceType.BATCH in source_types:
            raise ValueError("RealtimeFeatureView must not have Batch sources")

        # UDF required
        if self._udf is None:
            raise ValueError("RealtimeFeatureView requires a UDF")

        # No aggregation
        if self._agg_method is not None:
            raise ValueError("RealtimeFeatureView must not have an agg_method")
        if self._granularity_sec is not None:
            raise ValueError("RealtimeFeatureView must not have granularity_sec")

    def _validate_source_fields(self) -> None:
        """Validate that each source has the correct fields for its type."""
        for source in self._sources:
            if source.source_type == SourceType.STREAM:
                if not source.columns:
                    raise ValueError(f"Stream source '{source.name}' must have columns")
                if source.source_version is not None:
                    raise ValueError(f"Stream source '{source.name}' must not have " f"source_version")
                if source.selected_features is not None:
                    raise ValueError(f"Stream source '{source.name}' must not have " f"selected_features")

            elif source.source_type == SourceType.REQUEST:
                if not source.columns:
                    raise ValueError(f"Request source '{source.name}' must have columns")
                if source.source_version is not None:
                    raise ValueError(f"Request source '{source.name}' must not have " f"source_version")
                if source.selected_features is not None:
                    raise ValueError(f"Request source '{source.name}' must not have " f"selected_features")

            elif source.source_type == SourceType.FEATURES:
                if not source.columns:
                    raise ValueError(f"Features source '{source.name}' must have columns")
                if source.source_version is None:
                    raise ValueError(f"Features source '{source.name}' requires " f"source_version")

            elif source.source_type == SourceType.BATCH:
                if not source.columns:
                    raise ValueError(f"Batch source '{source.name}' must have columns")
                if source.source_version is not None:
                    raise ValueError(f"Batch source '{source.name}' must not have " f"source_version")
                if source.selected_features is not None:
                    raise ValueError(f"Batch source '{source.name}' must not have " f"selected_features")
