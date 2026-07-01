from __future__ import annotations

import datetime
import functools
import json
import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import packaging.version as pkg_version
from typing_extensions import Concatenate, ParamSpec

import snowflake.ml.feature_store.feature_view as fv_mod
import snowflake.ml.version as snowml_version
from snowflake.ml import dataset
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    dataset_errors,
    error_codes,
    exceptions as snowml_exceptions,
    sql_error_codes,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.sql_identifier import (
    SqlIdentifier,
    get_fully_qualified_name,
    parse_fully_qualified_name,
    to_sql_identifiers,
)
from snowflake.ml.dataset.dataset_metadata import FeatureStoreMetadata
from snowflake.ml.feature_store import (
    feature_group as fg_mod,
    feature_view_append_only_validation,
    feature_view_refresh_freq,
    online_service,
    online_service_http_client,
    realtime_dataset as rtfv_dataset,
)
from snowflake.ml.feature_store.entity import _ENTITY_NAME_LENGTH_LIMIT, Entity
from snowflake.ml.feature_store.feature_group import FeatureGroup
from snowflake.ml.feature_store.feature_view import (
    _FEATURE_OBJ_TYPE,
    _FEATURE_VIEW_NAME_DELIMITER,
    _LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS,
    _ONLINE_TABLE_SUFFIX,
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineStoreType,
    StorageConfig,
    StorageFormat,
    _FeatureViewMetadata,
    _FeatureViewSchemaNotReadyWarning,
)
from snowflake.ml.feature_store.metadata_manager import (
    AggregationMetadata,
    AppendOnlyMetadata,
    FeatureStoreMetadataManager,
    FeatureViewMetadataConfig,
    FvSourceRefsMetadata,
    MetadataObjectType,
    MetadataType,
    RealtimeConfigMetadata,
)
from snowflake.ml.feature_store.spec import (
    table_schema_evolution as fs_table_schema_evolution,
)
from snowflake.ml.feature_store.spec.builder import (
    BatchSource,
    FeatureViewSpecBuilder,
    SnowflakeTableInfo,
)
from snowflake.ml.feature_store.spec.enums import (
    ENTITY_TAG_PREFIX,
    FeatureAggregationMethod,
    FeatureViewKind,
    TableType,
)
from snowflake.ml.feature_store.spec.models import (
    FeatureViewSpec,
    validate_spec_oft_offline_table_schema,
    validate_spec_oft_tiled_offline_table_schema,
)
from snowflake.ml.feature_store.stream_source import (
    _LIST_STREAM_SOURCE_SCHEMA,
    StreamSource,
)
from snowflake.ml.feature_store.tile_sql_generator import (
    _TILE_START_COL,
    MergingSqlGenerator,
    RollupSqlGenerator,
)
from snowflake.ml.utils import sql_client
from snowflake.snowpark import DataFrame, Row, Session, functions as F
from snowflake.snowpark._internal import utils as snowpark_utils
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.types import (
    ArrayType,
    BooleanType,
    DataType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

if TYPE_CHECKING:
    import pandas as pd

_Args = ParamSpec("_Args")
_RT = TypeVar("_RT")

logger = logging.getLogger(__name__)


def _stream_source_schema_field_names(schema: StructType) -> list[str]:
    return [f.name for f in schema.fields]


def _normalize_stream_ingest_records(
    records: Union[list[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, dict):
        return [records]
    if isinstance(records, list):
        if not records:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("records must be non-empty for stream_ingest."),
            )
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(records):
            if not isinstance(row, dict):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=TypeError(
                        f"records must be a list of dicts; element {i} has type {type(row)!r}."
                    ),
                )
            rows.append(row)
        return rows
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INVALID_ARGUMENT,
        original_exception=TypeError(f"records must be a dict or non-empty list[dict]; got {type(records)!r}."),
    )


def _validate_stream_ingest_record_keys(expected: list[str], row: dict[str, Any], row_index: int) -> None:
    exp_set = set(expected)
    keys = set(row.keys())
    if keys != exp_set:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Record {row_index} column keys must match StreamSource schema exactly: "
                f"expected={expected!r}, missing={sorted(exp_set - keys)!r}, extra={sorted(keys - exp_set)!r}."
            ),
        )


def _store_type_from_oft_show_row(row: Row) -> OnlineStoreType:
    """Map SHOW ONLINE FEATURE TABLES ``store_type`` to ``OnlineStoreType``.

    Args:
        row: A Snowpark ``Row`` from SHOW ONLINE FEATURE TABLES.

    Returns:
        OnlineStoreType: ``HYBRID_TABLE`` if missing or unrecognized.

    When the column is absent (older Snowflake builds), defaults to ``HYBRID_TABLE``
    to match ``OnlineConfig`` / ``from_json`` behavior without ``store_type``.
    """
    raw: Any = None
    if "store_type" in row:
        raw = row["store_type"]
    elif "STORE_TYPE" in row:
        raw = row["STORE_TYPE"]
    else:
        return OnlineStoreType.HYBRID_TABLE

    if raw is None:
        return OnlineStoreType.HYBRID_TABLE
    s = str(raw).strip()
    if not s:
        return OnlineStoreType.HYBRID_TABLE

    key = s.lower()
    if key == OnlineStoreType.POSTGRES.value:
        return OnlineStoreType.POSTGRES
    if key == OnlineStoreType.HYBRID_TABLE.value:
        return OnlineStoreType.HYBRID_TABLE

    logger.warning(
        "Unknown SHOW ONLINE FEATURE TABLES store_type %r; defaulting to hybrid_table",
        raw,
    )
    return OnlineStoreType.HYBRID_TABLE


# Module-local alias kept so existing callers continue to reference
# ``_ENTITY_TAG_PREFIX``; the canonical value lives in
# :data:`snowflake.ml.feature_store.spec.enums.ENTITY_TAG_PREFIX`.
_ENTITY_TAG_PREFIX = ENTITY_TAG_PREFIX
_FEATURE_STORE_OBJECT_TAG = "SNOWML_FEATURE_STORE_OBJECT"
_FEATURE_VIEW_METADATA_TAG = "SNOWML_FEATURE_VIEW_METADATA"
_SNAPSHOT_STATUS_TABLE = "SNOWML_SNAPSHOT_STATUS"
# Sentinel value to distinguish "caller didn't pass refresh_freq" from "caller explicitly
# passed refresh_freq=None".  This is needed because None is a valid value for refresh_freq
# (it means "remove the schedule"), so we cannot use None as the default to detect whether the
# caller provided the argument.  Using a unique object() instance lets update_feature_view
# skip refresh_freq handling entirely when the caller omits it, while still accepting None
# as an intentional value — preserving backward compatibility with callers that never set it.
_UNSET: Any = object()


def _sql_string_literal(value: str) -> str:
    """Quote a string for safe inclusion as a Snowflake DDL string literal.

    Doubles embedded single quotes (the Snowflake convention for escaping
    inside single-quoted literals) and wraps the result in single quotes so
    callers cannot forget the surrounding quotes. Use this everywhere a
    user-controlled string is interpolated into a single-quoted SQL literal
    such as ``COMMENT = ...`` or ``ALLOWED_VALUES ...`` — Snowflake DDL does
    not accept ``?`` bind parameters in those positions, so quote-doubling
    is the only safe option.

    Args:
        value: Raw string to escape.

    Returns:
        ``'<value with `'` doubled>'`` — already quoted, ready to drop into
        SQL with no surrounding quotes at the call site.
    """
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class _FeatureStoreObjInfo:
    type: _FeatureStoreObjTypes
    pkg_version: str

    def to_json(self) -> str:
        state_dict = self.__dict__.copy()
        state_dict["type"] = state_dict["type"].value
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, json_str: str) -> _FeatureStoreObjInfo:
        json_dict = json.loads(json_str)
        # since we may introduce new fields in the json blob in the future,
        # in order to guarantee compatibility, we need to select ones that can be
        # decoded in the current version
        state_dict = {}
        state_dict["type"] = _FeatureStoreObjTypes.parse(json_dict["type"])
        state_dict["pkg_version"] = json_dict["pkg_version"]
        return cls(**state_dict)  # type: ignore[arg-type]


class _FeatureStoreObjTypes(Enum):
    UNKNOWN = "UNKNOWN"  # for forward compatibility
    MANAGED_FEATURE_VIEW = "MANAGED_FEATURE_VIEW"  # Snowflake manages the refresh for the user
    EXTERNAL_FEATURE_VIEW = "EXTERNAL_FEATURE_VIEW"
    FEATURE_VIEW_REFRESH_TASK = "FEATURE_VIEW_REFRESH_TASK"
    FEATURE_VIEW_BACKFILL_PROC = "FEATURE_VIEW_BACKFILL_PROC"
    FEATURE_VIEW_BACKFILL_UDTF = "FEATURE_VIEW_BACKFILL_UDTF"
    TRAINING_DATA = "TRAINING_DATA"
    ONLINE_FEATURE_TABLE = "ONLINE_FEATURE_TABLE"
    UDF_TRANSFORMED_TABLE = "UDF_TRANSFORMED_TABLE"
    SNAPSHOT_TABLE = "SNAPSHOT_TABLE"
    INTERNAL_METADATA_TABLE = "INTERNAL_METADATA_TABLE"
    FEATURE_GROUP = "FEATURE_GROUP"
    # Online feature table backing a RealtimeFeatureView (OFT-only, no DT/View).
    REALTIME_FEATURE_VIEW = "REALTIME_FEATURE_VIEW"

    @classmethod
    def parse(cls, val: str) -> _FeatureStoreObjTypes:
        try:
            return cls(val)
        except ValueError:
            return cls.UNKNOWN


class _MaterializedResourceKind(Enum):
    """High-level kinds of resources materialized by feature_view registration / update.

    Distinct from `_FeatureStoreObjTypes` (which labels persisted Snowflake object tags):
    these are the in-memory categories used by the materialization flow to label
    compensating-action stacks and to make rollback ordering explicit.
    """

    OFFLINE_FEATURE_VIEW = "OFFLINE_FEATURE_VIEW"  # Dynamic Table or VIEW
    ONLINE_FEATURE_TABLE = "ONLINE_FEATURE_TABLE"
    REFRESH_TASK = "REFRESH_TASK"
    SNAPSHOT_SCHEMA_EVOLUTION = "SNAPSHOT_SCHEMA_EVOLUTION"


_PROJECT = "FeatureStore"
_DT_OR_VIEW_QUERY_PATTERN = re.compile(
    r"""CREATE\ (OR\ REPLACE\ )?(?P<obj_type>(DYNAMIC\ ICEBERG\ TABLE|DYNAMIC\ TABLE|VIEW))\ .*
        COMMENT\ =\ '(?P<comment>.*)'\s*
        TAG.*?{fv_metadata_tag}\ =\ '(?P<fv_metadata>.*?)',?.*?
        AS\ (?P<query>.*)
    """.format(
        fv_metadata_tag=_FEATURE_VIEW_METADATA_TAG,
    ),
    flags=re.DOTALL | re.IGNORECASE | re.X,
)

_DT_INITIALIZE_PATTERN = re.compile(
    r"""CREATE\ DYNAMIC\ TABLE\ .*
        initialize\ =\ '(?P<initialize>.*)'\ .*?
        AS\ .*
    """,
    flags=re.DOTALL | re.IGNORECASE | re.X,
)

# Sentinel for update_feature_view: distinguishes "leave unchanged" (default)
# from an explicit None, which clears (UNSETs) the initialization warehouse.
_KEEP_CURRENT: Any = object()


def _initialization_warehouse_clause(feature_view: FeatureView) -> str:
    """Build the trailing ``INITIALIZATION_WAREHOUSE = ...`` fragment for CREATE DYNAMIC TABLE.

    Returns an empty string when the feature view has no initialization
    warehouse, leaving the DDL identical to the single-warehouse behavior.

    Args:
        feature_view: The feature view being materialized.

    Returns:
        ``\\n  INITIALIZATION_WAREHOUSE=<wh>`` or an empty string.
    """
    iw = feature_view.initialization_warehouse
    return f"\n                INITIALIZATION_WAREHOUSE = {iw}" if iw is not None else ""


# ``list_feature_views`` output schemas. Row builders always populate the verbose
# schema; non-verbose callers receive _LIST_FEATURE_VIEW_SCHEMA after trimming
# verbose-only trailing fields.
# ``kind`` is the per-row discriminator: BATCH / STREAMING / REALTIME.
_LIST_FEATURE_VIEW_BASE_FIELDS = [
    StructField("name", StringType()),
    StructField("version", StringType()),
    StructField("database_name", StringType()),
    StructField("schema_name", StringType()),
    StructField("created_on", TimestampType()),
    StructField("owner", StringType()),
    StructField("desc", StringType()),
    StructField("entities", ArrayType(StringType())),
    StructField("refresh_freq", StringType()),
    StructField("refresh_mode", StringType()),
    StructField("scheduling_state", StringType()),
    StructField("warehouse", StringType()),
    StructField("cluster_by", StringType()),
    StructField("online_config", StringType()),
    StructField("storage_config", StringType()),
    StructField("stream_config", StringType()),
    StructField("kind", StringType()),
    StructField("append_only", BooleanType()),
]

_LIST_FEATURE_VIEW_VERBOSE_EXTRA_FIELDS = [
    # Initialization warehouse used for the initial build / reinitialization of a
    # managed FV's dynamic table. Verbose-only: ``None`` for FVs without one.
    StructField("initialization_warehouse", StringType()),
    # JSON-encoded authored source-ref list, or ``None`` when no
    # ``FV_SOURCE_REFS`` metadata row was written for this FV.
    StructField("source_refs", StringType()),
    StructField("backup_source", StringType()),
]

_LIST_FEATURE_VIEW_SCHEMA = StructType(_LIST_FEATURE_VIEW_BASE_FIELDS)
_LIST_FEATURE_VIEW_SCHEMA_VERBOSE = StructType(_LIST_FEATURE_VIEW_BASE_FIELDS + _LIST_FEATURE_VIEW_VERBOSE_EXTRA_FIELDS)


def _create_list_feature_views_dataframe(
    session: Session,
    output_values: list[list[Any]],
    output_values_extra: list[list[Any]],
    *,
    verbose: bool,
) -> DataFrame:
    """Build the ``list_feature_views`` result DataFrame.

    ``output_values`` contains base-schema fields matching
    ``_LIST_FEATURE_VIEW_SCHEMA``. ``output_values_extra`` contains the
    verbose-only fields matching ``_LIST_FEATURE_VIEW_VERBOSE_EXTRA_FIELDS``.
    When ``verbose=True`` the two are merged per-row to produce the verbose
    schema; otherwise only ``output_values`` is used.

    Args:
        session: Snowpark session used to materialize the listing DataFrame.
        output_values: Base-field rows populated by ``_extract_feature_view_info``
            and RTFV listing helpers.
        output_values_extra: Verbose-only field rows, one per row in
            ``output_values``, in the same order.
        verbose: When True, include verbose-only columns such as ``backup_source``.

    Returns:
        Snowpark DataFrame whose schema matches ``verbose``.
    """
    if verbose:
        merged = [base + extra for base, extra in zip(output_values, output_values_extra)]
        return session.create_dataframe(merged, schema=_LIST_FEATURE_VIEW_SCHEMA_VERBOSE)
    return session.create_dataframe(output_values, schema=_LIST_FEATURE_VIEW_SCHEMA)


# Per-row kind discriminator emitted by ``list_feature_views``.
_FV_KIND_BATCH = "BATCH"
_FV_KIND_STREAMING = "STREAMING"
_FV_KIND_REALTIME = "REALTIME"


# Default storage config JSON strings for list_feature_views output
_DEFAULT_STORAGE_CONFIG_JSON = StorageConfig().to_json()  # {"format": "snowflake"}
_DEFAULT_ICEBERG_STORAGE_CONFIG_JSON = StorageConfig(format=StorageFormat.ICEBERG).to_json()  # {"format": "iceberg"}

CreationMode = sql_client.CreationOption
CreationMode.__module__ = __name__


@dataclass(frozen=True)
class _FeatureStoreConfig:
    database: SqlIdentifier
    schema: SqlIdentifier

    @property
    def full_schema_path(self) -> str:
        return f"{self.database}.{self.schema}"


def switch_warehouse(
    f: Callable[Concatenate[FeatureStore, _Args], _RT],
) -> Callable[Concatenate[FeatureStore, _Args], _RT]:
    @functools.wraps(f)
    def wrapper(self: FeatureStore, /, *args: _Args.args, **kargs: _Args.kwargs) -> _RT:
        original_warehouse = self._session.get_current_warehouse()
        if original_warehouse is not None:
            original_warehouse = SqlIdentifier(original_warehouse)
        warehouse_updated = False
        try:
            if original_warehouse != self._default_warehouse:
                self._session.use_warehouse(self._default_warehouse)
                warehouse_updated = True
            return f(self, *args, **kargs)
        finally:
            if warehouse_updated and original_warehouse is not None:
                self._session.use_warehouse(original_warehouse)

    return wrapper


_CTE_MERGE_BATCH_SIZE = 10


class _FvJoinInfo(NamedTuple):
    cte_name: str
    key_sig: tuple[tuple[str, ...], bool]
    join_keys: list[SqlIdentifier]
    has_ts: bool
    select_cols: list[str]
    col_names: list[str]


def _make_fv_join_clause(
    lhs_alias: str,
    cte_name: str,
    join_keys: list[SqlIdentifier],
    has_ts: bool,
    spine_timestamp_col: Optional[SqlIdentifier],
) -> str:
    conditions = [f"{lhs_alias}.{k.identifier()} = {cte_name}.{k.identifier()}" for k in join_keys]
    if has_ts and spine_timestamp_col:
        ts_id = spine_timestamp_col.identifier()
        conditions.append(f"{lhs_alias}.{ts_id} = {cte_name}.{ts_id}")
    return f"\n    LEFT JOIN {cte_name}\n    ON {' AND '.join(conditions)}"


def dispatch_decorator(
    *,
    skip_wh_switch: Optional[Callable[..., bool]] = None,
    skip_telemetry: Optional[Callable[..., bool]] = None,
) -> Callable[[Callable[Concatenate[FeatureStore, _Args], _RT]], Callable[Concatenate[FeatureStore, _Args], _RT],]:
    def decorator(
        f: Callable[Concatenate[FeatureStore, _Args], _RT],
    ) -> Callable[Concatenate[FeatureStore, _Args], _RT]:
        wrapped_with_wh = switch_warehouse(f)
        if skip_wh_switch is not None:

            @functools.wraps(f)
            def maybe_switch_wh(self: FeatureStore, /, *args: _Args.args, **kargs: _Args.kwargs) -> _RT:
                if skip_wh_switch(self, *args, **kargs):
                    return f(self, *args, **kargs)
                return wrapped_with_wh(self, *args, **kargs)

            after_wh: Callable[Concatenate[FeatureStore, _Args], _RT] = maybe_switch_wh
        else:
            after_wh = wrapped_with_wh

        wrapped_with_telemetry = telemetry.send_api_usage_telemetry(project=_PROJECT)(after_wh)
        if skip_telemetry is not None:

            @functools.wraps(f)
            def maybe_telemetry(self: FeatureStore, /, *args: _Args.args, **kargs: _Args.kwargs) -> _RT:
                if skip_telemetry(self, *args, **kargs):
                    # Mirror ``send_api_usage_telemetry``'s exception unwrap so the
                    # hot/cold paths surface the same caller-visible error types.
                    try:
                        return after_wh(self, *args, **kargs)
                    except snowml_exceptions.SnowflakeMLException as e:
                        raise e.original_exception from e
                return wrapped_with_telemetry(self, *args, **kargs)

            return maybe_telemetry

        return wrapped_with_telemetry

    return decorator


def _read_feature_view_is_realtime(*args: Any, **kwargs: Any) -> bool:
    """True iff the call is an RTFV read.

    Drives the warehouse-switch and telemetry-skip predicates, so order matters:
    when a ``FeatureView`` argument is present we use its ``is_realtime_feature_view`` flag
    directly (a non-RTFV with ``request_context`` must not skip telemetry). The
    string-form ``read_feature_view("name", "v1", ...)`` falls back to the
    ``request_context``-presence heuristic, since there is no FV object to
    inspect without a metadata round-trip.

    Args:
        *args: Positional arguments to ``read_feature_view``; ``args[1]`` (if present)
            is the ``feature_view`` parameter.
        **kwargs: Keyword arguments to ``read_feature_view``; ``feature_view`` and
            ``request_context`` are consulted.

    Returns:
        True if this call should be dispatched to the RealtimeFeatureView read path.
    """
    feature_view = kwargs.get("feature_view") if "feature_view" in kwargs else (args[1] if len(args) > 1 else None)
    if isinstance(feature_view, FeatureView):
        return feature_view.is_realtime_feature_view
    return kwargs.get("request_context") is not None


def _predicate_read_feature_view_skip_wh_switch(*args: Any, **kwargs: Any) -> bool:
    """Skip warehouse switch when opted-in or targeting an HTTP-only path (Postgres online / RTFV)."""
    if bool(kwargs.get("use_session_warehouse", False)):
        return True
    # RTFVs ignore store_type and always use the Query API; skip regardless of the
    # store_type kwarg so the public OFFLINE default doesn't incur warehouse churn.
    if _read_feature_view_is_realtime(*args, **kwargs):
        return True
    if _get_store_type(kwargs.get("store_type", fv_mod.StoreType.OFFLINE)) != fv_mod.StoreType.ONLINE:
        return False
    # String (name, version) inputs would require a metadata round-trip to resolve online_config,
    # defeating the savings; fall through for them.
    feature_view = kwargs.get("feature_view") if "feature_view" in kwargs else (args[1] if len(args) > 1 else None)
    if not isinstance(feature_view, FeatureView):
        return False
    return feature_view.online_config is not None and feature_view.online_config.store_type == OnlineStoreType.POSTGRES


def _predicate_read_feature_view_skip_telemetry(*args: Any, **kwargs: Any) -> bool:
    # RTFV reads are HTTP-only and latency-sensitive; skip the SQL telemetry wrapper
    # the same way the ONLINE path does, regardless of the caller's store_type.
    if _read_feature_view_is_realtime(*args, **kwargs):
        return True
    return _get_store_type(kwargs.get("store_type", fv_mod.StoreType.OFFLINE)) == fv_mod.StoreType.ONLINE


# FG read is ONLINE-only today (OFFLINE is rejected upstream); both predicates
# always skip — reads use HTTP (no warehouse) and run on a latency-sensitive path.
# When OFFLINE FG reads land, gate on ``store_type`` like the FV predicates above.
def _predicate_read_feature_group_skip_wh_switch(*args: Any, **kwargs: Any) -> bool:
    return True


def _predicate_read_feature_group_skip_telemetry(*args: Any, **kwargs: Any) -> bool:
    return True


class FeatureStore:
    """
    FeatureStore provides APIs to create, materialize, retrieve and manage feature pipelines.
    """

    # Prefixes for temporary objects used during shadow swap operations
    _TMP_VIEW_PREFIX = "_TMP_VIEW_"
    _TMP_DT_PREFIX = "_TMP_TABLE_"

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def __init__(
        self,
        session: Session,
        database: str,
        name: str,
        default_warehouse: str,
        *,
        creation_mode: CreationMode = CreationMode.FAIL_IF_NOT_EXIST,
        default_iceberg_external_volume: Optional[str] = None,
        online_service_access: Optional[online_service.OnlineServiceAccess] = None,
    ) -> None:
        """
        Creates a FeatureStore instance.

        Args:
            session: Snowpark Session to interact with Snowflake backend.
            database: Database to create the FeatureStore instance.
            name: Target FeatureStore name, maps to a schema in the database.
            default_warehouse: Default warehouse for feature store compute.
            creation_mode: If FAIL_IF_NOT_EXIST, feature store throws when required resources not already exist; If
                CREATE_IF_NOT_EXIST, feature store will create required resources if they not already exist. Required
                resources include schema and tags. Note database must already exist in either mode.
            default_iceberg_external_volume: Default external volume for Iceberg-backed Feature Views. If set,
                Feature Views using StorageFormat.ICEBERG can omit external_volume in their StorageConfig.
            online_service_access: Override the URL the Online Service is reached on. Defaults to
                auto-routing: PrivateLink when the account advertises it, otherwise public; SPCS-internal
                when running inside SPCS.

        Raises:
            SnowflakeMLException: [ValueError] default_warehouse does not exist.
            SnowflakeMLException: [ValueError] Required resources not exist when mode is FAIL_IF_NOT_EXIST.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
            SnowflakeMLException: [RuntimeError] Failed to create feature store.

        Example::

            >>> from snowflake.ml.feature_store import (
            ...     FeatureStore,
            ...     CreationMode,
            ... )
            <BLANKLINE>
            >>> # Create a new Feature Store:
            >>> fs = FeatureStore(
            ...     session=session,
            ...     database="MYDB",
            ...     name="MYSCHEMA",
            ...     default_warehouse="MYWH",
            ...     creation_mode=CreationMode.CREATE_IF_NOT_EXIST
            ... )
            <BLANKLINE>
            >>> # Connect to an existing Feature Store:
            >>> fs = FeatureStore(
            ...     session=session,
            ...     database="MYDB",
            ...     name="MYSCHEMA",
            ...     default_warehouse="MYWH",
            ...     creation_mode=CreationMode.FAIL_IF_NOT_EXIST
            ... )

        """

        database = SqlIdentifier(database)
        name = SqlIdentifier(name)

        self._telemetry_stmp = telemetry.get_function_usage_statement_params(_PROJECT)
        self._session: Session = session
        if online_service_access is not None and not isinstance(
            online_service_access, online_service.OnlineServiceAccess
        ):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=TypeError(
                    "online_service_access must be an OnlineServiceAccess member or None, "
                    f"got {type(online_service_access).__name__}."
                ),
            )
        self._online_service_access = online_service_access
        self._config = _FeatureStoreConfig(
            database=database,
            schema=name,
        )
        self._metadata_manager = FeatureStoreMetadataManager(
            session=session,
            schema_path=self._config.full_schema_path,
            fs_object_tag_path=self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG),
            telemetry_stmp=self._telemetry_stmp,
        )
        self._asof_join_enabled = None

        # A dict from object name to tuple of search space and object domain.
        # search space used in query "SHOW <object_TYPE> LIKE <object_name> IN <search_space>"
        # object domain used in query "TAG_REFERENCE(<object_name>, <object_domain>)"
        self._obj_search_spaces = {
            "DATASETS": (self._config.full_schema_path, "DATASET"),
            "DYNAMIC TABLES": (self._config.full_schema_path, "TABLE"),
            "VIEWS": (self._config.full_schema_path, "TABLE"),
            "ONLINE FEATURE TABLES": (self._config.full_schema_path, "TABLE"),
            "SCHEMAS": (f"DATABASE {self._config.database}", "SCHEMA"),
            "TAGS": (self._config.full_schema_path, None),
            "TASKS": (self._config.full_schema_path, "TASK"),
            "WAREHOUSES": (None, None),
        }

        self.update_default_warehouse(default_warehouse)
        self._default_iceberg_external_volume = default_iceberg_external_volume

        self._check_database_exists_or_throw()
        if creation_mode == CreationMode.FAIL_IF_NOT_EXIST:
            self._check_internal_objects_exist_or_throw()
            # Best-effort: ensure the snapshot status table exists even in connect-only mode.
            # The table is an internal implementation object (not user-visible) that must be
            # present for scheduled snapshot tasks to succeed. Swallow any error — the caller
            # may lack CREATE TABLE privilege, and the absence of the table does not prevent
            # non-append_only feature store operations from working.
            try:
                self._ensure_snapshot_status_table_exists()
            except Exception:
                logger.debug(
                    "feature store: could not ensure snapshot status table exists "
                    "(insufficient privilege or schema not writable); skipping."
                )

        else:
            try:
                # Explicitly check if schema exists first since we may not have CREATE SCHEMA privilege
                if len(self._find_object("SCHEMAS", self._config.schema)) == 0:
                    self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._config.full_schema_path}").collect(
                        statement_params=self._telemetry_stmp
                    )
                for tag in to_sql_identifiers([_FEATURE_VIEW_METADATA_TAG, _FEATURE_STORE_OBJECT_TAG]):
                    self._session.sql(f"CREATE TAG IF NOT EXISTS {self._get_fully_qualified_name(tag)}").collect(
                        statement_params=self._telemetry_stmp
                    )
                # Metadata table for aggregation configs is created lazily by metadata manager
                self._ensure_snapshot_status_table_exists()
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"Failed to create feature store {name}: {e}."),
                ) from e
        self._check_feature_store_object_versions()
        self._online_http_client: Optional[online_service_http_client.OnlineServiceHttpClient] = None
        logger.info(f"Successfully connected to feature store: {self._config.full_schema_path}.")

    def _get_or_create_online_http_client(
        self,
    ) -> online_service_http_client.OnlineServiceHttpClient:
        """Return the FeatureStore-scoped HTTP client, creating it on first use."""
        if self._online_http_client is None:
            self._online_http_client = online_service_http_client.OnlineServiceHttpClient(session=self._session)
        return self._online_http_client

    def close(self) -> None:
        """Release per-FeatureStore resources (currently: the Online Service HTTP pool). Idempotent."""
        if self._online_http_client is not None:
            self._online_http_client.close()
            self._online_http_client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup at GC
            pass

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def update_default_warehouse(self, warehouse_name: str) -> None:
        """Update default warehouse for feature store.

        Args:
            warehouse_name: Name of warehouse.

        Raises:
            SnowflakeMLException: If warehouse does not exists.

        Example::

            >>> fs = FeatureStore(...)
            >>> fs.update_default_warehouse("MYWH_2")
            >>> draft_fv = FeatureView("my_fv", ...)
            >>> registered_fv = fs.register_feature_view(draft_fv, '2.0')
            >>> print(registered_fv.warehouse)
            MYWH_2

        """
        warehouse = SqlIdentifier(warehouse_name)
        warehouse_result = self._find_object("WAREHOUSES", warehouse)
        if len(warehouse_result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find warehouse {warehouse}"),
            )

        self._default_warehouse = warehouse

    @dispatch_decorator()
    def register_entity(self, entity: Entity) -> Entity:
        """
        Register Entity in the FeatureStore.

        Args:
            entity: Entity object to be registered.

        Returns:
            A registered entity object.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = Entity('BAR', ['A'], desc='entity bar')
            >>> fs.register_entity(e)
            >>> fs.list_entities().show()
            --------------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"      |"OWNER"     |
            --------------------------------------------------
            |BAR     |["A"]        |entity bar  |REGTEST_RL  |
            --------------------------------------------------

        """
        tag_name = self._get_entity_name(entity.name)
        found_rows = self._find_object("TAGS", tag_name)
        if len(found_rows) > 0:
            warnings.warn(
                f"Entity {entity.name} already exists. Skip registration.",
                stacklevel=2,
                category=UserWarning,
            )
            return entity

        # allowed_values will add double-quotes around each value, thus use resolved str here.
        join_keys = [f"{key.resolved()}" for key in entity.join_keys]
        join_keys_str = ",".join(join_keys)
        full_tag_name = self._get_fully_qualified_name(tag_name)
        try:
            self._session.sql(
                f"""CREATE TAG IF NOT EXISTS {full_tag_name}
                    ALLOWED_VALUES {_sql_string_literal(join_keys_str)}
                    COMMENT = {_sql_string_literal(entity.desc)}
                """
            ).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to register entity `{entity.name}`: {e}."),
            ) from e

        logger.info(f"Registered Entity {entity}.")

        return self.get_entity(entity.name)

    def update_entity(self, name: str, *, desc: Optional[str] = None) -> Optional[Entity]:
        """Update a registered entity with provided information.

        Args:
            name: Name of entity to update.
            desc: Optional new description to apply. Default to None.

        Raises:
            SnowflakeMLException: Error happen when updating.

        Returns:
            A new entity with updated information or None if the entity doesn't exist.

        Example::

            >>> fs = FeatureStore(...)
            <BLANKLINE>
            >>> e = Entity(name='foo', join_keys=['COL_1'], desc='old desc')
            >>> fs.list_entities().show()
            ------------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"    |"OWNER"     |
            ------------------------------------------------
            |FOO     |["COL_1"]    |old desc  |REGTEST_RL  |
            ------------------------------------------------
            <BLANKLINE>
            >>> fs.update_entity('foo', desc='NEW DESC')
            >>> fs.list_entities().show()
            ------------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"    |"OWNER"     |
            ------------------------------------------------
            |FOO     |["COL_1"]    |NEW DESC  |REGTEST_RL  |
            ------------------------------------------------

        """
        name = SqlIdentifier(name)
        found_rows = (
            self.list_entities().filter(F.col("NAME") == name.resolved()).collect(statement_params=self._telemetry_stmp)
        )

        if len(found_rows) == 0:
            warnings.warn(
                f"Entity {name} does not exist.",
                stacklevel=2,
                category=UserWarning,
            )
            return None

        new_desc = desc if desc is not None else found_rows[0]["DESC"]

        try:
            full_name = f"{self._config.full_schema_path}.{self._get_entity_name(name)}"
            self._session.sql(f"ALTER TAG {full_name} SET COMMENT = {_sql_string_literal(new_desc)}").collect(
                statement_params=self._telemetry_stmp
            )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to update entity `{name}`: {e}."),
            ) from e

        logger.info(f"Successfully updated Entity {name}.")
        return self.get_entity(name)

    # TODO: add support to update column desc once SNOW-894249 is fixed
    @dispatch_decorator()
    def register_feature_view(
        self,
        feature_view: FeatureView,
        version: str,
        *,
        block: bool = True,
        overwrite: bool = False,
    ) -> FeatureView:
        """
        Materialize a FeatureView to Snowflake backend.
        Incremental maintenance for updates on the source data will be automated if refresh_freq is set.
        NOTE: Each new materialization will trigger a full FeatureView history refresh for the data included in the
              FeatureView.

        Args:
            feature_view: FeatureView instance to materialize.
            version: version of the registered FeatureView.
                NOTE: Version only accepts letters, numbers and underscore. Also version will be capitalized.
            block: Deprecated. To make the initial refresh asynchronous, set the `initialize`
                argument on the `FeatureView` to `"ON_SCHEDULE"`. Default is true.
            overwrite: Overwrite the existing FeatureView with same version. This is the same as dropping the
                FeatureView first then recreate. NOTE: there will be backfill cost associated if the FeatureView is
                being continuously maintained.

        Returns:
            A materialized FeatureView object.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView entity has not been registered.
            SnowflakeMLException: [ValueError] Warehouse or default warehouse is not specified.
            SnowflakeMLException: [RuntimeError] Failed to create dynamic table, task, or view.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
            Exception: Unexpected error during registration.

        Example::

            >>> fs = FeatureStore(...)
            >>> # draft_fv is a local object that hasn't materialized to Snowflake backend yet.
            >>> feature_df = session.sql("select f_1, f_2 from source_table")
            >>> draft_fv = FeatureView("my_fv", [entities], feature_df)
            >>> print(draft_fv.status)
            FeatureViewStatus.DRAFT
            <BLANKLINE>
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |        |           |                    |
            -------------------------------------------
            <BLANKLINE>
            >>> # registered_fv is a local object that maps to a Snowflake backend object.
            >>> registered_fv = fs.register_feature_view(draft_fv, "v1")
            >>> print(registered_fv.status)
            FeatureViewStatus.ACTIVE
            <BLANKLINE>
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |ACTIVE              |
            -------------------------------------------

        """
        version = FeatureViewVersion(version)

        if block is False:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    'block=False is deprecated. Use FeatureView(..., initialize="ON_SCHEDULE") '
                    "for async initial refresh."
                ),
            )

        # Defense-in-depth: re-run the full snapshot config validation for append_only FVs.
        # FeatureView.__init__ already runs this, but the FV object may have been
        # mutated after construction (e.g. attributes patched directly on a draft
        # before registration), so re-check before any Snowflake-side mutation.
        if feature_view.append_only:
            feature_view_append_only_validation.validate_snapshot_config_for_register(
                feature_view,
                overwrite=overwrite,
            )

        if feature_view.status != FeatureViewStatus.DRAFT:
            try:
                return self._get_feature_view_if_exists(feature_view.name, str(version))
            except Exception:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(
                        f"FeatureView {feature_view.name}/{feature_view.version} status is {feature_view.status}, "
                        + "but it doesn't exist."
                    ),
                )

        for e in feature_view.entities:
            if not self._validate_entity_exists(e.name):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"Entity {e.name} has not been registered."),
                )

        # RTFVs are OFT-only (no DT/View, no preamble/postamble).
        if feature_view.is_realtime_feature_view:
            logger.warning("RealtimeFeatureView is in Public Preview since 1.43.0.")
            from snowflake.ml.feature_store.realtime_registration import (
                register_realtime_feature_view,
            )

            return register_realtime_feature_view(
                feature_store=self,
                feature_view=feature_view,
                version=version,
                overwrite=overwrite,
            )

        feature_view_name = FeatureView._get_physical_name(feature_view.name, version)
        if not overwrite:
            try:
                return self._get_feature_view_if_exists(feature_view.name, str(version))
            except Exception:
                pass

        created_resources: list[tuple[_FeatureStoreObjTypes, str]] = []
        streaming_preamble = None
        streaming_ref_count_incremented = False
        try:
            fully_qualified_name = self._get_fully_qualified_name(feature_view_name)
            refresh_freq = feature_view.refresh_freq

            if refresh_freq is None:
                obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW, snowml_version.VERSION)
            else:
                obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, snowml_version.VERSION)

            self._resolve_storage_config(feature_view, fully_qualified_name)

            tagging_clause_str = self._build_tagging_clause(feature_view, obj_info)

            if feature_view.online and feature_view.online_config is not None:
                if feature_view.online_config.store_type == OnlineStoreType.POSTGRES:
                    logger.warning("POSTGRES online store type is in Public Preview since 1.43.0.")
                    online_service.assert_online_service_running_with_query_endpoint(
                        self._session,
                        self._config.database,
                        self._config.schema,
                        statement_params=self._telemetry_stmp,
                    )

            # Streaming preamble: create udf_transformed table, then complete FV initialization
            if feature_view.is_streaming:
                from snowflake.ml.feature_store.streaming_registration import (
                    run_streaming_preamble,
                )

                streaming_preamble = run_streaming_preamble(
                    session=self._session,
                    feature_view=feature_view,
                    version=version,
                    feature_view_name=feature_view_name,
                    overwrite=overwrite,
                    metadata_manager=self._metadata_manager,
                    telemetry_stmp=self._telemetry_stmp,
                    get_stream_source_fn=self.get_stream_source,
                    get_fully_qualified_name_fn=self._get_fully_qualified_name,
                )
                created_resources.append(
                    (
                        _FeatureStoreObjTypes.UDF_TRANSFORMED_TABLE,
                        streaming_preamble.fq_udf_table,
                    )
                )
                created_resources.append(
                    (
                        _FeatureStoreObjTypes.UDF_TRANSFORMED_TABLE,
                        streaming_preamble.fq_backfill_table,
                    )
                )

                # Complete FeatureView initialization with the transformed schema.
                # After this, feature_view.query, output_schema, feature_names all
                # reflect the udf_transformed table — existing code paths work as-is.
                udf_df = self._session.table(streaming_preamble.fq_udf_table)
                feature_view._initialize_from_feature_df(udf_df)
                feature_view._validate()

            # Set authoring package version before tile query generation. A tiled
            # rollup FV reads its parent's tile columns and emits the same column
            # layout, so it must inherit the parent's authoring version (legacy
            # vs. new distinct-N format); otherwise the rollup SQL would reference
            # columns that don't exist on a legacy parent's DT. Every other feature
            # view records the current snowml version.
            if feature_view.is_tiled and feature_view.rollup_config is not None:
                feature_view._authoring_pkg_version = feature_view.rollup_config.source.authoring_pkg_version
            else:
                feature_view._authoring_pkg_version = snowml_version.VERSION

            # For tiled feature views, skip column definitions since the tiling query
            # produces different columns (TILE_START, partial aggregates)
            column_descs = "" if feature_view.is_tiled else self._build_column_descs(feature_view)

            # Steps 1-2: offline DT/View + OFT. Routed through the shared materialization
            # entry point so any future resource added to the registration flow is
            # automatically picked up by update_feature_view's recreate path too.
            self._materialize_feature_view_resources(
                mode="register",
                feature_view=feature_view,
                feature_view_name=feature_view_name,
                fully_qualified_name=fully_qualified_name,
                version=str(version),
                column_descs=column_descs,
                tagging_clause_str=tagging_clause_str,
                block=block,
                overwrite=overwrite,
                created_resources=created_resources,
            )

            # Step 3: Save aggregation metadata for tiled feature views (atomically)
            if feature_view.is_tiled:
                agg_metadata = AggregationMetadata(
                    feature_granularity=feature_view.feature_granularity,  # type: ignore[arg-type]
                    features=feature_view.aggregation_specs,  # type: ignore[arg-type]
                    feature_aggregation_method=(
                        feature_view.feature_aggregation_method.value
                        if feature_view.feature_aggregation_method is not None
                        else None
                    ),
                    # Stored in FEATURE_SPECS so ``get_feature_view``
                    # reconstructs the exact secondary key order registered.
                    aggregation_secondary_keys=(
                        list(feature_view.aggregation_secondary_keys)
                        if feature_view.aggregation_secondary_keys
                        else None
                    ),
                )
                # Convert SqlIdentifier keys to strings if descriptions exist
                descs = None
                if feature_view.feature_descs:
                    descs = {k.identifier(): v for k, v in feature_view.feature_descs.items()}
                # Save specs, descs, and FV metadata config atomically in a single
                # statement. Persist the version actually used to author the tiles
                # (a rollup FV inherits its parent's version) so reloads pick the
                # matching tile-column layout. Legacy FVs carry None here (a rollup on
                # a legacy parent inherits None) and persist no version row.
                authored_version = feature_view.authoring_pkg_version
                fv_meta_config = (
                    FeatureViewMetadataConfig(authoring_pkg_version=authored_version)
                    if authored_version is not None
                    else None
                )
                self._metadata_manager.save_feature_view_metadata(
                    feature_view.name, version, agg_metadata, descs, fv_metadata_config=fv_meta_config
                )

            # Persist the authored source-ref list when the caller
            # supplied one; callers that leave ``source_refs`` unset skip
            # the metadata write and no ``FV_SOURCE_REFS`` row is created.
            if feature_view.source_refs:
                self._metadata_manager.save_feature_view_source_refs(
                    feature_view.name,
                    version,
                    FvSourceRefsMetadata(sources=list(feature_view.source_refs)),
                )

            # Step 4: Save rollup metadata for PIT-correct training queries
            if feature_view._rollup_metadata is not None:
                self._metadata_manager.save_rollup_metadata(
                    feature_view.name, version, feature_view._rollup_metadata.to_dict()
                )

            # Step 4b: Persist append-only config (backup_source) in the
            # metadata table — not the DT tag — to avoid the 256-char tag limit.
            if feature_view.append_only and feature_view.backup_source is not None:
                self._metadata_manager.save_append_only_metadata(
                    feature_view.name,
                    str(version),
                    AppendOnlyMetadata(backup_source=feature_view.backup_source),
                )

            # Step 5: Streaming postamble (metadata + ref_count + server-side backfill task graph)
            if streaming_preamble is not None:
                from snowflake.ml.feature_store.streaming_registration import (
                    run_streaming_postamble,
                )

                # Track each backfill resource as it's created so failures
                # later in the postamble (metadata save, ref-count) still
                # leave a complete cleanup trail. Reverse-order rollback
                # then drops finalize -> root -> proc -> UDTF.
                _streaming_kind_to_obj_type = {
                    "BACKFILL_UDTF": _FeatureStoreObjTypes.FEATURE_VIEW_BACKFILL_UDTF,
                    "BACKFILL_PROC": _FeatureStoreObjTypes.FEATURE_VIEW_BACKFILL_PROC,
                    "BACKFILL_ROOT_TASK": _FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK,
                    "BACKFILL_FINALIZE_TASK": _FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK,
                }

                def _track_streaming_resource(kind: str, fq_name: str) -> None:
                    # For ``BACKFILL_UDTF``, ``fq_name`` already carries the
                    # argument-type signature so ``DROP FUNCTION`` can use it.
                    created_resources.append((_streaming_kind_to_obj_type[kind], fq_name))

                run_streaming_postamble(
                    session=self._session,
                    feature_view=feature_view,
                    version=version,
                    feature_view_name=feature_view_name,
                    preamble=streaming_preamble,
                    metadata_manager=self._metadata_manager,
                    default_warehouse=self._default_warehouse,
                    get_fully_qualified_name_fn=self._get_fully_qualified_name,
                    telemetry_stmp=self._telemetry_stmp,
                    on_resource_created=_track_streaming_resource,
                )
                streaming_ref_count_incremented = True

        except Exception as e:
            # We can't rollback in case of overwrite.
            if not overwrite:
                self._rollback_created_resources(created_resources)
                # Cleanup metadata (covers tiled, streaming, and append-only config)
                if (
                    feature_view.is_tiled
                    or streaming_preamble is not None
                    or (feature_view.append_only and feature_view.backup_source is not None)
                ):
                    self._metadata_manager.delete_feature_view_metadata(str(feature_view.name), version)
                # Only decrement ref_count if postamble successfully incremented it
                if streaming_preamble is not None and streaming_ref_count_incremented:
                    try:
                        self._metadata_manager.decrement_stream_source_ref_count(
                            streaming_preamble.resolved_source_name
                        )
                    except Exception as rollback_err:
                        logger.warning(
                            f"Best-effort rollback: failed to decrement stream source ref_count "
                            f"for {streaming_preamble.resolved_source_name}: {rollback_err}"
                        )

            if isinstance(e, snowml_exceptions.SnowflakeMLException):
                raise
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to register feature view {feature_view.name}/{version}: {e}"),
            ) from e

        logger.info(f"Registered FeatureView {feature_view.name}/{version} successfully.")
        # Suppress the warning that fires when the just-created dynamic table hasn't had
        # its first refresh yet.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=_FeatureViewSchemaNotReadyWarning)
            return self.get_feature_view(feature_view.name, str(version))

    @overload
    def update_feature_view(
        self,
        name: str,
        version: str,
        *,
        refresh_freq: Optional[str] = _UNSET,
        warehouse: Optional[str] = None,
        initialization_warehouse: Optional[str] = _KEEP_CURRENT,
        desc: Optional[str] = None,
        online_config: Optional[fv_mod.OnlineConfig] = None,
        updated_feature_df: Optional[DataFrame] = None,
    ) -> FeatureView:
        ...

    @overload
    def update_feature_view(
        self,
        name: FeatureView,
        version: Optional[str] = None,
        *,
        refresh_freq: Optional[str] = _UNSET,
        warehouse: Optional[str] = None,
        initialization_warehouse: Optional[str] = _KEEP_CURRENT,
        desc: Optional[str] = None,
        online_config: Optional[fv_mod.OnlineConfig] = None,
        updated_feature_df: Optional[DataFrame] = None,
    ) -> FeatureView:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def update_feature_view(
        self,
        name: Union[FeatureView, str],
        version: Optional[str] = None,
        *,
        refresh_freq: Optional[str] = _UNSET,
        warehouse: Optional[str] = None,
        initialization_warehouse: Optional[str] = _KEEP_CURRENT,
        desc: Optional[str] = None,
        online_config: Optional[fv_mod.OnlineConfig] = None,
        updated_feature_df: Optional[DataFrame] = None,
    ) -> FeatureView:
        """Update a registered feature view.
            Check feature_view.py for which fields are allowed to be updated after registration.

        Args:
            name: FeatureView object or name to suspend.
            version: Optional version of feature view. Must set when argument feature_view is a str.
            refresh_freq: updated refresh frequency.
            warehouse: updated warehouse.
            initialization_warehouse: updated initialization warehouse, used for the initial build and
                reinitializations of the backing dynamic table. Pass a warehouse name to set it, or ``None`` to clear
                it (all refreshes then run on ``warehouse``). When omitted, the existing value is left unchanged.
                Not supported for static feature views.
            desc: description of feature view.
            online_config: updated online configuration for the online feature table.
                If provided with enable=True, creates online feature table if absent.
                If provided with enable=False, drops online feature table if present.
                If None (default), no change to online status.
                During update, only explicitly set fields in the OnlineConfig will be updated.
            updated_feature_df: Optional replacement Snowpark ``DataFrame`` for the feature view
                definition. When set, the returned :class:`~snowflake.ml.feature_store.feature_view.FeatureView`
                is reinitialized from this dataframe (same session as the ``FeatureStore``). Not supported for
                streaming or rollup feature views.

        Returns:
            Updated FeatureView.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = FeatureView(
            ...     name='foo',
            ...     entities=[e1, e2],
            ...     feature_df=session.sql('...'),
            ...     desc='this is old description',
            ... )
            >>> fv = fs.register_feature_view(feature_view=fv, version='v1')
            >>> fs.list_feature_views().select("name", "version", "desc").show()
            ------------------------------------------------
            |"NAME"  |"VERSION"  |"DESC"                   |
            ------------------------------------------------
            |FOO     |v1         |this is old description  |
            ------------------------------------------------
            <BLANKLINE>
            >>> # update_feature_view will apply new arguments to the registered feature view.
            >>> new_fv = fs.update_feature_view(
            ...     name='foo',
            ...     version='v1',
            ...     desc='that is new descption',
            ... )
            >>> fs.list_feature_views().select("name", "version", "desc").show()
            ------------------------------------------------
            |"NAME"  |"VERSION"  |"DESC"                   |
            ------------------------------------------------
            |FOO     |v1         |THAT IS NEW DESCRIPTION  |
            ------------------------------------------------
            <BLANKLINE>
            >>> # Enable online storage with custom configuration
            >>> config = OnlineConfig(enable=True, target_lag='15s')
            >>> online_fv = fs.update_feature_view(
            ...     name='foo',
            ...     version='v1',
            ...     online_config=config,
            ... )
            >>> print(online_fv.online)
            True
            >>> # Evolve the schema of an append_only feature view. Schema evolution is
            >>> # extend-only: existing columns must keep their original positions, so any
            >>> # new column must be appended at the end of the SELECT.
            >>> updated_fv = fs.update_feature_view(
            ...     name='foo',
            ...     version='v1',
            ...     updated_feature_df=session.sql('SELECT id, name, ts, age FROM source'),
            ... )
            >>> 'AGE' in [f.name for f in updated_fv.feature_df.schema.fields]
            True

        Raises:
            SnowflakeMLException: [ValueError] If updated_feature_df is provided for a non-append_only FV.
            SnowflakeMLException: [RuntimeError] If FeatureView is not managed and refresh_freq is defined.
            SnowflakeMLException: [RuntimeError] Failed to update feature view.
        """
        if online_config is not None and online_config.store_type == OnlineStoreType.POSTGRES:
            logger.warning("POSTGRES online store type is in Public Preview since 1.43.0.")

        # Step 1: Validate inputs
        feature_view = self._validate_feature_view_name_and_version_input(name, version)
        # None when _UNSET (user didn't pass refresh_freq); the existing
        # _refresh_freq is preserved below via the conditional assignment.
        actual_refresh_freq: Optional[str] = refresh_freq if refresh_freq is not _UNSET else None
        new_desc = desc if desc is not None else feature_view.desc

        init_wh_changed = initialization_warehouse is not _KEEP_CURRENT

        if refresh_freq is not _UNSET:
            # Validate the prospective refresh_freq against the snapshot-config contract
            # without mutating the registered FV. Shallow copy is sufficient: the update
            # variant only reads ``_append_only`` and ``refresh_freq``. The structural
            # invariants (is_batch_view, feature_df, timestamp_col, entities, refresh_mode)
            # cannot change for an already-registered FV, so they don't need to be
            # re-checked here — the register-time gate is the source of truth for those.
            validation_fv = feature_view.copy()
            validation_fv._refresh_freq = refresh_freq
            feature_view_append_only_validation.validate_snapshot_config_for_update(validation_fv)

        # Validate static feature view constraints
        if feature_view.status == FeatureViewStatus.STATIC and (actual_refresh_freq or warehouse or init_wh_changed):
            full_name = f"{feature_view.name}/{feature_view.version}"
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=RuntimeError(
                    f"Static feature view '{full_name}' does not support refresh_freq, warehouse, "
                    "and initialization_warehouse."
                ),
            )

        # Fail fast on an online store type that cannot back this feature view (e.g. HYBRID_TABLE
        # for a tiled FV) before any planning/resource work, so it surfaces as a clean
        # INVALID_ARGUMENT rather than being wrapped by the update-failure rollback handler. The
        # rule is a feature-view invariant; check the target config on a copy since the live rebuild
        # path does not re-run __init__ validation.
        if online_config is not None:
            probe_fv = feature_view.copy()
            probe_fv._online_config = online_config
            try:
                probe_fv._validate_online_store_supported()
            except ValueError as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT, original_exception=e
                ) from e

        if updated_feature_df is not None:
            if not feature_view.append_only:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        "updated_feature_df is only supported for append_only feature views. "
                        "Delete and re-register the feature view to change its query."
                    ),
                )
            if updated_feature_df.session is not self._session:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        "updated_feature_df must be built using the FeatureStore's Snowpark session."
                    ),
                )

            if feature_view._version is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(
                        "feature store: feature view has no version after retrieval; this is a bug."
                    ),
                )
            feature_view_name = FeatureView._get_physical_name(feature_view.name, feature_view._version)
            fully_qualified_name = self._get_fully_qualified_name(feature_view_name)

            new_schema = fs_table_schema_evolution.schema_upper_name_map(updated_feature_df.schema)

            # Build a desired-state copy and apply the new state to it, leaving the registered
            # `feature_view` untouched.  Mutating `feature_view` here would corrupt the caller's
            # FV instance and would also break `_plan_feature_view_update_operations` for any
            # follow-up call that diffs registered-vs-target state.
            new_feature_view = feature_view.copy()
            cluster_by_kw = None
            if feature_view.cluster_by:
                cluster_by_kw = [c.resolved() for c in feature_view.cluster_by]
            new_feature_view._initialize_from_feature_df(updated_feature_df, cluster_by=cluster_by_kw)
            if feature_view._feature_desc is not None and new_feature_view._feature_desc is not None:
                # Preserve descriptions for unchanged feature columns. Extend-only schema
                # evolution may append new columns, which intentionally start with empty
                # comments until the caller sets them explicitly.
                for feature_name in new_feature_view._feature_desc:
                    if feature_name in feature_view._feature_desc:
                        new_feature_view._feature_desc[feature_name] = feature_view._feature_desc[feature_name]

            new_feature_view.desc = new_desc
            if warehouse is not None:
                new_feature_view.warehouse = warehouse
            if actual_refresh_freq is not None:
                new_feature_view._refresh_freq = actual_refresh_freq
            if online_config is not None:
                new_feature_view._online_config = online_config  # no public setter; mirrors __init__

            new_feature_view._validate()
            # update_feature_view cannot mutate the structural invariants that
            # validate_snapshot_config_for_register checks (entities, timestamp_col,
            # refresh_mode, etc.); they were enforced once at registration. Only the
            # cron refresh_freq can change here, so the narrower update gate suffices.
            feature_view_append_only_validation.validate_snapshot_config_for_update(new_feature_view)

            self._materialize_feature_view_resources(
                mode="update",
                feature_view=new_feature_view,
                feature_view_name=feature_view_name,
                fully_qualified_name=fully_qualified_name,
                version=str(feature_view.version),
                old_feature_view=feature_view,
                new_schema=new_schema,
            )

            # The CREATE OR REPLACE recreate above already applied refresh_freq / warehouse /
            # desc / online_config; calling _plan_feature_view_update_operations on this path
            # would emit redundant ALTERs whose rollback SQL would no longer reflect the
            # registered state.  Return the canonical FV directly.
            return self.get_feature_view(name=feature_view.name, version=str(feature_view.version))

        # Step 2: Plan all operations
        rollback_operations: list[Any] = []
        try:
            operations, rollback_operations = self._plan_feature_view_update_operations(
                feature_view,
                actual_refresh_freq,
                warehouse,
                initialization_warehouse,
                new_desc,
                online_config,
            )

            # Step 3: Execute atomically
            self._execute_atomic_operations(operations)

        except Exception as e:
            # Step 4: Rollback on failure
            self._handle_update_failure(e, rollback_operations, feature_view)

        return self.get_feature_view(name=feature_view.name, version=str(feature_view.version))

    @overload
    def read_feature_view(
        self,
        feature_view: str,
        version: str,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: Literal[False],
    ) -> DataFrame:
        ...

    @overload
    def read_feature_view(
        self,
        feature_view: str,
        version: str,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: Literal[True],
    ) -> pd.DataFrame:
        ...

    @overload
    def read_feature_view(
        self,
        feature_view: str,
        version: str,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: None = None,
    ) -> Union[DataFrame, pd.DataFrame]:
        ...

    @overload
    def read_feature_view(
        self,
        feature_view: FeatureView,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: Literal[False],
    ) -> DataFrame:
        ...

    @overload
    def read_feature_view(
        self,
        feature_view: FeatureView,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: Literal[True],
    ) -> pd.DataFrame:
        ...

    @overload
    def read_feature_view(
        self,
        feature_view: FeatureView,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: None = None,
    ) -> Union[DataFrame, pd.DataFrame]:
        ...

    @dispatch_decorator(  # type: ignore[misc]
        skip_wh_switch=_predicate_read_feature_view_skip_wh_switch,
        skip_telemetry=_predicate_read_feature_view_skip_telemetry,
    )
    def read_feature_view(
        self,
        feature_view: Union[FeatureView, str],
        version: Optional[str] = None,
        *,
        keys: Optional[list[list[str]]] = None,
        feature_names: Optional[list[str]] = None,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
        use_session_warehouse: bool = False,
        request_context: Optional[pd.DataFrame] = None,
        as_pandas: Optional[bool] = None,
    ) -> Union[DataFrame, pd.DataFrame]:
        """
        Read values from a FeatureView from either offline or online store.

        Args:
            feature_view: A FeatureView object to read from, or the name of feature view.
                If name is provided then version also must be provided.
            version: Optional version of feature view. Must set when argument feature_view is a str.
            keys: Optional list of primary key value lists to filter by. Each inner list should contain
                values in the same order as the entity join_keys. Works for both offline and online stores.
                Example: [["user1"], ["user2"]] for single key,
                [["user1", "item1"], ["user2", "item2"]] for composite keys.
                If None, returns all data (hybrid online tables only). For **Postgres**-backed online
                feature views, ``keys`` must be **non-empty**; the Online Service Query API does not support
                unbounded scans from SnowML.
            feature_names: Optional list of feature names to return. If None, returns all features.
                For **Postgres** online reads, a non-empty list is sent to the Query API as JSON
                ``features`` so the service can return only those columns; the DataFrame still contains
                only that subset (and join keys). Offline and hybrid online SQL paths use the same list
                in the SELECT clause.
            store_type: Store to read from - StoreType.ONLINE or StoreType.OFFLINE (default).
                Ignored for RealtimeFeatureViews, which always compute values at request time
                via the online Query API regardless of this setting.
            use_session_warehouse: If True, use the session's current warehouse instead of the feature
                store's configured warehouse. No-op for Postgres-backed online reads.
            request_context: Per-row request context for ``RealtimeFeatureView`` reads. A
                ``pandas.DataFrame`` whose columns are a superset of the RTFV's
                ``RequestSource.schema`` field names and whose row count equals
                ``len(keys)``. Single-row reads pass a 1-row frame. Required for RTFVs that
                were registered with a ``RequestSource``; must be omitted for RTFVs without
                one and is rejected for all non-RealtimeFeatureView kinds.
            as_pandas: Return type. ``None`` (default) returns ``pandas.DataFrame`` for Postgres-backed
                online reads and Snowpark ``DataFrame`` everywhere else. ``True`` forces
                ``pandas.DataFrame`` (rejected for ``store_type=StoreType.OFFLINE``). ``False`` forces
                Snowpark ``DataFrame``.

        Returns:
            Snowpark DataFrame (or ``pandas.DataFrame`` when ``as_pandas`` resolves to True)
            containing the FeatureView data.

        Raises:
            SnowflakeMLException: [ValueError] version argument is missing when argument feature_view is a str.
            SnowflakeMLException: [ValueError] FeatureView is not registered.
            SnowflakeMLException: [ValueError] Online store is not enabled for this feature view.
            SnowflakeMLException: [ValueError] Invalid store type.
            SnowflakeMLException: [ValueError] ``as_pandas=True`` with ``store_type=StoreType.OFFLINE``.
            SnowflakeMLException: [ValueError] ``request_context`` provided for a non-RealtimeFeatureView.
            SnowflakeMLException: [ValueError] ``request_context`` is missing, is not a
                ``pandas.DataFrame``, is missing required columns, or whose row count does
                not match ``len(keys)`` for an RTFV read.

        Example::

            >>> fs = FeatureStore(...)
            >>> # Read all data from offline store
            >>> fs.read_feature_view('foo', 'v1', store_type=StoreType.OFFLINE).show()
            ------------------------------------------
            |"NAME"  |"ID"  |"TITLE"  |"AGE"  |"TS"  |
            ------------------------------------------
            |jonh    |1     |boss     |20     |100   |
            |porter  |2     |manager  |30     |200   |
            ------------------------------------------
            <BLANKLINE>
            >>> # Filter by keys in offline store
            >>> fs.read_feature_view('foo', 'v1', keys=[["1"], ["2"]], store_type=StoreType.OFFLINE).show()
            ------------------------------------------
            |"NAME"  |"ID"  |"TITLE"  |"AGE"  |"TS"  |
            ------------------------------------------
            |jonh    |1     |boss     |20     |100   |
            |porter  |2     |manager  |30     |200   |
            ------------------------------------------
            <BLANKLINE>
            >>> # Read from online store with specific keys (same API)
            >>> fs.read_feature_view('foo', 'v1', keys=[["1"], ["2"]], store_type=StoreType.ONLINE).show()
            --------------------------------
            |"ID"  |"TITLE"  |"AGE"       |
            --------------------------------
            |1     |boss     |20          |
            |2     |manager  |30          |
            --------------------------------
            <BLANKLINE>
            >>> # Select specific features (works for both stores)
            >>> fs.read_feature_view('foo', 'v1', keys=[["1"]], feature_names=["TITLE", "AGE"]).show()
            ----------------------
            |"TITLE"  |"AGE"    |
            ----------------------
            |boss     |20       |
            ----------------------
            >>> # Postgres online reads default to pandas (local-build fast path).
            >>> # Pass as_pandas=False to force a Snowpark DataFrame instead.
            >>> pdf = fs.read_feature_view('foo', 'v1', keys=[["1"]], store_type=StoreType.ONLINE)
            >>> type(pdf).__name__
            'DataFrame'

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        store_type = _get_store_type(store_type)

        # RealtimeFeatureView reads run a dedicated branch BEFORE the
        # general online/offline dispatch. RTFVs have no offline backing
        # and no SQL-backed online path -- they always route through the
        # Postgres Online Service Query API and require a per-row
        # request_context payload. The ``store_type`` kwarg is ignored
        # here (mirrors ``read_feature_group``, which has no store_type
        # at all) so callers don't have to override the public default.
        if feature_view.is_realtime_feature_view:
            return self._read_realtime_feature_view(
                feature_view,
                keys=keys,
                feature_names=feature_names,
                request_context=request_context,
                as_pandas=as_pandas,
            )

        if request_context is not None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "request_context is only supported for RealtimeFeatureView reads. "
                    f"Feature view {feature_view.name}/{feature_view.version} is not realtime; "
                    "remove the request_context kwarg."
                ),
            )

        # Resolve the as_pandas default. Postgres online reads default to pandas because
        # the Online Service Query API already returns row-oriented JSON; the historical
        # Snowpark default required a wasteful create_dataframe(VALUES ...) round-trip.
        # All other paths default to Snowpark DataFrame to preserve existing behavior.
        if as_pandas is None:
            as_pandas = (
                store_type == fv_mod.StoreType.ONLINE
                and feature_view.online_config is not None
                and feature_view.online_config.store_type == OnlineStoreType.POSTGRES
            )

        if store_type == fv_mod.StoreType.ONLINE:
            return self._read_from_online_store(feature_view, keys, feature_names, as_pandas=as_pandas)
        elif store_type == fv_mod.StoreType.OFFLINE:
            if as_pandas:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        "as_pandas=True is not supported for store_type=OFFLINE; offline reads can be arbitrarily "
                        "large and a local pandas materialization invites OOM. Call .to_pandas() on the returned "
                        "Snowpark DataFrame instead if you accept the full-scan cost."
                    ),
                )
            return self._read_from_offline_store(feature_view, keys, feature_names)
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid store type: {store_type}"),
            )

    @dispatch_decorator()
    def list_feature_views(
        self,
        *,
        entity_name: Optional[str] = None,
        feature_view_name: Optional[str] = None,
        verbose: bool = False,
    ) -> DataFrame:
        """
        List FeatureViews in the FeatureStore.
        If entity_name is specified, FeatureViews associated with that Entity will be listed.
        If feature_view_name is specified, further reducing the results to only match the specified name.

        Args:
            entity_name: Entity name.
            feature_view_name: FeatureView name.
            verbose: When True, include the ``initialization_warehouse``, ``source_refs`` and
                ``backup_source`` columns in the output. ``initialization_warehouse`` (string,
                nullable) is the warehouse used for the initial build / reinitialization of the
                backing dynamic table (``None`` when unset). ``source_refs`` (string, nullable) is
                the JSON-encoded list of authored source bindings captured at registration time
                (``None`` when no source refs were recorded). ``backup_source`` (string, nullable)
                contains the fully-qualified name of the historical snapshot table cloned at
                registration time (only set for append-only feature views registered with a
                ``backup_source``; ``None`` otherwise). Defaults to False.

        Returns:
            FeatureViews information as a Snowpark DataFrame. Each row always includes
            ``append_only`` (bool). ``initialization_warehouse``, ``source_refs`` and
            ``backup_source`` (all string, nullable) are included only when ``verbose=True``.

        Example::

            >>> fs = FeatureStore(...)
            >>> draft_fv = FeatureView(
            ...     name='foo',
            ...     entities=[e1, e2],
            ...     feature_df=session.sql('...'),
            ...     desc='this is description',
            ... )
            >>> fs.register_feature_view(feature_view=draft_fv, version='v1')
            >>> fs.list_feature_views().select("name", "version", "desc").show()
            --------------------------------------------
            |"NAME"  |"VERSION"  |"DESC"               |
            --------------------------------------------
            |FOO     |v1         |this is description  |
            --------------------------------------------

        """
        if feature_view_name is not None:
            feature_view_name = SqlIdentifier(feature_view_name)

        if entity_name is not None:
            entity_name = SqlIdentifier(entity_name)
            return self._optimized_find_feature_views(entity_name, feature_view_name, verbose=verbose)
        else:
            fv_rows = self._get_fv_backend_representations(feature_view_name, prefix_match=True)

            output_values: list[list[Any]] = []
            output_values_extra: list[list[Any]] = []
            iceberg_config_cache: dict[str, StorageConfig] = {}
            for row, _ in fv_rows:
                self._extract_feature_view_info(row, output_values, output_values_extra, iceberg_config_cache)

            from snowflake.ml.feature_store.realtime_registration import (
                append_realtime_listing_rows,
            )

            append_realtime_listing_rows(
                feature_store=self,
                feature_view_name_prefix=feature_view_name,
                output_values=output_values,
                output_values_extra=output_values_extra,
                fv_kind_realtime=_FV_KIND_REALTIME,
                default_storage_config_json=_DEFAULT_STORAGE_CONFIG_JSON,
            )

            return _create_list_feature_views_dataframe(
                self._session, output_values, output_values_extra, verbose=verbose
            )

    @dispatch_decorator()
    def get_feature_view(self, name: str, version: str) -> FeatureView:
        """
        Retrieve previously registered FeatureView.

        Args:
            name: FeatureView name.
            version: FeatureView version.

        Returns:
            FeatureView object.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView with name and version is not found,
                or incurred exception when reconstructing the FeatureView object.

        Example::

            >>> fs = FeatureStore(...)
            >>> # draft_fv is a local object that hasn't materialized to Snowflake backend yet.
            >>> draft_fv = FeatureView(
            ...     name='foo',
            ...     entities=[e1],
            ...     feature_df=session.sql('...'),
            ...     desc='this is description',
            ... )
            >>> fs.register_feature_view(feature_view=draft_fv, version='v1')
            <BLANKLINE>
            >>> # fv is a local object that maps to a Snowflake backend object.
            >>> fv = fs.get_feature_view('foo', 'v1')
            >>> print(f"name: {fv.name}")
            >>> print(f"version:{fv.version}")
            >>> print(f"desc:{fv.desc}")
            name: FOO
            version:v1
            desc:this is description

        """
        name = SqlIdentifier(name)
        version = FeatureViewVersion(version)

        fv_name = FeatureView._get_physical_name(name, version)
        results = self._get_fv_backend_representations(fv_name)
        if len(results) == 1:
            return self._compose_feature_view(
                results[0][0],
                results[0][1],
                self.list_entities().collect(statement_params=self._telemetry_stmp),
            )
        if len(results) == 0:
            # RTFVs are OFT-only; the metadata table is the source of truth.
            rtfv_meta = self._metadata_manager.get_realtime_config(name.resolved(), str(version))
            if rtfv_meta is not None:
                from snowflake.ml.feature_store.realtime_registration import (
                    compose_rtfv_from_metadata,
                )

                return compose_rtfv_from_metadata(self, rtfv_meta)

        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError(f"Failed to find FeatureView {name}/{version}: {results}"),
        )

    @dispatch_decorator()
    def create_online_service(self, producer_role: str, consumer_role: str) -> online_service.OnlineServiceResult:
        """Create the Online Service for this store's schema.

        Requires schema OWNERSHIP and account feature enablement. After SUCCESS, poll
        :meth:`get_online_service_status` until ``RUNNING`` before using online feature
        reads or stream ingestion.

        Args:
            producer_role: Role name granted producer privileges on the Online Service.
            consumer_role: Role name granted consumer privileges on the Online Service.

        Returns:
            Parsed create result from the Online Service.
        """
        return online_service.create_online_service(
            self._session,
            self._config.database,
            self._config.schema,
            producer_role,
            consumer_role,
            statement_params=self._telemetry_stmp,
        )

    @dispatch_decorator()
    def get_online_service_status(self) -> online_service.OnlineServiceStatus:
        """Return Online Service status.

        Poll this method after :meth:`create_online_service` until ``status`` is
        ``RUNNING`` before using online feature reads or stream ingestion.

        Returns:
            OnlineServiceStatus with ``status``, ``message``, and ``endpoints``.
        """
        return online_service.get_online_service_status(
            self._session,
            self._config.database,
            self._config.schema,
            statement_params=self._telemetry_stmp,
        )

    @dispatch_decorator()
    def drop_online_service(self) -> online_service.OnlineServiceResult:
        """Drop the Online Service.

        Stops the Online Service and releases its resources. Online feature reads
        and stream ingestion will no longer be available after this call.

        Returns:
            OnlineServiceResult with ``status`` and ``message``.
        """
        return online_service.drop_online_service(
            self._session,
            self._config.database,
            self._config.schema,
            statement_params=self._telemetry_stmp,
        )

    @overload
    def refresh_feature_view(
        self,
        feature_view: str,
        version: str,
        *,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
    ) -> None:
        ...

    @overload
    def refresh_feature_view(
        self,
        feature_view: FeatureView,
        version: Optional[str] = None,
        *,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
    ) -> None:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def refresh_feature_view(
        self,
        feature_view: Union[FeatureView, str],
        version: Optional[str] = None,
        *,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
    ) -> None:
        """Manually refresh a feature view.

        Args:
            feature_view: A registered feature view object, or the name of feature view.
            version: Optional version of feature view. Must set when argument feature_view is a str.
            store_type: Specify which storage to refresh. Can be StoreType.OFFLINE or StoreType.ONLINE.
                - StoreType.OFFLINE (default): Refreshes the offline feature view.
                - StoreType.ONLINE: Refreshes the online feature table for real-time serving.
                  Only available for feature views with online=True.
                Defaults to StoreType.OFFLINE.

        Raises:
            SnowflakeMLException: [ValueError] Invalid store type.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            <BLANKLINE>
            >>> # refresh with name and version
            >>> fs.refresh_feature_view('MY_FV', 'v1')
            >>> fs.get_refresh_history('MY_FV', 'v1').show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------
            <BLANKLINE>
            >>> # refresh with feature view object
            >>> fs.refresh_feature_view(fv)
            >>> fs.get_refresh_history(fv).show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:54:06.680000  |2024-07-10 14:54:07.226000  |INCREMENTAL       |
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        store_type = _get_store_type(store_type)

        if feature_view.append_only:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Feature view {feature_view.name}/{feature_view.version} has append_only enabled. "
                    "Manual refresh is not supported for append-only feature views because "
                    "refreshes are managed by the scheduled snapshot task."
                ),
            )

        if store_type == fv_mod.StoreType.ONLINE:
            # Refresh online feature table only
            if not feature_view.online:
                warnings.warn(
                    f"Feature view {feature_view.name}/{feature_view.version} does not have online storage enabled.",
                    stacklevel=2,
                    category=UserWarning,
                )
                return

            if (
                feature_view.online_config is not None
                and feature_view.online_config.store_type == fv_mod.OnlineStoreType.POSTGRES
            ):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        "Manual refresh is not supported for Postgres online feature tables. "
                        "Postgres online stores are refreshed automatically by the Online Service."
                    ),
                )

            # Use the unified method but specify online-only refresh
            self._update_feature_view_status(feature_view, "REFRESH", store_type=fv_mod.StoreType.ONLINE)
        elif store_type == fv_mod.StoreType.OFFLINE:
            # Refresh offline feature view only
            if feature_view.status == FeatureViewStatus.STATIC:
                warnings.warn(
                    "Static feature view can't be refreshed. You must set refresh_freq when register_feature_view().",
                    stacklevel=2,
                    category=UserWarning,
                )
                return
            self._update_feature_view_status(feature_view, "REFRESH", store_type=fv_mod.StoreType.OFFLINE)
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid store type: {store_type}"),
            )

    @overload
    def get_refresh_history(
        self,
        feature_view: FeatureView,
        version: Optional[str] = None,
        *,
        verbose: bool = False,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
    ) -> DataFrame:
        ...

    @overload
    def get_refresh_history(
        self,
        feature_view: str,
        version: str,
        *,
        verbose: bool = False,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
    ) -> DataFrame:
        ...

    def get_refresh_history(
        self,
        feature_view: Union[FeatureView, str],
        version: Optional[str] = None,
        *,
        verbose: bool = False,
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.OFFLINE,
    ) -> DataFrame:
        """Get refresh history statistics about a feature view.

        Args:
            feature_view: A registered feature view object, or the name of feature view.
            version: Optional version of feature view. Must set when argument feature_view is a str.
            verbose: Return more detailed history when set true.
            store_type: Store to get refresh history from - StoreType.ONLINE or StoreType.OFFLINE (default).
                - StoreType.OFFLINE (default): Returns refresh history for the offline feature view (dynamic table).
                - StoreType.ONLINE: Returns refresh history for the online feature table.
                  Only available for feature views with online=True.

        Returns:
            A dataframe contains the refresh history information.

        Raises:
            SnowflakeMLException: [ValueError]
                If store_type is ONLINE but feature view doesn't have online storage enabled.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            >>> # Get offline refresh history (default)
            >>> fs.refresh_feature_view('MY_FV', 'v1')
            >>> fs.get_refresh_history('MY_FV', 'v1').show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------
            <BLANKLINE>
            >>> # Get online refresh history (for feature views with online storage)
            >>> fs.get_refresh_history('MY_FV', 'v1', store_type=StoreType.ONLINE).show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"          |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1$ONLINE |SUCCEEDED  |2024-07-10 14:54:01.200000  |2024-07-10 14:54:02.100000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------
            <BLANKLINE>
            >>> # Verbose mode works for both storage types
            >>> fs.get_refresh_history(fv, verbose=True, store_type=StoreType.OFFLINE).show()
            >>> fs.get_refresh_history(fv, verbose=True, store_type=StoreType.ONLINE).show()

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        store_type = _get_store_type(store_type)

        if feature_view.status == FeatureViewStatus.STATIC:
            warnings.warn(
                "Static feature view never refreshes.",
                stacklevel=2,
                category=UserWarning,
            )
            return self._session.create_dataframe([Row()])

        if feature_view.status == FeatureViewStatus.DRAFT:
            warnings.warn(
                "This feature view has not been registered thus has no refresh history.",
                stacklevel=2,
                category=UserWarning,
            )
            return self._session.create_dataframe([Row()])

        # Validate online store request
        if store_type == fv_mod.StoreType.ONLINE:
            if not feature_view.online:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"Feature view '{feature_view.name}' version '{feature_view.version}' "
                        "does not have online storage enabled. Cannot retrieve online refresh history."
                    ),
                )
            return self._get_online_refresh_history(feature_view, verbose)
        elif store_type == fv_mod.StoreType.OFFLINE:
            return self._get_offline_refresh_history(feature_view, verbose)
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid store type: {store_type}"),
            )

    def _get_offline_refresh_history(self, feature_view: FeatureView, verbose: bool) -> DataFrame:
        """Get refresh history for an offline feature view (dynamic table).

        For streaming FVs with a server-side backfill task graph, also
        includes ``INFORMATION_SCHEMA.TASK_HISTORY`` rows for the backfill
        root and any future window children, projected into the same
        5-column shape with ``REFRESH_ACTION='BACKFILL'``. The internal
        finalizer task is filtered out — it has no user-meaningful
        refresh semantics. Verbose mode keeps the DT-only schema and
        emits a ``UserWarning`` (the two history views have incompatible
        verbose columns).

        Args:
            feature_view: Feature view to query history for.
            verbose: When True, return all columns from
                ``DYNAMIC_TABLE_REFRESH_HISTORY`` and skip the backfill
                ``UNION``.

        Returns:
            DataFrame with columns ``name, state, refresh_start_time,
            refresh_end_time, refresh_action`` (or all DT columns when
            ``verbose=True``).
        """
        fv_resolved_name = FeatureView._get_physical_name(
            feature_view.name,
            feature_view.version,  # type: ignore[arg-type]
        ).resolved()
        schema_resolved = self._config.schema.resolved()
        db_resolved = self._config.database.resolved()

        # UNION backfill task history only for streaming FVs that have a
        # recorded ``backfill_root_task_name``; legacy streaming FVs predate
        # the task-graph migration and fall through to DT-only history.
        #
        # ``user_visible_*`` patterns drive the user-facing UNION; the
        # ``all_backfill`` pattern is kept for the verbose-mode hint that
        # points users at full ``TASK_HISTORY`` (incl. the finalizer) for
        # debugging.
        user_visible_root_pattern: Optional[str] = None
        user_visible_window_pattern: Optional[str] = None
        all_backfill_pattern: Optional[str] = None
        if feature_view.is_streaming:
            try:
                streaming_meta = self._metadata_manager.get_streaming_metadata(
                    str(feature_view.name), str(feature_view.version)
                )
            except Exception:
                streaming_meta = None
            if streaming_meta is not None and streaming_meta.backfill_root_task_name:
                from snowflake.ml.feature_store.streaming_registration import (
                    _get_backfill_task_name_pattern,
                    _get_user_visible_backfill_task_name_patterns,
                )

                physical = FeatureView._get_physical_name(
                    feature_view.name,
                    feature_view.version,  # type: ignore[arg-type]
                )
                all_backfill_pattern = _get_backfill_task_name_pattern(physical)
                user_visible_root_pattern, user_visible_window_pattern = _get_user_visible_backfill_task_name_patterns(
                    physical
                )

        if verbose:
            if all_backfill_pattern is not None:
                warnings.warn(
                    "Streaming feature view backfill task history is not included in "
                    "verbose mode because DYNAMIC_TABLE_REFRESH_HISTORY and TASK_HISTORY "
                    "have different verbose schemas. Use the default (non-verbose) call "
                    "to see backfill rows, or query INFORMATION_SCHEMA.TASK_HISTORY "
                    f"directly with WHERE NAME LIKE '{all_backfill_pattern}' ESCAPE '\\\\' "
                    "for full task-graph details (including the internal finalizer task).",
                    UserWarning,
                    stacklevel=3,
                )
            return self._session.sql(
                f"""
                SELECT *
                FROM TABLE (
                    {db_resolved}.INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY (RESULT_LIMIT => 10000)
                )
                WHERE NAME = '{fv_resolved_name}'
                AND SCHEMA_NAME = '{schema_resolved}'
                """
            )

        dt_history_sql = f"""
            SELECT name, state, refresh_start_time, refresh_end_time, refresh_action
            FROM TABLE (
                {db_resolved}.INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY (RESULT_LIMIT => 10000)
            )
            WHERE NAME = '{fv_resolved_name}'
            AND SCHEMA_NAME = '{schema_resolved}'
            """

        if user_visible_root_pattern is None or user_visible_window_pattern is None:
            return self._session.sql(dt_history_sql)

        # User-facing projection: surface the FV's physical name (matching
        # the dynamic-table rows above) instead of the underlying task
        # name; ``REFRESH_ACTION='BACKFILL'`` is the only label that
        # distinguishes a backfill row from incremental refreshes. The
        # finalizer task is intentionally excluded — it is internal cleanup
        # plumbing with no user-meaningful refresh semantics. The ``LIKE``
        # for window children is harmless in Phase 1 (no rows match) and
        # extends to Phase 2 ``$BACKFILL_W<NNN>`` rows without DDL change.
        backfill_history_sql = f"""
            SELECT
                '{fv_resolved_name}'                       AS name,
                STATE                                      AS state,
                COALESCE(QUERY_START_TIME, SCHEDULED_TIME) AS refresh_start_time,
                COMPLETED_TIME                             AS refresh_end_time,
                'BACKFILL'                                 AS refresh_action
            FROM TABLE (
                {db_resolved}.INFORMATION_SCHEMA.TASK_HISTORY (RESULT_LIMIT => 10000)
            )
            WHERE DATABASE_NAME = '{db_resolved}'
              AND SCHEMA_NAME   = '{schema_resolved}'
              AND (NAME = '{user_visible_root_pattern}' OR NAME LIKE '{user_visible_window_pattern}')
            """

        return self._session.sql(
            f"""
            SELECT name, state, refresh_start_time, refresh_end_time, refresh_action
            FROM (
                {dt_history_sql}
                UNION ALL
                {backfill_history_sql}
            )
            ORDER BY refresh_start_time DESC NULLS LAST
            """
        )

    def _get_online_refresh_history(self, feature_view: FeatureView, verbose: bool) -> DataFrame:
        """Get refresh history for online feature table."""
        online_table_name = FeatureView._get_online_table_name(feature_view.name, feature_view.version)
        select_cols = "*" if verbose else "name, state, refresh_start_time, refresh_end_time, refresh_action"
        name = (
            f"{self._config.database.identifier()}."
            f"{self._config.schema.identifier()}."
            f"{online_table_name.identifier()}"
        )
        return self._session.sql(
            f"""
            SELECT
                {select_cols}
            FROM TABLE (
                {self._config.database}.INFORMATION_SCHEMA.ONLINE_FEATURE_TABLE_REFRESH_HISTORY (
                    NAME => '{name}'
                )
            )
            """
        )

    @overload
    def resume_feature_view(self, feature_view: FeatureView) -> FeatureView:
        ...

    @overload
    def resume_feature_view(self, feature_view: str, version: str) -> FeatureView:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def resume_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> FeatureView:
        """
        Resume a previously suspended FeatureView.

        This operation resumes both the offline feature view (dynamic table and associated task)
        and the online feature table (if it exists) to ensure consistent state across all storage types.

        Args:
            feature_view: FeatureView object or name to resume.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Returns:
            A new feature view with updated status.

        Example::

            >>> fs = FeatureStore(...)
            >>> # you must already have feature views registered
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            >>> fs.suspend_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |SUSPENDED           |
            -------------------------------------------
            <BLANKLINE>
            >>> fs.resume_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |ACTIVE              |
            -------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        # Plan atomic resume operations
        operations, rollback_operations = self._plan_feature_view_status_operations(feature_view, "RESUME")

        try:
            # Execute all operations atomically
            self._execute_atomic_operations(operations)
            logger.info(f"Successfully RESUME FeatureView {feature_view.name}/{feature_view.version}.")
        except Exception as e:
            # Handle failure with rollback
            self._handle_status_operation_failure(e, rollback_operations, feature_view, "RESUME")

        return self.get_feature_view(feature_view.name, str(feature_view.version))

    @overload
    def suspend_feature_view(self, feature_view: FeatureView) -> FeatureView:
        ...

    @overload
    def suspend_feature_view(self, feature_view: str, version: str) -> FeatureView:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def suspend_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> FeatureView:
        """
        Suspend an active FeatureView.

        This operation suspends both the offline feature view (dynamic table and associated task)
        and the online feature table (if it exists).

        Args:
            feature_view: FeatureView object or name to suspend.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Returns:
            A new feature view with updated status.

        Example::

            >>> fs = FeatureStore(...)
            >>> # assume you already have feature views registered
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            >>> fs.suspend_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |SUSPENDED           |
            -------------------------------------------
            <BLANKLINE>
            >>> fs.resume_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |ACTIVE              |
            -------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        # Plan atomic suspend operations
        operations, rollback_operations = self._plan_feature_view_status_operations(feature_view, "SUSPEND")

        try:
            # Execute all operations atomically
            self._execute_atomic_operations(operations)
            logger.info(f"Successfully suspended FeatureView {feature_view.name}/{feature_view.version}.")
        except Exception as e:
            # Handle failure with rollback
            self._handle_status_operation_failure(e, rollback_operations, feature_view, "SUSPEND")

        return self.get_feature_view(feature_view.name, str(feature_view.version))

    @overload
    def delete_feature_view(self, feature_view: FeatureView) -> None:
        ...

    @overload
    def delete_feature_view(self, feature_view: str, version: str) -> None:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def delete_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> None:
        """
        Delete a FeatureView.

        Args:
            feature_view: FeatureView object or name to delete.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not registered.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = FeatureView('FV0', ...)
            >>> fv1 = fs.register_feature_view(fv, 'FIRST')
            >>> fv2 = fs.register_feature_view(fv, 'SECOND')
            >>> fs.list_feature_views().select('NAME', 'VERSION').show()
            ----------------------
            |"NAME"  |"VERSION"  |
            ----------------------
            |FV0     |SECOND     |
            |FV0     |FIRST      |
            ----------------------
            <BLANKLINE>
            >>> # delete with name and version
            >>> fs.delete_feature_view('FV0', 'FIRST')
            >>> fs.list_feature_views().select('NAME', 'VERSION').show()
            ----------------------
            |"NAME"  |"VERSION"  |
            ----------------------
            |FV0     |SECOND     |
            ----------------------
            <BLANKLINE>
            >>> # delete with feature view object
            >>> fs.delete_feature_view(fv2)
            >>> fs.list_feature_views().select('NAME', 'VERSION').show()
            ----------------------
            |"NAME"  |"VERSION"  |
            ----------------------
            |        |           |
            ----------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        # TODO: we should leverage lineage graph to check downstream deps, and block the deletion
        # if there're other FVs depending on this
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        # RTFVs are OFT-only — no DT/View teardown.
        if feature_view.is_realtime_feature_view:
            from snowflake.ml.feature_store.realtime_registration import (
                delete_realtime_feature_view,
            )

            delete_realtime_feature_view(feature_store=self, feature_view=feature_view)
            return

        fully_qualified_name = feature_view.fully_qualified_name()
        if feature_view.status == FeatureViewStatus.STATIC:
            self._session.sql(f"DROP VIEW IF EXISTS {fully_qualified_name}").collect(
                statement_params=self._telemetry_stmp
            )
        else:
            # Both regular Dynamic Tables and Dynamic Iceberg Tables use DROP DYNAMIC TABLE
            self._session.sql(f"DROP DYNAMIC TABLE IF EXISTS {fully_qualified_name}").collect(
                statement_params=self._telemetry_stmp
            )

        # Drop companion task first (if any), then snapshot artifacts for this FV.
        # This mirrors the overwrite cleanup ordering and guarantees we don’t
        # leave a running CRON task targeting a deleted snapshot table.
        fv_physical_name = FeatureView._get_physical_name(feature_view.name, feature_view.version)
        self._cleanup_stale_feature_view_resources(
            feature_view=feature_view,
            feature_view_name=fv_physical_name,
            fully_qualified_name=fully_qualified_name,
            new_has_task=False,
            drop_snapshot_table=True,
        )

        # Delete online feature table if it exists
        if feature_view.online:
            fully_qualified_online_name = feature_view.fully_qualified_online_table_name()
            try:
                self._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {fully_qualified_online_name}").collect(
                    statement_params=self._telemetry_stmp
                )
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(
                        f"Failed to delete online feature table {fully_qualified_online_name}: {e}"
                    ),
                )

        # Clean up streaming FV resources (udf_transformed table + ref count)
        if feature_view.is_streaming:
            from snowflake.ml.feature_store.streaming_registration import (
                cleanup_streaming_feature_view,
            )

            feature_view_name = FeatureView._get_physical_name(feature_view.name, feature_view.version)
            cleanup_streaming_feature_view(
                session=self._session,
                feature_view_name=feature_view_name,
                version=str(feature_view.version),
                fv_name=str(feature_view.name),
                fv_metadata=feature_view._metadata(),
                metadata_manager=self._metadata_manager,
                get_fully_qualified_name_fn=self._get_fully_qualified_name,
                telemetry_stmp=self._telemetry_stmp,
            )

        # Delete aggregation metadata and feature descriptions if exist
        self._metadata_manager.delete_feature_view_metadata(str(feature_view.name), str(feature_view.version))

        logger.info(f"Deleted FeatureView {feature_view.name}/{feature_view.version}.")

    # =========================================================================
    # FeatureGroup CRUD
    # =========================================================================

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def register_feature_group(self, feature_group: FeatureGroup, version: str) -> FeatureGroup:
        """Materialize a FeatureGroup as a Postgres-backed Online Feature Table.

        Args:
            feature_group: Draft FeatureGroup to register.
            version: User-facing FeatureGroup version.

        Returns:
            FeatureGroup: equivalent to the input, with :attr:`FeatureGroup.version` populated.

        Raises:
            SnowflakeMLException: ``[ValueError]`` if any precondition is
                violated (invalid version, unregistered / offline / non-Postgres
                source, missing entity, mismatched join keys, or name collision).  # noqa: DAR402
            SnowflakeMLException: ``[RuntimeError]`` if OFT creation, tagging, or
                metadata write fails.  # noqa: DAR402

        # noqa: DAR401
        """
        logger.warning("FeatureGroup is in Public Preview since 1.43.0.")
        return fg_mod.register_feature_group(self, feature_group, version)

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def list_feature_groups(self) -> DataFrame:
        """List FeatureGroups registered in this FeatureStore.

        Returns:
            Snowpark DataFrame with one row per registered ``(name, version)``
            and columns ``NAME, VERSION, DESC, OWNER, AUTO_PREFIX, SOURCES,
            OUTPUT_COLUMNS``. Empty when no FGs are registered.
        """
        return fg_mod.list_feature_groups(self)

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def get_feature_group(self, name: str, version: str) -> FeatureGroup:
        """Retrieve a previously registered FeatureGroup.

        Args:
            name: FeatureGroup name.
            version: FeatureGroup version.

        Returns:
            FeatureGroup: equivalent to the registered original, with version populated.

        Raises:
            SnowflakeMLException: ``[ValueError]`` if no FeatureGroup with the
                given *(name, version)* exists.  # noqa: DAR402
        """
        return fg_mod.get_feature_group(self, name, version)

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def delete_feature_group(self, name: str, version: str) -> None:
        """Delete a registered FeatureGroup (OFT and metadata row).

        Idempotent on both the OFT (``DROP ... IF EXISTS``) and the metadata row.

        Args:
            name: FeatureGroup name.
            version: FeatureGroup version.

        Raises:
            SnowflakeMLException: ``[RuntimeError]`` if ``DROP ONLINE FEATURE TABLE``
                fails.  # noqa: DAR402
        """
        fg_mod.delete_feature_group(self, name, version)

    @dispatch_decorator(
        skip_wh_switch=_predicate_read_feature_group_skip_wh_switch,
        skip_telemetry=_predicate_read_feature_group_skip_telemetry,
    )
    def read_feature_group(
        self,
        feature_group: Union[FeatureGroup, str],
        version: Optional[str] = None,
        *,
        keys: list[list[Any]],
        store_type: Union[fv_mod.StoreType, str] = fv_mod.StoreType.ONLINE,
        request_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Read feature values from a registered :class:`FeatureGroup` for a batch of entity rows.

        Issues a single ``POST /api/v1/query`` against the Postgres-backed OFT
        with ``object_type=feature_group``; the server returns the FG's
        predetermined output column set (no per-call feature subsetting).

        Args:
            feature_group: A hydrated :class:`FeatureGroup` (from
                :meth:`get_feature_group`) or the FG name as a string. When a
                string, *version* is required.
            version: Required when *feature_group* is a string. When a
                :class:`FeatureGroup` is passed, this is optional but must
                agree with :attr:`~FeatureGroup.version` if supplied.
            keys: Non-empty list of entity rows; each inner list aligns
                column-wise with the FG's shared join keys.
            store_type: Defaults to :attr:`StoreType.ONLINE` (the only path
                supported today). :attr:`StoreType.OFFLINE` raises until
                offline FG reads land.
            request_context: Required when the FG has at least one
                :class:`RealtimeFeatureView` source that declares a
                :class:`RequestSource`; rejected otherwise.
                A ``pandas.DataFrame`` whose columns are a superset of the
                union of every contributing RTFV source's
                ``RequestSource.schema`` field names (case-insensitive);
                extras are dropped with a :class:`UserWarning`. Row count
                must equal ``len(keys)``; row ``i`` is the per-call
                context for the entity row ``keys[i]``.

        Returns:
            ``pandas.DataFrame`` with the join-key columns followed by the
            FG's :attr:`~FeatureGroup.output_columns` (in source order).

        Raises:
            SnowflakeMLException: ``[ValueError]`` if *keys* is empty, the
                version is missing/disagrees, the FG is unregistered, the
                Online Service is not yet RUNNING, or *request_context* is
                missing / extraneous / shape-invalid.
                ``[NotImplementedError]`` if *store_type* is anything other
                than :attr:`StoreType.ONLINE`.  # noqa: DAR402
        """
        return fg_mod.read_feature_group(
            self,
            feature_group,
            version,
            keys=keys,
            store_type=store_type,
            request_context=request_context,
        )

    @dispatch_decorator()
    def list_entities(self) -> DataFrame:
        """
        List all Entities in the FeatureStore.

        Returns:
            Snowpark DataFrame containing the results.

        Example::

            >>> fs = FeatureStore(...)
            >>> e_1 = Entity("my_entity", ['col_1'], desc='My first entity.')
            >>> fs.register_entity(e_1)
            >>> fs.list_entities().show()
            -----------------------------------------------------------
            |"NAME"     |"JOIN_KEYS"  |"DESC"            |"OWNER"     |
            -----------------------------------------------------------
            |MY_ENTITY  |["COL_1"]    |My first entity.  |REGTEST_RL  |
            -----------------------------------------------------------

        """
        prefix_len = len(_ENTITY_TAG_PREFIX) + 1
        return cast(
            DataFrame,
            self._session.sql(
                f"SHOW TAGS LIKE '{_ENTITY_TAG_PREFIX}%' IN SCHEMA {self._config.full_schema_path}"
            ).select(
                F.col('"name"').substr(prefix_len, _ENTITY_NAME_LENGTH_LIMIT).alias("NAME"),
                F.call_builtin("REPLACE", F.col('"allowed_values"'), F.lit(","), F.lit('", "')).alias("JOIN_KEYS"),
                F.col('"comment"').alias("DESC"),
                F.col('"owner"').alias("OWNER"),
            ),
        )

    @dispatch_decorator()
    def get_entity(self, name: str) -> Entity:
        """
        Retrieve previously registered Entity object.

        Args:
            name: Entity name.

        Returns:
            Entity object.

        Raises:
            SnowflakeMLException: [ValueError] Entity is not found.
            SnowflakeMLException: [RuntimeError] Failed to retrieve tag reference information.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> # e_1 is a local object that hasn't registered to Snowflake backend yet.
            >>> e_1 = Entity("my_entity", ['col_1'], desc='My first entity.')
            >>> fs.register_entity(e_1)
            <BLANKLINE>
            >>> # e_2 is a local object that points a backend object in Snowflake.
            >>> e_2 = fs.get_entity("my_entity")
            >>> print(e_2)
            Entity(name=MY_ENTITY, join_keys=['COL_1'], owner=REGTEST_RL, desc=My first entity.)

        """
        name = SqlIdentifier(name)
        try:
            result = (
                self.list_entities()
                .filter(F.col("NAME") == name.resolved())
                .collect(statement_params=self._telemetry_stmp)
            )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to list entities: {e}"),
            ) from e
        if len(result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find Entity with name: {name}."),
            )

        join_keys = [f'"{k}"' for k in json.loads(result[0]["JOIN_KEYS"])]

        return Entity._construct_entity(
            name=SqlIdentifier(result[0]["NAME"], case_sensitive=True).identifier(),
            join_keys=join_keys,
            desc=result[0]["DESC"],
            owner=result[0]["OWNER"],
        )

    @dispatch_decorator()
    def delete_entity(self, name: str) -> None:
        """
        Delete a previously registered Entity.

        Args:
            name: Name of entity to be deleted.

        Raises:
            SnowflakeMLException: [ValueError] Entity with given name not exists.
            SnowflakeMLException: [RuntimeError] Failed to alter schema or drop tag.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> e_1 = Entity("my_entity", ['col_1'], desc='My first entity.')
            >>> fs.register_entity(e_1)
            >>> fs.list_entities().show()
            -----------------------------------------------------------
            |"NAME"     |"JOIN_KEYS"  |"DESC"            |"OWNER"     |
            -----------------------------------------------------------
            |MY_ENTITY  |["COL_1"]    |My first entity.  |REGTEST_RL  |
            -----------------------------------------------------------
            <BLANKLINE>
            >>> fs.delete_entity("my_entity")
            >>> fs.list_entities().show()
            -------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"  |"OWNER"  |
            -------------------------------------------
            |        |             |        |         |
            -------------------------------------------

        """
        name = SqlIdentifier(name)

        if not self._validate_entity_exists(name):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Entity {name} does not exist."),
            )

        active_feature_views = self.list_feature_views(entity_name=name).collect(statement_params=self._telemetry_stmp)

        if len(active_feature_views) > 0:
            active_fvs = [r["NAME"] for r in active_feature_views]
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_DELETE_FAILED,
                original_exception=ValueError(f"Cannot delete Entity {name} due to active FeatureViews: {active_fvs}."),
            )

        tag_name = self._get_fully_qualified_name(self._get_entity_name(name))
        try:
            self._session.sql(f"DROP TAG IF EXISTS {tag_name}").collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to delete entity: {e}."),
            ) from e
        logger.info(f"Deleted Entity {name}.")

    # =========================================================================
    # Stream Source APIs
    # =========================================================================

    @dispatch_decorator()
    def register_stream_source(self, stream_source: StreamSource) -> StreamSource:
        """
        Register a StreamSource in the FeatureStore.

        Args:
            stream_source: StreamSource object to be registered.

        Returns:
            A registered StreamSource object with owner populated.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to save stream source metadata.

        Example::

            >>> fs = FeatureStore(...)
            >>> from snowflake.snowpark.types import (
            ...     StructType, StructField, StringType, FloatType, TimestampType,
            ... )
            >>> txn_source = StreamSource(
            ...     name="transaction_events",
            ...     schema=StructType([
            ...         StructField("user_id", StringType()),
            ...         StructField("amount", FloatType()),
            ...         StructField("event_time", TimestampType()),
            ...     ]),
            ...     desc="Real-time transaction events",
            ... )
            >>> fs.register_stream_source(txn_source)
            >>> fs.list_stream_sources().show()
            -------------------------------------------------------------------------------------...
            |"NAME"                |"SCHEMA"  |"DESC"                    |"OWNER"     |
            -------------------------------------------------------------------------------------...
            |TRANSACTION_EVENTS    |[...]     |Real-time transaction...  |REGTEST_RL  |
            -------------------------------------------------------------------------------------...

        """
        logger.warning("StreamSource is in Public Preview since 1.43.0.")

        if self._metadata_manager.stream_source_exists(stream_source.name.resolved()):
            warnings.warn(
                f"StreamSource {stream_source.name} already exists. Skip registration.",
                stacklevel=2,
                category=UserWarning,
            )
            return self.get_stream_source(stream_source.name)

        owner = self._session.get_current_role()
        stream_source.owner = owner

        metadata = stream_source._to_dict()
        try:
            self._metadata_manager.save_stream_source(
                name=stream_source.name.resolved(),
                metadata=metadata,
            )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to register stream source `{stream_source.name}`: {e}."),
            ) from e

        logger.info(f"Registered StreamSource {stream_source}.")

        return self.get_stream_source(stream_source.name)

    @dispatch_decorator()
    def get_stream_source(self, name: str) -> StreamSource:
        """
        Retrieve a previously registered StreamSource object.

        Args:
            name: StreamSource name.

        Returns:
            StreamSource object.

        Raises:
            SnowflakeMLException: [ValueError] StreamSource is not found.
            SnowflakeMLException: [RuntimeError] Failed to retrieve stream source metadata.

        Example::

            >>> fs = FeatureStore(...)
            >>> ss = fs.get_stream_source("transaction_events")
            >>> print(ss)
            StreamSource(name=TRANSACTION_EVENTS, ...)

        """
        name_id = SqlIdentifier(name)
        try:
            metadata = self._metadata_manager.get_stream_source_metadata(name_id.resolved())
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to retrieve stream source: {e}"),
            ) from e

        if metadata is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find StreamSource with name: {name_id}."),
            )

        return StreamSource._from_dict(metadata)

    @dispatch_decorator()
    def stream_ingest(
        self,
        stream_source: Union[str, StreamSource],
        records: Union[list[dict[str, Any]], dict[str, Any]],
        *,
        timeout_sec: float = 120.0,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> int:
        """Send rows to the Online Service for ingestion.

        Requires the ``SNOWFLAKE_PAT`` environment variable to be set with a valid
        Snowflake Programmatic Access Token.

        ``records`` may be one row as a dict or a non-empty ``list`` of row dicts.
        Each row's keys must match the registered ``StreamSource`` schema exactly
        (no missing or extra columns).

        Args:
            stream_source: Registered stream source name or a ``StreamSource`` instance.
            records: Rows to ingest.
            timeout_sec: Timeout in seconds for the ingest request.
            statement_params: Optional Snowpark statement parameters (for example telemetry).

        Returns:
            Count of records accepted by the Online Service. On partial success, the returned
            count may be less than the number of rows sent.

        Raises:
            SnowflakeMLException: If the Online Service or ingest endpoint is unavailable, ``SNOWFLAKE_PAT`` is
                unset, the request fails, or rows do not match the stream source schema.
        """
        if isinstance(stream_source, StreamSource):
            src = stream_source
            stream_name_wire = stream_source.name.resolved()
        else:
            src = self.get_stream_source(str(stream_source))
            stream_name_wire = src.name.resolved()

        stmp = self._telemetry_stmp if statement_params is None else statement_params
        st = online_service.assert_online_service_running_with_ingest_endpoint(
            self._session,
            self._config.database,
            self._config.schema,
            statement_params=stmp,
        )
        ingest_url = online_service.endpoint_url(st, "ingest", access=self._online_service_access)
        if not ingest_url:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("Online Service returned no ingest endpoint."),
            )
        ingest_base = ingest_url

        expected_cols = _stream_source_schema_field_names(src.schema)
        rows = _normalize_stream_ingest_records(records)
        for i, row in enumerate(rows):
            _validate_stream_ingest_record_keys(expected_cols, row, i)

        return online_service.stream_ingest_records(
            self._session,
            ingest_base,
            stream_name_wire,
            rows,
            timeout_sec=timeout_sec,
            http_client=self._get_or_create_online_http_client(),
        )

    @dispatch_decorator()
    def list_stream_sources(self) -> DataFrame:
        """
        List all StreamSources in the FeatureStore.

        Returns:
            Snowpark DataFrame containing the results with columns:
            NAME, SCHEMA, DESC, OWNER.

        Raises:
            SnowflakeMLException: If the metadata query fails.

        Example::

            >>> fs = FeatureStore(...)
            >>> fs.list_stream_sources().show()
            -------------------------------------------------------------------------------------...
            |"NAME"                |"SCHEMA"  |"DESC"                    |"OWNER"     |
            -------------------------------------------------------------------------------------...
            |TRANSACTION_EVENTS    |[...]     |Real-time transaction...  |REGTEST_RL  |
            -------------------------------------------------------------------------------------...

        """
        if not self._metadata_manager.table_exists():
            return self._session.create_dataframe([], schema=_LIST_STREAM_SOURCE_SCHEMA)

        try:
            return self._session.sql(
                f"""
                    SELECT
                        METADATA:name::VARCHAR AS "NAME",
                        METADATA:schema::VARCHAR AS "SCHEMA",
                        METADATA:desc::VARCHAR AS "DESC",
                        METADATA:owner::VARCHAR AS "OWNER"
                    FROM {self._metadata_manager.table_path}
                    WHERE OBJECT_TYPE = '{MetadataObjectType.STREAM_SOURCE.value}'
                    AND METADATA_TYPE = '{MetadataType.STREAM_SOURCE_CONFIG.value}'
                    ORDER BY "NAME"
                    """
            )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to list stream sources: {e}"),
            ) from e

    @dispatch_decorator()
    def delete_stream_source(self, name: str) -> None:
        """
        Delete a previously registered StreamSource.

        The StreamSource can only be deleted if no active FeatureViews reference it.

        Args:
            name: Name of the StreamSource to be deleted.

        Raises:
            SnowflakeMLException: [ValueError] StreamSource with given name not found.
            SnowflakeMLException: [ValueError] Cannot delete due to active references.
            SnowflakeMLException: [RuntimeError] Failed to delete stream source.

        Example::

            >>> fs = FeatureStore(...)
            >>> fs.delete_stream_source("transaction_events")

        """
        name_id = SqlIdentifier(name)

        if not self._metadata_manager.stream_source_exists(name_id.resolved()):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"StreamSource {name_id} does not exist."),
            )

        # Check for active feature view references
        ref_count = self._metadata_manager.get_stream_source_ref_count(name_id.resolved())
        if ref_count > 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_DELETE_FAILED,
                original_exception=ValueError(
                    f"Cannot delete StreamSource {name_id}: has {ref_count} active reference(s)."
                ),
            )

        try:
            self._metadata_manager.delete_stream_source_metadata(name_id.resolved())
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to delete stream source: {e}."),
            ) from e
        logger.info(f"Deleted StreamSource {name_id}.")

    @dispatch_decorator()
    def update_stream_source(self, name: str, *, desc: Optional[str] = None) -> Optional[StreamSource]:
        """Update a registered StreamSource description.

        Args:
            name: Name of the StreamSource to update.
            desc: New description to apply. Default to None (no change).

        Returns:
            Updated StreamSource object, or None if the StreamSource doesn't exist.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to update stream source.

        Example::

            >>> fs = FeatureStore(...)
            >>> fs.update_stream_source("transaction_events", desc="Updated description")
            >>> fs.get_stream_source("transaction_events").desc
            'Updated description'

        """
        name_id = SqlIdentifier(name)

        if not self._metadata_manager.stream_source_exists(name_id.resolved()):
            warnings.warn(
                f"StreamSource {name_id} does not exist.",
                stacklevel=2,
                category=UserWarning,
            )
            return None

        if desc is not None:
            try:
                self._metadata_manager.update_stream_source_desc(name_id.resolved(), desc)
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"Failed to update stream source `{name_id}`: {e}."),
                ) from e

        logger.info(f"Successfully updated StreamSource {name_id}.")
        return self.get_stream_source(name_id)

    @dispatch_decorator()
    def retrieve_feature_values(
        self,
        spine_df: DataFrame,
        features: Union[list[Union[FeatureView, FeatureViewSlice]], list[str]],
        *,
        spine_timestamp_col: Optional[str] = None,
        exclude_columns: Optional[list[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        join_method: Literal["sequential", "cte"] = "sequential",
    ) -> DataFrame:
        """
        Enrich spine dataframe with feature values for inference.

        Uses the latest entity mappings for rollup feature views. For point-in-time
        correct entity mappings (e.g., during training or backtesting with temporal
        rollup FVs), use ``generate_training_set`` instead.

        If spine_timestamp_col is specified, point-in-time feature values will be fetched.

        Args:
            spine_df: Snowpark DataFrame to join features into.
            features: List of features to join into the spine_df. Can be a list of FeatureView or FeatureViewSlice,
                or a list of serialized feature objects from Dataset.
            spine_timestamp_col: Timestamp column in spine_df for point-in-time feature value lookup.
            exclude_columns: Column names to exclude from the result dataframe.
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.
            auto_prefix: If True, automatically prefix all feature columns with
                '{feature_view_name}_{version}_' to avoid name collisions.
                Default False. Use FeatureView.with_name() for custom prefixes.
            join_method: Method for feature joins. "sequential" for layer-by-layer joins (default),
                "cte" for CTE method. (Internal use only - subject to change)

        Returns:
            Snowpark DataFrame containing the joined results.

        Raises:
            ValueError: if features is empty.

        Example::

            >>> fs = FeatureStore(...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view('my_fv', 'v1')
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> fs.retrieve_feature_values(spine_df, [fv]).show()
            --------------------
            |"END_STATION_ID"  |
            --------------------
            |505               |
            |347               |
            |466               |
            --------------------

        """
        if spine_timestamp_col is not None:
            spine_timestamp_col = SqlIdentifier(spine_timestamp_col)

        if len(features) == 0:
            raise ValueError("features cannot be empty")
        if isinstance(features[0], str):
            features = self._load_serialized_feature_views(cast(list[str], features))

        df, _ = self._join_features(
            spine_df,
            cast(list[Union[FeatureView, FeatureViewSlice]], features),
            spine_timestamp_col,
            include_feature_view_timestamp_col,
            auto_prefix,
            join_method,
        )

        if exclude_columns is not None:
            df = self._exclude_columns(df, exclude_columns)

        return df

    @overload
    def generate_training_set(
        self,
        spine_df: DataFrame,
        features: list[Union[FeatureView, FeatureViewSlice]],
        *,
        save_as: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        join_method: Literal["sequential", "cte"] = "sequential",
    ) -> DataFrame:
        ...

    @overload
    def generate_training_set(
        self,
        spine_df: DataFrame,
        *,
        feature_group: Union[FeatureGroup, tuple[str, str]],
        save_as: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[list[str]] = None,
    ) -> DataFrame:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def generate_training_set(
        self,
        spine_df: DataFrame,
        features: Optional[list[Union[FeatureView, FeatureViewSlice]]] = None,
        *,
        feature_group: Optional[Union[FeatureGroup, tuple[str, str]]] = None,
        save_as: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        join_method: Literal["sequential", "cte"] = "sequential",
    ) -> DataFrame:
        """
        Generate a training set from a spine DataFrame and either a list of FeatureViews or a FeatureGroup.

        For rollup feature views with temporal entity mappings (``mapping_valid_from_col``
        and ``mapping_valid_to_col``), this method uses point-in-time correct entity
        resolution via a range JOIN with bounded validity windows ``[valid_from, valid_to)``,
        supporting 1:N mappings (one child entity to multiple parent entities simultaneously).
        For inference with latest mappings, use ``retrieve_feature_values`` instead.

        Pass exactly one of ``features`` or ``feature_group``. With ``feature_group``,
        the FG-owned parameters (``auto_prefix``, ``join_method``, ``exclude_columns``,
        ``include_feature_view_timestamp_col``) are predetermined by the FG and
        must not be set explicitly.

        Realtime feature views: when ``features`` (or the resolved ``feature_group``)
        contains one or more realtime feature views, the spine DataFrame must carry
        a column for every field declared on each realtime feature view's
        ``RequestSource.schema``. Those columns supply the per-row request context
        the realtime ``compute_fn`` would receive at inference time. Each realtime
        feature view's ``compute_fn`` is evaluated against its upstream feature
        view rows joined from the offline backing -- no online service is called.
        ``FeatureViewSlice`` of a realtime feature view is supported and behaves
        as a column projection over ``compute_fn``'s declared output. Spine columns
        that are not declared on a realtime feature view's request source are
        ignored without warning (the spine legitimately carries labels, timestamps,
        and other non-feature columns).

        Result is materialized to a Snowflake Table if ``save_as`` is specified.

        Args:
            spine_df: Snowpark DataFrame to join features into.
            features: A list of FeatureView or FeatureViewSlice which contains features to be
                joined. Mutually exclusive with ``feature_group``.
            feature_group: A registered :class:`FeatureGroup` or a ``(name, version)`` tuple.
                Mutually exclusive with ``features``.
            save_as: If specified, a new table containing the produced result will be created. Name can be a fully
                qualified name or an unqualified name. If unqualified, defaults to the Feature Store database and schema
            spine_timestamp_col: Name of timestamp column in spine_df that will be used to join
                time-series features. If spine_timestamp_col is not none, the input features also must have
                timestamp_col.
            spine_label_cols: Name of column(s) in spine_df that contains labels.
            exclude_columns: Name of column(s) to exclude from the resulting training set.
                Only valid with the ``features`` overload (FGs return a predetermined column set).
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.
                Only valid with the ``features`` overload.
            auto_prefix: If True, automatically prefix all feature columns with
                '{feature_view_name}_{version}_'. Default False.
                Use FeatureView.with_name() for custom prefixes.
                Only valid with the ``features`` overload (FGs use ``FeatureGroup.auto_prefix``).
            join_method: Method for feature joins. "sequential" for layer-by-layer joins (default),
                "cte" for CTE method. (Internal use only - subject to change)
                Only valid with the ``features`` overload (FGs always use "cte").

        Returns:
            Returns a Snowpark DataFrame representing the training set.

        Raises:
            SnowflakeMLException: [INVALID_ARGUMENT] Neither or both of ``features`` /
                ``feature_group`` were provided, or an FG-incompatible parameter
                (``exclude_columns``, ``include_feature_view_timestamp_col``,
                ``auto_prefix``, non-default ``join_method``) was set with a FeatureGroup.
            SnowflakeMLException: [RuntimeError] Materialized table name already exists
            SnowflakeMLException: [RuntimeError] Failed to create materialized table.

        Example::

            >>> fs = FeatureStore(session, ...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view("MY_FV", "1")
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> training_set = fs.generate_training_set(
            ...     spine_df,
            ...     [fv],
            ...     save_as="my_training_set",
            ... )
            >>> print(type(training_set))
            <class 'snowflake.snowpark.table.Table'>
            <BLANKLINE>
            >>> print(training_set.queries)
            {'queries': ['SELECT  *  FROM (my_training_set)'], 'post_actions': []}

        """
        if (features is None) == (feature_group is None):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "generate_training_set requires exactly one of `features` or `feature_group`."
                ),
            )

        if feature_group is not None:
            features, auto_prefix, join_method = fg_mod.prepare_training_set_args(
                self,
                feature_group=feature_group,
                exclude_columns=exclude_columns,
                include_feature_view_timestamp_col=include_feature_view_timestamp_col,
                auto_prefix=auto_prefix,
                join_method=join_method,
            )

        features = cast(list[Union[FeatureView, FeatureViewSlice]], features)

        if spine_timestamp_col is not None:
            spine_timestamp_col = SqlIdentifier(spine_timestamp_col)
        if spine_label_cols is not None:
            spine_label_cols = to_sql_identifiers(spine_label_cols)  # type: ignore[assignment]

        direct_refs, rtfv_refs = rtfv_dataset.partition_features(features)

        if rtfv_refs:
            rtfv_dataset.validate_rtfv_dataset_inputs(features, list(spine_df.columns))
            augmented_spine = rtfv_dataset.attach_synthetic_row_id(spine_df)

            if direct_refs:
                user_visible_df, _join_keys = self._join_features(
                    augmented_spine,
                    direct_refs,
                    spine_timestamp_col,
                    include_feature_view_timestamp_col,
                    auto_prefix,
                    join_method,
                    is_training=True,
                )
            else:
                user_visible_df = augmented_spine

            result_df = rtfv_dataset.apply_rtfvs(
                self,
                user_visible_df,
                rtfv_refs_in_order=rtfv_refs,
                original_features=features,
                augmented_spine=augmented_spine,
                spine_timestamp_col=str(spine_timestamp_col) if spine_timestamp_col is not None else None,
                auto_prefix=auto_prefix,
            )
        else:
            result_df, join_keys = self._join_features(
                spine_df,
                features,
                spine_timestamp_col,
                include_feature_view_timestamp_col,
                auto_prefix,
                join_method,
                is_training=True,
            )

        if exclude_columns is not None:
            result_df = self._exclude_columns(result_df, exclude_columns)

        if save_as is not None:
            try:
                save_as = self._get_fully_qualified_name(save_as)
                result_df.write.mode("errorifexists").save_as_table(save_as, statement_params=self._telemetry_stmp)

                # Add tag
                task_obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.TRAINING_DATA, snowml_version.VERSION)
                self._session.sql(
                    f"""
                    ALTER TABLE {save_as}
                    SET TAG {self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}='{task_obj_info.to_json()}'
                    """
                ).collect(statement_params=self._telemetry_stmp)

                return self._session.table(save_as)

            except SnowparkSQLException as e:
                if e.sql_error_code == sql_error_codes.OBJECT_ALREADY_EXISTS:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.OBJECT_ALREADY_EXISTS,
                        original_exception=RuntimeError(str(e)),
                    ) from e
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"An error occurred during training set materialization: {e}."),
                ) from e
        return result_df

    @overload
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: list[Union[FeatureView, FeatureViewSlice]],
        *,
        version: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        desc: str = "",
        output_type: Literal["dataset"] = "dataset",
        join_method: Literal["sequential", "cte"] = "sequential",
    ) -> dataset.Dataset:
        ...

    @overload
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: list[Union[FeatureView, FeatureViewSlice]],
        *,
        output_type: Literal["table"],
        version: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        desc: str = "",
        join_method: Literal["sequential", "cte"] = "sequential",
    ) -> DataFrame:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: list[Union[FeatureView, FeatureViewSlice]],
        *,
        version: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[list[str]] = None,
        exclude_columns: Optional[list[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        desc: str = "",
        output_type: Literal["dataset", "table"] = "dataset",
        join_method: Literal["sequential", "cte"] = "sequential",
    ) -> Union[dataset.Dataset, DataFrame]:
        """
        Generate dataset by given source table and feature views.

        Args:
            name: The name of the Dataset to be generated. Datasets are uniquely identified within a schema
                by their name and version.
            spine_df: Snowpark DataFrame to join features into.
            features: A list of FeatureView or FeatureViewSlice which contains features to be joined.
            version: The version of the Dataset to be generated. If none specified, the current timestamp
                will be used instead.
            spine_timestamp_col: Name of timestamp column in spine_df that will be used to join
                time-series features. If spine_timestamp_col is not none, the input features also must have
                timestamp_col.
            spine_label_cols: Name of column(s) in spine_df that contains labels.
            exclude_columns: Name of column(s) to exclude from the resulting training set.
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.
            auto_prefix: If True, automatically prefix all feature columns with
                '{feature_view_name}_{version}_'. Default False.
                Use FeatureView.with_name() for custom prefixes.
            desc: A description about this dataset.
            output_type: (Deprecated) The type of Snowflake storage to use for the generated training data.
            join_method: Method for feature joins. "sequential" for layer-by-layer joins (default),
                "cte" for CTE method. (Internal use only - subject to change)

        Returns:
            If output_type is "dataset" (default), returns a Dataset object.
            If output_type is "table", returns a Snowpark DataFrame representing the table.

        Raises:
            SnowflakeMLException: [ValueError] Invalid output_type specified.
            SnowflakeMLException: [RuntimeError] Dataset name/version already exists.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(session, ...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view("MY_FV", "1")
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> my_dataset = fs.generate_dataset(
            ...     "my_dataset"
            ...     spine_df,
            ...     [fv],
            ... )
            >>> # Current timestamp will be used as default version name.
            >>> # You can explicitly overwrite by setting a version.
            >>> my_dataset.list_versions()
            ['2024_07_12_11_26_22']
            <BLANKLINE>
            >>> my_dataset.read.to_snowpark_dataframe().show(n=3)
            -------------------------------------------------------
            |"QUALITY"  |"FIXED_ACIDITY"     |"VOLATILE_ACIDITY"  |
            -------------------------------------------------------
            |3          |11.600000381469727  |0.5799999833106995  |
            |3          |8.300000190734863   |1.0199999809265137  |
            |3          |7.400000095367432   |1.184999942779541   |
            -------------------------------------------------------

        """
        if output_type not in {"table", "dataset"}:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid output_type: {output_type}."),
            )

        # Convert name to fully qualified name if not already fully qualified
        name = self._get_fully_qualified_name(name)
        version = version or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        fs_meta = FeatureStoreMetadata(
            spine_query=spine_df.queries["queries"][-1],
            compact_feature_views=[fv._get_compact_repr().to_json() for fv in features],
            spine_timestamp_col=spine_timestamp_col,
        )

        # Only set a save_as name if output_type is table
        table_name = f"{name}_{version}" if output_type == "table" else None
        result_df = self.generate_training_set(
            spine_df,
            features,
            spine_timestamp_col=spine_timestamp_col,
            spine_label_cols=spine_label_cols,
            exclude_columns=exclude_columns,
            include_feature_view_timestamp_col=include_feature_view_timestamp_col,
            auto_prefix=auto_prefix,
            save_as=table_name,
            join_method=join_method,
        )
        if output_type == "table":
            warnings.warn(
                "Generating a table from generate_dataset() is deprecated and will be removed in a future release,"
                " use generate_training_set() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return result_df

        try:
            assert output_type == "dataset"
            if not self._is_dataset_enabled():
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.SNOWML_CREATE_FAILED,
                    original_exception=RuntimeError(
                        "Dataset is not enabled in your account. Ask your account admin to set"
                        " FEATURE_DATASET=ENABLED or use generate_training_set() instead"
                        " to generate the data as a Snowflake Table."
                    ),
                )

            # Cache the result to a temporary table before creating the dataset
            # to ensure single query evaluation:
            has_tiled_fv = any(fg_mod.unwrap_fv(fv).is_tiled for fv in features)
            has_rtfv = any(fg_mod.unwrap_fv(fv).is_realtime_feature_view for fv in features)
            if has_tiled_fv or has_rtfv:
                result_df = result_df.cache_result()

            # TODO: Add feature store tag once Dataset (version) supports tags
            ds: dataset.Dataset = dataset.create_from_dataframe(
                self._session,
                name,
                version,
                input_dataframe=result_df,
                exclude_cols=([spine_timestamp_col] if spine_timestamp_col is not None else []),
                label_cols=spine_label_cols,
                properties=fs_meta,
                comment=desc,
            )
            return ds

        except dataset_errors.DatasetExistError as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.OBJECT_ALREADY_EXISTS,
                original_exception=RuntimeError(str(e)),
            ) from e
        except SnowparkSQLException as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"An error occurred during dataset generation: {e}."),
            ) from e

    @dispatch_decorator()
    def load_feature_views_from_dataset(self, ds: dataset.Dataset) -> list[Union[FeatureView, FeatureViewSlice]]:
        """
        Retrieve FeatureViews used during Dataset construction.

        Args:
            ds: Dataset object created from feature store.

        Returns:
            List of FeatureViews used during Dataset construction.

        Raises:
            ValueError: if dataset object is not generated from feature store.

        Example::

            >>> fs = FeatureStore(session, ...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view("MY_FV", "1.0")
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> my_dataset = fs.generate_dataset(
            ...     "my_dataset"
            ...     spine_df,
            ...     [fv],
            ... )
            >>> fvs = fs.load_feature_views_from_dataset(my_dataset)
            >>> print(len(fvs))
            1
            <BLANKLINE>
            >>> print(type(fvs[0]))
            <class 'snowflake.ml.feature_store.feature_view.FeatureView'>
            <BLANKLINE>
            >>> print(fvs[0].name)
            MY_FV
            <BLANKLINE>
            >>> print(fvs[0].version)
            1.0

        """
        assert ds.selected_version is not None
        source_meta = ds.selected_version._get_metadata()
        if (
            source_meta is None
            or not isinstance(source_meta.properties, FeatureStoreMetadata)
            or (
                source_meta.properties.serialized_feature_views is None
                and source_meta.properties.compact_feature_views is None
            )
        ):
            raise ValueError(f"Dataset {ds} does not contain valid feature view information.")

        properties = source_meta.properties
        if properties.serialized_feature_views:
            return self._load_serialized_feature_views(properties.serialized_feature_views)
        else:
            return self._load_compact_feature_views(properties.compact_feature_views)  # type: ignore[arg-type]

    def _rollback_created_resources(self, created_resources: list[tuple[_FeatureStoreObjTypes, str]]) -> None:
        """Rollback created resources in reverse order.

        Args:
            created_resources: List of (resource_type, resource_name) tuples to clean up
        """
        for resource_type, resource_name in reversed(created_resources):
            try:
                if resource_type == _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW:
                    self._session.sql(f"DROP DYNAMIC TABLE IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                elif resource_type == _FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW:
                    self._session.sql(f"DROP VIEW IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                elif resource_type == _FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK:
                    # Suspend before drop: streaming-FV root tasks may be
                    # mid-fire by the time rollback runs. SUSPEND is
                    # idempotent (and a no-op for already-suspended tiled-FV
                    # tasks). Suspend failures are logged so DROP still runs.
                    try:
                        self._session.sql(f"ALTER TASK IF EXISTS {resource_name} SUSPEND").collect(
                            statement_params=self._telemetry_stmp
                        )
                    except Exception as suspend_error:
                        logger.warning(
                            f"Rollback: failed to suspend task {resource_name} before drop "
                            f"(continuing with DROP anyway): {suspend_error}"
                        )
                    self._session.sql(f"DROP TASK IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                elif resource_type == _FeatureStoreObjTypes.FEATURE_VIEW_BACKFILL_PROC:
                    from snowflake.ml.feature_store.streaming_registration import (
                        _get_backfill_proc_signature,
                    )

                    self._session.sql(
                        f"DROP PROCEDURE IF EXISTS {resource_name}{_get_backfill_proc_signature()}"
                    ).collect(statement_params=self._telemetry_stmp)
                elif resource_type == _FeatureStoreObjTypes.FEATURE_VIEW_BACKFILL_UDTF:
                    # ``resource_name`` already carries the UDTF arg-type
                    # signature appended at registration time.
                    self._session.sql(f"DROP FUNCTION IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                elif resource_type == _FeatureStoreObjTypes.ONLINE_FEATURE_TABLE:
                    self._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                elif resource_type == _FeatureStoreObjTypes.UDF_TRANSFORMED_TABLE:
                    self._session.sql(f"DROP TABLE IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                elif resource_type == _FeatureStoreObjTypes.SNAPSHOT_TABLE:
                    self._session.sql(f"DROP TABLE IF EXISTS {resource_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                logger.info(f"Rollback: Successfully dropped {resource_type.value} {resource_name}")
            except Exception as rollback_error:
                # Log but don't fail the rollback process
                logger.warning(f"Rollback: Failed to drop {resource_type.value} {resource_name}: {rollback_error}")

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def _create_updated_feature_view(
        self, base_fv: FeatureView, online_config: Optional[fv_mod.OnlineConfig] = None
    ) -> FeatureView:
        """Return a copy of ``base_fv`` with a new online configuration applied.

        A shallow copy preserves the full feature-view identity (tiled / streaming / realtime /
        append-only / source-refs) automatically, so only the online config changes. Hand-copying
        individual fields here would silently drop any field not listed and drift as new fields are
        added (the same class of bug this avoids). Mirrors the ``copy() + _online_config`` pattern
        used elsewhere in ``update_feature_view``.

        Args:
            base_fv: The feature view to copy.
            online_config: The online configuration to apply to the copy.

        Returns:
            A shallow copy of ``base_fv`` with ``online_config`` applied.
        """
        fv = base_fv.copy()
        fv._online_config = online_config
        return fv

    def _build_offline_update_queries(
        self,
        feature_view: FeatureView,
        refresh_freq: Optional[str],
        warehouse: Optional[str],
        initialization_warehouse: Optional[str],
        desc: str,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Build offline update operations and their rollback operations.

        For CRON-based refresh, the DT must keep ``TARGET_LAG = 'DOWNSTREAM'``
        and the actual cron schedule lives on a companion Task — Dynamic Tables
        do not accept cron expressions in ``TARGET_LAG``. Duration-based refresh
        sets ``TARGET_LAG`` directly on the DT and there is no Task.

        The presence/absence of the companion Task is what disambiguates a
        user-set ``refresh_freq='DOWNSTREAM'`` from a cron-driven FV (whose DT
        also carries ``TARGET_LAG='DOWNSTREAM'``) when reading state back via
        ``get_feature_view``. Maintain that invariant across all four
        ``(old_kind, new_kind)`` transitions:

            duration ↔ duration : DT-only ALTER; no Task involvement.
            duration → cron     : ALTER DT to DOWNSTREAM, then create the Task.
            cron     → duration : ALTER DT to the new lag, then drop the Task.
            cron     ↔ cron     : ALTER DT, then SUSPEND/SET SCHEDULE/RESUME the Task.

        ("duration" above includes a literal ``"DOWNSTREAM"`` value.)

        Args:
            feature_view: The currently registered ``FeatureView`` whose
                offline state is being mutated; supplies the existing
                refresh_freq/warehouse/desc used to construct the rollback.
            refresh_freq: New refresh frequency requested by the caller, or
                ``None`` to keep the existing value.
            warehouse: New warehouse identifier requested by the caller, or
                ``None`` to keep the existing value.
            initialization_warehouse: New initialization warehouse. ``_KEEP_CURRENT``
                leaves it unchanged; ``None`` clears it (``UNSET``); any other
                value sets it (``SET``).
            desc: New description to set on the DT/view.

        Returns:
            A tuple of ``(operations, rollback_operations)`` where each is a
            list of ``(op_type, sql)`` tuples consumable by
            ``_execute_atomic_operations``.
        """
        fqn = feature_view.fully_qualified_name()

        if feature_view.status == FeatureViewStatus.STATIC:
            return [("OFFLINE_UPDATE", f"ALTER VIEW {fqn} SET COMMENT = {_sql_string_literal(desc)}")], []

        # Managed (Dynamic-Table-backed) FVs always have a refresh_freq — the
        # only refresh_freq=None case is an external (View-backed) FV, which
        # is STATIC and handled above.
        assert feature_view.refresh_freq is not None, "Managed FV must have a refresh_freq"
        new_warehouse = SqlIdentifier(warehouse) if warehouse else feature_view.warehouse
        old_warehouse = feature_view.warehouse
        effective_freq: str = refresh_freq if refresh_freq is not None else feature_view.refresh_freq
        new_is_cron = feature_view_refresh_freq._is_cron_refresh_freq(effective_freq)
        old_freq: str = feature_view.refresh_freq
        old_is_cron = feature_view_refresh_freq._is_cron_refresh_freq(old_freq)

        new_target_lag = "DOWNSTREAM" if new_is_cron else effective_freq
        old_target_lag = "DOWNSTREAM" if old_is_cron else old_freq

        def alter_dt(op_type: str, *, target_lag: str, wh: Optional[SqlIdentifier], comment: str) -> tuple[str, str]:
            return (
                op_type,
                f"ALTER DYNAMIC TABLE {fqn} SET TARGET_LAG = '{target_lag}'"
                f" WAREHOUSE = {wh} COMMENT = {_sql_string_literal(comment)}",
            )

        def create_task_ops(op_type: str, *, cron_expr: str, wh: Optional[SqlIdentifier]) -> list[tuple[str, str]]:
            # CREATE OR REPLACE keeps the operation idempotent even if a stale
            # task with the same name happens to exist (e.g. partial prior failure).
            task_obj_info = _FeatureStoreObjInfo(
                _FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK, snowml_version.VERSION
            )
            tag_path = self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)
            return [
                (
                    op_type,
                    f"CREATE OR REPLACE TASK {fqn} WAREHOUSE = {wh} "
                    f"SCHEDULE = 'USING CRON {cron_expr}' "
                    f"AS ALTER DYNAMIC TABLE {fqn} REFRESH",
                ),
                (
                    op_type,
                    f"ALTER TASK {fqn} SET TAG {tag_path}='{task_obj_info.to_json()}'",
                ),
                (op_type, f"ALTER TASK {fqn} RESUME"),
            ]

        def drop_task_op(op_type: str) -> tuple[str, str]:
            return (op_type, f"DROP TASK IF EXISTS {fqn}")

        def alter_task_schedule_ops(op_type: str, *, cron_expr: str) -> list[tuple[str, str]]:
            # ALTER TASK SET SCHEDULE requires the task to be suspended.
            return [
                (op_type, f"ALTER TASK {fqn} SUSPEND"),
                (op_type, f"ALTER TASK {fqn} SET SCHEDULE = 'USING CRON {cron_expr}'"),
                (op_type, f"ALTER TASK {fqn} RESUME"),
            ]

        operations: list[tuple[str, str]] = [
            alter_dt(
                "OFFLINE_UPDATE",
                target_lag=new_target_lag,
                wh=new_warehouse,
                comment=desc,
            )
        ]
        if old_is_cron and new_is_cron:
            operations.extend(alter_task_schedule_ops("OFFLINE_UPDATE", cron_expr=effective_freq))
        elif old_is_cron and not new_is_cron:
            operations.append(drop_task_op("OFFLINE_UPDATE"))
        elif not old_is_cron and new_is_cron:
            operations.extend(create_task_ops("OFFLINE_UPDATE", cron_expr=effective_freq, wh=new_warehouse))

        rollback_ops: list[tuple[str, str]] = [
            alter_dt(
                "OFFLINE_ROLLBACK",
                target_lag=old_target_lag,
                wh=old_warehouse,
                comment=feature_view.desc,
            )
        ]
        if old_is_cron and new_is_cron:
            rollback_ops.extend(alter_task_schedule_ops("OFFLINE_ROLLBACK", cron_expr=old_freq))
        elif old_is_cron and not new_is_cron:
            rollback_ops.extend(create_task_ops("OFFLINE_ROLLBACK", cron_expr=old_freq, wh=old_warehouse))
        elif not old_is_cron and new_is_cron:
            rollback_ops.append(drop_task_op("OFFLINE_ROLLBACK"))

        # INITIALIZATION_WAREHOUSE is a standalone SET/UNSET. _KEEP_CURRENT means
        # the caller didn't pass the argument; an explicit None clears it.
        if initialization_warehouse is not _KEEP_CURRENT:
            old_init_wh = feature_view.initialization_warehouse

            def set_or_unset_init_wh(op_type: str, value: Optional[SqlIdentifier]) -> tuple[str, str]:
                if value is None:
                    return (op_type, f"ALTER DYNAMIC TABLE {fqn} UNSET INITIALIZATION_WAREHOUSE")
                return (op_type, f"ALTER DYNAMIC TABLE {fqn} SET INITIALIZATION_WAREHOUSE = {value}")

            new_init_wh = SqlIdentifier(initialization_warehouse) if initialization_warehouse is not None else None
            operations.append(set_or_unset_init_wh("OFFLINE_UPDATE", new_init_wh))
            rollback_ops.append(set_or_unset_init_wh("OFFLINE_ROLLBACK", old_init_wh))

        return operations, rollback_ops

    @dataclass(frozen=True)
    class _OnlineUpdateStrategy:
        """Encapsulates online update operations and their rollbacks."""

        operations: list[tuple[str, Union[str, FeatureView]]]
        rollback_operations: list[tuple[str, Union[str, FeatureView]]]
        final_config: Optional[fv_mod.OnlineConfig]

    def _plan_online_update(
        self, feature_view: FeatureView, online_config: Optional[fv_mod.OnlineConfig]
    ) -> _OnlineUpdateStrategy:
        """Plan online update operations based on current state and target config.

        Handles three cases:
        - enable is None: Preserve current online state, only update if currently online
        - enable is True: Enable online storage (create if needed, update if exists)
        - enable is False: Disable online storage (drop if exists)

        Args:
            feature_view: The FeatureView object to check current online state.
            online_config: The OnlineConfig with target enable and lag settings.

        Returns:
            _OnlineUpdateStrategy containing operations and their rollbacks.
        """
        if online_config is None:
            return self._OnlineUpdateStrategy([], [], None)

        current_online = feature_view.online
        target_online = online_config.enable

        # Case 1: enable is None - preserve current online state, only update if currently online
        if target_online is None:
            if current_online and (online_config.target_lag is not None):
                # Online is currently enabled and user wants to update lag
                return self._plan_online_update_existing(feature_view, online_config)
            else:
                # No online changes needed (either not online, or lag not specified)
                return self._OnlineUpdateStrategy([], [], None)

        # Case 2: Enable online (create table)
        if target_online and not current_online:
            return self._plan_online_enable(feature_view, online_config)

        # Case 3: Disable online (drop table)
        elif not target_online and current_online:
            return self._plan_online_disable(feature_view)

        # Case 4: Update existing online table
        elif target_online and current_online:
            return self._plan_online_update_existing(feature_view, online_config)

        # Case 5: No change needed
        else:
            return self._OnlineUpdateStrategy([], [], online_config)

    def _plan_online_enable(
        self, feature_view: FeatureView, online_config: fv_mod.OnlineConfig
    ) -> _OnlineUpdateStrategy:
        """Plan operations to enable online storage."""
        # Get default target_lag from existing config or use default
        default_target_lag = (
            feature_view.online_config.target_lag
            if feature_view.online_config and feature_view.online_config.target_lag
            else fv_mod._BATCH_OFT_TARGET_LAG
        )
        final_config = fv_mod.OnlineConfig(
            enable=True,
            target_lag=(online_config.target_lag if online_config.target_lag is not None else default_target_lag),
            store_type=online_config.store_type,
        )

        temp_fv = self._create_updated_feature_view(feature_view, final_config)

        operations: list[tuple[str, Union[str, FeatureView]]] = [("CREATE_ONLINE", temp_fv)]
        rollback_ops: list[tuple[str, Union[str, FeatureView]]] = [
            ("DELETE_ONLINE", temp_fv.fully_qualified_online_table_name())
        ]

        return self._OnlineUpdateStrategy(operations, rollback_ops, final_config)

    def _plan_online_disable(self, feature_view: FeatureView) -> _OnlineUpdateStrategy:
        """Plan operations to disable online storage."""
        table_name = feature_view.fully_qualified_online_table_name()

        operations: list[tuple[str, Union[str, FeatureView]]] = [("DELETE_ONLINE", table_name)]
        rollback_ops: list[tuple[str, Union[str, FeatureView]]] = [
            (
                "CREATE_ONLINE",
                self._create_updated_feature_view(feature_view, feature_view.online_config),
            )
        ]

        # Create disabled config to properly represent the new state
        disabled_config = fv_mod.OnlineConfig(enable=False)

        return self._OnlineUpdateStrategy(operations, rollback_ops, disabled_config)

    def _plan_online_update_existing(
        self, feature_view: FeatureView, online_config: fv_mod.OnlineConfig
    ) -> _OnlineUpdateStrategy:
        """Plan operations to update existing online table configuration."""
        existing_config = feature_view.online_config or fv_mod.OnlineConfig(
            enable=True, target_lag=fv_mod._BATCH_OFT_TARGET_LAG
        )
        if online_config.target_lag is None or online_config.target_lag == existing_config.target_lag:
            return self._OnlineUpdateStrategy([], [], existing_config)

        table_name = feature_view.fully_qualified_online_table_name()
        update_query = f"ALTER ONLINE FEATURE TABLE {table_name} SET TARGET_LAG = '{online_config.target_lag}'"
        rollback_query = f"ALTER ONLINE FEATURE TABLE {table_name} SET TARGET_LAG = '{existing_config.target_lag}'"

        operations: list[tuple[str, Union[str, FeatureView]]] = [("UPDATE_ONLINE", update_query)]
        rollback_ops: list[tuple[str, Union[str, FeatureView]]] = [("UPDATE_ONLINE", rollback_query)]

        final_config = fv_mod.OnlineConfig(
            enable=True,
            target_lag=online_config.target_lag,
        )

        return self._OnlineUpdateStrategy(operations, rollback_ops, final_config)

    def _plan_feature_view_update_operations(
        self,
        feature_view: FeatureView,
        refresh_freq: Optional[str],
        warehouse: Optional[str],
        initialization_warehouse: Optional[str],
        desc: str,
        online_config: Optional[fv_mod.OnlineConfig],
    ) -> tuple[list[tuple[str, Union[str, FeatureView]]], list[tuple[str, Union[str, FeatureView]]],]:
        """Plan all update operations and their rollbacks."""
        operations: list[tuple[str, Union[str, FeatureView]]] = []
        rollback_operations: list[tuple[str, Union[str, FeatureView]]] = []

        # Plan offline updates
        offline_ops, offline_rollback_ops = self._build_offline_update_queries(
            feature_view, refresh_freq, warehouse, initialization_warehouse, desc
        )
        operations.extend(offline_ops)
        rollback_operations.extend(offline_rollback_ops)

        # Plan online updates
        online_strategy = self._plan_online_update(feature_view, online_config)
        operations.extend(online_strategy.operations)
        rollback_operations.extend(online_strategy.rollback_operations)

        return operations, rollback_operations

    def _plan_feature_view_status_operations(
        self, feature_view: FeatureView, operation: str
    ) -> tuple[list[tuple[str, Union[str, FeatureView]]], list[tuple[str, Union[str, FeatureView]]],]:
        """Plan atomic operations for suspend/resume operations.

        Args:
            feature_view: The feature view to operate on
            operation: "SUSPEND" or "RESUME"

        Returns:
            Tuple of (operations, rollback_operations)
        """
        assert operation in [
            "SUSPEND",
            "RESUME",
        ], f"Operation {operation} not supported"

        operations: list[tuple[str, Union[str, FeatureView]]] = []
        rollback_operations: list[tuple[str, Union[str, FeatureView]]] = []

        fully_qualified_name = feature_view.fully_qualified_name()

        # Define the reverse operation for rollback
        reverse_operation = "RESUME" if operation == "SUSPEND" else "SUSPEND"

        # Plan offline operations (dynamic table + task)
        offline_sql = f"ALTER DYNAMIC TABLE {fully_qualified_name} {operation}"
        offline_rollback_sql = f"ALTER DYNAMIC TABLE {fully_qualified_name} {reverse_operation}"

        task_sql = f"ALTER TASK IF EXISTS {fully_qualified_name} {operation}"
        task_rollback_sql = f"ALTER TASK IF EXISTS {fully_qualified_name} {reverse_operation}"

        operations.append(("OFFLINE_STATUS", offline_sql))
        operations.append(("TASK_STATUS", task_sql))

        # Rollback operations (in reverse order)
        rollback_operations.insert(0, ("TASK_STATUS", task_rollback_sql))
        rollback_operations.insert(0, ("OFFLINE_STATUS", offline_rollback_sql))

        # Plan online operations if applicable
        if feature_view.online:
            fully_qualified_online_name = feature_view.fully_qualified_online_table_name()
            online_sql = f"ALTER ONLINE FEATURE TABLE {fully_qualified_online_name} {operation}"
            online_rollback_sql = f"ALTER ONLINE FEATURE TABLE {fully_qualified_online_name} {reverse_operation}"

            operations.append(("ONLINE_STATUS", online_sql))
            # Add to front of rollback operations to maintain reverse order
            rollback_operations.insert(0, ("ONLINE_STATUS", online_rollback_sql))

        return operations, rollback_operations

    def _handle_update_failure(
        self,
        error: Exception,
        rollback_operations: list[tuple[str, Union[str, FeatureView]]],
        feature_view: FeatureView,
    ) -> None:
        """Handle update failure with rollback."""
        logger.warning(f"Update failed, attempting rollback: {error}")
        try:
            self._execute_atomic_operations(rollback_operations)
            logger.info("Rollback completed successfully")
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"Update failed and rollback failed. Original error: {error}. Rollback error: {rollback_error}"
                ),
            ) from error

        # Re-raise original error
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"Update feature view {feature_view.name}/{feature_view.version} failed: {error}"
            ),
        ) from error

    def _handle_status_operation_failure(
        self,
        error: Exception,
        rollback_operations: list[tuple[str, Union[str, FeatureView]]],
        feature_view: FeatureView,
        operation: str,
    ) -> None:
        """Handle status operation failure (suspend/resume) with rollback."""
        logger.warning(f"{operation} failed, attempting rollback: {error}")
        try:
            self._execute_atomic_operations(rollback_operations)
            logger.info("Rollback completed successfully")
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"{operation} failed and rollback failed. "
                    f"Operation error: {error}. "
                    f"Rollback error: {rollback_error}"
                ),
            ) from error

        # Re-raise original error
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"{operation} feature view {feature_view.name}/{feature_view.version} failed: {error}"
            ),
        ) from error

    def _execute_atomic_operations(self, operations: list[tuple[str, Union[str, FeatureView]]]) -> None:
        """Execute a list of operations atomically.

        Args:
            operations: List of (operation_type, operation_data) tuples
        """
        for op_type, op_data in operations:
            if op_type in (
                "OFFLINE_UPDATE",
                "OFFLINE_ROLLBACK",
                "UPDATE_ONLINE",
                "OFFLINE_STATUS",
                "TASK_STATUS",
                "ONLINE_STATUS",
            ):
                assert isinstance(op_data, str)
                self._session.sql(op_data).collect(statement_params=self._telemetry_stmp)
            elif op_type == "CREATE_ONLINE":
                assert isinstance(op_data, FeatureView)
                assert op_data.version is not None
                feature_view_name = FeatureView._get_physical_name(op_data.name, op_data.version)
                self._create_online_feature_table(op_data, feature_view_name, version=str(op_data.version))
            elif op_type == "DELETE_ONLINE":
                assert isinstance(op_data, str)
                self._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {op_data}").collect(
                    statement_params=self._telemetry_stmp
                )

    def _materialize_feature_view_resources(
        self,
        *,
        mode: Literal["register", "update"],
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        fully_qualified_name: str,
        version: str,
        # register-only kwargs
        column_descs: str = "",
        tagging_clause_str: str = "",
        block: bool = True,
        overwrite: bool = False,
        created_resources: Optional[list[tuple[_FeatureStoreObjTypes, str]]] = None,
        # update-only kwargs
        old_feature_view: Optional[FeatureView] = None,
        new_schema: Optional[dict[str, DataType]] = None,
    ) -> None:
        """Single named entry point for materializing a feature view's offline + online resources.

        Both ``register_feature_view`` and the ``updated_feature_df`` path of
        ``update_feature_view`` route through this method, giving the two flows a single
        discoverable pivot. The dispatcher itself does not enforce parity — each ``mode``
        still owns its own helper (``_materialize_register_path`` for register,
        ``_recreate_append_only_feature_view_atomically`` for update) — so when adding a
        new materialization resource (e.g. a new metadata table) the wiring must be added
        to *both* helpers. Centralizing the dispatch here at least makes the second
        helper easy to find from the first.

        ``mode="register"`` performs fresh creation. The caller owns top-level rollback
        via ``created_resources`` (any partial creations are unwound by
        ``_rollback_created_resources``).

        ``mode="update"`` performs an atomic-ish recreate of the offline DT, refresh
        task, and OFT (and extends the snapshot schema first). Rollback is handled
        internally via compensating actions; this mode requires ``old_feature_view`` and
        ``new_schema``.

        Args:
            mode: ``"register"`` for fresh creation, ``"update"`` for append_only recreate.
            feature_view: The desired-state FeatureView to materialize. For ``"update"``,
                this is the new (target) FV; the prior FV must be passed as
                ``old_feature_view``.
            feature_view_name: Versioned physical name of the FV (e.g. ``MY_FV$V1``).
            fully_qualified_name: Fully qualified name of the offline DT.
            version: Version string for the FV (used by online table creation).
            column_descs: ``"register"``-only. Column-comment clause for DT/View DDL.
            tagging_clause_str: ``"register"``-only. Tag-set clause for the offline object.
            block: ``"register"``-only. Whether DT creation should block on first refresh.
            overwrite: ``"register"``-only. Whether to ``CREATE OR REPLACE``.
            created_resources: ``"register"``-only. Mutable list the helpers append to so
                the caller's outer try/except can roll back partial state.
            old_feature_view: ``"update"``-only. The currently registered FV (state to
                restore on rollback).
            new_schema: ``"update"``-only. Target column types in canonical order, used to
                derive extend-only snapshot DDL.

        Raises:
            ValueError: If required kwargs for the chosen ``mode`` are missing.
        """
        if mode == "register":
            if created_resources is None:
                raise ValueError("created_resources is required when mode='register'")
            self._materialize_register_path(
                feature_view=feature_view,
                feature_view_name=feature_view_name,
                fully_qualified_name=fully_qualified_name,
                version=version,
                column_descs=column_descs,
                tagging_clause_str=tagging_clause_str,
                block=block,
                overwrite=overwrite,
                created_resources=created_resources,
            )
        else:
            if old_feature_view is None or new_schema is None:
                raise ValueError("old_feature_view and new_schema are required when mode='update'")
            self._recreate_append_only_feature_view_atomically(
                old_feature_view=old_feature_view,
                new_feature_view=feature_view,
                feature_view_name=feature_view_name,
                fully_qualified_name=fully_qualified_name,
                version=version,
                new_schema=new_schema,
            )

    def _materialize_register_path(
        self,
        *,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        fully_qualified_name: str,
        version: str,
        column_descs: str,
        tagging_clause_str: str,
        block: bool,
        overwrite: bool,
        created_resources: list[tuple[_FeatureStoreObjTypes, str]],
    ) -> None:
        """Create the offline DT/View and the online feature table for a fresh registration.

        Body is the Step 1 + Step 2 of ``register_feature_view``, lifted into a method so
        ``_materialize_feature_view_resources`` is the single dispatch point shared with
        the update path. Both helpers append to ``created_resources`` so partial progress
        is rolled back by ``_rollback_created_resources`` if anything raises mid-way.

        Args:
            feature_view: The FeatureView being registered.
            feature_view_name: Versioned physical name (e.g. ``MY_FV$V1``).
            fully_qualified_name: Fully qualified name of the offline object (DT/View).
            version: Version string for the FV (used by online table creation).
            column_descs: Pre-rendered ``COMMENT`` clauses for the offline DT columns.
            tagging_clause_str: Pre-rendered ``WITH TAG (...)`` clause for the offline object.
            block: When True, the DT creation waits for the initial refresh to complete.
            overwrite: When True, the offline object is created with ``OR REPLACE`` and
                any dangling online feature table from the prior registration is dropped.
            created_resources: Append-only audit trail of resources created during this
                registration; consumed by ``_rollback_created_resources`` on failure.
        """
        # Step 1: Create offline feature view (Dynamic Table or View).
        if feature_view.is_rollup:
            self._create_rollup_feature_view(
                feature_view=feature_view,
                feature_view_name=feature_view_name,
                fully_qualified_name=fully_qualified_name,
                tagging_clause_str=tagging_clause_str,
                overwrite=overwrite,
                created_resources=created_resources,
            )
        else:
            self._create_offline_feature_view(
                feature_view=feature_view,
                feature_view_name=feature_view_name,
                fully_qualified_name=fully_qualified_name,
                column_descs=column_descs,
                tagging_clause_str=tagging_clause_str,
                block=block,
                overwrite=overwrite,
                created_resources=created_resources,
            )

        # Step 2: Create online feature table if requested
        # (for streaming FVs, the existing POSTGRES path builds the streaming spec internally)
        if feature_view.online:
            online_table_name = self._create_online_feature_table(
                feature_view, feature_view_name, version=version, overwrite=overwrite
            )
            created_resources.append(
                (
                    _FeatureStoreObjTypes.ONLINE_FEATURE_TABLE,
                    self._get_fully_qualified_name(online_table_name),
                )
            )
        elif overwrite:
            # Delete dangling online feature table when overwriting online-enabled FV with online-disabled FV
            online_table_name = FeatureView._get_online_table_name(feature_view_name)
            fully_qualified_online_name = self._get_fully_qualified_name(online_table_name)
            self._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {fully_qualified_online_name}").collect(
                statement_params=self._telemetry_stmp
            )

    def _recreate_append_only_feature_view_atomically(
        self,
        *,
        old_feature_view: FeatureView,
        new_feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        fully_qualified_name: str,
        version: str,
        new_schema: dict[str, DataType],
    ) -> None:
        """Recreate the offline DT, online table, and refresh task for an append_only FV update.

        The four mutations executed by ``update_feature_view(updated_feature_df=...)`` —
        snapshot ``ALTER TABLE ADD COLUMN``, DT ``CREATE OR REPLACE``, task ``CREATE OR
        REPLACE``, and online feature table ``CREATE OR REPLACE`` — are not transactional
        in Snowflake. Without compensating actions, a failure midway leaves the feature
        view in an inconsistent steady state (e.g. snapshot has new columns but the task
        body still ``INSERT``s the old column list, so the new columns never get
        populated).

        This helper sequences the mutations and pairs each one with a compensating action.
        For helpers that can partially succeed before raising (for example ``CREATE OR
        REPLACE`` succeeds and a follow-up validation or ``RESUME`` fails), the
        compensating action is registered before the forward call so the failure handler
        can still restore the prior state.

        OFT is sequenced last to match the initial-create path
        (``_materialize_register_path``); cross-service work (POSTGRES online service
        health, spec validation, managed-cluster bootstrap) sits at the end of the
        forward DAG so failures earlier in the chain don't leave partial external state.

        Compensating actions are best-effort: if one fails the failure is logged but does
        not mask the original forward error, which is wrapped and re-raised.

        Args:
            old_feature_view: The currently registered FeatureView. Used as the source of
                truth for compensating actions (its query / refresh_freq / warehouse / desc
                describe the state we restore on rollback).
            new_feature_view: The desired-state FeatureView built from
                ``updated_feature_df``. Used for the forward CREATE OR REPLACE statements.
            feature_view_name: Versioned physical name of the FV (e.g. ``MY_FV$V1``).
            fully_qualified_name: Fully qualified name of the offline DT.
            version: Version string for the FV (used by online table creation).
            new_schema: Target column types in canonical order, used to derive the
                extend-only snapshot DDL.

        Raises:
            SnowflakeMLException: [INVALID_ARGUMENT] if the schema change is not
                extend-only (drop/reorder/type change).
            SnowflakeMLException: [INTERNAL_SNOWPARK_ERROR] if any forward step fails;
                the original error is chained as ``__cause__``.
        """
        snapshot_table_fqn = new_feature_view.fully_qualified_snapshot_table_name()
        snapshot_cols = fs_table_schema_evolution.schema_upper_name_map(self._session.table(snapshot_table_fqn).schema)
        forward_schema_cmds, rollback_schema_cmds = self._compute_extend_only_schema_evolution_cmds(
            snapshot_table_fqn, snapshot_cols, new_schema
        )

        obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, snowml_version.VERSION)

        # append_only FVs cannot be tiled (rejected at append-only registration validation), so the
        # tiled-FV "" branch in _build_column_descs's callers is unreachable here.
        new_column_descs = self._build_column_descs(new_feature_view)
        new_tagging_clause = self._build_tagging_clause(new_feature_view, obj_info)
        new_refresh_freq = new_feature_view.refresh_freq
        new_schedule_task = feature_view_refresh_freq._is_cron_refresh_freq(new_refresh_freq)
        new_warehouse = (
            new_feature_view.warehouse if new_feature_view.warehouse is not None else self._default_warehouse
        )

        old_column_descs = self._build_column_descs(old_feature_view)
        old_tagging_clause = self._build_tagging_clause(old_feature_view, obj_info)
        old_refresh_freq = old_feature_view.refresh_freq
        old_schedule_task = feature_view_refresh_freq._is_cron_refresh_freq(old_refresh_freq)
        old_warehouse = (
            old_feature_view.warehouse if old_feature_view.warehouse is not None else self._default_warehouse
        )

        fully_qualified_online_name = self._get_fully_qualified_name(
            FeatureView._get_online_table_name(feature_view_name)
        )

        # Return type is Any because compensating callbacks dispatch through helpers with
        # heterogeneous return values (None, str, list[Row]); the values are unused.
        compensating_actions: list[tuple[_MaterializedResourceKind, Callable[[], Any]]] = []

        def _run_sql_list(cmds: list[str]) -> None:
            for cmd in cmds:
                self._session.sql(cmd).collect(statement_params=self._telemetry_stmp)

        try:
            # Step 1: extend the snapshot schema. Partial-DDL failures (cmd[k] succeeds,
            # cmd[k+1] fails) are recovered inline by the helper so we never push a
            # half-applied compensating action onto the outer list.
            self._execute_extend_only_schema_evolution_cmds(forward_schema_cmds, rollback_schema_cmds)
            compensating_actions.append(
                (
                    _MaterializedResourceKind.SNAPSHOT_SCHEMA_EVOLUTION,
                    lambda: _run_sql_list(rollback_schema_cmds),
                )
            )

            # Step 2: recreate the DT with the new query/schema.
            compensating_actions.append(
                (
                    _MaterializedResourceKind.OFFLINE_FEATURE_VIEW,
                    lambda: self._create_dynamic_table(
                        feature_view_name,
                        old_feature_view,
                        fully_qualified_name,
                        old_column_descs,
                        old_tagging_clause,
                        old_schedule_task,
                        old_warehouse,
                        block=True,
                        override=True,
                    ),
                )
            )
            self._create_dynamic_table(
                feature_view_name,
                new_feature_view,
                fully_qualified_name,
                new_column_descs,
                new_tagging_clause,
                new_schedule_task,
                new_warehouse,
                block=True,
                override=True,
            )

            # Step 3: refresh task. The task body itself is schema-agnostic (``INSERT INTO
            # snap SELECT * FROM dt``), so this recreate is not driven by column-list
            # changes; it exists to pick up updates to ``refresh_freq`` / ``warehouse``
            # and to bridge transitions between cron-scheduled and duration-scheduled FVs
            # (the latter has no companion task).
            #
            # ``schedule_task`` is True iff refresh_freq is a cron expression (not a
            # duration and not "DOWNSTREAM"). Forward / compensating-action pairing:
            #   new_cron | old_cron | forward                 | compensating
            #   ---------+----------+-------------------------+----------------------------
            #   True     | True     | create-or-replace task  | recreate task from old FV
            #   True     | False    | create task             | drop task
            #   False    | True     | drop task               | recreate task from old FV
            #   False    | False    | (nothing)               | (nothing)
            if new_schedule_task:
                if old_schedule_task:
                    compensating_actions.append(
                        (
                            _MaterializedResourceKind.REFRESH_TASK,
                            lambda: self._create_scheduled_refresh_task(
                                " OR REPLACE",
                                old_feature_view,
                                fully_qualified_name,
                                old_warehouse,
                                feature_view_name=feature_view_name,
                            ),
                        )
                    )
                else:
                    compensating_actions.append(
                        (
                            _MaterializedResourceKind.REFRESH_TASK,
                            lambda: self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                                statement_params=self._telemetry_stmp
                            ),
                        )
                    )
                self._create_scheduled_refresh_task(
                    " OR REPLACE",
                    new_feature_view,
                    fully_qualified_name,
                    new_warehouse,
                    feature_view_name=feature_view_name,
                )
            elif old_schedule_task:
                # New FV is duration-based; drop the orphaned task left over from the
                # prior cron schedule. Compensating action restores it from the prior FV.
                compensating_actions.append(
                    (
                        _MaterializedResourceKind.REFRESH_TASK,
                        lambda: self._create_scheduled_refresh_task(
                            " OR REPLACE",
                            old_feature_view,
                            fully_qualified_name,
                            old_warehouse,
                            feature_view_name=feature_view_name,
                        ),
                    )
                )
                self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                    statement_params=self._telemetry_stmp
                )

            # Step 4: online feature table. Mirrors register_feature_view's gating —
            # only create when the new FV is online; if the prior FV was online and the
            # new one isn't, drop the dangling table. OFT is sequenced last to mirror
            # the initial-create path; its forward call (POSTGRES service health, spec
            # validation, managed-cluster provisioning) can partially succeed before
            # raising, so the compensating action is registered before the forward call
            # so a partial-success failure still triggers rollback.
            #
            # Forward / compensating-action pairing:
            #   new.online | old.online | forward                | compensating
            #   -----------+------------+------------------------+----------------------
            #   True       | True       | create-or-replace OFT  | recreate from old FV
            #   True       | False      | create OFT             | drop OFT
            #   False      | True       | drop OFT               | recreate from old FV
            #   False      | False      | (nothing)              | (nothing)
            if new_feature_view.online:
                if old_feature_view.online:
                    compensating_actions.append(
                        (
                            _MaterializedResourceKind.ONLINE_FEATURE_TABLE,
                            lambda: self._create_online_feature_table(
                                old_feature_view,
                                feature_view_name,
                                version=version,
                                overwrite=True,
                            ),
                        )
                    )
                else:
                    compensating_actions.append(
                        (
                            _MaterializedResourceKind.ONLINE_FEATURE_TABLE,
                            lambda: self._session.sql(
                                f"DROP ONLINE FEATURE TABLE IF EXISTS {fully_qualified_online_name}"
                            ).collect(statement_params=self._telemetry_stmp),
                        )
                    )
                self._create_online_feature_table(new_feature_view, feature_view_name, version=version, overwrite=True)
            elif old_feature_view.online:
                compensating_actions.append(
                    (
                        _MaterializedResourceKind.ONLINE_FEATURE_TABLE,
                        lambda: self._create_online_feature_table(
                            old_feature_view,
                            feature_view_name,
                            version=version,
                            overwrite=True,
                        ),
                    )
                )
                self._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {fully_qualified_online_name}").collect(
                    statement_params=self._telemetry_stmp
                )
        except Exception as forward_error:
            logger.warning(
                "update_feature_view recreate failed for %s/%s: %s; running compensating actions",
                old_feature_view.name,
                old_feature_view.version,
                forward_error,
            )
            # Roll back in dependency order, not reverse-append. The running pipeline
            # (DT, then OFT, then refresh task) is restored first so a partially-recovered
            # FV is queryable as soon as possible; the snapshot ALTER DROP COLUMN cleanup
            # runs last because nothing else depends on it. Within the pipeline tier:
            # ONLINE_FEATURE_TABLE rollback in HYBRID_TABLE mode runs ``CREATE ... FROM
            # <source>``, which inherits its column shape from the current source DT, so
            # OFFLINE_FEATURE_VIEW (DT) must be restored *before* the OFT rollback
            # inherits from it. REFRESH_TASK is column-agnostic (its body uses
            # ``SELECT *``) and depends on neither, so it slots in after OFT. The
            # ordering also covers partial-failure cases: DT, task, and OFT register
            # their compensating actions *before* the forward call, so they appear here
            # even when the forward step never completed.
            for rollback_step in (
                _MaterializedResourceKind.OFFLINE_FEATURE_VIEW,
                _MaterializedResourceKind.ONLINE_FEATURE_TABLE,
                _MaterializedResourceKind.REFRESH_TASK,
                _MaterializedResourceKind.SNAPSHOT_SCHEMA_EVOLUTION,
            ):
                for step_kind, action in compensating_actions:
                    if step_kind != rollback_step:
                        continue
                    try:
                        action()
                        logger.info("Rollback step '%s' succeeded", step_kind.value)
                    except Exception as rollback_error:
                        logger.exception(
                            "Rollback step '%s' failed: %s",
                            step_kind.value,
                            rollback_error,
                        )
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"update_feature_view {old_feature_view.name}/{old_feature_view.version} failed: "
                    f"{forward_error}"
                ),
            ) from forward_error

    def _read_from_offline_store(
        self,
        feature_view: FeatureView,
        keys: Optional[list[list[str]]],
        feature_names: Optional[list[str]],
    ) -> DataFrame:
        """Read feature values from the offline store (main feature view table).

        For tiled feature views, this computes aggregated features at current time
        by creating a synthetic spine with unique entity combinations.

        Args:
            feature_view: The feature view to read from.
            keys: Optional list of key values to filter by.
            feature_names: Optional list of feature names to return.

        Returns:
            Snowpark DataFrame containing the feature values.
        """
        table_name = feature_view.fully_qualified_name()

        # For tiled FVs, compute features at current time
        if feature_view.is_tiled:
            return self._read_tiled_fv_at_current_time(feature_view, keys, feature_names)

        # Build SELECT and WHERE clauses using helper methods
        select_clause = self._build_select_clause_and_validate(feature_view, feature_names, include_join_keys=True)
        where_clause, _ = self._build_where_clause_for_keys(feature_view, keys)

        query = f"SELECT {select_clause} FROM {table_name}{where_clause}"
        return self._session.sql(query)

    def _read_tiled_fv_at_current_time(
        self,
        feature_view: FeatureView,
        keys: Optional[list[list[str]]],
        feature_names: Optional[list[str]],
    ) -> DataFrame:
        """Read tiled feature view by computing aggregated features at current time.

        Creates a synthetic spine with unique entity combinations from the tile table,
        uses CURRENT_TIMESTAMP as the query time, and merges tiles to compute features.

        Args:
            feature_view: The tiled feature view to read from.
            keys: Optional list of key values to filter by.
            feature_names: Optional list of feature names to return.

        Returns:
            Snowpark DataFrame containing the computed feature values.
        """
        table_name = feature_view.fully_qualified_name()

        # Get join keys from entities
        join_keys: list[SqlIdentifier] = []
        for entity in feature_view.entities:
            join_keys.extend(entity.join_keys)

        join_keys_str = ", ".join(join_keys)

        # Build WHERE clause for key filtering (if any)
        where_clause, _ = self._build_where_clause_for_keys(feature_view, keys)

        # Step 1: Create spine CTE with unique entities + CURRENT_TIMESTAMP
        spine_cte = f"""
            SELECT DISTINCT {join_keys_str},
                   CURRENT_TIMESTAMP() AS "_QUERY_TS"
            FROM {table_name}{where_clause}
        """

        # Step 2: Generate merge CTEs using MergingSqlGenerator
        assert feature_view.aggregation_specs is not None
        assert feature_view.feature_granularity is not None
        assert feature_view.timestamp_col is not None

        generator = MergingSqlGenerator(
            tile_table=table_name,
            join_keys=join_keys,
            timestamp_col=feature_view.timestamp_col,
            feature_granularity=feature_view.feature_granularity,
            features=feature_view.aggregation_specs,
            spine_timestamp_col="_QUERY_TS",
            fv_index=0,
            authoring_pkg_version=feature_view.authoring_pkg_version,
        )

        merge_ctes = generator.generate_all_ctes()

        # Step 3: Build full query
        cte_parts = [f"SPINE AS ({spine_cte})"]
        for cte_name, cte_body in merge_ctes:
            cte_parts.append(f"{cte_name} AS ({cte_body})")

        # Get feature columns for final SELECT
        all_feature_cols = [spec.get_sql_column_name() for spec in feature_view.aggregation_specs]
        if feature_names:
            # Filter to requested features
            requested = {SqlIdentifier(f).resolved() for f in feature_names}
            keys_cols_to_keep: set[str] = set()
            keys_spec_by_window_offset = {
                (spec.window, spec.offset): spec
                for spec in feature_view.aggregation_specs
                if spec.function.is_secondary_key_array()
            }
            if keys_spec_by_window_offset:
                for spec in feature_view.aggregation_specs:
                    if spec.function.is_secondary_key_array():
                        continue
                    if SqlIdentifier(spec.get_sql_column_name()).resolved() in requested:
                        keys_spec = keys_spec_by_window_offset.get((spec.window, spec.offset))
                        if keys_spec is not None:
                            keys_cols_to_keep.add(keys_spec.get_sql_column_name())
            all_feature_cols = [
                c for c in all_feature_cols if SqlIdentifier(c).resolved() in requested or c in keys_cols_to_keep
            ]

        feature_cols_str = ", ".join(all_feature_cols)
        # CTE name format matches generator: FV{index:03d}
        final_select = f"SELECT {join_keys_str}, {feature_cols_str} FROM FV000"

        full_query = f"WITH {', '.join(cte_parts)} {final_select}"
        return self._session.sql(full_query)

    def _oft_full_schema(self, feature_view: FeatureView) -> StructType:
        """Schema of the OFT object (PK columns + feature columns).

        BFV/SFV ``output_schema`` already includes both. RealtimeFeatureView
        ``output_schema`` is ``realtime_config.output_schema`` -- feature
        columns only -- so the join-key fields are synthesized from upstream
        source FVs. If the RTFV author also re-declared a join key in
        ``output_schema``, the upstream-derived field wins (it carries the
        upstream's authoritative datatype) and the duplicate is dropped.

        Args:
            feature_view: The feature view whose OFT-level schema is needed.

        Returns:
            ``StructType`` containing both join-key fields and feature fields.

        Raises:
            SnowflakeMLException: ``[RuntimeError]`` with
                :data:`error_codes.INTERNAL_PYTHON_ERROR` when an RTFV is
                missing its realtime configuration (registry corruption).
        """
        if not feature_view.is_realtime_feature_view:
            return feature_view.output_schema
        from snowflake.ml.feature_store.realtime_registration import (
            resolve_realtime_join_key_fields,
        )

        pk_fields = resolve_realtime_join_key_fields(feature_view)
        if feature_view.realtime_config is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_PYTHON_ERROR,
                original_exception=RuntimeError(
                    f"realtime feature view {feature_view.name}/{feature_view.version}: "
                    "realtime configuration is missing. Re-register the feature view."
                ),
            )
        pk_canonical = {SqlIdentifier(f.name).resolved() for f in pk_fields}
        feature_fields = [
            f
            for f in feature_view.realtime_config.output_schema.fields
            if SqlIdentifier(f.name).resolved() not in pk_canonical
        ]
        return StructType([*pk_fields, *feature_fields])

    def _postgres_online_read_struct_type(
        self, feature_view: FeatureView, feature_names: Optional[list[str]]
    ) -> StructType:
        join_names = [k.resolved() for e in feature_view.entities for k in e.join_keys]
        full_schema = self._oft_full_schema(feature_view)
        if feature_names:
            wanted = set(join_names) | set(feature_names)
            fields = [f for f in full_schema.fields if f.name in wanted]
        else:
            fields = list(full_schema.fields)
        return StructType(fields)

    def _empty_dataframe_for_postgres_online_read(
        self, feature_view: FeatureView, feature_names: Optional[list[str]]
    ) -> DataFrame:
        return self._session.create_dataframe(
            [],
            schema=self._postgres_online_read_struct_type(feature_view, feature_names),
        )

    def _read_postgres_online_via_query_api(
        self,
        feature_view: FeatureView,
        keys: Optional[list[list[str]]],
        feature_names: Optional[list[str]],
        *,
        as_pandas: bool = False,
        request_context: Optional[list[dict[str, Any]]] = None,
    ) -> Union[DataFrame, pd.DataFrame]:
        query_url = getattr(feature_view, "_postgres_online_query_url", None)
        if not query_url:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Online read for this Postgres-backed feature view requires a hydrated query endpoint. "
                    "Call get_feature_view(name, version) again after the Online Service is RUNNING."
                ),
            )
        if not keys:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Online read from a Postgres-backed feature view requires at least one row in `keys`; "
                    "unbounded table scans are not supported via the Online Service Query API."
                ),
            )
        if feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Online read requires a registered feature view with a version for Postgres-backed online tables."
                ),
            )
        # RTFV reads reject feature_names upfront in read_feature_view, so the
        # select-clause validator is only invoked for BFV/SFV.
        if not feature_view.is_realtime_feature_view:
            self._build_select_clause_and_validate(feature_view, feature_names, include_join_keys=True)

        join_names = [k.resolved() for e in feature_view.entities for k in e.join_keys]
        join_set = set(join_names)
        full_schema = self._oft_full_schema(feature_view)
        join_key_field_types = {f.name: f.datatype for f in full_schema.fields if f.name in join_set}

        rows, schema = online_service.read_postgres_online_features(
            session=self._session,
            query_url=query_url,
            feature_view_name=str(feature_view.name),
            feature_view_version=str(feature_view.version),
            join_key_names=join_names,
            keys=keys,
            feature_names=feature_names,
            join_key_field_types=join_key_field_types,
            request_context=request_context,
            http_client=self._get_or_create_online_http_client(),
        )
        if as_pandas:
            return online_service.rows_to_pandas_for_postgres_online(rows, schema)
        if not rows:
            return self._empty_dataframe_for_postgres_online_read(feature_view, feature_names)
        coerced = [online_service._coerce_row_values_for_snowpark_local_schema(r, schema) for r in rows]
        return self._session.create_dataframe(coerced, schema=schema)

    def _read_realtime_feature_view(
        self,
        feature_view: FeatureView,
        *,
        keys: Optional[list[list[str]]],
        feature_names: Optional[list[str]],
        request_context: Optional[Any],
        as_pandas: Optional[bool],
    ) -> Union[DataFrame, pd.DataFrame]:
        """Dispatch ``read_feature_view`` for a RealtimeFeatureView.

        Runs the RTFV-specific validation order (online backing, keys,
        request_context type/shape/length) before converting the
        ``pd.DataFrame`` request_context to the per-row ``list[dict]`` shape
        the Online Service expects, and delegates to the shared Postgres-online
        read implementation. The ``store_type`` kwarg from the public API is
        ignored: RealtimeFeatureViews have no offline backing and always
        compute values at request time via the online path.

        Args:
            feature_view: A registered RealtimeFeatureView.
            keys: Per-row join key tuples. Must be non-empty.
            feature_names: Rejected for RTFV reads (server returns the full
                ``compute_fn`` output set). Pass ``None``.
            request_context: ``pandas.DataFrame`` whose columns are a superset
                of the RTFV's ``RequestSource.schema`` field names and whose
                row count equals ``len(keys)``. Pass ``None`` when the RTFV
                was registered without a ``RequestSource``.
            as_pandas: If ``None``, defaults to ``True`` for RTFV reads
                (consistent with FG and Postgres-online behavior).

        Returns:
            ``pandas.DataFrame`` (default) or Snowpark ``DataFrame`` containing
            join keys plus the RTFV's compute_fn output columns.

        Raises:
            SnowflakeMLException: ``[ValueError]`` for any RTFV-specific
                validation failure listed in :meth:`read_feature_view`.
        """
        # pandas is imported lazily to avoid a hard runtime dependency in
        # callers that don't touch the read path; the rest of the FS module
        # gates pandas under TYPE_CHECKING.
        import pandas as pandas_mod

        rt_cfg = feature_view.realtime_config
        if rt_cfg is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_PYTHON_ERROR,
                original_exception=RuntimeError(
                    f"realtime feature view {feature_view.name}/{feature_view.version}: "
                    "realtime configuration is missing. Re-register the feature view."
                ),
            )

        online_config = feature_view.online_config
        if online_config is None or online_config.store_type != OnlineStoreType.POSTGRES:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"realtime feature view {feature_view.name}/{feature_view.version}: "
                    "online store must be POSTGRES (the only supported backing). "
                    f"Found store_type={online_config.store_type if online_config else None!r}."
                ),
            )

        if feature_names is not None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"realtime feature view {feature_view.name}/{feature_view.version}: "
                    "feature_names filtering is not supported; the Online Service returns "
                    "the full compute_fn output set. Drop columns client-side via "
                    "pandas .drop() if you need a subset."
                ),
            )

        if keys is None or len(keys) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"realtime feature view {feature_view.name}/{feature_view.version}: "
                    "`keys` must be a non-empty list of join-key tuples; RealtimeFeatureView "
                    "reads do not support unbounded scans."
                ),
            )

        request_source = rt_cfg.request_source
        request_context_records: Optional[list[dict[str, Any]]] = None

        if request_source is None:
            if request_context is not None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"realtime feature view {feature_view.name}/{feature_view.version}: "
                        "this RealtimeFeatureView was registered without a RequestSource; "
                        "`request_context` must be omitted."
                    ),
                )
        else:
            request_columns = [f.name for f in request_source.schema.fields]
            # Match request_context column names against the RequestSource schema using
            # Snowflake identifier normalization rules: unquoted "amount" and the
            # Snowpark-normalized "AMOUNT" refer to the same column. We canonicalize
            # both sides to the resolved form and rename pandas columns to the canonical
            # name before serializing so the server payload keys match the RequestSource
            # field names verbatim.
            canonical_required = {SqlIdentifier(c).resolved(): c for c in request_columns}

            if request_context is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"realtime feature view {feature_view.name}/{feature_view.version}: "
                        "`request_context` is required for realtime reads. Pass a pandas DataFrame "
                        f"with columns {request_columns} and one row per entry in `keys`, e.g. "
                        "pd.DataFrame({" + ", ".join(f"{c!r}: [...]" for c in request_columns) + "})."
                    ),
                )

            # Shared with the FG-with-RTFV read path so the two paths
            # cannot disagree on shape, casing, or length.
            from snowflake.ml.feature_store.realtime_registration import (
                canonicalize_request_context,
            )

            request_context_records = canonicalize_request_context(
                request_context=request_context,
                required=canonical_required,
                keys=keys,
                error_prefix=f"realtime feature view {feature_view.name}/{feature_view.version}",
                pandas_mod=pandas_mod,
            )

        # RTFV reads default to pandas (consistent with FG and the Postgres-online fast path).
        if as_pandas is None:
            as_pandas = True

        return self._read_postgres_online_via_query_api(
            feature_view,
            keys,
            feature_names=None,
            as_pandas=as_pandas,
            request_context=request_context_records,
        )

    def _read_from_online_store(
        self,
        feature_view: FeatureView,
        keys: Optional[list[list[str]]],
        feature_names: Optional[list[str]],
        *,
        as_pandas: bool = False,
        request_context: Optional[list[dict[str, Any]]] = None,
    ) -> Union[DataFrame, pd.DataFrame]:
        """Read feature values from the online store with optional key filtering.

        Uses bind variables for single-key lookups and literal interpolation
        for batch lookups.

        Args:
            feature_view: The registered feature view to read from.
            keys: Optional list of key value lists to filter by.
            feature_names: Optional list of feature names to return.
            as_pandas: If True, return a ``pandas.DataFrame`` instead of a Snowpark ``DataFrame``.
            request_context: Per-row request context, forwarded to Postgres-online reads.
                Always ``None`` here -- this method is only reached for non-RTFV feature views,
                which the public ``read_feature_view`` API has already rejected request_context
                for.

        Returns:
            Snowpark DataFrame (or pandas DataFrame when ``as_pandas`` is True) containing the
            feature view data.

        Raises:
            SnowflakeMLException: If online store is not enabled.
        """
        if not feature_view.online:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Online store is not enabled for feature view {feature_view.name}/{feature_view.version}"
                ),
            )

        if feature_view.online_config is not None and feature_view.online_config.store_type == OnlineStoreType.POSTGRES:
            return self._read_postgres_online_via_query_api(
                feature_view,
                keys,
                feature_names,
                as_pandas=as_pandas,
                request_context=request_context,
            )

        fully_qualified_online_name = feature_view.fully_qualified_online_table_name()

        select_clause = self._build_select_clause_and_validate(feature_view, feature_names, include_join_keys=True)
        # NOTE: We use snowflake sql query binds for single-point lookups to ensure the best performance.
        # We don't use binds for multi-point lookups because we don't have any evidence that it improves
        # performance, but we may need to revisit this in the future.
        use_binds = keys is not None and len(keys) == 1
        where_clause, params = self._build_where_clause_for_keys(feature_view, keys, use_binds=use_binds)

        query = f"SELECT {select_clause} FROM {fully_qualified_online_name}{where_clause}"
        df = self._session.sql(query, params=params) if params else self._session.sql(query)
        if as_pandas:
            return df.to_pandas(statement_params=self._telemetry_stmp)
        return df

    @dispatch_decorator()
    def _clear(self, dryrun: bool = True) -> None:
        """
        Clear all feature views and entities. Note Feature Store schema and metadata will NOT be purged
        together. Use SQL to delete schema and metadata instead.

        Args:
            dryrun: Print a list of objects will be deleted but not actually perform the deletion when true.
        """
        warnings.warn(
            "It will clear ALL feature views and entities in this Feature Store. Make sure your role"
            " has sufficient access to all feature views and entities. Insufficient access to some feature"
            " views or entities will leave Feature Store in an incomplete state.",
            stacklevel=2,
            category=UserWarning,
        )

        all_fvs_df = self.list_feature_views()
        all_entities_df = self.list_entities()
        all_fvs_rows = all_fvs_df.collect(statement_params=self._telemetry_stmp)
        all_entities_rows = all_entities_df.collect(statement_params=self._telemetry_stmp)

        if dryrun:
            logger.info(
                "Following feature views and entities will be deleted."
                + " Set 'dryrun=False' to perform the actual deletion.",
            )
            logger.info(f"Total {len(all_fvs_rows)} Feature views to be deleted:")
            all_fvs_df.show(n=len(all_fvs_rows))
            logger.info(f"\nTotal {len(all_entities_rows)} Entities to be deleted:")
            all_entities_df.show(n=len(all_entities_rows))
            return

        for fv_row in all_fvs_rows:
            fv = self.get_feature_view(
                SqlIdentifier(fv_row["NAME"], case_sensitive=True).identifier(),
                fv_row["VERSION"],
            )
            self.delete_feature_view(fv)

        for entity_row in all_entities_rows:
            self.delete_entity(SqlIdentifier(entity_row["NAME"], case_sensitive=True).identifier())

        logger.info(f"Feature store {self._config.full_schema_path} has been cleared.")

    def _build_tagging_clause(
        self,
        feature_view: FeatureView,
        obj_info: _FeatureStoreObjInfo,
    ) -> str:
        """Build the SET TAG clause for a feature view's DT/view."""
        parts = [
            f"{self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)} = '{obj_info.to_json()}'",
            f"{self._get_fully_qualified_name(_FEATURE_VIEW_METADATA_TAG)} = '"
            f"{feature_view._metadata().to_json()}'",
        ]
        for e in feature_view.entities:
            join_keys = [f"{key.resolved()}" for key in e.join_keys]
            parts.append(f"{self._get_fully_qualified_name(self._get_entity_name(e.name))} = '{','.join(join_keys)}'")
        return ",\n".join(parts)

    @staticmethod
    def _build_column_descs(feature_view: FeatureView) -> str:
        """Build the column COMMENT clause for a feature view's DT/view."""
        if feature_view.feature_descs is None:
            return ""
        descs: list[str] = []
        for col in feature_view.output_schema.fields:
            col_desc = feature_view.feature_descs.get(SqlIdentifier(col.name), None)
            comment = "" if col_desc is None else f"COMMENT '{col_desc}'"
            descs.append(f"{col.name} {comment}")
        return ", ".join(descs)

    def _get_feature_view_if_exists(self, name: str, version: str) -> FeatureView:
        existing_fv = self.get_feature_view(name, version)
        warnings.warn(
            f"FeatureView {name}/{version} already exists. Skip registration."
            + " Set `overwrite` to True if you want to replace existing FeatureView.",
            stacklevel=2,
            category=UserWarning,
        )
        return existing_fv

    def _recompose_join_keys(self, join_key: str) -> list[str]:
        # ALLOWED_VALUES in TAG will follow format ["key_1,key2,..."]
        # since keys are already resolved following the SQL identifier rule on the write path,
        # we simply parse the keys back and wrap them with quotes to preserve cases
        # Example join_key repr from TAG value: "[key1,key2,key3]"
        join_keys = join_key[2:-2].split(",")
        res = []
        for k in join_keys:
            res.append(f'"{k}"')
        return res

    def _create_dynamic_table(
        self,
        feature_view_name: SqlIdentifier,
        feature_view: FeatureView,
        fully_qualified_name: str,
        column_descs: str,
        tagging_clause: str,
        schedule_task: bool,
        warehouse: SqlIdentifier,
        block: bool,
        override: bool,
    ) -> None:
        # TODO: cluster by join keys once DT supports that
        query = ""
        try:
            override_clause = " OR REPLACE" if override else ""
            query = self._create_dynamic_table_query(
                override_clause,
                fully_qualified_name,
                column_descs,
                schedule_task,
                feature_view,
                tagging_clause,
                warehouse,
            )
            self._session.sql(query).collect(block=block, statement_params=self._telemetry_stmp)
        except Exception as e:
            # Check for type conflict: VIEW exists but we want to create a DYNAMIC TABLE
            if (
                override
                and isinstance(e, SnowparkSQLException)
                and e.error_code == error_codes.SQL_COMPILATION_ERROR
                and self._get_existing_feature_view_object_type(feature_view_name) == "VIEW"
            ):
                logger.info(
                    f"Existing VIEW detected, performing shadow swap to DYNAMIC TABLE for {fully_qualified_name}"
                )
                self._shadow_swap_to_dynamic_table(
                    feature_view,
                    fully_qualified_name,
                    column_descs,
                    tagging_clause,
                    schedule_task,
                    warehouse,
                    block,
                )
            else:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(
                        f"Create dynamic table [\n{query}\n] or task {fully_qualified_name} failed: {e}."
                    ),
                ) from e

        if block:
            self._check_dynamic_table_refresh_mode(feature_view_name)

    def _create_dynamic_table_query(
        self,
        override_clause: str,
        table_name: str,
        column_descs: str,
        schedule_task: bool,
        feature_view: FeatureView,
        tagging_clause: str,
        warehouse: str,
    ) -> str:
        # Use tiling query for tiled feature views
        if feature_view.is_tiled:
            source_query = feature_view._get_tile_query()
        else:
            source_query = feature_view.query

        # Include column definitions only if provided (skip for tiled feature views)
        column_clause = f" ({column_descs})" if column_descs else ""

        init_wh_clause = _initialization_warehouse_clause(feature_view)
        storage_config = feature_view.storage_config
        if storage_config is not None and storage_config.format == StorageFormat.ICEBERG:
            # These should be validated by FeatureView constructor and _resolve_storage_config
            assert storage_config.external_volume is not None, "external_volume is required for ICEBERG format"
            assert storage_config.base_location is not None, "base_location is required for ICEBERG format"
            query = f"""CREATE{override_clause} DYNAMIC ICEBERG TABLE {table_name}{column_clause}
                TARGET_LAG = '{'DOWNSTREAM' if schedule_task else feature_view.refresh_freq}'
                COMMENT = {_sql_string_literal(feature_view.desc)}
                TAG (
                    {tagging_clause}
                )
                WAREHOUSE = {warehouse}{init_wh_clause}
                REFRESH_MODE = {feature_view.refresh_mode}
                INITIALIZE = {feature_view.initialize}
                CATALOG = 'SNOWFLAKE'
                EXTERNAL_VOLUME = {SqlIdentifier(storage_config.external_volume)}
                BASE_LOCATION = '{storage_config.base_location.replace("'", "''")}'
            """
        else:
            query = f"""CREATE{override_clause} DYNAMIC TABLE {table_name}{column_clause}
                TARGET_LAG = '{'DOWNSTREAM' if schedule_task else feature_view.refresh_freq}'
                COMMENT = {_sql_string_literal(feature_view.desc)}
                TAG (
                    {tagging_clause}
                )
                WAREHOUSE = {warehouse}{init_wh_clause}
                REFRESH_MODE = {feature_view.refresh_mode}
                INITIALIZE = {feature_view.initialize}
            """
        if feature_view.cluster_by:
            # For tiled FVs, replace timestamp column with TILE_START in cluster_by
            if feature_view.is_tiled and feature_view.timestamp_col:
                ts_col_upper = feature_view.timestamp_col.upper()
                cluster_by_cols = [
                    _TILE_START_COL if col.upper() == ts_col_upper else col for col in feature_view.cluster_by
                ]
            else:
                cluster_by_cols = [str(col) for col in feature_view.cluster_by]
            cluster_by_clause = f"CLUSTER BY ({', '.join(cluster_by_cols)})"
            query += f"{cluster_by_clause}"

        query += f"""
            AS {source_query}
        """
        return query

    def _ensure_snapshot_status_table_exists(self) -> None:
        """Create the snapshot status table if it does not already exist.

        Called from feature-store initialization (``CREATE_IF_NOT_EXIST``) and before
        append-only snapshot registration so status-row cleanup never depends on a
        prior append-only registration in the schema.
        """
        status_table_fqn = self._get_fully_qualified_name(SqlIdentifier(_SNAPSHOT_STATUS_TABLE))
        self._session.sql(
            f"""CREATE TABLE IF NOT EXISTS {status_table_fqn} (
                FV_FQN VARCHAR NOT NULL,
                SNAPSHOT_START_TIME TIMESTAMP_NTZ NOT NULL,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )"""
        ).collect(statement_params=self._telemetry_stmp)

    def _drop_orphaned_snapshot_table(self, feature_view_name: SqlIdentifier) -> None:
        """Drop the snapshot table and snapshot status row for a feature view.

        Used by ``register_feature_view(overwrite=True)`` to remove any snapshot
        table left over from a previous append_only registration of the same
        name/version.  The new registration is guaranteed non-append_only on the
        overwrite path (the append_only + overwrite combination is rejected
        upfront), so the prior snapshot state is unconditionally stale.

        Args:
            feature_view_name: The versioned physical name of the feature view, for
                example ``MY_FV$V1``.
        """
        snapshot_table_name = FeatureView._get_snapshot_table_name(feature_view_name)
        snapshot_fqn = self._get_fully_qualified_name(snapshot_table_name)
        self._session.sql(f"DROP TABLE IF EXISTS {snapshot_fqn}").collect(statement_params=self._telemetry_stmp)

        status_table_fqn = self._get_fully_qualified_name(SqlIdentifier(_SNAPSHOT_STATUS_TABLE))
        fully_qualified_name = self._get_fully_qualified_name(feature_view_name)
        fully_qualified_name_lit = snowpark_utils.escape_single_quotes(  # type: ignore[no-untyped-call]
            fully_qualified_name
        )
        try:
            self._session.sql(f"DELETE FROM {status_table_fqn} WHERE FV_FQN = '{fully_qualified_name_lit}'").collect(
                statement_params=self._telemetry_stmp
            )
        except SnowparkSQLException as e:
            logger.warning(
                "feature store: could not delete snapshot status row for %s: %s",
                fully_qualified_name,
                e,
            )

    def _cleanup_stale_feature_view_resources(
        self,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        fully_qualified_name: str,
        *,
        new_has_task: bool,
        drop_snapshot_table: Optional[bool] = None,
    ) -> None:
        """Drop stale feature-view resources during overwrite/delete operations.

        Called after the new offline object (View or Dynamic Table) has been
        successfully created on the ``register_feature_view(overwrite=True)``
        path.  Cleans up two classes of orphan from the prior registration:

        - **Stale Task.** Only dropped when the new FV does *not* itself have a
          Task (``new_has_task=False``).  Views never have one, and
          duration-based Dynamic Tables use the DT's internal scheduler.
          CRON-based DTs and ``append_only`` FVs do have a companion Task —
          but that case is handled by ``_create_scheduled_refresh_task`` with
          ``OR REPLACE``, which owns the Task lifecycle and takes precedence
          here.  ``IF EXISTS`` makes the drop a no-op when no prior Task exists.

        - **Stale snapshot table.** By default, dropped when the new FV is not
        ``append_only`` on overwrite paths because ``append_only=True`` + ``overwrite=True``
        is rejected upfront by ``register_feature_view``.  Delete flows pass
        ``drop_snapshot_table=True`` explicitly.
          ``IF EXISTS`` makes the drop a no-op when no prior snapshot exists.
        If ``drop_snapshot_table`` is ``None``, it defaults to
        ``not feature_view.append_only``.

        Args:
            feature_view: The newly registered feature view.
            feature_view_name: Versioned physical name (e.g. ``MY_FV$V1``).
            fully_qualified_name: Fully qualified name of the offline object
                (Dynamic Table or View) — also the Task name when one exists.
            new_has_task: True iff the registration is creating a Task for
                the new FV (CRON-based DT or ``append_only``).  When True,
                Task cleanup is skipped because the create-or-replace path
                already owns the Task lifecycle.
            drop_snapshot_table: Whether to drop the prior snapshot table.
                When ``None``, defaults to ``not feature_view.append_only``;
                delete flows pass ``True`` explicitly.
        """
        if not new_has_task:
            self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                statement_params=self._telemetry_stmp
            )
        if drop_snapshot_table is None:
            drop_snapshot_table = not feature_view.append_only
        if drop_snapshot_table:
            self._drop_orphaned_snapshot_table(feature_view_name)

    def _create_snapshot_table(
        self,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        dt_fqn: str,
        *,
        backfill_table: Optional[str] = None,
    ) -> str:
        """Create the snapshot accumulation table for an append-only feature view.

        Without ``backfill_table``: creates an empty table mirroring the DT
        schema via ``CREATE TABLE ... LIKE`` and applies clustering from
        ``feature_view.cluster_by``.

        With ``backfill_table``: zero-copy clones the backfill table, then
        reconciles schema via the extend-only helper — every DT column missing
        from the clone becomes ``ADD COLUMN``; the timestamp column and all
        entity join keys must already be present in the backfill (otherwise
        the helper rejects the call); and any backfill column not present in
        the DT, or any reorder/type change, is rejected. Clustering is then
        applied.

        The snapshot table is brand new at this point, so any failure after
        the initial ``CREATE TABLE`` (failed schema reconciliation, failed
        ``CLUSTER BY``, failed ``SET TAG``, …) is compensated by dropping the
        snapshot table — leaving a half-built table behind would block retries
        and confuse subsequent registration attempts. The ``ADD COLUMN``
        rollback that the extend-only helper would otherwise perform is
        deliberately suppressed (``rollback_cmds=[]``) because the outer
        ``DROP TABLE`` subsumes it.

        Args:
            feature_view: The feature view definition.
            feature_view_name: The versioned physical name (e.g. ``SNAP_FV$V1``).
            dt_fqn: Fully qualified name of the DT.
            backfill_table: Fully-qualified name of a historical snapshot table
                to clone.  When ``None``, creates an empty snapshot table.

        Returns:
            The fully qualified name of the created snapshot table.

        Raises:
            Exception: Re-raises any exception after best-effort cleanup of the
                half-built snapshot table (``DROP TABLE IF EXISTS``).
        """
        self._ensure_snapshot_status_table_exists()
        snapshot_table_name = FeatureView._get_snapshot_table_name(feature_view_name)
        snapshot_fqn = self._get_fully_qualified_name(snapshot_table_name)
        snapshot_obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.SNAPSHOT_TABLE, snowml_version.VERSION)

        if backfill_table is None:
            self._session.sql(f"CREATE TABLE {snapshot_fqn} LIKE {dt_fqn}").collect(
                statement_params=self._telemetry_stmp
            )
        else:
            self._session.sql(f"CREATE TABLE {snapshot_fqn} CLONE {backfill_table}").collect(
                statement_params=self._telemetry_stmp
            )

        try:
            if backfill_table is not None:
                # After CLONE, the snapshot mirrors the backfill schema. Reconcile to the DT
                # schema: each extra DT column becomes an ADD COLUMN; any backfill column
                # not present in the DT (or any reorder/type change) is rejected by the
                # extend-only helper. ``required_old_columns`` additionally fails with a
                # domain-friendly message when the user-supplied backfill is missing the
                # timestamp or any entity join key, which would otherwise either trigger a
                # confusing position-mismatch error (when those columns are at the front of
                # the DT, the common case) or silently land NULL key values for backfilled
                # rows (rare DT orderings where features come first).
                backfill_schema = fs_table_schema_evolution.schema_upper_name_map(
                    self._session.table(snapshot_fqn).schema
                )
                dt_schema = fs_table_schema_evolution.schema_upper_name_map(self._session.table(dt_fqn).schema)
                required_columns: list[str] = []
                if feature_view.timestamp_col:
                    required_columns.append(feature_view.timestamp_col.resolved())
                for entity in feature_view.entities:
                    for key in entity.join_keys:
                        required_columns.append(key.resolved())
                forward_cmds, _ = self._compute_extend_only_schema_evolution_cmds(
                    snapshot_fqn,
                    backfill_schema,
                    dt_schema,
                    required_old_columns=required_columns,
                )
                # rollback_cmds=[] — the outer except drops the entire snapshot table, so
                # per-column rollback would be wasted work.
                self._execute_extend_only_schema_evolution_cmds(forward_cmds, [])

            self._apply_snapshot_clustering(feature_view, snapshot_fqn)

            self._session.sql(
                f"""ALTER TABLE {snapshot_fqn}
                    SET TAG {self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}='{snapshot_obj_info.to_json()}'
                """
            ).collect(statement_params=self._telemetry_stmp)
        except Exception:
            try:
                self._session.sql(f"DROP TABLE IF EXISTS {snapshot_fqn}").collect(statement_params=self._telemetry_stmp)
            except Exception:
                logger.exception("Failed to drop half-built snapshot table %s", snapshot_fqn)
            raise

        return snapshot_fqn

    def _apply_snapshot_clustering(self, feature_view: FeatureView, snapshot_fqn: str) -> None:
        """Apply clustering to the snapshot table, matching the DT's cluster_by.

        Args:
            feature_view: The feature view whose cluster_by to apply.
            snapshot_fqn: Fully qualified name of the snapshot table.
        """
        if feature_view.cluster_by:
            cluster_expr = ", ".join(str(col) for col in feature_view.cluster_by)
            self._session.sql(f"ALTER TABLE {snapshot_fqn} CLUSTER BY ({cluster_expr})").collect(
                statement_params=self._telemetry_stmp
            )

    def _compute_extend_only_schema_evolution_cmds(
        self,
        table_fqn: str,
        old_schema: dict[str, DataType],
        new_schema: dict[str, DataType],
        *,
        required_old_columns: Optional[Iterable[str]] = None,
    ) -> tuple[list[str], list[str]]:
        """Compute the forward + rollback DDL to evolve ``table_fqn`` extend-only.

        Pure planning (no SQL is issued). Use together with
        :meth:`_execute_extend_only_schema_evolution_cmds` to run the forward DDL,
        and pass the rollback commands to a caller that needs to register a
        compensating action (e.g. :meth:`_recreate_append_only_feature_view_atomically`).

        Args:
            table_fqn: Fully qualified name of the table to evolve.
            old_schema: Existing column types in canonical order.
            new_schema: Target column types in canonical order.
            required_old_columns: Optional column names that must already be
                present in ``old_schema``. Use when ``old_schema`` describes
                user-supplied data (e.g. an offline backfill table) so the
                caller sees a domain-friendly missing-columns error before the
                structural prefix check fires.

        Returns:
            ``(forward_cmds, rollback_cmds)``. Both lists are empty when ``new_schema``
            equals ``old_schema``.

        Raises:
            SnowflakeMLException: If ``required_old_columns`` are absent from
                ``old_schema``, or if ``new_schema`` would drop, reorder, or
                change types of columns in ``old_schema``.
        """
        try:
            return fs_table_schema_evolution.get_table_schema_evolution_extend_only_commands(
                table_fqn,
                old_schema,
                new_schema,
                required_old_columns=required_old_columns,
            )
        except ValueError as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=e,
            )

    def _execute_extend_only_schema_evolution_cmds(
        self,
        forward_cmds: list[str],
        rollback_cmds: list[str],
    ) -> None:
        """Run ``forward_cmds`` sequentially; on failure, run ``rollback_cmds`` inline and re-raise.

        Inline rollback handles the partial-DDL case (cmd[k] succeeds, cmd[k+1] fails)
        so callers never see a half-applied schema. Each rollback step is best-effort:
        failures are logged but do not mask the original forward error.

        Args:
            forward_cmds: DDL to apply in order.
            rollback_cmds: DDL that undoes ``forward_cmds`` (typically the rollback list
                returned by :meth:`_compute_extend_only_schema_evolution_cmds`).

        Raises:
            Exception: Re-raised after best-effort rollback when one of the forward
                DDL statements fails partway through.
        """
        try:
            for cmd in forward_cmds:
                self._session.sql(cmd).collect(statement_params=self._telemetry_stmp)
        except Exception:
            for rollback_cmd in rollback_cmds:
                try:
                    self._session.sql(rollback_cmd).collect(statement_params=self._telemetry_stmp)
                except Exception:
                    logger.exception("Failed to roll back schema-evolution DDL: %s", rollback_cmd)
            raise

    def _build_snapshot_task_body(self, feature_view_name: SqlIdentifier, fully_qualified_name: str) -> str:
        """Build a Task SQL body that refreshes the DT and snapshots it.

        Uses a ``_SNAPSHOT_STATUS`` table to coordinate crash recovery.
        Idempotency is guaranteed by the transactional pairing of the
        snapshot INSERT and status row DELETE.

        On each Task execution:

        Branch 1 — status row exists (crash recovery):
          A previous run inserted the status row but the snapshot
          transaction did not commit. Check
          ``DYNAMIC_TABLE_REFRESH_HISTORY`` for a ``SUCCEEDED``
          refresh whose ``REFRESH_END_TIME`` is at or after
          ``SNAPSHOT_START_TIME``. If found, skip the next
          ``ALTER REFRESH`` and go straight to the common snapshot
          transaction — the DT already has the data the prior run
          was about to snapshot.

          Only ``SUCCEEDED`` refreshes are matched. In-progress
          (``RUNNING``) and ``FAILED`` refreshes are not detected
          here; the task falls through to branch 2 and issues a
          fresh ``ALTER REFRESH``. ``ALTER DYNAMIC TABLE … REFRESH``
          tolerates an in-flight refresh (queues or no-ops), so the
          extra call is harmless.

        Branch 2 — no status row, or branch 1 fell through:
          1. ``MERGE`` a status row keyed by ``FV_FQN`` with
             ``CURRENT_TIMESTAMP()`` as the start time. ``MATCHED``
             updates the existing row; ``NOT MATCHED`` inserts a
             new one.
          2. ``ALTER DYNAMIC TABLE … REFRESH``.
          3. Fall through to the common snapshot transaction.

        Common transaction (both branches):
          ``BEGIN TRANSACTION`` → INSERT DT rows into the snapshot table →
          DELETE the status row → ``COMMIT``.  If the transaction rolls
          back, the status row persists and the next run retries via
          Branch 1.

        Args:
            feature_view_name: The versioned physical name (e.g. ``SNAP_FV$V1``).
            fully_qualified_name: Fully qualified name of the DT.

        Returns:
            A Snowflake SQL scripting block (DECLARE ... BEGIN ... END).
        """
        snapshot_table_name = FeatureView._get_snapshot_table_name(feature_view_name)
        snapshot_fqn = self._get_fully_qualified_name(snapshot_table_name)
        status_table_fqn = self._get_fully_qualified_name(SqlIdentifier(_SNAPSHOT_STATUS_TABLE))

        # Defensively escape every value that gets baked into a SQL string literal in the
        # task body.  In practice these are derived from SqlIdentifiers and the configured
        # database/schema, none of which can carry a single quote today, but the task body
        # is persisted server-side and outlives the registration call — a future change to
        # the identifier-quoting rules or the FQN composition shouldn't be able to
        # silently break (or worse, inject SQL into) an already-deployed task.
        fv_fqn_lit = snowpark_utils.escape_single_quotes(fully_qualified_name)  # type: ignore[no-untyped-call]
        dt_name_lit = snowpark_utils.escape_single_quotes(feature_view_name.resolved())  # type: ignore[no-untyped-call]
        schema_lit = snowpark_utils.escape_single_quotes(  # type: ignore[no-untyped-call]
            self._config.schema.resolved()
        )

        history_after_query = f"""SELECT STATE
        FROM TABLE({self._config.database}.INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY(
            RESULT_LIMIT => 100
        ))
        WHERE NAME = '{dt_name_lit}'
        AND SCHEMA_NAME = '{schema_lit}'
        AND STATE IN ('SUCCEEDED')
        AND REFRESH_END_TIME >= :snapshot_start_time
        ORDER BY REFRESH_END_TIME DESC
        LIMIT 1"""

        # Use ``SELECT *`` so the persisted task body is schema-agnostic — schema-extension
        # updates do not need to rebuild it to capture new columns. The snapshot table is
        # created via ``CREATE TABLE ... CLONE`` from the DT and kept in lock-step with the
        # DT via extend-only ALTER ADD COLUMN, so in steady state ``snapshot.cols`` equals
        # ``dt.cols`` (same names, same order) and the positional ``SELECT *`` is correct.
        # During the brief window of an ``update_feature_view`` where the snapshot has been
        # extended but the DT has not yet been recreated, an in-flight task fire would see
        # ``snapshot.cols > dt.cols`` and fail with a column-count mismatch; the status-row
        # crash-recovery branch above retries on the next fire after the DT recreate
        # completes (one cycle of data is delayed; no data is lost).
        snapshot_insert = f"""INSERT INTO {snapshot_fqn}
            SELECT * FROM {fully_qualified_name}"""

        return f"""DECLARE
    snapshot_start_time TIMESTAMP_NTZ;
    refresh_state VARCHAR;
BEGIN
    -- Check status table for a pending snapshot (crash recovery)
    SELECT SNAPSHOT_START_TIME INTO :snapshot_start_time
        FROM {status_table_fqn}
        WHERE FV_FQN = '{fv_fqn_lit}'
        LIMIT 1;
    LET skip_dynamic_table_refresh BOOLEAN := false;
    IF (snapshot_start_time IS NOT NULL) THEN
        -- Branch 1: Status row exists — previous run did not complete.
        -- Check if a DT refresh already happened after snapshot_start_time.
        SELECT STATE INTO :refresh_state FROM ({history_after_query});

        IF (refresh_state IS NOT NULL) THEN
            skip_dynamic_table_refresh := true;
        END IF;
    END IF;

    IF (:skip_dynamic_table_refresh = false) THEN
        -- Branch 2: Fresh run — record start time and refresh DT.
        MERGE INTO {status_table_fqn} AS tgt
            USING (SELECT '{fv_fqn_lit}' AS FV_FQN, CURRENT_TIMESTAMP() AS SNAPSHOT_START_TIME) AS src
            ON tgt.FV_FQN = src.FV_FQN
            WHEN MATCHED THEN UPDATE SET tgt.SNAPSHOT_START_TIME = src.SNAPSHOT_START_TIME
            WHEN NOT MATCHED THEN INSERT (FV_FQN, SNAPSHOT_START_TIME) VALUES (src.FV_FQN, src.SNAPSHOT_START_TIME);

        ALTER DYNAMIC TABLE {fully_qualified_name} REFRESH;
    END IF;

    -- Common: Atomic snapshot INSERT + status cleanup
    BEGIN TRANSACTION;
        {snapshot_insert};
        DELETE FROM {status_table_fqn} WHERE FV_FQN = '{fv_fqn_lit}';
    COMMIT;
END;"""

    def _create_scheduled_refresh_task(
        self,
        override_clause: str,
        feature_view: FeatureView,
        fully_qualified_name: str,
        warehouse: SqlIdentifier,
        *,
        feature_view_name: Optional[SqlIdentifier] = None,
    ) -> None:
        """Create a Snowflake Task to refresh a Dynamic Table on a CRON schedule.

        For snapshot-enabled (append_only) feature views, the task also appends the
        current DT contents to the snapshot table.

        Note:
            Dynamic Iceberg tables are managed with ALTER DYNAMIC TABLE (no ICEBERG keyword),
            so this approach works the same way for both regular Dynamic Tables and
            Dynamic Iceberg Tables.

        Args:
            override_clause: SQL clause for CREATE OR REPLACE behavior.
            feature_view: The FeatureView containing refresh configuration.
            fully_qualified_name: Fully qualified name for the task (same as the DT).
            warehouse: Warehouse to use for task execution.
            feature_view_name: The versioned physical name (required for append_only FVs).

        Raises:
            SnowflakeMLException: [INTERNAL_SNOWML_ERROR] when ``feature_view.append_only``
                is True but ``feature_view_name`` is not provided.
            Exception: Re-raises any exception after best-effort cleanup. Cleanup is gated
                on ``override_clause`` — see the inline comment in the except block.
        """
        task_obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK, snowml_version.VERSION)
        try:
            if feature_view.append_only:
                if feature_view_name is None:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_SNOWML_ERROR,
                        original_exception=RuntimeError(
                            "Feature Store: append-only feature views require the versioned physical name "
                            "to build the refresh task."
                        ),
                    )
                task_body = self._build_snapshot_task_body(feature_view_name, fully_qualified_name)
            else:
                task_body = f"ALTER DYNAMIC TABLE {fully_qualified_name} REFRESH"

            self._session.sql(
                f"""CREATE{override_clause} TASK {fully_qualified_name}
                    WAREHOUSE = {warehouse}
                    SCHEDULE = 'USING CRON {feature_view.refresh_freq}'
                    AS {task_body}
                """
            ).collect(statement_params=self._telemetry_stmp)
            self._session.sql(
                f"""
                ALTER TASK {fully_qualified_name}
                SET TAG {self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}='{task_obj_info.to_json()}'
            """
            ).collect(statement_params=self._telemetry_stmp)
            self._session.sql(f"ALTER TASK {fully_qualified_name} RESUME").collect(
                statement_params=self._telemetry_stmp
            )
        except Exception:
            # Internal cleanup runs only on the fresh-create path (override_clause empty).
            #
            # On the OR REPLACE path, the caller is authoritative for rollback:
            #   - register_feature_view(overwrite=True) deliberately does not roll back
            #     created_resources (see register_feature_view's except block); leaving
            #     the just-replaced DT in place lets the user retry register without
            #     losing the prior FV's downstream consumers' object identity.
            #   - update_feature_view(updated_feature_df=...) drives recreate via
            #     _recreate_append_only_feature_view_atomically, which registers
            #     compensating actions for each forward step. Dropping the DT here
            #     would defeat those compensating actions (e.g. an online-table
            #     rollback that recreates ``ONLINE FEATURE TABLE FROM <dt>`` would
            #     fail because we just dropped the DT).
            is_fresh_create = not override_clause.strip()  # "" → True, " OR REPLACE" → False
            if is_fresh_create:
                self._session.sql(f"DROP DYNAMIC TABLE IF EXISTS {fully_qualified_name}").collect(
                    statement_params=self._telemetry_stmp
                )
                self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                    statement_params=self._telemetry_stmp
                )
                if feature_view.append_only and feature_view_name is not None:
                    snap_name = FeatureView._get_snapshot_table_name(feature_view_name)
                    self._session.sql(f"DROP TABLE IF EXISTS {self._get_fully_qualified_name(snap_name)}").collect(
                        statement_params=self._telemetry_stmp
                    )
            raise

    def _create_rollup_feature_view(
        self,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        fully_qualified_name: str,
        tagging_clause_str: str,
        overwrite: bool,
        created_resources: list[tuple[_FeatureStoreObjTypes, str]],
    ) -> None:
        """Create a rollup feature view Dynamic Table.

        Rollup feature views aggregate tiles from a parent tiled FV to a coarser
        entity level using a mapping table.

        Resources are appended to ``created_resources`` as they are successfully
        created so the caller can roll them back even if this method raises
        partway through (e.g. DT succeeds, then Task creation fails).

        Args:
            feature_view: The rollup feature view definition.
            feature_view_name: The physical name for the feature view.
            fully_qualified_name: Fully qualified name for the created DT.
            tagging_clause_str: Tagging clause for the CREATE statement.
            overwrite: Whether to replace existing objects.
            created_resources: Mutable list that this method appends successfully
                created resources to (in creation order) for caller-side rollback.

        Raises:
            SnowflakeMLException: If rollup_config is invalid or DT creation fails.
        """
        rollup_config = feature_view.rollup_config
        if rollup_config is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("Rollup feature view must have rollup_config"),
            )

        parent_fv = rollup_config.source

        # Get parent tile table name
        physical_name = FeatureView._get_physical_name(parent_fv.name, parent_fv.version)  # type: ignore[arg-type]
        parent_tile_table = f"{parent_fv.database}.{parent_fv.schema}.{physical_name}"

        # Get join keys
        parent_join_keys = [str(k) for e in parent_fv.entities for k in e.join_keys]
        new_join_keys = [str(k) for e in feature_view.entities for k in e.join_keys]

        # Get mapping query from DataFrame
        mapping_query = rollup_config.mapping_df.queries["queries"][-1]

        # Build and store RollupMetadata on the feature view for PIT training support
        from snowflake.ml.feature_store.feature_view import RollupMetadata

        rollup_metadata = RollupMetadata(
            parent_tile_table=parent_tile_table,
            parent_join_keys=parent_join_keys,
            mapping_query=mapping_query,
            mapping_valid_from_col=rollup_config.mapping_valid_from_col,
            mapping_valid_to_col=rollup_config.mapping_valid_to_col,
        )
        feature_view._rollup_metadata = rollup_metadata

        # For the materialized DT (inference path), use a flat JOIN.
        # When temporal columns are set (SCD Type 2), filter to currently-active
        # mappings only. No ROW_NUMBER dedup — supports 1:N mappings where one
        # child entity maps to multiple parent entities simultaneously.
        flat_mapping_query = mapping_query
        if rollup_config.mapping_valid_from_col is not None and rollup_config.mapping_valid_to_col is not None:
            vfc = SqlIdentifier(rollup_config.mapping_valid_from_col).identifier()
            vtc = SqlIdentifier(rollup_config.mapping_valid_to_col).identifier()
            flat_mapping_query = (
                f"SELECT * FROM ({mapping_query}) "
                f"WHERE {vfc} <= CURRENT_TIMESTAMP() AND ({vtc} IS NULL OR {vtc} > CURRENT_TIMESTAMP())"
            )

        # The rollup FV inherits its parent's authoring version (set at
        # registration), so its tile-column naming matches the parent's DT.
        generator = RollupSqlGenerator(
            parent_tile_table=parent_tile_table,
            parent_join_keys=parent_join_keys,
            new_join_keys=new_join_keys,
            mapping_query=flat_mapping_query,
            aggregation_specs=feature_view.aggregation_specs or [],
            authoring_pkg_version=feature_view.authoring_pkg_version,
        )
        rollup_sql = generator.generate()

        # Create the Dynamic Table
        warehouse = feature_view.warehouse if feature_view.warehouse is not None else self._default_warehouse
        refresh_freq = feature_view.refresh_freq or "DOWNSTREAM"
        overwrite_clause = " OR REPLACE" if overwrite else ""

        # If refresh_freq is a cron, we need to set TARGET_LAG = 'DOWNSTREAM' and create a Task.
        schedule_task = feature_view_refresh_freq._is_cron_refresh_freq(refresh_freq)
        target_lag = "DOWNSTREAM" if schedule_task else refresh_freq

        try:
            query = f"""CREATE{overwrite_clause} DYNAMIC TABLE {fully_qualified_name}
                TARGET_LAG = '{target_lag}'
                COMMENT = {_sql_string_literal(feature_view.desc)}
                TAG (
                    {tagging_clause_str}
                )
                WAREHOUSE = {warehouse}{_initialization_warehouse_clause(feature_view)}
                REFRESH_MODE = {feature_view.refresh_mode}
                INITIALIZE = {feature_view.initialize}
            """
            if feature_view.cluster_by:
                cluster_by_clause = f"CLUSTER BY ({', '.join(str(c) for c in feature_view.cluster_by)})"
                query += f"{cluster_by_clause}"

            query += f"""
                AS {rollup_sql}
            """
            self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            created_resources.append((_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, fully_qualified_name))

            # Create scheduled task after DT creation for cron-based refresh
            if schedule_task:
                task_overwrite_clause = " OR REPLACE" if overwrite else ""
                self._create_scheduled_refresh_task(
                    task_overwrite_clause,
                    feature_view,
                    fully_qualified_name,
                    warehouse,
                )
                created_resources.append(
                    (
                        _FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK,
                        fully_qualified_name,
                    )
                )
            elif overwrite:
                # Clean up any existing task when overwriting with a non-CRON feature view.
                # This handles the case: CRON rollup DT → duration/DOWNSTREAM rollup DT
                self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                    statement_params=self._telemetry_stmp
                )

        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Create rollup dynamic table [\n{query}\n] failed: {e}."),
            ) from e

    def _create_offline_feature_view(
        self,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        fully_qualified_name: str,
        column_descs: str,
        tagging_clause_str: str,
        block: bool,
        overwrite: bool,
        created_resources: list[tuple[_FeatureStoreObjTypes, str]],
    ) -> None:
        """Create the offline representation for a feature view.

        Depending on `refresh_freq`, this creates either a Dynamic Table (managed feature view)
        or a View (external feature view). Resources are appended to ``created_resources`` as
        they are successfully created so the caller can roll them back even if this method
        raises partway through (e.g. DT created, then snapshot-table reconciliation fails for
        an ``append_only`` FV).

        Args:
            feature_view: The feature view definition to materialize.
            feature_view_name: The physical name object for the feature view.
            fully_qualified_name: Fully qualified name for the created view/dynamic table.
            column_descs: Column descriptions clause used in the CREATE statement.
            tagging_clause_str: Tagging clause used in the CREATE statement.
            block: Whether to block until the initial refresh completes when applicable.
            overwrite: Whether to replace existing objects if they already exist.
            created_resources: Mutable list that this method appends successfully created
                resources to (in creation order) for caller-side rollback.

        Raises:
            SnowflakeMLException: [RuntimeError] If creating the view or dynamic table fails.
        """
        refresh_freq = feature_view.refresh_freq

        # External feature view via View (no refresh schedule)
        if refresh_freq is None:
            overwrite_clause = " OR REPLACE" if overwrite else ""
            query = self._create_offline_feature_view_view_query(
                overwrite_clause,
                fully_qualified_name,
                column_descs,
                feature_view,
                tagging_clause_str,
            )
            try:
                self._session.sql(query).collect(statement_params=self._telemetry_stmp)
                created_resources.append((_FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW, fully_qualified_name))
            except Exception as e:
                # Only check for type conflict when SQL_COMPILATION_ERROR is thrown
                if (
                    overwrite
                    and isinstance(e, SnowparkSQLException)
                    and e.error_code == error_codes.SQL_COMPILATION_ERROR
                    and self._get_existing_feature_view_object_type(feature_view_name) == "DYNAMIC TABLE"
                ):
                    # A DYNAMIC TABLE exists but we want to create a VIEW - use shadow swap
                    logger.info(
                        f"Existing DYNAMIC TABLE detected, performing shadow swap to VIEW for "
                        f"{fully_qualified_name}"
                    )
                    self._shadow_swap_to_view(
                        fully_qualified_name,
                        column_descs,
                        feature_view,
                        tagging_clause_str,
                        block,
                    )
                    created_resources.append(
                        (
                            _FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW,
                            fully_qualified_name,
                        )
                    )
                else:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                        original_exception=RuntimeError(f"Create view {fully_qualified_name} failed: {e}"),
                    ) from e
            if overwrite:
                # Views never have a companion Task, so always clean up any
                # task left over from a prior CRON-based DT registration.
                self._cleanup_stale_feature_view_resources(
                    feature_view,
                    feature_view_name,
                    fully_qualified_name,
                    new_has_task=False,
                )
            return
        # Managed feature view via Dynamic Table (and optional Task)
        #
        # Refresh behavior based on refresh_freq input type:
        # ┌────────────────────┬───────────────┬─────────────────┬──────────────────────────────────────────┐
        # │ Input Type         │ schedule_task │ TARGET_LAG      │ Who triggers the refresh?                │
        # ├────────────────────┼───────────────┼─────────────────┼──────────────────────────────────────────┤
        # │ Duration ('5m')    │ False         │ '5 minutes'     │ The Dynamic Table's internal scheduler.  │
        # │ CRON ('* * * UTC') │ True          │ 'DOWNSTREAM'    │ An external Snowflake Task.              │
        # │ DOWNSTREAM         │ False         │ 'DOWNSTREAM'    │ Only a consumer (or manual refresh).     │
        # └────────────────────┴───────────────┴─────────────────┴──────────────────────────────────────────┘
        #
        # Note: Dynamic Iceberg tables are managed with ALTER DYNAMIC TABLE (no ICEBERG keyword),
        # so CRON-based refresh works the same way as regular dynamic tables.
        schedule_task = feature_view_refresh_freq._is_cron_refresh_freq(refresh_freq)
        warehouse = feature_view.warehouse if feature_view.warehouse is not None else self._default_warehouse
        self._create_dynamic_table(
            feature_view_name,
            feature_view,
            fully_qualified_name,
            column_descs,
            tagging_clause_str,
            schedule_task,
            warehouse,
            block,
            overwrite,
        )
        created_resources.append((_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, fully_qualified_name))

        if feature_view.append_only:
            # Pre-register the snapshot table for rollback before invoking
            # _create_snapshot_table — that method runs CREATE TABLE … CLONE
            # first and may then raise during schema reconciliation, leaving a
            # partially-created table that the caller must clean up.  The FQN
            # is fully determined by feature_view_name, so we can append it
            # ahead of time; rollback uses DROP TABLE IF EXISTS so it's a no-op
            # if the table was never actually created.
            snapshot_table_name = FeatureView._get_snapshot_table_name(feature_view_name)
            snapshot_fqn = self._get_fully_qualified_name(snapshot_table_name)
            created_resources.append((_FeatureStoreObjTypes.SNAPSHOT_TABLE, snapshot_fqn))
            self._create_snapshot_table(
                feature_view,
                feature_view_name,
                fully_qualified_name,
                backfill_table=feature_view.backup_source,
            )

        if schedule_task:
            task_override_clause = " OR REPLACE" if overwrite else ""
            self._create_scheduled_refresh_task(
                task_override_clause,
                feature_view,
                fully_qualified_name,
                warehouse,
                feature_view_name=(feature_view_name if feature_view.append_only else None),
            )
            created_resources.append((_FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK, fully_qualified_name))

        if overwrite:
            self._cleanup_stale_feature_view_resources(
                feature_view,
                feature_view_name,
                fully_qualified_name,
                new_has_task=schedule_task,
            )

    def _shadow_replace(
        self,
        old_obj_fqn: str,
        old_obj_type: str,
        new_obj_fqn: str,
        new_obj_type: str,
        fully_qualified_name: str,
    ) -> None:
        try:
            # Swap: drop old <object type>, rename new <object type> to target
            self._session.sql(
                f"""ALTER {old_obj_type} IF EXISTS {fully_qualified_name} RENAME TO {old_obj_fqn}"""
            ).collect()
            self._session.sql(
                f"""ALTER {new_obj_type} IF EXISTS {new_obj_fqn} RENAME TO {fully_qualified_name}"""
            ).collect()
            self._session.sql(f"""DROP {old_obj_type} IF EXISTS {old_obj_fqn}""").collect()
        except Exception as tx_e:
            # If rename fails, recover at best-effort
            self._session.sql(
                f"""ALTER {old_obj_type} IF EXISTS {old_obj_fqn} RENAME TO {fully_qualified_name}"""
            ).collect()
            self._session.sql(f"""DROP {new_obj_type} IF EXISTS {new_obj_fqn}""").collect()
            raise tx_e

        logger.info(f"Shadow swap for {fully_qualified_name} completed successfully.")

    def _get_existing_feature_view_object_type(self, feature_view_name: SqlIdentifier) -> Optional[str]:
        """Check if a VIEW or DYNAMIC TABLE with the given name already exists.

        Args:
            feature_view_name: The name of the feature view object to check.

        Returns:
            "VIEW" if a view exists, "DYNAMIC TABLE" if a dynamic table exists,
            or None if no object with that name exists.
        """
        # Check for existing VIEW
        found_views = self._find_object("VIEWS", feature_view_name)
        if len(found_views) > 0:
            return "VIEW"

        # Check for existing DYNAMIC TABLE
        found_dts = self._find_object("DYNAMIC TABLES", feature_view_name)
        if len(found_dts) > 0:
            return "DYNAMIC TABLE"

        return None

    def _shadow_swap_to_view(
        self,
        fully_qualified_name: str,
        column_descs: str,
        feature_view: FeatureView,
        tagging_clause_str: str,
        block: bool,
    ) -> None:
        """Replace an existing DYNAMIC TABLE with a VIEW using shadow swap.

        Creates a temporary VIEW, then atomically swaps it with the existing DYNAMIC TABLE.
        Also cleans up any scheduled refresh task and online feature table associated with the DYNAMIC TABLE.

        Args:
            fully_qualified_name: Fully qualified name for the target object.
            column_descs: Column descriptions clause used in the CREATE statement.
            feature_view: The feature view definition.
            tagging_clause_str: Tagging clause used in the CREATE statement.
            block: Whether to block until completion.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to replace dynamic table with view.
        """
        import uuid

        temp_suffix = uuid.uuid4().hex[:8]
        db, schema, obj = parse_fully_qualified_name(fully_qualified_name)
        new_view_name = identifier.concat_names([obj.identifier(), self._TMP_VIEW_PREFIX, temp_suffix])
        new_table_name = identifier.concat_names([obj.identifier(), self._TMP_DT_PREFIX, temp_suffix])
        temp_view_fqn = get_fully_qualified_name(db, schema, SqlIdentifier(new_view_name))
        temp_table_fqn = get_fully_qualified_name(db, schema, SqlIdentifier(new_table_name))

        # Create the new version as a shadow view first
        query = self._create_offline_feature_view_view_query(
            "",  # Don't use override_clause for temp objects
            temp_view_fqn,
            column_descs,
            feature_view,
            tagging_clause_str,
        )

        try:
            self._session.sql(query).collect(block=block, statement_params=self._telemetry_stmp)
            self._shadow_replace(
                temp_table_fqn,
                "DYNAMIC TABLE",
                temp_view_fqn,
                "VIEW",
                fully_qualified_name,
            )

            logger.info(f"Successfully replaced DYNAMIC TABLE with VIEW for {fully_qualified_name}")
        except Exception as e:
            # Cleanup shadow object if creation failed
            self._session.sql(f"""DROP VIEW IF EXISTS {temp_view_fqn}""").collect()
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Shadow swap to VIEW for {fully_qualified_name} failed: {e}"),
            ) from e

    def _shadow_swap_to_dynamic_table(
        self,
        feature_view: FeatureView,
        fully_qualified_name: str,
        column_descs: str,
        tagging_clause_str: str,
        schedule_task: bool,
        warehouse: SqlIdentifier,
        block: bool,
    ) -> None:
        """Replace an existing VIEW with a DYNAMIC TABLE using shadow swap.

        Creates a temporary DYNAMIC TABLE, then atomically swaps it with the existing VIEW.

        Args:
            feature_view: The feature view definition.
            fully_qualified_name: Fully qualified name for the target object.
            column_descs: Column descriptions clause used in the CREATE statement.
            tagging_clause_str: Tagging clause used in the CREATE statement.
            schedule_task: Whether a scheduled refresh task will be created (affects TARGET_LAG).
            warehouse: The warehouse to use for the dynamic table.
            block: Whether to block until completion.

        raises:
            SnowflakeMLException: [RuntimeError] Failed to replace view with dynamic table.
        """
        import uuid

        temp_suffix = uuid.uuid4().hex[:8]
        db, schema, obj = parse_fully_qualified_name(fully_qualified_name)
        new_view_name = identifier.concat_names([obj.identifier(), self._TMP_VIEW_PREFIX, temp_suffix])
        new_table_name = identifier.concat_names([obj.identifier(), self._TMP_DT_PREFIX, temp_suffix])
        temp_view_fqn = get_fully_qualified_name(db, schema, SqlIdentifier(new_view_name))
        temp_table_fqn = get_fully_qualified_name(db, schema, SqlIdentifier(new_table_name))

        # Create the new version as a shadow dynamic table first
        query = self._create_dynamic_table_query(
            "",  # Don't use override_clause for temp objects
            temp_table_fqn,
            column_descs,
            schedule_task,
            feature_view,
            tagging_clause_str,
            warehouse,
        )

        try:
            self._session.sql(query).collect(block=block, statement_params=self._telemetry_stmp)
            self._shadow_replace(temp_view_fqn, "VIEW", temp_table_fqn, "TABLE", fully_qualified_name)

            logger.info(f"Successfully replaced VIEW with DYNAMIC TABLE for {fully_qualified_name}")
        except Exception as e:
            # Cleanup shadow object if creation failed
            self._session.sql(f"""DROP DYNAMIC TABLE IF EXISTS {temp_table_fqn}""").collect()
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(f"Shadow swap to DYNAMIC TABLE for {fully_qualified_name} failed: {e}"),
            ) from e

    def _create_offline_feature_view_view_query(
        self,
        overwrite_clause: str,
        view_name: str,
        column_descs: str,
        feature_view: FeatureView,
        tagging_clause_str: str,
    ) -> str:
        return f"""CREATE{overwrite_clause} VIEW {view_name} ({column_descs})
            COMMENT = {_sql_string_literal(feature_view.desc)}
            TAG (
                {tagging_clause_str}
            )
            AS {feature_view.query}
        """

    def _check_dynamic_table_refresh_mode(self, feature_view_name: SqlIdentifier) -> None:
        found_dts = self._find_object("DYNAMIC TABLES", feature_view_name)
        if len(found_dts) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Can not find dynamic table: `{feature_view_name}`."),
            )
        if found_dts[0]["refresh_mode"] != "INCREMENTAL":
            warnings.warn(
                "Your pipeline won't be incrementally refreshed due to: "
                + f"\"{found_dts[0]['refresh_mode_reason']}\".",
                stacklevel=2,
                category=UserWarning,
            )

    def _validate_entity_exists(self, name: SqlIdentifier) -> bool:
        full_entity_tag_name = self._get_entity_name(name)
        found_rows = self._find_object("TAGS", full_entity_tag_name)
        return len(found_rows) == 1

    def _get_feature_prefix(
        self,
        feature_view: Union[FeatureView, FeatureViewSlice],
        auto_prefix: bool,
    ) -> Optional[str]:
        """Thin wrapper over :func:`feature_view.get_feature_prefix`.

        Kept on the class for backward compatibility with internal callers.

        Args:
            feature_view: The feature view instance.
            auto_prefix: Whether to auto-generate prefix.

        Returns:
            Prefix string with trailing underscore, or ``None``.
        """
        return fv_mod.get_feature_prefix(feature_view, auto_prefix)

    def _build_cte_query(
        self,
        feature_views: list[FeatureView],
        feature_columns: list[str],
        spine_ref: str,
        spine_timestamp_col: Optional[SqlIdentifier],
        include_feature_view_timestamp_col: bool = False,
        auto_prefix: bool = False,
        is_training: bool = False,
    ) -> str:
        """
        Build a CTE query with the spine query and the feature views.

        This method supports feature views with different join keys by:
        1. Creating a spine CTE that includes all possible join keys
        2. For each feature view, creating a deduplicated spine subquery with only that FV's join keys
        3. For tiled FVs: Using MergingSqlGenerator to generate tile merging CTEs
        4. For non-tiled FVs: Performing ASOF JOINs on the deduplicated spine when timestamp columns exist
        5. Performing LEFT JOINs on the deduplicated spine when timestamp columns are missing
        6. Combining results by LEFT JOINing each FV CTE back to the original SPINE

        For large numbers of feature views (> _CTE_MERGE_BATCH_SIZE), the final join is restructured
        into batched merge CTEs to bound the join depth per batch. FVs are grouped by entity key
        signature, and each group is chunked into batches. Non-primary batches use a deduplicated
        spine (SELECT DISTINCT on their entity keys) to reduce cardinality for narrower entity groups.

        For temporal rollup FVs in training mode (``is_training=True`` and
        ``rollup_metadata.mapping_valid_from_col`` is set), an additional CTE is
        injected that range JOINs the parent tile table with the temporal mapping to
        produce PIT-correct rollup tiles using bounded validity windows
        ``[valid_from, valid_to)``. This supports 1:N mappings. This CTE replaces
        the flat DT as input to MergingSqlGenerator.

        Args:
            feature_views: A list of feature views to join.
            feature_columns: A list of feature column strings for each feature view.
            spine_ref: The spine query.
            spine_timestamp_col: The timestamp column from spine. Can be None if spine has no timestamp column.
            include_feature_view_timestamp_col: Whether to include the timestamp column of
                the feature view in the result. Default to false.
            auto_prefix: Whether to automatically prefix feature columns.
            is_training: If True, use PIT-correct entity mappings for temporal rollup FVs.

        Returns:
            A SQL query string with CTE structure for joining feature views.
        """
        if not feature_views:
            return f"SELECT * FROM ({spine_ref})"

        # Create spine CTE with the spine query for reuse
        spine_cte = f"""SPINE AS (
            SELECT * FROM ({spine_ref})
        )"""

        ctes = [spine_cte]
        cte_names = []
        for i, feature_view in enumerate(feature_views):
            cte_name = f"FV{i:03d}"
            cte_names.append(cte_name)

            feature_timestamp_col = feature_view.timestamp_col

            # Get the specific join keys for this feature view
            fv_join_keys = list({k for e in feature_view.entities for k in e.join_keys})
            join_keys_str = ", ".join(fv_join_keys)

            # Handle tiled feature views using MergingSqlGenerator
            if feature_view.is_tiled and spine_timestamp_col is not None:
                assert feature_timestamp_col is not None
                tile_table = feature_view.fully_qualified_name()

                # For temporal rollup FVs in training mode, inject a PIT-correct
                # rollup CTE that range JOINs parent tiles with the temporal mapping
                # using bounded [valid_from, valid_to) windows. Supports 1:N mappings.
                rm = feature_view.rollup_metadata
                if is_training and rm is not None and rm.mapping_valid_from_col is not None:
                    logger.info(
                        f"Injecting PIT-correct rollup CTE for {feature_view.name} "
                        f"using range JOIN with mapping_valid_from_col={rm.mapping_valid_from_col}, "
                        f"mapping_valid_to_col={rm.mapping_valid_to_col}. "
                        f"Tiles outside the mapping validity window will be excluded."
                    )
                    new_join_keys = [str(k) for k in fv_join_keys]
                    pit_generator = RollupSqlGenerator(
                        parent_tile_table=rm.parent_tile_table,
                        parent_join_keys=rm.parent_join_keys,
                        new_join_keys=new_join_keys,
                        mapping_query=rm.mapping_query,
                        aggregation_specs=feature_view.aggregation_specs or [],
                        mapping_valid_from_col=rm.mapping_valid_from_col,
                        mapping_valid_to_col=rm.mapping_valid_to_col,
                        authoring_pkg_version=feature_view.authoring_pkg_version,
                    )
                    pit_cte_name = f"PIT_ROLLUP_FV{i}"
                    cte_name_str, cte_body = pit_generator.generate_as_cte(pit_cte_name)
                    ctes.append(f"{cte_name_str} AS (\n{cte_body}\n)")
                    tile_table = pit_cte_name

                generator = MergingSqlGenerator(
                    tile_table=tile_table,
                    join_keys=fv_join_keys,
                    timestamp_col=feature_timestamp_col,
                    feature_granularity=feature_view.feature_granularity,  # type: ignore[arg-type]
                    features=feature_view.aggregation_specs,  # type: ignore[arg-type]
                    spine_timestamp_col=spine_timestamp_col,
                    fv_index=i,
                    authoring_pkg_version=feature_view.authoring_pkg_version,
                )
                # Add all CTEs from the merging generator
                for cte_tuple in generator.generate_all_ctes():
                    ctes.append(f"{cte_tuple[0]} AS (\n{cte_tuple[1]}\n)")

            # Use ASOF JOIN if both spine and feature view have timestamp columns, otherwise use LEFT JOIN
            elif spine_timestamp_col is not None and feature_timestamp_col is not None:
                # Build the deduplicated spine columns set (join keys + timestamp)
                spine_dedup_cols_set = set(fv_join_keys)
                if spine_timestamp_col not in spine_dedup_cols_set:
                    spine_dedup_cols_set.add(spine_timestamp_col)
                spine_dedup_cols_str = ", ".join(col.identifier() for col in spine_dedup_cols_set)

                # Build the JOIN condition using only this feature view's join keys
                join_conditions_dedup = [
                    f"SPINE_DEDUP.{col.identifier()} = FEATURE.{col.identifier()}" for col in fv_join_keys
                ]

                quoted_spine_ts = spine_timestamp_col.identifier()
                quoted_fv_ts = feature_timestamp_col.identifier()

                if include_feature_view_timestamp_col:
                    f_ts_col_alias = identifier.concat_names(
                        [
                            feature_view.name,
                            "_",
                            str(feature_view.version),
                            "_",
                            feature_timestamp_col,
                        ]
                    )
                    f_ts_col_str = f"FEATURE.{feature_timestamp_col} AS {f_ts_col_alias},"
                else:
                    f_ts_col_str = ""
                if feature_view.append_only:
                    asof_table_name = feature_view.fully_qualified_snapshot_table_name()
                else:
                    asof_table_name = feature_view.fully_qualified_name()
                ctes.append(
                    f"""{cte_name} AS (
    SELECT
        SPINE_DEDUP.*,
        {f_ts_col_str}
        FEATURE.* EXCLUDE ({join_keys_str}, {feature_timestamp_col})
    FROM (
        SELECT DISTINCT {spine_dedup_cols_str}
        FROM SPINE
    ) SPINE_DEDUP
    ASOF JOIN (
        SELECT {join_keys_str}, {feature_timestamp_col}, {feature_columns[i]}
        FROM {asof_table_name}
    ) FEATURE
    MATCH_CONDITION (SPINE_DEDUP.{quoted_spine_ts} >= FEATURE.{quoted_fv_ts})
    ON {" AND ".join(join_conditions_dedup)}
)"""
                )
            else:
                # Build the deduplicated spine columns list (just join keys, no timestamp)
                spine_dedup_cols_str = ", ".join(col.identifier() for col in fv_join_keys)

                # Build the JOIN condition using only this feature view's join keys
                join_conditions_dedup = [
                    f"SPINE_DEDUP.{col.identifier()} = FEATURE.{col.identifier()}" for col in fv_join_keys
                ]

                ctes.append(
                    f"""{cte_name} AS (
    SELECT
        SPINE_DEDUP.*,
        FEATURE.* EXCLUDE ({join_keys_str})
    FROM (
        SELECT DISTINCT {spine_dedup_cols_str}
        FROM SPINE
    ) SPINE_DEDUP
    LEFT JOIN (
        SELECT {join_keys_str}, {feature_columns[i]}
        FROM {feature_view.fully_qualified_name()}
    ) FEATURE
    ON {" AND ".join(join_conditions_dedup)}
)"""
                )

        # Collect per-FV metadata for the final join assembly
        fv_infos: list[_FvJoinInfo] = []
        for i, cte_name in enumerate(cte_names):
            feature_view = feature_views[i]
            fv_join_keys = list({k for e in feature_view.entities for k in e.join_keys})
            has_ts = spine_timestamp_col is not None and feature_view.timestamp_col is not None
            key_sig = (tuple(sorted(k.resolved() for k in fv_join_keys)), has_ts)

            select_cols: list[str] = []
            col_names: list[str] = []
            if (
                include_feature_view_timestamp_col
                and feature_view.timestamp_col is not None
                and not feature_view.is_tiled
            ):
                f_ts_col_alias = identifier.concat_names(
                    [
                        feature_view.name,
                        "_",
                        str(feature_view.version),
                        "_",
                        feature_view.timestamp_col,
                    ]
                )
                select_cols.append(f"{cte_name}.{f_ts_col_alias} AS {f_ts_col_alias}")
                col_names.append(f_ts_col_alias)

            prefix = self._get_feature_prefix(feature_view, auto_prefix)
            if feature_view.is_tiled and feature_view.aggregation_specs:
                for spec in feature_view.aggregation_specs:
                    col_name = spec.get_sql_column_name()
                    if prefix:
                        alias = identifier.concat_names([prefix, col_name])
                        select_cols.append(f"{cte_name}.{col_name} AS {alias}")
                        col_names.append(alias)
                    else:
                        select_cols.append(f"{cte_name}.{col_name}")
                        col_names.append(col_name)
            else:
                for col in feature_columns[i].split(", "):
                    col_clean = col.strip()
                    if prefix:
                        alias = identifier.concat_names([prefix, col_clean])
                        select_cols.append(f"{cte_name}.{col_clean} AS {alias}")
                        col_names.append(alias)
                    else:
                        select_cols.append(f"{cte_name}.{col_clean}")
                        col_names.append(col_clean)

            fv_infos.append(
                _FvJoinInfo(
                    cte_name=cte_name,
                    key_sig=key_sig,
                    join_keys=fv_join_keys,
                    has_ts=has_ts,
                    select_cols=select_cols,
                    col_names=col_names,
                )
            )

        # For small FV counts, use the existing flat join path
        if len(cte_names) <= _CTE_MERGE_BATCH_SIZE:
            all_select = [col for fi in fv_infos for col in fi.select_cols]
            all_joins = "".join(
                _make_fv_join_clause("SPINE", fi.cte_name, fi.join_keys, fi.has_ts, spine_timestamp_col)
                for fi in fv_infos
            )
            query = f"""WITH
{', '.join(ctes)}
SELECT
    SPINE.*,
    {', '.join(all_select)}
FROM SPINE{all_joins}
"""
            return query

        return self._build_batched_cte_merge_query(ctes, fv_infos, spine_timestamp_col)

    @staticmethod
    def _build_batched_cte_merge_query(
        ctes: list[str],
        fv_infos: list[_FvJoinInfo],
        spine_timestamp_col: Optional[SqlIdentifier],
    ) -> str:
        """Build a batched CTE merge query for large numbers of feature views.

        Groups FVs by entity key signature, chunks them into batches of at most
        _CTE_MERGE_BATCH_SIZE, and assembles batch CTEs that are joined in a final
        merge step.  BATCH0 carries all spine columns; subsequent batches deduplicate
        the spine to their entity keys via SELECT DISTINCT.

        Args:
            ctes: Accumulated list of CTE definition strings (modified in-place with batch CTEs).
            fv_infos: Per-feature-view join metadata collected by _build_cte_query.
            spine_timestamp_col: The timestamp column from the spine, or None.

        Returns:
            A SQL query string with batched CTE merge structure.
        """
        # Group FVs by entity key signature
        groups: dict[tuple[tuple[str, ...], bool], list[_FvJoinInfo]] = {}
        for fi in fv_infos:
            if fi.key_sig not in groups:
                groups[fi.key_sig] = []
            groups[fi.key_sig].append(fi)

        # BATCH0 gets the widest key set (most keys; ties broken by has_ts=True,
        # then lexicographic key order — deterministic regardless of insertion order).
        batch0_sig = max(groups.keys(), key=lambda sig: (len(sig[0]), sig[1], sig[0]))

        # Ordered list of (fv_chunk, key_sig): BATCH0 group first, then others
        # sorted for deterministic batch ordering across runs
        batches: list[tuple[list[_FvJoinInfo], tuple[tuple[str, ...], bool]]] = []
        batch0_fvs = groups.pop(batch0_sig)
        for start in range(0, len(batch0_fvs), _CTE_MERGE_BATCH_SIZE):
            batches.append((batch0_fvs[start : start + _CTE_MERGE_BATCH_SIZE], batch0_sig))
        for sig in sorted(groups.keys()):
            fvs = groups[sig]
            for start in range(0, len(fvs), _CTE_MERGE_BATCH_SIZE):
                batches.append((fvs[start : start + _CTE_MERGE_BATCH_SIZE], sig))

        batch_cte_names: list[str] = []
        batch_col_names: list[list[str]] = []
        batch_key_ids: list[list[SqlIdentifier]] = []
        batch_has_ts_flags: list[bool] = []

        for batch_idx, (chunk, key_sig) in enumerate(batches):
            batch_cte_name = f"_BATCH{batch_idx}"
            batch_cte_names.append(batch_cte_name)

            # All FVs in a group share the same resolved key signature, so chunk[0].join_keys
            # is a canonical SqlIdentifier list for SQL rendering (preserves case / quoting rules).
            group_key_ids = chunk[0].join_keys
            batch_key_ids.append(group_key_ids)
            batch_has_ts = key_sig[1]
            batch_has_ts_flags.append(batch_has_ts)

            batch_select = [col for fi in chunk for col in fi.select_cols]
            batch_col_names.append([cn for fi in chunk for cn in fi.col_names])

            if batch_idx == 0:
                batch_joins = "".join(
                    _make_fv_join_clause(
                        "SPINE",
                        fi.cte_name,
                        fi.join_keys,
                        fi.has_ts,
                        spine_timestamp_col,
                    )
                    for fi in chunk
                )
                cte_body = f"SELECT SPINE.*, {', '.join(batch_select)} FROM SPINE{batch_joins}"
            else:
                dedup_alias = f"_SD{batch_idx}"
                dedup_cols = [k.identifier() for k in group_key_ids]
                if batch_has_ts and spine_timestamp_col is not None:
                    dedup_cols.append(spine_timestamp_col.identifier())

                batch_joins = "".join(
                    _make_fv_join_clause(
                        dedup_alias,
                        fi.cte_name,
                        fi.join_keys,
                        fi.has_ts,
                        spine_timestamp_col,
                    )
                    for fi in chunk
                )

                key_select = ", ".join(f"{dedup_alias}.{k.identifier()}" for k in group_key_ids)
                ts_select = (
                    f", {dedup_alias}.{spine_timestamp_col.identifier()}"
                    if batch_has_ts and spine_timestamp_col is not None
                    else ""
                )

                cte_body = (
                    f"SELECT {key_select}{ts_select}, {', '.join(batch_select)}\n"
                    f"    FROM (SELECT DISTINCT {', '.join(dedup_cols)} FROM SPINE) {dedup_alias}"
                    f"{batch_joins}"
                )

            ctes.append(f"{batch_cte_name} AS (\n{cte_body}\n)")

        # Final merge: BATCH0.* + feature columns from other batches
        merge_select_parts = [f"{batch_cte_names[0]}.*"]
        merge_join_parts: list[str] = []

        for bi in range(1, len(batches)):
            bcn = batch_cte_names[bi]
            for col_name in batch_col_names[bi]:
                merge_select_parts.append(f"{bcn}.{col_name}")
            merge_join_parts.append(
                _make_fv_join_clause(
                    batch_cte_names[0],
                    bcn,
                    batch_key_ids[bi],
                    batch_has_ts_flags[bi],
                    spine_timestamp_col,
                )
            )

        merge_select_str = ",\n    ".join(merge_select_parts)
        query = f"""WITH
{', '.join(ctes)}
SELECT
    {merge_select_str}
FROM {batch_cte_names[0]}{''.join(merge_join_parts)}
"""
        return query

    def _join_features(
        self,
        spine_df: DataFrame,
        features: list[Union[FeatureView, FeatureViewSlice]],
        spine_timestamp_col: Optional[SqlIdentifier],
        include_feature_view_timestamp_col: bool,
        auto_prefix: bool = False,
        join_method: Literal["sequential", "cte"] = "sequential",
        is_training: bool = False,
    ) -> tuple[DataFrame, list[SqlIdentifier]]:
        # Validate join_method parameter
        if join_method not in ["sequential", "cte"]:
            raise ValueError(f"Invalid join_method '{join_method}'. Must be 'sequential' or 'cte'.")

        # Detect tiled and append_only feature views in a single pass; both inform downstream
        # validation of join_method and spine_timestamp_col requirements. Short-circuit once
        # both flags are populated — neither downstream check needs more than one example.
        has_tiled_fv = False
        first_append_only_fv: Optional[FeatureView] = None
        for feature in features:
            fv = fg_mod.unwrap_fv(feature)
            if fv.is_tiled:
                has_tiled_fv = True
            if fv.append_only and first_append_only_fv is None:
                first_append_only_fv = fv
            if has_tiled_fv and first_append_only_fv is not None:
                break

        if has_tiled_fv and join_method != "cte":
            raise ValueError(
                "Tiled feature views require join_method='cte'. "
                "Please set join_method='cte' when using feature views with tile-based aggregations."
            )

        if spine_timestamp_col is None:
            if has_tiled_fv:
                raise ValueError(
                    "Tiled feature views require a spine_timestamp_col for point-in-time joins. "
                    "Please provide spine_timestamp_col when using feature views with tile-based aggregations."
                )
            if first_append_only_fv is not None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"FeatureView {first_append_only_fv.name} is append_only and requires "
                        "spine_timestamp_col for point-in-time lookups against its snapshot table. "
                        "Without a timestamp column, only the latest feature values would be returned, "
                        "bypassing snapshot history."
                    ),
                )

        feature_views: list[FeatureView] = []
        # Extract column selections for each feature view
        feature_columns: list[str] = []
        for feature in features:
            fv = fg_mod.unwrap_fv(feature)
            # Defense-in-depth: realtime FVs should be routed through apply_rtfvs.
            assert (
                not fv.is_realtime_feature_view
            ), f"realtime feature view {fv.name} reached _join_features; should have been routed through apply_rtfvs"
            if fv.status == FeatureViewStatus.DRAFT:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"FeatureView {fv.name} has not been registered."),
                )
            for e in fv.entities:
                for k in e.join_keys:
                    if k not in to_sql_identifiers(spine_df.columns):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_ARGUMENT,
                            original_exception=ValueError(
                                f"join_key {k} from Entity {e.name} in FeatureView {fv.name} "
                                "is not found in spine_df."
                            ),
                        )
            feature_views.append(fv)
            if isinstance(feature, FeatureViewSlice):
                cols = feature.names
            else:
                cols = feature.feature_names
            feature_columns.append(", ".join(col.identifier() for col in cols))
        # TODO (SNOW-2396184): remove this check and the non-ASOF join path as ASOF join is enabled by default now.
        if self._asof_join_enabled is None:
            self._asof_join_enabled = self._is_asof_join_enabled()

        # TODO: leverage Snowpark dataframe for more concise syntax once it supports AsOfJoin
        query = spine_df.queries["queries"][-1]
        join_keys: list[SqlIdentifier] = []

        if join_method == "cte":
            logger.info(f"Using the CTE method with {len(features)} feature views")

            query = self._build_cte_query(
                feature_views,
                feature_columns,
                spine_df.queries["queries"][-1],
                spine_timestamp_col,
                include_feature_view_timestamp_col,
                auto_prefix,
                is_training,
            )
        else:
            # Use sequential joins layer by layer
            logger.info(f"Using the sequential join method with {len(features)} feature views")
            layer = 0
            for feature in features:
                if isinstance(feature, FeatureViewSlice):
                    cols = feature.names
                    feature = feature.feature_view_ref
                else:
                    cols = feature.feature_names

                join_keys = list({k for e in feature.entities for k in e.join_keys})
                join_keys_str = ", ".join(join_keys)
                assert feature.version is not None
                join_table_name = feature.fully_qualified_name()

                if spine_timestamp_col is not None and feature.timestamp_col is not None:
                    if feature.append_only:
                        join_table_name = feature.fully_qualified_snapshot_table_name()
                    if self._asof_join_enabled:
                        if include_feature_view_timestamp_col:
                            f_ts_col_alias = identifier.concat_names(
                                [
                                    feature.name,
                                    "_",
                                    feature.version,
                                    "_",
                                    feature.timestamp_col,
                                ]
                            )
                            f_ts_col_str = f"r_{layer}.{feature.timestamp_col} AS {f_ts_col_alias},"
                        else:
                            f_ts_col_str = ""

                        # Build feature column selection with optional prefix
                        prefix = self._get_feature_prefix(feature, auto_prefix)
                        if prefix:
                            feature_cols_list = []
                            for col in cols:
                                col_name = col.identifier()
                                alias = identifier.concat_names([prefix, col_name])
                                feature_cols_list.append(f"r_{layer}.{col_name} AS {alias}")
                            feature_cols_str = ", ".join(feature_cols_list)
                        else:
                            feature_cols_str = f"r_{layer}.* EXCLUDE ({join_keys_str}, {feature.timestamp_col})"

                        query = f"""
                            SELECT
                                l_{layer}.*,
                                {f_ts_col_str}
                                {feature_cols_str}
                            FROM ({query}) l_{layer}
                            ASOF JOIN (
                                SELECT {join_keys_str}, {feature.timestamp_col},
                                    {', '.join(col.identifier() for col in cols)}
                                FROM {join_table_name}
                            ) r_{layer}
                            MATCH_CONDITION (l_{layer}.{spine_timestamp_col} >= r_{layer}.{feature.timestamp_col})
                            ON {' AND '.join([f'l_{layer}.{k} = r_{layer}.{k}' for k in join_keys])}
                        """
                    else:
                        assert feature.feature_df is not None
                        query = self._composed_union_window_join_query(
                            layer=layer,
                            s_query=query,
                            s_ts_col=spine_timestamp_col,
                            f_df=feature.feature_df,
                            f_table_name=join_table_name,
                            f_ts_col=feature.timestamp_col,
                            join_keys=join_keys,
                        )
                else:
                    # Build feature column selection with optional prefix
                    prefix = self._get_feature_prefix(feature, auto_prefix)
                    if prefix:
                        feature_cols_list = []
                        for col in cols:
                            col_name = col.identifier()
                            alias = identifier.concat_names([prefix, col_name])
                            feature_cols_list.append(f"r_{layer}.{col_name} AS {alias}")
                        feature_cols_str = ", ".join(feature_cols_list)
                    else:
                        feature_cols_str = f"r_{layer}.* EXCLUDE ({join_keys_str})"

                    query = f"""
                        SELECT
                            l_{layer}.*,
                            {feature_cols_str}
                        FROM ({query}) l_{layer}
                        LEFT JOIN (
                            SELECT {join_keys_str}, {', '.join(col.identifier() for col in cols)}
                            FROM {join_table_name}
                        ) r_{layer}
                        ON {' AND '.join([f'l_{layer}.{k} = r_{layer}.{k}' for k in join_keys])}
                    """
                layer += 1

        # TODO: construct result dataframe with datframe APIs once ASOF join is supported natively.
        # Below code manually construct result dataframe from private members of spine dataframe, which
        # likely will cause unintentional issues. This step is needed because spine_df might contains
        # prerequisite queries and post actions that must be carried over to result dataframe.
        result_df = self._session.sql(query)
        result_df._plan.queries = spine_df._plan.queries[:-1] + result_df._plan.queries
        result_df._plan.post_actions = spine_df._plan.post_actions

        return result_df, join_keys

    def _check_database_exists_or_throw(self) -> None:
        resolved_db_name = self._config.database.resolved()
        dbs = self._session.sql(
            f"""
            SHOW DATABASES LIKE '{resolved_db_name}' STARTS WITH '{resolved_db_name}'
        """
        ).collect(statement_params=self._telemetry_stmp)
        if len(dbs) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Database {resolved_db_name} does not exist."),
            )

    def _check_internal_objects_exist_or_throw(self) -> None:
        schema_result = self._find_object("SCHEMAS", self._config.schema)
        if len(schema_result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(
                    f"Feature store schema {self._config.schema} does not exist. "
                    "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
                ),
            )
        for tag_name in to_sql_identifiers(
            [
                _FEATURE_STORE_OBJECT_TAG,
                _FEATURE_VIEW_METADATA_TAG,
            ]
        ):
            tag_result = self._find_object("TAGS", tag_name)
            if len(tag_result) == 0:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(
                        f"Feature store internal tag {tag_name} does not exist. "
                        "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
                    ),
                )

    def _is_asof_join_enabled(self) -> bool:
        result = None
        try:
            result = self._session.sql(
                """
                WITH
                spine AS (
                    SELECT "ID", "TS" FROM ( SELECT $1 AS "ID", $2 AS "TS" FROM VALUES (1 :: INT, 100 :: INT))
                ),
                feature AS (
                    SELECT "ID", "TS" FROM ( SELECT $1 AS "ID", $2 AS "TS" FROM VALUES (1 :: INT, 100 :: INT))
                )
                SELECT * FROM spine
                ASOF JOIN feature
                MATCH_CONDITION ( spine.ts >= feature.ts )
                ON spine.id = feature.id;
            """
            ).collect(statement_params=self._telemetry_stmp)
        except SnowparkSQLException:
            return False
        return result is not None and len(result) == 1

    # Visualize how the query works:
    #   https://docs.google.com/presentation/d/15fT2F34OFp5RPv2-hZirHw6wliPRVRlPHvoCMIB00oY/edit#slide=id.g25ab53e6c8d_0_32
    def _composed_union_window_join_query(
        self,
        layer: int,
        s_query: str,
        s_ts_col: SqlIdentifier,
        f_df: DataFrame,
        f_table_name: str,
        f_ts_col: SqlIdentifier,
        join_keys: list[SqlIdentifier],
    ) -> str:
        s_df = self._session.sql(s_query)
        s_only_cols = [col for col in to_sql_identifiers(s_df.columns) if col not in [*join_keys, s_ts_col]]
        f_only_cols = [col for col in to_sql_identifiers(f_df.columns) if col not in [*join_keys, f_ts_col]]
        join_keys_str = ", ".join(join_keys)
        temp_prefix = "_FS_TEMP_"

        def join_cols(cols: list[SqlIdentifier], end_comma: bool, rename: bool, prefix: str = "") -> str:
            if not cols:
                return ""
            cols = [f"{prefix}{col}" for col in cols]  # type: ignore[misc]
            if rename:
                cols = [f"{col} AS {col.replace(temp_prefix, '')}" for col in cols]  # type: ignore[misc]
            line_end = "," if end_comma else ""
            return ", ".join(cols) + line_end

        # Part 1: CTE of spine query
        spine_cte = f"""
            WITH spine_{layer} AS (
                {s_query}
            ),"""

        # Part 2: create union of spine table and feature tables
        s_select = f"""
            SELECT
                'SPINE' {temp_prefix}src,
                {s_ts_col},
                {join_keys_str},
                {join_cols(s_only_cols, end_comma=True, rename=False)}
                {join_cols(f_only_cols, end_comma=False, rename=False, prefix='null AS ')}
            FROM ({s_query})"""
        f_select = f"""
            SELECT
                'FEATURE' {temp_prefix}src,
                {f_ts_col} {s_ts_col},
                {join_keys_str},
                {join_cols(s_only_cols, end_comma=True, rename=False, prefix='null AS ')}
                {join_cols(f_only_cols, end_comma=False, rename=False)}
            FROM {f_table_name}"""
        union_cte = f"""
            unioned_{layer} AS (
                {s_select}
                UNION ALL
                {f_select}
            ),"""

        # Part 3: create window cte and add window column
        window_select = f"SELECT {temp_prefix}src, {s_ts_col}, {join_keys_str}"
        for f_col in f_only_cols:
            window_select = (
                window_select
                + f"""
                ,last_value({f_col}) IGNORE NULLS OVER (
                    PARTITION BY {join_keys_str}
                    ORDER BY {s_ts_col} ASC, {temp_prefix}src ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS {temp_prefix}{f_col}"""
            )
        window_select = window_select + f" FROM unioned_{layer}"
        window_cte = f"""
            windowed_{layer} AS (
                {window_select}
            )"""

        # Part 4: join original spine table with window table
        prefix_f_only_cols = to_sql_identifiers(
            [f"{temp_prefix}{name.resolved()}" for name in f_only_cols],
            case_sensitive=True,
        )
        last_select = f"""
            SELECT
                {join_keys_str},
                {s_ts_col},
                {join_cols(s_only_cols, end_comma=True, rename=False)}
                {join_cols(prefix_f_only_cols, end_comma=False, rename=True)}
            FROM spine_{layer}
            JOIN windowed_{layer}
            USING ({join_keys_str}, {s_ts_col})
            WHERE windowed_{layer}.{temp_prefix}src = 'SPINE'"""

        # Part 5: complete query
        complete_query = spine_cte + union_cte + window_cte + last_select

        return complete_query

    def _resolve_task_schedule(self, fv_name: SqlIdentifier, fallback: str) -> str:
        """Look up the Task schedule for a cron-driven feature view.

        CRON-based FVs use ``TARGET_LAG = 'DOWNSTREAM'`` on the DT while the
        actual cron schedule lives on a companion Task with the same name.
        This helper retrieves that schedule so the reconstructed
        ``FeatureView`` carries the real cron expression instead of
        ``'DOWNSTREAM'``.

        Args:
            fv_name: Resolved physical name of the feature view (e.g. ``FV$V1``).
            fallback: Value to return when no Task is found (i.e. the FV was
                truly registered with ``refresh_freq='DOWNSTREAM'``).

        Returns:
            The cron expression from the Task (with the ``USING CRON `` prefix
            stripped), or ``fallback`` if no Task exists.
        """
        task_rows = self._session.sql(
            f"SHOW TASKS LIKE '{fv_name.resolved()}' IN SCHEMA {self._config.full_schema_path}"
        ).collect(statement_params=self._telemetry_stmp)
        if not task_rows:
            return fallback
        schedule: str = task_rows[0]["schedule"]
        if schedule.upper().startswith("USING CRON "):
            schedule = schedule[len("USING CRON ") :]
        return schedule

    def _get_entity_name(self, raw_name: SqlIdentifier) -> SqlIdentifier:
        return SqlIdentifier(identifier.concat_names([_ENTITY_TAG_PREFIX, raw_name]))

    def _get_fully_qualified_name(self, name: Union[SqlIdentifier, str]) -> str:
        # Do a quick check to see if we can skip regex operations
        if "." not in name:
            return f"{self._config.full_schema_path}.{name}"

        db_name, schema_name, object_name = identifier.parse_schema_level_object_identifier(name)
        return "{}.{}.{}".format(
            db_name or self._config.database,
            schema_name or self._config.schema,
            object_name,
        )

    def _resolve_storage_config(self, feature_view: FeatureView, fully_qualified_name: str) -> None:
        """Resolve storage_config with defaults for Iceberg tables.

        For Iceberg feature views:
        - If no external_volume is provided, use the default_iceberg_external_volume from FeatureStore.
        - If no base_location is provided, auto-generate one from the fully qualified name.

        Args:
            feature_view: The feature view whose storage config should be resolved.
            fully_qualified_name: The fully qualified name (DB.SCHEMA.TABLE) for path generation.

        Raises:
            SnowflakeMLException: If Iceberg storage is requested but no external_volume is available
                (neither in StorageConfig nor as FeatureStore default).
        """
        if feature_view.storage_config is not None and feature_view.storage_config.format == StorageFormat.ICEBERG:
            external_volume = feature_view.storage_config.external_volume
            base_location = feature_view.storage_config.base_location

            # Apply default external_volume if not set
            if external_volume is None:
                if self._default_iceberg_external_volume is not None:
                    external_volume = self._default_iceberg_external_volume
                else:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ARGUMENT,
                        original_exception=ValueError(
                            "Iceberg storage requires an external_volume. Either provide external_volume in "
                            "StorageConfig or set default_iceberg_external_volume when creating FeatureStore."
                        ),
                    )

            # Auto-generate base_location if not set
            if base_location is None:
                base_location = f"snowflake/feature_store/iceberg/{fully_qualified_name.replace('.', '/')}"

            feature_view._storage_config = StorageConfig(
                format=feature_view.storage_config.format,
                external_volume=external_volume,
                base_location=base_location,
            )

    def _get_all_iceberg_storage_configs(self) -> dict[str, StorageConfig]:
        """Get storage configs for all Iceberg tables in the feature store schema.

        Executes a single SHOW ICEBERG TABLES IN SCHEMA query and returns a dictionary
        mapping table names to their StorageConfig. This is more efficient than calling
        _get_iceberg_storage_config for each table individually.

        Returns:
            Dictionary mapping table name (resolved, i.e. actual stored name) to StorageConfig.

        Raises:
            SnowflakeMLException: If the SHOW ICEBERG TABLES query fails.
        """
        query = f"SHOW ICEBERG TABLES IN SCHEMA {self._config.full_schema_path}"
        try:
            rows = self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            return {
                row["name"]: StorageConfig(
                    format=StorageFormat.ICEBERG,
                    external_volume=row["external_volume_name"],
                    base_location=row["base_location"],
                )
                for row in rows
            }
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(
                    f"Failed to retrieve Iceberg storage configs from schema {self._config.full_schema_path}: {e}"
                ),
            ) from e

    def _get_iceberg_storage_config(self, table_name: SqlIdentifier) -> Optional[StorageConfig]:
        """Get storage config for an Iceberg table using SHOW ICEBERG TABLES.

        Args:
            table_name: The table name to look up.

        Returns:
            StorageConfig with Iceberg format if found, None if not an Iceberg table.

        Raises:
            SnowflakeMLException: If the SHOW ICEBERG TABLES query fails.
        """
        # Use resolved() to get the actual stored name (without SQL quoting/escaping)
        raw_name = table_name.resolved()
        # Escape single quotes for SQL string literal safety.
        # The table name goes inside single quotes as a LIKE pattern (string literal),
        # so single quotes must be escaped as '' (e.g., "foo'bar" -> "foo''bar").
        # The schema path is a SQL identifier (not a string literal), and is already
        # properly formatted by full_schema_path using SqlIdentifier utilities.
        escaped_name = raw_name.replace("'", "''")
        query = f"SHOW ICEBERG TABLES LIKE '{escaped_name}' IN SCHEMA {self._config.full_schema_path}"

        try:
            iceberg_tables = self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            if iceberg_tables:
                row = iceberg_tables[0]
                return StorageConfig(
                    format=StorageFormat.ICEBERG,
                    external_volume=row["external_volume_name"],
                    base_location=row["base_location"],
                )
            return None
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(f"Failed to retrieve Iceberg storage config for {raw_name}: {e}"),
            ) from e

    # TODO: SHOW DYNAMIC TABLES is very slow while other show objects are fast, investigate with DT in SNOW-902804.
    def _get_fv_backend_representations(
        self, object_name: Optional[SqlIdentifier], prefix_match: bool = False
    ) -> list[tuple[Row, _FeatureStoreObjTypes]]:
        dynamic_table_results = [
            (d, _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW)
            for d in self._find_object("DYNAMIC TABLES", object_name, prefix_match)
        ]
        view_results = [
            (d, _FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW)
            for d in self._find_object("VIEWS", object_name, prefix_match)
        ]
        return dynamic_table_results + view_results

    def _update_feature_view_status(
        self,
        feature_view: FeatureView,
        operation: str,
        store_type: Optional[fv_mod.StoreType] = None,
    ) -> FeatureView:
        assert operation in [
            "RESUME",
            "SUSPEND",
            "REFRESH",
        ], f"Operation: {operation} not supported"
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        fully_qualified_name = feature_view.fully_qualified_name()

        # Handle offline feature view (default for suspend/resume, or when explicitly requested)
        if store_type is None or store_type == fv_mod.StoreType.OFFLINE:
            # Note: Dynamic Iceberg tables are managed with ALTER DYNAMIC TABLE (no ICEBERG keyword)
            # just like regular dynamic tables, so the same commands work.
            try:
                self._session.sql(f"ALTER DYNAMIC TABLE {fully_qualified_name} {operation}").collect(
                    statement_params=self._telemetry_stmp
                )
                if operation != "REFRESH":
                    self._session.sql(f"ALTER TASK IF EXISTS {fully_qualified_name} {operation}").collect(
                        statement_params=self._telemetry_stmp
                    )
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(
                        f"Failed to update feature view {fully_qualified_name}'s status: {e}"
                    ),
                ) from e

        elif store_type == fv_mod.StoreType.ONLINE and operation in [
            "SUSPEND",
            "RESUME",
            "REFRESH",
        ]:
            if feature_view.online:
                fully_qualified_online_name = feature_view.fully_qualified_online_table_name()
                try:
                    self._session.sql(f"ALTER ONLINE FEATURE TABLE {fully_qualified_online_name} {operation}").collect(
                        statement_params=self._telemetry_stmp
                    )
                    logger.info(
                        f"Successfully {operation.lower()}ed online feature table for "
                        f"{feature_view.name}/{feature_view.version}"
                    )
                except Exception as e:
                    # For refresh operations, raise the exception; for suspend/resume, just log warning
                    if operation == "REFRESH":
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                            original_exception=RuntimeError(
                                f"Failed to refresh online feature table {fully_qualified_online_name}: {e}"
                            ),
                        ) from e
                    else:
                        # Log warning but don't fail the entire operation if online table operation
                        # fails for suspend/resume
                        logger.warning(f"Failed to {operation} online feature table {fully_qualified_online_name}: {e}")

        logger.info(f"Successfully {operation.lower()}ed FeatureView {feature_view.name}/{feature_view.version}.")
        return self.get_feature_view(feature_view.name, feature_view.version)

    def _optimized_find_feature_views(
        self,
        entity_name: SqlIdentifier,
        feature_view_name: Optional[SqlIdentifier],
        *,
        verbose: bool = False,
    ) -> DataFrame:
        if not self._validate_entity_exists(entity_name):
            return _create_list_feature_views_dataframe(self._session, [], [], verbose=verbose)

        # TODO: this can be optimized further by directly getting all possible FVs and filter by tag
        # it's easier to rewrite the code once we can remove the tag_reference path
        all_fvs = self._get_fv_backend_representations(object_name=None)
        fv_maps = {SqlIdentifier(r["name"], case_sensitive=True): r for r, _ in all_fvs}

        filters = [lambda d: d["entityName"].startswith(feature_view_name.resolved())] if feature_view_name else None
        res = self._lookup_tagged_objects(self._get_entity_name(entity_name), filters)

        output_values: list[list[Any]] = []
        output_values_extra: list[list[Any]] = []
        iceberg_config_cache: dict[str, StorageConfig] = {}
        rtfv_oft_entity_names: set[str] = set()
        for r in res:
            # Tagged entries cover both DT/View FVs and RTFV OFTs. DT/View
            # entries match ``fv_maps``; RTFV OFTs are identified by the
            # ``$ONLINE`` suffix.
            key = SqlIdentifier(r["entityName"], case_sensitive=True)
            if key in fv_maps:
                self._extract_feature_view_info(fv_maps[key], output_values, output_values_extra, iceberg_config_cache)
            else:
                resolved_oft_name = r["entityName"]
                if resolved_oft_name.endswith(_ONLINE_TABLE_SUFFIX):
                    rtfv_oft_entity_names.add(resolved_oft_name)

        rtfv_meta_by_oft: dict[str, RealtimeConfigMetadata] = {}
        if rtfv_oft_entity_names:
            for meta in self._metadata_manager.list_realtime_config_metadata():
                oft_phys = FeatureView._get_online_table_name(meta.name, meta.version).resolved()
                if oft_phys in rtfv_oft_entity_names:
                    rtfv_meta_by_oft[oft_phys] = meta

        if rtfv_meta_by_oft:
            from snowflake.ml.feature_store.realtime_registration import (
                append_realtime_listing_row,
            )

            # Single SHOW sweep instead of one _find_object per RTFV.
            oft_row_by_phys: dict[str, Row] = {
                SqlIdentifier(r["name"], case_sensitive=True).resolved(): r
                for r in self._find_object("ONLINE FEATURE TABLES", None)
            }
            for oft_phys, meta in rtfv_meta_by_oft.items():
                if feature_view_name is not None and not meta.name.startswith(feature_view_name.resolved()):
                    continue
                append_realtime_listing_row(
                    feature_store=self,
                    rtfv_metadata=meta,
                    oft_show_row=oft_row_by_phys.get(oft_phys),
                    output_values=output_values,
                    output_values_extra=output_values_extra,
                    fv_kind_realtime=_FV_KIND_REALTIME,
                    default_storage_config_json=_DEFAULT_STORAGE_CONFIG_JSON,
                )

        return _create_list_feature_views_dataframe(self._session, output_values, output_values_extra, verbose=verbose)

    def _extract_feature_view_info(
        self,
        row: Row,
        output_values: list[list[Any]],
        output_values_extra: list[list[Any]],
        iceberg_config_cache: dict[str, StorageConfig],
    ) -> None:
        """Extract feature view information from a backend row.

        Args:
            row: Row from SHOW DYNAMIC TABLES or SHOW VIEWS.
            output_values: Base-field rows matching ``_LIST_FEATURE_VIEW_SCHEMA``.
            output_values_extra: Verbose-only field rows matching
                ``_LIST_FEATURE_VIEW_VERBOSE_EXTRA_FIELDS``.
            iceberg_config_cache: Mutable dictionary mapping table names to StorageConfig.
                Populated lazily on first Iceberg FV encountered. Caller should pass the same
                dictionary instance across calls to avoid repeated SHOW ICEBERG TABLES queries.
        """
        name, version = row["name"].split(_FEATURE_VIEW_NAME_DELIMITER)
        fv_metadata, _ = self._lookup_feature_view_metadata(row, FeatureView._get_physical_name(name, version))

        values: list[Any] = []
        values.append(name)
        values.append(version)
        values.append(row["database_name"])
        values.append(row["schema_name"])
        values.append(row["created_on"])
        values.append(row["owner"])
        values.append(row["comment"])
        values.append(fv_metadata.entities)
        # CRON-based FVs use TARGET_LAG='DOWNSTREAM' on the DT and store the actual
        # cron expression on a companion Task. Recover it so list_feature_views()
        # surfaces the same value the user originally passed in (matching the
        # round-trip behavior of get_feature_view / register_feature_view).
        target_lag = row["target_lag"] if "target_lag" in row else None
        if target_lag == "DOWNSTREAM":
            target_lag = self._resolve_task_schedule(FeatureView._get_physical_name(name, version), target_lag)
        values.append(target_lag)
        values.append(row["refresh_mode"] if "refresh_mode" in row else None)
        values.append(row["scheduling_state"] if "scheduling_state" in row else None)
        values.append(row["warehouse"] if "warehouse" in row else None)
        values.append(json.dumps(self._extract_cluster_by_columns(row["cluster_by"])) if "cluster_by" in row else None)
        # Verbose-only field, emitted into output_values_extra below.
        initialization_warehouse = row["initialization_warehouse"] if "initialization_warehouse" in row else None

        online_config_json = self._determine_online_config_from_oft(name, version, include_online_service_metadata=True)
        values.append(online_config_json)

        # Use fv_metadata.is_iceberg to skip iceberg lookup for non-Iceberg FVs
        if fv_metadata.is_iceberg:
            fv_name = FeatureView._get_physical_name(name, version)
            # Fetch iceberg configs lazily on first Iceberg FV encountered
            if not iceberg_config_cache:
                iceberg_config_cache.update(self._get_all_iceberg_storage_configs())
            storage_config = iceberg_config_cache.get(fv_name.resolved())

            if storage_config:
                storage_config_json = storage_config.to_json()
            else:
                # FV is marked as Iceberg but we couldn't retrieve its config - log warning
                logger.warning(
                    f"Feature view {name}/{version} is marked as Iceberg but SHOW ICEBERG TABLES "
                    "returned no results. Storage config details may be incomplete."
                )
                storage_config_json = _DEFAULT_ICEBERG_STORAGE_CONFIG_JSON
        else:
            storage_config_json = _DEFAULT_STORAGE_CONFIG_JSON
        values.append(storage_config_json)

        # Stream config (streaming FVs only). Schema:
        # ``{"stream_source", "transformation_fn", "backfill_start_time"?,
        #    "backfill_status"?}``. Surfacing ``backfill_status`` here (rather
        # than as a top-level column) keeps streaming-only state scoped to
        # streaming-only context — non-streaming FVs leave ``stream_config``
        # null and never see a misleading "backfill" field that doesn't
        # apply to their full-refresh semantics. Backfill task-graph names
        # stay internal — used by cleanup/overwrite and ``get_refresh_history``.
        if fv_metadata.is_streaming:
            streaming_meta = self._metadata_manager.get_streaming_metadata(name, version)
            if streaming_meta:
                stream_config_data: dict[str, Any] = {
                    "stream_source": streaming_meta.stream_source_name,
                    "transformation_fn": streaming_meta.transformation_fn_name,
                }
                if streaming_meta.backfill_start_time is not None:
                    stream_config_data["backfill_start_time"] = streaming_meta.backfill_start_time
                if streaming_meta.backfill_state is not None:
                    stream_config_data["backfill_status"] = streaming_meta.backfill_state
                stream_config_json = json.dumps(stream_config_data)
            else:
                stream_config_json = json.dumps({"stream_source": "unknown"})
        else:
            stream_config_json = None
        values.append(stream_config_json)

        # RTFV rows are emitted separately; this helper only sees DT/View FVs.
        values.append(_FV_KIND_STREAMING if fv_metadata.is_streaming else _FV_KIND_BATCH)

        values.append(fv_metadata.is_append_only)

        output_values.append(values)

        # Surface the authored source-ref list (JSON-encoded) when a
        # ``FV_SOURCE_REFS`` row exists; otherwise leave the column null.
        # Verbose-only: goes into output_values_extra alongside backup_source.
        source_refs_meta = self._metadata_manager.get_feature_view_source_refs(name, version)
        source_refs_json = json.dumps(source_refs_meta.sources) if source_refs_meta is not None else None

        backup_source: Optional[str] = None
        if fv_metadata.is_append_only:
            append_only_meta = self._metadata_manager.get_append_only_metadata(name, version)
            if append_only_meta is not None:
                backup_source = AppendOnlyMetadata.from_dict(append_only_meta).backup_source
        output_values_extra.append([initialization_warehouse, source_refs_json, backup_source])

    def _determine_online_config_from_oft(
        self, name: str, version: str, *, include_online_service_metadata: bool = False
    ) -> str:
        """Determine online configuration by checking for corresponding online feature table.

        When an online feature table exists, ``store_type`` is taken from the
        ``SHOW ONLINE FEATURE TABLES`` row when that column is present; otherwise
        ``OnlineStoreType.HYBRID_TABLE`` is used (same as older clients without the column).

        Args:
            name: Feature view name. Accepts either the raw stored name or the SQL-quoted form.
            version: Feature view version
            include_online_service_metadata: If True, includes additional Online Service metadata
                (refresh_mode, scheduling_state) in the JSON for display purposes.
                If False, returns only OnlineConfig-compatible JSON.

        Returns:
            JSON string of OnlineConfig with enable=True, table's target_lag, and store_type
            (from SHOW when available) if online table exists,
            otherwise default config with enable=False. When include_online_service_metadata=True,
            may include additional fields not part of OnlineConfig.

        Raises:
            SnowflakeMLException: If multiple online feature tables found for the given name/version,
                or if the online feature table is missing required 'target_lag' column.
        """
        # SQL-quoted input would be re-wrapped downstream and miss the OFT for case-sensitive names.
        if not isinstance(name, SqlIdentifier) and len(name) >= 2 and name[0] == '"' and name[-1] == '"':
            name = SqlIdentifier(name, case_sensitive=False).resolved()

        online_table_name = FeatureView._get_online_table_name(name, version)

        online_tables = self._find_object(object_type="ONLINE FEATURE TABLES", object_name=online_table_name)

        if online_tables:
            if len(online_tables) != 1:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(
                        f"Expected exactly 1 online feature table for {online_table_name}, "
                        f"but found {len(online_tables)}"
                    ),
                )

            oft_row = online_tables[0]

            def extract_field(row: Row, field_name: str) -> str:
                if field_name in row:
                    return str(row[field_name])
                elif field_name.upper() in row:
                    return str(row[field_name.upper()])
                else:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_SNOWML_ERROR,
                        original_exception=RuntimeError(
                            f"Online feature table {online_table_name} missing required '{field_name}' column"
                        ),
                    )

            # Extract required fields using consistent pattern
            target_lag = extract_field(oft_row, "target_lag")

            online_config = fv_mod.OnlineConfig(
                enable=True,
                target_lag=target_lag,
                store_type=_store_type_from_oft_show_row(oft_row),
            )

            if include_online_service_metadata:
                display_data = json.loads(online_config.to_json())

                display_data["refresh_mode"] = extract_field(oft_row, "refresh_mode")
                display_data["scheduling_state"] = extract_field(oft_row, "scheduling_state")

                return json.dumps(display_data)
            else:
                return online_config.to_json()
        else:
            # No online feature table found - return default disabled config
            online_config = fv_mod.OnlineConfig(enable=False, target_lag=fv_mod._BATCH_OFT_TARGET_LAG)
            return online_config.to_json()

    def _lookup_feature_view_metadata(self, row: Row, fv_name: str) -> tuple[_FeatureViewMetadata, str]:
        if len(row["text"]) == 0:
            # NOTE: if this is a shared feature view, then text column will be empty due to privacy constraints.
            # So instead of looking at original query text, we will obtain metadata by querying the tag value.
            # For query body, we will just use a simple select instead of original DDL query since shared feature views
            # are read-only.
            try:
                res = self._lookup_tags(
                    domain="table",
                    obj_name=fv_name,
                    filter_fns=[lambda d: d["tagName"] == _FEATURE_VIEW_METADATA_TAG],
                )
                fv_metadata = _FeatureViewMetadata.from_json(res[0]["tagValue"])
                query = f"SELECT * FROM {self._get_fully_qualified_name(fv_name)}"
                return (fv_metadata, query)
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(f"Failed to extract feature_view metadata for {fv_name}: {e}."),
                )
        else:
            # Parse the CREATE statement text to extract FeatureView metadata.
            # This regex expects FeatureStore-specific clauses (COMMENT, TAG with _FEATURE_VIEW_METADATA_TAG).
            # If a raw DYNAMIC TABLE/VIEW exists without these clauses (i.e., not created by FeatureStore),
            # the regex won't match and we'll raise an error. This is intentional - we can only
            # reconstruct FeatureViews that were created by FeatureStore with proper metadata tags.
            m = re.match(_DT_OR_VIEW_QUERY_PATTERN, row["text"])
            if m is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(f"Failed to parse query text for FeatureView {fv_name}: {row}."),
                )
            fv_metadata = _FeatureViewMetadata.from_json(m.group("fv_metadata"))
            query = m.group("query")

            return (fv_metadata, query)

    def _compose_feature_view(self, row: Row, obj_type: _FeatureStoreObjTypes, entity_list: list[Row]) -> FeatureView:
        def find_and_compose_entity(name: str) -> Entity:
            name = SqlIdentifier(name).resolved()
            for e in entity_list:
                if e["NAME"] == name:
                    return Entity(
                        name=SqlIdentifier(e["NAME"], case_sensitive=True).identifier(),
                        join_keys=[f'"{k}"' for k in json.loads(e["JOIN_KEYS"])],
                        desc=e["DESC"],
                    )
            raise RuntimeError(f"Cannot find entity {name} from retrieved entity list: {entity_list}")

        name, version = row["name"].split(_FEATURE_VIEW_NAME_DELIMITER)
        name = SqlIdentifier(name, case_sensitive=True)
        fv_name = FeatureView._get_physical_name(name, version)
        fv_metadata, query = self._lookup_feature_view_metadata(row, fv_name)

        infer_schema_df = self._session.sql(f"SELECT * FROM {self._get_fully_qualified_name(fv_name)}")
        desc = row["comment"]

        online_config_json = self._determine_online_config_from_oft(name.resolved(), version)
        online_config = fv_mod.OnlineConfig.from_json(online_config_json)

        # Get storage_config from SHOW ICEBERG TABLES if marked as Iceberg in metadata
        storage_config: Optional[StorageConfig] = None
        if fv_metadata.is_iceberg:
            storage_config = self._get_iceberg_storage_config(fv_name)
            if storage_config is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(
                        f"Failed to retrieve Iceberg storage config for {fv_name.resolved()}. "
                        "The feature view is marked as Iceberg but SHOW ICEBERG TABLES returned no results."
                    ),
                )

        # Load feature metadata if present (for tiled feature views)
        agg_metadata = self._metadata_manager.get_feature_specs(name.identifier(), version)
        feature_granularity = agg_metadata.feature_granularity if agg_metadata else None
        aggregation_specs = agg_metadata.features if agg_metadata else None
        is_tiled = agg_metadata is not None
        # Restored from the FEATURE_SPECS row; None when no tiling metadata.
        aggregation_secondary_keys = agg_metadata.aggregation_secondary_keys if agg_metadata else None
        # The persisted aggregation method must be known before _construct_feature_view runs
        # validation: reconstruction routes through the batch path, where window/offset alignment
        # is enforced for TILES but relaxed for CONTINUOUS. Setting it afterward would be too late.
        feature_aggregation_method = (
            FeatureAggregationMethod(agg_metadata.feature_aggregation_method)
            if agg_metadata and agg_metadata.feature_aggregation_method
            else (FeatureAggregationMethod.TILES if is_tiled else None)
        )

        # Rehydrate the authored source-ref list onto the reconstructed
        # FV when a ``FV_SOURCE_REFS`` row was persisted at register time.
        source_refs_meta = self._metadata_manager.get_feature_view_source_refs(name.identifier(), version)
        restored_source_refs = source_refs_meta.sources if source_refs_meta is not None else None

        if obj_type == _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW:
            df = self._session.sql(query)
            entities = [find_and_compose_entity(n) for n in fv_metadata.entities]
            ts_col = fv_metadata.timestamp_col
            timestamp_col = ts_col if ts_col not in _LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS else None
            re_initialize = re.match(_DT_INITIALIZE_PATTERN, row["text"])
            initialize = re_initialize.group("initialize") if re_initialize is not None else "ON_CREATE"
            # INITIALIZATION_WAREHOUSE is surfaced as a dedicated SHOW column (like WAREHOUSE),
            # not echoed in the DDL text, so read it from the column.
            init_wh_value = row["initialization_warehouse"] if "initialization_warehouse" in row else None
            initialization_warehouse = (
                SqlIdentifier(init_wh_value, case_sensitive=True).identifier() if init_wh_value else None
            )

            # For tiled FVs, get descriptions from metadata table; otherwise from DT columns
            if is_tiled:
                feature_descs = self._metadata_manager.get_feature_descs(name.identifier(), version) or {}
            else:
                feature_descs = self._fetch_column_descs("DYNAMIC TABLE", fv_name)

            # CRON-based FVs use TARGET_LAG='DOWNSTREAM' on the DT and store the
            # actual cron expression on a companion Task. Recover it so the
            # round-trip (register → get → update) preserves what the user passed.
            refresh_freq = row["target_lag"]
            if refresh_freq == "DOWNSTREAM":
                refresh_freq = self._resolve_task_schedule(fv_name, refresh_freq)

            backup_source: Optional[str] = None
            if fv_metadata.is_append_only:
                append_only_meta = self._metadata_manager.get_append_only_metadata(name.identifier(), version)
                if append_only_meta is not None:
                    backup_source = AppendOnlyMetadata.from_dict(append_only_meta).backup_source

            fv = FeatureView._construct_feature_view(
                name=name,
                entities=entities,
                feature_df=df,
                timestamp_col=timestamp_col,
                desc=desc,
                version=version,
                status=(
                    FeatureViewStatus(row["scheduling_state"])
                    if len(row["scheduling_state"]) > 0
                    else FeatureViewStatus.MASKED
                ),
                feature_descs=feature_descs,
                refresh_freq=refresh_freq,
                database=self._config.database.identifier(),
                schema=self._config.schema.identifier(),
                warehouse=(
                    SqlIdentifier(row["warehouse"], case_sensitive=True).identifier()
                    if len(row["warehouse"]) > 0
                    else None
                ),
                initialization_warehouse=initialization_warehouse,
                refresh_mode=row["refresh_mode"],
                refresh_mode_reason=row["refresh_mode_reason"],
                initialize=initialize,
                owner=row["owner"],
                infer_schema_df=infer_schema_df,
                session=self._session,
                cluster_by=self._extract_cluster_by_columns(row["cluster_by"]),
                online_config=online_config,
                storage_config=storage_config,
                feature_granularity=feature_granularity,
                aggregation_specs=aggregation_specs,
                aggregation_secondary_keys=aggregation_secondary_keys,
                feature_aggregation_method=feature_aggregation_method,
                is_streaming=fv_metadata.is_streaming,
                append_only=fv_metadata.is_append_only,
                backup_source=backup_source,
                source_refs=restored_source_refs,
            )
            self._hydrate_postgres_online_service(fv)

            # Load authoring package version from metadata table. Legacy FVs have no
            # row -> None.
            if is_tiled:
                fv_meta_config = self._metadata_manager.get_feature_view_metadata_config(name.identifier(), version)
                fv._authoring_pkg_version = fv_meta_config.authoring_pkg_version if fv_meta_config else None

            # For streaming FVs, attach the function source from metadata
            if fv_metadata.is_streaming:
                streaming_meta = self._metadata_manager.get_streaming_metadata(name.identifier(), version)
                if streaming_meta and streaming_meta.transformation_fn_source:
                    fv._transformation_fn_source = streaming_meta.transformation_fn_source

            # Load rollup metadata if this is a rollup FV
            if fv_metadata.is_rollup:
                from snowflake.ml.feature_store.feature_view import RollupMetadata

                rollup_data = self._metadata_manager.get_rollup_metadata(name.identifier(), version)
                if rollup_data is not None:
                    fv._rollup_metadata = RollupMetadata.from_dict(rollup_data)

            return fv
        else:
            df = self._session.sql(query)
            entities = [find_and_compose_entity(n) for n in fv_metadata.entities]
            ts_col = fv_metadata.timestamp_col
            timestamp_col = ts_col if ts_col not in _LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS else None

            fv = FeatureView._construct_feature_view(
                name=name,
                entities=entities,
                feature_df=df,
                timestamp_col=timestamp_col,
                desc=desc,
                version=version,
                status=FeatureViewStatus.STATIC,
                feature_descs=self._fetch_column_descs("VIEW", fv_name),
                refresh_freq=None,
                database=self._config.database.identifier(),
                schema=self._config.schema.identifier(),
                warehouse=None,
                refresh_mode=None,
                refresh_mode_reason=None,
                initialize="ON_CREATE",
                owner=row["owner"],
                infer_schema_df=infer_schema_df,
                session=self._session,
                online_config=online_config,
                feature_granularity=feature_granularity,
                aggregation_specs=aggregation_specs,
                aggregation_secondary_keys=aggregation_secondary_keys,
                feature_aggregation_method=feature_aggregation_method,
                storage_config=storage_config,
                is_streaming=fv_metadata.is_streaming,
                source_refs=restored_source_refs,
            )
            self._hydrate_postgres_online_service(fv)

            if is_tiled:
                fv_meta_config = self._metadata_manager.get_feature_view_metadata_config(name.identifier(), version)
                fv._authoring_pkg_version = fv_meta_config.authoring_pkg_version if fv_meta_config else None

            return fv

    def _resolve_postgres_online_query_url(self, *, log_label: str) -> Optional[str]:
        """Fetch the Postgres OFT query endpoint URL or warn and return ``None``.

        Failures (status not RUNNING, missing endpoint URL, transient
        SYSTEM$ errors) are downgraded to ``UserWarning`` so callers can
        still construct the target object and see the "Online Service not
        RUNNING" error on first read.

        Args:
            log_label: Caller name embedded in the warning log
                (e.g. ``"get_feature_view"``, ``"get_feature_group"``).

        Returns:
            The query endpoint URL when the Online Service is RUNNING and
            advertises a ``query`` endpoint; ``None`` otherwise.
        """
        try:
            st = online_service.fetch_online_service_status(
                self._session,
                self._config.database,
                self._config.schema,
                statement_params=self._telemetry_stmp,
            )
        except Exception as ex:
            logger.warning("Could not fetch Online Service status during %s: %s", log_label, ex)
            warnings.warn(
                online_service.online_service_not_ready_message(),
                category=UserWarning,
                stacklevel=4,
            )
            return None
        query_url = online_service.endpoint_url(st, "query", access=self._online_service_access)
        if st.status == "RUNNING" and query_url:
            # Temporary: SYSTEM$ may return host-only query URLs until server sends full https URLs.
            return query_url
        warnings.warn(
            online_service.online_service_not_ready_message(),
            category=UserWarning,
            stacklevel=4,
        )
        return None

    def _hydrate_postgres_online_service(self, fv: FeatureView) -> None:
        """Set ``_postgres_online_query_url`` and warn when Postgres online reads are not ready."""
        if not fv.online or fv.online_config is None or fv.online_config.store_type != OnlineStoreType.POSTGRES:
            return
        fv._postgres_online_query_url = self._resolve_postgres_online_query_url(log_label="get_feature_view")

    def _hydrate_fg_postgres_online_service(self, fg: FeatureGroup) -> None:
        """Set ``FeatureGroup._postgres_online_query_url`` for read-time use.

        Args:
            fg: The reconstructed FeatureGroup whose query URL should be populated.
        """
        fg._postgres_online_query_url = self._resolve_postgres_online_query_url(log_label="get_feature_group")

    def _fetch_column_descs(self, obj_type: str, obj_name: SqlIdentifier) -> dict[str, str]:
        res = self._session.sql(f"DESC {obj_type} {self._get_fully_qualified_name(obj_name)}").collect(
            statement_params=self._telemetry_stmp
        )

        descs = {}
        for r in res:
            if r["comment"] is not None:
                descs[SqlIdentifier(r["name"], case_sensitive=True).identifier()] = r["comment"]
        return descs

    def _tag_oft(self, fully_qualified_oft_name: str, obj_type: _FeatureStoreObjTypes) -> None:
        """Apply the FS object tag to a freshly-created OFT.

        Wraps :func:`feature_view.execute_oft_set_tag`, supplying the
        per-FeatureStore context (tag FQN, package version, telemetry params)
        so the SQL builder stays a free function.

        Args:
            fully_qualified_oft_name: ``DB.SCHEMA.OFT_NAME``.
            obj_type: ``_FeatureStoreObjTypes`` value embedded in the tag JSON;
                discriminates FVs (``ONLINE_FEATURE_TABLE``) from FGs
                (``FEATURE_GROUP``) for downstream listing.
        """
        obj_info = _FeatureStoreObjInfo(obj_type, snowml_version.VERSION)
        fv_mod.execute_oft_set_tag(
            self._session,
            fully_qualified_oft_name=fully_qualified_oft_name,
            fully_qualified_tag_name=self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG),
            tag_value_json=obj_info.to_json(),
            statement_params=self._telemetry_stmp,
        )

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def _create_online_feature_table(
        self,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        version: str,
        overwrite: bool = False,
    ) -> str:
        """Create online feature table for the feature view.

        For ``HYBRID_TABLE`` store type, creates the OFT via ``CREATE ... FROM <source_table>``.
        For ``POSTGRES`` store type, builds a :class:`~snowflake.ml.feature_store.spec.models.FeatureViewSpec`
        and creates the OFT via ``CREATE ... FROM SPECIFICATION $$<json>$$``.

        Args:
            feature_view: The FeatureView object for which to create the online feature table.
            feature_view_name: The physical name of the feature view (DT/View).
            version: The version string for the feature view.
            overwrite: Whether to overwrite existing online feature table. Defaults to False.

        Returns:
            The name of the created online table (without schema qualification).

        Raises:
            SnowflakeMLException: [ValueError] If OnlineConfig is required but not provided.
            SnowflakeMLException: If creating the online feature table fails.
        """
        # Defense-in-depth: also covers the registration path. Rejects store types that cannot back
        # a tiled feature view before any DDL is issued. The rule lives on FeatureView.
        try:
            feature_view._validate_online_store_supported()
        except ValueError as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT, original_exception=e
            ) from e

        online_table_name = FeatureView._get_online_table_name(feature_view_name)

        fully_qualified_online_name = self._get_fully_qualified_name(online_table_name)
        source_table_name = self._get_fully_qualified_name(feature_view_name)

        # Build online config clauses
        config = feature_view.online_config
        if not config:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("OnlineConfig is required to create online feature table"),
            )

        # Extract join keys for PRIMARY KEY (preserve order and ensure unique)
        ordered_join_keys: list[str] = []
        seen_join_keys: set[str] = set()
        for entity in feature_view.entities:
            for join_key in entity.join_keys:
                resolved_key = join_key.resolved()
                if resolved_key not in seen_join_keys:
                    seen_join_keys.add(resolved_key)
                    ordered_join_keys.append(resolved_key)
        # GS dedupes ingest on the OFT primary key from spec; Quake strips
        # the secondary key back out of the PK after we add it.
        if config.store_type == OnlineStoreType.POSTGRES and feature_view.aggregation_secondary_keys:
            for sk in feature_view.aggregation_secondary_keys:
                if sk not in seen_join_keys:
                    seen_join_keys.add(sk)
                    ordered_join_keys.append(sk)
        primary_key_clause = fv_mod.build_oft_primary_key_clause(ordered_join_keys)
        target_lag_value = config.target_lag if config.target_lag is not None else fv_mod._BATCH_OFT_TARGET_LAG
        # StreamingFeatureView OFTs are spec-backed and require request-time freshness.
        if feature_view.is_streaming:
            target_lag_value = fv_mod._NON_BATCH_OFT_TARGET_LAG

        warehouse_clause = fv_mod.build_oft_warehouse_clause(feature_view.warehouse, self._default_warehouse)

        refresh_mode_clause = ""
        if feature_view.refresh_mode:
            refresh_mode_clause = f"REFRESH_MODE='{feature_view.refresh_mode}'"

        timestamp_clause = ""
        if feature_view.timestamp_col:
            timestamp_clause = f"TIMESTAMP_COLUMN='{feature_view.timestamp_col}'"

        # Determine source clause based on online store type
        store_type = config.store_type
        if store_type == OnlineStoreType.POSTGRES:
            online_service.assert_online_service_running_with_query_endpoint(
                self._session,
                self._config.database,
                self._config.schema,
                statement_params=self._telemetry_stmp,
            )
            # Read offline column shapes from the materialized DT/View so storage-side
            # lengths land in the spec. Non-tiled streaming is already materialized-backed
            # via _initialize_from_feature_df, so skip the extra describe.
            pg_offline_dt_schema: Optional[StructType] = None
            try:
                if feature_view.is_tiled or not feature_view.is_streaming:
                    pg_offline_dt_schema = self._session.table(source_table_name).schema
                    validate_spec_oft_tiled_offline_table_schema(pg_offline_dt_schema)
                else:
                    validate_spec_oft_offline_table_schema(feature_view.output_schema)
            except ValueError as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"Feature view '{feature_view.name}' contains column types not supported by "
                        f"online store. {e} "
                        f"Consider casting unsupported columns to a supported type before creating the feature view."
                    ),
                ) from e

            if feature_view.is_streaming:
                from snowflake.ml.feature_store.streaming_registration import (
                    _build_streaming_feature_view_spec,
                )

                assert feature_view.stream_config is not None  # guaranteed by is_streaming
                stream_source = self.get_stream_source(feature_view.stream_config.get_stream_source_name())

                udf_table_name = FeatureView._get_udf_transformed_table_name(feature_view_name)
                fq_udf_table = self._get_fully_qualified_name(udf_table_name)
                if feature_view.is_tiled:
                    udf_transformed_schema = self._session.table(fq_udf_table).schema
                else:
                    udf_transformed_schema = feature_view.output_schema

                spec = _build_streaming_feature_view_spec(
                    feature_view=feature_view,
                    feature_view_name=feature_view_name,
                    version=version,
                    target_lag=target_lag_value,
                    stream_source=stream_source,
                    udf_transformed_schema=udf_transformed_schema,
                    tiled_materialized_schema=pg_offline_dt_schema,
                    database=self._config.database.resolved(),
                    schema=self._config.schema.resolved(),
                )
            else:
                spec = self._build_batch_feature_view_spec(
                    feature_view=feature_view,
                    feature_view_name=feature_view_name,
                    version=version,
                    target_lag=target_lag_value,
                    offline_materialized_schema=pg_offline_dt_schema,
                )
            spec_json = spec.to_json()
            source_clause = f"FROM SPECIFICATION $${spec_json}$$"
        else:
            source_clause = f"FROM {source_table_name}"

        # Create online feature table
        try:
            query = fv_mod.build_oft_create_sql(
                fully_qualified_oft_name=fully_qualified_online_name,
                primary_key_clause=primary_key_clause,
                target_lag=target_lag_value,
                source_clause=source_clause,
                warehouse_clause=warehouse_clause,
                refresh_mode_clause=refresh_mode_clause,
                timestamp_clause=timestamp_clause,
                overwrite=overwrite,
            )
            self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            self._tag_oft(fully_qualified_online_name, _FeatureStoreObjTypes.ONLINE_FEATURE_TABLE)
        except Exception as e:
            logger.error(f"Failed to create online feature table for {feature_view.name}: {e}")
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"Create online feature table {fully_qualified_online_name} failed: {e}"
                ),
            ) from e

        return online_table_name

    def _build_batch_feature_view_spec(
        self,
        feature_view: FeatureView,
        feature_view_name: SqlIdentifier,
        version: str,
        target_lag: str,
        *,
        offline_materialized_schema: Optional[StructType] = None,
    ) -> FeatureViewSpec:
        """Build a validated FeatureView spec for a batch feature view.

        Constructs a :class:`FeatureViewSpec` from the FeatureView metadata and
        the physical DT/View name.

        Args:
            feature_view: The FeatureView definition.
            feature_view_name: Physical name of the DT/View (already created).
            version: Feature view version string.
            target_lag: Resolved target lag string (e.g., ``"30s"``).
            offline_materialized_schema: Snowpark schema of the materialized DT/View
                (from ``Session.table(fq_dt).schema``). Required when ``feature_view.is_tiled``;
                preferred for non-tiled batch so column shapes track storage. When omitted,
                falls back to ``feature_view.output_schema``.

        Returns:
            FeatureViewSpec instance ready for serialization.

        Raises:
            ValueError: If tiled batch is missing *offline_materialized_schema*.
        """
        database = self._config.database.resolved()
        schema = self._config.schema.resolved()

        # Entity join key column names (deduplicated, preserving order)
        entity_columns: list[str] = []
        seen_entity_cols: set[str] = set()
        for entity in feature_view.entities:
            for jk in entity.join_keys:
                resolved = jk.resolved()
                if resolved not in seen_entity_cols:
                    seen_entity_cols.add(resolved)
                    entity_columns.append(resolved)

        # GS dedupes on ``ordered_entity_column_names`` and ignores
        # ``ordered_secondary_key_column_names``; Quake strips SKs back out
        # before storing.
        if feature_view.aggregation_secondary_keys:
            for sk in feature_view.aggregation_secondary_keys:
                if sk not in seen_entity_cols:
                    entity_columns.append(sk)

        if feature_view.is_tiled:
            if offline_materialized_schema is None:
                raise ValueError(
                    "Tiled batch feature view spec requires offline_materialized_schema "
                    "(Snowflake DT schema from Session.table(...).schema)."
                )
            offline_columns = offline_materialized_schema
        else:
            offline_columns = (
                offline_materialized_schema if offline_materialized_schema is not None else feature_view.output_schema
            )
        # Batch FVs always register the materialized table as BatchSource; Tiled/UDFTransformed are streaming-only.
        offline_table_type = TableType.BATCH_SOURCE

        # Offline config: the DT/View that was just created
        offline_table_info = SnowflakeTableInfo(
            table_type=offline_table_type,
            database=database,
            schema=schema,
            table=feature_view_name.resolved(),
            columns=offline_columns,
        )

        builder = (
            FeatureViewSpecBuilder(
                FeatureViewKind.BatchFeatureView,
                database=database,
                schema=schema,
                name=feature_view.name.resolved(),
                version=version,
            )
            .set_offline_configs([offline_table_info])
            .set_properties(
                entity_columns=entity_columns,
                secondary_key_columns=feature_view.aggregation_secondary_keys,
                timestamp_field=(feature_view.timestamp_col.resolved() if feature_view.timestamp_col else None),
                granularity=(feature_view.feature_granularity if feature_view.is_tiled else None),
                agg_method=(FeatureAggregationMethod.TILES if feature_view.is_tiled else None),
                target_lag=target_lag,
            )
        )

        # For tiled batch FVs, set the raw source schema and aggregation specs.
        # Rollup FVs inherit agg specs from the parent; the raw source columns
        # that those specs reference live in the parent's feature_df.
        if feature_view.is_tiled and feature_view.aggregation_specs:
            if feature_view.is_rollup:
                assert feature_view.rollup_config is not None
                source_df = feature_view.rollup_config.source.feature_df
            else:
                source_df = feature_view.feature_df
            assert source_df is not None
            builder.set_sources([BatchSource(schema=source_df.schema)])
            builder.set_features(feature_view.aggregation_specs)

        return builder.build()

    def _find_object(
        self,
        object_type: str,
        object_name: Optional[SqlIdentifier],
        prefix_match: bool = False,
    ) -> list[Row]:
        """Try to find an object by given type and name pattern.

        Args:
            object_type: Type of the object. Could be TABLES, TAGS etc.
            object_name: Name of object. It will match everything of object_type is object_name is None.
            prefix_match: Will search all objects with object_name as prefix if set True. Otherwise
                will do exact on object_name. Default to false. If object_name is empty and prefix_match is
                True, then it will match everything of object_type.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to find resource.

        Returns:
            Return a list of rows round.
        """
        if object_name is None:
            match_name = "%"
        elif prefix_match:
            match_name = object_name.resolved() + "%"
        else:
            match_name = object_name.resolved()

        search_space, obj_domain = self._obj_search_spaces[object_type]
        all_rows = []
        fs_tag_objects = []
        tag_free_object_types = ["TAGS", "SCHEMAS", "WAREHOUSES", "DATASETS"]
        try:
            search_scope = f"IN {search_space}" if search_space is not None else ""
            all_rows = self._session.sql(f"SHOW {object_type} LIKE '{match_name}' {search_scope}").collect(
                statement_params=self._telemetry_stmp
            )
            # There could be non-FS objects under FS schema, thus filter on objects with FS special tag.
            if object_type not in tag_free_object_types and len(all_rows) > 0:
                fs_obj_rows = self._lookup_tagged_objects(
                    _FEATURE_STORE_OBJECT_TAG, [lambda d: d["domain"] == obj_domain]
                )
                fs_tag_objects = [row["entityName"] for row in fs_obj_rows]
        except Exception as e:
            # ONLINE FEATURE TABLE preview feature may raise SQL error if not enabled
            # Return empty list for discovery flows in this case
            if (
                object_type == "ONLINE FEATURE TABLES"
                and isinstance(e, SnowparkSQLException)
                and ("unexpected 'online'" in str(e).lower())
            ):
                return []
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to find object : {e}"),
            ) from e

        result = []
        for row in all_rows:
            found_name = row["name"]
            prefix = object_name.resolved() if object_name is not None else ""
            if found_name.startswith(prefix) and (object_type in tag_free_object_types or found_name in fs_tag_objects):
                result.append(row)
        return result

    def _load_serialized_feature_views(
        self, serialized_feature_views: list[str]
    ) -> list[Union[FeatureView, FeatureViewSlice]]:
        results: list[Union[FeatureView, FeatureViewSlice]] = []
        for obj in serialized_feature_views:
            try:
                obj_type = json.loads(obj)[_FEATURE_OBJ_TYPE]
            except Exception as e:
                raise ValueError(f"Malformed serialized feature object: {obj}") from e

            if obj_type == FeatureView.__name__:
                results.append(FeatureView.from_json(obj, self._session))
            elif obj_type == FeatureViewSlice.__name__:
                results.append(FeatureViewSlice.from_json(obj, self._session))
            else:
                raise ValueError(f"Unsupported feature object type: {obj_type}")
        return results

    def _load_compact_feature_views(
        self, compact_feature_views: list[str]
    ) -> list[Union[FeatureView, FeatureViewSlice]]:
        results: list[Union[FeatureView, FeatureViewSlice]] = []
        for obj in compact_feature_views:
            results.append(FeatureView._load_from_compact_repr(self._session, obj))
        return results

    def _exclude_columns(self, df: DataFrame, exclude_columns: list[str]) -> DataFrame:
        exclude_columns = to_sql_identifiers(exclude_columns)  # type: ignore[assignment]
        df_cols = to_sql_identifiers(df.columns)
        for col in exclude_columns:
            if col not in df_cols:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"{col} in exclude_columns not exists in dataframe columns: {df_cols}"
                    ),
                )
        return cast(DataFrame, df.drop(exclude_columns))  # type: ignore[redundant-cast]

    def _is_dataset_enabled(self) -> bool:
        try:
            self._session.sql(f"SHOW DATASETS IN SCHEMA {self._config.full_schema_path}").collect(
                statement_params=self._telemetry_stmp
            )
            return True
        except SnowparkSQLException:
            return False

    def _check_feature_store_object_versions(self) -> None:
        versions = self._collapse_object_versions()
        if len(versions) > 0 and pkg_version.parse(snowml_version.VERSION) < versions[0]:
            warnings.warn(
                "The current snowflake-ml-python version out of date, package upgrade recommended "
                + f"(current={snowml_version.VERSION}, recommended>={str(versions[0])})",
                stacklevel=2,
                category=UserWarning,
            )

    def _filter_results(
        self,
        results: list[dict[str, str]],
        filter_fns: Optional[list[Callable[[dict[str, str]], bool]]] = None,
    ) -> list[dict[str, str]]:
        if filter_fns is None:
            return results

        filtered_results = []
        for r in results:
            if all([fn(r) for fn in filter_fns]):
                filtered_results.append(r)
        return filtered_results

    def _lookup_tags(
        self,
        domain: str,
        obj_name: str,
        filter_fns: Optional[list[Callable[[dict[str, str]], bool]]] = None,
    ) -> list[dict[str, str]]:
        """
        Lookup tag values for a given object, optionally apply filters on the results.

        Args:
            domain: Domain of the obj to look for tag. E.g. table
            obj_name: Name of the obj.
            filter_fns: List of filter functions applied on the results.

        Returns:
            List of tag values in dictionary format.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to lookup tags.

        Example::

            self._lookup_tags("TABLE", "MY_FV", [lambda d: d["tagName"] == "TARGET_TAG_NAME"])

        """
        # NOTE: use ENTITY_DETAIL system fn to query tags for given object for it to work in
        # processes using owner's right. e.g. Streamlit, or stored procedure
        try:
            res = self._session.sql(
                f"""
                SELECT ENTITY_DETAIL('{domain}','{self._get_fully_qualified_name(obj_name)}', '["TAG_REFERENCES"]');
            """
            ).collect(statement_params=self._telemetry_stmp)
            entity_detail = json.loads(res[0][0])
            results = entity_detail["tagReferencesInfo"]["tagReferenceList"]
            return self._filter_results(results, filter_fns)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to lookup tags for object for {obj_name}: {e}"),
            ) from e

    def _lookup_tagged_objects(
        self,
        tag_name: str,
        filter_fns: Optional[list[Callable[[dict[str, str]], bool]]] = None,
    ) -> list[dict[str, str]]:
        """
        Lookup objects based on specified tag name, optionally apply filters on the results.

        Args:
            tag_name: Name of the tag.
            filter_fns: List of filter functions applied on the results.

        Returns:
            List of objects in dictionary format.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to lookup tagged objects.

        Example::

            self._lookup_tagged_objects("TARGET_TAG_NAME", [lambda d: d["entityName"] == "MY_FV"])

        """
        # NOTE: use ENTITY_DETAIL system fn to query objects from tag for it to work in
        # processes using owner's right. e.g. Streamlit, or stored procedure
        try:
            res = self._session.sql(
                f"""
                SELECT ENTITY_DETAIL('TAG','{self._get_fully_qualified_name(tag_name)}', '["TAG_REFERENCES_INTERNAL"]');
            """
            ).collect(statement_params=self._telemetry_stmp)
            entity_detail = json.loads(res[0][0])
            results = entity_detail["referencedEntities"]["tagReferenceList"]
            return self._filter_results(results, filter_fns)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to lookup tagged objects for {tag_name}: {e}"),
            ) from e

    def _collapse_object_versions(self) -> list[pkg_version.Version]:
        try:
            res = self._lookup_tagged_objects(_FEATURE_STORE_OBJECT_TAG)
        except Exception:
            # since this is a best effort user warning to upgrade pkg versions
            # we are treating failures as benign error
            return []
        versions = set()
        compatibility_breakage_detected = False
        for r in res:
            info = _FeatureStoreObjInfo.from_json(r["tagValue"])
            if info.type == _FeatureStoreObjTypes.UNKNOWN:
                compatibility_breakage_detected = True
            versions.add(pkg_version.parse(info.pkg_version))

        sorted_versions = sorted(versions, reverse=True)
        if compatibility_breakage_detected:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_PACKAGE_OUTDATED,
                original_exception=RuntimeError(
                    f"The current snowflake-ml-python version {snowml_version.VERSION} is out of date, "
                    + f"please upgrade to at least {sorted_versions[0]}."
                ),
            )
        return sorted_versions

    def _validate_feature_view_name_and_version_input(
        self, feature_view: Union[FeatureView, str], version: Optional[str] = None
    ) -> FeatureView:
        if isinstance(feature_view, str):
            if version is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError("Version must be provided when argument feature_view is a str."),
                )
            feature_view = self.get_feature_view(feature_view, version)
        elif not isinstance(feature_view, FeatureView):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Invalid type of argument feature_view. It must be either str or FeatureView type."
                ),
            )

        return feature_view

    @staticmethod
    def _extract_cluster_by_columns(cluster_by_clause: str) -> list[str]:
        # Use regex to extract elements inside the parentheses.
        match = re.search(r"\((.*?)\)", cluster_by_clause)
        if match:
            # Handle both quoted and unquoted column names.
            return re.findall(identifier.SF_IDENTIFIER_RE, match.group(1))
        return []

    def _build_select_clause_and_validate(
        self,
        feature_view: FeatureView,
        feature_names: Optional[list[str]],
        include_join_keys: bool = True,
    ) -> str:
        """Build SELECT clause for feature view queries and validate feature names.

        Args:
            feature_view: The feature view to build the clause for
            feature_names: Optional list of feature names to include
            include_join_keys: Whether to include join keys in the select clause

        Returns:
            SELECT clause string

        Raises:
            SnowflakeMLException: If requested feature names don't exist
        """
        if feature_names:
            # Validate feature names exist
            available_features = [f.name for f in feature_view.output_schema.fields]
            for feature_name in feature_names:
                if feature_name not in available_features:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ARGUMENT,
                        original_exception=ValueError(
                            f"Feature '{feature_name}' not found in feature view. "
                            f"Available features: {available_features}"
                        ),
                    )

            # Build select clause with join keys and requested features
            select_columns = []
            if include_join_keys:
                all_join_keys = []
                for entity in feature_view.entities:
                    all_join_keys.extend([key.resolved() for key in entity.join_keys])
                select_columns.extend([f'"{key}"' for key in all_join_keys])

            select_columns.extend([f'"{name}"' for name in feature_names])
            return ", ".join(select_columns)
        else:
            # Select all columns
            return "*"

    def _build_where_clause_for_keys(
        self,
        feature_view: FeatureView,
        keys: Optional[list[list[Any]]],
        use_binds: bool = False,
    ) -> tuple[str, list[Any]]:
        """Build WHERE clause for key filtering.

        Args:
            feature_view: The feature view to build the clause for.
            keys: Optional list of key value lists to filter by.
            use_binds: If True, use ``?`` placeholders and return bind params.
                If False, interpolate values as string literals.

        Returns:
            Tuple of (WHERE clause string, bind params list). Params is empty
            when use_binds is False or no keys are provided.

        Raises:
            SnowflakeMLException: If key structure is invalid.
        """
        if not keys:
            return "", []

        all_join_keys = []
        for entity in feature_view.entities:
            all_join_keys.extend([key.resolved() for key in entity.join_keys])

        for key_values in keys:
            if len(key_values) != len(all_join_keys):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"Each key must have {len(all_join_keys)} values for join keys {all_join_keys}, "
                        f"got {len(key_values)} values"
                    ),
                )

        params: list[Any] = []
        where_conditions = []
        for key_values in keys:
            key_conditions = []
            for join_key, value in zip(all_join_keys, key_values):
                if use_binds:
                    key_conditions.append(f'"{join_key}" = ?')
                    params.append(value)
                else:
                    safe_value = str(value).replace("'", "''")
                    key_conditions.append(f"\"{join_key}\" = '{safe_value}'")
            where_conditions.append(f"({' AND '.join(key_conditions)})")

        return f" WHERE {' OR '.join(where_conditions)}", params


def _get_store_type(store_type: Union[fv_mod.StoreType, str]) -> fv_mod.StoreType:
    """Return a StoreType enum from a Union[StoreType, str].

    Args:
        store_type: Store type enum or string value.

    Returns:
        StoreType enum value.
    """
    if isinstance(store_type, str):
        return fv_mod.StoreType(store_type.lower())
    return store_type
