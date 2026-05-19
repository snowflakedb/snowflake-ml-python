"""Group multiple FeatureViews under one Postgres-backed Online Feature Table.

Example::

    fg = FeatureGroup(
        name="fraud_features",
        features=[
            user_fv.slice(["total_spend_30d"]),
            txn_fv.with_name("txn"),
        ],
        desc="Features for fraud detection",
        auto_prefix=True,
    )
    fs.register_feature_group(fg, "v1")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import snowflake.ml.feature_store.feature_view as fv_mod
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import online_service
from snowflake.ml.feature_store.feature_view import (
    _FEATURE_VIEW_NAME_DELIMITER,
    _FEATURE_VIEW_VERSION_MAX_LENGTH,
    _FEATURE_VIEW_VERSION_RE,
    _ONLINE_TABLE_SUFFIX,
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
    OnlineStoreType,
    get_feature_prefix,
)
from snowflake.ml.feature_store.metadata_manager import (
    FeatureGroupMetadata,
    FeatureGroupSourceRef,
)
from snowflake.ml.feature_store.spec.builder import FeatureViewSpecBuilder
from snowflake.ml.feature_store.spec.enums import FeatureViewKind
from snowflake.snowpark.types import BooleanType, StringType, StructField, StructType

logger = logging.getLogger(__name__)


class FeatureGroupVersion(str):
    """User-facing FeatureGroup version; same alphabet and length cap as :class:`FeatureViewVersion`."""

    def __new__(cls, version: str) -> FeatureGroupVersion:
        if not _FEATURE_VIEW_VERSION_RE.match(version) or len(version) > _FEATURE_VIEW_VERSION_MAX_LENGTH:
            raise ValueError(
                f"`{version}` is not a valid feature group version. "
                "It must start with letter or digit, and followed by letter, digit, '_', '-' or '.'. "
                f"The length limit is {_FEATURE_VIEW_VERSION_MAX_LENGTH}."
            )
        return super().__new__(cls, version)

    def __init__(self, version: str) -> None:
        super().__init__()


if TYPE_CHECKING:
    import pandas as pd

    from snowflake.ml.feature_store.feature_store import FeatureStore
    from snowflake.ml.feature_store.spec.models import FeatureViewSpec
    from snowflake.snowpark import DataFrame


_FEATURE_GROUP_NAME_MAX_LENGTH = 255


def _validate_feature_group_name(name: str) -> None:
    """Reject FG names that aren't safe to use as a SQL identifier.

    Mirrors :meth:`FeatureView._validate`: any string is permitted except
    the FG/FV version delimiter ``$``. Round-trip stability under different
    casing comes from ``SqlIdentifier(name).resolved()`` at the
    register/list/get/delete call sites, the same mechanism FV uses.

    Args:
        name: Candidate FG name.

    Raises:
        ValueError: If *name* is empty, exceeds 255 characters, or contains
            the FV/FG version delimiter ``$``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("FeatureGroup name must be a non-empty string.")
    if len(name) > _FEATURE_GROUP_NAME_MAX_LENGTH:
        raise ValueError(f"FeatureGroup name `{name}` exceeds the {_FEATURE_GROUP_NAME_MAX_LENGTH} character limit.")
    if _FEATURE_VIEW_NAME_DELIMITER in name:
        raise ValueError(f"FeatureGroup name `{name}` contains invalid character `{_FEATURE_VIEW_NAME_DELIMITER}`.")


def unwrap_fv(item: Union[FeatureView, FeatureViewSlice]) -> FeatureView:
    """Return the underlying :class:`FeatureView` for an FG/training-set source item.

    Args:
        item: FG/training-set source (FeatureView or FeatureViewSlice).

    Returns:
        ``item.feature_view_ref`` when ``item`` is a slice, otherwise ``item`` itself.
    """
    return item.feature_view_ref if isinstance(item, FeatureViewSlice) else item


def _ref_key(feature: Union[FeatureView, FeatureViewSlice]) -> tuple[str, Optional[str]]:
    """Return the ``(name, version)`` identity tuple for a feature item.

    Args:
        feature: A :class:`FeatureView` or :class:`FeatureViewSlice`.

    Returns:
        Tuple of ``(resolved_name, version_str_or_None)`` used as the
        canonical identity of the underlying feature view.
    """
    fv = unwrap_fv(feature)
    name = fv.name.resolved()
    version = str(fv.version) if fv.version else None
    return (name, version)


class FeatureGroup:
    """Group of registered FeatureViews materialized as one Postgres OFT.

    The OFT's primary key is the ordered union of the source FVs' join keys,
    so sources may join at different grains (coarser sources broadcast over
    the wider key). Authoring-only: source state checks (registered, online,
    POSTGRES) run at registration time inside :class:`FeatureStore`.
    """

    def __init__(
        self,
        name: str,
        features: list[Union[FeatureView, FeatureViewSlice]],
        *,
        desc: str = "",
        auto_prefix: bool = True,
    ) -> None:
        """Construct a draft FeatureGroup.

        Args:
            name: User-facing FG name (any non-empty string up to 255 chars
                without the ``$`` version delimiter).
            features: Non-empty list of :class:`FeatureView` /
                :class:`FeatureViewSlice` references to include.
            desc: Human-readable description.
            auto_prefix: When ``True``, prefix each output column with
                ``"<fv_name>_<fv_version>_"``. Overridden per item by an
                explicit :meth:`FeatureView.with_name`.

        Raises:
            ValueError: If *name* is invalid, *features* is empty, an item is
                not a :class:`FeatureView` / :class:`FeatureViewSlice`, or two
                items share the same ``(fv_name, fv_version)``.
        """
        _validate_feature_group_name(name)

        if not features:
            raise ValueError("FeatureGroup requires at least one feature view.")

        seen: dict[tuple[str, Optional[str]], int] = {}
        duplicates: list[str] = []
        for idx, item in enumerate(features):
            if not isinstance(item, (FeatureView, FeatureViewSlice)):
                raise ValueError(
                    f"FeatureGroup features[{idx}] must be a FeatureView or FeatureViewSlice; "
                    f"got {type(item).__name__}."
                )
            key = _ref_key(item)
            if key in seen:
                duplicates.append(f"{key[0]}@{key[1]}")
            else:
                seen[key] = idx
        if duplicates:
            raise ValueError(
                "FeatureGroup features must be unique by (name, version); duplicates: "
                f"{sorted(set(duplicates))}. To attach multiple slices/aliases of the same FV, "
                "merge them into one slice or use two separate FeatureGroups."
            )

        self._name: str = name
        self._features: list[Union[FeatureView, FeatureViewSlice]] = list(features)
        self._desc: str = desc
        self._auto_prefix: bool = auto_prefix
        # Set on register/get_feature_group; ``None`` for drafts (mirrors :attr:`FeatureView._version`).
        self._version: Optional[FeatureGroupVersion] = None
        # Hydrated on ``get_feature_group``; ``read_feature_group`` raises if still ``None``.
        self._postgres_online_query_url: Optional[str] = None

    # -----------------------------------------------------------------------
    # Public surface
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        """User-facing FeatureGroup name."""
        return self._name

    @property
    def version(self) -> Optional[FeatureGroupVersion]:
        """User-facing FeatureGroup version.

        Set by :meth:`FeatureStore.register_feature_group` and
        :meth:`FeatureStore.get_feature_group`; ``None`` on locally-constructed
        drafts that have not been registered.

        Returns:
            The :class:`FeatureGroupVersion` assigned at registration, or
            ``None`` for unregistered drafts.
        """
        return self._version

    @property
    def features(self) -> list[Union[FeatureView, FeatureViewSlice]]:
        """Defensive copy of the source FV references."""
        return list(self._features)

    @property
    def desc(self) -> str:
        """Free-text description (persisted as the OFT ``COMMENT``)."""
        return self._desc

    @property
    def auto_prefix(self) -> bool:
        """Whether columns are auto-prefixed by ``<fv>_<version>_``."""
        return self._auto_prefix

    @property
    def output_columns(self) -> list[str]:
        """Resolved output column names in source order, with prefixes applied per item.

        Never raises on collisions; duplicate detection lives in
        :meth:`FeatureStore.register_feature_group`.

        Returns:
            Output column names in source-emission order.
        """
        cols: list[str] = []
        for item in self._features:
            prefix = get_feature_prefix(item, self._auto_prefix)
            base_names = self._feature_column_names(item)
            for n in base_names:
                raw = f"{prefix}{n}" if prefix else n
                cols.append(identifier.concat_names([raw]))
        return cols

    # -----------------------------------------------------------------------
    # Spec construction
    # -----------------------------------------------------------------------

    def _to_spec(self, *, database: str, schema: str, version: str) -> FeatureViewSpec:
        """Translate this draft FeatureGroup into a validated spec payload.

        Builds an internal ``(name, version)``-keyed prefix map and delegates
        to :class:`FeatureViewSpecBuilder`. Returns a :class:`FeatureViewSpec`
        ready for ``CREATE ONLINE FEATURE TABLE ... FROM SPECIFICATION``.

        Args:
            database: FeatureStore database name.
            schema: FeatureStore schema name.
            version: User-facing FeatureGroup version (already validated by
                :class:`FeatureGroupVersion`). Embedded in the spec metadata
                and surfaced to the Online Service Query API at read time.

        Returns:
            FeatureViewSpec: validated spec for this FeatureGroup, ready for
            ``CREATE ONLINE FEATURE TABLE ... FROM SPECIFICATION``.
        """
        prefix_map: dict[tuple[str, Optional[str]], str] = {}
        for item in self._features:
            prefix = get_feature_prefix(item, self._auto_prefix)
            if not prefix:
                continue
            key = _ref_key(item)
            prefix_map[key] = prefix

        builder = FeatureViewSpecBuilder(
            FeatureViewKind.FeatureGroup,
            database=database,
            schema=schema,
            name=self._name,
            version=version,
        )
        builder.set_sources(list(self._features))
        builder.set_source_prefixes(prefix_map)
        return builder.build()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _feature_column_names(item: Union[FeatureView, FeatureViewSlice]) -> list[str]:
        """Return the upstream feature column names contributed by *item*.

        For a :class:`FeatureViewSlice`, preserves the caller-requested
        slice order. For a full :class:`FeatureView`, returns
        ``feature_names`` in declaration order.

        Args:
            item: Source FeatureView or FeatureViewSlice.

        Returns:
            Resolved feature column names in source-emission order.
        """
        if isinstance(item, FeatureViewSlice):
            return [n.resolved() for n in item.names]
        return [n.resolved() for n in item.feature_names]

    def __repr__(self) -> str:
        refs = ", ".join(f"{k[0]}@{k[1]}" for k in (_ref_key(f) for f in self._features))
        version_repr = repr(str(self._version)) if self._version is not None else "None"
        return (
            f"FeatureGroup(name={self._name!r}, version={version_repr}, features=[{refs}], "
            f"desc={self._desc!r}, auto_prefix={self._auto_prefix!r})"
        )


# Registration-time helpers used by FeatureStore.{register,get,delete,list}_feature_group.
# Kept here (rather than in feature_store.py) because every concern they encode
# is FG-specific: OFT naming, source-FV preconditions, and metadata <-> FeatureGroup
# round-trip. The OFT's primary key is the ordered union of source FVs' join keys
# (derived by the spec builder), so coarser sources broadcast over the wider PK.


# Schema returned by ``list_feature_groups``. Mirrors the shape of
# ``_LIST_FEATURE_VIEW_SCHEMA`` minus FV-only refresh/storage/stream columns.
_LIST_FEATURE_GROUP_SCHEMA = StructType(
    [
        StructField("name", StringType()),
        StructField("version", StringType()),
        StructField("desc", StringType()),
        StructField("owner", StringType()),
        StructField("auto_prefix", BooleanType()),
        StructField("sources", StringType()),  # JSON-encoded list of FeatureGroupSourceRef dicts
        StructField("output_columns", StringType()),  # JSON-encoded list of resolved column names
    ]
)


def feature_group_oft_name(name: Union[SqlIdentifier, str], version: Union[FeatureGroupVersion, str]) -> SqlIdentifier:
    """Return the SqlIdentifier ``<name>$<version>$ONLINE`` for the OFT backing a FeatureGroup.

    Inputs are canonicalized so any caller-supplied casing yields the same
    identifier as the metadata key written at register time.

    Args:
        name: FeatureGroup name.
        version: FeatureGroup version.

    Returns:
        SqlIdentifier for the OFT physical name.
    """
    canonical_name = name.resolved() if isinstance(name, SqlIdentifier) else SqlIdentifier(name).resolved()
    canonical_version = str(version) if isinstance(version, FeatureGroupVersion) else str(FeatureGroupVersion(version))
    return SqlIdentifier(
        identifier.concat_names(
            [
                canonical_name,
                _FEATURE_VIEW_NAME_DELIMITER,
                canonical_version,
                _ONLINE_TABLE_SUFFIX,
            ]
        )
    )


def validate_sources_online_postgres(
    features: list[Union[FeatureView, FeatureViewSlice]],
    *,
    consumer_label: str,
) -> None:
    """Reject source FVs that are not online-enabled on Postgres.

    FG OFTs materialize as a single Postgres-backed table, so every source
    FV must already be online on Postgres for the FG OFT to read its rows
    directly.

    Args:
        features: Source items (FeatureViews or FeatureViewSlices).
        consumer_label: Human-readable label for the caller (used in error
            messages, e.g., ``"FeatureGroup"``).

    Raises:
        SnowflakeMLException: ``[ValueError]`` if any source FV is not online
            or its store_type is not POSTGRES.
    """
    offenders: list[str] = []
    for f in features:
        fv = unwrap_fv(f)
        if not fv.online:
            offenders.append(f"{fv.name.resolved()}@{fv.version} (online=False)")
            continue
        cfg = fv.online_config
        if cfg is None or cfg.store_type != OnlineStoreType.POSTGRES:
            store = cfg.store_type.value if cfg is not None else "None"
            offenders.append(f"{fv.name.resolved()}@{fv.version} (store_type={store})")
    if offenders:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"{consumer_label} requires every source FeatureView to be registered with "
                f"online=True and store_type=OnlineStoreType.POSTGRES. Offending sources: "
                f"{sorted(set(offenders))}."
            ),
        )


def reject_name_collision(feature_store: FeatureStore, fg_name: str, fg_version: str) -> None:
    """Reject FG ``(fg_name, fg_version)`` if it collides with an existing FV or FG.

    Cross-checks against:

    - Existing OFTs at ``<fg_name>$<fg_version>$ONLINE`` (another FG with the
      same name/version already registered).
    - Registered FeatureViews whose resolved user-facing name equals the
      candidate FG name (the FV/FG user-facing namespaces are shared).

    Args:
        feature_store: Calling FeatureStore (for backend lookups).
        fg_name: The candidate FG name.
        fg_version: The candidate FG version.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if a collision is detected.
    """
    normalized = SqlIdentifier(fg_name).resolved()
    fv_rows = feature_store._get_fv_backend_representations(SqlIdentifier(fg_name), prefix_match=True)
    for row, _ in fv_rows:
        phys = row["name"]
        head = phys.split(_FEATURE_VIEW_NAME_DELIMITER)[0]
        if head == normalized:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.OBJECT_ALREADY_EXISTS,
                original_exception=ValueError(
                    f"Cannot register FeatureGroup {fg_name}: a FeatureView with the same name already exists."
                ),
            )

    oft_name = feature_group_oft_name(fg_name, fg_version)
    existing_oft = feature_store._find_object("ONLINE FEATURE TABLES", oft_name)
    if existing_oft:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.OBJECT_ALREADY_EXISTS,
            original_exception=ValueError(f"FeatureGroup {fg_name}/{fg_version} already exists."),
        )


def build_source_refs(
    features: list[Union[FeatureView, FeatureViewSlice]],
) -> list[FeatureGroupSourceRef]:
    """Translate FG features into ``FeatureGroupSourceRef`` for persistence.

    Slices are encoded as ``slice_columns``; an unsliced FV gets ``None``.
    ``alias`` round-trips :meth:`FeatureView.with_name` so the reconstructed
    FG produces identical output column names.

    Args:
        features: Source items (FeatureViews or FeatureViewSlices) from the
            FeatureGroup being persisted.

    Returns:
        Persistable ``FeatureGroupSourceRef`` list, one per source item, in
        the original order.
    """
    refs: list[FeatureGroupSourceRef] = []
    for f in features:
        if isinstance(f, FeatureViewSlice):
            fv = f.feature_view_ref
            slice_columns: Optional[list[str]] = [n.resolved() for n in f.names]
        else:
            fv = f
            slice_columns = None
        assert fv.version is not None  # checked by register_feature_group precondition
        refs.append(
            FeatureGroupSourceRef(
                fv_name=fv.name.resolved(),
                fv_version=str(fv.version),
                slice_columns=slice_columns,
                alias=fv.column_alias,
            )
        )
    return refs


def compose_from_metadata(feature_store: FeatureStore, meta: FeatureGroupMetadata) -> FeatureGroup:
    """Reconstruct a :class:`FeatureGroup` from its persisted metadata.

    Args:
        feature_store: Calling FeatureStore (used to fetch each source FV).
        meta: Persisted FG metadata previously written by ``register_feature_group``.

    Returns:
        A FeatureGroup equivalent to the one originally registered, with
        :attr:`FeatureGroup.version` populated from ``meta.version``.
    """
    features: list[Union[FeatureView, FeatureViewSlice]] = []
    for src in meta.sources:
        fv = feature_store.get_feature_view(src.fv_name, src.fv_version)
        item: Union[FeatureView, FeatureViewSlice] = (
            fv.slice(src.slice_columns) if src.slice_columns is not None else fv
        )
        if src.alias is not None:
            item = item.with_name(src.alias)
        features.append(item)
    fg = FeatureGroup(
        name=meta.name,
        features=features,
        desc=meta.desc,
        auto_prefix=meta.auto_prefix,
    )
    fg._version = FeatureGroupVersion(meta.version)
    return fg


def delete_feature_group(fs: FeatureStore, name: str, version: str) -> None:
    """Drop the OFT and metadata row for a registered :class:`FeatureGroup`.

    Idempotent on both the OFT (``DROP ... IF EXISTS``) and the metadata row.

    Args:
        fs: The :class:`FeatureStore` invoking this operation.
        name: FeatureGroup name.
        version: FeatureGroup version.

    Raises:
        SnowflakeMLException: ``[RuntimeError]`` if ``DROP ONLINE FEATURE TABLE``
            fails.
    """
    canonical_version = FeatureGroupVersion(version)
    oft_name = feature_group_oft_name(name, canonical_version)
    fully_qualified_oft = fs._get_fully_qualified_name(oft_name)
    try:
        fs._session.sql(f"DROP ONLINE FEATURE TABLE IF EXISTS {fully_qualified_oft}").collect(
            statement_params=fs._telemetry_stmp
        )
    except Exception as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(f"Failed to drop FeatureGroup OFT {fully_qualified_oft}: {e}"),
        ) from e
    fs._metadata_manager.delete_feature_group_metadata(SqlIdentifier(name).resolved(), str(canonical_version))
    logger.info(f"Deleted FeatureGroup {name}/{version}.")


def get_feature_group(fs: FeatureStore, name: str, version: str) -> FeatureGroup:
    """Retrieve a previously registered FeatureGroup.

    Args:
        fs: The :class:`FeatureStore` invoking this operation.
        name: FeatureGroup name.
        version: FeatureGroup version.

    Returns:
        FeatureGroup: equivalent to the registered original, with version populated.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if no FeatureGroup with the
            given *(name, version)* exists.
    """
    canonical = SqlIdentifier(name).resolved()
    canonical_version = FeatureGroupVersion(version)
    meta = fs._metadata_manager.get_feature_group_metadata(canonical, str(canonical_version))
    if meta is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError(f"FeatureGroup {name}/{version} is not registered."),
        )
    fg = compose_from_metadata(fs, meta)
    # Cache the query URL so ``read_feature_group`` skips the per-call status round-trip.
    fs._hydrate_fg_postgres_online_service(fg)
    return fg


def list_feature_groups(fs: FeatureStore) -> DataFrame:
    """List FeatureGroups registered in this FeatureStore.

    Args:
        fs: The :class:`FeatureStore` invoking this operation.

    Returns:
        Snowpark DataFrame with one row per registered ``(name, version)`` and
        columns ``NAME, VERSION, DESC, OWNER, AUTO_PREFIX, SOURCES,
        OUTPUT_COLUMNS``. Empty when no FGs are registered.
    """
    # Imported lazily to avoid a circular import at module load.
    from snowflake.ml.feature_store.feature_store import (
        _FEATURE_STORE_OBJECT_TAG,
        _FeatureStoreObjInfo,
        _FeatureStoreObjTypes,
    )

    all_meta = fs._metadata_manager.list_feature_group_metadata()
    owner_by_oft: dict[str, Any] = {}
    # Tag-scan rows return ``entityName`` as a bare (unquoted) string; normalize
    # through ``SqlIdentifier(case_sensitive=True).resolved()`` so the join key
    # agrees byte-for-byte with the OFT physical name computed below.
    for r in fs._lookup_tagged_objects(
        _FEATURE_STORE_OBJECT_TAG,
        [
            lambda d: d.get("domain") == "TABLE",
            lambda d: _FeatureStoreObjInfo.from_json(d["tagValue"]).type == _FeatureStoreObjTypes.FEATURE_GROUP,
        ],
    ):
        owner_by_oft[SqlIdentifier(r["entityName"], case_sensitive=True).resolved()] = r.get("entityOwner")

    output_values: list[list[Any]] = []
    for meta in all_meta:
        oft_phys = feature_group_oft_name(meta.name, meta.version).resolved()
        owner = owner_by_oft.get(oft_phys)
        sources_json = json.dumps([s.to_dict() for s in meta.sources])
        # Legacy rows (no ``output_columns`` field) render as ``[]``; we
        # deliberately do NOT fall back to source rehydration here.
        output_cols = list(meta.output_columns or [])
        output_values.append(
            [
                meta.name,
                meta.version,
                meta.desc,
                owner,
                meta.auto_prefix,
                sources_json,
                json.dumps(output_cols),
            ]
        )

    return fs._session.create_dataframe(output_values, schema=_LIST_FEATURE_GROUP_SCHEMA)


def register_feature_group(fs: FeatureStore, feature_group: FeatureGroup, version: str) -> FeatureGroup:
    """Materialize a FeatureGroup as a Postgres-backed Online Feature Table.

    Validates preconditions (registered + online + POSTGRES source FVs, no name
    collision) before any side effect, then creates the OFT via ``CREATE ONLINE
    FEATURE TABLE ... FROM SPECIFICATION`` and persists ``FeatureGroupMetadata``
    so :func:`get_feature_group` can round-trip. The OFT is keyed by the ordered
    union of the source FVs' join keys, so heterogeneous-grain sources broadcast
    over the wider primary key.

    Args:
        fs: The :class:`FeatureStore` invoking this operation.
        feature_group: Draft FeatureGroup to register.
        version: User-facing FeatureGroup version (validated by
            :class:`FeatureGroupVersion`).

    Returns:
        FeatureGroup: equivalent to the input, with :attr:`FeatureGroup.version` populated.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if any precondition is violated
            (invalid version, unregistered / offline / non-Postgres source,
            missing entity, or name collision).
        SnowflakeMLException: ``[RuntimeError]`` if OFT creation, tagging, or
            metadata write fails.

    # noqa: DAR401
    """
    # Imported lazily to avoid a circular import at module load.
    from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes

    if not isinstance(feature_group, FeatureGroup):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"register_feature_group expects a FeatureGroup; got {type(feature_group).__name__}."
            ),
        )

    validated_version = FeatureGroupVersion(version)

    # DRAFT FVs cannot back an OFT.
    unregistered = [
        f"{fv.name.resolved()}@{fv.version}"
        for fv in (f.feature_view_ref if isinstance(f, FeatureViewSlice) else f for f in feature_group.features)
        if fv.status == FeatureViewStatus.DRAFT or fv.version is None
    ]
    if unregistered:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError(
                "FeatureGroup source FeatureView(s) are not registered: "
                f"{sorted(set(unregistered))}. Call fs.register_feature_view(...) on each source first."
            ),
        )

    validate_sources_online_postgres(feature_group.features, consumer_label="FeatureGroup")

    # Mirrors the entity-exists check in register_feature_view.
    for fv_ref in (f.feature_view_ref if isinstance(f, FeatureViewSlice) else f for f in feature_group.features):
        for e in fv_ref.entities:
            if not fs._validate_entity_exists(e.name):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(
                        f"Entity {e.name} (referenced by source FV "
                        f"{fv_ref.name.resolved()}@{fv_ref.version}) is not registered."
                    ),
                )

    reject_name_collision(fs, feature_group.name, validated_version)

    online_service.assert_online_service_running_with_query_endpoint(
        fs._session,
        fs._config.database,
        fs._config.schema,
        statement_params=fs._telemetry_stmp,
    )

    # Building the spec also surfaces builder-side rules (duplicate columns,
    # allowed upstream FV kinds) and derives the OFT's ordered superset PK
    # across heterogeneous-key sources, before any side effect.
    spec = feature_group._to_spec(
        database=fs._config.database.resolved(),
        schema=fs._config.schema.resolved(),
        version=validated_version,
    )
    primary_key = list(spec.spec.ordered_entity_column_names)
    assert primary_key, "FeatureGroup spec must derive at least one entity column"

    oft_name = feature_group_oft_name(feature_group.name, validated_version)
    fully_qualified_oft = fs._get_fully_qualified_name(oft_name)
    spec_json = spec.to_json()
    canonical_name = SqlIdentifier(feature_group.name).resolved()

    # Track only what THIS call created so the failure rollback never tears
    # down a concurrent caller's resources (mirrors register_feature_view).
    created_resources: list[tuple[_FeatureStoreObjTypes, str]] = []
    metadata_saved = False
    try:
        # Reuse the FV OFT SQL builders so clause ordering / quoting / WAREHOUSE
        # precedence stay in lock-step with register_feature_view; FG-specific
        # bits are the spec-backed source clause and ``_NON_BATCH_OFT_TARGET_LAG``.
        create_sql = fv_mod.build_oft_create_sql(
            fully_qualified_oft_name=fully_qualified_oft,
            primary_key_clause=fv_mod.build_oft_primary_key_clause(primary_key),
            target_lag=fv_mod._NON_BATCH_OFT_TARGET_LAG,
            source_clause=f"FROM SPECIFICATION $${spec_json}$$",
            warehouse_clause=fv_mod.build_oft_warehouse_clause(None, fs._default_warehouse),
        )
        fs._session.sql(create_sql).collect(statement_params=fs._telemetry_stmp)
        created_resources.append((_FeatureStoreObjTypes.ONLINE_FEATURE_TABLE, fully_qualified_oft))
        fs._tag_oft(fully_qualified_oft, _FeatureStoreObjTypes.FEATURE_GROUP)

        # Persist under the resolved (canonical) name so OFT-entity-name lookups
        # (which Snowflake upper-cases for unquoted identifiers) hit this row.
        # Capture ``output_columns`` now while source FV objects are live so
        # ``list_feature_groups`` does not have to rehydrate them per row.
        metadata = FeatureGroupMetadata(
            name=canonical_name,
            version=str(validated_version),
            desc=feature_group.desc,
            auto_prefix=feature_group.auto_prefix,
            sources=build_source_refs(feature_group.features),
            output_columns=list(feature_group.output_columns),
        )
        fs._metadata_manager.save_feature_group_metadata(metadata)
        metadata_saved = True
    except Exception as e:
        fs._rollback_created_resources(created_resources)
        if metadata_saved:
            try:
                fs._metadata_manager.delete_feature_group_metadata(canonical_name, str(validated_version))
            except Exception as cleanup_err:
                logger.warning(
                    f"Best-effort rollback failed to delete FG metadata for "
                    f"{feature_group.name}/{validated_version}: {cleanup_err}"
                )
        if isinstance(e, snowml_exceptions.SnowflakeMLException):
            raise
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
            original_exception=RuntimeError(
                f"Failed to register FeatureGroup {feature_group.name}/{validated_version}: {e}"
            ),
        ) from e

    logger.info(f"Registered FeatureGroup {feature_group.name}/{validated_version} successfully.")
    return fs.get_feature_group(feature_group.name, str(validated_version))


def read_feature_group(
    fs: FeatureStore,
    feature_group: Union[FeatureGroup, str],
    version: Optional[str],
    *,
    keys: list[list[Any]],
    store_type: Union[fv_mod.StoreType, str],
) -> pd.DataFrame:
    """Read FG values via the Online Service Query API. See :meth:`FeatureStore.read_feature_group`.

    Args:
        fs: The :class:`FeatureStore` invoking this operation.
        feature_group: A hydrated :class:`FeatureGroup` or its name.
        version: Required when *feature_group* is a string; otherwise optional
            and validated against the passed FG.
        keys: Non-empty list of entity rows aligned with the FG's join keys.
        store_type: Only :attr:`StoreType.ONLINE` is supported today.

    Returns:
        ``pandas.DataFrame`` with the join-key columns followed by the FG's
        :attr:`~FeatureGroup.output_columns`.

    Raises:
        SnowflakeMLException: ``[ValueError]`` for empty keys, missing /
            disagreeing version, unregistered FG, or pre-RUNNING Online
            Service. ``[NotImplementedError]`` if *store_type* is not
            :attr:`StoreType.ONLINE`.
    """
    # Lazy import: feature_store imports this module at top level.
    from snowflake.ml.feature_store.feature_store import _get_store_type

    if _get_store_type(store_type) != fv_mod.StoreType.ONLINE:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=NotImplementedError(
                "read_feature_group(store_type=StoreType.OFFLINE) is not yet supported. "
                "Only StoreType.ONLINE (Postgres-backed) reads are available today; "
                "offline reads will be added in a future release."
            ),
        )

    if isinstance(feature_group, str):
        if version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "read_feature_group requires `version` when `feature_group` is a string."
                ),
            )
        fg = fs.get_feature_group(feature_group, version)
    else:
        if version is not None and FeatureGroupVersion(version) != feature_group.version:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"version={version!r} disagrees with the passed FeatureGroup "
                    f"({feature_group.name}/{feature_group.version}); pass only one."
                ),
            )
        fg = feature_group

    if fg.version is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "read_feature_group requires a registered FeatureGroup; " "call fs.register_feature_group(...) first."
            ),
        )

    if not keys:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "read_feature_group requires at least one row in `keys`; "
                "unbounded table scans are not supported via the Online Service Query API."
            ),
        )

    query_url = fg._postgres_online_query_url
    if not query_url:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Online read for this FeatureGroup requires a hydrated query endpoint. "
                "Call get_feature_group(name, version) again after the Online Service is RUNNING."
            ),
        )

    # Ordered union of the source FVs' join keys — same derivation the spec
    # builder uses to compute the OFT's primary key, so the read request PK
    # and the materialized OFT PK stay in lock-step.
    seen: set[str] = set()
    join_names: list[str] = []
    for source in fg.features:
        for entity in unwrap_fv(source).entities:
            for jk in entity.join_keys:
                resolved = jk.resolved()
                if resolved not in seen:
                    seen.add(resolved)
                    join_names.append(resolved)
    # Each source FV contributes types only for the join keys it carries;
    # broader sources fill in keys that narrower sources don't have.
    join_key_field_types: dict[str, Any] = {}
    for source in fg.features:
        source_fv = unwrap_fv(source)
        for f in source_fv.output_schema.fields:
            if f.name in seen and f.name not in join_key_field_types:
                join_key_field_types[f.name] = f.datatype

    rows, schema = online_service.read_postgres_online_features(
        session=fs._session,
        query_url=query_url,
        feature_view_name=str(fg.name),
        feature_view_version=str(fg.version),
        join_key_names=join_names,
        keys=keys,
        feature_names=None,
        join_key_field_types=join_key_field_types,
        object_type="feature_group",
        http_client=fs._get_or_create_online_http_client(),
    )
    return online_service.rows_to_pandas_for_postgres_online(rows, schema)


def prepare_training_set_args(
    fs: FeatureStore,
    *,
    feature_group: Union[FeatureGroup, tuple[str, str]],
    exclude_columns: Optional[list[str]],
    include_feature_view_timestamp_col: bool,
    auto_prefix: bool,
    join_method: Literal["sequential", "cte"],
) -> tuple[list[Union[FeatureView, FeatureViewSlice]], bool, Literal["sequential", "cte"]]:
    """Validate FG-incompatible ``generate_training_set`` params and return ``(features, auto_prefix, join_method)``.

    The FG owns its features, prefixing, and join strategy, so this helper
    rejects overrides rather than silently dropping them, then resolves the
    FG (looking up by ``(name, version)`` if a tuple is passed).

    Args:
        fs: The :class:`FeatureStore` invoking this operation.
        feature_group: A registered :class:`FeatureGroup` or a
            ``(name, version)`` tuple.
        exclude_columns: Must be ``None``; FGs return a predetermined set.
        include_feature_view_timestamp_col: Must be ``False`` for FGs.
        auto_prefix: Must be ``False``; FGs use their own ``auto_prefix``.
        join_method: Must be ``"sequential"`` (the default); FGs always join
            via ``"cte"``.

    Returns:
        Tuple ``(features, auto_prefix, join_method)`` ready to feed into
        the existing FV-path join code.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if any FG-incompatible
            parameter is set, or the FG is unregistered.
    """
    if exclude_columns is not None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "exclude_columns is not supported with `feature_group`; "
                "FeatureGroups return a predetermined output column set. Drop columns post-hoc."
            ),
        )
    if include_feature_view_timestamp_col:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("include_feature_view_timestamp_col is not supported with `feature_group`."),
        )
    if auto_prefix:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "auto_prefix is not supported with `feature_group`; "
                "the FeatureGroup's own auto_prefix setting is used."
            ),
        )
    if join_method != "sequential":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "join_method is not supported with `feature_group`; FGs always join via 'cte'."
            ),
        )

    if isinstance(feature_group, FeatureGroup):
        fg = feature_group
        if fg.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "generate_training_set requires a registered FeatureGroup; "
                    "call fs.register_feature_group(...) first."
                ),
            )
    else:
        fg_name, fg_version = feature_group
        fg = fs.get_feature_group(fg_name, fg_version)

    return list(fg.features), fg.auto_prefix, "cte"
