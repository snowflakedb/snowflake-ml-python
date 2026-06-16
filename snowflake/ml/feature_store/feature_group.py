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
    from snowflake.snowpark import DataFrame, Session


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

        Drops any per-source feature name that resolves to one of the FG's
        superset primary-key columns; otherwise an RTFV source whose
        ``realtime_config.output_schema`` re-declares the join key (the
        canonical ``compute_fn`` return shape) would double-emit the PK on
        top of the OFT's PK columns.

        Never raises on the remaining collisions; duplicate detection lives
        in :meth:`FeatureStore.register_feature_group` via the spec builder.

        Returns:
            Output column names in source-emission order.
        """
        return _resolve_fg_output_columns(self)

    # -----------------------------------------------------------------------
    # Spec construction
    # -----------------------------------------------------------------------

    def _to_spec(
        self, *, database: str, schema: str, version: str, session: Optional[Session] = None
    ) -> FeatureViewSpec:
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
            session: Optional Snowpark session used to read upstream
                FeatureView column shapes from the materialized DT/View, so
                source column shapes match the upstream's stored
                ``OutputColumn`` for the Online Service exact-shape check.

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
            session=session,
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
        ``feature_names`` in declaration order. For an RTFV source whose
        ``feature_names`` is empty (rehydrated RTFVs leave ``_feature_desc``
        empty), falls back to the canonical ``realtime_config.output_schema``
        minus the RTFV's own entity columns, so the source contributes its
        computed feature columns rather than nothing.

        Args:
            item: Source FeatureView or FeatureViewSlice.

        Returns:
            Resolved feature column names in source-emission order.
        """
        if isinstance(item, FeatureViewSlice):
            return [n.resolved() for n in item.names]
        if item.realtime_config is not None:
            entity_cols = set(item.ordered_entity_columns)
            return [f.name for f in item.realtime_config.output_schema.fields if f.name not in entity_cols]
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


def reject_name_collision(
    feature_store: FeatureStore,
    fg_name: str,
    fg_version: str,
    *,
    consumer_label: str = "FeatureGroup",
    oft_name: Optional[SqlIdentifier] = None,
) -> None:
    """Reject ``(name, version)`` if it collides with an existing FV or OFT.

    Cross-checks against:

    - Registered FeatureViews whose user-facing name equals the candidate
      (FV and FG / RTFV share the ``<name>$<version>$ONLINE`` OFT shape).
    - Existing OFTs at the resolved OFT name.

    Args:
        feature_store: Calling FeatureStore (for backend lookups).
        fg_name: The candidate name.
        fg_version: The candidate version.
        consumer_label: Human-readable caller label used in error messages
            (e.g. ``"FeatureGroup"`` or ``"realtime feature view"``).
        oft_name: Optional pre-resolved OFT identifier. Falls back to the
            FG-style :func:`feature_group_oft_name` when omitted.

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
                    f"Cannot register {consumer_label} {fg_name}: a FeatureView with the same name already exists."
                ),
            )

    resolved_oft_name = oft_name if oft_name is not None else feature_group_oft_name(fg_name, fg_version)
    existing_oft = feature_store._find_object("ONLINE FEATURE TABLES", resolved_oft_name)
    if existing_oft:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.OBJECT_ALREADY_EXISTS,
            original_exception=ValueError(f"{consumer_label} {fg_name}/{fg_version} already exists."),
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


def hydrate_source_refs(
    feature_store: FeatureStore,
    sources: list[FeatureGroupSourceRef],
) -> list[Union[FeatureView, FeatureViewSlice]]:
    """Inverse of :func:`build_source_refs`: re-fetch + re-project each source FV.

    Args:
        feature_store: Calling FeatureStore (used to fetch each source FV).
        sources: Persistable source refs previously written by
            :func:`build_source_refs`.

    Returns:
        Live FeatureView / FeatureViewSlice objects in the original order,
        with slice / alias projection reapplied.
    """
    features: list[Union[FeatureView, FeatureViewSlice]] = []
    for src in sources:
        fv = feature_store.get_feature_view(src.fv_name, src.fv_version)
        item: Union[FeatureView, FeatureViewSlice] = (
            fv.slice(src.slice_columns) if src.slice_columns is not None else fv
        )
        if src.alias is not None:
            item = item.with_name(src.alias)
        features.append(item)
    return features


def compose_from_metadata(feature_store: FeatureStore, meta: FeatureGroupMetadata) -> FeatureGroup:
    """Reconstruct a :class:`FeatureGroup` from its persisted metadata.

    Args:
        feature_store: Calling FeatureStore (used to fetch each source FV).
        meta: Persisted FG metadata previously written by ``register_feature_group``.

    Returns:
        A FeatureGroup equivalent to the one originally registered, with
        :attr:`FeatureGroup.version` populated from ``meta.version``.
    """
    features = hydrate_source_refs(feature_store, meta.sources)
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


def _fg_superset_pk(fg: FeatureGroup) -> list[str]:
    """Ordered union of source FVs' join keys (canonicalized), in source order.

    Mirrors the derivation the spec builder uses to compute the OFT's primary
    key, so register-time / read-time / property logic stays in lock-step
    with the materialized OFT.

    Args:
        fg: The FeatureGroup whose source FVs' join keys to union.

    Returns:
        Resolved join-key names in first-seen order across ``fg.features``.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for source in fg.features:
        for entity in unwrap_fv(source).entities:
            for jk in entity.join_keys:
                resolved = jk.resolved()
                if resolved not in seen:
                    seen.add(resolved)
                    ordered.append(resolved)
    return ordered


def _resolve_fg_output_columns(fg: FeatureGroup) -> list[str]:
    """Centralized output-column derivation used by the property and ``_to_spec``.

    Walks ``fg.features`` in declaration order, applies the same prefix
    rules as :meth:`FeatureGroup.output_columns`, and drops any per-source
    feature name that resolves to an FG superset PK column. The dedupe is
    case-insensitive via :class:`SqlIdentifier` resolution.

    Args:
        fg: The FeatureGroup whose output columns to resolve.

    Returns:
        Output column names in source-emission order, with FG-PK columns
        removed.
    """
    pk = {SqlIdentifier(k).resolved() for k in _fg_superset_pk(fg)}
    cols: list[str] = []
    for item in fg.features:
        prefix = get_feature_prefix(item, fg.auto_prefix)
        for name in FeatureGroup._feature_column_names(item):
            if SqlIdentifier(name).resolved() in pk:
                continue
            raw = f"{prefix}{name}" if prefix else name
            cols.append(identifier.concat_names([raw]))
    return cols


def _fg_join_key_field_types(fg: FeatureGroup, *, strict: bool = False) -> dict[str, Any]:
    """Resolve ``{join_key -> Snowpark datatype}`` across every source FV.

    Shared between register-time validation and the read-time response
    schema so they cannot disagree. BFV/SFV sources contribute types
    from their ``output_schema``; RTFVs delegate to
    :func:`resolve_realtime_join_key_fields` because their own
    ``output_schema`` is feature-only.

    ``strict=True`` propagates resolver errors instead of skipping the
    source — read-time wants this so registry drift on an RTFV upstream
    surfaces clearly. Register-time leaves it ``False`` so other FG
    validators can produce the user-facing error.

    Args:
        fg: The FeatureGroup being read or registered.
        strict: See note above.

    Returns:
        Canonical join-key name to Snowpark datatype.

    Raises:
        SnowflakeMLException: ``[ValueError]`` when two sources disagree
            on a shared join key's datatype (both sources named).
    """
    from snowflake.ml.feature_store.realtime_registration import (
        resolve_realtime_join_key_fields,
    )

    pk = set(_fg_superset_pk(fg))
    type_by_name: dict[str, Any] = {}
    source_by_name: dict[str, str] = {}
    conflicts: list[str] = []

    for source in fg.features:
        fv = unwrap_fv(source)
        label = f"{fv.name.resolved()}@{fv.version}"
        if fv.realtime_config is not None:
            if strict:
                # Read-time path: surface registry drift on an RTFV upstream
                # (e.g. an upstream FV re-registered with a different
                # join-key datatype) instead of silently dropping the join
                # key from the response schema.
                fields = resolve_realtime_join_key_fields(fv)
            else:
                # Register-time defense-in-depth: the RTFV's own
                # register-time validation would have rejected an
                # inconsistent upstream view, so skipping the source lets
                # FG-level validators still surface a clean ValueError.
                try:
                    fields = resolve_realtime_join_key_fields(fv)
                except snowml_exceptions.SnowflakeMLException:
                    continue
            iterator = ((f.name, f.datatype) for f in fields)
        else:
            iterator = ((f.name, f.datatype) for f in fv.output_schema.fields if f.name in pk)
        for name, datatype in iterator:
            previous = type_by_name.get(name)
            if previous is None:
                type_by_name[name] = datatype
                source_by_name[name] = label
            elif previous != datatype:
                conflicts.append(
                    f"join key {name!r}: source {source_by_name[name]} declares {previous}; "
                    f"source {label} declares {datatype}"
                )

    if conflicts:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "feature group: source feature views disagree on the datatype of a shared join key. "
                "All sources that include the column must declare the same Snowpark datatype. "
                f"Conflicts: {sorted(set(conflicts))}."
            ),
        )

    return type_by_name


def _rtfv_sources(fg: FeatureGroup) -> list[FeatureView]:
    """RTFV source FVs in declaration order, unwrapped (slices dropped).

    Args:
        fg: The FeatureGroup to scan.

    Returns:
        :class:`FeatureView` objects whose ``realtime_config`` is set,
        in source order.
    """
    return [unwrap_fv(s) for s in fg.features if unwrap_fv(s).realtime_config is not None]


def _fg_has_realtime_source(fg: FeatureGroup) -> bool:
    """Whether *fg* has any RTFV source.

    Args:
        fg: The FeatureGroup to inspect.

    Returns:
        ``True`` iff any source FV has a ``realtime_config``.
    """
    return any(unwrap_fv(s).realtime_config is not None for s in fg.features)


def _fg_has_request_source_rtfv(fg: FeatureGroup) -> bool:
    """Whether any RTFV source declares a ``RequestSource``.

    Drives read-time dispatch: when ``True``, the read requires a
    ``request_context``; otherwise it must be omitted. RTFVs without a
    ``RequestSource`` don't move this needle.

    Args:
        fg: The FeatureGroup to inspect.

    Returns:
        ``True`` iff at least one RTFV source has a ``request_source``.
    """
    for source in fg.features:
        rtc = unwrap_fv(source).realtime_config
        if rtc is not None and rtc.request_source is not None:
            return True
    return False


def _resolve_fg_request_context(
    fg: FeatureGroup,
    *,
    request_context: Optional[pd.DataFrame],
    keys: list[list[Any]],
    pandas_mod: Any,
) -> Optional[list[dict[str, Any]]]:
    """Validate ``request_context`` for the FG read and return the per-row payload.

    Required iff at least one RTFV source declares a ``RequestSource``;
    rejected otherwise. The shape/missing/extra/length checks are
    delegated to :func:`canonicalize_request_context` so the FG and
    single-RTFV paths cannot disagree.

    Args:
        fg: The FeatureGroup being read.
        request_context: Caller-supplied DataFrame, or ``None``.
        keys: Entity rows; used for length match.
        pandas_mod: Imported ``pandas`` module.

    Returns:
        ``None`` when no payload is needed, otherwise the canonicalized
        per-row payload as ``list[dict[canonical_name, value]]`` in
        ``keys`` order.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if ``request_context`` is
            supplied when not needed, missing when needed, or fails the
            shared shape/missing/extra/length validation.
    """
    requires_request_context = _fg_has_request_source_rtfv(fg)
    required = _fg_required_request_columns(fg)

    if not requires_request_context:
        if request_context is not None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"feature group {fg.name}/{fg.version}: `request_context` is only "
                    "supported when at least one RealtimeFeatureView source declares a "
                    "RequestSource. Pass request_context=None for this FeatureGroup."
                ),
            )
        return None

    if request_context is None:
        display_required = list(required.values())
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"feature group {fg.name}/{fg.version}: `request_context` is required when "
                "any RealtimeFeatureView source declares a RequestSource. Pass a pandas "
                f"DataFrame with columns {display_required} and one row per entry in `keys`."
            ),
        )

    from snowflake.ml.feature_store.realtime_registration import (
        canonicalize_request_context,
    )

    return canonicalize_request_context(
        request_context=request_context,
        required=required,
        keys=keys,
        error_prefix=f"feature group {fg.name}/{fg.version}",
        pandas_mod=pandas_mod,
    )


def _fg_required_request_columns(fg: FeatureGroup) -> dict[str, str]:
    """Ordered ``canonical_name -> display_name`` over every RTFV source's ``RequestSource``.

    Walks RTFV sources in declaration order, then each source's declared
    schema field order. Same canonical name across sources merges to the
    first source's display; register-time validation already guarantees
    datatype agreement. Order matters so missing-column errors,
    extras-dropped warnings, and the projection are deterministic.

    Args:
        fg: The FeatureGroup being read.

    Returns:
        Canonical-to-display mapping. Empty when no RTFV declares a
        ``RequestSource``.
    """
    required: dict[str, str] = {}
    for fv in _rtfv_sources(fg):
        rtc = fv.realtime_config
        assert rtc is not None  # filtered by _rtfv_sources
        if rtc.request_source is None:
            continue
        for field in rtc.request_source.schema.fields:
            canonical = SqlIdentifier(field.name).resolved()
            if canonical not in required:
                required[canonical] = field.name
    return required


def validate_fg_request_context_contract(fg: FeatureGroup) -> None:
    """Reject cross-RTFV ``RequestSource.schema`` datatype disagreements.

    The shared ``request_context`` payload at read time is keyed by canonical
    column name only, so two RTFV sources that declare the same canonical
    column with conflicting Snowpark datatypes have no coherent client-side
    representation. Same-name + same-datatype is the de-facto shared-column
    case and passes. Matching is case-insensitive via
    :class:`SqlIdentifier` resolution; the user-facing display name is
    preserved from the first declaring source.

    Delegates to :func:`realtime_dataset.validate_rtfvs_request_context_contract`
    so the same logic runs at FG registration and at dataset-time validation
    for the FV-list overload of ``generate_training_set``.

    Args:
        fg: The FeatureGroup whose RTFV sources' RequestSource schemas to
            check.
    """
    # Lazy import: realtime_dataset imports this module for unwrap_fv.
    from snowflake.ml.feature_store import realtime_dataset

    realtime_dataset.validate_rtfvs_request_context_contract(_rtfv_sources(fg))


def validate_fg_request_source_pk_overlap(fg: FeatureGroup) -> None:
    """Reject any RTFV source whose ``RequestSource.schema`` overlaps the FG superset PK.

    Mirrors :func:`validate_rtfv_entity_contract`'s RTFV-level check at the
    FG level: a RequestSource column that collides with the FG's join keys
    would produce a duplicate column in the server-side request dataframe.
    The FG superset PK is wider than any single RTFV's own entity keys, so
    this catch fires even if the per-RTFV check already passed.

    Args:
        fg: The FeatureGroup whose RTFV sources to validate.

    Raises:
        SnowflakeMLException: ``[ValueError]`` if any RTFV source declares
            a RequestSource column that overlaps the FG's superset PK
            (offending source FV and column names both reported).
    """
    canonical_pk = {SqlIdentifier(k).resolved() for k in _fg_superset_pk(fg)}
    offenders: list[str] = []
    for fv in _rtfv_sources(fg):
        rtc = fv.realtime_config
        assert rtc is not None  # filtered by _rtfv_sources
        if rtc.request_source is None:
            continue
        overlapping = [
            f.name for f in rtc.request_source.schema.fields if SqlIdentifier(f.name).resolved() in canonical_pk
        ]
        if overlapping:
            offenders.append(
                f"{fv.name.resolved()}@{fv.version} declares RequestSource columns {sorted(set(overlapping))}"
            )
    if offenders:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "feature group: realtime feature view source(s) declare RequestSource columns that "
                f"overlap the FeatureGroup's superset primary key {sorted(canonical_pk)}. Entity "
                "join keys are supplied at read time via ``keys=[[...]]`` and prepended server-side "
                "to the request payload; declaring them in RequestSource.schema produces a duplicate "
                f"column in the compute_fn's input DataFrame. Offending source(s): {sorted(set(offenders))}."
            ),
        )


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

    # Cross-source consistency checks: surface conflicts as ValueErrors before
    # any side effect. Order matters for the user-facing message: PK datatype
    # disagreement is the broadest failure mode and runs first; per-RTFV
    # checks follow.
    _fg_join_key_field_types(feature_group)
    validate_fg_request_source_pk_overlap(feature_group)
    validate_fg_request_context_contract(feature_group)

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
        session=fs._session,
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
    request_context: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Read FG values via the Online Service Query API. See :meth:`FeatureStore.read_feature_group`.

    Args:
        fs: The :class:`FeatureStore` invoking this operation.
        feature_group: A hydrated :class:`FeatureGroup` or its name.
        version: Required when *feature_group* is a string; otherwise optional
            and validated against the passed FG.
        keys: Non-empty list of entity rows aligned with the FG's join keys.
        store_type: Only :attr:`StoreType.ONLINE` is supported today.
        request_context: Required when at least one RTFV source declares a
            ``RequestSource``; rejected otherwise. Columns are the union of
            every contributing RTFV's ``RequestSource.schema`` field names
            (case-insensitive); extras are dropped with a
            :class:`UserWarning`; row count must match ``len(keys)``.

    Returns:
        ``pandas.DataFrame`` with the join-key columns followed by the FG's
        :attr:`~FeatureGroup.output_columns`.

    Raises:
        SnowflakeMLException: ``[ValueError]`` for empty keys, missing /
            disagreeing version, unregistered FG, or pre-RUNNING Online
            Service; for ``request_context`` shape / contract violations;
            for ``request_context`` missing on an FG whose RTFV source(s)
            declare a ``RequestSource`` or supplied otherwise. ``[NotImplementedError]`` if
            *store_type* is not :attr:`StoreType.ONLINE`.
    """
    # Lazy import: feature_store imports this module at top level.
    import pandas as pandas_mod

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
    join_names = _fg_superset_pk(fg)
    # ``strict=True`` so registry drift on an RTFV upstream raises with
    # the offending source named rather than silently dropping a join
    # key from the response schema.
    join_key_field_types = _fg_join_key_field_types(fg, strict=True)

    request_records = _resolve_fg_request_context(fg, request_context=request_context, keys=keys, pandas_mod=pandas_mod)

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
        request_context=request_records,
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

    if _fg_has_realtime_source(fg):
        logger.info(
            "FeatureGroup %s/%s contains realtime feature view sources; training-set "
            "generation will evaluate compute_fn(s) over the joined upstream rows.",
            fg.name,
            fg.version,
        )

    return list(fg.features), fg.auto_prefix, "cte"
