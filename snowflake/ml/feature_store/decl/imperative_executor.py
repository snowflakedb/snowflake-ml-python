"""Imperative executor for the declarative feature store.

Maps ``PlanOp`` objects to ``FeatureStore`` method calls.  All imports from
``snowflake.ml.feature_store`` are **lazy** (inside functions) so this module
can be included in the standalone ``decl`` wheel without pulling in heavy
transitive dependencies at import time.

This module MUST NOT import from ``snowflake.ml.dataset``,
``snowflake.ml.data``, or any other heavy snowml package.
"""

from __future__ import annotations

import logging
from typing import Any

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.errors import (
    DependencyError,
    FeatureStoreNotInitializedError,
)
from snowflake.ml.feature_store.decl.types import ApplyResult, Plan, PlanOptions
from snowflake.ml.feature_store.spec.enums import ENTITY_TAG_PREFIX

logger = logging.getLogger(__name__)

# ``OpKind.RECREATE_SOURCE`` is owned by W1A (``decl/enums.py``).  We
# discover it via :func:`getattr` so the source-lifecycle dispatch in
# :func:`execute_plan` degrades gracefully on a stale wheel that
# pre-dates the enum extension — without the value the dispatch simply
# never sees a ``RECREATE_SOURCE`` op.
_RECREATE_SOURCE_OP_KIND: Any = getattr(OpKind, "RECREATE_SOURCE", None)


def _resolve_online_target_lag(payload: dict[str, Any]) -> str | None:
    """Pick the OFT ``target_lag`` string from a plan payload.

    The authoring ``target_lag`` / ``target_lag_sec`` field is OFT
    staleness only — strictly the value the imperative
    ``OnlineConfig.target_lag`` receives (which Snowflake then writes
    into ``CREATE ONLINE FEATURE TABLE … TARGET_LAG='<val>'``).  The
    Dynamic Table refresh cadence is a separate authoring knob
    (``refresh_freq``) and is **not** read from here.

    Streaming and realtime feature views always run at 0 seconds target
    lag — the Snowflake runtime stamps ``target_lag_sec: 0`` regardless
    of the authored value, the Pydantic validator
    (``FeatureView._reject_target_lag_on_stream_or_realtime``) rejects
    any authored value on the loader path, and the spec compiler strips
    the key from streaming / realtime compiled specs.  As defence-in-
    depth this helper short-circuits to ``None`` for streaming /
    realtime kinds so a hand-built payload that bypasses the compiler
    strip cannot leak a non-zero value into ``OnlineConfig.target_lag``.
    The caller's ``"0 seconds"`` default fires instead — which is the
    only value Snowflake accepts for streaming / realtime OFTs.

    Resolution order for non-streaming/realtime kinds (first match wins):

    1. ``target_lag`` (raw authoring shape — string or int seconds).
    2. ``target_lag_sec`` (the post-``normalize_durations`` integer shape
       that the loader writes for every YAML).  Formatted as
       ``"<n> seconds"`` when only the integer key is present.

    Returns ``None`` when neither key is set, so the caller can fall
    back to its own default — ``_BATCH_OFT_TARGET_LAG`` ("10 seconds")
    for online ``BatchFeatureView`` per the imperative API, or
    ``"0 seconds"`` for streaming / realtime / FG payloads.

    Args:
        payload: A plan op payload dict (post-``_model_to_dict`` /
            ``normalize_durations``).

    Returns:
        A target-lag string ready for ``OnlineConfig(target_lag=...)``,
        or ``None`` when the payload omits both keys (or carries a
        streaming / realtime kind).
    """
    kind = payload.get("kind", "") if isinstance(payload.get("kind"), str) else ""
    if "Streaming" in kind or "Realtime" in kind:
        return None
    raw = payload.get("target_lag")
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, int):
        return f"{raw} seconds"
    sec = payload.get("target_lag_sec")
    if isinstance(sec, int) and sec > 0:
        return f"{sec} seconds"
    if isinstance(sec, str) and sec.strip().isdigit():
        return f"{int(sec)} seconds"
    return None


def assert_feature_store_initialized(
    session: Any,
    database: str,
    schema: str,
    warehouse: str,
) -> Any:
    """Construct a ``FeatureStore(FAIL_IF_NOT_EXIST)`` or raise a clean
    init-required error pointing operators at ``snow feature init``.

    This is the single enforcement point for the declarative client's
    init-first invariant: every ``snow feature`` command except ``init``
    must fail fast if the target schema lacks the bootstrap tags
    (``SNOWML_FEATURE_STORE_OBJECT`` / ``SNOWML_FEATURE_VIEW_METADATA``).
    The snowml-core ``FeatureStore`` constructor already enforces that
    invariant by raising a ``NOT_FOUND`` ``SnowflakeMLException`` when
    either tag is absent (see ``_check_internal_objects_exist_or_throw``);
    we catch that specific error code and rewrap it as
    :class:`FeatureStoreNotInitializedError` so the CLI can render a
    one-line, actionable error instead of leaking snowml-core internals.

    Args:
        session: Snowpark ``Session`` to bind to the new ``FeatureStore``.
        database: Snowflake database name.
        schema: Snowflake schema name (the ``FeatureStore`` "name").
        warehouse: Default warehouse for the ``FeatureStore``.

    Returns:
        A constructed ``FeatureStore`` bound to ``database.schema`` so
        callers can reuse it without paying for a second
        ``FAIL_IF_NOT_EXIST`` check.

    Raises:
        FeatureStoreNotInitializedError: If the schema is missing the
            internal bootstrap tags or does not exist as a feature
            store.  Triggered either by a ``SnowflakeMLException`` with
            ``error_code == NOT_FOUND`` (direct call-path) or by a
            ``ValueError`` whose ``str()`` starts with the
            ``(<NOT_FOUND>)`` marker (snowml-core's telemetry
            decorator unwraps ``SnowflakeMLException`` to its
            ``original_exception`` before re-raising).
        snowml_exceptions.SnowflakeMLException: Any other
            ``SnowflakeMLException`` raised by ``FeatureStore.__init__``
            (e.g. auth, permissions, warehouse) is propagated verbatim
            — only ``NOT_FOUND`` is rewrapped as the init-required error.
    """
    from snowflake.ml._internal.exceptions import (
        error_codes,
        exceptions as snowml_exceptions,
    )
    from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore

    not_found_prefix = f"({error_codes.NOT_FOUND})"

    try:
        return FeatureStore(
            session,
            database,
            schema,
            warehouse,
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )
    except snowml_exceptions.SnowflakeMLException as exc:
        # Direct (un-telemetry-wrapped) path: ``FeatureStore.__init__``
        # raised the typed snowml-core exception verbatim.  Happens in
        # unit tests and when the imperative codepath is invoked
        # without telemetry context.
        if exc.error_code == error_codes.NOT_FOUND:
            raise FeatureStoreNotInitializedError(database, schema, exc) from exc
        raise
    except ValueError as exc:
        # Telemetry-unwrapped path: in real execution
        # ``snowflake.ml._internal.telemetry`` catches the
        # ``SnowflakeMLException`` raised by
        # ``_check_internal_objects_exist_or_throw`` and re-raises
        # ``e.original_exception`` (a ``ValueError`` whose ``str()``
        # starts with the ``(<error_code>)`` marker).  We match the
        # ``(2101)`` prefix because ``error_code`` is no longer
        # available on the unwrapped exception.
        if str(exc).startswith(not_found_prefix):
            raise FeatureStoreNotInitializedError(database, schema, exc) from exc
        raise


def fetch_entity_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch entity tag rows by delegating to ``FeatureStore.list_entities()``.

    The read-path counterpart to the entity-DDL ops in
    :func:`execute_plan` — both route through the imperative
    ``FeatureStore`` API so the declarative module never emits raw
    entity-tag SQL.  When the target schema lacks the bootstrap
    tags, :func:`assert_feature_store_initialized` rewraps the
    snowml-core ``NOT_FOUND`` as
    :class:`FeatureStoreNotInitializedError` so the CLI surfaces a
    clear "run ``snow feature init``" message.

    The imperative ``list_entities()`` returns a Snowpark DataFrame whose
    rows have keys ``NAME`` (with the ``SNOWML_FEATURE_STORE_ENTITY_``
    prefix already stripped), ``JOIN_KEYS`` (a JSON-decodable string),
    ``DESC``, and ``OWNER``.  The legacy CLI applied-state parser
    (:func:`state._build_entity_object`) and the list-renderer
    (:func:`api.enrich_list_results`) both expect the raw ``SHOW TAGS``
    shape (``name`` *with* the prefix, ``database_name``, ``schema_name``,
    ``allowed_values``, ``comment``).  This function translates between
    those two shapes so existing consumers keep working unchanged.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name to scope the listing to.
        schema: Snowflake schema name (the FeatureStore "name") to scope
            the listing to.
        warehouse: Default warehouse for the imperative ``FeatureStore``
            constructor.  ``list_entities()`` itself only issues a
            ``SHOW TAGS`` so the warehouse is not actually used; an
            empty string is accepted, but a real value is recommended
            when the caller has one available so future imperative
            calls work without re-priming.

    Returns:
        A list of row dicts in the SHOW TAGS shape.  Returns an empty
        list when no entities are registered.
    """
    fs = assert_feature_store_initialized(session, database, schema, warehouse)
    listed = fs.list_entities().collect()

    translated: list[dict[str, Any]] = []
    for row in listed:
        if hasattr(row, "as_dict"):
            data = row.as_dict()
        elif isinstance(row, dict):
            data = dict(row)
        else:
            try:
                data = dict(row)
            except (TypeError, ValueError):
                logger.debug("Skipping un-mappable entity row: %r", row)
                continue

        # ``FeatureStore.list_entities()`` returns rows with the
        # ``SNOWML_FEATURE_STORE_ENTITY_`` prefix already stripped, plus
        # ``JOIN_KEYS`` (JSON string), ``DESC``, ``OWNER``.  Re-add the
        # prefix and copy the JSON join_keys / desc into the SHOW TAGS
        # column names that downstream consumers
        # (``state._build_entity_object``, ``api.enrich_list_results``)
        # expect.
        raw_name = data.get("NAME") or data.get("name") or ""
        if not raw_name:
            continue
        tag_name = f"{ENTITY_TAG_PREFIX}{raw_name}"
        allowed = data.get("JOIN_KEYS") or data.get("join_keys") or ""
        comment = data.get("DESC") or data.get("desc") or ""
        owner = data.get("OWNER") or data.get("owner") or ""

        translated.append(
            {
                "name": tag_name,
                "database_name": database,
                "schema_name": schema,
                "allowed_values": allowed,
                "comment": comment,
                "owner": owner,
            }
        )

    return translated


# Suffix appended to offline Dynamic Table backing every FV.  The
# imperative ``FeatureView._get_physical_name`` joins ``<NAME>`` and
# ``<VERSION>`` with this delimiter; the offline DT carries no
# ``$ONLINE`` suffix (that suffix is reserved for the OFT).
_OFFLINE_DT_DELIMITER = "$"


def _parse_online_enabled(raw: Any) -> bool:
    """Decide whether an FV row's ``online_config`` indicates the OFT
    is enabled.

    ``FeatureStore.list_feature_views`` emits ``online_config`` as a
    JSON string that is either ``OnlineConfig.to_json()`` (when the
    OFT exists — ``enable=True``) or a default config with
    ``enable=False`` when the OFT is absent.  This helper handles
    both the JSON-string shape and the unlikely raw-dict shape; any
    other value collapses to ``False`` so the merge logic in
    :func:`state.fetch_applied_state` defaults to "no OFT, this is
    an offline-only FV" — the safe direction (the OFT path will
    overwrite the offline path when both are present, never the
    other way around).

    Args:
        raw: The ``online_config`` cell value from a
            ``FeatureStore.list_feature_views`` row.

    Returns:
        ``True`` when ``online_config.enable`` is truthy; ``False`` for
        absent / falsey / unparsable values.
    """
    import json

    if isinstance(raw, dict):
        return bool(raw.get("enable"))
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return False
        if isinstance(decoded, dict):
            return bool(decoded.get("enable"))
    return False


def _parse_entities_cell(raw: Any) -> list[str]:
    """Coerce the ``entities`` column of a list-FV row to a ``list[str]``.

    Snowpark surfaces ARRAY columns as Python lists when the cell is
    materialised through ``Row.as_dict()``.  When the row crosses a
    cursor boundary that does not understand ARRAY types, the cell
    arrives as a JSON-string.  Both shapes carry the same data; this
    helper normalises to a Python list so callers can iterate without
    branching.

    Args:
        raw: The ``entities`` cell value from a
            ``FeatureStore.list_feature_views`` row.

    Returns:
        A list of entity-name strings.  Returns ``[]`` for ``None``,
        non-string / non-list values, or unparsable JSON.
    """
    import json

    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if isinstance(decoded, list):
            return [str(x) for x in decoded if x is not None]
    return []


def fetch_feature_view_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch feature-view rows by delegating to ``FeatureStore.list_feature_views()``.

    Mirror of :func:`fetch_entity_rows` for FeatureView enumeration.
    The imperative ``FeatureStore.list_feature_views`` walks the
    ``SHOW DYNAMIC TABLES`` / ``SHOW VIEWS`` backends and returns one
    row per FV — including offline-only ``BatchFeatureView``s that
    have no Online Feature Table.  Without this path the declarative
    ``fetch_applied_state`` cannot see offline-only BFVs and the
    planner re-emits ``CREATE_FV`` on every plan after a successful
    ``apply`` (see ``plans/offline_bfv_state_fix_b9da0006.plan.md``).

    The returned rows match the Phase-1 contract documented in
    Section 7 of that plan: a narrow dict that
    :func:`state.fetch_applied_state` translates into an
    ``AppliedObject`` for offline-only FVs.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name.
        schema: Snowflake schema name (the FeatureStore "name").
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of FV row dicts.  Each row carries:
        ``name``, ``version``, ``database_name``, ``schema_name``,
        ``kind`` (``BATCH`` / ``STREAMING`` / ``REALTIME``),
        ``entities`` (parsed ``list[str]``),
        ``online_enabled`` (bool), ``target_lag``,
        ``refresh_freq``, ``warehouse``, ``desc``, and
        ``physical_dt_name`` (``<NAME>$<VERSION>`` for
        ``dt_text_map`` lookup).  Returns ``[]`` when no FV is
        registered.  Init-first symmetry with
        :func:`fetch_entity_rows` — :func:`assert_feature_store_initialized`
        bubbles up :class:`FeatureStoreNotInitializedError` when the
        target schema lacks the bootstrap feature-store tags.
    """
    fs = assert_feature_store_initialized(session, database, schema, warehouse)
    listed = fs.list_feature_views().collect()

    translated: list[dict[str, Any]] = []
    for row in listed:
        if hasattr(row, "as_dict"):
            data = row.as_dict()
        elif isinstance(row, dict):
            data = dict(row)
        else:
            try:
                data = dict(row)
            except (TypeError, ValueError):
                logger.debug("Skipping un-mappable feature-view row: %r", row)
                continue

        name = _row_get(data, "name") or ""
        version = _row_get(data, "version") or ""
        if not name or not version:
            logger.debug("Skipping list-FV row with missing name/version: %r", data)
            continue

        kind_raw = _row_get(data, "kind") or ""
        kind = str(kind_raw).upper() if kind_raw else ""
        physical_dt = f"{str(name)}{_OFFLINE_DT_DELIMITER}{str(version)}"

        row_dict: dict[str, Any] = {
            "name": str(name),
            "version": str(version),
            "database_name": _row_get(data, "database_name") or database,
            "schema_name": _row_get(data, "schema_name") or schema,
            "kind": kind,
            "entities": _parse_entities_cell(_row_get(data, "entities")),
            "online_enabled": _parse_online_enabled(_row_get(data, "online_config")),
            "target_lag": _row_get(data, "target_lag") or "",
            "refresh_freq": _row_get(data, "refresh_freq") or "",
            "warehouse": _row_get(data, "warehouse") or "",
            "cluster_by": _row_get(data, "cluster_by") or "",
            "refresh_mode": _row_get(data, "refresh_mode") or "",
            "desc": _row_get(data, "desc") or "",
            "physical_dt_name": physical_dt,
            # ``source_refs`` is a JSON-encoded list[dict] populated by
            # the ``_LIST_FEATURE_VIEW_SCHEMA.source_refs`` projection
            # (plan section A1).  ``None`` / missing for legacy FVs that
            # pre-date the metadata row; ``decl/state.fetch_applied_state``
            # then emits a once-per-FV warning and falls back to the
            # legacy ``_build_datasources_by_table`` name-lookup shim.
            "source_refs": _decode_json_cell(_row_get(data, "source_refs")),
        }

        # Enrich BatchFV rows with the full ``FeatureViewSpec`` dict so the
        # applied-state recovery can populate ``features``, ``timestamp_field``,
        # ``feature_granularity_sec``, ``aggregation_secondary_keys``,
        # ``cluster_by``, ``initialize``, and ``refresh_mode`` for FVs whose
        # online presence is absent (``online: false``) and therefore cannot
        # be recovered via ``DESCRIBE … TYPE = SPECIFICATION``.  Failures
        # fall through silently: the legacy minimal-payload reconstruction
        # remains the fallback so this path is purely additive.
        if kind == "BATCH":
            spec_dict = _serialize_batch_fv_spec(fs, session, row_dict)
            if spec_dict is not None:
                row_dict["spec_text"] = spec_dict

        translated.append(row_dict)

    return translated


def fetch_feature_view_object(
    session: Any,
    database: str,
    schema: str,
    warehouse: str,
    name: str,
    version: str,
) -> Any:
    """Return a fully-rehydrated ``FeatureView`` object for ``(name, version)``.

    Lazy-imports :class:`FeatureStore` and delegates to
    :meth:`FeatureStore.get_feature_view`, surfacing the same
    ``FeatureStoreNotInitializedError`` rewrap on missing bootstrap
    tags so callers get a uniform init-first error story.

    Used by the applied-state recovery in :mod:`decl.state` to read
    imperative-only fields that the list-FV row does not carry
    (``initialize``, ``aggregation_specs``, ``aggregation_secondary_keys``,
    ``feature_granularity``, ``timestamp_col``, ``source_refs``) without
    re-parsing the offline DT's CREATE statement.  Plan section B2 calls
    this helper once per tiled BatchFV during the OFT enrichment pass.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name.
        schema: Snowflake schema name (the FeatureStore "name").
        warehouse: Default warehouse forwarded to the ``FeatureStore``
            constructor.
        name: FeatureView name (case-canonical).
        version: FeatureView version string.

    Returns:
        FeatureView object — the same instance
        :meth:`FeatureStore.get_feature_view` would have returned, with
        every metadata-restored attribute populated (``source_refs``,
        ``aggregation_secondary_keys``, ``feature_granularity``, …).
        Errors from :func:`assert_feature_store_initialized` (which
        rewraps the missing-bootstrap-tags case as
        ``FeatureStoreNotInitializedError``) and any
        ``snowml_exceptions.SnowflakeMLException`` raised by
        :meth:`FeatureStore.get_feature_view` (FV not found, auth, etc.)
        propagate to the caller unchanged.
    """
    fs = assert_feature_store_initialized(session, database, schema, warehouse)
    return fs.get_feature_view(name, version)


def _serialize_batch_fv_spec(
    fs: Any,
    session: Any,
    row_dict: dict[str, Any],
) -> dict[str, Any] | None:
    """Serialize a registered BatchFV into a SPECIFICATION-shaped dict.

    Calls the imperative ``FeatureStore.get_feature_view()`` to compose a
    fully-populated :class:`FeatureView`, then runs the same internal
    ``_build_batch_feature_view_spec`` the OFT ``CREATE … FROM
    SPECIFICATION`` path uses so the dict shape matches what
    :func:`state.fetch_applied_state` would have parsed out of a live
    ``DESCRIBE … TYPE = SPECIFICATION`` payload.

    Returns ``None`` on any failure so the caller can fall back to the
    legacy minimal reconstruction in
    :func:`state._build_offline_fv_object`.

    Args:
        fs: A live ``FeatureStore`` instance for the target schema.
        session: A live ``snowflake.snowpark.Session``; used to resolve
            the offline Dynamic Table's materialised schema for tiled
            BFVs (the ``_build_batch_feature_view_spec`` contract requires
            ``offline_materialized_schema`` when ``is_tiled``).
        row_dict: The translated row dict (must carry ``name``,
            ``version``, ``database_name``, ``schema_name``,
            ``physical_dt_name``, and any cadence fields).

    Returns:
        A dict matching :meth:`FeatureViewSpec.to_dict` output, or
        ``None`` when reconstruction failed.
    """
    try:
        fv = fs.get_feature_view(row_dict["name"], row_dict["version"])
    except Exception as e:  # noqa: BLE001 — defensive: keep legacy fallback
        logger.warning(
            "feature view state: could not load feature view %s (version %s) from the feature store; "
            "falling back to minimal state, which can cause repeated feature-view updates. Cause: %s",
            row_dict.get("name"),
            row_dict.get("version"),
            e,
        )
        return None

    target_lag = row_dict.get("refresh_freq") or row_dict.get("target_lag") or "0 seconds"

    offline_materialized_schema = None
    try:
        if getattr(fv, "is_tiled", False):
            db = row_dict.get("database_name", "")
            sch = row_dict.get("schema_name", "")
            physical_dt = row_dict.get("physical_dt_name", "")
            fq = f"{db}.{sch}.{physical_dt}"
            offline_materialized_schema = session.table(fq).schema
    except Exception as e:  # noqa: BLE001
        # Fall through — spec reconstruction will fail and we return None.
        logger.warning(
            "feature view state: could not read the offline table schema for feature view %s (version %s); "
            "falling back to minimal state, which can cause repeated feature-view updates. Cause: %s",
            row_dict.get("name"),
            row_dict.get("version"),
            e,
        )

    # snowml's ``FeatureStore.get_feature_view()`` reconstructs ``fv.feature_df``
    # from the materialized DT (post-aggregation: ``USER_ID, …, TILE_START,
    # _PARTIAL_SUM_AMOUNT``) for tiled BFVs.  ``_build_batch_feature_view_spec``
    # then validates ``aggregation_specs`` against this schema — failing with
    # ``Column 'AMOUNT' not found in resolution pool`` because the raw source
    # column ``AMOUNT`` no longer exists.  Recover the raw source schema by
    # reading ``fv.source_refs`` (populated authoritatively from
    # ``FV_SOURCE_REFS`` metadata — see plan section A1) and substitute a
    # synthetic empty DataFrame carrying that schema onto ``fv._feature_df``
    # so the spec builder's validation pool matches the aggregation source
    # columns.  Replaces the legacy ``SHOW DYNAMIC TABLES`` + regex parse of
    # the DT body (Phase B1 — no SQL/DT-text parsing in ``decl/``).
    if getattr(fv, "is_tiled", False) and not getattr(fv, "is_rollup", False):
        try:
            source_refs = getattr(fv, "source_refs", None) or []
            raw_source = ""
            src_db = ""
            src_schema = ""
            if source_refs and isinstance(source_refs[0], dict):
                raw_source = str(source_refs[0].get("table") or "")
                src_db = str(source_refs[0].get("source_database") or "") or row_dict.get("database_name", "")
                src_schema = str(source_refs[0].get("source_schema") or "") or row_dict.get("schema_name", "")
            if raw_source:
                if "." in raw_source:
                    fq = raw_source
                else:
                    fq = f"{src_db}.{src_schema}.{raw_source}"
                raw_schema = session.table(fq).schema
                empty_df = session.create_dataframe([], schema=raw_schema)
                fv._feature_df = empty_df
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "feature view state: could not read the source table schema for feature view %s (version %s); "
                "falling back to minimal state, which can cause repeated feature-view updates. Cause: %s",
                row_dict.get("name"),
                row_dict.get("version"),
                e,
            )

    try:
        from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier as _SqlId

        physical_dt_name = _SqlId(row_dict.get("physical_dt_name", ""))
        spec = fs._build_batch_feature_view_spec(
            feature_view=fv,
            feature_view_name=physical_dt_name,
            version=row_dict["version"],
            target_lag=target_lag,
            offline_materialized_schema=offline_materialized_schema,
        )
        result: dict[str, Any] = spec.to_dict()
        _emulate_quake_postprocess(result)
        _canonicalize_enums_in_place(result)
        # snowml's spec builder stamps the default ``online_store_type``
        # onto every BatchFV result even when the authoring YAML has
        # ``online: false`` (no OFT will be created).  The local-compile
        # side omits the key entirely for offline-only FVs, so leaving
        # the stamped value in would trigger a spurious ``UPDATE_FV``
        # via :func:`planner._batch_fv_operational_drift`.  Strip it
        # whenever the row reports no online deployment.
        if not row_dict.get("online_enabled"):
            result.pop("online_store_type", None)
        # The applied-state recovery has no in-spec carrier for the
        # refresh ``warehouse`` (DESCRIBE … TYPE = SPECIFICATION omits
        # it and ``_inject_advanced_bfv_fields_from_dt_text`` does not
        # yet parse it from DT text).  ``planner._warehouse_drifted``
        # treats an authored-but-not-recovered warehouse as drift, so
        # an offline-only BFV that authors ``warehouse:`` always
        # flagged ``UPDATE_FV`` after a clean apply.  ``list_feature_views``
        # already surfaces the deployed warehouse on the FV row; inject
        # it into ``spec.warehouse`` so the planner's drift comparison
        # finds a match on a clean round-trip.
        warehouse = row_dict.get("warehouse") or ""
        inner = result.get("spec") if isinstance(result.get("spec"), dict) else None
        if warehouse and isinstance(inner, dict) and "warehouse" not in inner:
            inner["warehouse"] = warehouse
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "feature view state: could not reconstruct the specification for feature view %s (version %s); "
            "falling back to minimal state, which can cause repeated feature-view updates. Cause: %s",
            row_dict.get("name"),
            row_dict.get("version"),
            e,
        )
        return None


def _canonicalize_enums_in_place(obj: Any) -> None:
    """Recursively replace any :class:`enum.Enum` value with its ``.value``.

    ``FeatureViewSpec.to_dict()`` returns a dict whose ``kind`` and
    ``feature_aggregation_method`` carry the source ``FeatureViewKind`` /
    ``FeatureAggregationMethod`` enum instances rather than their canonical
    string values — the live ``DESCRIBE … TYPE = SPECIFICATION`` JSON
    path always returns strings (Snowflake serialises the spec server-side
    before sending it over the wire), so the divergence is invisible to
    the online-OFT recovery codepath and only surfaces when the
    offline-only enrichment in :func:`_serialize_batch_fv_spec` short-circuits
    through ``spec.to_dict()`` directly.

    Without canonicalisation, the exporter writes the enum-bearing payload
    through PyYAML's safe-load contract and emits the
    ``!!python/object/apply:snowflake.ml.feature_store.spec.enums.<Enum>``
    tag.  The next ``snow feature plan`` then fails the YAML safe-load with
    ``could not determine a constructor for the tag …`` because the
    loader uses ``yaml.safe_load``, which (correctly) rejects Python
    object reconstruction tags.

    Mutates ``obj`` in place: walks nested dicts and lists, replacing every
    ``Enum`` leaf with its ``.value``.  Non-enum leaves are left untouched.

    Args:
        obj: Any value.  Dicts and lists are recursed; ``Enum`` instances
            are replaced with their ``.value`` in the parent container;
            other types are no-ops.
    """
    from enum import Enum

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, Enum):
                obj[key] = value.value
            else:
                _canonicalize_enums_in_place(value)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            if isinstance(value, Enum):
                obj[idx] = value.value
            else:
                _canonicalize_enums_in_place(value)


def _emulate_quake_postprocess(result: dict[str, Any]) -> None:
    """Mirror Snowflake-server's Quake transforms on a client-side spec dict.

    ``FeatureStore._build_batch_feature_view_spec`` returns snowml's
    *pre-deploy* spec shape: secondary keys are folded into
    ``ordered_entity_column_names`` and surfaced via
    ``ordered_secondary_key_column_names``, the aggregation builder
    auto-emits ``<SK>_KEYS_<window>s`` array features per secondary key,
    and tiled output columns are wrapped in ``ArrayType`` with an inner
    ``element_type``.  When the OFT is created via ``CREATE … FROM
    SPECIFICATION``, the Snowflake server (Quake) strips the secondary
    keys back out, drops the SK array features, and unwraps tiled
    ``ArrayType`` output columns to their base element type before
    persisting.  The applied-side ``DESCRIBE … TYPE = SPECIFICATION``
    JSON returns this post-Quake shape — which is what the local-compile
    side already produces from the YAML.  For offline-only tiled BFVs
    (no OFT), we have to apply the same transform client-side so the
    hash round-trip matches.

    Mutates ``result`` in place.  Operates only on
    ``kind == "BatchFeatureView"`` payloads; other kinds are a no-op so
    the helper is safe to invoke unconditionally.

    Args:
        result: A ``FeatureViewSpec.to_dict()`` payload (``{kind,
            metadata, offline_configs, online_store_type, spec}``).
    """
    if not isinstance(result, dict) or result.get("kind") != "BatchFeatureView":
        return
    inner = result.get("spec")
    if not isinstance(inner, dict):
        return

    sks_raw = inner.pop("ordered_secondary_key_column_names", None)
    sks: list[str] = []
    if isinstance(sks_raw, list):
        sks = [str(k) for k in sks_raw if k]
    if sks:
        inner["aggregation_secondary_keys"] = sks
        entities = inner.get("ordered_entity_column_names")
        if isinstance(entities, list):
            sks_uc = {s.upper() for s in sks}
            inner["ordered_entity_column_names"] = [e for e in entities if str(e).upper() not in sks_uc]

    features = inner.get("features")
    if isinstance(features, list):
        sks_uc = {s.upper() for s in sks}
        cleaned: list[Any] = []
        for feat in features:
            if not isinstance(feat, dict):
                cleaned.append(feat)
                continue
            src = feat.get("source_column")
            src_name = str(src.get("name", "")).upper() if isinstance(src, dict) else ""
            if sks_uc and src_name in sks_uc and "function" not in feat:
                continue
            oc = feat.get("output_column")
            if isinstance(oc, dict) and oc.get("type") == "ArrayType":
                elem = oc.get("element_type")
                if isinstance(elem, str) and elem:
                    oc["type"] = elem
                    oc.pop("element_type", None)
            cleaned.append(feat)
        inner["features"] = cleaned


def _decode_json_cell(raw: Any) -> Any:
    """Coerce a JSON-string cell to a Python object.

    The metadata store persists ``SOURCES`` and ``OUTPUT_COLUMNS`` as JSON
    variants.  Snowpark surfaces those columns as JSON strings via
    ``Row.as_dict()``; the cursor path may surface them already-decoded
    (a Python list / dict).  This helper accepts either.

    Args:
        raw: The cell value from a ``FeatureStore.list_feature_groups()`` row.

    Returns:
        The JSON-decoded Python object (typically ``list``); the original
        value when it was already a list / dict; ``None`` when the cell
        is missing, empty, or unparsable.
    """
    import json

    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return None
    return None


def fetch_feature_group_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch FeatureGroup rows by delegating to ``FeatureStore.list_feature_groups()``.

    Mirror of :func:`fetch_feature_view_rows` for FeatureGroup enumeration.
    The imperative ``FeatureStore.list_feature_groups`` reads from
    :class:`FeatureGroupMetadata` rows and returns one row per registered
    ``(name, version)``.  This function lazy-imports
    ``snowflake.ml.feature_store.feature_store.FeatureStore`` and
    constructs it via :func:`assert_feature_store_initialized` so an
    uninitialised schema raises :class:`FeatureStoreNotInitializedError`
    (matching the entity / FV reads), letting the CLI emit the actionable
    "run ``snow feature init``" message.

    The returned rows decode the JSON-string ``SOURCES`` / ``OUTPUT_COLUMNS``
    columns into native Python objects so downstream consumers
    (``state.fetch_applied_state``, ``exporter.export_specs``) can iterate
    without re-parsing.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name.
        schema: Snowflake schema name (the FeatureStore "name").
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of FG row dicts in a narrow shape with keys ``name``,
        ``version``, ``desc``, ``owner``, ``auto_prefix``, ``sources``
        (list of ``{fv_name, fv_version, slice_columns?, alias?}``),
        ``output_columns`` (``list[str] | None`` — ``None`` for legacy
        rows written before the column existed), ``database_name``,
        and ``schema_name``.  Returns ``[]`` when no FG is registered.

    Raises:
        ValueError: If any row returned by
            ``FeatureStore.list_feature_groups()`` maps to an empty
            ``name`` or ``version``.  Silently skipping such a row
            would mask upstream metadata drift (Snowpark column rename
            on ``_LIST_FEATURE_GROUP_SCHEMA``, corrupt metadata, or a
            legacy FG written without a version) and surface as a
            spurious ``CREATE_FG`` on the next ``snow feature plan``.
    """
    fs = assert_feature_store_initialized(session, database, schema, warehouse)
    listed = fs.list_feature_groups().collect()

    translated: list[dict[str, Any]] = []
    for row in listed:
        if hasattr(row, "as_dict"):
            data = row.as_dict()
        elif isinstance(row, dict):
            data = dict(row)
        else:
            try:
                data = dict(row)
            except (TypeError, ValueError):
                logger.debug("Skipping un-mappable feature-group row: %r", row)
                continue

        name = _row_get(data, "name") or ""
        version = _row_get(data, "version") or ""
        if not name or not version:
            # Strict-fail (was silent-skip): a row mapping to empty
            # name/version signals either Snowpark column-rename drift
            # on _LIST_FEATURE_GROUP_SCHEMA, a corrupt metadata entry,
            # or a legacy FG written without a version.  Skipping it
            # masks the root cause — the next snow feature plan would
            # diff against an empty applied state and surface a
            # spurious CREATE_FG instead of pointing at the broken
            # row.  Raising here puts the offending row shape in the
            # operator's hands so they can fix the source.
            raise ValueError(
                f"FeatureStore.list_feature_groups() returned a row with "
                f"missing name/version: name={name!r}, version={version!r}, "
                f"row={data!r} — refusing to silently drop it"
            )

        sources_decoded = _decode_json_cell(_row_get(data, "sources"))
        if not isinstance(sources_decoded, list):
            sources_decoded = []
        output_cols_decoded = _decode_json_cell(_row_get(data, "output_columns"))
        if output_cols_decoded is not None and not isinstance(output_cols_decoded, list):
            output_cols_decoded = None

        translated.append(
            {
                "name": str(name),
                "version": str(version),
                "desc": _row_get(data, "desc") or "",
                "owner": _row_get(data, "owner") or "",
                "auto_prefix": bool(_row_get(data, "auto_prefix")),
                "sources": sources_decoded,
                "output_columns": output_cols_decoded,
                "database_name": _row_get(data, "database_name") or database,
                "schema_name": _row_get(data, "schema_name") or schema,
            }
        )

    return translated


def _row_get(data: dict[str, Any], key: str) -> Any:
    """Return ``data[key]`` accepting either lower- or upper-cased keys.

    ``list_feature_views`` lowercases column names per the
    ``_LIST_FEATURE_VIEW_SCHEMA`` definition; cursor paths may surface
    them upper-cased.  Both shapes carry the same data; callers never
    see the divergence.

    Args:
        data: A row dict from ``Row.as_dict()`` or ``DictCursor``.
        key: Lower-case column name to read.

    Returns:
        The matching column value, or ``None`` when neither casing is
        present.
    """
    if key in data:
        return data[key]
    up = key.upper()
    if up in data:
        return data[up]
    return None


def fetch_stream_source_rows(
    session: Any,
    database: str,
    schema: str,
    warehouse: str = "",
) -> list[dict[str, Any]]:
    """Fetch stream-source rows by delegating to ``FeatureStore.list_stream_sources()``.

    Mirror of :func:`fetch_entity_rows` / :func:`fetch_feature_view_rows` /
    :func:`fetch_feature_group_rows` for the streaming-source read path.
    The imperative ``FeatureStore.list_stream_sources()`` returns a
    Snowpark DataFrame whose rows carry ``NAME``, ``SCHEMA`` (JSON
    string), ``DESC``, and ``OWNER`` columns (see
    ``feature_store.py:list_stream_sources``).  Decoding the ``SCHEMA``
    JSON into a ``list[{"name", "type"}]`` here keeps downstream
    consumers (:mod:`state` and the planner's source-diff helper) free
    of JSON-parsing boilerplate.

    Init-first symmetry: :func:`assert_feature_store_initialized`
    rewraps the snowml-core ``NOT_FOUND`` as
    :class:`FeatureStoreNotInitializedError` so the CLI surfaces a clear
    "run ``snow feature init``" message when the target schema lacks
    the bootstrap tags.  Malformed ``SCHEMA`` JSON degrades that row's
    ``schema`` to ``[]`` (logged at ``debug`` level) so the planner can
    fall through to a ``recreate`` decision rather than crashing on
    metadata drift.

    See ``plans/stream_source_contract.md`` §3 for the row-shape
    contract this function pins.

    Args:
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name.
        schema: Snowflake schema name (the FeatureStore "name").
        warehouse: Default warehouse forwarded to the imperative
            ``FeatureStore`` constructor; an empty string is accepted.

    Returns:
        A list of row dicts in the contract §3 shape::

            {
                "name": str,
                "schema": list[{"name": str, "type": str}],
                "desc": str,
                "owner": str,
            }

        Returns ``[]`` when no stream source is registered.
    """
    import json

    fs = assert_feature_store_initialized(session, database, schema, warehouse)
    listed = fs.list_stream_sources().collect()

    translated: list[dict[str, Any]] = []
    for row in listed:
        if hasattr(row, "as_dict"):
            data = row.as_dict()
        elif isinstance(row, dict):
            data = dict(row)
        else:
            try:
                data = dict(row)
            except (TypeError, ValueError):
                logger.debug("Skipping un-mappable stream-source row: %r", row)
                continue

        name = data.get("NAME") or data.get("name") or ""
        if not name:
            logger.debug("Skipping stream-source row with missing NAME: %r", data)
            continue

        raw_schema = data.get("SCHEMA") or data.get("schema") or ""
        parsed_schema: list[dict[str, Any]] = []
        if isinstance(raw_schema, list):
            parsed_schema = [c for c in raw_schema if isinstance(c, dict)]
        elif isinstance(raw_schema, str) and raw_schema.strip():
            try:
                decoded = json.loads(raw_schema)
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "Stream source %s has malformed SCHEMA JSON; degrading to []: %s",
                    name,
                    exc,
                )
                decoded = None
            if isinstance(decoded, list):
                parsed_schema = [c for c in decoded if isinstance(c, dict)]

        translated.append(
            {
                "name": str(name),
                "schema": parsed_schema,
                "desc": data.get("DESC") or data.get("desc") or "",
                "owner": data.get("OWNER") or data.get("owner") or "",
            }
        )

    return translated


def execute_plan(
    plan: Plan,
    session: Any,
    database: str,
    schema: str,
    warehouse: str,
    options: PlanOptions,
) -> ApplyResult:
    """Execute a Plan by delegating each PlanOp to FeatureStore methods.

    All ``snowflake.ml.feature_store`` imports are lazy — they happen here,
    not at module level.  After the upstream lazy-import fixes (dataset and
    circular feature_store imports made lazy), importing these classes does
    NOT pull in numpy/pandas/pyarrow.

    Args:
        plan: The execution plan from ``generate_plan``.
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name.
        schema: Snowflake schema name (FeatureStore "name").
        warehouse: Snowflake warehouse name.
        options: Plan options (overwrite, allow_recreate, etc.).

    Returns:
        ApplyResult with per-operation status.

    Raises:
        Exception: If any plan operation fails during execution.
    """
    # Destructive-op guard (apply-time policy enforcement).
    #
    # The planner is policy-free: it always emits ``RECREATE_FV`` /
    # ``DROP_FV`` / ``DROP_ENTITY`` for destructive diffs so operators
    # can see them in the ``snow feature plan`` output before deciding
    # whether to apply.  The ``allow_recreate`` flag is the operator
    # opt-in that gates *apply*, not plan generation — refusing here
    # (before any FV side-effect or even ``FeatureStore`` construction)
    # leaves the plan file unrenamed under L5 (Mark-Failed-Stays-
    # Unapplied) so a follow-up ``snow feature apply --allow-recreate``
    # consumes the same plan.  See BUG_BASH §14 and
    # plans/apply_allow_recreate_destructive_gate_*.plan.md.
    if not options.allow_recreate:
        destructive_ops = [op for op in plan.ops if op.destructive]
        if destructive_ops:
            refused_rows = [
                {
                    "operation": op.kind.value,
                    "name": op.name,
                    "reason": op.reason,
                    "destructive": True,
                    "status": "refused",
                }
                for op in destructive_ops
            ]
            skipped_rows = [
                {
                    "operation": op.kind.value,
                    "name": op.name,
                    "reason": op.reason,
                    "destructive": False,
                    "status": "skipped",
                }
                for op in plan.ops
                if not op.destructive
            ]
            return ApplyResult(
                status="refused",
                ops=refused_rows + skipped_rows,
                warnings=list(plan.warnings),
                errors=[
                    f"Apply refused: {len(destructive_ops)} destructive "
                    f"operation(s) require --allow-recreate. "
                    f"Re-run with `snow feature apply --allow-recreate <path>`."
                ],
            )

    # Eager ``FeatureStore`` construction.  Every entity + FV op now
    # routes through the imperative API (no raw entity-tag SQL
    # remains in this module), so the init-first invariant must fire
    # uniformly across entity-only, FV-only, and mixed plans:
    # ``assert_feature_store_initialized`` rewraps the snowml-core
    # ``NOT_FOUND`` as :class:`FeatureStoreNotInitializedError` so
    # the CLI's outer handler can surface the
    # "run ``snow feature init``" guidance.
    fs = assert_feature_store_initialized(session, database, schema, warehouse)

    ops: list[dict[str, Any]] = []
    errors: list[str] = []

    for op in plan.ops:
        if op.kind == OpKind.NO_CHANGE:
            ops.append(
                {
                    "operation": op.kind.value,
                    "name": op.name,
                    "reason": op.reason,
                    "destructive": op.destructive,
                    "status": "skipped",
                }
            )
            continue

        # Source-side ops dispatch table (contract §4 of
        # ``plans/stream_source_contract.md``).  ``StreamingSource`` ops
        # route through the imperative ``FeatureStore`` stream-source
        # API; ``BatchSource`` ops are informational no-ops because
        # ``BatchSource`` is virtual at the runtime level (its identity
        # lives inside the consuming FV's offline Dynamic Table).  Each
        # ``_execute_*_stream_source`` helper handles the
        # ``payload['kind']`` branching internally so this dispatcher
        # stays flat; the legacy ``CREATE_SOURCE`` branch is kept
        # explicit because it pre-dates the StreamingSource read path
        # and its informational no-op for non-StreamingSource kinds
        # historically reported ``status: skipped`` (preserve that
        # back-compat shape for ``CREATE_SOURCE`` only).
        if op.kind == OpKind.CREATE_SOURCE:
            if op.payload.get("kind") == "StreamingSource":
                try:
                    _execute_create_stream_source(fs, op.payload)
                    ops.append(
                        {
                            "operation": op.kind.value,
                            "name": op.name,
                            "reason": op.reason,
                            "destructive": op.destructive,
                            "status": "success",
                        }
                    )
                except Exception as e:
                    logger.error("Failed to execute %s for %s: %s", op.kind.value, op.name, e)
                    ops.append(
                        {
                            "operation": op.kind.value,
                            "name": op.name,
                            "reason": op.reason,
                            "destructive": op.destructive,
                            "status": "error",
                            "error": str(e),
                        }
                    )
                    errors.append(f"{op.kind.value} {op.name}: {e}")
                    raise
            else:
                ops.append(
                    {
                        "operation": op.kind.value,
                        "name": op.name,
                        "reason": op.reason,
                        "destructive": op.destructive,
                        "status": "skipped",
                    }
                )
            continue

        if (
            op.kind == OpKind.UPDATE_SOURCE
            or op.kind == OpKind.DROP_SOURCE
            or (_RECREATE_SOURCE_OP_KIND is not None and op.kind == _RECREATE_SOURCE_OP_KIND)
        ):
            try:
                if op.kind == OpKind.UPDATE_SOURCE:
                    _execute_update_stream_source(fs, op)
                elif op.kind == OpKind.DROP_SOURCE:
                    _execute_drop_stream_source(fs, op)
                else:
                    _execute_recreate_stream_source(fs, op)
                ops.append(
                    {
                        "operation": op.kind.value,
                        "name": op.name,
                        "reason": op.reason,
                        "destructive": op.destructive,
                        "status": "success",
                    }
                )
            except Exception as e:
                logger.error("Failed to execute %s for %s: %s", op.kind.value, op.name, e)
                ops.append(
                    {
                        "operation": op.kind.value,
                        "name": op.name,
                        "reason": op.reason,
                        "destructive": op.destructive,
                        "status": "error",
                        "error": str(e),
                    }
                )
                errors.append(f"{op.kind.value} {op.name}: {e}")
                raise
            continue

        try:
            _execute_op(fs, session, op, database, schema, warehouse, options)
            ops.append(
                {
                    "operation": op.kind.value,
                    "name": op.name,
                    "reason": op.reason,
                    "destructive": op.destructive,
                    "status": "success",
                }
            )
        except Exception as e:
            logger.error("Failed to execute %s for %s: %s", op.kind.value, op.name, e)
            ops.append(
                {
                    "operation": op.kind.value,
                    "name": op.name,
                    "reason": op.reason,
                    "destructive": op.destructive,
                    "status": "error",
                    "error": str(e),
                }
            )
            errors.append(f"{op.kind.value} {op.name}: {e}")
            raise

    status = "applied" if not errors else "partial_failure"
    return ApplyResult(
        status=status,
        ops=ops,
        warnings=list(plan.warnings),
        errors=errors,
    )


def _execute_op(
    fs: Any,
    session: Any,
    op: Any,
    database: str,
    schema: str,
    warehouse: str,
    options: PlanOptions,
) -> None:
    """Dispatch a single PlanOp to the appropriate executor.

    All entity + FV ops route through the imperative ``FeatureStore``
    instance ``fs`` — no raw entity-tag SQL is issued from this
    module.  The ``fs`` is constructed eagerly by
    :func:`execute_plan` via
    :func:`assert_feature_store_initialized`, so the init-first
    invariant has already fired by the time this dispatcher runs.

    Args:
        fs: The constructed ``FeatureStore`` instance bound to
            ``database.schema``.  Reused across every op in the plan.
        session: Snowpark session for ``_build_feature_view`` /
            ``_build_feature_df`` (DataFrame construction).
        op: The plan op to execute.
        database: Default database for entity / FV resolution.
        schema: Default schema for entity / FV resolution.
        warehouse: Default warehouse forwarded to FV constructors.
        options: Plan-wide execution options (e.g. ``overwrite``).
    """
    if op.kind == OpKind.CREATE_ENTITY:
        _execute_create_entity(fs, op)

    elif op.kind == OpKind.DROP_ENTITY:
        _execute_drop_entity(fs, op)

    elif op.kind == OpKind.UPDATE_ENTITY:
        _execute_update_entity(fs, op)

    elif op.kind == OpKind.CREATE_FV:
        fv, version = _build_feature_view(op.payload, session, database, schema, warehouse, fs=fs)
        backfill_block = op.payload.get("backfill") if isinstance(op.payload, dict) else None
        backfill_overwrite = bool(backfill_block.get("overwrite")) if isinstance(backfill_block, dict) else False
        # ``PlanOptions.overwrite`` is the legacy global opt-in (force-apply
        # past version-conflict checks); the FV-level ``backfill.overwrite``
        # is the per-FV opt-in for the imperative "backfill cost" path.  A
        # True from either source forwards as ``register_feature_view(overwrite=True)``.
        fs.register_feature_view(
            fv,
            version,
            overwrite=options.overwrite or backfill_overwrite,
        )

    elif op.kind == OpKind.UPDATE_FV:
        _execute_update_feature_view(fs, op, warehouse)

    elif op.kind == OpKind.RECREATE_FV:
        name = op.payload.get("name", "")
        version = op.payload.get("version", "V1")
        try:
            fs.delete_feature_view(name, version)
        except Exception as e:
            logger.warning("delete_feature_view(%s, %s) failed during RECREATE: %s", name, version, e)
        fv, new_version = _build_feature_view(op.payload, session, database, schema, warehouse, fs=fs)
        fs.register_feature_view(fv, new_version, overwrite=options.allow_recreate or True)

    elif op.kind == OpKind.DROP_FV:
        name = op.payload.get("name", "")
        version = op.payload.get("version", "V1")
        fs.delete_feature_view(name, version)

    elif op.kind == OpKind.CREATE_FG:
        _execute_create_feature_group(fs, op)

    elif op.kind == OpKind.DROP_FG:
        _execute_drop_feature_group(fs, op)


def _execute_create_feature_group(fs: Any, op: Any) -> None:
    """Apply ``OpKind.CREATE_FG`` via ``FeatureStore.register_feature_group``.

    Two variants share this entry point:

    * **Non-destructive create** (``op.destructive is False``): register the
      newly-built ``FeatureGroup`` directly.  The ``--allow-recreate`` gate
      did not refuse the plan because no existing FG is being clobbered.
    * **Destructive recreate** (``op.destructive is True``): delete-then-
      register pair.  ``delete_feature_group`` is best-effort — a missing
      existing FG must not fail the recreate (snowml-core makes the OFT
      drop ``IF EXISTS`` and the metadata delete idempotent, but a stale
      schema where the OFT was already dropped manually still has to
      succeed).  Mirrors the FV-level ``RECREATE_FV`` pattern.

    The ``--allow-recreate`` gate at :func:`execute_plan` already refused
    destructive FG plans before calling this function, so by the time we
    get here the operator has explicitly opted in.

    Args:
        fs: Active ``FeatureStore`` instance.
        op: The ``CREATE_FG`` plan op carrying the FG payload.
    """
    payload = op.payload or {}
    name = str(payload.get("name", "") or op.name)
    version = str(payload.get("version", "V1") or "V1")

    if op.destructive:
        # Best-effort delete; a missing FG is a no-op.  The error is
        # logged for diagnostics but does not bubble up — the register
        # half is the load-bearing side-effect.
        try:
            fs.delete_feature_group(name, version)
        except Exception as exc:  # noqa: BLE001 — defensive, see docstring
            logger.warning(
                "delete_feature_group(%s, %s) failed during destructive CREATE_FG: %s",
                name,
                version,
                exc,
            )

    fg, fg_version = _build_feature_group(fs, payload, version)
    fs.register_feature_group(fg, fg_version)


def _execute_drop_feature_group(fs: Any, op: Any) -> None:
    """Apply ``OpKind.DROP_FG`` via ``FeatureStore.delete_feature_group``."""
    payload = op.payload or {}
    name = str(payload.get("name", "") or op.name)
    version = str(payload.get("version", "V1") or "V1")
    fs.delete_feature_group(name, version)


def _build_feature_group(
    fs: Any,
    payload: dict[str, Any],
    version: str,
) -> tuple[Any, str]:
    """Construct an imperative ``FeatureGroup`` from a CREATE_FG payload.

    **Critical contract (do NOT relax)**.  Each source ref is hydrated by
    calling ``fs.get_feature_view(fv_name, fv_version)`` — never by
    instantiating ``FeatureView(...)`` from a payload.  Rationale: the
    imperative ``get_feature_view`` is the only code path that knows about
    the six advanced BFV fields (``warehouse``, ``cluster_by``,
    ``refresh_mode``, ``initialize``, ``storage_config``,
    ``aggregation_secondary_keys``) and every future FV authoring field.
    Reconstructing an FV from the FG payload would silently strip those
    fields; going through ``get_feature_view`` makes the FG path forward-
    compatible with the advanced BFV plan and any successor by construction.

    Slice and alias are layered on top of the hydrated FV:

    * ``slice_columns`` (if present) → ``fv.slice(slice_columns)``.
    * ``alias`` (if present, including ``alias = ""`` which means
      "no prefix" overriding ``auto_prefix``) → ``.with_name(alias)``.
    * ``alias = None`` (the default) is left alone — the FG's
      ``auto_prefix`` flag determines column naming at register time.

    Args:
        fs: Active ``FeatureStore`` instance.
        payload: A CREATE_FG plan-op payload (dict) carrying ``name``,
            ``desc``, ``auto_prefix``, and ``feature_views[]`` of
            ``{name, version, slice_columns?, alias?}`` refs.
        version: User-facing FG version (passed through; the imperative
            ``register_feature_group`` takes this on its second positional
            argument).

    Returns:
        Tuple of ``(FeatureGroup, version)``.  ``register_feature_group``
        consumes both directly.

    Raises:
        ValueError: If any source ref in ``payload['feature_views']`` is
            missing the required ``name`` or ``version`` field.
    """
    # Lazy import — keeps ``execute_plan`` import-light and matches the
    # pattern used by ``_build_feature_view``.
    from snowflake.ml.feature_store.feature_group import FeatureGroup as ImperativeFG

    name = str(payload.get("name", ""))
    desc = str(payload.get("desc", "") or "")
    auto_prefix = bool(payload.get("auto_prefix", True))

    features: list[Any] = []
    for fv_ref in payload.get("feature_views", []) or []:
        if not isinstance(fv_ref, dict):
            continue
        fv_name = str(fv_ref.get("name", ""))
        fv_version = str(fv_ref.get("version", ""))
        if not fv_name or not fv_version:
            raise ValueError(f"FeatureGroup '{name}' has a source ref missing name or version: {fv_ref!r}")

        # The single-line load-bearing call.  See the docstring above for
        # why direct ``FeatureView(...)`` reconstruction is forbidden.
        item = fs.get_feature_view(fv_name, fv_version)

        slice_cols = fv_ref.get("slice_columns")
        if slice_cols:
            item = item.slice(list(slice_cols))

        if "alias" in fv_ref and fv_ref["alias"] is not None:
            # ``alias = ""`` is preserved (semantically: "no prefix");
            # only ``alias = None`` skips the call.
            item = item.with_name(fv_ref["alias"])

        features.append(item)

    fg = ImperativeFG(
        name,
        features=features,
        desc=desc,
        auto_prefix=auto_prefix,
    )
    return fg, version


# UPDATE_FV is now valid for all three FV kinds.  Each kind has its
# own operational subset (see ``planner._FV_OPERATIONAL_FIELDS_BY_KIND``):
#
# * BatchFeatureView   — full surface: desc, refresh_freq, warehouse,
#   online_config.
# * StreamingFeatureView — desc, warehouse, online_config (no
#   refresh_freq — the spec validator
#   ``FeatureView._reject_refresh_freq_on_stream_or_realtime``
#   rejects authoring on this kind, and the runtime stamps
#   ``target_lag_sec=0`` regardless of any cadence).  A5 extended
#   ``FeatureStore.update_feature_view`` to route the streaming kwargs
#   through the ALTER DT + ALTER OFT + ALTER TASK helpers (the
#   StreamingFV materialises as a Dynamic Table over
#   ``$UDF_TRANSFORMED``, so the offline-update path applies unchanged).
# * RealtimeFeatureView — restricted surface: only desc + online_config.
#   ``refresh_freq`` / ``warehouse`` are not forwarded because A5
#   rejects them at the ``FeatureStore.update_feature_view`` layer
#   (RTFVs are OFT-only — no Dynamic Table, no refresh Task).
_UPDATE_FV_SUPPORTED_KINDS: frozenset[str] = frozenset(
    {"BatchFeatureView", "StreamingFeatureView", "RealtimeFeatureView"}
)


def _payload_has_aggregation_windows(payload: dict[str, Any]) -> bool:
    """Whether a plan payload describes a tiled FV — i.e. any feature
    declares an aggregation window.

    Mirrors the compiler's ``has_windows`` check and the imperative
    ``FeatureView.is_tiled``: a tiled FV materialises its tiles as a
    managed Dynamic Table whose ``TARGET_LAG`` is driven by
    ``refresh_freq``.

    Args:
        payload: The FV authoring/plan payload dict.

    Returns:
        ``True`` if any ``features[]`` entry carries ``window`` or
        ``window_sec``, else ``False``.
    """
    return any(
        isinstance(f, dict) and (f.get("window") is not None or f.get("window_sec") is not None)
        for f in (payload.get("features") or [])
    )


def _payload_forwards_refresh_freq(payload: dict[str, Any]) -> bool:
    """Whether ``refresh_freq`` should reach the imperative FV surface for
    this payload.

    ``refresh_freq`` is the offline Dynamic Table cadence, so it is
    forwarded only for kinds that build an offline DT: BatchFeatureView,
    and **tiled** StreamingFeatureView (whose tiles are a managed DT).
    Non-tiled streaming FVs materialise to a zero-lag VIEW and realtime
    FVs compute on lookup — neither has a DT to schedule — so the field
    is dropped there as defence-in-depth against a hand-built payload
    that bypassed the spec validator.

    Args:
        payload: The FV authoring/plan payload dict.

    Returns:
        ``True`` if ``refresh_freq`` should be forwarded for this kind.
    """
    kind = payload.get("kind", "")
    if kind == "BatchFeatureView":
        return True
    if kind == "StreamingFeatureView":
        return _payload_has_aggregation_windows(payload)
    return False


def _execute_update_feature_view(fs: Any, op: Any, default_warehouse: str) -> None:
    """Apply ``OpKind.UPDATE_FV`` via ``FeatureStore.update_feature_view``.

    Routes the planner's operational-drift edits through the imperative
    update path.  Post-B7, all three FV kinds are supported — see the
    :data:`_UPDATE_FV_SUPPORTED_KINDS` docstring for the per-kind
    operational surface.

    The recreate-only authoring fields (``stream_source``,
    ``transformation_fn``, ``feature_granularity``,
    ``aggregation_secondary_keys``, ``backfill_table``,
    ``backfill_start_time``, ``source_refs``) are intentionally not
    alterable here — mutating any of them requires a destructive
    ``RECREATE_FV`` via :meth:`delete_feature_view` +
    :meth:`register_feature_view`.

    Args:
        fs: Active ``FeatureStore`` instance.
        op: Plan op whose ``payload`` is the authoring-format spec dict.
        default_warehouse: Connection warehouse when the payload omits one.

    Raises:
        ValueError: If ``payload["kind"]`` is not a supported FV kind.
    """
    from snowflake.ml.feature_store.feature_view import OnlineConfig, OnlineStoreType

    payload = op.payload or {}
    name = str(payload.get("name", "") or op.name)
    version = str(payload.get("version", "V1") or "V1")
    kind = payload.get("kind", "")
    if kind not in _UPDATE_FV_SUPPORTED_KINDS:
        raise ValueError(
            f"UPDATE_FV is only supported for {sorted(_UPDATE_FV_SUPPORTED_KINDS)} in the "
            f"declarative client; got kind={kind!r} for {name}/{version}."
        )

    is_realtime = kind == "RealtimeFeatureView"

    kwargs: dict[str, Any] = {}
    # ``refresh_freq`` is the offline Dynamic Table cadence.  It is
    # forwarded for kinds that build an offline DT: BatchFeatureView and
    # **tiled** StreamingFeatureView (whose tiles are a managed DT — the
    # spec validator ``FeatureView._reject_refresh_freq_on_stream_or_realtime``
    # accepts it there and the ``STREAM_FV_TILING_REFRESH`` invariant
    # requires it).  Non-tiled streaming and realtime FVs have no DT to
    # schedule, so the field is dropped there as defence-in-depth against
    # a hand-built payload that bypassed the validator.  On a tiled
    # streaming UPDATE_FV this alters the tile DT ``TARGET_LAG``
    # (``ALTER DYNAMIC TABLE … SET TARGET_LAG``).
    if _payload_forwards_refresh_freq(payload):
        schedule = payload.get("refresh_freq")
        if schedule:
            kwargs["refresh_freq"] = str(schedule)

    # ``warehouse`` is skipped for RealtimeFV because A5 rejects it at
    # the imperative layer (no Dynamic Table, no refresh Task).
    if not is_realtime:
        wh = payload.get("warehouse") or default_warehouse
        if wh:
            kwargs["warehouse"] = wh

    if "description" in payload:
        kwargs["desc"] = str(payload.get("description") or "")

    online = payload.get("online")
    if online is not None:
        target_lag = _resolve_online_target_lag(payload)
        if target_lag is None:
            # Kind-aware default mirroring the CREATE_FV path:
            #
            # * ``BatchFeatureView`` → ``_BATCH_OFT_TARGET_LAG``
            #   ("10 seconds").  The spec validator
            #   ``_reject_target_lag_on_offline_batch_fv`` only allows
            #   authored ``target_lag`` on online BFVs, so this branch
            #   covers an online BFV that elided the field.
            # * ``StreamingFeatureView`` / ``RealtimeFeatureView`` →
            #   ``"0 seconds"``.  The spec validator
            #   ``_reject_target_lag_on_stream_or_realtime`` forbids
            #   authoring ``target_lag`` on these kinds entirely
            #   (Snowflake stamps ``target_lag_sec: 0`` on the deployed
            #   OFT regardless and rejects any non-zero value), so the
            #   default MUST be ``"0 seconds"`` — using the BFV
            #   default here surfaces as ``Invalid TARGET_LAG value
            #   '10 seconds' specified for online feature table``.
            if kind == "BatchFeatureView":
                from snowflake.ml.feature_store.feature_view import (
                    _BATCH_OFT_TARGET_LAG,
                )

                target_lag = _BATCH_OFT_TARGET_LAG
            else:
                target_lag = "0 seconds"
        kwargs["online_config"] = OnlineConfig(
            enable=bool(online),
            target_lag=target_lag,
            store_type=OnlineStoreType.POSTGRES,
        )

    if not kwargs:
        return

    fs.update_feature_view(name, version, **kwargs)


def _build_entity(payload: dict[str, Any]) -> Any:
    """Construct an Entity from a spec payload dict.

    Args:
        payload: Dict from a PlanOp with kind=CREATE_ENTITY.
            Expected keys: name, join_keys (list of dicts with "name").

    Returns:
        An ``Entity`` instance.
    """
    from snowflake.ml.feature_store.entity import Entity

    name = payload.get("name", "")
    join_keys_raw = payload.get("join_keys", [])
    join_keys = [jk["name"] if isinstance(jk, dict) else str(jk) for jk in join_keys_raw]
    desc = payload.get("description", "") or ""
    return Entity(name=name, join_keys=join_keys, desc=desc)


def _spec_columns_to_struct_type(columns: list[Any]) -> Any:
    """Translate a spec ``columns:`` list to a Snowpark ``StructType``.

    The spec compiler normalises column types to the
    ``snowflake.snowpark.types`` class names listed in
    :data:`snowflake.ml.feature_store.stream_source._TYPE_NAME_TO_CLASS`
    (``StringType``, ``LongType``, ``DoubleType``, ``DecimalType``,
    ``BooleanType``, ``TimestampType``).  This helper maps that
    string-typed authoring shape onto the real Snowpark types that
    ``StreamSource``'s schema validator expects.

    Args:
        columns: List of spec column dicts; each carries ``name`` and
            ``type`` plus optional precision / scale / length / tz.

    Returns:
        A ``snowflake.snowpark.types.StructType`` matching ``columns``.

    Raises:
        ValueError: If a column carries an unsupported type name.
        RuntimeError: If ``snowflake-snowpark-python`` is not installed —
            the decl wheel does not pull it in transitively, so streaming
            registration requires the imperative feature-store stack.
    """
    try:
        from snowflake.snowpark.types import (
            BooleanType,
            DecimalType,
            DoubleType,
            LongType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )
    except ImportError as e:
        raise RuntimeError(
            "snowflake-snowpark-python is required to register streaming sources "
            "or streaming feature views from a declarative spec; install it via the "
            "imperative feature-store wheel."
        ) from e

    type_map: dict[str, Any] = {
        "StringType": StringType,
        "LongType": LongType,
        "DoubleType": DoubleType,
        "DecimalType": DecimalType,
        "BooleanType": BooleanType,
        "TimestampType": TimestampType,
    }

    fields: list[Any] = []
    for col in columns:
        if not isinstance(col, dict):
            raise ValueError(f"Stream-source column entry is not a dict: {col!r}")
        col_name = col.get("name", "")
        type_name = col.get("type", "")
        cls = type_map.get(type_name)
        if cls is None:
            raise ValueError(
                f"Unsupported column type '{type_name}' for stream-source field '{col_name}'. "
                f"Supported: {sorted(type_map)}"
            )
        if type_name == "DecimalType":
            dt = cls(col.get("precision", 38), col.get("scale", 0))
        elif type_name == "StringType":
            dt = cls(col.get("length")) if col.get("length") else cls()
        else:
            dt = cls()
        fields.append(StructField(col_name, dt))
    return StructType(fields)


def _execute_create_stream_source(fs: Any, payload: dict[str, Any]) -> Any:
    """Register a streaming source in Snowflake's feature-store metadata.

    Mirrors the imperative form: ``fs.register_stream_source(StreamSource
    (name, schema, desc=...))``.  Idempotent — if the source is already
    registered, snowml-core emits a ``UserWarning`` and returns the
    existing object without raising.

    Args:
        fs: A ``FeatureStore`` instance.
        payload: ``CREATE_SOURCE`` op payload with ``name``, ``columns``,
            and optional ``description``.

    Returns:
        The registered ``StreamSource`` (or the pre-existing one when
        idempotency triggers).

    Raises:
        ValueError: If ``payload['name']`` is empty.
    """
    from snowflake.ml.feature_store.stream_source import StreamSource

    name = payload.get("name", "")
    if not name:
        raise ValueError("CREATE_SOURCE for StreamingSource requires a non-empty 'name'.")
    desc = payload.get("description", "") or payload.get("desc", "") or ""
    schema = _spec_columns_to_struct_type(payload.get("columns", []))
    stream_source = StreamSource(name=name, schema=schema, desc=desc)
    return fs.register_stream_source(stream_source)


def _execute_update_stream_source(fs: Any, op: Any) -> Any:
    """Apply ``OpKind.UPDATE_SOURCE`` for a stream-source plan op.

    ``StreamingSource`` payloads route to
    ``fs.update_stream_source(name, desc=desc)`` — the only field
    ``FeatureStore.update_stream_source`` accepts is ``desc`` (see
    ``feature_store.py:update_stream_source``).  ``BatchSource`` payloads
    are informational no-ops because ``BatchSource`` is virtual at the
    runtime level (its identity lives inside the consuming FV's offline
    Dynamic Table); the planner still emits ``UPDATE_SOURCE`` for parity
    with the StreamingSource diff path so the operator sees the change in
    the plan output, but the executor must NOT call any FS method.

    See ``plans/stream_source_contract.md`` §4 for the dispatch table
    that pins this behaviour.

    Args:
        fs: Active ``FeatureStore`` instance.
        op: The ``UPDATE_SOURCE`` plan op carrying the source payload.

    Returns:
        The updated ``StreamSource`` (or ``None`` for the BatchSource
        no-op path).  The dispatcher only inspects the absence of an
        exception, so the return value is informational.

    Raises:
        ValueError: If the StreamingSource payload is missing ``name``.
    """
    payload = op.payload or {}
    kind = payload.get("kind", "")
    if kind != "StreamingSource":
        # ``BatchSource`` (or any future virtual source kind) — no
        # runtime metadata to update; informational success status is
        # recorded by the dispatcher when this function returns
        # without raising.
        return None

    name = str(payload.get("name", "") or op.name)
    if not name:
        raise ValueError("UPDATE_SOURCE for StreamingSource requires a non-empty 'name'.")
    desc = payload.get("description", "") or payload.get("desc", "") or ""
    return fs.update_stream_source(name, desc=desc)


def _execute_recreate_stream_source(fs: Any, op: Any) -> Any:
    """Apply ``OpKind.RECREATE_SOURCE`` for a stream-source plan op.

    Destructive form for streaming-source schema or binding changes that
    ``update_stream_source`` cannot express (it only supports ``desc``).
    ``StreamingSource`` payloads dispatch a two-step
    ``fs.delete_stream_source(name)`` followed by
    ``_execute_create_stream_source(fs, payload)``; the delete-first
    order matters because :func:`_execute_create_stream_source` is
    idempotent in snowml-core (re-registering an existing source emits a
    ``UserWarning`` and returns the existing object without rewriting
    the schema), so a register-without-delete would silently fail to
    apply the new schema.  ``BatchSource`` payloads are informational
    no-ops for the same reason as
    :func:`_execute_update_stream_source` — the actual recreate flows
    through the consuming FVs' ``RECREATE_FV`` ops.

    The ``--allow-recreate`` gate at the top of :func:`execute_plan`
    refuses the plan when ``destructive=True`` and the operator did not
    opt in, so by the time this function runs the operator has already
    consented to the destructive change.

    Args:
        fs: Active ``FeatureStore`` instance.
        op: The ``RECREATE_SOURCE`` plan op carrying the source payload.

    Returns:
        The newly-registered ``StreamSource`` (or ``None`` for the
        BatchSource no-op path).

    Raises:
        ValueError: If the StreamingSource payload is missing ``name``.
    """
    payload = op.payload or {}
    kind = payload.get("kind", "")
    if kind != "StreamingSource":
        return None

    name = str(payload.get("name", "") or op.name)
    if not name:
        raise ValueError("RECREATE_SOURCE for StreamingSource requires a non-empty 'name'.")
    fs.delete_stream_source(name)
    return _execute_create_stream_source(fs, payload)


def _execute_drop_stream_source(fs: Any, op: Any) -> None:
    """Apply ``OpKind.DROP_SOURCE`` for a stream-source plan op.

    ``StreamingSource`` payloads dispatch ``fs.delete_stream_source(name)``.
    Snowml-core's ``delete_stream_source`` raises a
    ``SNOWML_DELETE_FAILED`` error when the source has ``ref_count > 0``
    (any FV still references it); the imperative branch must let that
    exception propagate so the planner's op records
    ``status: partial_failure`` and the operator can see it (contract
    §4).  ``BatchSource`` payloads are informational no-ops.

    Args:
        fs: Active ``FeatureStore`` instance.
        op: The ``DROP_SOURCE`` plan op carrying the source payload.

    Raises:
        ValueError: If the StreamingSource payload is missing ``name``.
    """
    payload = op.payload or {}
    kind = payload.get("kind", "")
    if kind != "StreamingSource":
        return

    name = str(payload.get("name", "") or op.name)
    if not name:
        raise ValueError("DROP_SOURCE for StreamingSource requires a non-empty 'name'.")
    fs.delete_stream_source(name)


def _execute_create_entity(fs: Any, op: Any) -> None:
    """Materialise an ``OpKind.CREATE_ENTITY`` via ``FeatureStore.register_entity``.

    Delegates fully to the imperative API.  ``register_entity`` is
    idempotent: when the entity tag already exists snowml-core emits
    a ``UserWarning`` and returns the pre-existing object instead of
    raising, which means re-applying an unchanged plan is a no-op
    (matching the planner's ``NO_CHANGE`` contract for entities).

    Args:
        fs: The constructed ``FeatureStore`` instance (built in
            :func:`execute_plan` via
            :func:`assert_feature_store_initialized`).
        op: The ``CREATE_ENTITY`` plan op carrying the entity payload.

    Raises:
        ValueError: When ``op.payload`` does not carry an entity name.
    """
    from snowflake.ml.feature_store.entity import Entity

    payload = op.payload
    name = payload.get("name", "") or op.name
    if not name:
        raise ValueError(f"CREATE_ENTITY op {op.name} has no entity name in its payload.")

    join_keys_raw = payload.get("join_keys", [])
    join_keys = [jk["name"] if isinstance(jk, dict) else str(jk) for jk in join_keys_raw]
    desc = payload.get("description", "") or payload.get("desc", "") or ""

    entity = Entity(name=name, join_keys=join_keys, desc=desc)
    fs.register_entity(entity)


def _execute_drop_entity(fs: Any, op: Any) -> None:
    """Materialise an ``OpKind.DROP_ENTITY`` via ``FeatureStore.delete_entity``.

    Delegates fully to the imperative API.  Two snowml-core failure
    modes need rewrapping for the declarative caller:

    * ``NOT_FOUND`` — the entity was already deleted out-of-band (or
      a previous partial apply removed it).  Treat as a soft skip so
      re-applying the plan stays idempotent; the planner's
      ``NO_CHANGE`` contract assumes drops are best-effort.
    * ``SNOWML_DELETE_FAILED`` — Snowflake rejected the underlying
      ``DROP TAG`` because an active feature view still references
      the entity.  Rewrap as :class:`DependencyError` so the CLI can
      surface a clear "drop the FV first" message instead of a
      generic backend error.

    Args:
        fs: The constructed ``FeatureStore`` instance (built in
            :func:`execute_plan` via
            :func:`assert_feature_store_initialized`).
        op: The ``DROP_ENTITY`` plan op carrying the entity payload.

    Raises:
        DependencyError: When the entity is still referenced by an
            active feature view.  The original exception is chained.
        snowml_exceptions.SnowflakeMLException: Any other
            ``SnowflakeMLException`` raised by ``delete_entity`` (e.g.
            permissions / connection errors) is propagated verbatim.
    """
    from snowflake.ml._internal.exceptions import (
        error_codes,
        exceptions as snowml_exceptions,
    )

    payload = op.payload or {}
    name = payload.get("name", "") or op.name

    try:
        fs.delete_entity(name)
    except snowml_exceptions.SnowflakeMLException as exc:
        if exc.error_code == error_codes.NOT_FOUND:
            logger.info(
                "delete_entity(%s) — entity already absent; treating DROP_ENTITY as no-op.",
                name,
            )
            return
        if exc.error_code == error_codes.SNOWML_DELETE_FAILED:
            raise DependencyError(
                f"Cannot drop entity {name!r} because it is still referenced "
                "by one or more deployed feature views. Drop the FV first, "
                f"then re-apply. Underlying error: {exc}"
            ) from exc
        raise


def _execute_update_entity(fs: Any, op: Any) -> None:
    """Materialise an ``OpKind.UPDATE_ENTITY`` via ``FeatureStore.update_entity``.

    Delegates to the imperative API. ``FeatureStore.update_entity``
    accepts ``desc=`` only; join keys are identity for a registered
    entity and are not updated through this path.

    Args:
        fs: The constructed ``FeatureStore`` instance (built in
            :func:`execute_plan` via
            :func:`assert_feature_store_initialized`).
        op: The ``UPDATE_ENTITY`` plan op carrying the new payload.

    Raises:
        ValueError: When ``op.payload`` does not carry an entity name.
    """
    payload = op.payload
    name = payload.get("name", "") or op.name
    if not name:
        raise ValueError(f"UPDATE_ENTITY op {op.name} has no entity name in its payload.")

    desc = payload.get("description", "") or ""

    fs.update_entity(name, desc=desc)


def _build_features(raw: list[dict[str, Any]]) -> list[Any]:
    """Convert planner-payload feature dicts to imperative Feature objects.

    The planner payload encodes each aggregated feature as a dict shaped
    like::

        {
            "function": "sum",
            "window_sec": 3600,
            "offset_sec": 0,                  # optional
            "function_params": {"n": 10},     # optional, e.g. last_n
            "source_column": {"name": "AMOUNT", "type": "DoubleType"},
            "output_column": {"name": "AMOUNT_SUM_1H", "type": "DoubleType"},
        }

    The imperative ``Feature`` builder accepts ``window`` / ``offset`` as
    duration strings (``"3600s"``, ``"60s"``) which round-trip through
    ``interval_to_seconds`` to identical seconds — choosing the
    seconds-suffix form means we never have to pretty-print durations
    here, and matches the shape ``DESCRIBE ... TYPE = SPECIFICATION``
    returns.  ``function`` strings map to ``AggregationType`` via the
    enum's value (case-insensitive lookup so the BUG_BASH-style
    lower-case ``"sum"`` / ``"max"`` payloads work as-is).  Any
    ``function_params`` dict is forwarded as ``**params`` to support
    ``last_n``-family / ``approx_percentile`` aggregations.

    Both ``Feature`` and ``AggregationType`` are imported lazily so
    importing this module does not pull in numpy/pandas/pyarrow at
    package load time — the architectural decl-isolation rule.

    Args:
        raw: The payload's ``features`` list (post-compile, with
            ``window_sec`` integers rather than authoring strings).

    Returns:
        A list of imperative ``Feature`` builders, each with its alias
        set to the payload's ``output_column.name``.  An empty input
        produces an empty list (the caller skips the kwarg entirely in
        that case so non-aggregated streaming FVs stay on the existing
        ``stream_config``-only path).

    Raises:
        ValueError: When an aggregation feature is missing ``function``,
            ``source_column``, or a resolved ``window_sec``.
    """
    from snowflake.ml.feature_store.aggregation import AggregationType
    from snowflake.ml.feature_store.feature import Feature

    features: list[Any] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        fn_name = entry.get("function")
        window_sec = entry.get("window_sec")
        has_window_sec = isinstance(window_sec, int)
        raw_window = entry.get("window")
        if not fn_name and not has_window_sec and raw_window is None:
            continue

        src_col_raw = entry.get("source_column", "")
        if isinstance(src_col_raw, dict):
            column = src_col_raw.get("name", "")
        else:
            column = str(src_col_raw)
        column = str(column or "").strip()

        out_col_raw = entry.get("output_column")
        if isinstance(out_col_raw, dict):
            alias = str(out_col_raw.get("name") or "").strip()
        else:
            alias = str(out_col_raw or "").strip()
        label = alias if alias else f"#{index}"
        prefix = f"aggregation {label}"

        if not fn_name:
            raise ValueError(f"{prefix} requires ``function``.")
        if not column:
            raise ValueError(f"{prefix} requires ``source_column``.")
        if not isinstance(window_sec, int):
            if raw_window is not None:
                raise ValueError(
                    f"{prefix} has unusable ``window: {raw_window!r}``. "
                    "Use a duration with a unit (``5m``, ``1h``); bare "
                    "numbers like ``300`` and fractional values like "
                    "``1.5m`` are rejected."
                )
            raise ValueError(f"{prefix} requires ``window`` (e.g. ``5m``) or ``window_sec``.")

        agg_type = AggregationType(str(fn_name).lower())
        window = f"{window_sec}s"

        offset_sec = entry.get("offset_sec")
        offset = f"{int(offset_sec)}s" if offset_sec else "0"

        params = entry.get("function_params") or {}
        if not isinstance(params, dict):
            params = {}

        feat = Feature(agg_type, column, window, offset, **params)
        if alias:
            feat = feat.alias(alias)

        features.append(feat)

    return features


def _normalize_join_key_token(raw: Any) -> str:
    """Normalise a join-key cell from ``list_entities`` JSON for comparisons."""
    s = str(raw).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()


def _get_entity_for_feature_view_ref(fs: Any, ref: str) -> Any:
    """Resolve a single ``entities`` entry to an ``Entity``.

    Authoring YAML usually lists **join-key column names** here (the same
    strings ``validate_specs`` checks against entity ``join_keys``).  The
    imperative ``FeatureStore.get_entity`` API keys off the **entity tag
    name** instead.  When the two coincide (``name: USER_ID`` with join key
    ``USER_ID`` — the streaming BUG_BASH shape) a direct ``get_entity`` lookup
    succeeds.  When they differ (``name: USER_BATCH_DECL`` with join key
    ``USER_ID``) we fall back to ``list_entities`` and match on declared join
    keys, then re-fetch by canonical entity name.

    Args:
        fs: Active ``FeatureStore`` instance.
        ref: One entry from the FV's ``entities`` list (already a plain
            string — dict ``{"name": ...}`` entries are flattened by callers).

    Returns:
        The ``Entity`` instance returned by ``fs.get_entity``.

    Raises:
        ValueError: When ``ref`` is empty, no entity matches, or multiple entities
            declare the same join key (ambiguous).
        RuntimeError: When ``list_entities().collect()`` fails during join-key
            resolution (after a ``NOT_FOUND`` direct lookup).
        snowml_exceptions.SnowflakeMLException: Propagated from ``get_entity`` /
            ``list_entities`` for non-``NOT_FOUND`` failures.
    """
    import json

    from snowflake.ml._internal.exceptions import (
        error_codes,
        exceptions as snowml_exceptions,
    )

    ref_n = str(ref or "").strip()
    if not ref_n:
        raise ValueError("FeatureView `entities` entry is empty.")

    try:
        return fs.get_entity(ref_n)
    except snowml_exceptions.SnowflakeMLException as exc:
        if exc.error_code != error_codes.NOT_FOUND:
            raise

    try:
        rows = fs.list_entities().collect()
    except Exception as e:
        raise RuntimeError(f"list_entities() failed while resolving entity ref {ref_n!r}: {e}") from e

    matches: list[str] = []
    ref_upper = ref_n.upper()
    for row in rows:
        r = row.as_dict() if hasattr(row, "as_dict") else dict(row)
        name = str(r.get("NAME", "") or "").strip()
        if not name:
            continue
        jk_raw = r.get("JOIN_KEYS", "")
        try:
            keys_raw = json.loads(jk_raw) if isinstance(jk_raw, str) else list(jk_raw or [])
        except json.JSONDecodeError:
            keys_raw = []
        norm_keys = {_normalize_join_key_token(k).upper() for k in keys_raw}
        if ref_upper in norm_keys:
            matches.append(name)

    if len(matches) == 1:
        return fs.get_entity(matches[0])
    if not matches:
        raise ValueError(f"No entity registered with name {ref_n!r} and no entity declares join key {ref_n!r}.")
    raise ValueError(
        f"Ambiguous entity resolution for join key {ref_n!r}: " f"multiple entities declare it: {sorted(matches)!r}"
    )


def _build_feature_view(
    payload: dict[str, Any],
    session: Any,
    database: str,
    schema: str,
    warehouse: str,
    *,
    fs: Any,
) -> tuple[Any, str]:
    """Construct a FeatureView from a spec payload dict.

    Resolves each entry in the payload's ``entities`` (the authoring-side
    name; matches the imperative ``FeatureView(entities=...)`` kwarg)
    against the live ``FeatureStore`` registry.  Entries may be
    **entity tag names** (direct ``get_entity`` hit) or **join-key
    column names** declared on exactly one entity in the store (see
    :func:`_get_entity_for_feature_view_ref`).  The planner
    topologically orders ``CREATE_ENTITY`` ops before dependent
    ``CREATE_FV`` ops so registration already exists when this runs.

    Args:
        payload: Dict from a PlanOp with kind=CREATE_FV/UPDATE_FV/RECREATE_FV.
        session: Snowpark Session for DataFrame construction.
        database: Default database.
        schema: Default schema.
        warehouse: Default warehouse.
        fs: The ``FeatureStore`` instance constructed eagerly by
            :func:`execute_plan`.  Used to look up each declared
            entity by name.

    Returns:
        Tuple of (FeatureView, version_string).
    """
    from snowflake.ml.feature_store.feature_view import (
        FeatureView,
        OnlineConfig,
        OnlineStoreType,
    )

    name = payload.get("name", "")
    version = payload.get("version", "V1")

    # The planner builds ``PlanOp.payload`` from ``_model_to_dict`` so
    # the dict already speaks the new authoring vocabulary (``entities``
    # / ``timestamp_col``).  Translation to the wire-form names only
    # happens inside :func:`spec_compiler.compile_to_spec`, which the
    # planner runs separately for hashing.
    entity_cols = payload.get("entities", [])
    entity_refs = [col["name"] if isinstance(col, dict) else str(col) for col in entity_cols]
    entities: list[Any] = []
    seen_entity_names: set[str] = set()
    for ref in entity_refs:
        ref_s = str(ref or "").strip()
        if not ref_s:
            continue
        ent = _get_entity_for_feature_view_ref(fs, ref_s)
        ename = str(getattr(ent, "name", "") or "")
        if ename and ename not in seen_entity_names:
            seen_entity_names.add(ename)
            entities.append(ent)

    # Optional parameters
    kwargs: dict[str, Any] = {}
    if payload.get("timestamp_col"):
        kwargs["timestamp_col"] = payload["timestamp_col"]
    # ``refresh_freq`` resolution.
    #
    # ``refresh_freq`` (free-form Dynamic-Table cadence string) is the
    # **single source of truth** for the offline DT refresh.  Same name
    # on both sides — the declarative authoring key and the imperative
    # ``FeatureView(refresh_freq=...)`` constructor kwarg agree, so no
    # rename layer remains here.  The authoring-side ``target_lag`` /
    # ``target_lag_sec`` field is OFT staleness only (it maps to
    # ``OnlineConfig.target_lag``) and is NOT read here.
    #
    # ``refresh_freq`` is the offline Dynamic Table cadence, so it is
    # forwarded only for kinds that build an offline DT: BatchFeatureView
    # and **tiled** StreamingFeatureView (see
    # ``_payload_forwards_refresh_freq``).  Non-tiled streaming and
    # realtime kinds have no DT to schedule — the spec validator
    # (``FeatureView._reject_refresh_freq_on_stream_or_realtime``) rejects
    # authoring on them and the runtime stamps ``target_lag_sec=0`` — so
    # the field is dropped there as defence-in-depth against a hand-built
    # payload that bypassed the validator.
    #
    # Tiled FVs (batch AND streaming) REQUIRE ``refresh_freq``:
    # snowml-core's ``FeatureView`` constructor raises otherwise (the
    # tile-based check in ``feature_view.py:_validate``), matching the
    # declarative-side invariants ``BATCH_FV_TILING_REFRESH`` /
    # ``STREAM_FV_TILING_REFRESH``.  There is no granularity-based
    # default: a tiled FV without an authored ``refresh_freq`` hits the
    # imperative error verbatim.
    if _payload_forwards_refresh_freq(payload) and payload.get("refresh_freq"):
        kwargs["refresh_freq"] = payload["refresh_freq"]
    if payload.get("warehouse", warehouse):
        kwargs["warehouse"] = payload.get("warehouse", warehouse)
    if payload.get("description"):
        kwargs["desc"] = payload["description"]
    # Advanced BFV structural knobs.  ``cluster_by`` is the only one wired
    # up so far (Phase 2); subsequent phases add ``refresh_mode``,
    # ``initialize``, ``storage_config``, and ``aggregation_secondary_keys``
    # at this same insertion point.  Each is forwarded verbatim as a kwarg
    # to the imperative ``FeatureView(...)`` constructor below.
    cluster_by = payload.get("cluster_by")
    if isinstance(cluster_by, list) and cluster_by:
        kwargs["cluster_by"] = [str(c) for c in cluster_by]
    refresh_mode = payload.get("refresh_mode")
    if isinstance(refresh_mode, str) and refresh_mode:
        kwargs["refresh_mode"] = refresh_mode.upper()
    storage_config_dict = payload.get("storage_config")
    if isinstance(storage_config_dict, dict) and storage_config_dict:
        from snowflake.ml.feature_store.feature_view import StorageConfig, StorageFormat

        fmt_raw = str(storage_config_dict.get("format", "snowflake")).lower()
        try:
            fmt = StorageFormat(fmt_raw)
        except ValueError:
            fmt = StorageFormat.SNOWFLAKE
        kwargs["storage_config"] = StorageConfig(
            format=fmt,
            external_volume=storage_config_dict.get("external_volume"),
            base_location=storage_config_dict.get("base_location"),
        )
    secondary_keys = payload.get("aggregation_secondary_keys")
    if isinstance(secondary_keys, list) and secondary_keys:
        kwargs["aggregation_secondary_keys"] = [str(c) for c in secondary_keys]
    if payload.get("feature_granularity"):
        fg = payload["feature_granularity"]
        kwargs["feature_granularity"] = str(fg) if isinstance(fg, int) else fg

    # Plan section A1 (Phase E A-bis fix-up) — stamp ``_source_refs``
    # onto the FeatureView so ``FeatureStore.register_feature_view``
    # persists the authored source bindings as
    # ``MetadataType.FV_SOURCE_REFS``.  Without this stamp the write
    # gate at ``feature_store.py:if feature_view.source_refs:`` is
    # forever false on declarative apply paths and replan recovery
    # falls back to the legacy DT-text shim (Phase B4), which leaves
    # ``BatchFeatureView.sources`` empty and triggers spurious
    # ``RECREATE_FV`` ops on the second plan for offline tiled BFVs
    # (the live-verify AS2 invalidation symptom on ``MY_ADV_BFV_DECL``).
    # ``payload["sources"]`` is the canonical ``list[dict]`` produced
    # by the planner's ``_model_to_dict`` over ``SourceRef`` entries;
    # it already has the ``name`` / ``source_type`` / ``table`` (or
    # ``query``) / ``columns`` shape ``FvSourceRefsMetadata`` expects
    # so it flows through verbatim without translation.  Defensive:
    # skip the stamp on an empty list so the write-gate behaviour
    # matches the imperative-callers contract.
    source_refs_payload = payload.get("sources")
    if isinstance(source_refs_payload, list) and source_refs_payload:
        canonical_refs = [dict(s) for s in source_refs_payload if isinstance(s, dict)]
        if canonical_refs:
            kwargs["_source_refs"] = canonical_refs

    # ``initialize`` resolution.  Phase 4 of the advanced BFV plan
    # promoted ``initialize`` to a first-class top-level authoring key,
    # but the legacy nested ``backfill.initialize`` shape is still
    # accepted for back-compat.  Resolution order (highest priority first):
    #
    #   1. ``payload["initialize"]``    — top-level value (the new
    #                                     canonical form).
    #   2. ``payload["backfill"].initialize`` — legacy nested form.
    #
    # Streaming FVs use StreamConfig lifecycle and never carry initialize on
    # the constructor (the spec validator already rejects this combination).
    fv_kind = payload.get("kind", "")
    if fv_kind == "BatchFeatureView":
        initialize = payload.get("initialize")
        if not initialize:
            backfill_block = payload.get("backfill") if isinstance(payload.get("backfill"), dict) else None
            if backfill_block:
                initialize = backfill_block.get("initialize")
        if initialize:
            kwargs["initialize"] = str(initialize).upper()

    # Online config — resolve ``OnlineConfig.target_lag`` from EITHER the
    # raw ``target_lag`` field (rare; the loader's ``normalize_durations``
    # pass rewrites human-friendly strings to ``target_lag_sec`` integers
    # before this executor runs) OR the normalised ``target_lag_sec``.
    #
    # Authoring ``target_lag`` is OFT staleness only — strictly the
    # value the imperative ``OnlineConfig.target_lag`` receives (see
    # ``feature_view.py:build_oft_create_sql`` for the
    # ``CREATE ONLINE FEATURE TABLE … TARGET_LAG='<val>'`` clause it
    # ends up driving).  The DT refresh comes from ``refresh_freq``
    # (above) and never crosses into this resolution path.
    #
    # Default policy when ``target_lag`` is absent:
    # * ``BatchFeatureView`` → ``_BATCH_OFT_TARGET_LAG`` ("10 seconds"),
    #   the same default the imperative ``register_feature_view`` /
    #   ``_plan_online_update_existing`` path applies to a fresh online
    #   batch FV.  Snowflake rejects ``"0 seconds"`` for batch OFTs.
    # * ``StreamingFeatureView`` / ``RealtimeFeatureView`` /
    #   ``FeatureGroup`` → ``"0 seconds"`` (Snowflake requires this for
    #   those kinds — the runtime enforces it regardless).
    if payload.get("online", False):
        target_lag = _resolve_online_target_lag(payload)
        if target_lag is None:
            online_kind = payload.get("kind", "") if isinstance(payload.get("kind"), str) else ""
            if online_kind == "BatchFeatureView":
                from snowflake.ml.feature_store.feature_view import (
                    _BATCH_OFT_TARGET_LAG,
                )

                target_lag = _BATCH_OFT_TARGET_LAG
            else:
                target_lag = "0 seconds"
        kwargs["online_config"] = OnlineConfig(
            enable=True,
            target_lag=target_lag,
            store_type=OnlineStoreType.POSTGRES,
        )

    # Streaming feature views go through StreamConfig — snowml-core's
    # streaming preamble (run_streaming_preamble) requires it to create
    # the udf_transformed and $BACKFILL tables.  Passing a stub
    # ``feature_df`` here would fall back to the batch CREATE VIEW path
    # and emit ``CREATE VIEW name () ... AS query`` (empty column list)
    # because schema introspection on an unresolvable placeholder
    # silently sets ``feature_descs = None``.
    if payload.get("kind") == "StreamingFeatureView":
        kwargs["stream_config"] = _build_stream_config(payload, session, database, schema, fs=fs)

        # Aggregation triple (BUG_BASH step 7) — when the planner
        # payload carries ``features`` + ``feature_granularity_sec`` +
        # ``feature_aggregation_method`` (the canonical streaming-FV
        # tiled aggregation shape), thread all three through to the
        # imperative ``FeatureView`` constructor so snowml-core's
        # ``is_tiled`` branch fires and ``_feature_desc`` is keyed off
        # the windowed-aggregation outputs (e.g. ``TOTAL_ENGAGEMENT_1H``)
        # instead of the UDF's raw output columns
        # (e.g. ``ENGAGEMENT_SCORE``).  See
        # :func:`_build_features` for the per-feature conversion.
        feature_granularity_sec = payload.get("feature_granularity_sec")
        if feature_granularity_sec:
            kwargs["feature_granularity"] = f"{int(feature_granularity_sec)}s"

        aggregation_method = payload.get("feature_aggregation_method")
        if aggregation_method:
            # ``FeatureAggregationMethod`` lives in ``feature_store.spec.enums``,
            # which the wheel-isolation tests forbid this module from
            # importing directly.  ``feature_view.py`` already re-exports
            # the enum (it imports it at module top-level for type hints),
            # so we go through that allowed path — same module we already
            # import ``FeatureView`` from on the line above.
            from snowflake.ml.feature_store import feature_view as _fv

            FeatureAggregationMethod = _fv.FeatureAggregationMethod  # type: ignore[attr-defined]
            kwargs["feature_aggregation_method"] = FeatureAggregationMethod(aggregation_method)

        features = _build_features(payload.get("features", []) or [])
        if features:
            kwargs["features"] = features

        # ``refresh_freq`` for a tiled streaming FV was already added to
        # ``kwargs`` above by the ``_payload_forwards_refresh_freq`` gate:
        # it drives the offline tile Dynamic Table's ``TARGET_LAG`` (the
        # tiles are a managed DT), and snowml-core's ``_validate`` requires
        # it for every tiled kind.  A non-tiled streaming FV never reaches
        # that gate (no aggregation windows) and materialises to a zero-lag
        # VIEW instead.  The OFT's ``target_lag_sec`` is a separate knob the
        # runtime stamps to 0 on the ingest path and is not read here.

        fv = FeatureView(
            name=name,
            entities=entities,
            **kwargs,
        )
        return fv, version

    feature_df = _build_feature_df(payload, session, database, schema)

    if payload.get("kind") == "BatchFeatureView":
        tiled = _build_features(payload.get("features", []) or [])
        if tiled:
            feature_granularity_sec = payload.get("feature_granularity_sec")
            if feature_granularity_sec:
                kwargs["feature_granularity"] = f"{int(feature_granularity_sec)}s"

            # NOTE: snowml-core's ``FeatureView.__init__`` rejects
            # ``feature_aggregation_method`` for non-streaming FVs (see
            # ``feature_view.py:974-977``: "feature_aggregation_method is
            # only supported for streaming feature views.").  For BFVs
            # the *tiled* shape is conveyed implicitly via
            # ``features=[FeatureSpec(function=..., window=...)]``; the
            # ``feature_aggregation_method:`` authoring key is accepted
            # (and round-tripped) but MUST NOT be forwarded as a kwarg
            # here.  Forwarding it broke the live ADVANCED_BVT bug-bash
            # apply with the exact error above.

            kwargs["features"] = tiled
            # The pre-fix ``feature_granularity → refresh_freq`` default
            # is removed: tiled BFVs without an authored ``refresh_freq``
            # now hit the imperative-side ``ValueError("refresh_freq is
            # required for tile-based aggregations.")`` directly.  The
            # validator side surfaces the friendlier
            # ``BATCH_FV_TILING_REFRESH`` message before apply runs.

    fv = FeatureView(
        name=name,
        entities=entities,
        feature_df=feature_df,
        **kwargs,
    )

    return fv, version


def _build_stream_config(
    payload: dict[str, Any],
    session: Any,
    database: str,
    schema: str,
    *,
    fs: Any,
) -> Any:
    """Construct a snowml-core ``StreamConfig`` from a streaming-FV payload.

    Wires three concerns:

    - ``stream_source``: the ``sources[0].name`` from the FV spec.  The
      planner's ``CREATE_SOURCE`` op (driven by
      :func:`fetch_stream_source_rows`-populated applied state) is the
      single source of truth for stream-source registration now — see
      ``plans/stream_source_contract.md`` §3 for the read-path contract
      and §4 for the planner's four-way source decision.  The previous
      defensive ``_execute_create_stream_source`` call inside this helper
      was the second source of ``UserWarning: StreamSource <name> already
      exists. Skip registration.`` noise the user reported; the planner
      now emits exactly one ``CREATE_SOURCE`` per genuinely-new source
      before any ``CREATE_FV`` op runs (topological-sort contract in
      :mod:`dependencies`), so the FV-side defensive register has been
      removed.  The resolved ``stream_source_name`` extracted from
      ``payload['sources']`` still flows through to ``StreamConfig`` so
      the streaming preamble's ``get_stream_source`` lookup names the
      correct source.
    - ``transformation_fn``: a real Python callable compiled from
      ``payload['udf']['function_definition']`` via
      :func:`udf_loader.compile_udf_callable`.  ``inspect.getsource``
      succeeds on the result so ``StreamConfig.__post_init__``'s AST
      import-guard runs end-to-end.
    - ``backfill_df``: a Snowpark DataFrame.  Resolved from the
      FV-level ``backfill.table`` FQN when declared
      (``session.table(<fqn>)``), else synthesized as a typed
      one-row in-memory DataFrame from the source's ``columns:``
      so the streaming preamble's ``.limit(10).to_pandas()`` probe
      is non-empty.
    - ``backfill_start_time``: optional ``datetime`` forwarded from
      the FV-level ``backfill.start_time`` (parsed from ISO 8601
      strings via :func:`_coerce_backfill_start_time`).  When unset,
      ``StreamConfig`` defaults to ``None`` (no time filter).

    Args:
        payload: ``CREATE_FV`` payload with ``kind == 'StreamingFeatureView'``.
        session: Snowpark session used for ``session.table`` /
            ``session.create_dataframe``.
        database: Default database (unused today; reserved for future
            use when stream sources gain DB / schema qualification).
        schema: Default schema (same as ``database``).
        fs: The ``FeatureStore`` instance constructed eagerly by
            :func:`execute_plan`.  Reserved for future use; kept on the
            signature so :func:`_build_feature_view` callers do not have
            to special-case the streaming branch.

    Returns:
        A ``StreamConfig`` instance.

    Raises:
        ValueError: If the spec is missing required fields (no streaming
            source in ``sources[]`` or no UDF / function_definition).
    """
    from snowflake.ml.feature_store.decl.udf_loader import compile_udf_callable
    from snowflake.ml.feature_store.stream_config import StreamConfig

    del database, schema, fs  # reserved for future qualification / stream-source ops

    sources = payload.get("sources", []) or []
    stream_source_spec = next(
        (s for s in sources if isinstance(s, dict)),
        None,
    )
    if stream_source_spec is None or not stream_source_spec.get("name"):
        raise ValueError(
            f"StreamingFeatureView '{payload.get('name', '')}' must declare a "
            "stream source in 'sources[]' with a 'name'."
        )
    stream_source_name: str = str(stream_source_spec["name"])

    udf = payload.get("udf") or {}
    if not isinstance(udf, dict):
        raise ValueError(
            f"StreamingFeatureView '{payload.get('name', '')}' has a malformed "
            "'udf' field — expected a dict with 'name' and 'function_definition'."
        )
    function_definition = udf.get("function_definition")
    udf_name = udf.get("name", "")
    if not function_definition or not udf_name:
        raise ValueError(
            f"StreamingFeatureView '{payload.get('name', '')}' must declare a "
            "'udf' with both 'name' and inlined 'function_definition' source."
        )
    if not isinstance(function_definition, str):
        # The compiler ordinarily inlines the .py file as a string; if a
        # caller passed an already-compiled callable, accept it verbatim.
        if callable(function_definition):
            transformation_fn = function_definition
        else:
            raise ValueError(
                f"StreamingFeatureView '{payload.get('name', '')}': "
                "'udf.function_definition' must be a Python source string "
                "(the spec compiler inlines udf.file as text)."
            )
    else:
        transformation_fn = compile_udf_callable(function_definition, udf_name)

    backfill_block = payload.get("backfill") or {}
    if not isinstance(backfill_block, dict):
        backfill_block = {}
    backfill_df = _build_streaming_backfill_df(backfill_block, stream_source_spec, session)
    backfill_start_time = _coerce_backfill_start_time(backfill_block.get("start_time"))
    sc_kwargs: dict[str, Any] = {
        "stream_source": stream_source_name,
        "transformation_fn": transformation_fn,
        "backfill_df": backfill_df,
    }
    if backfill_start_time is not None:
        sc_kwargs["backfill_start_time"] = backfill_start_time
    # Plan section A3 (Phase E A-bis fix-up) — stamp ``backfill_table``
    # onto the StreamConfig so ``streaming_registration.save_streaming_metadata``
    # persists the authored FQN into ``StreamingMetadata.backfill_table``.
    # Without this stamp the exporter / list-FV row cannot re-emit the
    # operator's ``backfill.table:`` block on the second plan and the
    # planner trips ``UPDATE_FV`` / ``RECREATE_FV`` whenever the
    # authored YAML carries ``backfill.table:`` (the L2 limitation
    # closure on streaming FVs).  ``backfill_table`` is the
    # pre-expansion FQN string exactly as authored — distinct from
    # ``backfill_df`` (the resolved Snowpark DataFrame) which already
    # flows through above; both live on ``StreamConfig`` per the
    # ``stream_config.py`` contract.
    raw_backfill_table = backfill_block.get("table")
    if isinstance(raw_backfill_table, str) and raw_backfill_table.strip():
        sc_kwargs["backfill_table"] = raw_backfill_table.strip()
    return StreamConfig(**sc_kwargs)


def _coerce_backfill_start_time(value: Any) -> Any:
    """Coerce the FV-level ``backfill.start_time`` field into a ``datetime``.

    Authoring YAML supplies an ISO 8601 string (e.g.
    ``"2026-05-18T22:00:00"``) but ``StreamConfig.backfill_start_time`` is
    typed as ``Optional[datetime.datetime]``.  This helper accepts:

    * ``None`` → ``None`` (no filter applied).
    * ``datetime.datetime`` → returned unchanged.
    * ``str`` → parsed via :meth:`datetime.datetime.fromisoformat`; a
      trailing ``Z`` is treated as UTC (``datetime.fromisoformat``
      rejected ``Z`` before Python 3.11; we strip it manually for
      forward compatibility with operator-friendly timestamps).

    Any other shape is returned verbatim so the underlying
    ``StreamConfig`` validator surfaces a clean type error.

    Args:
        value: The raw ``backfill.start_time`` value from the spec dict.

    Returns:
        ``None``, a ``datetime``, or the original value when no coercion
        applies.
    """
    import datetime as _dt

    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return _dt.datetime.fromisoformat(text)
        except ValueError:
            return value
    return value


def _build_streaming_backfill_df(
    backfill_block: dict[str, Any],
    stream_source_spec: dict[str, Any],
    session: Any,
) -> Any:
    """Return a Snowpark DataFrame to use as ``StreamConfig.backfill_df``.

    Two-tier resolution:

    1. If the FV-level ``backfill.table`` is set, resolve via
       ``session.table(<fqn>)`` so backfill rows come from a real
       historical table.
    2. Otherwise synthesize a single typed-sentinel row from the stream
       source's ``columns:`` and return
       ``session.create_dataframe([row], schema=...)``.  This satisfies
       the streaming preamble's ``.limit(10).to_pandas()`` probe (which
       fails on zero rows) without requiring a Snowflake round-trip —
       REST stream sources without a historical backfill table still
       register cleanly.

    Args:
        backfill_block: The FV-level ``backfill`` block (may be empty).
            Carries the optional ``table`` field.
        stream_source_spec: The dict for the streaming source as it
            appears in the FV's ``sources[]`` list (carries ``name`` and
            ``columns``).  Used for the synthesized fallback only.
        session: Snowpark session.

    Returns:
        A Snowpark DataFrame suitable as ``StreamConfig.backfill_df``.
    """
    backfill_table = backfill_block.get("table") if isinstance(backfill_block, dict) else None
    if backfill_table:
        return session.table(backfill_table)

    columns = stream_source_spec.get("columns", []) or []
    schema = _spec_columns_to_struct_type(columns)
    sentinel_row = _typed_sentinel_row(columns)
    return session.create_dataframe([sentinel_row], schema)


def _typed_sentinel_row(columns: list[Any]) -> list[Any]:
    """Return one type-zero sentinel value per column for synthetic backfill.

    snowml-core's streaming preamble probes ``backfill_df.limit(10).to_pandas()``
    and rejects empty frames; one synthetic row is enough to pass.  Using
    ``None`` for every column would lose type information at the Snowpark
    serialisation boundary, so we emit type-zero values instead — empty
    string, ``0`` / ``0.0``, ``False``, and a real ``datetime`` — which
    snowml's pandas-dtype inference can resolve.

    Args:
        columns: The spec's ``columns:`` list (each entry has ``type``).

    Returns:
        A list of one Python value per column, in spec order.
    """
    import datetime
    from decimal import Decimal

    sentinels: dict[str, Any] = {
        "StringType": "",
        "LongType": 0,
        "DoubleType": 0.0,
        "BooleanType": False,
        "TimestampType": datetime.datetime.now(),
        "DecimalType": Decimal(0),
    }
    row: list[Any] = []
    for col in columns:
        type_name = col.get("type", "") if isinstance(col, dict) else ""
        row.append(sentinels.get(type_name, ""))
    return row


def _build_feature_df(
    payload: dict[str, Any],
    session: Any,
    database: str,
    schema: str,
) -> Any:
    """Construct a Snowpark DataFrame from a non-streaming FV's sources.

    Two first-class source shapes are supported per
    :class:`~snowflake.ml.feature_store.decl.spec_models.BatchSource`:

    * ``table:`` — returns ``session.table(<src_db>.<src_schema>.<table>)``.
      Source-side qualifiers (``source_database`` / ``source_schema``)
      override the FV's deployment qualifier when present; the FV's
      database / schema are the fallback for unqualified tables.

    * ``query:`` — returns ``session.sql(query)``. The SQL body is
      passed verbatim; whitespace normalization is a compile-time
      concern handled by
      :func:`snowflake.ml.feature_store.decl.compiler.normalize_sql_whitespace`.
      The Phase-4 DT-text recovery path
      (:func:`snowflake.ml.feature_store.decl.state._inject_batch_fv_source_from_dt_text`)
      reconstructs the same shape from the deployed
      ``CREATE DYNAMIC TABLE … AS <query>`` body so re-applies are
      no-ops — see ``docs/ARCHITECTURE.md`` apply pipeline section.
      Source-side qualifiers are ignored for query-backed sources;
      any qualification belongs in the SQL itself.

    ``table:`` takes precedence over ``query:`` if both are present
    (defensive — the spec validator
    :class:`~snowflake.ml.feature_store.decl.spec_models.BatchSource`
    already rejects this combination at authoring time).

    Streaming feature views never reach this function — they go through
    :func:`_build_stream_config` instead.  An older revision of this code
    fell back to a synthetic ``SELECT cols FROM <db>.<schema>.PLACEHOLDER
    WHERE 1=0`` query for streaming sources without a table; that query
    is unresolvable at registration time and caused snowml-core to emit
    ``CREATE VIEW name () AS query`` (empty column list) — see
    BUG_BASH §6 for the original bug.  The fallback is removed; a
    non-streaming FV without a resolvable source now fails loudly.

    Args:
        payload: Parsed feature view spec dict.  Must NOT have
            ``kind == 'StreamingFeatureView'`` — caller routes streaming
            FVs through :func:`_build_stream_config`.
        session: A ``snowflake.snowpark.Session`` instance.
        database: Snowflake database name for resolving unqualified tables.
        schema: Snowflake schema name for resolving unqualified tables.

    Returns:
        A Snowpark DataFrame representing the feature view's source query.

    Raises:
        ValueError: If no source declares a usable ``table`` or ``query``.
    """
    sources = payload.get("sources", [])

    for src in sources:
        if isinstance(src, dict):
            table = src.get("table")
            if table:
                src_db = src.get("source_database", database)
                src_schema = src.get("source_schema", schema)
                fqn = f"{src_db}.{src_schema}.{table}"
                return session.table(fqn)

            query = src.get("query")
            if query:
                return session.sql(query)

    raise ValueError(
        f"FeatureView '{payload.get('name', '')}' (kind={payload.get('kind', '')}) "
        "has no resolvable source: at least one entry in 'sources[]' must "
        "declare 'table' or 'query', or the spec must declare "
        "'kind: StreamingFeatureView' (which uses a StreamConfig instead)."
    )
