"""Pydantic v2 authoring-format models for the declarative feature store.

These models represent the human-facing spec format: string-based type
identifiers, human-friendly duration strings, and rich references.
They are fully standalone — no imports from ``snowflake.ml.feature_store.spec``,
``snowflake.snowpark``, or ``snowflake.connector``.

A compilation step (``decl/compiler.py``) transforms these into the JSON
payload format expected by ``CREATE ONLINE FEATURE TABLE ... FROM SPECIFICATION``.
"""

import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class FSColumn(BaseModel):
    """A single column with name and string-based type information.

    The ``default`` field is required when adding a new output column to
    a feature view that has already been deployed — it tells downstream
    consumers what value to use for historical rows that predate the column.
    """

    name: str
    type: str  # FSBaseType value or alias — normalized during compilation
    length: Optional[int] = None  # For StringType
    precision: Optional[int] = None  # For DecimalType
    scale: Optional[int] = None  # For DecimalType
    tz: Optional[str] = None  # For TimestampType
    default: Optional[Any] = None  # Backfill value for new columns on deployed views


class SpecBase(BaseModel):
    """Common fields shared by all top-level spec kinds."""

    kind: str = ""
    name: str = ""
    version: Optional[str] = None
    database: Optional[str] = None
    schema_: Optional[str] = None
    description: Optional[str] = None


class Entity(SpecBase):
    """Top-level Entity definition."""

    kind: str = "Entity"
    join_keys: list[FSColumn] = []


class StreamingSource(SpecBase):
    """Top-level StreamingSource definition (real-time REST push source).

    Backfill is no longer attached to the source.  Use the FV-level
    :class:`Backfill` block on :class:`FeatureView` to wire
    ``StreamConfig.backfill_df`` and ``StreamConfig.backfill_start_time``.

    A YAML or dict carrying the legacy ``backfill_table`` key raises a
    ``ValidationError`` with an actionable migration message.
    """

    kind: str = "StreamingSource"
    type: str = "REST"
    columns: list[FSColumn] = []

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_backfill_table(cls, data: Any) -> Any:
        if isinstance(data, dict) and "backfill_table" in data:
            raise ValueError(
                "StreamingSource.backfill_table has been removed; backfill is "
                "now an FV-level concern that mirrors the imperative API. "
                "Move this value onto the dependent FeatureView as "
                "`backfill: { table: <FQN> }` (and add `start_time:` if you "
                "need StreamConfig.backfill_start_time)."
            )
        return data


class BatchSource(SpecBase):
    """Top-level BatchSource definition (offline batch source).

    A ``BatchSource`` describes the data the ``BatchFeatureView`` reads from.
    Exactly one of these three fields must be set:

    * ``table`` — fully qualified or bare table name; resolved via
      ``session.table(...)`` at apply time.
    * ``query`` — inline SQL string; resolved via ``session.sql(query)``.
      Whitespace is normalized at compile time so the recovered DT body
      compares equal across re-applies.
    * ``query_file`` — path (relative to the spec file) to a sibling
      ``.sql`` file; the compiler reads it and inlines the contents into
      ``query``. Mirrors the ``udf.file`` sidecar convention.

    ``source_database`` and ``source_schema`` identify where ``table``
    lives and may differ from the deployment ``database``/``schema_``
    inherited from SpecBase. They are unused for ``query`` / ``query_file``
    sources, where any qualification is part of the SQL itself.
    """

    kind: str = "BatchSource"
    source_database: Optional[str] = None
    source_schema: Optional[str] = None
    table: Optional[str] = None
    query: Optional[str] = None
    query_file: Optional[str] = None
    columns: list[FSColumn] = []

    @model_validator(mode="after")
    def _validate_exactly_one_source(self) -> "BatchSource":
        present = [field for field in ("table", "query", "query_file") if getattr(self, field)]
        if len(present) == 0:
            raise ValueError(
                f"BatchSource '{self.name}' must set exactly one of " "{table, query, query_file}; none were provided."
            )
        if len(present) > 1:
            raise ValueError(
                f"BatchSource '{self.name}' must set exactly one of "
                "{table, query, query_file}; got "
                f"{', '.join(sorted(present))}."
            )
        return self


class SourceRef(BaseModel):
    """A reference to a named source within a feature view.

    Authoring YAML usually lists only ``name`` and ``source_type``; the
    declarative ``resolve_datasource_columns`` pass copies ``columns`` plus
    ``table`` / ``query`` / ``source_database`` / ``source_schema`` from the
    matching top-level datasource spec so :func:`decl.imperative_executor._build_feature_df`
    can resolve ``session.table(...)`` without duplicating location fields on
    every feature view.
    """

    name: str
    source_type: str  # SourceType value
    columns: list[FSColumn] = Field(default_factory=list)
    table: Optional[str] = None
    query: Optional[str] = None
    source_database: Optional[str] = None
    source_schema: Optional[str] = None


class UDF(BaseModel):
    """User-defined function configuration.

    In the authoring format, ``file`` points to a co-located Python file
    and ``name`` identifies the function within it. During compilation
    the CLI reads the file and inlines the source as ``function_definition``.
    """

    name: str
    engine: str = "pandas"
    file: Optional[str] = None  # Path to .py file (YAML authoring)
    function_definition: Optional[Union[str, Any]] = None  # String source or callable
    output_columns: list[FSColumn] = []


class Feature(BaseModel):
    """A feature mapping from source column to output column.

    Optionally includes an aggregation function and time window.
    Durations use human-friendly strings in the authoring format
    (e.g. ``"5m"``, ``"1h"``) and are normalized to ``_sec`` integers
    during compilation.

    The ``_sec`` integer variants (``window_sec`` / ``offset_sec``) are
    accepted directly so a freshly-exported YAML — which carries the
    seconds form returned by ``DESCRIBE ... TYPE = SPECIFICATION`` —
    survives the round-trip through ``model_validate`` / ``model_dump``
    without losing the duration information.
    """

    source_column: FSColumn = FSColumn(name="", type="")
    output_column: FSColumn = FSColumn(name="", type="")
    function: Optional[str] = None
    window: Optional[Union[str, int]] = None  # Authoring: "5m", "1h"
    offset: Optional[Union[str, int]] = None  # Authoring: "1m"
    window_sec: Optional[int] = None  # Imperative-shape (DESCRIBE / exporter)
    offset_sec: Optional[int] = None  # Imperative-shape (DESCRIBE / exporter)
    function_params: Optional[dict[str, Any]] = None


class Backfill(BaseModel):
    """FeatureView-level backfill configuration.

    Mirrors the imperative API's two backfill surfaces in a single
    declarative block:

    * **Streaming** (``kind: StreamingFeatureView``) — ``table`` resolves to
      ``StreamConfig.backfill_df`` (via ``session.table(<table>)``) and
      ``start_time`` is forwarded as ``StreamConfig.backfill_start_time``.
      When ``table`` is omitted the imperative executor still synthesises a
      single typed-sentinel row from the source's declared ``columns`` so
      streaming registration works for REST sources without history.
    * **Batch** (``kind: BatchFeatureView``) — ``overwrite`` is forwarded to
      ``FeatureStore.register_feature_view(overwrite=...)`` and
      ``initialize`` to ``FeatureView(initialize=...)`` so authors can opt
      into the imperative ``ON_CREATE`` / ``ON_SCHEDULE`` semantics.

    Cross-field-kind validation lives on :class:`FeatureView` so the
    error message can name the FV kind that produced the misuse.
    """

    table: Optional[str] = None
    start_time: Optional[Union[datetime.datetime, str]] = None
    overwrite: Optional[bool] = None
    initialize: Optional[Literal["ON_CREATE", "ON_SCHEDULE"]] = None


class StorageConfig(BaseModel):
    """Storage-backend configuration for a managed Feature View.

    Mirrors the imperative ``snowflake.ml.feature_store.feature_view.StorageConfig``
    dataclass in YAML authoring form.  ``format`` is the only required
    key; ``external_volume`` / ``base_location`` are only meaningful when
    ``format: iceberg``.  Iceberg-without-external_volume validation
    fires from :mod:`invariants` (``BATCH_FV_STORAGE_ICEBERG_NO_VOLUME``)
    rather than here so the error message can name the offending FV.
    """

    format: Literal["snowflake", "iceberg"] = "snowflake"
    external_volume: Optional[str] = None
    base_location: Optional[str] = None


class FeatureView(SpecBase):
    """Top-level FeatureView definition.

    The ``sources`` list accepts ``SourceRef`` name-reference dicts or
    direct source-object dicts. Source objects are resolved to ``SourceRef``
    dicts during compilation.

    ``entities`` accepts plain column-name strings or full :class:`Entity`
    objects (the serializer expands ``Entity`` objects into their join-key
    column names).  The field name matches the imperative
    ``snowflake.ml.feature_store.FeatureView`` constructor's ``entities``
    kwarg.

    ``timestamp_col`` is the name of the timestamp column for point-in-time
    lookup.  Matches the imperative ``FeatureView(timestamp_col=...)`` kwarg.

    ``backfill`` is an FV-level :class:`Backfill` block; see that class for
    the streaming / batch field map.

    The legacy authoring keys ``ordered_entity_column_names`` and
    ``timestamp_field`` are rejected at load time with a migration error
    so authors get a clear pointer instead of a silently-dropped field.
    """

    kind: str = "StreamingFeatureView"
    online: bool = False
    offline: bool = False
    timestamp_col: Optional[str] = None
    feature_granularity: Optional[Union[str, int]] = None
    feature_granularity_sec: Optional[int] = None  # Imperative-shape (DESCRIBE / exporter)
    feature_aggregation_method: Optional[str] = None
    target_lag: Optional[Union[str, int]] = None
    target_lag_sec: Optional[int] = None  # Imperative-shape (DESCRIBE / exporter)
    refresh_freq: Optional[str] = None
    entities: list[Any] = []  # str or Entity
    sources: list[Any] = []  # SourceRef or StreamingSource/BatchSource
    udf: Optional[UDF] = None
    features: list[Feature] = []
    backfill: Optional[Backfill] = None
    # Advanced BFV authoring knobs that map 1:1 onto kwargs of the imperative
    # ``snowflake.ml.feature_store.FeatureView`` constructor.  ``warehouse``
    # is operational (``UPDATE_FV`` via ``FeatureStore.update_feature_view``);
    # ``cluster_by`` is structural (RECREATE_FV).  Remaining slots
    # (refresh_mode / initialize / storage_config / aggregation_secondary_keys)
    # land in Phases 3-6 as each field's TDD pass ships.
    warehouse: Optional[str] = None
    cluster_by: Optional[list[str]] = None
    refresh_mode: Optional[str] = None
    # Promoted from ``backfill.initialize`` to first-class top-level in
    # Phase 4 of the advanced BFV plan.  The legacy nested form is still
    # accepted by ``compile_to_spec`` (back-compat alias); when both are
    # set, the top-level value wins so the compiled-spec hash has a
    # single source of truth.  Anything other than the two canonical enum
    # values raises ``ValidationError`` at load time.
    initialize: Optional[Literal["ON_CREATE", "ON_SCHEDULE"]] = None
    storage_config: Optional[StorageConfig] = None
    # ``aggregation_secondary_keys`` is private-preview; valid on both
    # tiled and non-tiled BFVs (on a non-tiled FV it is a spec/OFT
    # identity column, not a no-op).  The only authoring constraint — the
    # length-1 cap — is enforced by
    # ``invariants._check_batch_feature_view_constraints`` so the error
    # message can name the offending FV.  Default ``None`` keeps the
    # field out of the compiled ``spec`` for the common case.
    aggregation_secondary_keys: Optional[list[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_authoring_keys(cls, data: Any) -> Any:
        """Reject the two legacy authoring keys with a migration message.

        The authoring surface was realigned with the imperative
        ``snowflake.ml.feature_store.FeatureView`` constructor:

        * ``ordered_entity_column_names`` -> ``entities``
        * ``timestamp_field``             -> ``timestamp_col``

        The stable substring ``"has been renamed to"`` in the error
        message is also what :func:`decl.loader._dict_to_spec` greps for
        to decide that a migration error must propagate to the user
        instead of being silently degraded to a bare ``SpecBase``.

        Args:
            data: Raw input passed to ``model_validate`` (a YAML- or
                JSON-loaded dict, or kwargs forwarded from a Pydantic
                model constructor).

        Returns:
            The input ``data`` unchanged when no legacy keys are present.

        Raises:
            ValueError: If the input carries ``ordered_entity_column_names``
                or ``timestamp_field``, with a migration message naming the
                new authoring key.
        """
        if not isinstance(data, dict):
            return data
        if "ordered_entity_column_names" in data:
            raise ValueError(
                "FeatureView field 'ordered_entity_column_names' has been "
                "renamed to 'entities' to match the imperative "
                "FeatureView(...) constructor. Update your YAML / Python "
                "authoring (this is a hard rename — there is no alias)."
            )
        if "timestamp_field" in data:
            raise ValueError(
                "FeatureView field 'timestamp_field' has been renamed to "
                "'timestamp_col' to match the imperative FeatureView(...) "
                "constructor. Update your YAML / Python authoring (this "
                "is a hard rename — there is no alias)."
            )
        if "batch_schedule" in data:
            raise ValueError(
                "FeatureView field 'batch_schedule' has been renamed to "
                "'refresh_freq' to match the imperative FeatureView(...) "
                "constructor. Update your YAML / Python authoring (this "
                "is a hard rename — there is no alias). Note that "
                "refresh_freq is only valid where it drives an offline "
                "Dynamic Table cadence: BatchFeatureView and tiled "
                "StreamingFeatureView. It is rejected on non-tiled "
                "StreamingFeatureView and RealtimeFeatureView (those run "
                "at zero target lag); remove the field entirely there."
            )
        return data

    @model_validator(mode="after")
    def _validate_backfill_against_kind(self) -> "FeatureView":
        b = self.backfill
        if b is None:
            return self
        kind = self.kind or ""
        is_streaming = "Streaming" in kind or "Realtime" in kind
        is_batch = kind == "BatchFeatureView"
        # Streaming: reject the batch-only fields.  Messages MUST contain
        # ``"is not valid on"`` so ``loader._dict_to_spec`` re-raises instead
        # of degrading to a bare SpecBase.
        if is_streaming:
            if b.overwrite is not None and b.overwrite is not False:
                raise ValueError(
                    f"FeatureView '{self.name}' (kind={kind}): "
                    f"backfill.overwrite is not valid on {kind} "
                    "(it maps to FeatureStore.register_feature_view(overwrite=...) "
                    "on BatchFeatureView). "
                    "Streaming feature views use StreamConfig lifecycle instead."
                )
            if b.initialize is not None:
                raise ValueError(
                    f"FeatureView '{self.name}' (kind={kind}): "
                    f"backfill.initialize is not valid on {kind} "
                    "(it maps to FeatureView(initialize=...) on BatchFeatureView). "
                    "Streaming feature views use StreamConfig lifecycle instead."
                )
        # Batch: reject the streaming-only fields.
        if is_batch:
            if b.table is not None:
                raise ValueError(
                    f"FeatureView '{self.name}' (kind={kind}): "
                    f"backfill.table is not valid on {kind} "
                    "(it maps to StreamConfig.backfill_df via session.table(...) "
                    "on StreamingFeatureView). "
                    "Batch feature views read history from their declared sources."
                )
            if b.start_time is not None:
                raise ValueError(
                    f"FeatureView '{self.name}' (kind={kind}): "
                    f"backfill.start_time is not valid on {kind} "
                    "(it maps to StreamConfig.backfill_start_time on "
                    "StreamingFeatureView). "
                    "Batch feature views derive freshness from their refresh cadence."
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def _enforce_always_online_for_stream_or_realtime(cls, data: Any) -> Any:
        """Streaming and realtime feature views are always online by design.

        The Snowflake runtime materialises an Online Feature Table for
        every deployed ``StreamingFeatureView`` / ``RealtimeFeatureView``
        instance, so an explicit ``online=False`` is a contradiction
        and an explicit ``online=True`` is redundant.  This validator
        therefore:

        * Defaults ``online`` to ``True`` when the authoring dict omits
          the field on a streaming / realtime kind.
        * Rejects an explicit ``online=False`` on those kinds with a
          message naming the FV and the phrase
          ``"always online by design"`` so authors get an actionable
          error instead of a silently-flipped intent.

        ``BatchFeatureView`` keeps the legacy ``online: bool = False``
        default — batch FVs can legitimately be offline-only and the
        field stays first-class for that kind.

        Resolves the ``kind`` discriminator from the input dict first
        and falls back to the model's class-level default so the rule
        applies uniformly across all three authoring entry points:
        direct subclass construction (``StreamingFeatureView(...)``),
        the YAML / Python loader's dispatched
        ``cls.model_validate(...)``, and a defensive
        ``FeatureView.model_validate({"kind": ..., ...})`` call.

        Args:
            data: Raw input passed to ``model_validate`` (a YAML- or
                JSON-loaded dict, or kwargs forwarded from a Pydantic
                model constructor).

        Returns:
            The input ``data`` (possibly updated with ``online=True``)
            when the kind is not streaming / realtime, or when the
            field was correctly defaulted / left explicit-True.

        Raises:
            ValueError: When the kind is streaming / realtime and the
                input carries an explicit ``online=False``.
        """
        if not isinstance(data, dict):
            return data
        kind = data.get("kind")
        if kind is None:
            kind_field = cls.model_fields.get("kind")
            if kind_field is not None:
                kind = kind_field.default
        kind = kind or ""
        is_stream_or_realtime = "Streaming" in kind or "Realtime" in kind
        if not is_stream_or_realtime:
            return data
        if "online" not in data:
            data = dict(data)
            data["online"] = True
            return data
        if data.get("online") is False:
            name = data.get("name", "<unnamed>")
            raise ValueError(
                f"FeatureView '{name}' (kind={kind}): online=False is "
                f"not valid on {kind} — streaming and realtime feature "
                "views are always online by design. Remove the field "
                "or set it to True."
            )
        return data

    @model_validator(mode="after")
    def _reject_target_lag_on_stream_or_realtime(self) -> "FeatureView":
        """Reject ``target_lag`` / ``target_lag_sec`` on streaming and
        realtime feature views.

        Streaming and realtime FVs always run at 0 seconds target lag —
        the Snowflake runtime enforces this and stamps
        ``target_lag_sec: 0`` onto the deployed
        ``DESCRIBE … TYPE = SPECIFICATION`` payload regardless of the
        authored value (see ``_RUNTIME_STAMPED_SPEC_KEYS`` in
        ``invariants.py``).  Accepting an authored value on these kinds
        is operator-confusing because the value is silently dropped at
        deploy time, so the declarative surface raises here instead.

        Strict semantics: even an explicit ``0`` is rejected so authors
        must remove the keys entirely.  The exporter is responsible for
        omitting both keys from streaming / realtime YAML to keep
        ``snow feature init`` → re-apply round-trips clean (see
        ``exporter.py``).

        Mirrors the kind predicate used by
        :meth:`_validate_backfill_against_kind`.

        Returns:
            ``self`` when the validator predicate does not apply (batch
            kind) or when neither ``target_lag`` nor ``target_lag_sec``
            is supplied on a streaming / realtime kind.

        Raises:
            ValueError: When ``target_lag`` or ``target_lag_sec`` is
                supplied (including an explicit ``0``) on a
                ``StreamingFeatureView`` or ``RealtimeFeatureView``.
        """
        kind = self.kind or ""
        is_stream_or_realtime = "Streaming" in kind or "Realtime" in kind
        if not is_stream_or_realtime:
            return self
        for field_name in ("target_lag", "target_lag_sec"):
            if getattr(self, field_name) is not None:
                raise ValueError(
                    f"FeatureView '{self.name}' (kind={kind}): "
                    f"{field_name} is not valid on {kind} — streaming and "
                    "realtime feature views always run at 0 seconds target "
                    "lag (the Snowflake runtime enforces this and stamps "
                    "target_lag_sec=0 onto the deployed SPECIFICATION "
                    "regardless of the authored value).  Remove the field "
                    "entirely."
                )
        return self

    def _has_aggregation_windows(self) -> bool:
        """Whether this FV is tiled — i.e. any feature declares an
        aggregation window.

        Mirrors the imperative ``FeatureView.is_tiled`` and the
        compiler's ``has_windows`` check: a tiled FV materialises its
        tiles as a managed Dynamic Table.

        Returns:
            ``True`` if any feature declares an aggregation window
            (``window`` or ``window_sec``), else ``False``.
        """
        return any(
            getattr(f, "window", None) is not None or getattr(f, "window_sec", None) is not None for f in self.features
        )

    @model_validator(mode="after")
    def _reject_refresh_freq_on_stream_or_realtime(self) -> "FeatureView":
        """Reject ``refresh_freq`` where it has no offline Dynamic Table to
        schedule: non-tiled streaming and all realtime feature views.

        ``refresh_freq`` controls the offline Dynamic Table's refresh
        cadence (the ``CREATE DYNAMIC TABLE … TARGET_LAG`` / ``SCHEDULE``
        clause); it maps 1:1 to the imperative
        ``FeatureView(refresh_freq=...)`` constructor kwarg.

        A **non-tiled** streaming FV materialises to a zero-lag VIEW and a
        realtime FV computes on demand at lookup time — neither has an
        offline DT to schedule, and the Snowflake runtime stamps
        ``target_lag_sec=0`` onto the deployed SPECIFICATION regardless of
        any authored cadence, so the field is silently dropped at deploy
        time.  The declarative surface rejects it at load time on those
        kinds so operators get a clear pointer rather than a
        silently-dropped value.  Strict semantics: even an explicit
        ``"0 seconds"`` is rejected.

        A **tiled** streaming FV (aggregation windows) is the exception:
        its offline object is a managed tile Dynamic Table whose
        ``TARGET_LAG`` is ``refresh_freq`` — the imperative
        ``FeatureView._validate`` *requires* it — so it is accepted here.
        The *requirement* that a tiled streaming FV carry ``refresh_freq``
        is enforced by the ``STREAM_FV_TILING_REFRESH`` invariant, not
        this validator.

        Returns:
            ``self`` when the validator predicate does not apply (batch
            kind, or tiled streaming), or when ``refresh_freq`` is unset.

        Raises:
            ValueError: When ``refresh_freq`` is supplied on a non-tiled
                ``StreamingFeatureView`` or any ``RealtimeFeatureView``.
        """
        kind = self.kind or ""
        is_realtime = "Realtime" in kind
        is_streaming = "Streaming" in kind
        if not (is_streaming or is_realtime):
            return self
        # Tiled streaming FVs schedule an offline tile Dynamic Table and
        # therefore legitimately carry refresh_freq.
        if is_streaming and self._has_aggregation_windows():
            return self
        if self.refresh_freq is not None:
            offline_object = "computes on demand at lookup time" if is_realtime else "materialises to a zero-lag VIEW"
            raise ValueError(
                f"FeatureView '{self.name}' (kind={kind}): "
                f"refresh_freq is not valid on {kind} — a non-tiled "
                f"streaming / realtime feature view {offline_object} and "
                "runs at 0 seconds target lag (the Snowflake runtime "
                "stamps target_lag_sec=0 regardless of any authored "
                "cadence). refresh_freq controls the offline Dynamic Table "
                "refresh cadence and is only meaningful on BatchFeatureView "
                "and tiled StreamingFeatureView. Remove the field entirely."
            )
        return self

    @model_validator(mode="after")
    def _reject_target_lag_on_offline_batch_fv(self) -> "FeatureView":
        """Reject ``target_lag`` / ``target_lag_sec`` on offline-only batch FVs.

        After the ``feature_granularity`` / ``refresh_freq`` /
        ``target_lag`` decoupling, authoring ``target_lag`` is OFT
        staleness only — strictly the value the imperative
        ``OnlineConfig.target_lag`` receives.  An offline-only
        ``BatchFeatureView`` (``online: False``) has no Online Feature
        Table, so any authored ``target_lag`` / ``target_lag_sec`` is
        meaningless.  The DT refresh cadence is authored separately as
        ``refresh_freq``.

        Streaming / realtime kinds are handled by the sibling
        ``_reject_target_lag_on_stream_or_realtime`` validator (they
        reject the field unconditionally).  This validator only fires
        for ``BatchFeatureView`` instances with ``online: False``.

        Returns:
            ``self`` when the validator predicate does not apply
            (online batch FV, or non-batch kind), or when neither
            ``target_lag`` nor ``target_lag_sec`` is supplied.

        Raises:
            ValueError: When ``target_lag`` or ``target_lag_sec`` is
                supplied on an offline-only ``BatchFeatureView``.
        """
        kind = self.kind or ""
        if kind != "BatchFeatureView":
            return self
        if self.online:
            return self
        for field_name in ("target_lag", "target_lag_sec"):
            if getattr(self, field_name) is not None:
                raise ValueError(
                    f"FeatureView '{self.name}' (kind={kind}, online=False): "
                    f"{field_name} is OFT staleness (online-only) and is not "
                    "valid on an offline-only BatchFeatureView.  Use "
                    "``refresh_freq`` for the offline Dynamic Table refresh "
                    "cadence, or set ``online: true`` if you intended the "
                    "value as Online Feature Table TARGET_LAG."
                )
        return self

    @model_validator(mode="after")
    def _reject_unsupported_refresh_mode(self) -> "FeatureView":
        """Reject unsupported ``refresh_mode`` values with an actionable message.

        ``refresh_mode`` accepts only ``"FULL"`` or ``"INCREMENTAL"``.  Any other
        value (e.g. ``"AUTO"``) must produce a propagated ``ValidationError``
        whose message contains ``"is not valid on"`` so that
        ``loader._dict_to_spec`` re-raises it rather than falling back to the
        sparse ``SpecBase`` fallback.  Omitting the field entirely is the
        correct way to let Snowflake choose the refresh strategy automatically.

        Returns:
            ``self`` when ``refresh_mode`` is ``None``, ``"FULL"``, or
            ``"INCREMENTAL"``.

        Raises:
            ValueError: When ``refresh_mode`` is set to any other value.
        """
        rm = self.refresh_mode
        if rm is not None and str(rm).upper() not in ("FULL", "INCREMENTAL"):
            raise ValueError(
                f"refresh_mode '{rm}' is not valid on FeatureView; "
                "omit the field to let Snowflake choose automatically, "
                "or set it to 'FULL' or 'INCREMENTAL' to pin a refresh strategy."
            )
        return self


class StreamingFeatureView(FeatureView):
    """``FeatureView`` subclass pinning ``kind`` to ``"StreamingFeatureView"``.

    The Python authoring form discriminates feature-view kinds by class
    type (see ``plans/python_form/python_authoring_form.md`` Q2 / Q8):
    authors write ``StreamingFeatureView(...)`` instead of
    ``FeatureView(kind="StreamingFeatureView", ...)``.  Every inherited
    field, validator, and downstream consumer behaves identically to
    the base class — the only difference is that ``kind`` is pre-set
    at the class level.

    ``loader._dict_to_spec`` also maps YAML / JSON dicts carrying
    ``kind: "StreamingFeatureView"`` to this subclass so the
    ``isinstance`` chain is uniform across all three authoring formats.

    ``online`` defaults to ``True`` here (and the base
    ``FeatureView._enforce_always_online_for_stream_or_realtime``
    validator rejects an explicit ``online=False``) because a streaming
    FV is always backed by an Online Feature Table — see that
    validator's docstring for the full contract.
    """

    kind: str = "StreamingFeatureView"
    online: bool = True


class BatchFeatureView(FeatureView):
    """``FeatureView`` subclass pinning ``kind`` to ``"BatchFeatureView"``.

    See :class:`StreamingFeatureView` for the discrimination contract;
    the same rules apply here.  Authors write ``BatchFeatureView(...)``
    and the loader / serializer treats it identically to a base
    ``FeatureView`` carrying ``kind="BatchFeatureView"``.
    """

    kind: str = "BatchFeatureView"


class RealtimeFeatureView(FeatureView):
    """``FeatureView`` subclass pinning ``kind`` to ``"RealtimeFeatureView"``.

    See :class:`StreamingFeatureView` for the discrimination contract;
    the same rules apply here.  Authors write ``RealtimeFeatureView(...)``
    and the loader / serializer treats it identically to a base
    ``FeatureView`` carrying ``kind="RealtimeFeatureView"``.

    ``online`` defaults to ``True`` here (and the base
    ``FeatureView._enforce_always_online_for_stream_or_realtime``
    validator rejects an explicit ``online=False``) because a realtime
    FV is always backed by an Online Feature Table.
    """

    kind: str = "RealtimeFeatureView"
    online: bool = True


class FeatureViewRef(BaseModel):
    """A name-and-version reference to a feature view within a feature group.

    Mirrors the imperative ``FeatureGroupSourceRef`` shape so a declarative
    YAML round-trips through ``FeatureStore.list_feature_groups()`` /
    ``register_feature_group(...)`` without translation.

    Fields:
        name: Source ``FeatureView`` name.
        version: Source ``FeatureView`` version.  Required — feature groups
            pin a specific version of each member FV.
        slice_columns: Optional subset of the source FV's output columns.
            ``None`` means "use the full FV".  ``[]`` is rejected (an empty
            slice is meaningless; use ``None`` instead).
        alias: Optional rename prefix.  ``None`` means "fall back to the
            owning ``FeatureGroup.auto_prefix`` rule".  ``""`` is preserved
            (semantically: "no prefix") and is distinct from ``None``.
    """

    name: str
    version: str
    slice_columns: Optional[list[str]] = None
    alias: Optional[str] = None

    @model_validator(mode="after")
    def _validate_feature_view_ref(self) -> "FeatureViewRef":
        if not self.name:
            raise ValueError("FeatureViewRef.name must not be empty.")
        if not self.version:
            raise ValueError(
                f"FeatureViewRef '{self.name}' is missing 'version'. "
                "FeatureGroups pin a specific FV version (mirrors the "
                "imperative FeatureGroupSourceRef shape)."
            )
        if self.slice_columns is not None and len(self.slice_columns) == 0:
            raise ValueError(
                f"FeatureViewRef '{self.name}' has an empty slice_columns "
                "list; omit the field (or set it to None) to use the full FV."
            )
        return self


class FeatureGroup(SpecBase):
    """Top-level FeatureGroup definition.

    A logical grouping of FeatureViews sharing the same schema and
    versioned together.  Imperatively this materialises as a Postgres-backed
    Online Feature Table; the declarative shape mirrors
    ``FeatureStore.register_feature_group(...)`` so a local edit / apply
    cycle is byte-identical with the imperative API.

    Fields:
        desc: Free-form description.  Default ``""`` matches snowml-core's
            ``FeatureGroup.__init__`` default.
        auto_prefix: When True (default), the source FV's name is added as
            a prefix to its output columns; per-source ``alias`` overrides.
        feature_views: Non-empty list of ``FeatureViewRef``.  ``(name,
            version)`` pairs must be unique within the list.
    """

    kind: str = "FeatureGroup"
    desc: str = ""
    auto_prefix: bool = True
    feature_views: list[FeatureViewRef] = []

    @model_validator(mode="before")
    @classmethod
    def _coerce_feature_view_objects(cls, data: Any) -> Any:
        """Accept ``FeatureView`` Python objects in ``feature_views``.

        Q5 of ``plans/python_form/python_authoring_form.md`` makes
        cross-spec references equally first-class for name-strings AND
        Python objects.  An author writing
        ``FeatureGroup(feature_views=[fv1, fv2])`` with FV instances
        must succeed: this validator collapses any ``FeatureView``
        instance to a ``FeatureViewRef`` with the matching ``name``
        (and ``version`` if the FV carries one) BEFORE Pydantic
        validates the field's typed ``list[FeatureViewRef]`` schema.

        The validator is a no-op for dict input (the YAML / JSON path)
        and for input that already carries ``FeatureViewRef`` instances.

        Args:
            data: Either a raw dict (YAML / JSON path) or a kwargs
                mapping (``model_validate``-from-kwargs path).

        Returns:
            The same data with any ``FeatureView`` instance under the
            ``feature_views`` key collapsed to a ``FeatureViewRef``.
        """
        if not isinstance(data, dict):
            return data
        fvs = data.get("feature_views")
        if not isinstance(fvs, list):
            return data
        coerced: list[Any] = []
        for item in fvs:
            if isinstance(item, FeatureView):
                ref: dict[str, Any] = {"name": item.name}
                version = getattr(item, "version", None)
                if version:
                    ref["version"] = version
                coerced.append(FeatureViewRef.model_validate(ref))
            else:
                coerced.append(item)
        data["feature_views"] = coerced
        return data

    @model_validator(mode="after")
    def _validate_feature_group(self) -> "FeatureGroup":
        if self.name and "$" in self.name:
            raise ValueError(
                f"FeatureGroup name '{self.name}' must not contain '$' "
                "(reserved as the Snowflake-side name/version delimiter)."
            )
        if not self.feature_views:
            raise ValueError(
                f"FeatureGroup '{self.name or '<unnamed>'}' must reference "
                "at least one FeatureView in 'feature_views'."
            )
        seen: set[tuple[str, str]] = set()
        for ref in self.feature_views:
            key = (ref.name, ref.version)
            if key in seen:
                raise ValueError(
                    f"FeatureGroup '{self.name or '<unnamed>'}' references "
                    f"FeatureView ({ref.name}, {ref.version}) more than once. "
                    "Each (name, version) pair must be unique."
                )
            seen.add(key)
        return self
