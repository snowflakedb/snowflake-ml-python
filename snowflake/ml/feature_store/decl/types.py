"""Pydantic v2 models for the planning and execution pipeline.

All types in this module serialize cleanly to JSON via ``model_dump()``.
They carry no connection handles, cursors, or runtime state — this makes
them suitable as REST request/response bodies when GS capabilities are added.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.spec_models import FSColumn, SpecBase


class ObjectKind:
    """Canonical applied-object kinds.

    These string constants are the values that appear in :class:`AppliedObject`
    ``kind`` fields and the ``type`` column of ``snow feature list``.  Existing
    feature-view subkinds (``StreamingFeatureView``, ``RealtimeFeatureView``,
    ``BatchFeatureView``) are still produced by the planner and continue to
    work alongside these constants.
    """

    FEATURE_VIEW = "FeatureView"
    ENTITY = "Entity"
    DATASOURCE = "Datasource"
    FEATURE_GROUP = "FeatureGroup"


class AppliedObject(BaseModel):
    """A single object as it exists in Snowflake (applied state snapshot).

    Attributes:
        key: Unique identifier in the form ``kind:database.schema:name``.
        kind: Object kind string (e.g. ``"Entity"``, ``"FeatureView"``,
            ``"Datasource"``, or one of the legacy feature-view subkinds
            ``"StreamingFeatureView"`` / ``"RealtimeFeatureView"`` /
            ``"BatchFeatureView"``).
        name: Object name.
        version: Deployed version string, or ``None`` if not versioned.
        content_hash: SHA-256 of the deployed spec (for idempotency checks).
        spec_payload: The full spec as deployed (reconstructed from Snowflake).
        columns: Column schemas for column-evolution comparison.
        from_specification: ``True`` when ``spec_payload`` came from the
            ``DESCRIBE ... TYPE = SPECIFICATION`` SQL — i.e. it carries the
            full original spec JSON and is suitable for full-spec diffing.
            ``False`` when the payload was reconstructed from
            ``DESCRIBE`` columns alone (legacy structural fingerprint path).
        details: Kind-specific extras that don't fit elsewhere — e.g.
            ``join_keys`` for Entity, ``source_type`` for Datasource.
    """

    key: str
    kind: str
    name: str
    version: Optional[str] = None
    content_hash: str = ""
    spec_payload: dict[str, Any] = {}
    columns: list[FSColumn] = []
    from_specification: bool = False
    details: dict[str, Any] = {}


class AppliedState(BaseModel):
    """A point-in-time snapshot of all deployed feature store objects.

    Attributes:
        objects: Mapping from ``kind:database.schema:name`` to ``AppliedObject``.
    """

    objects: dict[str, AppliedObject] = {}


class PlanOp(BaseModel):
    """A single operation in an execution plan.

    Attributes:
        kind: The operation type (CREATE_ENTITY, UPDATE_FV, etc.).
        name: The object name this operation targets.
        depends_on: Names of objects this operation depends on.
        destructive: Whether this is a breaking/destructive change.
        reason: Human-readable explanation of why this op was generated.
        payload: The spec payload dict for this operation.
    """

    kind: OpKind
    name: str
    depends_on: list[str] = []
    destructive: bool = False
    reason: str = ""
    payload: dict[str, Any] = {}


class Plan(BaseModel):
    """An ordered, validated execution plan.

    Attributes:
        ops: Topologically-sorted list of operations to execute.
        warnings: Non-blocking warnings generated during plan creation.
    """

    ops: list[PlanOp] = []
    warnings: list[str] = []


class ValidationResult(BaseModel):
    """A single validation finding (error or warning).

    Attributes:
        severity: Either ``"ERROR"`` (blocking) or ``"WARNING"`` (non-blocking).
        code: A short machine-readable code (e.g. ``"COLUMN_MISSING_DEFAULT"``).
        message: Human-readable description of the finding.
        object_name: The name of the object that triggered the finding.
    """

    severity: Literal["ERROR", "WARNING"]
    code: str
    message: str
    object_name: str = ""


class PlanOptions(BaseModel):
    """Options that control plan generation behaviour.

    Attributes:
        overwrite: Force apply even when version or column checks fail.
        allow_recreate: Allow destructive schema changes that require re-materialization.
        full_directory_mode: Only True when ``./...`` is specified; enables deletion detection.
    """

    overwrite: bool = False
    allow_recreate: bool = False
    full_directory_mode: bool = False


class SpecBatch(BaseModel):
    """A parsed batch of specs ready for validation and planning.

    Attributes:
        specs: The parsed spec objects (``SpecBase`` subclass instances).
        source_files: The file paths that were loaded to produce this batch.
    """

    specs: list[SpecBase] = []
    source_files: list[str] = []


class PlanFile(BaseModel):
    """A serializable plan file that can be saved to disk and loaded by apply.

    Attributes:
        version: File format version (currently ``"1"``).
        created_at: ISO 8601 timestamp when the plan was generated.
        target_database: The Snowflake database the plan targets.
        target_schema: The Snowflake schema the plan targets.
        target_name: Manifest target name (e.g. ``"DEV"`` / ``"PROD"``)
            the plan was generated against. Defaults to ``""`` so plan
            files written before D4-ext (Phase 3+4) deserialise cleanly
            and continue to apply when no ``--target`` is requested.
            Apply rejects with ``status="target_mismatch"`` when this
            field is non-empty and disagrees with the requested
            ``--target`` (case-insensitive after upper()).
        source_files: Input spec file paths used to generate the plan.
        plan: The generated execution plan.
        summary: Op-kind counts (e.g. ``{"CREATE_ENTITY": 1, "CREATE_FV": 2}``).
    """

    version: str = "1"
    created_at: str = ""
    target_database: str = ""
    target_schema: str = ""
    target_name: str = ""
    source_files: list[str] = []
    plan: Plan = Plan()
    summary: dict[str, int] = {}


class ApplyResult(BaseModel):
    """Result of an imperative ``execute_plan`` call.

    Returned by ``api.execute_plan()`` so the CLI can display each
    operation's status without re-deriving it from raw FeatureStore
    return values.

    Attributes:
        status: One of ``"applied"`` (every op succeeded), ``"refused"``
            (the plan carried at least one ``destructive=True`` op and
            ``PlanOptions.allow_recreate`` was False — no op was
            executed; the operator must re-run with
            ``--allow-recreate`` to consume the same plan file under
            L5), ``"partial_failure"`` (one or more ops raised at
            execution time), or ``"validation_failed"`` (planner-side
            ERROR severities surfaced before execution).
        ops: Per-operation details for display
            (``operation`` / ``name`` / ``status`` / ...). Each op's
            ``status`` cell is one of ``"success"``, ``"skipped"``,
            ``"error"``, or ``"refused"``.
        warnings: Non-blocking warnings from planning.
        errors: Blocking errors. For ``"validation_failed"`` these are
            the planner-side ERROR severities; for ``"refused"`` it is
            a single human-readable directive naming the destructive
            op count and the ``--allow-recreate`` remediation.
    """

    status: str = "applied"
    ops: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
