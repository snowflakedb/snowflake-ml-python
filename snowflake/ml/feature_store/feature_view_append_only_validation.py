"""Append-only (snapshot accumulation) feature view validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)

if TYPE_CHECKING:
    from snowflake.ml.feature_store.feature_view import FeatureView

from snowflake.ml.feature_store import feature_view_refresh_freq


def validate_snapshot_config_for_update(feature_view: FeatureView) -> None:
    """Re-validate snapshot constraints that can change after registration.

    Called from ``update_feature_view`` paths that may mutate
    ``refresh_freq``. Only checks the constraints the update API can
    violate — that ``refresh_freq`` is set and is a cron expression. The
    remaining structural invariants (batch feature view shape, ``feature_df``,
    ``timestamp_col``, ``entities``, ``refresh_mode``) cannot change for
    an already-registered FV; they are enforced once at registration via
    :func:`validate_snapshot_config_for_register`.

    No-op for non-append_only feature views.

    Args:
        feature_view: The feature view being updated or re-validated.

    Raises:
        SnowflakeMLException: If ``refresh_freq`` is missing or not a
            cron expression for an append_only feature view.
    """
    if not feature_view._append_only:
        return

    if feature_view.refresh_freq is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Snapshot accumulation requires refresh_freq to be set (feature view cannot be a view)."
            ),
        )
    if not feature_view_refresh_freq._is_cron_refresh_freq(feature_view.refresh_freq):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Snapshot accumulation requires refresh_freq to be a cron expression, "
                f"got '{feature_view.refresh_freq}'."
            ),
        )


def validate_snapshot_config_for_register(feature_view: FeatureView, *, overwrite: bool = False) -> None:
    """Full snapshot-config validation — called at registration time.

    Verifies every invariant required for an append_only feature view: the
    structural shape (batch feature view, non-rollup, non-tiled, ``refresh_mode='FULL'``,
    non-empty ``feature_df`` / ``timestamp_col`` / ``entities``), the cron
    ``refresh_freq``, plus the ``overwrite=True`` rejection. Also validates
    that ``backup_source`` is only set when ``append_only=True``.

    Update flows should use :func:`validate_snapshot_config_for_update`
    instead — the structural invariants checked here cannot change for an
    already-registered FV, so re-running them on every update is wasted
    work and would misattribute a long-standing problem to the caller's
    most recent change.

    Args:
        feature_view: The feature view being registered.
        overwrite: Whether ``register_feature_view`` was called with
            ``overwrite=True``. Rejected for append_only feature views.

    Raises:
        SnowflakeMLException: If any snapshot constraint is violated.
    """
    if feature_view._backup_source is not None and not feature_view._append_only:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("backup_source is not valid if append_only=False"),
        )

    if not feature_view._append_only:
        return

    if overwrite:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "append_only feature views do not support overwrite=True. "
                "Drop the feature view first with delete_feature_view(), then re-register."
            ),
        )

    if not feature_view.is_batch_feature_view:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Snapshot accumulation is only supported for batch feature views, "
                "not streaming or real-time feature views."
            ),
        )
    if feature_view.is_rollup:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Snapshot accumulation is only supported for batch feature views, not rollup feature views."
            ),
        )
    if feature_view.is_tiled:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("Snapshot accumulation is not supported for tiled feature views."),
        )
    if feature_view.refresh_mode is None or feature_view.refresh_mode.upper() != "FULL":
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                f"Snapshot accumulation requires refresh_mode='FULL', got '{feature_view.refresh_mode}'."
            ),
        )
    if feature_view.feature_df is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "Snapshot accumulation requires a feature DataFrame or SQL query "
                "so the output schema can be inferred at registration time."
            ),
        )
    if feature_view.timestamp_col is None:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("Snapshot accumulation requires timestamp_col to be specified."),
        )
    if not feature_view.entities:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError("Snapshot accumulation requires at least one entity."),
        )

    # Delegate the refresh_freq checks so the register and update gates
    # share a single source of truth for the cron contract.
    validate_snapshot_config_for_update(feature_view)
