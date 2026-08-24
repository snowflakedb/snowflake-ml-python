"""Tests for the ``decl.api.validate_specs`` facade.

Pins the contract that the public facade in :mod:`decl.api` forwards
its ``dev_mode`` kwarg into :func:`decl.invariants.validate_specs`.

The CLI manager goes through this facade only — never directly into
``decl.invariants`` — so the facade is the single seam where ``dev_mode``
plumbing must be respected.  A regression here surfaces as
``snow feature plan --dev`` still flagging ``MISSING_VERSION`` errors,
because the api wrapper silently dropped the kwarg.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from snowflake.ml.feature_store.decl import api as decl_api


class TestApiValidateSpecsForwardsDevMode:
    """Pin the api -> invariants forwarding contract for dev_mode."""

    def test_forwards_dev_mode_true(self) -> None:
        with patch(
            "snowflake.ml.feature_store.decl.invariants.validate_specs",
            return_value=[],
        ) as mock_validate:
            decl_api.validate_specs(
                batch=None,  # type: ignore[arg-type]
                applied_state=None,  # type: ignore[arg-type]
                target_database="DB",
                target_schema="SCH",
                dev_mode=True,
            )

        kwargs = mock_validate.call_args.kwargs
        assert kwargs.get("dev_mode") is True, (
            "api.validate_specs must forward dev_mode=True into the " f"invariants module; got kwargs={kwargs!r}"
        )

    def test_forwards_dev_mode_false(self) -> None:
        with patch(
            "snowflake.ml.feature_store.decl.invariants.validate_specs",
            return_value=[],
        ) as mock_validate:
            decl_api.validate_specs(
                batch=None,  # type: ignore[arg-type]
                applied_state=None,  # type: ignore[arg-type]
                target_database="DB",
                target_schema="SCH",
                dev_mode=False,
            )

        kwargs = mock_validate.call_args.kwargs
        assert kwargs.get("dev_mode") is False, (
            "api.validate_specs must forward dev_mode=False explicitly; " f"got kwargs={kwargs!r}"
        )

    def test_default_dev_mode_is_false(self) -> None:
        """Legacy callers (no dev_mode kwarg) must reach the invariants
        module with ``dev_mode=False`` so production validation stays
        strict by default."""
        with patch(
            "snowflake.ml.feature_store.decl.invariants.validate_specs",
            return_value=[],
        ) as mock_validate:
            decl_api.validate_specs(
                batch=None,  # type: ignore[arg-type]
                applied_state=None,  # type: ignore[arg-type]
                target_database="DB",
                target_schema="SCH",
            )

        kwargs = mock_validate.call_args.kwargs
        assert kwargs.get("dev_mode", "<unset>") is False, (
            "Backwards compatibility: omitting dev_mode must surface as "
            f"dev_mode=False inside invariants; got {kwargs!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
