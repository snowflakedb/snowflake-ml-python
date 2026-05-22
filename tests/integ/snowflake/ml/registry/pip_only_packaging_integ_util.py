"""Helpers for pip-only packaging integration tests before account-wide launch."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from snowflake.ml._internal import platform_capabilities as pc


@contextmanager
def force_enable_pip_only_packaging() -> Iterator[None]:
    """Force ``ENABLE_PIP_ONLY_PACKAGING`` until the platform parameter is launched."""
    with pc.PlatformCapabilities.mock_features({pc.ENABLE_PIP_ONLY_PACKAGING: True}):
        yield


class PipOnlyPackagingIntegMixin:
    """Mixin that mocks pip-only packaging for registry integ tests."""

    def _test_registry_model_deployment(self, *args: Any, **kwargs: Any) -> Any:
        with force_enable_pip_only_packaging():
            return super()._test_registry_model_deployment(*args, **kwargs)  # type: ignore[misc]

    def _test_registry_batch_inference(self, *args: Any, **kwargs: Any) -> Any:
        with force_enable_pip_only_packaging():
            return super()._test_registry_batch_inference(*args, **kwargs)  # type: ignore[misc]

    def _test_registry_model_target_platforms(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        with force_enable_pip_only_packaging():
            super()._test_registry_model_target_platforms(*args, **kwargs)  # type: ignore[misc]
