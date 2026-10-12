from unittest import mock

from snowflake.ml._internal import platform_capabilities

_PATCHER = mock.patch.object(
    platform_capabilities.PlatformCapabilities,
    "is_lora_adapters_enabled",
    return_value=True,
    autospec=True,
)
_started = False


def enable() -> None:
    """Turn on the LoRA adapters client capability for this process.

    Starts at most once so multiple LoRA integ modules can share one pytest
    process without restacking the same patch.
    """
    global _started
    if _started:
        return
    _PATCHER.start()
    _started = True
