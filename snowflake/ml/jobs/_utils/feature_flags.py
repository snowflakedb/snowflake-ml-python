import os
from enum import Enum
from typing import Optional


def parse_bool_env_value(value: Optional[str], default: bool = False) -> bool:
    """Parse a boolean value from an environment variable string.

    Args:
        value: The environment variable value to parse (may be None).
        default: The default value to return if the value is None or unrecognized.

    Returns:
        True if the value is a truthy string (true, 1, yes, on - case insensitive),
        False if the value is a falsy string (false, 0, no, off - case insensitive),
        or the default value if the value is None or unrecognized.
    """
    if value is None:
        return default

    normalized_value = value.strip().lower()
    if normalized_value in ("true", "1", "yes", "on"):
        return True
    elif normalized_value in ("false", "0", "no", "off"):
        return False
    else:
        # For unrecognized values, return the default
        return default


class FeatureFlags(Enum):
    USE_SUBMIT_JOB_V2 = "MLRS_USE_SUBMIT_JOB_V2"
    ENABLE_RUNTIME_VERSIONS = "MLRS_ENABLE_RUNTIME_VERSIONS"

    def is_enabled(self, default: bool = False) -> bool:
        """Check if the feature flag is enabled.

        Args:
            default: The default value to return if the environment variable is not set.

        Returns:
            True if the environment variable is set to a truthy value,
            False if set to a falsy value, or the default value if not set.
        """
        return parse_bool_env_value(os.getenv(self.value), default)

    def __str__(self) -> str:
        return self.value
