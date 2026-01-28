import os
from typing import Callable, Optional, Union

from snowflake.ml._internal.utils.snowflake_env import SnowflakeCloudType
from snowflake.snowpark import context as sp_context

# Default value type: can be a bool or a callable that returns a bool
DefaultValue = Union[bool, Callable[[], bool]]


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


def _enabled_in_clouds(*clouds: SnowflakeCloudType) -> Callable[[], bool]:
    """Create a callable that checks if the current environment is in any of the specified clouds.

    This factory function returns a callable that can be used as a dynamic default
    for feature flags. The returned callable will check if the current Snowflake
    session is connected to a region in any of the specified cloud providers.

    Args:
        *clouds: One or more SnowflakeCloudType values to check against.

    Returns:
        A callable that returns True if running in any of the specified clouds,
        False otherwise (including when no session is available).

    Example:
        >>> # Enable feature only in GCP
        >>> default=_enabled_in_clouds(SnowflakeCloudType.GCP)
        >>>
        >>> # Enable feature in both GCP and Azure
        >>> default=_enabled_in_clouds(SnowflakeCloudType.GCP, SnowflakeCloudType.AZURE)
    """
    cloud_set = frozenset(clouds)

    def check() -> bool:
        try:
            from snowflake.ml._internal.utils.snowflake_env import get_current_cloud

            session = sp_context.get_active_session()
            current_cloud = get_current_cloud(session, default=SnowflakeCloudType.AWS)
            return current_cloud in cloud_set
        except Exception:
            # If we can't determine the cloud (no session, SQL error, etc.),
            # default to False for safety
            return False

    return check


class _FeatureFlag:
    """A feature flag backed by an environment variable with a configurable default.

    The default value can be a constant boolean or a callable that dynamically
    determines the default based on runtime context (e.g., cloud provider).
    """

    def __init__(self, env_var: str, default: DefaultValue = False) -> None:
        """Initialize a feature flag.

        Args:
            env_var: The environment variable name that controls this flag.
            default: The default value when the env var is not set. Can be:
                - A boolean constant (True/False)
                - A callable that returns a boolean (evaluated at check time)
        """
        self._env_var = env_var
        self._default = default

    @property
    def value(self) -> str:
        """Return the environment variable name (for compatibility with Enum-style access)."""
        return self._env_var

    def _get_default(self) -> bool:
        """Get the default value, calling it if it's a callable."""
        if callable(self._default):
            return self._default()
        return self._default

    def is_enabled(self) -> bool:
        """Check if the feature flag is enabled.

        First checks the environment variable. If not set or unrecognized,
        falls back to the configured default value.

        Returns:
            True if the feature is enabled, False otherwise.
        """
        env_value = os.getenv(self._env_var)
        if env_value is not None:
            # Environment variable is set, parse it
            result = parse_bool_env_value(env_value, default=self._get_default())
            return result
        else:
            # Environment variable not set, use the default
            return self._get_default()

    def __str__(self) -> str:
        return self._env_var


class FeatureFlags:
    """Collection of feature flags for ML Jobs."""

    ENABLE_RUNTIME_VERSIONS = _FeatureFlag("MLRS_ENABLE_RUNTIME_VERSIONS", default=True)
    ENABLE_STAGE_MOUNT_V2 = _FeatureFlag(
        "MLRS_ENABLE_STAGE_MOUNT_V2",
        default=_enabled_in_clouds(SnowflakeCloudType.GCP),
    )
