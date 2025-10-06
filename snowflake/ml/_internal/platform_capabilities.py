import json
import logging
from contextlib import contextmanager
from typing import Any, Optional

from packaging import version

from snowflake.ml import version as snowml_version
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import (
    exceptions as snowpark_exceptions,
    session as snowpark_session,
)

logger = logging.getLogger(__name__)

LIVE_COMMIT_PARAMETER = "ENABLE_LIVE_VERSION_IN_SDK"
INLINE_DEPLOYMENT_SPEC_PARAMETER = "ENABLE_INLINE_DEPLOYMENT_SPEC_FROM_CLIENT_VERSION"
SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST = "SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST"


class PlatformCapabilities:
    """Class that retrieves platform feature values for the currently running server.

    Example usage:
    ```
    pc = PlatformCapabilities.get_instance(session)
    if pc.is_inlined_deployment_spec_enabled():
        # Inline deployment spec is enabled.
        print("Inline deployment spec is enabled.")
    else:
        # Inline deployment spec is disabled.
        print("Inline deployment spec is disabled or not supported.")
    ```
    """

    _instance: Optional["PlatformCapabilities"] = None
    # Used for unittesting only. This is to avoid the need to mock the session object or reaching out to Snowflake
    _mock_features: Optional[dict[str, Any]] = None

    @classmethod
    def get_instance(cls, session: Optional[snowpark_session.Session] = None) -> "PlatformCapabilities":
        # Used for unittesting only. In this situation, _instance is not initialized.
        if cls._mock_features is not None:
            return cls(features=cls._mock_features)
        if not cls._instance:
            cls._instance = cls(session=session)
        return cls._instance

    @classmethod
    def set_mock_features(cls, features: Optional[dict[str, Any]] = None) -> None:
        cls._mock_features = features

    @classmethod
    def clear_mock_features(cls) -> None:
        cls._mock_features = None

    # For contextmanager, we need to have return type Iterator[Never]. However, Never type is introduced only in
    # Python 3.11. So, we are ignoring the type for this method.
    _dummy_features: dict[str, Any] = {"dummy": "dummy"}

    @classmethod  # type: ignore[arg-type]
    @contextmanager
    def mock_features(cls, features: dict[str, Any] = _dummy_features) -> None:  # type: ignore[misc]
        logger.debug(f"Setting mock features: {features}")
        cls.set_mock_features(features)
        try:
            yield
        finally:
            logger.debug(f"Clearing mock features: {features}")
            cls.clear_mock_features()

    def is_inlined_deployment_spec_enabled(self) -> bool:
        return self._is_version_feature_enabled(INLINE_DEPLOYMENT_SPEC_PARAMETER)

    def is_set_module_functions_volatility_from_manifest(self) -> bool:
        return self._get_bool_feature(SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST, False)

    def is_live_commit_enabled(self) -> bool:
        return self._get_bool_feature(LIVE_COMMIT_PARAMETER, False)

    @staticmethod
    def _get_features(session: snowpark_session.Session) -> dict[str, Any]:
        try:
            result = (
                query_result_checker.SqlResultValidator(
                    session=session,
                    query="SELECT SYSTEM$ML_PLATFORM_CAPABILITIES() AS FEATURES;",
                )
                .has_dimensions(expected_rows=1, expected_cols=1)
                .has_column("FEATURES")
                .validate()[0]
            )
            if "FEATURES" in result:
                capabilities_json: str = result["FEATURES"]
                try:
                    parsed_json = json.loads(capabilities_json)
                    assert isinstance(parsed_json, dict), f"Expected JSON object, got {type(parsed_json)}"
                    return parsed_json
                except json.JSONDecodeError as e:
                    message = f"""Unable to parse JSON from: "{capabilities_json}"; Error="{e}"."""
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_SNOWML_ERROR, original_exception=RuntimeError(message)
                    )
        except snowpark_exceptions.SnowparkSQLException as e:
            logger.debug(f"Failed to retrieve platform capabilities: {e}")
            # This can happen is server side is older than 9.2. That is fine.
        return {}

    def __init__(
        self, *, session: Optional[snowpark_session.Session] = None, features: Optional[dict[str, Any]] = None
    ) -> None:
        # This is for testing purposes only.
        if features:
            self.features = features
            return
        if not session:
            session = next(iter(snowpark_session._get_active_sessions()))
            assert session, "Missing active session object"
        self.features = PlatformCapabilities._get_features(session)

    def _get_bool_feature(self, feature_name: str, default_value: bool) -> bool:
        value = self.features.get(feature_name, default_value)
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in [0, 1]:
            return value == 1
        if isinstance(value, str):
            if value.lower() in ["true", "1"]:
                return True
            elif value.lower() in ["false", "0"]:
                return False
            else:
                raise ValueError(f"Invalid boolean string: {value} for feature {feature_name}")
        raise ValueError(f"Invalid boolean feature value: {value} for feature {feature_name}")

    def _get_version_feature(self, feature_name: str) -> version.Version:
        """Get a version feature value, returning a large version number on failure or missing feature.

        Args:
            feature_name: The name of the feature to retrieve.

        Returns:
            version.Version: The parsed version, or a large version number (999.999.999) if parsing fails
            or the feature is missing.
        """
        # Large version number to use as fallback
        large_version = version.Version("999.999.999")

        value = self.features.get(feature_name)
        if value is None:
            logger.debug(f"Feature {feature_name} not found, returning large version number")
            return large_version

        try:
            # Convert to string if it's not already
            version_str = str(value)
            return version.Version(version_str)
        except (version.InvalidVersion, ValueError, TypeError) as e:
            logger.debug(
                f"Failed to parse version from feature {feature_name} with value '{value}': {e}. "
                f"Returning large version number"
            )
            return large_version

    def _is_version_feature_enabled(self, feature_name: str) -> bool:
        """Check if the current package version is greater than or equal to the version feature.

        Args:
            feature_name: The name of the version feature to compare against.

        Returns:
            bool: True if current package version >= feature version, False otherwise.
        """
        current_version = version.Version(snowml_version.VERSION)
        feature_version = self._get_version_feature(feature_name)

        result = current_version >= feature_version
        logger.debug(
            f"Version comparison for feature {feature_name}: "
            f"current={current_version}, feature={feature_version}, enabled={result}"
        )
        return result
