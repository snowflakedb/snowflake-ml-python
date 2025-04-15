import json
from contextlib import contextmanager
from typing import Any, Dict, Optional

from absl import logging

from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import query_result_checker
from snowflake.snowpark import (
    exceptions as snowpark_exceptions,
    session as snowpark_session,
)


class PlatformCapabilities:
    """Class that retrieves platform feature values for the currently running server.

    Example usage:
    ```
    pc = PlatformCapabilities.get_instance(session)
    if pc.is_nested_function_enabled():
        # Nested functions are enabled.
        print("Nested functions are enabled.")
    else:
        # Nested functions are disabled.
        print("Nested functions are disabled or not supported.")
    ```
    """

    _instance: Optional["PlatformCapabilities"] = None
    # Used for unittesting only. This is to avoid the need to mock the session object or reaching out to Snowflake
    _mock_features: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls, session: Optional[snowpark_session.Session] = None) -> "PlatformCapabilities":
        # Used for unittesting only. In this situation, _instance is not initialized.
        if cls._mock_features is not None:
            return cls(features=cls._mock_features)
        if not cls._instance:
            cls._instance = cls(session=session)
        return cls._instance

    @classmethod
    def set_mock_features(cls, features: Optional[Dict[str, Any]] = None) -> None:
        cls._mock_features = features

    @classmethod
    def clear_mock_features(cls) -> None:
        cls._mock_features = None

    # For contextmanager, we need to have return type Iterator[Never]. However, Never type is introduced only in
    # Python 3.11. So, we are ignoring the type for this method.
    @classmethod  # type: ignore[arg-type]
    @contextmanager
    def mock_features(cls, features: Dict[str, Any]) -> None:  # type: ignore[misc]
        logging.debug(f"Setting mock features: {features}")
        cls.set_mock_features(features)
        try:
            yield
        finally:
            logging.debug(f"Clearing mock features: {features}")
            cls.clear_mock_features()

    def is_nested_function_enabled(self) -> bool:
        return self._get_bool_feature("SPCS_MODEL_ENABLE_EMBEDDED_SERVICE_FUNCTIONS", False)

    def is_inlined_deployment_spec_enabled(self) -> bool:
        return self._get_bool_feature("ENABLE_INLINE_DEPLOYMENT_SPEC", False)

    def is_live_commit_enabled(self) -> bool:
        return self._get_bool_feature("ENABLE_BUNDLE_MODULE_CHECKOUT", False)

    @staticmethod
    def _get_features(session: snowpark_session.Session) -> Dict[str, Any]:
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
            logging.debug(f"Failed to retrieve platform capabilities: {e}")
            # This can happen is server side is older than 9.2. That is fine.
        return {}

    def __init__(
        self, *, session: Optional[snowpark_session.Session] = None, features: Optional[Dict[str, Any]] = None
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
