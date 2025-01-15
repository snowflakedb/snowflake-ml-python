import json
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

    @classmethod
    def get_instance(cls, session: Optional[snowpark_session.Session] = None) -> "PlatformCapabilities":
        if not cls._instance:
            cls._instance = cls(session)
        return cls._instance

    def is_nested_function_enabled(self) -> bool:
        return self._get_bool_feature("SPCS_MODEL_ENABLE_EMBEDDED_SERVICE_FUNCTIONS", False)

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

    def __init__(self, session: Optional[snowpark_session.Session] = None) -> None:
        if not session:
            session = next(iter(snowpark_session._get_active_sessions()))
            assert session, "Missing active session object"
        self.features: Dict[str, Any] = PlatformCapabilities._get_features(session)

    def _get_bool_feature(self, feature_name: str, default_value: bool) -> bool:
        value = self.features.get(feature_name, default_value)
        assert isinstance(value, bool), f"Expected bool, got {type(value)}"
        return value
