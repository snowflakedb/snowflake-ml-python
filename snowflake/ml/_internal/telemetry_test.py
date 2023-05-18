import inspect
import time
from typing import Any, Dict, Optional
from unittest import mock

import pytest
from absl.testing import absltest, parameterized

from snowflake import connector
from snowflake.connector import telemetry as connector_telemetry
from snowflake.ml._internal import env, telemetry as utils_telemetry
from snowflake.snowpark import dataframe, session
from snowflake.snowpark._internal import error_message, server_connection

_SOURCE = env.SOURCE
_PROJECT = "Project"
_SUBPROJECT = "Subproject"
_VERSION = env.VERSION
_PYTHON_VERSION = env.PYTHON_VERSION
_OS = env.OS


class TelemetryTest(parameterized.TestCase):
    """Testing telemetry functions."""

    def setUp(self) -> None:
        self.mock_session = absltest.mock.MagicMock(spec=session.Session)
        self.mock_server_conn = absltest.mock.MagicMock(spec=server_connection.ServerConnection)
        self.mock_snowflake_conn = absltest.mock.MagicMock(spec=connector.SnowflakeConnection)
        self.mock_telemetry = absltest.mock.MagicMock(spec=connector_telemetry.TelemetryClient)
        self.mock_session._conn = self.mock_server_conn
        self.mock_server_conn._conn = self.mock_snowflake_conn
        self.mock_snowflake_conn._telemetry = self.mock_telemetry
        self.telemetry_type = f"{_SOURCE.lower()}_{utils_telemetry.TelemetryField.TYPE_FUNCTION_USAGE.value}"

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_telemetry(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test basic send_api_usage_telemetry."""
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                func_params_to_log=["param"],
                custom_tags={"custom_tag": "tag"},
            )
            def foo(self, param: Any) -> None:
                pass

        test_obj = DummyObject()
        test_obj.foo(param="val")
        self.mock_telemetry.try_add_log_to_batch.assert_called()
        self.assertEqual(
            utils_telemetry._SourceTelemetryClient.DEFAULT_FORCE_FLUSH_SIZE, self.mock_telemetry._flush_size
        )

        message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
        data = message["data"]

        # message
        assert message[connector_telemetry.TelemetryField.KEY_SOURCE.value] == _SOURCE
        assert message[utils_telemetry.TelemetryField.KEY_PROJECT.value] == _PROJECT
        assert message[utils_telemetry.TelemetryField.KEY_SUBPROJECT.value] == _SUBPROJECT
        assert message[connector_telemetry.TelemetryField.KEY_TYPE.value] == self.telemetry_type
        assert message[utils_telemetry.TelemetryField.KEY_VERSION.value] == _VERSION
        assert message[utils_telemetry.TelemetryField.KEY_PYTHON_VERSION.value] == _PYTHON_VERSION
        assert message[utils_telemetry.TelemetryField.KEY_OS.value] == _OS

        # data
        assert (
            data[utils_telemetry.TelemetryField.KEY_CATEGORY.value]
            == utils_telemetry.TelemetryField.FUNC_CAT_USAGE.value
        )
        assert "DummyObject.foo" in data[utils_telemetry.TelemetryField.KEY_FUNC_NAME.value]
        assert data[utils_telemetry.TelemetryField.KEY_FUNC_PARAMS.value] == {"param": "'val'"}
        assert data[utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value] == {"custom_tag": "tag"}

        # TODO(hayu): [SNOW-750523] Add json level comparisons in telemetry unit tests.

    def test_client_telemetry_conn_member_name_session(self) -> None:
        """Test send_api_usage_telemetry with `conn_member_name` and object has a session."""
        mock_session = absltest.mock.MagicMock(spec=session.Session)
        mock_session._conn = self.mock_server_conn

        class DummyObject:
            def __init__(self) -> None:
                self.session = mock_session

            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                conn_attr_name="s" "" "ession._conn._conn",
            )
            def foo(self) -> None:
                pass

        test_obj = DummyObject()
        test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_called()

    def test_client_telemetry_conn_member_name_conn(self) -> None:
        """Test send_api_usage_telemetry with `conn_member_name` and object has a SnowflakeConnection."""
        mock_snowflake_conn = absltest.mock.MagicMock(spec=connector.SnowflakeConnection)
        mock_snowflake_conn._telemetry = self.mock_telemetry

        class DummyObject:
            def __init__(self) -> None:
                self.conn = mock_snowflake_conn

            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                conn_attr_name="conn",
            )
            def foo(self) -> None:
                pass

        test_obj = DummyObject()
        test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_called()

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_telemetry_api_calls_extractor(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry with api calls extractor."""
        mock_get_active_sessions.return_value = {self.mock_session}

        def extract_api_calls(captured: Any) -> Any:
            assert isinstance(captured, DummyObject)
            return captured.api_calls

        class DummyObject:
            def __init__(self) -> None:
                self.api_calls = [time.time, time.sleep]

            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                api_calls_extractor=extract_api_calls,
            )
            def foo(self) -> None:
                time.time()
                time.sleep(0.001)  # sleep for 1 ms

        test_obj = DummyObject()
        test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_called()

        data = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]["data"]
        full_func_name_time = utils_telemetry._get_full_func_name(time.time)
        full_func_name_sleep = utils_telemetry._get_full_func_name(time.sleep)
        api_call_time = {utils_telemetry.TelemetryField.NAME.value: full_func_name_time}
        api_call_sleep = {utils_telemetry.TelemetryField.NAME.value: full_func_name_sleep}
        assert api_call_time in data[utils_telemetry.TelemetryField.KEY_API_CALLS.value]
        assert api_call_sleep in data[utils_telemetry.TelemetryField.KEY_API_CALLS.value]

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_error(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry when the decorated function raises an error."""
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                func_params_to_log=["param"],
            )
            def foo(self) -> None:
                raise RuntimeError("foo error")

        test_obj = DummyObject()
        with pytest.raises(RuntimeError):
            test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_called()
        self.mock_telemetry.send_batch.assert_called()

        message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
        self.assertEqual(repr(RuntimeError("foo error")), message[utils_telemetry.TelemetryField.KEY_ERROR_INFO.value])

    def test_get_statement_params_full_func_name(self) -> None:
        """Test get_statement_params_full_func_name."""

        class DummyObject:
            def foo(self) -> str:
                frame = inspect.currentframe()
                func_name: str = (
                    utils_telemetry.get_statement_params_full_func_name(frame, "DummyObject")
                    if frame
                    else "DummyObject.foo"
                )
                return func_name

        test_obj = DummyObject()
        actual_func_name = test_obj.foo()
        module = inspect.getmodule(inspect.currentframe())
        expected_func_name = f"{module.__name__}.DummyObject.foo" if module else "DummyObject.foo"
        assert actual_func_name == expected_func_name

    def test_get_function_usage_statement_params(self) -> None:
        """Test get_function_usage_statement_params."""

        class DummyObject:
            def foo(self, param: Any) -> Dict[str, Any]:
                frame = inspect.currentframe()
                func_name = (
                    utils_telemetry.get_statement_params_full_func_name(frame, "DummyObject")
                    if frame
                    else "DummyObject.foo"
                )
                statement_params: Dict[str, Any] = utils_telemetry.get_function_usage_statement_params(
                    project=_PROJECT,
                    subproject=_SUBPROJECT,
                    function_name=func_name,
                    function_parameters={"param": param},
                    api_calls=[time.time, time.sleep],
                    custom_tags={"custom_tag": "tag"},
                )
                time.time()
                time.sleep(0.001)  # sleep for 1 ms
                return statement_params

        test_obj = DummyObject()
        actual_statement_params = test_obj.foo(param="val")
        module = inspect.getmodule(inspect.currentframe())
        expected_func_name = f"{module.__name__}.DummyObject.foo" if module else "DummyObject.foo"
        full_func_name_time = utils_telemetry._get_full_func_name(time.time)
        full_func_name_sleep = utils_telemetry._get_full_func_name(time.sleep)
        api_call_time = {utils_telemetry.TelemetryField.NAME.value: full_func_name_time}
        api_call_sleep = {utils_telemetry.TelemetryField.NAME.value: full_func_name_sleep}
        expected_statement_params = {
            connector_telemetry.TelemetryField.KEY_SOURCE.value: _SOURCE,
            utils_telemetry.TelemetryField.KEY_PROJECT.value: _PROJECT,
            utils_telemetry.TelemetryField.KEY_SUBPROJECT.value: _SUBPROJECT,
            connector_telemetry.TelemetryField.KEY_TYPE.value: self.telemetry_type,
            utils_telemetry.TelemetryField.KEY_OS.value: _OS,
            utils_telemetry.TelemetryField.KEY_VERSION.value: _VERSION,
            utils_telemetry.TelemetryField.KEY_PYTHON_VERSION.value: _PYTHON_VERSION,
            utils_telemetry.TelemetryField.KEY_CATEGORY.value: utils_telemetry.TelemetryField.FUNC_CAT_USAGE.value,
            utils_telemetry.TelemetryField.KEY_FUNC_NAME.value: expected_func_name,
            utils_telemetry.TelemetryField.KEY_FUNC_PARAMS.value: {"param": "val"},
            utils_telemetry.TelemetryField.KEY_API_CALLS.value: [api_call_time, api_call_sleep],
            utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: {"custom_tag": "tag"},
        }
        assert actual_statement_params == expected_statement_params

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_telemetry_multiple_active_sessions(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry when there are multiple active sessions."""
        mock_session2 = absltest.mock.MagicMock(spec=session.Session)
        mock_session2._conn = self.mock_server_conn
        mock_get_active_sessions.return_value = {self.mock_session, mock_session2}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
            )
            def foo(self) -> None:
                pass

        test_obj = DummyObject()
        test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_called()

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_telemetry_no_default_session(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry when there is no default session."""
        mock_get_active_sessions.side_effect = error_message.SnowparkClientExceptionMessages.SERVER_NO_DEFAULT_SESSION()

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
            )
            def foo(self, ex: bool = False) -> None:
                if ex:
                    raise RuntimeError("foo error")
                return

        test_obj = DummyObject()
        test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_not_called()
        with self.assertRaises(RuntimeError) as context:
            test_obj.foo(True)
        self.assertEqual("foo error", str(context.exception))

    @parameterized.parameters(  # type: ignore
        {"params": {"default_stmt_params": None}},
        {"params": {"default_stmt_params": {}}},
        {"params": {"default_stmt_params": {"default": 0}}},
    )
    def test_add_stmt_params_to_df(self, params: Dict[str, Any]) -> None:
        class DummyObject:
            @utils_telemetry.add_stmt_params_to_df(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                func_params_to_log=["default_stmt_params"],
                api_calls=[time.time, time.sleep],
                custom_tags={"custom_tag": "tag"},
            )
            def foo(self, default_stmt_params: Optional[Dict[str, Any]] = None) -> dataframe.DataFrame:
                mock_df: dataframe.DataFrame = absltest.mock.MagicMock(spec=dataframe.DataFrame)
                if default_stmt_params is not None:
                    mock_df._statement_params = default_stmt_params.copy()  # type: ignore[assignment]
                return mock_df

            @utils_telemetry.add_stmt_params_to_df(
                project=_PROJECT,
            )
            def foo2(self) -> "DummyObject":
                return self

        test_obj = DummyObject()
        returned_df = test_obj.foo(**params)
        actual_statement_params = returned_df._statement_params
        full_func_name_time = utils_telemetry._get_full_func_name(time.time)
        full_func_name_sleep = utils_telemetry._get_full_func_name(time.sleep)
        api_call_time = {utils_telemetry.TelemetryField.NAME.value: full_func_name_time}
        api_call_sleep = {utils_telemetry.TelemetryField.NAME.value: full_func_name_sleep}
        expected_statement_params = {
            connector_telemetry.TelemetryField.KEY_SOURCE.value: _SOURCE,
            utils_telemetry.TelemetryField.KEY_PROJECT.value: _PROJECT,
            utils_telemetry.TelemetryField.KEY_SUBPROJECT.value: _SUBPROJECT,
            connector_telemetry.TelemetryField.KEY_TYPE.value: self.telemetry_type,
            utils_telemetry.TelemetryField.KEY_OS.value: _OS,
            utils_telemetry.TelemetryField.KEY_VERSION.value: _VERSION,
            utils_telemetry.TelemetryField.KEY_PYTHON_VERSION.value: _PYTHON_VERSION,
            utils_telemetry.TelemetryField.KEY_CATEGORY.value: utils_telemetry.TelemetryField.FUNC_CAT_USAGE.value,
            utils_telemetry.TelemetryField.KEY_FUNC_PARAMS.value: {
                "default_stmt_params": repr(params["default_stmt_params"])
            },
            utils_telemetry.TelemetryField.KEY_API_CALLS.value: [api_call_time, api_call_sleep],
            utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: {"custom_tag": "tag"},
        }
        self.assertIsNotNone(actual_statement_params)
        assert actual_statement_params is not None, "actual_statement_params cannot be None"  # mypy
        self.assertLessEqual(expected_statement_params.items(), actual_statement_params.items())
        self.assertIn("DummyObject.foo", actual_statement_params[utils_telemetry.TelemetryField.KEY_FUNC_NAME.value])
        self.assertFalse(hasattr(test_obj.foo2(), "_statement_params"))


if __name__ == "__main__":
    absltest.main()
