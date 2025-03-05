import inspect
import pickle
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional
from unittest import mock

import cloudpickle
from absl.testing import absltest, parameterized

from snowflake import connector
from snowflake.connector import cursor, telemetry as connector_telemetry
from snowflake.ml._internal import env, telemetry as utils_telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
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
        self.mock_snowflake_conn._session_parameters = {}
        self.mock_snowflake_conn.is_closed.return_value = False
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
                time.sleep(0.1)  # sleep for 100 ms

        test_obj = DummyObject()
        test_obj.foo(param="val")
        self.mock_telemetry.try_add_log_to_batch.assert_called()

        message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
        data = message["data"]

        # message
        self.assertEqual(message[connector_telemetry.TelemetryField.KEY_SOURCE.value], _SOURCE)
        self.assertEqual(message[utils_telemetry.TelemetryField.KEY_PROJECT.value], _PROJECT)
        self.assertEqual(message[utils_telemetry.TelemetryField.KEY_SUBPROJECT.value], _SUBPROJECT)
        self.assertEqual(message[connector_telemetry.TelemetryField.KEY_TYPE.value], self.telemetry_type)
        self.assertEqual(message[utils_telemetry.TelemetryField.KEY_VERSION.value], _VERSION)
        self.assertEqual(message[utils_telemetry.TelemetryField.KEY_PYTHON_VERSION.value], _PYTHON_VERSION)
        self.assertEqual(message[utils_telemetry.TelemetryField.KEY_OS.value], _OS)
        self.assertIsInstance(message[utils_telemetry.TelemetryField.KEY_DURATION.value], float)
        self.assertGreaterEqual(message[utils_telemetry.TelemetryField.KEY_DURATION.value], 0.1)

        # data
        self.assertEqual(
            data[utils_telemetry.TelemetryField.KEY_CATEGORY.value], utils_telemetry.TelemetryField.FUNC_CAT_USAGE.value
        )
        self.assertIn("DummyObject.foo", data[utils_telemetry.TelemetryField.KEY_FUNC_NAME.value])
        self.assertEqual(data[utils_telemetry.TelemetryField.KEY_FUNC_PARAMS.value], {"param": "'val'"})
        self.assertEqual(data[utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value], {"custom_tag": "tag"})

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
        self.assertIn(api_call_time, data[utils_telemetry.TelemetryField.KEY_API_CALLS.value])
        self.assertIn(api_call_sleep, data[utils_telemetry.TelemetryField.KEY_API_CALLS.value])

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_telemetry_error(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry when the decorated function raises an error."""
        mock_get_active_sessions.return_value = {self.mock_session}
        message = "foo error"

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
            )
            def foo(self) -> None:
                raise RuntimeError(message)

        test_obj = DummyObject()
        with self.assertRaises(RuntimeError):
            test_obj.foo()
        self.mock_telemetry.try_add_log_to_batch.assert_called()
        self.mock_telemetry.send_batch.assert_called()

        telemetry_message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
        expected_error = exceptions.SnowflakeMLException(error_codes.UNDEFINED, RuntimeError(message))
        self.assertEqual(error_codes.UNDEFINED, telemetry_message[utils_telemetry.TelemetryField.KEY_ERROR_CODE.value])
        self.assertEqual(repr(expected_error), telemetry_message[utils_telemetry.TelemetryField.KEY_ERROR_INFO.value])
        self.assertIn(
            "raise RuntimeError(message)", telemetry_message[utils_telemetry.TelemetryField.KEY_STACK_TRACE.value]
        )

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
        self.assertEqual(actual_func_name, expected_func_name)

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
        self.assertEqual(actual_statement_params, expected_statement_params)

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

    @parameterized.parameters(  # type: ignore[misc]
        {"params": {"default_stmt_params": None}},
        {"params": {"default_stmt_params": {}}},
        {"params": {"default_stmt_params": {"default": 0}}},
    )
    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_add_stmt_params_to_df(self, mock_get_active_sessions: mock.MagicMock, params: Dict[str, Any]) -> None:
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
                func_params_to_log=["default_stmt_params"],
                api_calls_extractor=extract_api_calls,
                custom_tags={"custom_tag": "tag"},
            )
            def foo(self, default_stmt_params: Optional[Dict[str, Any]] = None) -> dataframe.DataFrame:
                mock_df: dataframe.DataFrame = absltest.mock.MagicMock(spec=dataframe.DataFrame)
                if default_stmt_params is not None:
                    mock_df._statement_params = default_stmt_params.copy()  # type: ignore[assignment]
                return mock_df

            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo2(self) -> "DummyObject":
                return self

        test_obj = DummyObject()
        returned_df = test_obj.foo(**params)
        actual_statement_params = returned_df._statement_params
        full_func_name_time = utils_telemetry._get_full_func_name(time.time)
        full_func_name_sleep = utils_telemetry._get_full_func_name(time.sleep)
        full_func_name_foo = utils_telemetry._get_full_func_name(DummyObject.foo)
        api_call_time = {utils_telemetry.TelemetryField.NAME.value: full_func_name_time}
        api_call_sleep = {utils_telemetry.TelemetryField.NAME.value: full_func_name_sleep}
        api_call_foo = {utils_telemetry.TelemetryField.NAME.value: full_func_name_foo}
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
            utils_telemetry.TelemetryField.KEY_API_CALLS.value: [api_call_time, api_call_sleep, api_call_foo],
            utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: {"custom_tag": "tag"},
        }
        self.assertIsNotNone(actual_statement_params)
        assert actual_statement_params is not None  # mypy
        self.assertEqual(actual_statement_params, actual_statement_params | expected_statement_params)
        self.assertIn("DummyObject.foo", actual_statement_params[utils_telemetry.TelemetryField.KEY_FUNC_NAME.value])
        self.assertFalse(hasattr(test_obj.foo2(), "_statement_params"))

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_client_telemetry_flush_size(self, mock_get_active_sessions: mock.MagicMock) -> None:
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
                subproject=_SUBPROJECT,
            )
            def foo(self) -> None:
                pass

        test_obj = DummyObject()
        for _ in range(utils_telemetry._FLUSH_SIZE):
            test_obj.foo()
        self.mock_telemetry.send_batch.assert_called()

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_native_error(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry when the decorated function raises a native error."""
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo(self) -> None:
                raise RuntimeError("foo error")

        def validate_traceback(ex: Exception) -> bool:
            stack = traceback.extract_tb(ex.__traceback__)
            self.assertEqual(stack[-1].name, DummyObject.foo.__name__)
            return True

        test_obj = DummyObject()
        with self.assertRaisesWithPredicateMatch(RuntimeError, predicate=validate_traceback):
            test_obj.foo()

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_snowml_error(self, mock_get_active_sessions: mock.MagicMock) -> None:
        """Test send_api_usage_telemetry when the decorated function raises a snowml error."""
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo(self) -> None:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_TEST,
                    original_exception=RuntimeError("foo error"),
                )

        test_obj = DummyObject()
        with self.assertRaises(RuntimeError) as ex:
            test_obj.foo()
        self.assertIn(error_codes.INTERNAL_TEST, str(ex.exception))

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_snowml_nested_error(self, mock_get_active_sessions: mock.MagicMock) -> None:
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo(self) -> None:
                self.nested_foo()

            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def nested_foo(self) -> None:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_TEST,
                    original_exception=RuntimeError("foo error"),
                )

        test_obj = DummyObject()
        with self.assertRaises(RuntimeError) as ex:
            test_obj.foo()
        self.assertIn(error_codes.INTERNAL_TEST, str(ex.exception))
        self.assertNotIn(error_codes.UNDEFINED, str(ex.exception))

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_snowml_nested_error_tb_1(self, mock_get_active_sessions: mock.MagicMock) -> None:
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo(self) -> None:
                self.nested_foo()

            def nested_foo(self) -> None:
                raise RuntimeError("foo error")

        test_obj = DummyObject()
        try:
            test_obj.foo()
        except RuntimeError:
            self.assertIn("nested_foo", traceback.format_exc())
            telemetry_message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
            self.assertIn("self.nested_foo()", telemetry_message[utils_telemetry.TelemetryField.KEY_STACK_TRACE.value])

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_snowml_nested_error_tb_2(self, mock_get_active_sessions: mock.MagicMock) -> None:
        mock_get_active_sessions.return_value = {self.mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo(self) -> None:
                self.nested_foo()

            def nested_foo(self) -> None:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_TEST,
                    original_exception=RuntimeError("foo error"),
                )

        test_obj = DummyObject()
        try:
            test_obj.foo()
        except RuntimeError:
            self.assertIn("nested_foo", traceback.format_exc())
            telemetry_message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
            self.assertIn("self.nested_foo()", telemetry_message[utils_telemetry.TelemetryField.KEY_STACK_TRACE.value])

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_disable_telemetry(self, mock_get_active_sessions: mock.MagicMock) -> None:
        mock_session = absltest.mock.MagicMock(spec=session.Session)
        mock_session._conn = self.mock_server_conn

        mock_session.telemetry_enabled = False
        mock_get_active_sessions.return_value = {mock_session}

        class DummyObject:
            @utils_telemetry.send_api_usage_telemetry(
                project=_PROJECT,
            )
            def foo(self) -> dataframe.DataFrame:
                return absltest.mock.MagicMock(spec=dataframe.DataFrame)  # type: ignore[no-any-return]

        test_obj = DummyObject()
        returned_df = test_obj.foo()
        actual_statement_params = returned_df._statement_params
        # No client telemetry sent.
        self.mock_telemetry.try_add_log_to_batch.assert_not_called()
        assert actual_statement_params is not None  # mypy
        # Statement parameters updated to the returned dataframe.
        self.assertEqual(actual_statement_params[connector_telemetry.TelemetryField.KEY_SOURCE.value], env.SOURCE)

    @mock.patch("snowflake.snowpark.session._get_active_sessions")
    def test_send_custom_usage(self, mock_get_active_sessions: mock.MagicMock) -> None:
        mock_get_active_sessions.return_value = {self.mock_session}

        project = "m_project"
        subproject = "m_subproject"
        telemetry_type = "m_telemetry_type"
        tag = "m_tag"
        data = {"k1": "v1", "k2": {"nested_k2": "nested_v2"}}
        kwargs = {utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: tag}

        with mock.patch.object(utils_telemetry._SourceTelemetryClient, "_send", return_value=None) as m_send:
            utils_telemetry.send_custom_usage(
                project=project, telemetry_type=telemetry_type, subproject=subproject, data=data, **kwargs
            )

            m_send.assert_called_once_with(
                msg={
                    "source": "SnowML",
                    "project": project,
                    "subproject": subproject,
                    "version": _VERSION,
                    "python_version": _PYTHON_VERSION,
                    "operating_system": _OS,
                    "type": telemetry_type,
                    "data": data,
                    "custom_tags": tag,
                }
            )

    def test_get_sproc_statement_params_kwargs(self) -> None:
        def test_sproc_no_statement_params() -> None:
            return None

        def test_sproc_statement_params(statement_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
            return statement_params

        statement_params = {"test": "test"}
        kwargs = utils_telemetry.get_sproc_statement_params_kwargs(test_sproc_statement_params, statement_params)
        self.assertIn("statement_params", kwargs)

        kwargs = utils_telemetry.get_sproc_statement_params_kwargs(test_sproc_no_statement_params, statement_params)
        self.assertNotIn("statement_params", kwargs)

    def test_add_statement_params_custom_tags(self) -> None:
        project = "m_project"
        subproject = "m_subproject"
        custom_tags = {"test": "TEST"}

        # Test empty statement parameters
        self.assertEqual(utils_telemetry.add_statement_params_custom_tags({}, custom_tags), {})
        self.assertEqual(utils_telemetry.add_statement_params_custom_tags(None, custom_tags), {})

        statement_params = utils_telemetry.get_statement_params(project=project, subproject=subproject)

        # Test adding custom tags works
        self.assertNotIn(utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value, statement_params)
        result = utils_telemetry.add_statement_params_custom_tags(statement_params, custom_tags)
        self.assertIn(utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value, result)
        self.assertEqual(result.get(utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value), custom_tags)

        # Test overwriting tag works
        statement_params[utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value] = {"test": "BAD"}
        result = utils_telemetry.add_statement_params_custom_tags(statement_params, custom_tags)
        self.assertIn(utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value, result)
        self.assertEqual(result.get(utils_telemetry.TelemetryField.KEY_CUSTOM_TAGS.value), custom_tags)

    def test_apply_statement_params_patch(self) -> None:
        patch_manager = utils_telemetry._StatementParamsPatchManager()

        mock_cursor = absltest.mock.MagicMock(spec=cursor.SnowflakeCursor)
        with mock.patch.object(self.mock_snowflake_conn, "cursor", return_value=mock_cursor):
            server_conn = server_connection.ServerConnection({}, self.mock_snowflake_conn)
            sess = session.Session(server_conn)
            try:
                patch_manager._patch_session(sess, throw_on_patch_fail=True)
            except Exception as e:
                self.fail(f"Patching failed with unexpected exception: {e}")

    def test_pickle_instrumented_function(self) -> None:
        @utils_telemetry.send_api_usage_telemetry(
            project=_PROJECT,
            subproject="PICKLE",
        )
        def _picklable_test_function(session: session.Session) -> None:
            """Used for test_pickle_instrumented_function"""
            session.sql("SELECT 1").collect()

        with self.assertRaises(pickle.PicklingError):
            _ = cloudpickle.dumps(self.mock_session)

        self._do_internal_statement_params_test(_picklable_test_function)
        try:
            function_pickled = cloudpickle.dumps(_picklable_test_function)
        except Exception as e:
            self.fail(f"Pickling failed with unexpected exception: {e}")

        function_unpickled = cloudpickle.loads(function_pickled)
        self._do_internal_statement_params_test(function_unpickled)

    def test_statement_params_internal_query(self) -> None:
        # Create and decorate a test function that calls some SQL query
        @utils_telemetry.send_api_usage_telemetry(
            project=_PROJECT,
            subproject=_SUBPROJECT,
        )
        def dummy_function(session: session.Session) -> None:
            session.sql("SELECT 1").collect()  # Intentionally omit statement_params arg

        self._do_internal_statement_params_test(dummy_function)

    def test_statement_params_nested_internal_query(self) -> None:
        @utils_telemetry.send_api_usage_telemetry(
            project="INNER_PROJECT",
            subproject=_SUBPROJECT,
        )
        def inner_function(session: session.Session) -> None:
            session.sql("SELECT 1").collect()  # Intentionally omit statement_params arg

        @utils_telemetry.send_api_usage_telemetry(
            project="OUTER_PROJECT",
            subproject=_SUBPROJECT,
        )
        def outer_function(session: session.Session) -> None:
            inner_function(session)

        self._do_internal_statement_params_test(outer_function, expected_params={"project": "OUTER_PROJECT"})

    def test_statement_params_internal_params_precedence(self) -> None:
        @utils_telemetry.send_api_usage_telemetry(
            project=_PROJECT,
            subproject=_SUBPROJECT,
        )
        def project_override(session: session.Session) -> None:
            session.sql("SELECT 1").collect(statement_params={"project": "MY_OVERRIDE"})

        self._do_internal_statement_params_test(
            project_override,
            expected_params={"project": "MY_OVERRIDE", "snowml_telemetry_type": "SNOWML_AUGMENT_TELEMETRY"},
        )

        @utils_telemetry.send_api_usage_telemetry(
            project=_PROJECT,
            subproject=_SUBPROJECT,
        )
        def telemetry_type_override(session: session.Session) -> None:
            session.sql("SELECT 1").collect(statement_params={"snowml_telemetry_type": "user override"})

        self._do_internal_statement_params_test(
            telemetry_type_override,
            expected_params={"snowml_telemetry_type": "user override"},
        )

    def test_statement_params_multithreading(self) -> None:
        query1 = "select 1"
        query2 = "select 2"

        @utils_telemetry.send_api_usage_telemetry(project="PROJECT_1")
        def test_function1(session: session.Session) -> None:
            time.sleep(0.1)
            session.sql(query1).collect()

        @utils_telemetry.send_api_usage_telemetry(project="PROJECT_2")
        def test_function2(session: session.Session) -> None:
            session.sql(query2).collect()

        # Set up a real Session with mocking starting at SnowflakeConnection
        # Do this manually instead of using _do_internal_statement_params_test
        # to make sure we're sharing a single cursor so that we don't erroneously pass
        # the test just because each thread is using their own cursor.
        mock_cursor = absltest.mock.MagicMock(spec=cursor.SnowflakeCursor)
        with mock.patch.object(self.mock_snowflake_conn, "cursor", return_value=mock_cursor):
            server_conn = server_connection.ServerConnection({}, self.mock_snowflake_conn)
            sess = session.Session(server_conn)
            with mock.patch.object(session, "_get_active_sessions", return_value={sess}):
                thread1 = threading.Thread(target=test_function1, args=(sess,))
                thread2 = threading.Thread(target=test_function2, args=(sess,))

                thread1.start()
                thread2.start()

                thread1.join()
                thread2.join()

        self.assertEqual(2, len(mock_cursor.execute.call_args_list))
        statement_params_by_query = {
            call[0][0]: call.kwargs.get("_statement_params", {}) for call in mock_cursor.execute.call_args_list
        }

        default_params = {"source": "SnowML", "snowml_telemetry_type": "SNOWML_AUTO_TELEMETRY"}
        self.assertEqual(
            statement_params_by_query[query1],
            statement_params_by_query[query1] | {**default_params, "project": "PROJECT_1"},
        )
        self.assertEqual(
            statement_params_by_query[query2],
            statement_params_by_query[query2] | {**default_params, "project": "PROJECT_2"},
        )

    def test_statement_params_external_function(self) -> None:
        # Create and decorate a test function that calls some SQL query
        @utils_telemetry.send_api_usage_telemetry(
            project=_PROJECT,
            subproject=_SUBPROJECT,
        )
        def dummy_function(session: session.Session) -> None:
            session.sql("SELECT 1").collect()

        def external_function(session: session.Session) -> None:
            session.sql("SELECT 2").collect()

        # Set up a real Session with mocking starting at SnowflakeConnection
        # Do this manually instead of using _do_internal_statement_params_test
        # to make sure we're sharing a single cursor so that we don't erroneously pass
        # the test just because we're using a fresh session
        mock_cursor = absltest.mock.MagicMock(spec=cursor.SnowflakeCursor)
        with mock.patch.object(self.mock_snowflake_conn, "cursor", return_value=mock_cursor):
            server_conn = server_connection.ServerConnection({}, self.mock_snowflake_conn)
            sess = session.Session(server_conn)
            with mock.patch.object(session, "_get_active_sessions", return_value={sess}):
                dummy_function(sess)
                external_function(sess)

        call_statement_params = [
            call.kwargs.get("_statement_params", {}) for call in mock_cursor.execute.call_args_list
        ]
        self.assertEqual(2, len(call_statement_params))
        self.assertIn("source", call_statement_params[0].keys())
        self.assertIn("snowml_telemetry_type", call_statement_params[0].keys())
        self.assertNotIn("source", call_statement_params[1].keys())
        self.assertNotIn("snowml_telemetry_type", call_statement_params[1].keys())

    def _do_internal_statement_params_test(
        self, func: Callable[[session.Session], None], expected_params: Optional[Dict[str, str]] = None
    ) -> None:
        # Set up a real Session with mocking starting at SnowflakeConnection
        mock_cursor = absltest.mock.MagicMock(spec=cursor.SnowflakeCursor)
        with mock.patch.object(self.mock_snowflake_conn, "cursor", return_value=mock_cursor):
            server_conn = server_connection.ServerConnection({}, self.mock_snowflake_conn)
            sess = session.Session(server_conn)
            with mock.patch.object(session, "_get_active_sessions", return_value={sess}):
                func(sess)

        # Validate that the mock cursor received statement params
        mock_cursor.execute.assert_called_once()
        statement_params = mock_cursor.execute.call_args.kwargs.get("_statement_params", None)
        self.assertIsNotNone(statement_params, "statement params not found in execute call")

        expected_dict = {"source": "SnowML", "snowml_telemetry_type": "SNOWML_AUTO_TELEMETRY"}
        expected_dict.update(expected_params or {})
        self.assertEqual(statement_params, statement_params | expected_dict)


if __name__ == "__main__":
    absltest.main()
