import os
import pickle
import sys
from unittest.mock import patch

from absl.testing import absltest, parameterized

from snowflake.ml.experiment import ExperimentTracking, _logging as experiment_logging
from snowflake.snowpark import session as snowpark_session
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentSerializationIntegTest(ExperimentTrackingIntegTestBase):
    def test_experiment_getstate_and_setstate(self) -> None:
        """Test getstate and setstate methods by pickling and then unpickling"""
        self.exp.set_experiment("TEST_EXPERIMENT")
        self.exp.start_run("TEST_RUN")

        pickled = pickle.dumps(self.exp)
        ExperimentTracking._instance = None  # Reset singleton for test

        # Make sure that there is only one active session when _get_active_session() in setstate is called
        with snowpark_session._session_management_lock:
            session_set = snowpark_session._active_sessions.copy()
            snowpark_session._active_sessions = {self._session}
            new_exp = pickle.loads(pickled)  # setstate is called here
            snowpark_session._active_sessions = session_set

        self.assertIsNot(new_exp, self.exp)
        self.assert_experiment_tracking_equality(self.exp, new_exp)

    def test_experiment_getstate_and_setstate_no_session(self) -> None:
        """Test that setstate creates a new session if no session is found"""
        self.exp.set_experiment("TEST_EXPERIMENT")
        self.exp.start_run("TEST_RUN")

        pickled = pickle.dumps(self.exp)
        ExperimentTracking._instance = None  # Reset singleton for test

        # Make sure that there are no active sessions when _get_active_session() in setstate is called
        with snowpark_session._session_management_lock:
            session_set = snowpark_session._active_sessions.copy()
            snowpark_session._active_sessions.clear()
            new_exp = pickle.loads(pickled)  # setstate is called here
            self.assertEqual(len(snowpark_session._active_sessions), 1)  # check that a new session has been created
            new_session = snowpark_session._active_sessions.pop()
            snowpark_session._active_sessions = session_set

        self.assertIsNot(new_exp, self.exp)
        self.assert_experiment_tracking_equality(self.exp, new_exp)
        # Check that the unpickled experiment uses the newly created session
        self.assertIs(new_exp._session, new_session)
        self.assertIsNot(new_exp._session, self._session)

        # Clean up the newly created session
        new_session.close()

    @parameterized.parameters(True, False)
    def test_patch_stdout_and_stderr(self, live_logging_status: bool) -> None:
        """Test that stdout and stderr are patched to log to ExperimentLogger when a run is active."""
        experiment_name = "TEST_EXPERIMENT_STDOUT_STDERR"
        run_name = "TEST_RUN_STDOUT_STDERR"
        stdout_message = "This is a test message to stdout"
        stderr_message = "This is a test message to stderr"
        unpatched_message = "No run in context; this should not be logged."

        with patch("snowflake.ml._internal.env_utils.get_execution_context", return_value="SPCS"):
            self.exp.set_live_logging_status(live_logging_status)
        self.exp.set_experiment(experiment_name=experiment_name)
        self.assertIsNone(self.exp._logging_context)
        print(unpatched_message)
        print(unpatched_message, file=sys.stderr)

        with self.exp.start_run(run_name=run_name):
            self.assertEqual(live_logging_status, self.exp._logging_context is not None)
            print(stdout_message)
            print(stderr_message, file=sys.stderr)

            # Test buffer preservation after pickling and unpickling (only when live logging is enabled)
            if live_logging_status:
                # Write partial messages (no newline) to create buffered content
                print("partial stdout", end="")
                print("partial stderr", end="", file=sys.stderr)

                # Verify that the partial messages are in the buffer
                assert self.exp._logging_context is not None
                self.assertEqual(self.exp._logging_context.stdout_logger._buffer, "partial stdout")
                self.assertEqual(self.exp._logging_context.stderr_logger._buffer, "partial stderr")

                pickled = pickle.dumps(self.exp)

                # Disable live logging on original instance before unpickling to avoid double patching
                with patch("snowflake.ml._internal.env_utils.get_execution_context", return_value="SPCS"):
                    self.exp.set_live_logging_status(False)

                ExperimentTracking._instance = None  # Reset singleton for test
                # Make sure that there is only one active session when _get_active_session() in setstate is called
                with snowpark_session._session_management_lock:
                    session_set = snowpark_session._active_sessions.copy()
                    snowpark_session._active_sessions = {self._session}
                    with patch("snowflake.ml._internal.env_utils.get_execution_context", return_value="SPCS"):
                        restored_exp = pickle.loads(pickled)
                    snowpark_session._active_sessions = session_set

                # Verify that the partial messages are in the buffer of the restored object
                assert restored_exp._logging_context is not None
                self.assertEqual(restored_exp._logging_context.stdout_logger._buffer, "partial stdout")
                self.assertEqual(restored_exp._logging_context.stderr_logger._buffer, "partial stderr")

                # Complete the partial messages with newlines
                print(" completed")
                print(" completed", file=sys.stderr)

                # Clean up restored_exp to unpatch stdout/stderr before exiting
                with patch("snowflake.ml._internal.env_utils.get_execution_context", return_value="SPCS"):
                    restored_exp.set_live_logging_status(False)

        self.assertIsNone(self.exp._logging_context)
        print(unpatched_message)
        print(unpatched_message, file=sys.stderr)

        # Verify that the log messages were written to the ExperimentLogger file
        experiment_fqn = f"{self._db_name}.{self._schema_name}.{experiment_name}"
        exp_id = self._session.sql(f"CALL SYSTEM$RESOLVE_EXPERIMENT_ID('{experiment_fqn}')").collect()[0][0]
        run_id = self._session.sql(
            f"CALL SYSTEM$RESOLVE_EXPERIMENT_RUN_ID('{experiment_fqn}', '{run_name}')"
        ).collect()[0][0]
        stdout_logfile_path = os.path.join(
            experiment_logging.ExperimentLogger.OUTPUT_DIRECTORY, str(exp_id), str(run_id), "STDOUT.log"
        )
        stderr_logfile_path = os.path.join(
            experiment_logging.ExperimentLogger.OUTPUT_DIRECTORY, str(exp_id), str(run_id), "STDERR.log"
        )
        # Log file should have been written if and only if live logging was enabled
        self.assertEqual(os.path.exists(stdout_logfile_path), live_logging_status)
        self.assertEqual(os.path.exists(stderr_logfile_path), live_logging_status)

        if live_logging_status:
            with open(stdout_logfile_path) as f:
                log_contents = f.read()
                self.assertIn(stdout_message, log_contents)
                self.assertNotIn(stderr_message, log_contents)
                self.assertNotIn(unpatched_message, log_contents)
                self.assertIn("partial stdout completed", log_contents)
            with open(stderr_logfile_path) as f:
                log_contents = f.read()
                self.assertNotIn(stdout_message, log_contents)
                self.assertIn(stderr_message, log_contents)
                self.assertNotIn(unpatched_message, log_contents)
                self.assertIn("partial stderr completed", log_contents)


if __name__ == "__main__":
    absltest.main()
