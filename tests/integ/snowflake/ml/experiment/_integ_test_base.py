import uuid

from absl.testing import parameterized

from snowflake.ml.experiment import ExperimentTracking, _logging as experiment_logging
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class ExperimentTrackingIntegTestBase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
        experiment_logging.ExperimentLogger.OUTPUT_DIRECTORY = "/tmp/experiment_tracking"

    def setUp(self) -> None:
        self.test_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self._session)
        self._schema_name = "PUBLIC"
        self._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.test_id, "TEST_EXPERIMENT_TRACKING"
        ).upper()
        self._db_manager.create_database(self._db_name, data_retention_time_in_days=1)
        self._db_manager.cleanup_databases(expire_hours=6)
        ExperimentTracking._instance = None
        self.exp = ExperimentTracking(
            self._session,
            database_name=self._db_name,
            schema_name=self._schema_name,
        )

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._db_name)
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

    def assert_experiment_tracking_equality(self, exp1: ExperimentTracking, exp2: ExperimentTracking) -> None:
        """Helper method to assert equality of two ExperimentTracking instances."""
        self.assertEqual(exp1._database_name, exp2._database_name)
        self.assertEqual(exp1._schema_name, exp2._schema_name)
        self.assertEqual(exp1._sql_client, exp2._sql_client)
        self.assertEqual(exp1._session is None, exp2._session is None)
        self.assertEqual(exp1._experiment is None, exp2._experiment is None)
        if exp1._experiment is not None and exp2._experiment is not None:
            self.assertEqual(exp1._experiment.name, exp2._experiment.name)
        self.assertEqual(exp1._run is None, exp2._run is None)
        if exp1._run is not None and exp2._run is not None:
            self.assertEqual(exp1._run.experiment_name, exp2._run.experiment_name)
            self.assertEqual(exp1._run.name, exp2._run.name)
        self.assertEqual(exp1._logging_context is None, exp2._logging_context is None)
        if exp1._logging_context is not None and exp2._logging_context is not None:
            self.assertEqual(exp1._logging_context.stdout_logger.exp_id, exp2._logging_context.stdout_logger.exp_id)
            self.assertEqual(exp1._logging_context.stdout_logger.run_id, exp2._logging_context.stdout_logger.run_id)
            self.assertEqual(exp1._logging_context.stdout_logger.stream, exp2._logging_context.stdout_logger.stream)
            self.assertEqual(exp1._logging_context.stdout_logger._buffer, exp2._logging_context.stdout_logger._buffer)
            self.assertEqual(
                exp1._logging_context.stdout_logger.file.name, exp2._logging_context.stdout_logger.file.name
            )
            self.assertEqual(exp1._logging_context.stderr_logger.exp_id, exp2._logging_context.stderr_logger.exp_id)
            self.assertEqual(exp1._logging_context.stderr_logger.run_id, exp2._logging_context.stderr_logger.run_id)
            self.assertEqual(exp1._logging_context.stderr_logger.stream, exp2._logging_context.stderr_logger.stream)
            self.assertEqual(exp1._logging_context.stderr_logger._buffer, exp2._logging_context.stderr_logger._buffer)
            self.assertEqual(
                exp1._logging_context.stderr_logger.file.name, exp2._logging_context.stderr_logger.file.name
            )
        self.assertEqual(exp1._registry is None, exp2._registry is None)
        if exp1._registry is not None and exp2._registry is not None:
            self.assertEqual(exp1._registry._database_name, exp2._registry._database_name)
            self.assertEqual(exp1._registry._schema_name, exp2._registry._schema_name)
            self.assertEqual(exp1._registry.enable_monitoring, exp2._registry.enable_monitoring)
            self.assertEqual(exp1._registry._model_manager._database_name, exp2._registry._model_manager._database_name)
            self.assertEqual(exp1._registry._model_manager._schema_name, exp2._registry._model_manager._schema_name)
            self.assertEqual(exp1._registry._model_manager._model_ops, exp2._registry._model_manager._model_ops)
            self.assertEqual(exp1._registry._model_manager._service_ops, exp2._registry._model_manager._service_ops)
            self.assertEqual(
                exp1._registry._model_monitor_manager._database_name,
                exp2._registry._model_monitor_manager._database_name,
            )
            self.assertEqual(
                exp1._registry._model_monitor_manager._schema_name, exp2._registry._model_monitor_manager._schema_name
            )
            self.assertEqual(
                exp1._registry._model_monitor_manager.statement_params,
                exp2._registry._model_monitor_manager.statement_params,
            )
            self.assertEqual(
                exp1._registry._model_monitor_manager._model_monitor_client._sql_client,
                exp2._registry._model_monitor_manager._model_monitor_client._sql_client,
            )
            self.assertEqual(
                exp1._registry._model_monitor_manager._model_monitor_client._database_name,
                exp2._registry._model_monitor_manager._model_monitor_client._database_name,
            )
            self.assertEqual(
                exp1._registry._model_monitor_manager._model_monitor_client._schema_name,
                exp2._registry._model_monitor_manager._model_monitor_client._schema_name,
            )
