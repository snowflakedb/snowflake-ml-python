from unittest import mock

import yaml
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import (
    _evaluate,
    _evaluation_job,
    _evaluation_spec,
    _experiment_job,
)
from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.dataset import dataset
from snowflake.ml.experiment._client import experiment_tracking_sql_client
from snowflake.ml.experiment._entities import experiment as experiment_entity
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as snowpark_exceptions


class EvaluateTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = mock.create_autospec(snowpark.Session, instance=True)
        self.client = mock.create_autospec(
            experiment_tracking_sql_client.ExperimentTrackingSQLClient,
            instance=True,
        )
        self.input_dataset = mock.create_autospec(dataset.Dataset, instance=True)
        selected_version = mock.Mock()
        selected_version.name = "V1"
        self.input_dataset.selected_version = selected_version
        self.input_dataset.fully_qualified_name = "DB.SCHEMA.EVAL_DATA"

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_creates_executes_and_returns_job(self, resolve_client: mock.Mock) -> None:
        experiment_identifier = mock.Mock()
        resolve_client.return_value = (self.client, experiment_identifier)
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.MY_EVAL"

        job = _evaluate.evaluate(
            session=self.session,
            data=self.input_dataset,
            scorers=["exact_match"],
            model="llama3.1-70b",
            prompt="Answer {0}",
            inputs=["QUESTION"],
            label_col="ANSWER",
            experiment="DB.SCHEMA.MY_EVAL",
        )

        self.assertEqual(
            resolve_client.call_args_list,
            [
                mock.call(self.session, "DB.SCHEMA.MY_EVAL"),
                mock.call(self.session, "DB.SCHEMA.MY_EVAL"),
            ],
        )
        self.client.create_experiment.assert_called_once()
        create_args = self.client.create_experiment.call_args
        self.assertIs(create_args.args[0], experiment_identifier)
        self.assertIsInstance(create_args.args[1], sql_client.CreationMode)
        self.assertEqual(
            create_args.kwargs["experiment_type"],
            experiment_entity.ExperimentType.AI_FUNCTION_EVALUATION,
        )
        self.assertEqual(
            yaml.safe_load(create_args.kwargs["specification"]),
            {
                "query_text": "AI_COMPLETE('llama3.1-70b', PROMPT('Answer {0}', QUESTION))::VARCHAR",
                "metrics": [{"name": "exact_match"}],
                "dataset": {
                    "name": "DB.SCHEMA.EVAL_DATA",
                    "version": "V1",
                    "ground_truth": "ANSWER",
                },
            },
        )
        self.client.execute_experiment.assert_called_once_with(experiment_identifier)
        self.assertIsInstance(job, _evaluation_job.EvaluationJob)
        self.assertEqual(job.experiment, "DB.SCHEMA.MY_EVAL")

    @mock.patch.object(_experiment_job, "resolve_client")
    @mock.patch("snowflake.cortex._evaluate.context.get_active_session")
    @mock.patch(
        "snowflake.ml.model._client.sql._base.snowpark_utils.generate_random_alphanumeric",
        return_value="abc123",
    )
    def test_uses_active_session_and_generates_experiment_name(
        self,
        unused_random: mock.Mock,
        get_active_session: mock.Mock,
        resolve_client: mock.Mock,
    ) -> None:
        get_active_session.return_value = self.session
        experiment_identifier = mock.Mock()
        resolve_client.return_value = (self.client, experiment_identifier)
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.GENAI_EVAL_ABC123"

        job = _evaluate.evaluate(
            data=self.input_dataset,
            scorers=[_evaluation_spec.EvaluationMetric(name="llm_judge", judge_model="mistral-large2")],
            model="llama3.1-70b",
            prompt="Answer {0}",
            inputs=["question"],
            label_col="ANSWER",
        )

        get_active_session.assert_called_once_with()
        self.assertEqual(
            resolve_client.call_args_list,
            [
                mock.call(self.session, "GENAI_EVAL_ABC123"),
                mock.call(self.session, "DB.SCHEMA.GENAI_EVAL_ABC123"),
            ],
        )
        self.assertEqual(job.experiment, "DB.SCHEMA.GENAI_EVAL_ABC123")
        specification = yaml.safe_load(self.client.create_experiment.call_args.kwargs["specification"])
        self.assertEqual(specification["metrics"], [{"name": "llm_judge", "judge_model": "mistral-large2"}])

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_rejects_multiple_scorers_before_creating_experiment(self, resolve_client: mock.Mock) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one metric"):
            _evaluate.evaluate(
                session=self.session,
                data=self.input_dataset,
                scorers=["exact_match", "llm_judge"],
                model="llama3.1-70b",
                prompt="Answer the question",
                inputs=[],
                label_col="ANSWER",
            )

        resolve_client.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_rejects_zero_scorers_before_creating_experiment(self, resolve_client: mock.Mock) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one metric"):
            self._evaluate(scorers=[])

        resolve_client.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_create_failure_does_not_execute(self, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.create_experiment.side_effect = RuntimeError("already exists; choose a new experiment name")

        with self.assertRaisesRegex(RuntimeError, "choose a new experiment name"):
            self._evaluate(experiment="DB.SCHEMA.EXISTING")

        self.client.execute_experiment.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_execute_failure_preserves_error_and_names_created_experiment(self, resolve_client: mock.Mock) -> None:
        experiment_identifier = mock.Mock()
        resolve_client.return_value = (self.client, experiment_identifier)
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.MY_EVAL"
        execute_error = snowpark_exceptions.SnowparkSQLException("Cortex role required", sql_error_code=400032)
        self.client.execute_experiment.side_effect = execute_error

        with self.assertRaisesRegex(RuntimeError, "was created but execution did not start") as raised:
            self._evaluate(experiment="MY_EVAL")

        self.assertIn("DB.SCHEMA.MY_EVAL", str(raised.exception))
        self.assertIn("Cortex role required", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, snowml_exceptions.SnowflakeMLException)
        self.assertIs(raised.exception.__cause__.__cause__, execute_error)
        self.client.create_experiment.assert_called_once()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_execute_client_error_is_not_reported_as_execution_failure(self, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.MY_EVAL"
        self.client.execute_experiment.side_effect = AttributeError("client defect")

        with self.assertRaisesRegex(AttributeError, "client defect"):
            self._evaluate(experiment="MY_EVAL")

    @mock.patch.object(_experiment_job, "resolve_client")
    @mock.patch("snowflake.cortex._evaluate.context.get_active_session")
    def test_no_active_session_creates_nothing(
        self,
        get_active_session: mock.Mock,
        resolve_client: mock.Mock,
    ) -> None:
        get_active_session.side_effect = RuntimeError("No active Snowpark session")

        with self.assertRaisesRegex(RuntimeError, "No active Snowpark session"):
            self._evaluate(session=None)

        resolve_client.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_invalid_metric_creates_nothing(self, resolve_client: mock.Mock) -> None:
        with self.assertRaisesRegex(ValueError, "judge model can only be used"):
            self._evaluate(scorers=[_evaluation_spec.EvaluationMetric("exact_match", judge_model="x")])

        resolve_client.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_job_uses_creation_context(self, resolve_client: mock.Mock) -> None:
        experiment_identifier = mock.Mock()
        resolve_client.return_value = (self.client, experiment_identifier)
        self.client.fully_qualified_object_name.return_value = "ORIGINAL_DB.ORIGINAL_SCHEMA.MY_EVAL"

        job = self._evaluate(experiment="MY_EVAL")

        self.assertEqual(job.experiment, "ORIGINAL_DB.ORIGINAL_SCHEMA.MY_EVAL")

    @mock.patch.object(_experiment_job, "resolve_client")
    @mock.patch(
        "snowflake.ml.model._client.sql._base.snowpark_utils.generate_random_alphanumeric",
        side_effect=["first", "second"],
    )
    def test_generated_names_are_distinct(self, unused_random: mock.Mock, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.fully_qualified_object_name.side_effect = [
            "DB.SCHEMA.GENAI_EVAL_FIRST",
            "DB.SCHEMA.GENAI_EVAL_FIRST",
            "DB.SCHEMA.GENAI_EVAL_SECOND",
            "DB.SCHEMA.GENAI_EVAL_SECOND",
        ]

        first = self._evaluate()
        second = self._evaluate()

        self.assertNotEqual(first.experiment, second.experiment)
        self.assertEqual(
            resolve_client.call_args_list,
            [
                mock.call(self.session, "GENAI_EVAL_FIRST"),
                mock.call(self.session, "DB.SCHEMA.GENAI_EVAL_FIRST"),
                mock.call(self.session, "GENAI_EVAL_SECOND"),
                mock.call(self.session, "DB.SCHEMA.GENAI_EVAL_SECOND"),
            ],
        )

    def _evaluate(self, **overrides: object) -> _evaluation_job.EvaluationJob:
        values: dict[str, object] = {
            "session": self.session,
            "data": self.input_dataset,
            "scorers": ["exact_match"],
            "model": "llama3.1-70b",
            "prompt": "Answer {0}",
            "inputs": ["QUESTION"],
            "label_col": "ANSWER",
        }
        values.update(overrides)
        return _evaluate.evaluate(**values)  # type: ignore[arg-type]


if __name__ == "__main__":
    absltest.main()
