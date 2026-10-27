from unittest import mock

import yaml
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import (
    _evaluation_spec,
    _experiment_job,
    _optimization_job,
    _optimize_prompt,
)
from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.dataset import dataset
from snowflake.ml.experiment._client import experiment_tracking_sql_client
from snowflake.ml.experiment._entities import experiment as experiment_entity
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as snowpark_exceptions


class OptimizePromptTest(absltest.TestCase):
    def setUp(self) -> None:
        self.session = mock.create_autospec(snowpark.Session, instance=True)
        self.client = mock.create_autospec(
            experiment_tracking_sql_client.ExperimentTrackingSQLClient,
            instance=True,
        )
        self.trainset = mock.create_autospec(dataset.Dataset, instance=True)
        selected_version = mock.Mock()
        selected_version.name = "V1"
        self.trainset.selected_version = selected_version
        self.trainset.fully_qualified_name = "DB.SCHEMA.TRAIN_DATA"
        self.holdout_data = mock.create_autospec(dataset.Dataset, instance=True)
        holdout_version = mock.Mock()
        holdout_version.name = "V2"
        self.holdout_data.selected_version = holdout_version
        self.holdout_data.fully_qualified_name = "DB.SCHEMA.HOLDOUT_DATA"

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_creates_executes_and_returns_job(self, resolve_client: mock.Mock) -> None:
        experiment_identifier = mock.Mock()
        resolve_client.return_value = (self.client, experiment_identifier)
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.MY_OPTIMIZATION"

        job = _optimize_prompt.optimize_prompt(
            session=self.session,
            trainset=self.trainset,
            metric="exact_match",
            model="llama3.1-70b",
            prompt="Answer {0}",
            inputs=["QUESTION"],
            label_col="ANSWER",
            holdout_data=self.holdout_data,
            models=["llama3.1-70b", "mistral-large2"],
            budget="medium",
            reflection_model="claude-sonnet-4-5",
            strategy="balanced",
            algorithm="gepa",
            experiment="DB.SCHEMA.MY_OPTIMIZATION",
        )

        self.assertEqual(
            resolve_client.call_args_list,
            [
                mock.call(self.session, "DB.SCHEMA.MY_OPTIMIZATION"),
                mock.call(self.session, "DB.SCHEMA.MY_OPTIMIZATION"),
            ],
        )
        self.client.create_experiment.assert_called_once()
        create_args = self.client.create_experiment.call_args
        self.assertIs(create_args.args[0], experiment_identifier)
        self.assertIsInstance(create_args.args[1], sql_client.CreationMode)
        self.assertEqual(
            create_args.kwargs["experiment_type"],
            experiment_entity.ExperimentType.AI_FUNCTION_OPTIMIZATION,
        )
        self.assertEqual(
            yaml.safe_load(create_args.kwargs["specification"]),
            {
                "function": {
                    "function_body": "AI_COMPLETE('llama3.1-70b', PROMPT('Answer {0}', QUESTION))",
                },
                "metrics": [{"name": "exact_match"}],
                "dataset": {
                    "name": "DB.SCHEMA.TRAIN_DATA",
                    "version": "V1",
                    "holdout_data": "DB.SCHEMA.HOLDOUT_DATA",
                    "holdout_version": "V2",
                    "column_mapping": {
                        "argument_mapping": {"$1": "QUESTION"},
                        "ground_truth": "ANSWER",
                    },
                },
                "optimization": {
                    "models": ["llama3.1-70b", "mistral-large2"],
                    "budget": "medium",
                    "reflection_model": "claude-sonnet-4-5",
                    "strategy": "balanced",
                    "algorithm": "gepa",
                },
            },
        )
        self.client.execute_experiment.assert_called_once_with(experiment_identifier)
        self.assertIsInstance(job, _optimization_job.OptimizationJob)
        self.assertEqual(job.experiment, "DB.SCHEMA.MY_OPTIMIZATION")

    @mock.patch.object(_experiment_job, "resolve_client")
    @mock.patch("snowflake.cortex._optimize_prompt.context.get_active_session")
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
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.GENAI_OPT_ABC123"

        job = _optimize_prompt.optimize_prompt(
            trainset=self.trainset,
            metric=_evaluation_spec.EvaluationMetric(name="llm_judge", judge_model="mistral-large2"),
            model="llama3.1-70b",
            prompt="Answer {0}",
            inputs=["QUESTION"],
            label_col="ANSWER",
        )

        get_active_session.assert_called_once_with()
        self.assertEqual(
            resolve_client.call_args_list,
            [
                mock.call(self.session, "GENAI_OPT_ABC123"),
                mock.call(self.session, "DB.SCHEMA.GENAI_OPT_ABC123"),
            ],
        )
        self.assertEqual(job.experiment, "DB.SCHEMA.GENAI_OPT_ABC123")
        specification = yaml.safe_load(self.client.create_experiment.call_args.kwargs["specification"])
        self.assertEqual(specification["metrics"], [{"name": "llm_judge", "judge_model": "mistral-large2"}])
        self.assertEqual(specification["optimization"]["models"], ["llama3.1-70b"])
        self.assertNotIn("budget", specification["optimization"])

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_execute_failure_names_created_experiment(self, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.MY_OPTIMIZATION"
        execute_error = snowpark_exceptions.SnowparkSQLException("Cortex role required", sql_error_code=400032)
        self.client.execute_experiment.side_effect = execute_error

        with self.assertRaisesRegex(RuntimeError, "was created but execution did not start") as raised:
            self._optimize(experiment="MY_OPTIMIZATION")

        self.assertIn("DB.SCHEMA.MY_OPTIMIZATION", str(raised.exception))
        self.assertIn("Cortex role required", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, snowml_exceptions.SnowflakeMLException)
        self.assertIs(raised.exception.__cause__.__cause__, execute_error)
        self.client.create_experiment.assert_called_once()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_execute_client_error_is_not_reported_as_execution_failure(self, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.fully_qualified_object_name.return_value = "DB.SCHEMA.MY_OPTIMIZATION"
        self.client.execute_experiment.side_effect = AttributeError("client defect")

        with self.assertRaisesRegex(AttributeError, "client defect"):
            self._optimize(experiment="MY_OPTIMIZATION")

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_create_failure_does_not_execute(self, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.create_experiment.side_effect = RuntimeError("already exists")

        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self._optimize(experiment="DB.SCHEMA.EXISTING")

        self.client.execute_experiment.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    @mock.patch("snowflake.cortex._optimize_prompt.context.get_active_session")
    def test_no_active_session_creates_nothing(
        self,
        get_active_session: mock.Mock,
        resolve_client: mock.Mock,
    ) -> None:
        get_active_session.side_effect = RuntimeError("No active Snowpark session")

        with self.assertRaisesRegex(RuntimeError, "No active Snowpark session"):
            self._optimize(session=None)

        resolve_client.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    def test_invalid_metric_creates_nothing(self, resolve_client: mock.Mock) -> None:
        with self.assertRaisesRegex(ValueError, "judge model can only be used"):
            self._optimize(metric=_evaluation_spec.EvaluationMetric("exact_match", judge_model="x"))

        resolve_client.assert_not_called()

    @mock.patch.object(_experiment_job, "resolve_client")
    @mock.patch(
        "snowflake.ml.model._client.sql._base.snowpark_utils.generate_random_alphanumeric",
        side_effect=["first", "second"],
    )
    def test_generated_names_are_distinct(self, unused_random: mock.Mock, resolve_client: mock.Mock) -> None:
        resolve_client.return_value = (self.client, mock.Mock())
        self.client.fully_qualified_object_name.side_effect = [
            "DB.SCHEMA.GENAI_OPT_FIRST",
            "DB.SCHEMA.GENAI_OPT_FIRST",
            "DB.SCHEMA.GENAI_OPT_SECOND",
            "DB.SCHEMA.GENAI_OPT_SECOND",
        ]

        first = self._optimize()
        second = self._optimize()

        self.assertNotEqual(first.experiment, second.experiment)
        self.assertEqual(
            resolve_client.call_args_list,
            [
                mock.call(self.session, "GENAI_OPT_FIRST"),
                mock.call(self.session, "DB.SCHEMA.GENAI_OPT_FIRST"),
                mock.call(self.session, "GENAI_OPT_SECOND"),
                mock.call(self.session, "DB.SCHEMA.GENAI_OPT_SECOND"),
            ],
        )

    def _optimize(self, **overrides: object) -> _optimization_job.OptimizationJob:
        values: dict[str, object] = {
            "session": self.session,
            "trainset": self.trainset,
            "metric": "exact_match",
            "model": "llama3.1-70b",
            "prompt": "Answer {0}",
            "inputs": ["QUESTION"],
            "label_col": "ANSWER",
        }
        values.update(overrides)
        return _optimize_prompt.optimize_prompt(**values)  # type: ignore[arg-type]


if __name__ == "__main__":
    absltest.main()
