import json
from typing import Any
from unittest import mock

from absl.testing import absltest

from snowflake.cortex import Finetune, FinetuneJob, FinetuneStatus
from snowflake.ml.test_utils import mock_data_frame


class FinetuneTest(absltest.TestCase):
    system_function_name = "SNOWFLAKE.CORTEX.FINETUNE"

    def setUp(self) -> None:
        self.list_jobs_return_value: list[dict[str, Any]] = [
            {"id": "1", "status": "SUCCESS"},
            {"id": "2", "status": "ERROR"},
        ]
        self.list_jobs_expected_result = [
            FinetuneJob(session=None, status=FinetuneStatus(**status)) for status in self.list_jobs_return_value
        ]

    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_create(self, mock_call_sql_function: mock.Mock) -> None:
        """Test call of finetune operation CREATE."""
        mock_call_sql_function.return_value = "workflow_id"
        cft = Finetune()
        cft.create("test_model", "base_model", "SELECT * FROM TRAINING", "SELECT * FROM VALIDATION")
        mock_call_sql_function.assert_called_with(
            self.system_function_name,
            None,
            "CREATE",
            "test_model",
            "base_model",
            "SELECT * FROM TRAINING",
            "SELECT * FROM VALIDATION",
            None,
        )

    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_create_with_snowpark_dataframe(self, mock_call_sql_function: mock.Mock) -> None:
        """Test call of finetune operation CREATE."""
        mock_call_sql_function.return_value = "workflow_id"
        training_df = mock_data_frame.MockDataFrame()
        training_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")
        validation_df = mock_data_frame.MockDataFrame()
        validation_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM VALIDATION")

        cft = Finetune()
        cft.create("test_model", "base_model", training_df, validation_df)
        mock_call_sql_function.assert_called_with(
            self.system_function_name,
            None,
            "CREATE",
            "test_model",
            "base_model",
            "SELECT PROMPT, COMPLETION FROM TRAINING",
            "SELECT PROMPT, COMPLETION FROM VALIDATION",
            None,
        )

    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_create_with_snowpark_dataframe_two_training_queries(
        self, mock_call_sql_function: mock.Mock
    ) -> None:
        """Test call of finetune operation CREATE with an incompatible training DataFrame."""
        training_df = mock_data_frame.MockDataFrame()
        training_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")
        training_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM VALIDATION")
        validation_df = mock_data_frame.MockDataFrame()
        validation_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM VALIDATION")

        cft = Finetune()
        with self.assertRaisesRegex(ValueError, r".*training_data.*queries.*"):
            cft.create("test_model", "base_model", training_df, validation_df)

    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_create_with_snowpark_dataframe_two_validation_queries(
        self, mock_call_sql_function: mock.Mock
    ) -> None:
        """Test call of finetune operation CREATE with an incompatible validation DataFrame."""
        training_df = mock_data_frame.MockDataFrame()
        training_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")
        validation_df = mock_data_frame.MockDataFrame()
        validation_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM VALIDATION")
        validation_df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")

        cft = Finetune()
        with self.assertRaisesRegex(ValueError, r"validation_data.*queries"):
            cft.create("test_model", "base_model", training_df, validation_df)

    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_create_with_options(self, mock_call_sql_function: mock.Mock) -> None:
        """Test call of finetune operation CREATE with options."""
        mock_call_sql_function.return_value = "workflow_id"
        cft = Finetune()
        cft.create("test_model", "base_model", "SELECT * FROM TRAINING", "SELECT * FROM VALIDATION", {"awesome": True})
        mock_call_sql_function.assert_called_with(
            self.system_function_name,
            None,
            "CREATE",
            "test_model",
            "base_model",
            "SELECT * FROM TRAINING",
            "SELECT * FROM VALIDATION",
            {"awesome": True},
        )

    @mock.patch("snowflake.cortex._finetune.Finetune.list_jobs")
    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_cancel(self, mock_call_sql_function: mock.Mock, mock_finetune_list_jobs: mock.Mock) -> None:
        """Test call of finetune operation CANCEL."""
        mock_call_sql_function.return_value = "job 2 cancelled"
        mock_finetune_list_jobs.return_value = self.list_jobs_expected_result
        cft = Finetune()
        run = cft.list_jobs()[1]
        run.cancel()
        mock_call_sql_function.assert_called_with(self.system_function_name, None, "CANCEL", "2")

    @mock.patch("snowflake.cortex._finetune.Finetune.list_jobs")
    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_describe(self, mock_call_sql_function: mock.Mock, mock_finetune_list_jobs: mock.Mock) -> None:
        """Test call of finetune operation DESCRIBE."""
        sql_return_value: dict[str, Any] = {
            "base_model": "llama3-8b",
            "created_on": 1728688216077,
            "finished_on": 1728688392137,
            "id": "CortexFineTuningWorkflow_4dbc8970-65d8-44b8-9054-da818c1593dd",
            "model": "CFT_DB.TASTY_EMAIL.test_api_1",
            "progress": 1.0,
            "status": "SUCCESS",
            "training_data": (
                "select MODIFIED_BODY as PROMPT, GOLDEN_JSON as COMPLETION "
                "from EMAIL_MODIFIED_WITH_RESPONSE_GOLDEN_10K_JSON where id % 10 = 0"
            ),
            "trained_tokens": 377100,
            "training_result": {"validation_loss": 0.8828646540641785, "training_loss": 0.8691850564418695},
            "validation_data": "",
        }
        expected_result = FinetuneStatus(**sql_return_value)
        mock_call_sql_function.return_value = json.dumps(sql_return_value)
        mock_finetune_list_jobs.return_value = self.list_jobs_expected_result
        run = Finetune().list_jobs()[0]
        self.assertEqual(
            run.describe(),
            expected_result,
        )
        mock_call_sql_function.assert_called_with(self.system_function_name, None, "DESCRIBE", "1")

    @mock.patch("snowflake.cortex._finetune.Finetune.list_jobs")
    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_describe_error(
        self, mock_call_sql_function: mock.Mock, mock_finetune_list_jobs: mock.Mock
    ) -> None:
        """Test call of finetune operation DESCRIBE with error message."""
        mock_finetune_list_jobs.return_value = self.list_jobs_expected_result
        sql_return_value: dict[str, Any] = {
            "base_model": "llama3-8b",
            "created_on": 1728670992861,
            "error": {
                "code": "INVALID_PARAMETER_VALUE",
                "message": (
                    'Failed to query input data for Fine-tuning. Failed to execute query: SELECT "PROMPT", '
                    '"COMPLETION" FROM (select MODIFIED_BODY as PROMPT, GOLDEN_JSON as COMPLETION from '
                    "EMAIL_MODIFIED_WITH_RESPONSE_GOLDEN_10K_JSON where id % 10 = 0;). 001003 (42000): "
                    "01b79faf-0003-dbfd-0022-b7876037dabe: SQL compilation error:\nsyntax error line 1 at position "
                    "161 unexpected ';'."
                ),
            },
            "finished_on": 1728670994015,
            "id": "CortexFineTuningWorkflow_54cfb2bb-ff69-4d5a-8513-320ba3fdb258",
            "progress": 0.0,
            "status": "ERROR",
            "training_data": (
                "select MODIFIED_BODY as PROMPT, GOLDEN_JSON as COMPLETION from "
                "EMAIL_MODIFIED_WITH_RESPONSE_GOLDEN_10K_JSON where id % 10 = 0;"
            ),
            "validation_data": "",
        }

        mock_call_sql_function.return_value = json.dumps(sql_return_value)
        run = Finetune().list_jobs()[0]
        self.assertEqual(run.describe(), FinetuneStatus(**sql_return_value))
        mock_call_sql_function.assert_called_with(self.system_function_name, None, "DESCRIBE", "1")

    @mock.patch("snowflake.cortex._finetune.call_sql_function_literals")
    def test_finetune_list_jobs(self, mock_call_sql_function: mock.Mock) -> None:
        """Test call of finetune operation list_jobs."""
        mock_call_sql_function.return_value = json.dumps(self.list_jobs_return_value)
        run_list = Finetune().list_jobs()
        self.assertTrue(isinstance(run_list, list))
        self.assertEqual(run_list, self.list_jobs_expected_result)
        mock_call_sql_function.assert_called_with(self.system_function_name, None, "SHOW")


if __name__ == "__main__":
    absltest.main()
