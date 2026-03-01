import json
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml.experiment._logging import experiment_logger


class ExperimentLoggerTest(absltest.TestCase):
    """Test cases for ExperimentLogger."""

    def setUp(self) -> None:
        """Set up an ExperimentLogger."""
        self.experiment_id = 123
        self.run_id = 456
        self.stream = "STDOUT"
        experiment_logger.ExperimentLogger.OUTPUT_DIRECTORY = "/tmp/experiment_tracking"
        self.logger = experiment_logger.ExperimentLogger(self.experiment_id, self.run_id, self.stream)
        self.logger.file = MagicMock()

    def test_close(self) -> None:
        """Test that closing ExperimentLogger closes the underlying file."""
        self.logger.close()
        self.logger.file.close.assert_called_once()  # type: ignore[attr-defined]

    def test_flush(self) -> None:
        """Test that flushing ExperimentLogger flushes the underlying file."""
        self.logger.flush()
        self.logger.file.flush.assert_called_once()  # type: ignore[attr-defined]

    def test_write(self) -> None:
        """Test that writing to ExperimentLogger writes the correct JSON format to the file."""
        data = "This is a test log message."
        logger_write_return_value = self.logger.write(data)
        self.assertEqual(logger_write_return_value, len(data))

        expected_log_message = {
            "body": data,
            "attributes": {
                "snow.experiment.id": self.experiment_id,
                "snow.experiment.run.id": self.run_id,
                "snow.experiment.stream": self.stream,
            },
        }
        expected_json_data = json.dumps(expected_log_message) + "\n"
        self.logger.file.write.assert_called_once_with(expected_json_data)  # type: ignore[attr-defined]

    def test_writelines(self) -> None:
        """Test that writelines writes each line correctly."""
        lines = ["first line", "second line"]
        self.logger.writelines(lines)

        for line in lines:
            expected_log_message = {
                "body": line,
                "attributes": {
                    "snow.experiment.id": self.experiment_id,
                    "snow.experiment.run.id": self.run_id,
                    "snow.experiment.stream": self.stream,
                },
            }
            expected_json_data = json.dumps(expected_log_message) + "\n"
            self.logger.file.write.assert_any_call(expected_json_data)  # type: ignore[attr-defined]


if __name__ == "__main__":
    absltest.main()
