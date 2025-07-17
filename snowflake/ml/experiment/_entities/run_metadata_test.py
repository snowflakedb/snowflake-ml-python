from absl.testing import absltest

from snowflake.ml.experiment._entities import run_metadata


class RunDataTest(absltest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.sample_metrics = [
            run_metadata.Metric(name="accuracy", value=0.95, step=100),
            run_metadata.Metric(name="loss", value=0.05, step=100),
        ]
        self.sample_params = [
            run_metadata.Param(name="learning_rate", value="0.001"),
            run_metadata.Param(name="batch_size", value="32"),
        ]

    def test_init(self) -> None:
        """Test RunData initialization."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=self.sample_metrics,
            parameters=self.sample_params,
        )

        self.assertEqual(run_data.status, run_metadata.RunStatus.RUNNING)
        self.assertEqual(len(run_data.metrics), 2)
        self.assertEqual(len(run_data.parameters), 2)
        self.assertEqual(run_data.metrics[0].name, "accuracy")
        self.assertEqual(run_data.parameters[0].name, "learning_rate")

    def test_from_dict_complete(self) -> None:
        """Test creating RunData from a complete dictionary."""
        metadata = {
            "status": "RUNNING",
            "metrics": [
                {"name": "accuracy", "value": 0.95, "step": 100},
                {"name": "loss", "value": 0.05, "step": 100},
            ],
            "parameters": [
                {"name": "learning_rate", "value": "0.001"},
                {"name": "batch_size", "value": "32"},
            ],
        }

        run_data = run_metadata.RunMetadata.from_dict(metadata)

        self.assertEqual(run_data.status, run_metadata.RunStatus.RUNNING)
        self.assertEqual(len(run_data.metrics), 2)
        self.assertEqual(len(run_data.parameters), 2)

        # Check metrics
        self.assertEqual(run_data.metrics[0].name, "accuracy")
        self.assertEqual(run_data.metrics[0].value, 0.95)
        self.assertEqual(run_data.metrics[0].step, 100)

        # Check parameters
        self.assertEqual(run_data.parameters[0].name, "learning_rate")
        self.assertEqual(run_data.parameters[0].value, "0.001")

    def test_from_dict_empty(self) -> None:
        """Test creating RunData from an empty dictionary."""
        metadata = {}  # type: ignore[var-annotated]

        run_data = run_metadata.RunMetadata.from_dict(metadata)

        self.assertEqual(run_data.status, run_metadata.RunStatus.UNKNOWN)
        self.assertEqual(len(run_data.metrics), 0)
        self.assertEqual(len(run_data.parameters), 0)

    def test_to_dict(self) -> None:
        """Test converting RunData to dictionary."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=self.sample_metrics,
            parameters=self.sample_params,
        )

        result = run_data.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], run_metadata.RunStatus.RUNNING)
        self.assertEqual(len(result["metrics"]), 2)
        self.assertEqual(len(result["parameters"]), 2)

        # Check that metrics are converted to dicts
        self.assertIsInstance(result["metrics"][0], dict)
        self.assertEqual(result["metrics"][0]["name"], "accuracy")
        self.assertEqual(result["metrics"][0]["value"], 0.95)
        self.assertEqual(result["metrics"][0]["step"], 100)

    def test_set_metric_new(self) -> None:
        """Test setting a new metric."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=[],
            parameters=[],
        )

        run_data.set_metric(key="accuracy", value=0.85, step=50)

        self.assertEqual(len(run_data.metrics), 1)
        self.assertEqual(run_data.metrics[0].name, "accuracy")
        self.assertEqual(run_data.metrics[0].value, 0.85)
        self.assertEqual(run_data.metrics[0].step, 50)

    def test_set_metric_update_existing(self) -> None:
        """Test updating an existing metric with the same name and step."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=[run_metadata.Metric(name="accuracy", value=0.80, step=50)],
            parameters=[],
        )

        run_data.set_metric(key="accuracy", value=0.90, step=50)

        self.assertEqual(len(run_data.metrics), 1)
        self.assertEqual(run_data.metrics[0].value, 0.90)

    def test_set_metric_different_step(self) -> None:
        """Test setting a metric with the same name but different step."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=[run_metadata.Metric(name="accuracy", value=0.80, step=50)],
            parameters=[],
        )

        run_data.set_metric(key="accuracy", value=0.90, step=100)

        self.assertEqual(len(run_data.metrics), 2)
        self.assertEqual(run_data.metrics[0].step, 50)
        self.assertEqual(run_data.metrics[1].step, 100)
        self.assertEqual(run_data.metrics[1].value, 0.90)

    def test_set_param_new(self) -> None:
        """Test setting a new parameter."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=[],
            parameters=[],
        )

        run_data.set_param(key="learning_rate", value=0.001)

        self.assertEqual(len(run_data.parameters), 1)
        self.assertEqual(run_data.parameters[0].name, "learning_rate")
        self.assertEqual(run_data.parameters[0].value, "0.001")

    def test_set_param_update_existing(self) -> None:
        """Test updating an existing parameter."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=[],
            parameters=[run_metadata.Param(name="learning_rate", value="0.001")],
        )

        run_data.set_param(key="learning_rate", value=0.01)

        self.assertEqual(len(run_data.parameters), 1)
        self.assertEqual(run_data.parameters[0].value, "0.01")

    def test_set_param_type_conversion(self) -> None:
        """Test that parameter values are converted to strings."""
        run_data = run_metadata.RunMetadata(
            status=run_metadata.RunStatus.RUNNING,
            metrics=[],
            parameters=[],
        )

        # Test with different types
        run_data.set_param(key="int_param", value=42)
        run_data.set_param(key="float_param", value=3.14)
        run_data.set_param(key="bool_param", value=True)
        run_data.set_param(key="list_param", value=[1, 2, 3])

        self.assertEqual(len(run_data.parameters), 4)

        # Check that all values are converted to strings
        param_dict = {p.name: p.value for p in run_data.parameters}
        self.assertEqual(param_dict["int_param"], "42")
        self.assertEqual(param_dict["float_param"], "3.14")
        self.assertEqual(param_dict["bool_param"], "True")
        self.assertEqual(param_dict["list_param"], "[1, 2, 3]")


if __name__ == "__main__":
    absltest.main()
