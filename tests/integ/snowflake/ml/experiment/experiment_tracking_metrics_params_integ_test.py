import json

import pandas as pd
from absl.testing import absltest

from snowflake.ml.experiment import ExperimentTracking
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentMetricsParamsIntegTest(ExperimentTrackingIntegTestBase):
    def test_log_metrics_and_params_comprehensive(self) -> None:
        """Comprehensive test for logging metrics and parameters with various scenarios."""
        experiment_name = "TEST_EXPERIMENT_METRICS_PARAMS"
        run_name = "TEST_RUN_COMPREHENSIVE"

        # Set up experiment and run
        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            # Log single metric
            self.exp.log_metric("accuracy", 0.95, step=1)

            # Log multiple metrics at once
            batch_metrics = {"precision": 0.92, "recall": 0.88, "f1_score": 0.90}
            self.exp.log_metrics(batch_metrics, step=5)

            # Log same metric at different steps
            self.exp.log_metric("loss", 0.8, step=1)
            self.exp.log_metric("loss", 0.6, step=2)
            self.exp.log_metric("loss", 0.4, step=3)

            # Update metric at same step (should overwrite)
            self.exp.log_metric("accuracy", 0.97, step=1)  # Should overwrite 0.95

            # Log single parameter
            self.exp.log_param("learning_rate", 0.001)

            # Log multiple parameters of different types
            batch_params = {"batch_size": 32, "model_type": "transformer", "dropout": 0.1, "use_attention": True}
            self.exp.log_params(batch_params)

            # Update parameter (should overwrite)
            self.exp.log_param("batch_size", 64)  # Should overwrite 32

            # Log additional metric and parameter
            self.exp.log_metric("train_accuracy", 0.85, step=10)
            self.exp.log_param("optimizer", "adam")

        # Verify all data was logged correctly
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        self.assertEqual(len(runs), 1)

        # Parse and verify metadata
        metadata = json.loads(runs[0]["metadata"])

        # Verify run is finished
        self.assertEqual(metadata["status"], "FINISHED")

        # Check metrics
        experiment_fqn = f"{self._db_name}.{self._schema_name}.{experiment_name}"
        metrics = self._session.sql(
            f"SELECT * FROM TABLE(SYSTEM$GET_EXPERIMENT_RUN_METRICS('{experiment_fqn}', '{run_name}'))"
        ).collect()
        # Should have: accuracy(updated), precision, recall, f1_score, loss(3 steps), train_accuracy = 8 total
        self.assertEqual(len(metrics), 8)
        metric_dict = {(m["METRIC_NAME"], m["STEP"]): float(m["VALUE"]) for m in metrics}
        # Verify single metric (updated value)
        self.assertEqual(metric_dict[("accuracy", 1)], 0.97)  # Updated value

        # Verify batch metrics
        self.assertEqual(metric_dict[("precision", 5)], 0.92)
        self.assertEqual(metric_dict[("recall", 5)], 0.88)
        self.assertEqual(metric_dict[("f1_score", 5)], 0.90)

        # Verify multi-step metrics
        self.assertEqual(metric_dict[("loss", 1)], 0.8)
        self.assertEqual(metric_dict[("loss", 2)], 0.6)
        self.assertEqual(metric_dict[("loss", 3)], 0.4)

        # Verify additional metric
        self.assertEqual(metric_dict[("train_accuracy", 10)], 0.85)

        # Check parameters
        parameters = self._session.sql(
            f"""SHOW RUN PARAMETERS IN
                EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}
                RUN {run_name}"""
        ).collect()
        self.assertEqual(6, len(parameters))
        parameter_dict = {p["name"]: p["value"] for p in parameters}
        self.assertEqual(parameter_dict["learning_rate"], "0.001")
        self.assertEqual(parameter_dict["batch_size"], "64")
        self.assertEqual(parameter_dict["model_type"], "transformer")
        self.assertEqual(parameter_dict["dropout"], "0.1")
        self.assertEqual(parameter_dict["use_attention"], "True")
        self.assertEqual(parameter_dict["optimizer"], "adam")

    def test_log_metrics_params_auto_run_creation(self) -> None:
        """Test that logging metrics/params without an active run creates a new run automatically."""
        # Log metric - should create a new run automatically
        self.exp.log_metric("accuracy", 0.95)

        # Log parameter - should use the same auto-created run
        self.exp.log_param("model", "bert")

        # Log more data to the auto-created run
        self.exp.log_metrics({"precision": 0.92, "recall": 0.88}, step=2)
        self.exp.log_params({"epochs": 10, "learning_rate": 0.001})

        # Verify a run was created automatically in the default experiment
        self.assertEqual(self.exp._experiment.name, "DEFAULT")
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.DEFAULT").collect()
        self.assertEqual(len(runs), 1)

        # Parse and verify run is RUNNING
        self.assertEqual(json.loads(runs[0]["metadata"])["status"], "RUNNING")
        run_name = runs[0]["name"]

        # Check metrics
        experiment_fqn = f"{self._db_name}.{self._schema_name}.DEFAULT"
        metrics = self._session.sql(
            f"SELECT * FROM TABLE(SYSTEM$GET_EXPERIMENT_RUN_METRICS('{experiment_fqn}', '{run_name}'))"
        ).collect()
        self.assertEqual(len(metrics), 3)
        metric_dict = {(m["METRIC_NAME"], m["STEP"]): float(m["VALUE"]) for m in metrics}
        self.assertEqual(metric_dict[("accuracy", 0)], 0.95)
        self.assertEqual(metric_dict[("precision", 2)], 0.92)
        self.assertEqual(metric_dict[("recall", 2)], 0.88)

        # Check parameters
        parameters = self._session.sql(
            f"""SHOW RUN PARAMETERS IN
                EXPERIMENT {self._db_name}.{self._schema_name}.DEFAULT
                RUN {run_name}"""
        ).collect()
        self.assertEqual(len(parameters), 3)
        parameter_dict = {p["name"]: p["value"] for p in parameters}
        self.assertEqual(parameter_dict["model"], "bert")
        self.assertEqual(parameter_dict["epochs"], "10")
        self.assertEqual(parameter_dict["learning_rate"], "0.001")

        # End run
        self.exp.end_run()
        runs = self._session.sql(f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.DEFAULT").collect()
        self.assertEqual(len(runs), 1)
        self.assertEqual(json.loads(runs[0]["metadata"])["status"], "FINISHED")

    def test_list_metrics(self) -> None:
        """Test list_metrics returns a pivoted Snowpark DataFrame with the highest step value for each metric."""
        experiment_name = "TEST_EXPERIMENT_LIST_METRICS"

        self.exp.set_experiment(experiment_name=experiment_name)

        with self.exp.start_run(run_name="RUN_A"):
            self.exp.log_metric("accuracy", 0.80, step=1)
            self.exp.log_metric("accuracy", 0.90, step=2)
            self.exp.log_metric("accuracy", 0.95, step=3)
            self.exp.log_metric("loss", 0.50, step=1)
            self.exp.log_metric("loss", 0.20, step=2)
            self.exp.log_metric("loss", 0.05, step=3)
            self.exp.log_metric("f1", 0.90, step=1)

        with self.exp.start_run(run_name="RUN_B"):
            self.exp.log_metric("accuracy", 0.70, step=1)
            self.exp.log_metric("accuracy", 0.80, step=2)

        # list_metrics returns a Snowpark DataFrame; convert to pandas for assertion.
        # Columns appear in alphabetical order (SHOW RUN METRICS sorts by run_name, name).
        all_metrics_df = self.exp.list_metrics().to_pandas().set_index("run_name").sort_index()
        expected_all_metrics_df = (
            pd.DataFrame(
                [
                    {"run_name": "RUN_A", "accuracy": 0.95, "f1": 0.90, "loss": 0.05},
                    {"run_name": "RUN_B", "accuracy": 0.80, "f1": float("nan"), "loss": float("nan")},
                ]
            )
            .set_index("run_name")
            .sort_index()
        )
        pd.testing.assert_frame_equal(all_metrics_df, expected_all_metrics_df)

        # List metrics filtered to a single run
        run_a_metrics = self.exp.list_metrics(run_name="RUN_A").collect()
        self.assertEqual(len(run_a_metrics), 1)
        self.assertEqual(run_a_metrics[0]["run_name"], "RUN_A")
        self.assertAlmostEqual(run_a_metrics[0]["accuracy"], 0.95)
        self.assertAlmostEqual(run_a_metrics[0]["loss"], 0.05)
        self.assertAlmostEqual(run_a_metrics[0]["f1"], 0.90)

    def test_list_params(self) -> None:
        """Test list_params returns a pivoted Snowpark DataFrame with parameters per run."""
        experiment_name = "TEST_EXPERIMENT_LIST_PARAMS"

        self.exp.set_experiment(experiment_name=experiment_name)

        with self.exp.start_run(run_name="RUN_A"):
            self.exp.log_params({"learning_rate": 0.01, "batch_size": 32})
            self.exp.log_param("optimizer", "adam")

        with self.exp.start_run(run_name="RUN_B"):
            self.exp.log_param("learning_rate", 0.02)

        # list_params returns a Snowpark DataFrame; convert to pandas for assertion.
        # Columns appear in alphabetical order (SHOW RUN PARAMETERS sorts by run_name, name).
        all_params_df = self.exp.list_params().to_pandas().set_index("run_name").sort_index()
        expected_all_params_df = (
            pd.DataFrame(
                [
                    {"run_name": "RUN_A", "batch_size": "32", "learning_rate": "0.01", "optimizer": "adam"},
                    {
                        "run_name": "RUN_B",
                        "batch_size": float("nan"),
                        "learning_rate": "0.02",
                        "optimizer": float("nan"),
                    },
                ]
            )
            .set_index("run_name")
            .sort_index()
        )
        pd.testing.assert_frame_equal(all_params_df, expected_all_params_df)

        # List params filtered to a single run
        run_a_params = self.exp.list_params(run_name="RUN_A").collect()
        self.assertEqual(len(run_a_params), 1)
        self.assertEqual(run_a_params[0]["run_name"], "RUN_A")
        self.assertEqual(run_a_params[0]["learning_rate"], "0.01")
        self.assertEqual(run_a_params[0]["batch_size"], "32")
        self.assertEqual(run_a_params[0]["optimizer"], "adam")

    def test_search_and_filter_metrics_and_params(self) -> None:
        """Test joining metrics and params DataFrames, filtering across both, and handling empty runs."""
        from snowflake.snowpark import functions as F

        experiment_name = "TEST_EXPERIMENT_SEARCH_AND_FILTER"
        self.exp.set_experiment(experiment_name=experiment_name)

        # Empty run with no metrics or params
        with self.exp.start_run(run_name="RUN_EMPTY"):
            pass

        # Empty run still appears with 1 row but no attribute columns
        empty_metrics = self.exp.list_metrics().collect()
        empty_params = self.exp.list_params().collect()
        self.assertEqual(len(empty_metrics), 1)
        self.assertEqual(empty_metrics[0]["run_name"], "RUN_EMPTY")
        self.assertEqual(len(empty_params), 1)
        self.assertEqual(empty_params[0]["run_name"], "RUN_EMPTY")

        # Populate runs with metrics and params
        with self.exp.start_run(run_name="RUN_A"):
            self.exp.log_metric("accuracy", 0.95, step=1)
            self.exp.log_metric("loss", 0.05, step=1)
            self.exp.log_params({"optimizer": "adam", "learning_rate": 0.01})

        with self.exp.start_run(run_name="RUN_B"):
            self.exp.log_metric("accuracy", 0.80, step=1)
            self.exp.log_metric("loss", 0.20, step=1)
            self.exp.log_params({"optimizer": "sgd", "learning_rate": 0.1})

        with self.exp.start_run(run_name="RUN_C"):
            self.exp.log_metric("accuracy", 0.92, step=1)
            self.exp.log_metric("loss", 0.08, step=1)
            self.exp.log_params({"optimizer": "adam", "learning_rate": 0.001})

        # All 4 runs appear (including empty one with NULLs)
        metrics_df = self.exp.list_metrics()
        params_df = self.exp.list_params()
        self.assertEqual(len(metrics_df.collect()), 4)
        self.assertEqual(len(params_df.collect()), 4)

        # Join metrics and params into a single view per run
        runs_df = metrics_df.join(params_df, on='"run_name"')

        # Filter: optimizer == "adam" AND accuracy > 0.90
        filtered = (
            runs_df.filter((F.col('"optimizer"') == "adam") & (F.col('"accuracy"') > 0.90)).sort('"run_name"').collect()
        )
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["run_name"], "RUN_A")
        self.assertEqual(filtered[1]["run_name"], "RUN_C")

        # Filter: learning_rate < 0.01 (param is a string, cast to float for comparison)
        low_lr = runs_df.filter(F.col('"learning_rate"').cast("float") < 0.01).collect()
        self.assertEqual(len(low_lr), 1)
        self.assertEqual(low_lr[0]["run_name"], "RUN_C")

        # Filter: loss < 0.10
        low_loss = runs_df.filter(F.col('"loss"') < 0.10).sort('"run_name"').collect()
        self.assertEqual(len(low_loss), 2)
        self.assertEqual(low_loss[0]["run_name"], "RUN_A")
        self.assertEqual(low_loss[1]["run_name"], "RUN_C")

    def test_list_metrics_and_params_no_experiment_raises(self) -> None:
        """Test that list_metrics/list_params raise without an experiment."""
        ExperimentTracking._instance = None
        exp_no_experiment = ExperimentTracking(
            self._session, database_name=self._db_name, schema_name=self._schema_name
        )
        with self.assertRaises(RuntimeError):
            exp_no_experiment.list_metrics()
        with self.assertRaises(RuntimeError):
            exp_no_experiment.list_params()


if __name__ == "__main__":
    absltest.main()
