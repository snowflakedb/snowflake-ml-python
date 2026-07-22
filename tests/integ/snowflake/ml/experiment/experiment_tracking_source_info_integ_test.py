import json
from typing import Any, Optional

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment import _source_info
from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentTrackingSourceInfoIntegTest(ExperimentTrackingIntegTestBase):
    """End-to-end coverage for ADD RUN ... WITH (SOURCE_INFO = ...).

    These exercise the real server contract: the client renders a dollar-quoted
    JSON payload and the deployed server must accept it and create the run.
    """

    def _add_run(self, experiment_name: str, run_name: str, source_info: Optional[_source_info.SourceInfo]) -> None:
        self.exp.set_experiment(experiment_name=experiment_name)
        source_info_json = None
        if source_info is not None and not source_info.is_empty():
            source_info_json = json.dumps(source_info.to_json_dict())
        self.exp._sql_client.add_run(
            experiment_name=sql_identifier.SqlIdentifier(experiment_name),
            run_name=sql_identifier.SqlIdentifier(run_name),
            source_info=source_info_json,
        )

    def _read_run_metadata(self, experiment_name: str, run_name: str) -> dict[str, Any]:
        runs = self._session.sql(
            f"SHOW RUNS IN EXPERIMENT {self._db_name}.{self._schema_name}.{experiment_name}"
        ).collect()
        by_name = {row["name"]: row for row in runs}
        self.assertIn(run_name, by_name, f"Run {run_name} was not created")
        metadata: dict[str, Any] = json.loads(by_name[run_name]["metadata"])
        return metadata

    def test_add_run_with_full_source_info(self) -> None:
        experiment_name = "TEST_EXPERIMENT_SOURCE_INFO_FULL"
        run_name = "RUN_FULL"
        source_info = _source_info.SourceInfo(
            entry_point="train/main.py",
            git=_source_info.GitInfo(
                remote_url="https://github.com/snowflakedb/snowml.git",
                commit_hash="0123456789abcdef0123456789abcdef01234567",
                branch="main",
            ),
            snowflake_file_domain_type="workspace",
            snowflake_file_domain_name='USER$.PUBLIC."ML Runtime Testing"',
        )

        self._add_run(experiment_name, run_name, source_info)

        # The run must exist; reaching here means the server accepted the SOURCE_INFO clause.
        self._read_run_metadata(experiment_name, run_name)

    def test_add_run_with_partial_source_info(self) -> None:
        experiment_name = "TEST_EXPERIMENT_SOURCE_INFO_PARTIAL"
        run_name = "RUN_PARTIAL"
        # Only an entry point and a commit hash; remote_url and branch are omitted.
        source_info = _source_info.SourceInfo(
            entry_point="notebook.ipynb",
            git=_source_info.GitInfo(commit_hash="0123456789abcdef0123456789abcdef01234567"),
        )

        self._add_run(experiment_name, run_name, source_info)

        self._read_run_metadata(experiment_name, run_name)

    def test_add_run_without_source_info(self) -> None:
        experiment_name = "TEST_EXPERIMENT_SOURCE_INFO_NONE"
        run_name = "RUN_NONE"

        self._add_run(experiment_name, run_name, None)

        self._read_run_metadata(experiment_name, run_name)

    def test_add_run_with_empty_source_info(self) -> None:
        # An empty SourceInfo must behave identically to omitting it (no WITH clause).
        experiment_name = "TEST_EXPERIMENT_SOURCE_INFO_EMPTY"
        run_name = "RUN_EMPTY"

        self._add_run(experiment_name, run_name, _source_info.SourceInfo())

        self._read_run_metadata(experiment_name, run_name)


if __name__ == "__main__":
    absltest.main()
