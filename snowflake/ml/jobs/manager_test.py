import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from snowflake.ml.jobs import job_definition
from snowflake.ml.jobs._utils import constants, type_utils


class ManagerTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        (
            pathlib.PureWindowsPath(r"\mnt\job_stage\system\mljob_launcher.py"),
            pathlib.PureWindowsPath(r"\mnt\job_stage\app\entry.py"),
        ),
        (
            pathlib.PurePosixPath("/mnt/job_stage/system/mljob_launcher.py"),
            pathlib.PurePosixPath("/mnt/job_stage/app/entry.py"),
        ),
    )
    def test_job_definition_preserves_windows_absolute_paths(
        self, launcher_script: pathlib.PurePath, entrypoint: pathlib.PurePath
    ) -> None:
        uploaded_payload = type_utils.UploadedPayload(
            stage_path=pathlib.PurePosixPath("@payload_stage/job_id"),
            entrypoint=[
                launcher_script,
                entrypoint,
            ],
            env_vars={},
        )

        session = MagicMock()
        session.get_current_warehouse.return_value = "TEST_WH"
        session.get_current_database.return_value = "TEST_DB"
        session.get_current_schema.return_value = "TEST_SCHEMA"

        with patch(
            "snowflake.ml.jobs.job_definition.payload_utils.JobPayload",
            return_value=MagicMock(upload=MagicMock(return_value=uploaded_payload)),
        ), patch(
            "snowflake.ml.jobs.job_definition.runtime_env_utils.get_runtime_image",
            return_value="/snowflake/image/image_repo/test_image:2.1.4",
        ):
            definition: job_definition.MLJobDefinition[Any, Any] = job_definition.MLJobDefinition.register(
                source="entry.py",
                entrypoint="entry.py",
                compute_pool="POOL",
                stage_name="@payload_stage/job_id",
                session=session,
                # Avoid hitting runtime image discovery logic, which may require external access.
                runtime_environment="test_runtime",
            )

        self.assertEqual(
            [
                "/mnt/job_stage/system/mljob_launcher.py",
                "/mnt/job_stage/app/entry.py",
            ],
            definition.entrypoint_args,
        )

    def test_delete_job_skips_injection_stage_path(self) -> None:
        """delete_job must not run a REMOVE built from an unvalidated, attacker-controlled stage path.

        A submitter can smuggle REMOVE/SQL-injection characters into the stage volume source (e.g. via
        spec_overrides). delete_job runs with the *deleting* principal's privileges, so it must refuse
        to interpolate a malformed stage path into its REMOVE f-string (CWE-441 / CWE-89).
        """
        malicious_source = "@EVIL_DB.EVIL_SCH.EVIL_STAGE/x'; DROP TABLE victim; --"
        service_spec = {
            "spec": {
                "containers": [{"name": "main", "env": {}}],
                "volumes": [{"name": constants.STAGE_VOLUME_NAME, "source": malicious_source}],
            }
        }

        executed_sql: list[str] = []

        def sql_side_effect(query: str) -> MagicMock:
            executed_sql.append(query)
            return MagicMock()

        mock_session = MagicMock()
        mock_session.sql.side_effect = sql_side_effect

        mock_job = jobs.MLJob[None]("EVIL_DB.EVIL_SCH.JOB", session=mock_session)
        mock_job._service_spec_cached = service_spec

        with patch("snowflake.ml.jobs.manager.query_helper.run_query", return_value=[]):
            jobs.delete_job(mock_job)

        remove_statements = [q for q in executed_sql if q.strip().upper().startswith("REMOVE")]
        self.assertEqual(
            [],
            remove_statements,
            msg=f"delete_job executed a REMOVE built from an invalid stage path: {remove_statements}",
        )


if __name__ == "__main__":
    absltest.main()
