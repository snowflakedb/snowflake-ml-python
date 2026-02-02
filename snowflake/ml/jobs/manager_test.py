import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.ml.jobs import job_definition
from snowflake.ml.jobs._utils import types


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
        uploaded_payload = types.UploadedPayload(
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


if __name__ == "__main__":
    absltest.main()
