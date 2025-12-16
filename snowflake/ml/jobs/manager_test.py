import json
import pathlib
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.ml.jobs import manager
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
    def test_do_submit_job_preserves_windows_absolute_paths(
        self, launcher_script: pathlib.PurePath, entrypoint: pathlib.PurePath
    ) -> None:
        payload = types.UploadedPayload(
            stage_path=pathlib.PurePosixPath("@payload_stage/job_id"),
            entrypoint=[
                launcher_script,
                entrypoint,
            ],
            env_vars={},
        )

        with patch(
            "snowflake.ml.jobs.manager.query_helper.run_query",
            return_value=[["JOB_ID"]],
        ) as run_query, patch("snowflake.ml.jobs.manager.get_job"):
            manager._do_submit_job(
                session=MagicMock(),
                payload=payload,
                args=None,
                env_vars={},
                spec_overrides={},
                compute_pool="POOL",
                job_id=None,
                external_access_integrations=None,
                query_warehouse=None,
                target_instances=1,
                min_instances=1,
                enable_metrics=True,
                use_async=True,
                runtime_environment=None,
            )

        spec_options = json.loads(run_query.call_args.kwargs["params"][2])
        self.assertEqual(
            [
                "/mnt/job_stage/system/mljob_launcher.py",
                "/mnt/job_stage/app/entry.py",
            ],
            spec_options["ARGS"],
        )


if __name__ == "__main__":
    absltest.main()
