from pathlib import PurePath, PurePosixPath
from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake.ml.jobs import job_definition
from snowflake.ml.jobs._utils import feature_flags, types


def _make_uploaded_payload() -> types.UploadedPayload:
    return types.UploadedPayload(
        stage_path=PurePosixPath("@payload_stage/entry"),
        entrypoint=[PurePath("/mnt/job_stage/app/launcher.py"), PurePath("/mnt/job_stage/app/entry.py")],
    )


class MLJobDefinitionTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = MagicMock()
        self.session.get_current_warehouse.return_value = "TEST_WH"
        self.session.get_current_database.return_value = "TEST_DB"
        self.session.get_current_schema.return_value = "TEST_SCHEMA"
        self.uploaded_payload = _make_uploaded_payload()

    def _expected_definition(self, runtime_value: str | None) -> job_definition.MLJobDefinition[[Any], Any]:
        spec_options = types.SpecOptions(
            stage_path=self.uploaded_payload.stage_path.as_posix(),
            args=None,
            enable_metrics=True,
            runtime=runtime_value,
            env_vars={},
            enable_stage_mount_v2=feature_flags.FeatureFlags.ENABLE_STAGE_MOUNT_V2.is_enabled(),
        )
        job_options = types.JobOptions(
            query_warehouse="TEST_WH",
            target_instances=1,
            min_instances=1,
            generate_suffix=False,
        )
        return job_definition.MLJobDefinition(
            job_options=job_options,
            spec_options=spec_options,
            stage_name=self.uploaded_payload.stage_path.as_posix(),
            compute_pool="E2E_TEST_POOL",
            entrypoint_args=[v.as_posix() if isinstance(v, PurePath) else v for v in self.uploaded_payload.entrypoint],
            session=self.session,
            database="TEST_DB",
            schema="TEST_SCHEMA",
            name="entry",
        )

    def test_register_with_runtime_env_passes_through_value(self) -> None:
        with patch(
            "snowflake.ml.jobs.job_definition.payload_utils.JobPayload",
            return_value=MagicMock(upload=MagicMock(return_value=self.uploaded_payload)),
        ), patch("snowflake.ml.jobs.job_definition.payload_utils.get_payload_name", return_value="entry"):
            result: job_definition.MLJobDefinition[[Any], Any] = job_definition.MLJobDefinition.register(
                source="entry.py",
                entrypoint="entry.py",
                compute_pool="E2E_TEST_POOL",
                stage_name="payload_stage",
                session=self.session,
                runtime_environment="test_flag",
                generate_suffix=False,
            )
        self.assertEqual(result.__dict__, self._expected_definition("test_flag").__dict__)


if __name__ == "__main__":
    absltest.main()
