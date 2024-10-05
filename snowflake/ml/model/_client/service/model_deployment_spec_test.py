import pathlib
import tempfile

import yaml
from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.service import model_deployment_spec


class ModelDeploymentSpecTest(absltest.TestCase):
    def test_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.save(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("service"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
                ingress_enabled=True,
                max_instances=1,
                gpu=None,
                num_workers=None,
                max_batch_rows=None,
                force_rebuild=False,
                external_access_integration=sql_identifier.SqlIdentifier("external_access_integration"),
            )

            file_path = mds.workspace_path / mds.DEPLOY_SPEC_FILE_REL_PATH
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "image_build": {
                            "compute_pool": "IMAGE_BUILD_COMPUTE_POOL",
                            "image_repo": "DB.SCHEMA.IMAGE_REPO",
                            "force_rebuild": False,
                            "external_access_integrations": ["EXTERNAL_ACCESS_INTEGRATION"],
                        },
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "SERVICE_COMPUTE_POOL",
                            "ingress_enabled": True,
                            "max_instances": 1,
                        },
                    },
                )

    def test_minimal_case_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.save(
                database_name=sql_identifier.SqlIdentifier("db", case_sensitive=True),
                schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("model", case_sensitive=True),
                version_name=sql_identifier.SqlIdentifier("version", case_sensitive=True),
                service_database_name=None,
                service_schema_name=None,
                service_name=sql_identifier.SqlIdentifier("service", case_sensitive=True),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier(
                    "image_build_compute_pool", case_sensitive=True
                ),
                service_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool", case_sensitive=True),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("image_repo", case_sensitive=True),
                ingress_enabled=True,
                max_instances=1,
                gpu=None,
                num_workers=None,
                max_batch_rows=None,
                force_rebuild=False,
                external_access_integration=sql_identifier.SqlIdentifier(
                    "external_access_integration", case_sensitive=True
                ),
            )

            file_path = mds.workspace_path / mds.DEPLOY_SPEC_FILE_REL_PATH
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": '"db"."schema"."model"', "version": '"version"'}],
                        "image_build": {
                            "compute_pool": '"image_build_compute_pool"',
                            "image_repo": '"db"."schema"."image_repo"',
                            "force_rebuild": False,
                            "external_access_integrations": ['"external_access_integration"'],
                        },
                        "service": {
                            "name": '"db"."schema"."service"',
                            "compute_pool": '"service_compute_pool"',
                            "ingress_enabled": True,
                            "max_instances": 1,
                        },
                    },
                )

    def test_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.save(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
                service_database_name=sql_identifier.SqlIdentifier("service_db"),
                service_schema_name=sql_identifier.SqlIdentifier("service_schema"),
                service_name=sql_identifier.SqlIdentifier("service"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                service_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                image_repo_database_name=sql_identifier.SqlIdentifier("image_repo_db"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("image_repo_schema"),
                image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
                ingress_enabled=True,
                max_instances=10,
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
                force_rebuild=True,
                external_access_integration=sql_identifier.SqlIdentifier("external_access_integration"),
            )

            file_path = mds.workspace_path / mds.DEPLOY_SPEC_FILE_REL_PATH
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "image_build": {
                            "compute_pool": "IMAGE_BUILD_COMPUTE_POOL",
                            "image_repo": "IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                            "force_rebuild": True,
                            "external_access_integrations": ["EXTERNAL_ACCESS_INTEGRATION"],
                        },
                        "service": {
                            "name": "SERVICE_DB.SERVICE_SCHEMA.SERVICE",
                            "compute_pool": "SERVICE_COMPUTE_POOL",
                            "ingress_enabled": True,
                            "max_instances": 10,
                            "gpu": "1",
                            "num_workers": 10,
                            "max_batch_rows": 1024,
                        },
                    },
                )


if __name__ == "__main__":
    absltest.main()
