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
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
                ingress_enabled=True,
                max_instances=1,
                cpu=None,
                memory=None,
                gpu=None,
                num_workers=None,
                max_batch_rows=None,
                force_rebuild=False,
                external_access_integrations=[sql_identifier.SqlIdentifier("external_access_integration")],
            )
            assert mds.workspace_path
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

    def test_minimal_inline_yaml(self) -> None:
        mds = model_deployment_spec.ModelDeploymentSpec()
        yaml_str = mds.save(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
            service_database_name=None,
            service_schema_name=None,
            service_name=sql_identifier.SqlIdentifier("service"),
            image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
            image_repo_database_name=None,
            image_repo_schema_name=None,
            image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
            ingress_enabled=True,
            max_instances=1,
            cpu=None,
            memory=None,
            gpu=None,
            num_workers=None,
            max_batch_rows=None,
            force_rebuild=False,
            external_access_integrations=[sql_identifier.SqlIdentifier("external_access_integration")],
        )
        assert yaml_str
        result = yaml.safe_load(yaml_str)
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
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool", case_sensitive=True),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("image_repo", case_sensitive=True),
                ingress_enabled=True,
                max_instances=1,
                cpu=None,
                memory=None,
                gpu=None,
                num_workers=None,
                max_batch_rows=None,
                force_rebuild=False,
                external_access_integrations=[
                    sql_identifier.SqlIdentifier("external_access_integration", case_sensitive=True)
                ],
            )
            assert mds.workspace_path
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
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                image_repo_database_name=sql_identifier.SqlIdentifier("image_repo_db"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("image_repo_schema"),
                image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
                ingress_enabled=True,
                max_instances=10,
                cpu="1",
                memory="1GiB",
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("external_access_integration")],
            )
            assert mds.workspace_path
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
                            "cpu": "1",
                            "memory": "1GiB",
                            "gpu": "1",
                            "num_workers": 10,
                            "max_batch_rows": 1024,
                        },
                    },
                )

    def test_no_eai(self) -> None:
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
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                image_repo_database_name=None,
                image_repo_schema_name=None,
                image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
                ingress_enabled=True,
                max_instances=1,
                cpu=None,
                memory=None,
                gpu=None,
                num_workers=None,
                max_batch_rows=None,
                force_rebuild=False,
                external_access_integrations=None,
            )
            assert mds.workspace_path
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
                        },
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "SERVICE_COMPUTE_POOL",
                            "ingress_enabled": True,
                            "max_instances": 1,
                        },
                    },
                )

    def test_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.save(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
                job_database_name=sql_identifier.SqlIdentifier("job_db"),
                job_schema_name=sql_identifier.SqlIdentifier("job_schema"),
                job_name=sql_identifier.SqlIdentifier("job"),
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("job_compute_pool"),
                image_repo_database_name=sql_identifier.SqlIdentifier("image_repo_db"),
                image_repo_schema_name=sql_identifier.SqlIdentifier("image_repo_schema"),
                image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
                cpu="1",
                memory="1GiB",
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("external_access_integration")],
                warehouse=sql_identifier.SqlIdentifier("warehouse"),
                target_method="predict",
                input_table_database_name=sql_identifier.SqlIdentifier("input_table_db"),
                input_table_schema_name=sql_identifier.SqlIdentifier("input_table_schema"),
                input_table_name=sql_identifier.SqlIdentifier("input_table"),
                output_table_database_name=sql_identifier.SqlIdentifier("output_table_db"),
                output_table_schema_name=sql_identifier.SqlIdentifier("output_table_schema"),
                output_table_name=sql_identifier.SqlIdentifier("output_table"),
            )
            assert mds.workspace_path
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
                        "job": {
                            "name": "JOB_DB.JOB_SCHEMA.JOB",
                            "compute_pool": "JOB_COMPUTE_POOL",
                            "cpu": "1",
                            "memory": "1GiB",
                            "gpu": "1",
                            "num_workers": 10,
                            "max_batch_rows": 1024,
                            "warehouse": "WAREHOUSE",
                            "target_method": "predict",
                            "input_table_name": "INPUT_TABLE_DB.INPUT_TABLE_SCHEMA.INPUT_TABLE",
                            "output_table_name": "OUTPUT_TABLE_DB.OUTPUT_TABLE_SCHEMA.OUTPUT_TABLE",
                        },
                    },
                )


if __name__ == "__main__":
    absltest.main()
