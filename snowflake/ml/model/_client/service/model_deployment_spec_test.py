import pathlib
import tempfile
from typing import Union, cast

import yaml
from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.feature_store import feature_view
from snowflake.ml.model import inference_engine
from snowflake.ml.model._client.service import model_deployment_spec


class ModelDeploymentSpecTest(parameterized.TestCase):
    def test_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=5,
                autocapture=None,
            )

            assert mds.workspace_path
            file_path = mds.save()
            with open(file_path, encoding="utf-8") as f:
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
                            "min_instances": 1,
                            "max_instances": 5,
                        },
                    },
                )

    def test_minimal_inline_yaml(self) -> None:
        mds = model_deployment_spec.ModelDeploymentSpec()  # No workspace path
        mds.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        mds.add_image_build_spec(
            image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
            fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
        )
        mds.add_service_spec(
            service_name=sql_identifier.SqlIdentifier("service"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
            ingress_enabled=True,
            min_instances=1,
            max_instances=5,
        )
        yaml_str = mds.save()

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
                },
                "service": {
                    "name": "DB.SCHEMA.SERVICE",
                    "compute_pool": "SERVICE_COMPUTE_POOL",
                    "ingress_enabled": True,
                    "min_instances": 1,
                    "max_instances": 5,
                },
            },
        )

    def test_minimal_case_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db", case_sensitive=True),
                schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("model", case_sensitive=True),
                version_name=sql_identifier.SqlIdentifier("version", case_sensitive=True),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier(
                    "image_build_compute_pool", case_sensitive=True
                ),
                fully_qualified_image_repo_name='"db"."schema"."image_repo"',
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service", case_sensitive=True),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool", case_sensitive=True),
                ingress_enabled=True,
                min_instances=1,
                max_instances=6,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
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
                        },
                        "service": {
                            "name": '"db"."schema"."service"',
                            "compute_pool": '"service_compute_pool"',
                            "ingress_enabled": True,
                            "min_instances": 1,
                            "max_instances": 6,
                        },
                    },
                )

    @parameterized.parameters(  # type: ignore[misc]
        {
            "force_rebuild": True,
            "ingress_enabled": True,
        },
        {
            "force_rebuild": False,
            "ingress_enabled": False,
        },
    )
    def test_full(
        self,
        force_rebuild: bool,
        ingress_enabled: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                force_rebuild=force_rebuild,
                external_access_integrations=[sql_identifier.SqlIdentifier("external_access_integration")],
            )
            mds.add_service_spec(
                service_database_name=sql_identifier.SqlIdentifier("service_db"),
                service_schema_name=sql_identifier.SqlIdentifier("service_schema"),
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=ingress_enabled,
                min_instances=0,
                max_instances=10,
                cpu="1",
                memory="1GiB",
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
                autocapture=True,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "image_build": {
                            "compute_pool": "IMAGE_BUILD_COMPUTE_POOL",
                            "image_repo": "IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                            "force_rebuild": force_rebuild,
                            "external_access_integrations": ["EXTERNAL_ACCESS_INTEGRATION"],
                        },
                        "service": {
                            "name": "SERVICE_DB.SERVICE_SCHEMA.SERVICE",
                            "compute_pool": "SERVICE_COMPUTE_POOL",
                            "ingress_enabled": ingress_enabled,
                            "min_instances": 0,
                            "max_instances": 10,
                            "cpu": "1",
                            "memory": "1GiB",
                            "gpu": "1",
                            "num_workers": 10,
                            "max_batch_rows": 1024,
                            "autocapture": True,
                        },
                    },
                )

    def test_no_eai(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
                external_access_integrations=None,  # Explicitly None
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=7,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
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
                            # external_access_integrations should be omitted
                        },
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "SERVICE_COMPUTE_POOL",
                            "ingress_enabled": True,
                            "min_instances": 1,
                            "max_instances": 7,
                        },
                    },
                )

    def test_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                force_rebuild=True,
                external_access_integrations=[sql_identifier.SqlIdentifier("external_access_integration")],
            )
            mds.add_job_spec(
                job_database_name=sql_identifier.SqlIdentifier("job_db"),
                job_schema_name=sql_identifier.SqlIdentifier("job_schema"),
                job_name=sql_identifier.SqlIdentifier("job"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("job_compute_pool"),
                cpu="1",
                memory="1GiB",
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
                warehouse=sql_identifier.SqlIdentifier("warehouse"),
                function_name="function_name",
                input_stage_location="input_stage_location",
                output_stage_location="output_stage_location",
                completion_filename="completion_filename",
                input_file_pattern="*",
                column_handling=None,
                params=None,
                block=False,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
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
                            "function_name": "function_name",
                            "input": {
                                "input_stage_location": "input_stage_location",
                                "input_file_pattern": "*",
                            },
                            "output": {
                                "output_stage_location": "output_stage_location",
                                "completion_filename": "completion_filename",
                            },
                            # TODO(SNOW-3321349): Add "sync": False once server-side support is rolled out.
                        },
                    },
                )

    def test_job_with_name_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_job_spec(
                job_database_name=sql_identifier.SqlIdentifier("job_db"),
                job_schema_name=sql_identifier.SqlIdentifier("job_schema"),
                job_name=sql_identifier.SqlIdentifier("job"),
                name_prefix="CUSTOM_PREFIX",
                inference_compute_pool_name=sql_identifier.SqlIdentifier("job_compute_pool"),
                warehouse=sql_identifier.SqlIdentifier("warehouse"),
                function_name="function_name",
                input_stage_location="input_stage_location",
                output_stage_location="output_stage_location",
                completion_filename="completion_filename",
                input_file_pattern="*",
                block=False,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "image_build": {
                            "compute_pool": "IMAGE_BUILD_COMPUTE_POOL",
                            "force_rebuild": False,
                            "image_repo": "DB.SCHEMA.IMAGE_REPO",
                        },
                        "job": {
                            "name": "JOB_DB.JOB_SCHEMA.JOB",
                            "name_prefix": "CUSTOM_PREFIX",
                            "compute_pool": "JOB_COMPUTE_POOL",
                            "warehouse": "WAREHOUSE",
                            "function_name": "function_name",
                            "input": {
                                "input_stage_location": "input_stage_location",
                                "input_file_pattern": "*",
                            },
                            "output": {
                                "output_stage_location": "output_stage_location",
                                "completion_filename": "completion_filename",
                            },
                            # TODO(SNOW-3321349): Add "sync": False once server-side support is rolled out.
                        },
                    },
                )

    def test_job_with_name_prefix_only(self) -> None:
        """Test job spec with name_prefix but no job_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_job_spec(
                name_prefix="CUSTOM_PREFIX",
                inference_compute_pool_name=sql_identifier.SqlIdentifier("job_compute_pool"),
                warehouse=sql_identifier.SqlIdentifier("warehouse"),
                function_name="function_name",
                input_stage_location="input_stage_location",
                output_stage_location="output_stage_location",
                completion_filename="completion_filename",
                input_file_pattern="*",
                block=False,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                # name should be omitted (None excluded)
                self.assertNotIn("name", result["job"])
                self.assertEqual(result["job"]["name_prefix"], "CUSTOM_PREFIX")

    def test_hf_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=8,
            )
            mds.add_hf_logger_spec(
                hf_model_name="hf_model",
                hf_task="hf_task",
                hf_token="hf_token",
                hf_tokenizer="hf_tokenizer",
                # log model args
                pip_requirements=["torch", "transformers"],
                conda_dependencies=["python", "numpy"],
                comment="comment",
                target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
                # kwargs
                trust_remote_code=False,
                max_tokens=100,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
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
                            "min_instances": 1,
                            "max_instances": 8,
                        },
                        "model_loggings": [
                            {
                                "hf_model": {
                                    "hf_model_name": "hf_model",
                                    "task": "hf_task",
                                    "token": "hf_token",
                                    "tokenizer": "hf_tokenizer",
                                    "trust_remote_code": False,
                                    "hf_model_kwargs": '{"trust_remote_code": false, "max_tokens": 100}',
                                },
                                "log_model_args": {
                                    "pip_requirements": ["torch", "transformers"],
                                    "conda_dependencies": ["python", "numpy"],
                                    "target_platforms": ["SNOWPARK_CONTAINER_SERVICES"],
                                    "comment": "comment",
                                },
                            },
                        ],
                    },
                )

    def test_hf_config_without_hf_model_name_raises(self) -> None:
        mds = model_deployment_spec.ModelDeploymentSpec()
        # Add other required specs first
        mds.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        mds.add_image_build_spec(
            image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
            fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
        )
        mds.add_service_spec(
            service_name=sql_identifier.SqlIdentifier("service"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
        )
        # Now try adding HF logger spec without model name
        with self.assertRaisesRegex(
            ValueError, "Hugging Face model name is required when using Hugging Face model deployment"
        ):
            mds.add_hf_logger_spec(
                hf_model_name="",  # Empty or None should fail
                hf_task="hf_task",
                hf_token="hf_token",
                hf_tokenizer="hf_tokenizer",
            )

    def test_missing_model_spec_raises(self) -> None:
        mds = model_deployment_spec.ModelDeploymentSpec()
        mds.add_image_build_spec(
            image_build_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            fully_qualified_image_repo_name="db.schema.repo",
        )
        mds.add_service_spec(
            service_name=sql_identifier.SqlIdentifier("service"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            service_database_name=sql_identifier.SqlIdentifier("db"),
            service_schema_name=sql_identifier.SqlIdentifier("schema"),
        )
        with self.assertRaisesRegex(ValueError, "Model specification is required"):
            mds.save()

    def test_missing_image_build_spec_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "POOL",
                            "ingress_enabled": True,
                            "min_instances": 0,
                            "max_instances": 1,
                        },
                    },
                )

    def test_missing_service_or_job_spec_raises(self) -> None:
        mds = model_deployment_spec.ModelDeploymentSpec()
        mds.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        mds.add_image_build_spec(
            image_build_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            fully_qualified_image_repo_name=sql_identifier.SqlIdentifier("repo"),
        )
        with self.assertRaisesRegex(ValueError, "Either service or job specification is required"):
            mds.save()

    def test_both_service_and_job_spec_raises_on_add(self) -> None:
        mds_service = model_deployment_spec.ModelDeploymentSpec()
        mds_service.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        mds_service.add_service_spec(
            service_name=sql_identifier.SqlIdentifier("service"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
        )
        with self.assertRaisesRegex(ValueError, "Cannot add a job spec when a service spec already exists"):
            mds_service.add_job_spec(
                job_name=sql_identifier.SqlIdentifier("job"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
                warehouse=sql_identifier.SqlIdentifier("wh"),
                function_name="function_name",
                input_stage_location="input_stage_location",
                output_stage_location="output_stage_location",
                completion_filename="completion_filename",
                input_file_pattern="*",
                column_handling=None,
                params=None,
                block=False,
            )

        mds_job = model_deployment_spec.ModelDeploymentSpec()
        mds_job.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        mds_job.add_job_spec(
            job_name=sql_identifier.SqlIdentifier("job"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            warehouse=sql_identifier.SqlIdentifier("wh"),
            function_name="function_name",
            input_stage_location="input_stage_location",
            output_stage_location="output_stage_location",
            completion_filename="completion_filename",
            input_file_pattern="*",
            column_handling=None,
            params=None,
            num_workers=1,
            block=False,
        )
        with self.assertRaisesRegex(ValueError, "Cannot add a service spec when a job spec already exists"):
            mds_job.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            )

    def test_clear_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            self.assertLen(mds._models, 1)
            mds.clear()
            self.assertLen(mds._models, 0)

    def test_image_build_spec_minimal_params(self) -> None:
        """Test add_image_build_spec with only required parameter and all optional parameters as None/default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=None,  # Explicitly None
                fully_qualified_image_repo_name=None,  # Explicitly None
                force_rebuild=False,  # Default value
                external_access_integrations=None,  # Explicitly None
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=9,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "SERVICE_COMPUTE_POOL",
                            "ingress_enabled": True,
                            "min_instances": 1,
                            "max_instances": 9,
                        },
                    },
                )

    def test_inference_engine_spec_with_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            )
            mds.add_inference_engine_spec(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args=[
                    "--some_vllm_arg=0.8",
                    "--tensor_parallel_size=2",
                ],
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "image_build": {
                            "compute_pool": "POOL",
                            "force_rebuild": False,
                            "image_repo": "DB.SCHEMA.REPO",
                        },
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "POOL",
                            "ingress_enabled": True,
                            "min_instances": 0,
                            "max_instances": 1,
                            "inference_engine_spec": {
                                "inference_engine_name": "vllm",
                                "inference_engine_args": [
                                    "--some_vllm_arg=0.8",
                                    "--tensor_parallel_size=2",
                                ],
                            },
                        },
                    },
                )
        mds.clear()

    def test_inference_engine_spec_with_service_blocklist_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            )
            mds.add_inference_engine_spec(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args=[
                    "--some_vllm_arg=0.8",
                    "--host=host",
                    "--port=8000",
                ],
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "image_build": {
                            "compute_pool": "POOL",
                            "force_rebuild": False,
                            "image_repo": "DB.SCHEMA.REPO",
                        },
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "POOL",
                            "ingress_enabled": True,
                            "min_instances": 0,
                            "max_instances": 1,
                            "inference_engine_spec": {
                                "inference_engine_name": "vllm",
                                "inference_engine_args": ["--some_vllm_arg=0.8"],
                            },
                        },
                    },
                )
        mds.clear()

    def test_inference_engine_spec_with_service_skip_image_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            )
            mds.add_inference_engine_spec(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args=["--some_vllm_arg=0.8", "--host=host", "--port=8000"],
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "service": {
                            "name": "DB.SCHEMA.SERVICE",
                            "compute_pool": "POOL",
                            "ingress_enabled": True,
                            "min_instances": 0,
                            "max_instances": 1,
                            "inference_engine_spec": {
                                "inference_engine_name": "vllm",
                                "inference_engine_args": ["--some_vllm_arg=0.8"],
                            },
                        },
                    },
                )
        mds.clear()

    def test_inference_engine_spec_with_job(self) -> None:
        """Test add_inference_engine_spec works with job spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_job_spec(
                job_name=sql_identifier.SqlIdentifier("batch_job"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
                function_name="predict",
                input_stage_location="@input_stage/",
                output_stage_location="@output_stage/",
                completion_filename="_SUCCESS",
                input_file_pattern="*.parquet",
                warehouse=sql_identifier.SqlIdentifier("warehouse"),
                gpu="4",
                replicas=2,
                block=False,
            )
            mds.add_inference_engine_spec(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args=[
                    "--tensor_parallel_size=4",
                    "--max-model-len=4096",
                ],
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                self.assertDictEqual(
                    result,
                    {
                        "models": [{"name": "DB.SCHEMA.MODEL", "version": "VERSION"}],
                        "job": {
                            "name": "DB.SCHEMA.BATCH_JOB",
                            "compute_pool": "POOL",
                            "warehouse": "WAREHOUSE",
                            "function_name": "predict",
                            "gpu": "4",
                            "input": {
                                "input_stage_location": "@input_stage/",
                                "input_file_pattern": "*.parquet",
                            },
                            "output": {
                                "output_stage_location": "@output_stage/",
                                "completion_filename": "_SUCCESS",
                            },
                            "replicas": 2,
                            # TODO(SNOW-3321349): Add "sync": False once server-side support is rolled out.
                            "inference_engine_spec": {
                                "inference_engine_name": "vllm",
                                "inference_engine_args": [
                                    "--tensor_parallel_size=4",
                                    "--max-model-len=4096",
                                ],
                            },
                        },
                    },
                )
        mds.clear()

    def test_inference_engine_spec_requires_service_or_job(self) -> None:
        """Test add_inference_engine_spec raises error when called before add_service_spec or add_job_spec."""
        mds = model_deployment_spec.ModelDeploymentSpec()
        mds.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        with self.assertRaises(ValueError) as cm:
            mds.add_inference_engine_spec(
                inference_engine=inference_engine.InferenceEngine.VLLM,
                inference_engine_args=["--tensor_parallel_size=2"],
            )

        self.assertIn(
            "Inference engine specification must be called after add_service_spec() or add_job_spec().",
            str(cm.exception),
        )

    def test_inference_engine_spec_with_job_skip_image_build(self) -> None:
        """Test that job spec with inference engine skips image build and produces valid yaml."""
        mds = model_deployment_spec.ModelDeploymentSpec()  # No workspace path - inline yaml
        mds.add_model_spec(
            database_name=sql_identifier.SqlIdentifier("db"),
            schema_name=sql_identifier.SqlIdentifier("schema"),
            model_name=sql_identifier.SqlIdentifier("model"),
            version_name=sql_identifier.SqlIdentifier("version"),
        )
        mds.add_job_spec(
            job_name=sql_identifier.SqlIdentifier("batch_job"),
            inference_compute_pool_name=sql_identifier.SqlIdentifier("pool"),
            function_name="predict",
            input_stage_location="@input_stage/",
            output_stage_location="@output_stage/",
            completion_filename="_SUCCESS",
            input_file_pattern="*.parquet",
            warehouse=sql_identifier.SqlIdentifier("warehouse"),
            block=False,
        )
        mds.add_inference_engine_spec(
            inference_engine=inference_engine.InferenceEngine.VLLM,
            inference_engine_args=None,  # No args
        )
        yaml_str = mds.save()

        assert yaml_str
        result = yaml.safe_load(yaml_str)
        # Verify no image_build key present
        self.assertNotIn("image_build", result)
        # Verify inference_engine_spec is in job
        self.assertIn("inference_engine_spec", result["job"])
        self.assertEqual(result["job"]["inference_engine_spec"]["inference_engine_name"], "vllm")
        self.assertEqual(result["job"]["inference_engine_spec"]["inference_engine_args"], [])

    def test_job_with_partition_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="IMAGE_REPO_DB.IMAGE_REPO_SCHEMA.IMAGE_REPO",
                force_rebuild=True,
            )
            mds.add_job_spec(
                job_database_name=sql_identifier.SqlIdentifier("job_db"),
                job_schema_name=sql_identifier.SqlIdentifier("job_schema"),
                job_name=sql_identifier.SqlIdentifier("job"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("job_compute_pool"),
                cpu="1",
                memory="1GiB",
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
                warehouse=sql_identifier.SqlIdentifier("warehouse"),
                function_name="function_name",
                input_stage_location="input_stage_location",
                output_stage_location="output_stage_location",
                completion_filename="completion_filename",
                input_file_pattern="*",
                column_handling=None,
                params=None,
                partition_columns=["PARTITION_COL"],
                block=False,
            )
            file_path_str = mds.save()

            assert mds.workspace_path
            file_path = pathlib.Path(file_path_str)
            with file_path.open("r", encoding="utf-8") as f:
                result = yaml.safe_load(f)
                # Verify partition_columns appears under job.input
                self.assertEqual(result["job"]["input"]["partition_columns"], ["PARTITION_COL"])

    def _make_fake_feature_view(
        self,
        *,
        database: str = "FS_DB",
        schema: str = "FS_SCHEMA",
        name: str = "USER_FEATURES",
        version: str = "V1",
    ) -> feature_view.FeatureView:
        # Returns a duck-typed stand-in for FeatureView. We construct the real type
        # via cast() so call sites stay typed; instantiating a real FeatureView
        # would require a Session and a registered entity, which is overkill for
        # the YAML-shape assertions exercised here. The runtime shape used by
        # _build_feature_retrieval_config is .database/.schema/.name (each
        # .identifier()) and .version, all of which _FakeFeatureView provides.
        return cast(
            feature_view.FeatureView,
            _FakeFeatureView(database=database, schema=schema, name=name, version=version),
        )

    def test_service_spec_with_feature_sources(self) -> None:
        # End-to-end YAML shape: feature_sources_per_function lands in
        # service.feature_retrieval.lookups with logical 3-part source name,
        # type "oft_vnext", and version in its own field.
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=5,
                feature_sources_per_function={
                    "predict": [self._make_fake_feature_view()],
                },
            )
            file_path = mds.save()
            with open(file_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
            self.assertEqual(
                result["service"]["feature_retrieval"],
                {
                    "lookups": {
                        "predict": [
                            {
                                "source": "FS_DB.FS_SCHEMA.USER_FEATURES",
                                "type": "oft_vnext",
                                "version": "V1",
                            }
                        ]
                    }
                },
            )

    def test_service_spec_without_feature_sources_omits_block(self) -> None:
        # Default path: when feature_sources_per_function is None (or not passed),
        # the feature_retrieval key MUST be absent from the YAML, otherwise GS
        # would route the deploy through the FR validator unnecessarily.
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=5,
            )
            file_path = mds.save()
            with open(file_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
            self.assertNotIn("feature_retrieval", result["service"])

    def test_service_spec_unregistered_feature_view_raises(self) -> None:
        # FeatureView with version=None is the canonical "unregistered" shape;
        # we surface this client-side so the user gets a localized error
        # instead of an opaque deploy-time GS validation failure.
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            unregistered = cast(
                feature_view.FeatureView,
                _FakeFeatureView(database="FS_DB", schema="FS_SCHEMA", name="UF", version=None),
            )
            with self.assertRaisesRegex(ValueError, "is not registered"):
                mds.add_service_spec(
                    service_name=sql_identifier.SqlIdentifier("service"),
                    inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                    ingress_enabled=True,
                    min_instances=1,
                    max_instances=5,
                    feature_sources_per_function={"predict": [unregistered]},
                )

    def test_service_spec_with_feature_sources_multi_method(self) -> None:
        # Multi-method coverage: each function gets its own lookups bucket
        # and per-bucket order is preserved.
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=5,
                feature_sources_per_function={
                    "predict": [self._make_fake_feature_view(name="USER_FEATURES", version="V1")],
                    "explain": [self._make_fake_feature_view(name="ITEM_FEATURES", version="V2")],
                },
            )
            file_path = mds.save()
            with open(file_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
            self.assertEqual(
                result["service"]["feature_retrieval"]["lookups"],
                {
                    "predict": [
                        {"source": "FS_DB.FS_SCHEMA.USER_FEATURES", "type": "oft_vnext", "version": "V1"},
                    ],
                    "explain": [
                        {"source": "FS_DB.FS_SCHEMA.ITEM_FEATURES", "type": "oft_vnext", "version": "V2"},
                    ],
                },
            )

    def test_service_spec_feature_source_uses_logical_name_not_physical_fqn(self) -> None:
        # Regression guard: the wire `source` MUST be the logical 3-part name
        # (DB.SCHEMA.NAME), NOT FeatureView.fully_qualified_name() which appends
        # `$<version>$ONLINE` and points at the physical OFT online table. A future
        # refactor that switches to fully_qualified_name() must fail this test loudly.
        fv = self._make_fake_feature_view(name="USER_FEATURES", version="V1")
        # Stub fully_qualified_name() to a recognizable, wrong shape so any
        # accidental use of it leaks into `source` and trips the assertion.
        fv.fully_qualified_name = lambda: "FS_DB.FS_SCHEMA.USER_FEATURES$V1$ONLINE"  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=5,
                feature_sources_per_function={"predict": [fv]},
            )
            file_path = mds.save()
            with open(file_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
            source = result["service"]["feature_retrieval"]["lookups"]["predict"][0]["source"]
        self.assertEqual(source, "FS_DB.FS_SCHEMA.USER_FEATURES")
        self.assertNotIn("$", source, f"source must not contain $version$ONLINE, got {source!r}")

    def test_service_spec_feature_source_preserves_case_sensitive_identifier(self) -> None:
        # Case-sensitive FV name (e.g. created with "lowercase") must round-trip
        # through SqlIdentifier.identifier() as a quoted form so the server
        # resolves the same physical object.
        fv = cast(
            feature_view.FeatureView,
            _FakeFeatureView(
                database="FS_DB",
                schema="FS_SCHEMA",
                name=sql_identifier.SqlIdentifier('"user_features"', case_sensitive=False),
                version="V1",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            mds.add_service_spec(
                service_name=sql_identifier.SqlIdentifier("service"),
                inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                ingress_enabled=True,
                min_instances=1,
                max_instances=5,
                feature_sources_per_function={"predict": [fv]},
            )
            file_path = mds.save()
            with open(file_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
        self.assertEqual(
            result["service"]["feature_retrieval"]["lookups"]["predict"][0]["source"],
            'FS_DB.FS_SCHEMA."user_features"',
        )

    def test_service_spec_empty_feature_sources_dict_raises(self) -> None:
        # Empty dict is treated symmetrically with empty list per method:
        # both surface a ValueError client-side instead of silently producing
        # a service spec without the feature_retrieval block.
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            with self.assertRaisesRegex(ValueError, "feature_sources_per_function is empty"):
                mds.add_service_spec(
                    service_name=sql_identifier.SqlIdentifier("service"),
                    inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                    ingress_enabled=True,
                    min_instances=1,
                    max_instances=5,
                    feature_sources_per_function={},
                )

    def test_service_spec_empty_feature_view_list_raises(self) -> None:
        # An empty list is ambiguous (a method opted in to FR but supplied no
        # source) and would produce an FR config block the server treats as
        # malformed; reject up front with a clearer message.
        with tempfile.TemporaryDirectory() as tmpdir:
            mds = model_deployment_spec.ModelDeploymentSpec(workspace_path=pathlib.Path(tmpdir))
            mds.add_model_spec(
                database_name=sql_identifier.SqlIdentifier("db"),
                schema_name=sql_identifier.SqlIdentifier("schema"),
                model_name=sql_identifier.SqlIdentifier("model"),
                version_name=sql_identifier.SqlIdentifier("version"),
            )
            mds.add_image_build_spec(
                image_build_compute_pool_name=sql_identifier.SqlIdentifier("image_build_compute_pool"),
                fully_qualified_image_repo_name="DB.SCHEMA.IMAGE_REPO",
            )
            with self.assertRaisesRegex(ValueError, "is empty"):
                mds.add_service_spec(
                    service_name=sql_identifier.SqlIdentifier("service"),
                    inference_compute_pool_name=sql_identifier.SqlIdentifier("service_compute_pool"),
                    ingress_enabled=True,
                    min_instances=1,
                    max_instances=5,
                    feature_sources_per_function={"predict": []},
                )


class _FakeFeatureView:
    """Duck-typed stand-in for snowflake.ml.feature_store.FeatureView used by
    feature-retrieval spec tests. We avoid importing FeatureView into the
    model-client test module to keep the dependency direction one-way
    (feature_store may depend on model client, never the reverse).
    """

    def __init__(
        self,
        *,
        database: str,
        schema: str,
        name: Union[str, sql_identifier.SqlIdentifier],
        version: object,
    ) -> None:
        self.database = sql_identifier.SqlIdentifier(database) if database else None
        self.schema = sql_identifier.SqlIdentifier(schema) if schema else None
        self.name = name if isinstance(name, sql_identifier.SqlIdentifier) else sql_identifier.SqlIdentifier(name)
        self.version = version


if __name__ == "__main__":
    absltest.main()
