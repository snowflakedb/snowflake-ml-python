import pathlib
import tempfile

import yaml
from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils import sql_identifier
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
                max_instances=1,
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
                            "max_instances": 1,
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
            max_instances=1,
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
                    "max_instances": 1,
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
                max_instances=1,
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
                            "max_instances": 1,
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
                max_instances=10,
                cpu="1",
                memory="1GiB",
                gpu="1",
                num_workers=10,
                max_batch_rows=1024,
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
                max_instances=1,
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
                            "max_instances": 1,
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
                        },
                    },
                )

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
                max_instances=1,
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
                            "max_instances": 1,
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
                max_instances=1,
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
                            "max_instances": 1,
                        },
                    },
                )

    def test_inference_engine_options_minimal(self) -> None:
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

    def test_inference_engine_options_minimal_with_blocklist_args(self) -> None:
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
                            "max_instances": 1,
                            "inference_engine_spec": {
                                "inference_engine_name": "vllm",
                                "inference_engine_args": ["--some_vllm_arg=0.8"],
                            },
                        },
                    },
                )
        mds.clear()

    def test_skip_image_build_with_inference_engine(self) -> None:
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
                            "max_instances": 1,
                            "inference_engine_spec": {
                                "inference_engine_name": "vllm",
                                "inference_engine_args": ["--some_vllm_arg=0.8"],
                            },
                        },
                    },
                )
        mds.clear()


if __name__ == "__main__":
    absltest.main()
