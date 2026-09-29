"""Deploy a pre-logged Model Garden LLM from SNOWFLAKE.MODELS.

The model must already exist in SNOWFLAKE.MODELS (catalog map set + refresh).
Set SNOWML_MODEL_GARDEN_HF_MODEL to the Hugging Face id used in the catalog map.
"""

import logging
import os
import re
import unittest

from absl.testing import absltest

from snowflake.ml.model._client.model import batch_inference_job_specs
from tests.integ.snowflake.ml.registry.services import (
    registry_model_garden_deployment_test as garden,
)

logger = logging.getLogger(__name__)

_HF_MODEL_ENV = "SNOWML_MODEL_GARDEN_HF_MODEL"
_CATALOG_KEY_ENV = "SNOWML_MODEL_GARDEN_CATALOG_KEY"
_HF_MODEL_NAME = os.environ.get(_HF_MODEL_ENV, "").strip()


def _quoted_model_name(model_name: str) -> str:
    stripped = model_name.strip().strip('"')
    escaped = stripped.replace('"', '""')
    return f'"{escaped}"'


@unittest.skipUnless(_HF_MODEL_NAME, f"{_HF_MODEL_ENV} must be set to run this test.")
class TestRegistryModelGardenPreloggedDeploymentInteg(garden.TestRegistryModelGardenDeploymentInteg):
    def setUp(self) -> None:
        super().setUp()
        self._hf_model_name = _HF_MODEL_NAME
        self._catalog_key = os.environ.get(_CATALOG_KEY_ENV, "").strip()
        basename = re.sub(r"[^A-Za-z0-9]", "_", self._hf_model_name.rsplit("/", 1)[-1])[:16]
        self._service_short_name = f"MG_{basename}_{self._run_id}".upper()

    def _resolve_model_garden_model(self) -> str | None:
        names = self._snowflake_models_names()
        catalog_key = self._catalog_key.strip('"')
        if catalog_key:
            for name in names:
                if name.upper() == catalog_key.upper():
                    return _quoted_model_name(name)
        expected = self._hf_model_name.rsplit("/", 1)[-1].upper()
        for name in names:
            if expected in name.upper() or name.upper() == expected:
                return _quoted_model_name(name)
        return None

    def _require_model_garden_model(self) -> str:
        model_name = self._resolve_model_garden_model()
        if model_name is None:
            self.skipTest(
                f"No SNOWFLAKE.MODELS object matching {_HF_MODEL_ENV}={self._hf_model_name!r} "
                f"catalog_key={self._catalog_key!r}; skipping."
            )
        return model_name

    def test_model_garden_service(self) -> None:
        model_name = self._require_model_garden_model()
        logger.info("Using pre-logged Model Garden model SNOWFLAKE.MODELS.%s", model_name)
        self._enable_ai_complete_session_params()

        service_short_name = self._service_short_name
        service_name = f"{self._test_db}.{self._test_schema}.{service_short_name}"
        mv = self._model_garden_model_version(model_name)
        mv.create_service(
            service_name=service_name,
            service_compute_pool=self._TEST_GPU_COMPUTE_POOL,
        )
        self._wait_for_named_service_status(service_short_name, expected_status="RUNNING")

        with self.subTest(invocation="service_function_sql"):
            self._assert_chat_completion(self._invoke_service_sql(service_name))
            logger.info("subTest passed: service_function_sql")

        with self.subTest(invocation="model_version_run"):
            self._assert_chat_completion(self._invoke_mv_run(service_name, model_name))
            logger.info("subTest passed: model_version_run")

        with self.subTest(invocation="ai_complete"):
            self._assert_ai_complete_text(self._invoke_ai_complete(service_name))
            logger.info("subTest passed: ai_complete")

        with self.subTest(lifecycle="suspend_resume"):
            self.session.sql(f"ALTER SERVICE {service_name} SUSPEND").collect()
            self._wait_for_named_service_status(service_short_name, expected_status="SUSPENDED")

            self.session.sql(f"ALTER SERVICE {service_name} RESUME").collect()
            self._wait_for_named_service_status(service_short_name, expected_status="RUNNING")

            self._assert_chat_completion(self._invoke_service_sql(service_name))
            logger.info("subTest passed: suspend_resume")

    def test_model_garden_batch_inference_job(self) -> None:
        model_name = self._require_model_garden_model()
        logger.info("Using pre-logged Model Garden model SNOWFLAKE.MODELS.%s", model_name)

        mv = self._model_garden_model_version(model_name)
        job_short_name = f"MG_BATCH_{self._run_id}".upper()
        job_name = f"{self._test_db}.{self._test_schema}.{job_short_name}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{job_short_name}/output/"
        input_df = self.session.create_dataframe(self._openai_call_input())

        batch_job = mv.run_batch(
            input_df,
            compute_pool=self._TEST_GPU_COMPUTE_POOL,
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            job_name=job_name,
        )

        try:
            batch_job.wait(timeout=1800)
        except TimeoutError:
            logger.warning("Batch job timed out after 30 minutes. Status: %s", batch_job.status)

        if batch_job.status != "DONE":
            logs = batch_job.get_logs(limit=100)
            self.fail(f"Job {batch_job.id} status is {batch_job.status}, expected DONE.\n\n{logs}")

        resolved_job_name = batch_job.id.split(".")[-1].strip('"')
        job_output_stage_location = output_stage_location.rstrip("/") + "/" + resolved_job_name + "/"
        success_file_path = job_output_stage_location.rstrip("/") + "/_SUCCESS"
        list_results = self.session.sql(f"LIST {success_file_path}").collect()
        self.assertGreater(len(list_results), 0, f"Batch job did not produce success file at: {success_file_path}")

        output_df = self.session.read.option("on_error", "CONTINUE").parquet(job_output_stage_location)
        self.assertEqual(output_df.count(), input_df.count())
        self._assert_chat_completion(output_df.to_pandas())
        logger.info("test_model_garden_batch_inference_job passed")


if __name__ == "__main__":
    absltest.main()
