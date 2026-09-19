import inspect
import json
import os
import sys
import tempfile
import textwrap
import unittest
from typing import Any, Callable

import numpy as np
from absl.testing import absltest, parameterized

from snowflake.ml import jobs
from snowflake.ml.jobs._utils import runtime_env_utils
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as sp_exceptions, session
from tests.integ.snowflake.ml.jobs import (
    reflection_utils,
    test_constants,
    test_file_helper,
)
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils

_TRAIN_MODEL_FUNC = "train_model"

_PREDICT_FUNC = "predict_result"


@absltest.skipIf(
    (region := test_env_utils.get_current_snowflake_region()) is None
    or region["cloud"] not in test_constants._SUPPORTED_CLOUDS,
    "Test only for SPCS supported clouds",
)
class JobTestBase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.session = session._get_active_session()
        except sp_exceptions.SnowparkSessionException:
            cls.session = test_env_utils.get_available_session()
        cls.session.use_database(test_constants._TEST_DB)
        cls.dbm = db_manager.DBManager(cls.session)
        cls.dbm.cleanup_schemas(prefix=test_constants._TEST_SCHEMA, expire_days=1)
        cls.db = cls.session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            cls.compute_pool = cls.dbm.create_compute_pool(
                test_constants._TEST_COMPUTE_POOL, sql_client.CreationMode(if_not_exists=True), max_nodes=20
            )
        except sp_exceptions.SnowparkSQLException:
            if not cls.dbm.show_compute_pools(test_constants._TEST_COMPUTE_POOL).count() > 0:
                raise cls.failureException(
                    f"Compute pool {test_constants._TEST_COMPUTE_POOL} not available and could not be created"
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbm.drop_schema(cls.schema, if_exists=True)
        cls.session.close()
        super().tearDownClass()

    def _submit_func_as_file(self, func: Callable[[], None], **kwargs: Any) -> jobs.MLJob[None]:
        # Insert default kwargs
        default_kwargs = dict(
            compute_pool=self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
        )
        kwargs = {**default_kwargs, **kwargs}

        func_source = inspect.getsource(func)
        payload_str = textwrap.dedent(func_source) + "\n\n__return__ = " + func.__name__ + "()\n"
        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            temp_file.write(payload_str.encode("utf-8"))
            temp_file.flush()
            job: jobs.MLJob[None] = jobs.submit_file(
                temp_file.name,
                **kwargs,
            )
            return job

    @staticmethod
    def _local_python_version() -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def _resolve_runtime_image(
        self, *, runtime_environment: str | None = None, python_version: str | None = None
    ) -> str:
        """Resolve a runtime selector through the backend the same way job registration does."""
        # Mirrors the selector that MLJobDefinition._register builds when the runtime versions feature
        # flag is enabled.
        selector: str | None = runtime_environment
        if python_version is not None:
            selector_dict = {"pythonVersion": python_version}
            if runtime_environment is not None:
                selector_dict["runtimeEnvironment"] = runtime_environment
            selector = json.dumps(selector_dict)
        return runtime_env_utils.get_runtime_image(self.session, self.compute_pool, selector)

    def _resolve_for_local_python(self, runtime_environment: str | None = None) -> str:
        """Resolve for the interpreter running the tests, skipping if it has no registered image."""
        python_version = self._local_python_version()
        try:
            return self._resolve_runtime_image(runtime_environment=runtime_environment, python_version=python_version)
        except sp_exceptions.SnowparkSQLException as e:
            # Deliberately narrow: only a missing image becomes a skip, so an unrelated failure of the
            # runtime lookup fails the test loudly instead of turning into a silent pass.
            if not any(signal in str(e).lower() for signal in ("runtime image is registered", "matched image")):
                raise
            raise unittest.SkipTest(f"No runtime image registered for Python {python_version}: {e}") from e


class ModelingJobTestBase(JobTestBase):
    def get_inference(self, model: Any, module_path: str) -> Any:
        return reflection_utils.run_reflected_func(module_path, _PREDICT_FUNC, model)

    def get_model(self, model_name: str, module_path: str) -> Any:
        return reflection_utils.run_reflected_func(module_path, _TRAIN_MODEL_FUNC, model_name)

    def train_models(
        self,
        model_name: str,
        model_script: str,
        pip_requirements: list[str] | None = None,
        external_access_integrations: list[str] | None = None,
    ) -> None:
        payload = test_file_helper.TestAsset(model_script)
        job = jobs.submit_file(
            payload.path,
            self.compute_pool,
            stage_name="payload_stage",
            session=self.session,
            pip_requirements=pip_requirements,
            args=[model_name],
            external_access_integrations=external_access_integrations,
        )
        self.assertIsNotNone(job)
        module_path = f"test_files.{os.path.splitext(model_script)[0].replace('/', '.')}"
        model_local = self.get_model(model_name, module_path)
        inference_local = self.get_inference(model_local, module_path)
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        model_remote = job.result()
        inference_remote = self.get_inference(model_remote, module_path)
        np.testing.assert_allclose(inference_local, inference_remote, atol=1e-6)
