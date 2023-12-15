import os
import tempfile

import pandas as pd
import pytest
from absl.testing import absltest

from snowflake.ml.model import (
    _api as model_api,
    deploy_platforms,
    type_hints as model_types,
)
from snowflake.ml.model.models import llm
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    spcs_integ_test_base,
    test_env_utils,
)


@pytest.mark.conda_incompatible
class TestSPCSLLMModelInteg(spcs_integ_test_base.SpcsIntegTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["HF_HOME"] = self.cache_dir.name

    def tearDown(self) -> None:
        super().tearDown()
        if self._original_hf_home:
            os.environ["HF_HOME"] = self._original_hf_home
        else:
            del os.environ["HF_HOME"]
        self.cache_dir.cleanup()

    def test_text_generation_pipeline(
        self,
    ) -> None:
        model = llm.LLM(
            model_id_or_path="facebook/opt-350m",
        )

        x_df = pd.DataFrame(
            [["Hello world"]],
        )

        stage_path = f"@{self._test_stage}/{self._run_id}"
        deployment_stage_path = f"@{self._test_stage}/{self._run_id}"
        model_api.save_model(  # type: ignore[call-overload]
            name="model",
            session=self._session,
            stage_path=stage_path,
            model=model,
            options={"embed_local_ml_library": True},
            conda_dependencies=[
                test_env_utils.get_latest_package_version_spec_in_conda("snowflake-snowpark-python"),
            ],
        )
        svc_func_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id,
            f"func_{self._run_id}",
        )

        deployment_options: model_types.SnowparkContainerServiceDeployOptions = {
            "compute_pool": self._TEST_GPU_COMPUTE_POOL,
            "num_gpus": 1,
            "model_in_image": True,
        }

        deploy_info = model_api.deploy(
            name=svc_func_name,
            session=self._session,
            stage_path=stage_path,
            deployment_stage_path=deployment_stage_path,
            model_id=svc_func_name,
            platform=deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
            options={
                **deployment_options,  # type: ignore[arg-type]
            },  # type: ignore[call-overload]
        )
        assert deploy_info is not None
        res = model_api.predict(session=self._session, deployment=deploy_info, X=x_df)
        self.assertIn("generated_text", res)
        self.assertEqual(len(res["generated_text"]), 1)
        self.assertNotEmpty(res["generated_text"][0])


if __name__ == "__main__":
    absltest.main()
