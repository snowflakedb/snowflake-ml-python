import json
import os
import tempfile
from typing import List, Optional

import pandas as pd
from absl.testing import absltest, parameterized

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class TestRegistryHuggingFacePipelineDeploymentModelInteg(
    registry_model_deployment_test_base.RegistryModelDeploymentTestBase
):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir.name

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self._original_cache_dir
        self.cache_dir.cleanup()

    @parameterized.product(  # type: ignore[misc]
        gpu_requests=[None, "1"],
        pip_requirements=[None, ["transformers"]],
    )
    def test_text_generation(
        self,
        gpu_requests: str,
        pip_requirements: Optional[List[str]],
    ) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="openai-community/gpt2",
        )

        x_df = pd.DataFrame(
            [['A descendant of the Lost City of Atlantis, who swam to Earth while saying, "']],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, str)
                resp = json.loads(row)
                self.assertIsInstance(resp, list)
                self.assertIn("generated_text", resp[0])

        self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (
                    x_df,
                    check_res,
                ),
            },
            options={"cuda_version": "11.8"} if gpu_requests else {},
            additional_dependencies=["pytorch==2.1.0"],
            gpu_requests=gpu_requests,
            pip_requirements=pip_requirements,
        )


if __name__ == "__main__":
    absltest.main()
