import json
import os
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryHuggingFacePipelineBatchInferenceGpuModelInteg(
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase
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
        pip_requirements=[None, ["transformers"]],
    )
    def test_text_generation(
        self,
        pip_requirements: Optional[list[str]],
    ) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-gpt2-with-chatml-template",
            max_length=200,
        )

        x = [
            [
                {"role": "system", "content": "Complete the sentence."},
                {
                    "role": "user",
                    "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",
                },
            ]
        ]

        x_df = pd.DataFrame([x], columns=["inputs"])

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["inputs", "outputs"]))

            for row in res["outputs"]:
                row = json.loads(row)
                self.assertIsInstance(row, list)
                self.assertIn("generated_text", row[0])

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        input_df = self.session.create_dataframe(x_df)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name, gpu_requests="1"),
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            pip_requirements=pip_requirements,
            prediction_assert_fn=check_res,
        )


if __name__ == "__main__":
    absltest.main()
