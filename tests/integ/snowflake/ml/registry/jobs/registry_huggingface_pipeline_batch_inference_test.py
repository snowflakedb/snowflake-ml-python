import json
import os
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import openai_signatures
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


class TestRegistryHuggingFacePipelineBatchInferenceInteg(
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
        pip_requirements=[None, ["transformers", "torch==2.6.0"]],
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

        NUM_CHOICES = 3
        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Complete the sentence.",
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",  # noqa: E501
                                },
                            ],
                        },
                    ],
                    "temperature": 0.9,
                    "max_completion_tokens": 250,
                    "stop": None,
                    "n": NUM_CHOICES,
                    "stream": False,
                    "top_p": 0.9,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.1,
                }
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(
                    [
                        "messages",
                        "temperature",
                        "max_completion_tokens",
                        "stop",
                        "n",
                        "stream",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                        "id",
                        "object",
                        "created",
                        "model",
                        "choices",
                        "usage",
                    ],
                    dtype="object",
                ),
                check_order=False,
            )

            for row in res["choices"]:
                row = json.loads(row)
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), NUM_CHOICES)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])

        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        input_df = self.session.create_dataframe(x_df)

        self._test_registry_batch_inference(
            model=model,
            options={},
            pip_requirements=pip_requirements,
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(job_name=job_name),
            prediction_assert_fn=check_res,
        )


if __name__ == "__main__":
    absltest.main()
