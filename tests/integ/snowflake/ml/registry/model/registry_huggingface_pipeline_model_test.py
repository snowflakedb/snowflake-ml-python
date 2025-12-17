import json
import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml.model import model_signature, openai_signatures
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class TestRegistryHuggingFacePipelineModelInteg(registry_model_test_base.RegistryModelTestBase):
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

    def test_fill_mask_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="fill-mask",
            model="google-bert/bert-base-uncased",
            top_k=1,
        )

        x_df = pd.DataFrame(
            [
                ["LynYuu is the [MASK] of the Grand Duchy of Yu."],
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertIn("score", row[0])
                self.assertIn("token", row[0])
                self.assertIn("token_str", row[0])
                self.assertIn("sequence", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_ner_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(task="ner", model="Isotonic/distilbert_finetuned_ai4privacy_v2")

        x_df = pd.DataFrame(
            [
                ["My name is Izumi and I live in Tokyo, Japan."],
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertIn("entity", row[0])
                self.assertIn("score", row[0])
                self.assertIn("index", row[0])
                self.assertIn("word", row[0])
                self.assertIn("start", row[0])
                self.assertIn("end", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_question_answering_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="question-answering",
            model="distilbert/distilbert-base-cased-distilled-squad",
            top_k=1,
        )

        x_df = pd.DataFrame(
            [
                {
                    "question": "What did Doris want to do?",
                    "context": (
                        "Doris is a cheerful mermaid from the ocean depths. She transformed into a bipedal creature "
                        'and came to see everyone because she wanted to "learn more about the world of athletics."'
                        " She dislikes cuisines with seafood."
                    ),
                }
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["score", "start", "end", "answer"]))

            self.assertEqual(res["score"].dtype.type, np.float64)
            self.assertEqual(res["start"].dtype.type, np.int64)
            self.assertEqual(res["end"].dtype.type, np.int64)
            self.assertEqual(res["answer"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_question_answering_pipeline_multiple_output(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="question-answering",
            model="distilbert/distilbert-base-cased-distilled-squad",
            top_k=3,
        )

        x_df = pd.DataFrame(
            [
                {
                    "question": "What did Doris want to do?",
                    "context": (
                        "Doris is a cheerful mermaid from the ocean depths. She transformed into a bipedal creature "
                        'and came to see everyone because she wanted to "learn more about the world of athletics."'
                        " She dislikes cuisines with seafood."
                    ),
                }
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answers"]))

            for row in res["answers"]:
                self.assertIsInstance(row, list)
                self.assertIn("score", row[0])
                self.assertIn("start", row[0])
                self.assertIn("end", row[0])
                self.assertIn("answer", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_summarization_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(task="summarization", model="Falconsai/text_summarization")

        x_df = pd.DataFrame(
            [
                [
                    (
                        "Neuro-sama is a chatbot styled after a female VTuber that hosts live streams on the Twitch "
                        'channel "vedal987". Her speech and personality are generated by an artificial intelligence'
                        " (AI) system  which utilizes a large language model, allowing her to communicate with "
                        "viewers in a live chat.  She was created by a computer programmer and AI-developer named "
                        "Jack Vedal, who decided to build  upon the concept of an AI VTuber by combining interactions "
                        "between AI game play and a computer-generated avatar. She debuted on Twitch on December 19, "
                        "2022 after four years of development."
                    )
                ]
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["summary_text"]))

            self.assertEqual(res["summary_text"].dtype.type, str)

        self._test_registry_model(
            model=model,
            additional_dependencies=[
                str(env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("sentencepiece")))
            ],
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_table_question_answering_pipeline(self) -> None:
        import transformers

        model_id = "microsoft/tapex-base-finetuned-wtq"
        # TODO: Use this model after upgrading pytorch>=2.6.0
        # model_id = "google/tapas-large-finetuned-wtq"
        model = transformers.pipeline(
            task="table-question-answering",
            model=model_id,
        )

        x_df = pd.DataFrame(
            [
                {
                    "query": "Which channel has the most subscribers?",
                    "table": json.dumps(
                        {
                            "Channel": [
                                "A.I.Channel",
                                "Kaguya Luna",
                                "Mirai Akari",
                                "Siro",
                            ],
                            "Subscribers": [
                                "3,020,000",
                                "872,000",
                                "694,000",
                                "660,000",
                            ],
                            "Videos": ["1,200", "113", "639", "1,300"],
                            "Created At": [
                                "Jun 30 2016",
                                "Dec 4 2017",
                                "Feb 28 2014",
                                "Jun 23 2017",
                            ],
                        }
                    ),
                }
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answer", "coordinates", "cells", "aggregator"]))

            self.assertEqual(res["answer"].dtype.type, str)
            if model_id == "google/tapas-large-finetuned-wtq":
                # only google/tapas-large-finetuned-wtq has coordinates, cells, and aggregator
                self.assertEqual(res["coordinates"].dtype.type, np.object_)
                self.assertIsInstance(res["coordinates"][0], list)
                self.assertEqual(res["cells"].dtype.type, np.object_)
                self.assertIsInstance(res["cells"][0], list)
                self.assertEqual(res["aggregator"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_text_classification_pair_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(task="text-classification", model="cross-encoder/ms-marco-MiniLM-L-12-v2")

        x_df = pd.DataFrame(
            [{"text": "I like you.", "text_pair": "I love you, too."}],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["label", "score"]))

            self.assertEqual(res["label"].dtype.type, str)
            self.assertEqual(res["score"].dtype.type, np.float32)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            signatures={
                "__call__": model_signature.ModelSignature(
                    inputs=[
                        model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="text"),
                        model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="text_pair"),
                    ],
                    outputs=[
                        model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="label"),
                        model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="score"),
                    ],
                ),
            },
        )

    def test_text_classification_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-classification",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            top_k=1,
        )

        x_df = pd.DataFrame(
            [
                {
                    "text": "I am wondering if I should have udon or rice for lunch",
                }
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))

            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertIn("label", row[0])
                self.assertIn("score", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_text_generation_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B",
        )

        x_df = pd.DataFrame(
            [['A descendant of the Lost City of Atlantis, who swam to Earth while saying, "']],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertIn("generated_text", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            signatures={
                "__call__": model_signature.ModelSignature(
                    inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="inputs")],
                    outputs=[
                        model_signature.FeatureGroupSpec(
                            name="outputs",
                            specs=[
                                model_signature.FeatureSpec(
                                    dtype=model_signature.DataType.STRING, name="generated_text"
                                )
                            ],
                            shape=(-1,),
                        )
                    ],
                )
            },
        )

    def test_text_generation_chat_template_pipeline(
        self,
    ) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-gpt2-with-chatml-template",
            max_length=200,
        )

        x_df = pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {"role": "system", "content": "Complete the sentence."},
                        {
                            "role": "user",
                            "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",
                        },
                    ],
                    "max_completion_tokens": 250,
                    "temperature": 0.9,
                    "stop": None,
                    "n": 3,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                }
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(
                res.columns,
                pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
                check_order=False,
            )

            for row in res["choices"]:
                self.assertIsInstance(row, list)
                self.assertIn("message", row[0])
                self.assertIn("content", row[0]["message"])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
        )

    def test_hf_pipeline_text2text_generation(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="text2text-generation",
            model="google-t5/t5-small",
        )

        x_df = pd.DataFrame(
            [['A descendant of the Lost City of Atlantis, who swam to Earth while saying, "']],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["generated_text"]))
            self.assertEqual(res["generated_text"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )

    def test_translation_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(task="translation_en_to_ja", model="Mitsua/elan-mt-tiny-en-ja")

        x_df = pd.DataFrame(
            [
                [
                    (
                        "Snowflake's Data Cloud is powered by an advanced data platform provided as a self-managed "
                        "service. Snowflake enables data storage, processing, and analytic solutions that are faster, "
                        "easier to use, and far more flexible than traditional offerings. The Snowflake data platform "
                        "is not built on any existing database technology or “big data” software platforms such as "
                        "Hadoop. Instead, Snowflake combines a completely new SQL query engine with an innovative "
                        "architecture natively designed for the cloud. To the user, Snowflake provides all of the "
                        "functionality of an enterprise analytic database, along with many additional special features "
                        "and unique capabilities."
                    )
                ]
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["translation_text"]))
            self.assertEqual(res["translation_text"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            additional_dependencies=[
                "sentencepiece",
            ],
        )

    def test_zero_shot_classification_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="zero-shot-classification",
            model="sileod/deberta-v3-base-tasksource-nli",
        )

        x_df = pd.DataFrame(
            [
                {
                    "sequences": "I have a problem with Snowflake that needs to be resolved asap!!",
                    "candidate_labels": ["urgent", "not urgent"],
                },
                {
                    "sequences": "I have a problem with Snowflake that needs to be resolved asap!!",
                    "candidate_labels": ["English", "Japanese"],
                },
            ],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["sequence", "labels", "scores"]))
            self.assertEqual(res["sequence"].dtype.type, str)
            self.assertEqual(
                res["sequence"][0],
                "I have a problem with Snowflake that needs to be resolved asap!!",
            )
            self.assertEqual(
                res["sequence"][1],
                "I have a problem with Snowflake that needs to be resolved asap!!",
            )
            self.assertEqual(res["labels"].dtype.type, np.object_)
            self.assertListEqual(sorted(res["labels"][0]), sorted(["urgent", "not urgent"]))
            self.assertListEqual(sorted(res["labels"][1]), sorted(["English", "Japanese"]))
            self.assertEqual(res["scores"].dtype.type, np.object_)
            self.assertIsInstance(res["scores"][0], list)
            self.assertIsInstance(res["scores"][1], list)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
