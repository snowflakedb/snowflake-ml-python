import json
import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env_utils
from snowflake.ml.model import model_signature, openai_signatures
from snowflake.ml.model.models import huggingface
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

        def check_res_with_top_k(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))
            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 3)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"top_k": 3},
                    check_res_with_top_k,
                ),
            },
        )

    def test_ner_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(task="ner", model="Isotonic/distilbert_finetuned_ai4privacy_v2")

        x_df = pd.DataFrame(
            [
                ["My name is Izumi and I live in Tokyo, Japan."],
                ["Gibberish jabberish jabberish"],
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                if len(row) > 0:
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
        model = huggingface.TransformersPipeline(
            task="question-answering",
            model="distilbert/distilbert-base-cased-distilled-squad",
            compute_pool_for_log=None,
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

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["score", "start", "end", "answer"]))
            self.assertEqual(res["score"].dtype.type, np.float64)
            for answer in res["answer"]:
                self.assertLessEqual(len(answer), 5)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        "doc_stride": 64,
                        "max_answer_len": 5,
                        "max_seq_len": 256,
                        "max_question_len": 32,
                        "handle_impossible_answer": True,
                        "align_to_words": False,
                    },
                    check_res_with_params,
                ),
            },
        )

    def test_question_answering_pipeline_multiple_output(self) -> None:
        model = huggingface.TransformersPipeline(
            task="question-answering",
            model="distilbert/distilbert-base-cased-distilled-squad",
            compute_pool_for_log=None,
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

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answers"]))
            for row in res["answers"]:
                self.assertIsInstance(row, list)
                self.assertIn("answer", row[0])
                for entry in row:
                    self.assertLessEqual(len(entry["answer"]), 5)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        "max_answer_len": 5,
                        "doc_stride": 64,
                        "max_seq_len": 256,
                        "max_question_len": 32,
                        "handle_impossible_answer": True,
                        "align_to_words": False,
                    },
                    check_res_with_params,
                ),
            },
        )

    def test_summarization_pipeline(self) -> None:
        model = huggingface.TransformersPipeline(
            task="summarization",
            model="Falconsai/text_summarization",
            compute_pool_for_log=None,
        )

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
        model_id = "microsoft/tapex-base-finetuned-wtq"
        # TODO: Use this model after upgrading pytorch>=2.6.0
        # model_id = "google/tapas-large-finetuned-wtq"
        model = huggingface.TransformersPipeline(
            task="table-question-answering",
            model=model_id,
            compute_pool_for_log=None,
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

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answer", "coordinates", "cells", "aggregator"]))
            self.assertEqual(res["answer"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"sequential": True, "padding": "max_length", "truncation": "only_first"},
                    check_res_with_params,
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

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))
            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 2)
                self.assertIn("score", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"top_k": 2, "function_to_apply": "sigmoid"},
                    check_res_with_params,
                ),
            },
        )

    def test_text_generation_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )

        x_df = pd.DataFrame(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": 'A descendant of the Lost City of Atlantis, who swam to Earth while saying, "',  # noqa: E501
                        },
                    ],
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
                        {
                            "role": "system",
                            "content": "Complete the sentence.",
                        },
                        {
                            "role": "user",
                            "content": "A descendant of the Lost City of Atlantis, who swam to Earth while saying, ",  # noqa: E501
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
            # testing non-default signature for text-generation chat template pipeline
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        )

    def test_hf_pipeline_text2text_generation(self) -> None:
        model = huggingface.TransformersPipeline(
            task="text2text-generation",
            model="google-t5/t5-small",
            compute_pool_for_log=None,
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
        model = huggingface.TransformersPipeline(
            task="translation_en_to_ja",
            model="Mitsua/elan-mt-tiny-en-ja",
            compute_pool_for_log=None,
        )

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

        def check_res_with_multi_label(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["sequence", "labels", "scores"]))
            self.assertEqual(res["scores"].dtype.type, np.object_)
            self.assertIsInstance(res["scores"][0], list)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"multi_label": True, "hypothesis_template": "This is about {}."},
                    check_res_with_multi_label,
                ),
            },
        )

    def test_image_classification_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
        )

        # Read test image as bytes
        with open("tests/integ/snowflake/ml/test_data/cat.jpeg", "rb") as f:
            image_bytes = f.read()

        x_df = pd.DataFrame({"images": [image_bytes] * 3})

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))

            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)
                self.assertIn("label", row[0])
                self.assertIn("score", row[0])

        def check_res_with_top_k(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))
            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 3)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"top_k": 3, "function_to_apply": "sigmoid"},
                    check_res_with_top_k,
                ),
            },
            additional_dependencies=["pillow==12.0"],
        )

    def test_automatic_speech_recognition_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
        )

        # Read test audio as bytes
        with open("tests/integ/snowflake/ml/test_data/batman_audio.mp3", "rb") as f:
            audio_bytes = f.read()

        x_df = pd.DataFrame({"audio": [audio_bytes] * 3})

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, dict)
                self.assertIn("text", row)

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))
            for row in res["outputs"]:
                self.assertIsInstance(row, dict)
                self.assertIn("text", row)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        # Length control (INT64)
                        "max_new_tokens": 20,
                        "max_length": 100,
                        "min_length": 1,
                        "min_new_tokens": 1,
                        # Stopping criteria (STRING, DOUBLE)
                        "early_stopping": "never",
                        "max_time": 60.0,
                        # Sampling strategy (BOOL, INT64, DOUBLE)
                        "do_sample": False,
                        "num_beams": 1,
                        "num_beam_groups": 1,
                        "penalty_alpha": 0.0,
                        # Sampling parameters (INT64, DOUBLE)
                        "temperature": 1.0,
                        "top_k": 50,
                        "top_p": 1.0,
                        "min_p": 0.0,
                        "typical_p": 1.0,
                        "epsilon_cutoff": 0.0,
                        "eta_cutoff": 0.0,
                        # Diversity / repetition (DOUBLE, INT64)
                        "diversity_penalty": 0.0,
                        "repetition_penalty": 1.0,
                        "encoder_repetition_penalty": 1.0,
                        "length_penalty": 1.0,
                        "no_repeat_ngram_size": 0,
                        # Token manipulation (BOOL, INT64)
                        "renormalize_logits": False,
                        "remove_invalid_values": False,
                        # Output control (INT64, BOOL)
                        "num_return_sequences": 1,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "output_scores": False,
                        "output_logits": False,
                        # Encoder-decoder (INT64)
                        "encoder_no_repeat_ngram_size": 0,
                        # Miscellaneous (DOUBLE, BOOL)
                        "guidance_scale": 1.0,
                        "low_memory": False,
                        "use_cache": True,
                        # Shaped params: list (INT64, shape=(-1,))
                        "suppress_tokens": [1, 2, 3],
                        "begin_suppress_tokens": [1, 2],
                        # Task-specific (STRING)
                        "return_timestamps": "word",
                    },
                    check_res_with_params,
                ),
            },
            additional_dependencies=["ffmpeg"],
        )

    # TODO: revisit this test later
    def test_video_classification_pipeline(self) -> None:
        self.skipTest("decord package not available in conda channel")
        import transformers

        model = transformers.pipeline(
            task="video-classification",
            model="nateraw/videomae-base-finetuned-ucf101-subset",
        )

        # Read test video as bytes
        with open("tests/integ/snowflake/ml/test_data/cutting_in_kitchen.avi", "rb") as f:
            video_bytes = f.read()

        x_df = pd.DataFrame(
            [
                {"video": video_bytes},
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))

            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)
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
            additional_dependencies=[
                "decord",
                "ffmpeg",
                "av",
                "pillow",
                "accelerate=1.12.0",
                "pytorch=2.9.1",
                "torchvision=0.24.1",
            ],
        )

    def test_object_detection_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="object-detection",
            model="hustvl/yolos-tiny",
        )

        with open("tests/integ/snowflake/ml/test_data/cat.jpeg", "rb") as f:
            image_bytes = f.read()

        x_df = pd.DataFrame({"images": [image_bytes]})

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["detections"]))
            for row in res["detections"]:
                self.assertIsInstance(row, list)
                if len(row) > 0:
                    self.assertIn("label", row[0])
                    self.assertIn("score", row[0])
                    self.assertIn("box", row[0])

        def check_res_with_threshold(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["detections"]))
            for row in res["detections"]:
                self.assertIsInstance(row, list)
                for det in row:
                    self.assertGreaterEqual(det["score"], 0.9)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"threshold": 0.9, "timeout": 30.0},
                    check_res_with_threshold,
                ),
            },
            additional_dependencies=["pillow==12.0"],
        )

    def test_zero_shot_image_classification_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="zero-shot-image-classification",
            model="wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M",
        )

        with open("tests/integ/snowflake/ml/test_data/cat.jpeg", "rb") as f:
            image_bytes = f.read()

        x_df = pd.DataFrame(
            [
                {
                    "images": image_bytes,
                    "candidate_labels": ["cat", "dog", "bird"],
                },
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))
            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)
                self.assertIn("label", row[0])
                self.assertIn("score", row[0])

        def check_res_with_template(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["labels"]))
            for row in res["labels"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"hypothesis_template": "A picture of a {}.", "timeout": 30.0},
                    check_res_with_template,
                ),
            },
            additional_dependencies=["pillow==12.0"],
        )

    def test_zero_shot_object_detection_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="zero-shot-object-detection",
            model="google/owlvit-base-patch32",
        )

        with open("tests/integ/snowflake/ml/test_data/cat.jpeg", "rb") as f:
            image_bytes = f.read()

        x_df = pd.DataFrame(
            [
                {
                    "images": image_bytes,
                    "candidate_labels": ["cat", "remote control"],
                },
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["detections"]))
            for row in res["detections"]:
                self.assertIsInstance(row, list)
                if len(row) > 0:
                    self.assertIn("label", row[0])
                    self.assertIn("score", row[0])
                    self.assertIn("box", row[0])

        def check_res_with_threshold(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["detections"]))
            for row in res["detections"]:
                self.assertIsInstance(row, list)
                for det in row:
                    self.assertGreaterEqual(det["score"], 0.5)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"threshold": 0.5, "top_k": 5, "timeout": 30.0},
                    check_res_with_threshold,
                ),
            },
            additional_dependencies=["pillow==12.0"],
        )

    def test_image_feature_extraction_pipeline(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="image-feature-extraction",
            model="google/vit-base-patch16-224",
        )

        with open("tests/integ/snowflake/ml/test_data/cat.jpeg", "rb") as f:
            image_bytes = f.read()

        x_df = pd.DataFrame({"images": [image_bytes]})

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["feature_extraction"]))
            for row in res["feature_extraction"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["feature_extraction"]))
            for row in res["feature_extraction"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"timeout": 30.0},
                    check_res_with_params,
                ),
            },
            additional_dependencies=["pillow==12.0"],
        )

    def test_visual_question_answering_pipeline(self) -> None:
        model = huggingface.TransformersPipeline(
            task="visual-question-answering",
            model="dandelin/vilt-b32-finetuned-vqa",
            compute_pool_for_log=None,
        )

        with open("tests/integ/snowflake/ml/test_data/cat.jpeg", "rb") as f:
            image_bytes = f.read()

        x_df = pd.DataFrame(
            [
                {
                    "image": image_bytes,
                    "question": "What animal is in the picture?",
                },
            ]
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answers"]))
            for row in res["answers"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)
                self.assertIn("answer", row[0])
                self.assertIn("score", row[0])

        def check_res_with_top_k(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answers"]))
            for row in res["answers"]:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), 3)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {"top_k": 3, "timeout": 30.0},
                    check_res_with_top_k,
                ),
            },
            additional_dependencies=["pillow==12.0"],
        )

    def test_text_generation_non_chat_pipeline_with_params(self) -> None:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model="hf-internal-testing/tiny-random-gpt2",
        )

        x_df = pd.DataFrame(
            [{"inputs": "Once upon a time"}],
        )

        def check_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))
            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)
                self.assertIn("generated_text", row[0])

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))
            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                self.assertGreater(len(row), 0)
                self.assertIn("generated_text", row[0])

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        # Length control (INT64)
                        "max_new_tokens": 10,
                        "min_new_tokens": 2,
                        "max_length": 50,
                        "min_length": 5,
                        # Stopping criteria (STRING, DOUBLE)
                        "early_stopping": "never",
                        "max_time": 60.0,
                        # Sampling strategy (BOOL, INT64, DOUBLE)
                        "do_sample": True,
                        "num_beams": 1,
                        "num_beam_groups": 1,
                        "penalty_alpha": 0.0,
                        # Sampling parameters (INT64, DOUBLE)
                        "temperature": 0.8,
                        "top_k": 50,
                        "top_p": 0.9,
                        "min_p": 0.0,
                        "typical_p": 0.95,
                        "epsilon_cutoff": 0.0,
                        "eta_cutoff": 0.0,
                        # Diversity / repetition (DOUBLE, INT64)
                        "diversity_penalty": 0.0,
                        "repetition_penalty": 1.2,
                        "encoder_repetition_penalty": 1.0,
                        "length_penalty": 1.0,
                        "no_repeat_ngram_size": 3,
                        # Token manipulation (BOOL, INT64)
                        "renormalize_logits": True,
                        "remove_invalid_values": False,
                        # Output control (INT64, BOOL)
                        "num_return_sequences": 1,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "output_scores": False,
                        "output_logits": False,
                        # Token IDs (INT64)
                        "pad_token_id": 0,
                        "eos_token_id": 50256,
                        # Encoder-decoder (INT64)
                        "encoder_no_repeat_ngram_size": 0,
                        # Assisted generation (INT64, STRING)
                        "num_assistant_tokens": 5,
                        "num_assistant_tokens_schedule": "heuristic",
                        # Miscellaneous (DOUBLE, BOOL)
                        "guidance_scale": 1.0,
                        "low_memory": False,
                        "use_cache": True,
                        # Shaped params: list (INT64, shape=(-1,))
                        "suppress_tokens": [50256],
                        "begin_suppress_tokens": [50256],
                        # Shaped params: list-of-lists (INT64, shape=(-1,-1))
                        "bad_words_ids": [[50256]],
                    },
                    check_res_with_params,
                ),
            },
        )

    def test_summarization_pipeline_with_params(self) -> None:
        model = huggingface.TransformersPipeline(
            task="summarization",
            model="Falconsai/text_summarization",
            compute_pool_for_log=None,
        )

        x_df = pd.DataFrame(
            [
                {
                    "documents": (
                        "Snowflake's Data Cloud is powered by an advanced data platform provided as a self-managed "
                        "service. Snowflake enables data storage, processing, and analytic solutions."
                    )
                }
            ],
        )

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["summary_text"]))
            self.assertEqual(res["summary_text"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res_with_params,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        # Task-specific params (BOOL, STRING)
                        "clean_up_tokenization_spaces": True,
                        "truncation": "longest_first",
                        # Length control (INT64)
                        "max_new_tokens": 10,
                        "max_length": 50,
                        "min_length": 1,
                        "min_new_tokens": 1,
                        # Stopping criteria (STRING, DOUBLE)
                        "early_stopping": "never",
                        "max_time": 60.0,
                        # Sampling (BOOL, INT64, DOUBLE)
                        "do_sample": True,
                        "num_beams": 1,
                        "num_beam_groups": 1,
                        "temperature": 0.5,
                        "top_k": 50,
                        "top_p": 0.9,
                        "min_p": 0.0,
                        "typical_p": 1.0,
                        # Repetition (DOUBLE, INT64)
                        "repetition_penalty": 1.1,
                        "no_repeat_ngram_size": 2,
                        "length_penalty": 1.0,
                        # Boolean flags
                        "renormalize_logits": True,
                        "remove_invalid_values": False,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "output_scores": False,
                        "low_memory": False,
                        "use_cache": True,
                        # INT64 scalars
                        "num_return_sequences": 1,
                        "encoder_no_repeat_ngram_size": 0,
                    },
                    check_res_with_params,
                ),
            },
            additional_dependencies=[
                "sentencepiece",
            ],
        )

    def test_translation_pipeline_with_params(self) -> None:
        model = huggingface.TransformersPipeline(
            task="translation_en_to_ja",
            model="Mitsua/elan-mt-tiny-en-ja",
            compute_pool_for_log=None,
        )

        x_df = pd.DataFrame([{"inputs": "Hello, how are you?"}])

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["translation_text"]))
            self.assertEqual(res["translation_text"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res_with_params,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        # Task-specific params (STRING)
                        "src_lang": "en",
                        "tgt_lang": "ja",
                        # Length control (INT64)
                        "max_new_tokens": 5,
                        "max_length": 30,
                        "min_length": 1,
                        # Stopping criteria (STRING, DOUBLE)
                        "early_stopping": "never",
                        "max_time": 60.0,
                        # Sampling (BOOL, INT64, DOUBLE)
                        "do_sample": False,
                        "num_beams": 1,
                        "temperature": 1.0,
                        "top_k": 50,
                        "top_p": 1.0,
                        # Repetition (DOUBLE, INT64)
                        "repetition_penalty": 1.0,
                        "no_repeat_ngram_size": 0,
                        "length_penalty": 1.0,
                        # Boolean flags
                        "renormalize_logits": False,
                        "use_cache": True,
                        # INT64 scalars
                        "num_return_sequences": 1,
                    },
                    check_res_with_params,
                ),
            },
            additional_dependencies=["sentencepiece"],
        )

    def test_text2text_generation_pipeline_with_params(self) -> None:
        model = huggingface.TransformersPipeline(
            task="text2text-generation",
            model="google-t5/t5-small",
            compute_pool_for_log=None,
        )

        x_df = pd.DataFrame(
            [{"inputs": "translate English to French: Hello, how are you?"}],
        )

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["generated_text"]))
            self.assertEqual(res["generated_text"].dtype.type, str)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res_with_params,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        # Task-specific params (BOOL, STRING)
                        "clean_up_tokenization_spaces": True,
                        "truncation": "longest_first",
                        # Length control (INT64)
                        "max_new_tokens": 10,
                        "max_length": 50,
                        "min_length": 1,
                        "min_new_tokens": 1,
                        # Stopping criteria (STRING, DOUBLE)
                        "early_stopping": "never",
                        "max_time": 60.0,
                        # Sampling (BOOL, INT64, DOUBLE)
                        "do_sample": True,
                        "num_beams": 1,
                        "num_beam_groups": 1,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.9,
                        "min_p": 0.0,
                        "typical_p": 1.0,
                        # Repetition (DOUBLE, INT64)
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 3,
                        "length_penalty": 1.0,
                        # Boolean flags
                        "renormalize_logits": True,
                        "remove_invalid_values": False,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "output_scores": False,
                        "low_memory": False,
                        "use_cache": True,
                        # INT64 scalars
                        "num_return_sequences": 1,
                        "encoder_no_repeat_ngram_size": 0,
                    },
                    check_res_with_params,
                ),
            },
        )

    def test_image_to_text_pipeline_with_params(self) -> None:
        model = huggingface.TransformersPipeline(
            task="image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            compute_pool_for_log=None,
        )

        from PIL import Image

        img = Image.new("RGB", (224, 224), color="red")
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        x_df = pd.DataFrame({"images": [img_bytes]})

        def check_res_with_params(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))
            for row in res["outputs"]:
                self.assertIsInstance(row, list)
                for entry in row:
                    self.assertIn("generated_text", entry)

        self._test_registry_model(
            model=model,
            prediction_assert_fns={
                "": (
                    x_df,
                    check_res_with_params,
                ),
            },
            params_assert_fns={
                "": (
                    x_df,
                    {
                        # Length control (INT64)
                        "max_new_tokens": 10,
                        "max_length": 50,
                        "min_length": 1,
                        # Stopping criteria (STRING, DOUBLE)
                        "early_stopping": "never",
                        "max_time": 60.0,
                        # Sampling (BOOL, INT64, DOUBLE)
                        "do_sample": False,
                        "num_beams": 1,
                        "temperature": 1.0,
                        "top_k": 50,
                        "top_p": 1.0,
                        # Repetition (DOUBLE, INT64)
                        "repetition_penalty": 1.0,
                        "no_repeat_ngram_size": 0,
                        "length_penalty": 1.0,
                        # Boolean flags
                        "renormalize_logits": False,
                        "remove_invalid_values": False,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "output_scores": False,
                        "low_memory": False,
                        "use_cache": True,
                        # INT64 scalars
                        "num_return_sequences": 1,
                    },
                    check_res_with_params,
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
