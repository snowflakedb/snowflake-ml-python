import copy
import json
import os
import tempfile
from typing import TYPE_CHECKING, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from absl.testing import absltest
from packaging import version

from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers.huggingface_pipeline import (
    HuggingFacePipelineHandler,
)
from snowflake.ml.model._signatures import utils
from snowflake.ml.model.models import huggingface_pipeline

if TYPE_CHECKING:
    import transformers


class HuggingFacePipelineHandlerTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["HF_HOME"] = self.cache_dir.name

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_hf_home:
            os.environ["HF_HOME"] = self._original_hf_home
        else:
            del os.environ["HF_HOME"]
        self.cache_dir.cleanup()

    def test_get_device_config(self) -> None:
        self.assertDictEqual(HuggingFacePipelineHandler._get_device_config(), {})
        self.assertDictEqual(HuggingFacePipelineHandler._get_device_config(use_gpu=False), {})
        self.assertDictEqual(HuggingFacePipelineHandler._get_device_config(use_gpu=True), {"device_map": "auto"})
        self.assertDictEqual(
            HuggingFacePipelineHandler._get_device_config(device_map="balanced"), {"device_map": "balanced"}
        )
        self.assertDictEqual(
            HuggingFacePipelineHandler._get_device_config(use_gpu=False, device_map="balanced"),
            {"device_map": "balanced"},
        )
        self.assertDictEqual(
            HuggingFacePipelineHandler._get_device_config(use_gpu=True, device_map="balanced"),
            {"device_map": "balanced"},
        )
        self.assertDictEqual(HuggingFacePipelineHandler._get_device_config(device="cuda:0"), {"device": "cuda:0"})
        self.assertDictEqual(
            HuggingFacePipelineHandler._get_device_config(use_gpu=False, device="cuda:0"), {"device": "cuda:0"}
        )
        self.assertDictEqual(
            HuggingFacePipelineHandler._get_device_config(use_gpu=True, device="cuda:0"), {"device": "cuda:0"}
        )

    def _check_loaded_pipeline_object(self, original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
        self.assertEqual(original.framework, loaded.framework)
        self.assertEqual(original.task, loaded.task)
        self.assertEqual(original._batch_size, loaded._batch_size)
        self.assertEqual(original._num_workers, loaded._num_workers)
        self.assertDictEqual(original._preprocess_params, loaded._preprocess_params)
        self.assertDictEqual(original._forward_params, loaded._forward_params)
        self.assertDictEqual(original._postprocess_params, loaded._postprocess_params)

    def _check_loaded_pipeline_wrapper_object(
        self,
        original: huggingface_pipeline.HuggingFacePipelineModel,
        loaded: huggingface_pipeline.HuggingFacePipelineModel,
        use_gpu: bool = False,
    ) -> None:
        original_dict = original.__dict__
        if use_gpu:
            original_dict = copy.deepcopy(original_dict)
            if "torch_dtype" not in original_dict:
                original_dict["torch_dtype"] = "auto"
            original_dict["device_map"] = "auto"
        self.assertDictEqual(original_dict, loaded.__dict__)

    def _basic_test_case(
        self,
        task: str,
        model_id: str,
        udf_test_input: pd.DataFrame,
        options: Dict[str, object],
        check_pipeline_fn: Callable[["transformers.Pipeline", "transformers.Pipeline"], None],
        check_udf_res_fn: Callable[[pd.DataFrame], None],
        check_gpu: bool = True,
    ) -> None:
        import transformers

        model = transformers.pipeline(task=task, model=model_id, **options)

        sig = utils.huggingface_pipeline_signature_auto_infer(task=task, params=options)
        assert sig

        s = {"__call__": sig}

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**s, "another_predict": s["__call__"]},
                    metadata={"author": "halu", "version": "1"},
                )
            with self.assertRaises(NotImplementedError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, transformers.Pipeline)
            self._check_loaded_pipeline_object(model, pk.model)

            check_pipeline_fn(model, pk.model)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            res = pk.model(udf_test_input.copy(deep=True))
            check_udf_res_fn(res)

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=model,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, transformers.Pipeline)
            self._check_loaded_pipeline_object(model, pk.model)

            check_pipeline_fn(model, pk.model)
            self.assertEqual(s, pk.meta.signatures)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            res = pk.model(udf_test_input.copy(deep=True))
            check_udf_res_fn(res)

        wrapper_model = huggingface_pipeline.HuggingFacePipelineModel(
            task=task, model=model_id, **options  # type:ignore[arg-type]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertWarns(UserWarning):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=wrapper_model,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                name="model1_no_sig",
                model=wrapper_model,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, huggingface_pipeline.HuggingFacePipelineModel)
            self._check_loaded_pipeline_wrapper_object(wrapper_model, pk.model)
            self.assertEqual(s, pk.meta.signatures)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            res = pk.model(udf_test_input.copy(deep=True))
            check_udf_res_fn(res)

            if check_gpu:
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
                pk.load(options={"use_gpu": True})
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, huggingface_pipeline.HuggingFacePipelineModel)
                self._check_loaded_pipeline_wrapper_object(wrapper_model, pk.model, use_gpu=True)
                self.assertEqual(s, pk.meta.signatures)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
                pk.load(as_custom_model=True, options={"use_gpu": True})
                assert pk.model
                assert pk.meta
                assert callable(pk.model)
                res = pk.model(udf_test_input.copy(deep=True))
                check_udf_res_fn(res)

    def test_conversational_pipeline(self) -> None:
        import transformers

        if version.parse(transformers.__version__) >= version.parse("4.42.0"):
            self.skipTest("This test is not compatible with transformers>=4.42.0")

        x = transformers.Conversation(
            text="Do you know how to say Snowflake in French?",
            past_user_inputs=["Do you speak French?"],
            generated_responses=["Yes I do."],
        )

        x_df = pd.DataFrame(
            [
                {
                    "user_inputs": x.past_user_inputs + [x.new_user_input],
                    "generated_responses": x.generated_responses,
                },
            ]
        )

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(copy.deepcopy(x))
            loaded_res = loaded(copy.deepcopy(x))
            assert isinstance(original_res, transformers.Conversation)
            assert isinstance(loaded_res, transformers.Conversation)
            self.assertListEqual(original_res.generated_responses, loaded_res.generated_responses)
            self.assertListEqual(original_res.past_user_inputs, loaded_res.past_user_inputs)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["generated_responses"]))

            for row in res["generated_responses"]:
                self.assertIsInstance(row, list)
                for resp in row:
                    self.assertIsInstance(resp, str)

        self._basic_test_case(
            task="conversational",
            model_id="ToddGoldfarb/Cadet-Tiny",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

    def test_fill_mask_pipeline(self) -> None:
        x = ["LynYuu is the <mask> of the Grand Duchy of Yu."]

        x_df = pd.DataFrame(
            [x],
            columns=["inputs"],
        )

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, str)
                resp = json.loads(row)
                self.assertIsInstance(resp, list)
                self.assertIn("score", resp[0])
                self.assertIn("token", resp[0])
                self.assertIn("token_str", resp[0])
                self.assertIn("sequence", resp[0])

        self._basic_test_case(
            task="fill-mask",
            model_id="sshleifer/tiny-distilroberta-base",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

        self._basic_test_case(
            task="fill-mask",
            model_id="sshleifer/tiny-distilroberta-base",
            udf_test_input=x_df,
            options={"top_k": 5},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

        self._basic_test_case(
            task="fill-mask",
            model_id="sshleifer/tiny-distilroberta-base",
            udf_test_input=x_df,
            options={"batch_size": 8},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

    def test_ner_pipeline(
        self,
    ) -> None:
        x = ["My name is Izumi and I live in Tokyo, Japan."]

        x_df = pd.DataFrame(
            [x],
            columns=["inputs"],
        )

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def get_check_udf_res_fn(
            aggregation_strategy: Optional[str] = None,
        ) -> Callable[[pd.DataFrame], None]:
            def check_udf_res(res: pd.DataFrame) -> None:
                pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

                for row in res["outputs"]:
                    self.assertIsInstance(row, str)
                    resp = json.loads(row)
                    self.assertIsInstance(resp, list)
                    if aggregation_strategy:
                        self.assertIn("entity_group", resp[0])
                    else:
                        self.assertIn("entity", resp[0])
                        self.assertIn("index", resp[0])

                    self.assertIn("score", resp[0])
                    self.assertIn("word", resp[0])
                    self.assertIn("start", resp[0])
                    self.assertIn("end", resp[0])

            return check_udf_res

        self._basic_test_case(
            task="ner",
            model_id="hf-internal-testing/tiny-bert-for-token-classification",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=get_check_udf_res_fn(),
            check_gpu=False,  # Model being used does not support accelerate.
        )

        self._basic_test_case(
            task="ner",
            model_id="hf-internal-testing/tiny-bert-for-token-classification",
            udf_test_input=x_df,
            options={"aggregation_strategy": "simple"},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=get_check_udf_res_fn(aggregation_strategy="simple"),
            check_gpu=False,  # Model being used does not support accelerate.
        )

    def test_question_answering_pipeline(
        self,
    ) -> None:
        x = [
            {
                "question": "What did Doris want to do?",
                "context": (
                    "Doris is a cheerful mermaid from the ocean depths. She transformed into a bipedal creature "
                    'and came to see everyone because she wanted to "learn more about the world of athletics."'
                    " She dislikes cuisines with seafood."
                ),
            }
        ]

        x_df = pd.DataFrame(x)

        def get_check_pipeline_fn(
            top_k: int = 1,
        ) -> Callable[["transformers.Pipeline", "transformers.Pipeline"], None]:
            def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
                original_res = original(x)
                loaded_res = loaded(x)
                if top_k == 1:
                    self.assertDictEqual(original_res, loaded_res)
                else:
                    self.assertListEqual(original_res, loaded_res)

            return check_pipeline

        def get_check_udf_res_fn(
            top_k: int = 1,
        ) -> Callable[[pd.DataFrame], None]:
            def check_udf_res(res: pd.DataFrame) -> None:
                if top_k == 1:
                    pd.testing.assert_index_equal(res.columns, pd.Index(["score", "start", "end", "answer"]))

                    self.assertEqual(res["score"].dtype.type, np.float64)
                    self.assertEqual(res["start"].dtype.type, np.int64)
                    self.assertEqual(res["end"].dtype.type, np.int64)
                    self.assertEqual(res["answer"].dtype.type, np.object_)
                else:
                    pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

                    for row in res["outputs"]:
                        self.assertIsInstance(row, str)
                        resp = json.loads(row)
                        self.assertIsInstance(resp, list)
                        self.assertIn("score", resp[0])
                        self.assertIn("start", resp[0])
                        self.assertIn("end", resp[0])
                        self.assertIn("answer", resp[0])

            return check_udf_res

        self._basic_test_case(
            task="question-answering",
            model_id="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=get_check_pipeline_fn(),
            check_udf_res_fn=get_check_udf_res_fn(),
            check_gpu=False,  # Model being used does not support accelerate.
        )

        self._basic_test_case(
            task="question-answering",
            model_id="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            udf_test_input=x_df,
            options={"align_to_words": True},
            check_pipeline_fn=get_check_pipeline_fn(),
            check_udf_res_fn=get_check_udf_res_fn(),
            check_gpu=False,  # Model being used does not support accelerate.
        )

        self._basic_test_case(
            task="question-answering",
            model_id="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            udf_test_input=x_df,
            options={"top_k": 5},
            check_pipeline_fn=get_check_pipeline_fn(top_k=5),
            check_udf_res_fn=get_check_udf_res_fn(top_k=5),
            check_gpu=False,  # Model being used does not support accelerate.
        )

    def test_summarization_pipeline(
        self,
    ) -> None:
        x = [
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

        x_df = pd.DataFrame([x], columns=["documents"])

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["summary_text"]))

            self.assertEqual(res["summary_text"].dtype.type, np.object_)

        self._basic_test_case(
            task="summarization",
            model_id="sshleifer/tiny-mbart",
            udf_test_input=x_df,
            # This model is stored in fp16, but the architecture does not support it,
            # it will messed up the auto dtype loading.
            options={"torch_dtype": torch.float32},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "when `return_tensors` set to `True` has not been supported yet.",
        ):
            self._basic_test_case(
                task="summarization",
                model_id="sshleifer/tiny-mbart",
                udf_test_input=x_df,
                options={"return_tensors": True, "torch_dtype": torch.float32},
                check_pipeline_fn=check_pipeline,
                check_udf_res_fn=check_udf_res,
            )

    def test_table_question_answering_pipeline(
        self,
    ) -> None:
        x = [
            {
                "query": "Which channel has the most subscribers?",
                "table": {
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
                },
            }
        ]

        x_df = pd.DataFrame([{"query": element["query"], "table": json.dumps(element["table"])} for element in x])

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertDictEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["answer", "coordinates", "cells", "aggregator"]))

            self.assertEqual(res["answer"].dtype.type, np.object_)
            self.assertEqual(res["coordinates"].dtype.type, np.object_)
            self.assertIsInstance(res["coordinates"][0], list)
            self.assertEqual(res["cells"].dtype.type, np.object_)
            self.assertIsInstance(res["cells"][0], list)
            self.assertEqual(res["aggregator"].dtype.type, np.object_)

        self._basic_test_case(
            task="table-question-answering",
            model_id="google/tapas-tiny-finetuned-wtq",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,  # Model being used does not support accelerate.
        )

        self._basic_test_case(
            task="table-question-answering",
            model_id="google/tapas-tiny-finetuned-wtq",
            udf_test_input=x_df,
            options={"sequential": True},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,  # Model being used does not support accelerate.
        )

    def test_text_classification_pair_pipeline(
        self,
    ) -> None:
        x = [{"text": "I like you.", "text_pair": "I love you, too."}]

        x_df = pd.DataFrame(x)

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["label", "score"]))

            self.assertEqual(res["label"].dtype.type, np.object_)
            self.assertEqual(res["score"].dtype.type, np.float64)

        self._basic_test_case(
            task="text-classification",
            model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,
        )

    def test_text_classification_pipeline(
        self,
    ) -> None:
        x = [
            {
                "text": "I am wondering if I should have udon or rice for lunch",
                "text_pair": "",
            }
        ]

        x_df = pd.DataFrame(x)

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def get_check_udf_res_fn(
            top_k: Optional[int] = None,
        ) -> Callable[[pd.DataFrame], None]:
            def check_udf_res(res: pd.DataFrame) -> None:
                if top_k:
                    pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

                    for row in res["outputs"]:
                        self.assertIsInstance(row, str)
                        resp = json.loads(row)
                        self.assertIsInstance(resp, list)
                        self.assertIn("label", resp[0])
                        self.assertIn("score", resp[0])
                else:
                    pd.testing.assert_index_equal(res.columns, pd.Index(["label", "score"]))

                    self.assertEqual(res["label"].dtype.type, np.object_)
                    self.assertEqual(res["score"].dtype.type, np.float64)

            return check_udf_res

        self._basic_test_case(
            task="text-classification",
            model_id="hf-internal-testing/tiny-random-distilbert",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=get_check_udf_res_fn(),
            check_gpu=False,
        )

        self._basic_test_case(
            task="text-classification",
            model_id="hf-internal-testing/tiny-random-distilbert",
            udf_test_input=x_df,
            options={"top_k": 1},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=get_check_udf_res_fn(top_k=1),
            check_gpu=False,
        )

        self._basic_test_case(
            task="text-classification",
            model_id="hf-internal-testing/tiny-random-distilbert",
            udf_test_input=x_df,
            options={"top_k": 3},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=get_check_udf_res_fn(top_k=3),
            check_gpu=False,
        )

    def test_text_generation_pipeline(
        self,
    ) -> None:
        x = ['A descendant of the Lost City of Atlantis, who swam to Earth while saying, "']

        x_df = pd.DataFrame([x], columns=["inputs"])

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["outputs"]))

            for row in res["outputs"]:
                self.assertIsInstance(row, str)
                resp = json.loads(row)
                self.assertIsInstance(resp, list)
                self.assertIn("generated_text", resp[0])

        self._basic_test_case(
            task="text-generation",
            model_id="sshleifer/tiny-ctrl",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "when `return_tensors` set to `True` has not been supported yet.",
        ):
            self._basic_test_case(
                task="text-generation",
                model_id="sshleifer/tiny-ctrl",
                udf_test_input=x_df,
                options={"return_tensors": True},
                check_pipeline_fn=check_pipeline,
                check_udf_res_fn=check_udf_res,
                check_gpu=False,
            )

        self._basic_test_case(
            task="text-generation",
            model_id="sshleifer/tiny-ctrl",
            udf_test_input=x_df,
            options={"num_return_sequences": 1},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,
        )

        self._basic_test_case(
            task="text-generation",
            model_id="sshleifer/tiny-ctrl",
            udf_test_input=x_df,
            options={"num_return_sequences": 4, "do_sample": True, "num_beams": 1},
            check_pipeline_fn=lambda x, y: None,  # ignore this check
            check_udf_res_fn=check_udf_res,
            check_gpu=False,
        )

    def test_text2text_generation_pipeline(
        self,
    ) -> None:
        x = ['A descendant of the Lost City of Atlantis, who swam to Earth while saying, "']

        x_df = pd.DataFrame([x], columns=["inputs"])

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["generated_text"]))
            self.assertEqual(res["generated_text"].dtype.type, np.object_)

        self._basic_test_case(
            task="text2text-generation",
            model_id="patrickvonplaten/t5-tiny-random",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "when `return_tensors` set to `True` has not been supported yet.",
        ):
            self._basic_test_case(
                task="text2text-generation",
                model_id="patrickvonplaten/t5-tiny-random",
                udf_test_input=x_df,
                options={"return_tensors": True},
                check_pipeline_fn=check_pipeline,
                check_udf_res_fn=check_udf_res,
            )

    def test_translation_pipeline(
        self,
    ) -> None:
        x = ["Snowflake's Data Cloud is powered by an advanced data platform provided as a self-managed service."]

        x_df = pd.DataFrame([x], columns=["inputs"])

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(x)
            loaded_res = loaded(x)
            self.assertListEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["translation_text"]))
            self.assertEqual(res["translation_text"].dtype.type, np.object_)

        self._basic_test_case(
            task="translation_en_to_ja",
            model_id="patrickvonplaten/t5-tiny-random",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

        self._basic_test_case(
            task="translation",
            model_id="patrickvonplaten/t5-tiny-random",
            udf_test_input=x_df,
            options={"src_lang": "en", "tgt_lang": "de"},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "when `return_tensors` set to `True` has not been supported yet.",
        ):
            self._basic_test_case(
                task="translation_en_to_ja",
                model_id="patrickvonplaten/t5-tiny-random",
                udf_test_input=x_df,
                options={"return_tensors": True},
                check_pipeline_fn=check_pipeline,
                check_udf_res_fn=check_udf_res,
            )

    def test_zero_shot_classification_pipeline(
        self,
    ) -> None:
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

        def check_pipeline(original: "transformers.Pipeline", loaded: "transformers.Pipeline") -> None:
            original_res = original(
                "I have a problem with Snowflake that needs to be resolved asap!!",
                ["urgent", "not urgent"],
            )
            loaded_res = loaded(
                "I have a problem with Snowflake that needs to be resolved asap!!",
                ["urgent", "not urgent"],
            )
            self.assertDictEqual(original_res, loaded_res)

        def check_udf_res(res: pd.DataFrame) -> None:
            pd.testing.assert_index_equal(res.columns, pd.Index(["sequence", "labels", "scores"]))
            self.assertEqual(res["sequence"].dtype.type, np.object_)
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
            self.assertIsInstance(res["labels"][0], list)
            self.assertIsInstance(res["labels"][1], list)

        self._basic_test_case(
            task="zero-shot-classification",
            model_id="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            udf_test_input=x_df,
            options={},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,
        )

        self._basic_test_case(
            task="zero-shot-classification",
            model_id="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            udf_test_input=x_df,
            options={"multi_label": True},
            check_pipeline_fn=check_pipeline,
            check_udf_res_fn=check_udf_res,
            check_gpu=False,
        )


if __name__ == "__main__":
    absltest.main()
