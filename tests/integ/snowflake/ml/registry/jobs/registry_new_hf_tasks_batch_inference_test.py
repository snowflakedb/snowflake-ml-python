import json
import math
import os
import tempfile

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model.batch import (
    FileEncoding,
    InputFormat,
    InputSpec,
    JobSpec,
    OutputSpec,
)
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base


# TODO(kdickerson): Reorganize these tests by input modality (image, audio, etc.) alongside
# existing multi-modality test files once we've built confidence in these tests.
class TestRegistryNewHFTasksBatchInferenceInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        cls.cache_dir.cleanup()

    @absltest.skip(
        "Requires tesseract system binary which is not available in the inference image. "
        "Re-enable when the inference image includes tesseract."
    )
    def test_document_question_answering(self) -> None:
        from transformers import pipeline

        model = pipeline(task="document-question-answering", model="impira/layoutlm-document-qa")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg", "What is in the image?"],
            [f"{input_files_stage_location}cat.jpeg", "What animal is this?"],
        ]
        column_names = ["image", "question"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGE": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_document_qa(res: pd.DataFrame) -> None:
            self.assertIn("answers", res.columns)
            self.assertEqual(len(res), 2)
            for _, row in res.iterrows():
                answers = json.loads(row["answers"])
                self.assertIsInstance(answers, list)
                self.assertGreater(len(answers), 0)
                for answer in answers:
                    self.assertIn("score", answer)
                    self.assertIn("answer", answer)
                    self.assertIsInstance(answer["score"], float)
                    self.assertIsInstance(answer["answer"], str)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow", "pytesseract"],
            prediction_assert_fn=check_document_qa,
        )

    @absltest.skip(
        "Requires torch>=2.6 for torch.load with weights_only=True (CVE security fix). "
        "Re-enable when inference image upgrades torch."
    )
    def test_visual_question_answering(self) -> None:
        from transformers import pipeline

        model = pipeline(task="visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg", "What is in the image?"],
            [f"{input_files_stage_location}cat.jpeg", "What animal is this?"],
        ]
        column_names = ["image", "question"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGE": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_vqa(res: pd.DataFrame) -> None:
            self.assertIn("answers", res.columns)
            self.assertEqual(len(res), 2)
            for _, row in res.iterrows():
                answers = json.loads(row["answers"])
                self.assertIsInstance(answers, list)
                self.assertGreater(len(answers), 0)
                for answer in answers:
                    self.assertIn("answer", answer)
                    self.assertIn("score", answer)
                # Check that at least one answer for this row contains "cat"
                row_answers = [a["answer"].lower() for a in answers]
                self.assertTrue(
                    any("cat" in a for a in row_answers),
                    f"Expected 'cat' in at least one answer, got: {row_answers}",
                )

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            prediction_assert_fn=check_vqa,
        )

    def test_image_feature_extraction(self) -> None:
        from transformers import pipeline

        model = pipeline(task="image-feature-extraction", model="google/vit-base-patch16-224-in21k")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
        ]
        column_names = ["images"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_image_features(res: pd.DataFrame) -> None:
            self.assertIn("feature_extraction", res.columns)
            self.assertEqual(len(res), 3)
            embeddings = []
            for _, row in res.iterrows():
                features = json.loads(row["feature_extraction"])
                self.assertIsInstance(features, list)
                self.assertGreater(len(features), 0)
                # Features may be nested (list of lists) — flatten to validate values
                flat = features
                if isinstance(features[0], list):
                    flat = [v for sublist in features for v in sublist]
                for val in flat:
                    self.assertIsInstance(val, float)
                    self.assertTrue(math.isfinite(val), f"Non-finite value in embedding: {val}")
                embeddings.append(features)
            # Same input image should produce identical embeddings
            self.assertEqual(embeddings[0], embeddings[1])
            self.assertEqual(embeddings[0], embeddings[2])

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            prediction_assert_fn=check_image_features,
        )

    @absltest.skip(
        "Blocked on model logger support from https://github.com/snowflake-eng/snowml/pull/5733; "
        "re-enable once that change lands in the model logger."
    )
    def test_image_to_text(self) -> None:
        from transformers import pipeline

        model = pipeline(task="image-to-text", model="microsoft/git-base")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
        ]
        column_names = ["images"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_image_to_text(res: pd.DataFrame) -> None:
            self.assertIn("outputs", res.columns)
            self.assertEqual(len(res), 3)
            all_captions = []
            for _, row in res.iterrows():
                outputs = json.loads(row["outputs"])
                self.assertIsInstance(outputs, list)
                self.assertGreater(len(outputs), 0)
                for output in outputs:
                    self.assertIn("generated_text", output)
                    self.assertIsInstance(output["generated_text"], str)
                    self.assertGreater(len(output["generated_text"]), 0)
                    all_captions.append(output["generated_text"].lower())
            self.assertTrue(
                any("cat" in caption for caption in all_captions),
                f"Expected 'cat' in at least one caption, got: {all_captions}",
            )

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            prediction_assert_fn=check_image_to_text,
        )

    def test_object_detection(self) -> None:
        from transformers import pipeline

        model = pipeline(task="object-detection", model="hustvl/yolos-tiny")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
            [f"{input_files_stage_location}cat.jpeg"],
        ]
        column_names = ["images"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_object_detection(res: pd.DataFrame) -> None:
            self.assertIn("detections", res.columns)
            self.assertEqual(len(res), 3)
            for _, row in res.iterrows():
                detections = json.loads(row["detections"])
                self.assertIsInstance(detections, list)
                for det in detections:
                    self.assertIn("label", det)
                    self.assertIn("score", det)
                    self.assertIn("box", det)
                    box = det["box"]
                    for key in ("xmin", "ymin", "xmax", "ymax"):
                        self.assertIn(key, box)
                        self.assertGreaterEqual(box[key], 0)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            prediction_assert_fn=check_object_detection,
        )

    @absltest.skip(
        "Requires torch>=2.6 for torch.load with weights_only=True (CVE security fix). "
        "Re-enable when inference image upgrades torch."
    )
    def test_zero_shot_image_classification(self) -> None:
        from transformers import pipeline

        model = pipeline(task="zero-shot-image-classification", model="openai/clip-vit-base-patch32")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg", ["cat", "dog", "bird"]],
            [f"{input_files_stage_location}cat.jpeg", ["animal", "vehicle", "furniture"]],
        ]
        column_names = ["images", "candidate_labels"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_zero_shot_image_classification(res: pd.DataFrame) -> None:
            self.assertIn("labels", res.columns)
            self.assertEqual(len(res), 2)
            for _, row in res.iterrows():
                labels = json.loads(row["labels"])
                self.assertIsInstance(labels, list)
                self.assertGreater(len(labels), 0)
                for entry in labels:
                    self.assertIn("label", entry)
                    self.assertIn("score", entry)
            # First row has candidate_labels ["cat", "dog", "bird"] — "cat" should be top-scoring
            first_row_labels = json.loads(res.iloc[0]["labels"])
            top_label = max(first_row_labels, key=lambda x: x["score"])
            self.assertEqual(
                top_label["label"],
                "cat",
                f"Expected top label to be 'cat', got '{top_label['label']}' with scores: {first_row_labels}",
            )
            # Second row has candidate_labels ["animal", "vehicle", "furniture"] — "animal" should be top-scoring
            second_row_labels = json.loads(res.iloc[1]["labels"])
            top_label = max(second_row_labels, key=lambda x: x["score"])
            self.assertEqual(
                top_label["label"],
                "animal",
                f"Expected top label to be 'animal', got '{top_label['label']}' with scores: {second_row_labels}",
            )

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            prediction_assert_fn=check_zero_shot_image_classification,
        )

    def test_zero_shot_object_detection(self) -> None:
        from transformers import pipeline

        model = pipeline(task="zero-shot-object-detection", model="google/owlvit-base-patch32")

        (
            job_name,
            output_stage_location,
            input_files_stage_location,
        ) = self._prepare_job_name_and_stage_for_batch_inference()

        cat_file_path = "tests/integ/snowflake/ml/test_data/cat.jpeg"
        self.session.sql(
            f"PUT 'file://{cat_file_path}' {input_files_stage_location} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ).collect()

        data = [
            [f"{input_files_stage_location}cat.jpeg", ["cat", "dog", "bird"]],
            [f"{input_files_stage_location}cat.jpeg", ["animal", "toy", "furniture"]],
            [f"{input_files_stage_location}cat.jpeg", ["pet", "plant", "food"]],
        ]
        column_names = ["images", "candidate_labels"]
        input_df = self.session.create_dataframe(data, schema=column_names)

        column_handling = {
            "IMAGES": {
                "input_format": InputFormat.FULL_STAGE_PATH,
                "convert_to": FileEncoding.RAW_BYTES,
            }
        }

        def check_zero_shot_object_detection(res: pd.DataFrame) -> None:
            self.assertIn("detections", res.columns)
            self.assertEqual(len(res), 3)
            for _, row in res.iterrows():
                detections = json.loads(row["detections"])
                self.assertIsInstance(detections, list)
                for det in detections:
                    self.assertIn("label", det)
                    self.assertIn("score", det)
                    self.assertIn("box", det)

        self._test_registry_batch_inference(
            model=model,
            X=input_df,
            compute_pool="SYSTEM_COMPUTE_POOL_CPU",
            output_spec=OutputSpec(stage_location=output_stage_location),
            input_spec=InputSpec(column_handling=column_handling),
            job_spec=JobSpec(job_name=job_name, replicas=1),
            pip_requirements=["pillow"],
            prediction_assert_fn=check_zero_shot_object_detection,
        )


if __name__ == "__main__":
    absltest.main()
