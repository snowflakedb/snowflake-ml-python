import contextlib
import http
import logging
import os
import tempfile
from typing import Any, Dict, List

from absl.testing import absltest
from absl.testing.absltest import mock
from starlette import testclient

from snowflake.ml._internal import file_utils
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model.models import llm

logger = logging.getLogger(__name__)


class MainVllmTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["HF_HOME"] = cls.cache_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            del os.environ["HF_HOME"]
        cls.cache_dir.cleanup()

    def setUp(self) -> None:
        super().setUp()

    def setup_lora_model(self) -> str:
        import peft

        ft_model = peft.AutoPeftModelForCausalLM.from_pretrained(  # type: ignore[attr-defined]
            "peft-internal-testing/opt-350m-lora",
            device_map="auto",
        )
        tmpdir = self.create_tempdir().full_path
        ft_model.save_pretrained(tmpdir)
        options = llm.LLMOptions(
            max_batch_size=100,
        )
        model = llm.LLM(tmpdir, options=options)
        tmpdir = self.create_tempdir()
        tmpdir_for_zip = self.create_tempdir()
        zip_full_path = os.path.join(tmpdir_for_zip.full_path, "model.zip")
        model_packager.ModelPackager(tmpdir.full_path).save(
            name="test_model",
            model=model,
            metadata={"author": "halu", "version": "1"},
        )
        file_utils.make_archive(zip_full_path, tmpdir.full_path)
        return zip_full_path

    def setup_pretrain_model(self) -> str:
        options = llm.LLMOptions(
            max_batch_size=100,
            enable_tp=True,
        )
        model = llm.LLM("facebook/opt-350m", options=options)
        tmpdir = self.create_tempdir()
        tmpdir_for_zip = self.create_tempdir()
        zip_full_path = os.path.join(tmpdir_for_zip.full_path, "model.zip")
        model_packager.ModelPackager(tmpdir.full_path).save(
            name="test_model",
            model=model,
            metadata={"author": "halu", "version": "1"},
        )
        file_utils.make_archive(zip_full_path, tmpdir.full_path)
        return zip_full_path

    @contextlib.contextmanager
    def common_helper(self, model_zip_path):  # type: ignore[no-untyped-def]
        with mock.patch.dict(
            os.environ,
            {
                "TARGET_METHOD": "infer",
                "MODEL_ZIP_STAGE_PATH": model_zip_path,
            },
        ):
            import main

            client = testclient.TestClient(main.app)
            yield main, client

    def generate_data(self, dfl: List[str]) -> Dict[str, Any]:
        res = []
        for i, v in enumerate(dfl):
            res.append(
                [
                    i,
                    {
                        "_ID": i,
                        "input": v,
                    },
                ]
            )
        return {"data": res}

    def test_happy_lora_case(self) -> None:
        model_zip_path = self.setup_lora_model()
        with self.common_helper(model_zip_path) as (_, client):
            prompts = ["1+1=", "2+2="]
            data = self.generate_data(prompts)
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, http.HTTPStatus.OK)

    def test_happy_pretrain_case(self) -> None:
        model_zip_path = self.setup_pretrain_model()
        with self.common_helper(model_zip_path) as (_, client):
            prompts = ["1+1=", "2+2="]
            data = self.generate_data(prompts)
            response = client.post("/predict", json=data)
            self.assertEqual(response.status_code, http.HTTPStatus.OK)


if __name__ == "__main__":
    absltest.main()
