import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model.models import llm


class LLMTest(absltest.TestCase):
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

    def test_llm(self) -> None:
        import peft

        ft_model = peft.AutoPeftModelForCausalLM.from_pretrained(  # type: ignore[attr-defined]
            "peft-internal-testing/tiny-OPTForCausalLM-lora",
            device_map="auto",
        )
        tmp_dir = self.create_tempdir().full_path
        ft_model.save_pretrained(tmp_dir)
        llm.LLM(model_id_or_path=tmp_dir)


if __name__ == "__main__":
    absltest.main()
