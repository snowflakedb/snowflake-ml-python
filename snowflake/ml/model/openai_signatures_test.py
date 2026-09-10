from absl.testing import absltest

from snowflake.ml.model import openai_signatures
from snowflake.ml.model._signatures import core

_MODEL_PARAM = core.ParamSpec(name="model", dtype=core.DataType.STRING, default_value=None)


class OpenAISignaturesTest(absltest.TestCase):
    def test_params_specs_end_with_optional_model(self) -> None:
        for spec in (
            openai_signatures._OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC,
            openai_signatures._OPENAI_CHAT_SIGNATURE_WITH_PARAMS_SPEC_WITH_CONTENT_FORMAT_STRING,
        ):
            model_param = spec.params[-1]
            self.assertIsInstance(model_param, core.ParamSpec)
            self.assertEqual(model_param, _MODEL_PARAM)

    def test_params_exports_include_model(self) -> None:
        for export in (
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
        ):
            self.assertEqual(list(export.keys()), ["__call__"])
            self.assertEqual(export["__call__"].params[-1], _MODEL_PARAM)

    def test_no_params_specs_do_not_gain_model_param(self) -> None:
        for spec in (
            openai_signatures._OPENAI_CHAT_SIGNATURE_SPEC,
            openai_signatures._OPENAI_CHAT_SIGNATURE_SPEC_WITH_CONTENT_FORMAT_STRING,
        ):
            self.assertFalse(any(param.name == "model" for param in spec.params))


if __name__ == "__main__":
    absltest.main()
