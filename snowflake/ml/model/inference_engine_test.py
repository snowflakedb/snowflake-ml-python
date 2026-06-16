from absl.testing import absltest

from snowflake.ml.model import inference_engine


class InferenceEngineFromValueTest(absltest.TestCase):
    def test_from_value_enum_pass_through(self) -> None:
        self.assertEqual(
            inference_engine.InferenceEngine.from_value(inference_engine.InferenceEngine.VLLM),
            inference_engine.InferenceEngine.VLLM,
        )
        self.assertEqual(
            inference_engine.InferenceEngine.from_value(inference_engine.InferenceEngine.PYTHON_GENERIC),
            inference_engine.InferenceEngine.PYTHON_GENERIC,
        )

    def test_from_value_string_variants(self) -> None:
        test_cases = [
            ("vllm", inference_engine.InferenceEngine.VLLM),
            ("vLLM", inference_engine.InferenceEngine.VLLM),
            ("VLLM", inference_engine.InferenceEngine.VLLM),
            ("  vllm  ", inference_engine.InferenceEngine.VLLM),
            ("python_generic", inference_engine.InferenceEngine.PYTHON_GENERIC),
            ("PYTHON_GENERIC", inference_engine.InferenceEngine.PYTHON_GENERIC),
        ]
        for engine_value, expected_engine in test_cases:
            with self.subTest(engine_value=engine_value):
                self.assertEqual(inference_engine.InferenceEngine.from_value(engine_value), expected_engine)

    def test_from_value_invalid_string(self) -> None:
        with self.assertRaises(ValueError) as error_context:
            inference_engine.InferenceEngine.from_value("foo")
        self.assertEqual(
            str(error_context.exception),
            "Unsupported inference engine 'foo'. Supported engines: vllm, python_generic.",
        )

    def test_from_value_invalid_type(self) -> None:
        with self.assertRaises(ValueError) as error_context:
            inference_engine.InferenceEngine.from_value(123)  # type: ignore[arg-type]
        self.assertEqual(
            str(error_context.exception),
            "Unsupported inference engine type int. Supported engines: vllm, python_generic.",
        )


if __name__ == "__main__":
    absltest.main()
