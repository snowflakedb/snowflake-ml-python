from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _utils
from snowflake.ml.model._packager.model_meta import model_meta


class UtilsTest(absltest.TestCase):
    def test_add_explain_method_signature(self) -> None:
        predict_sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="feature1"),
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="feature2"),
            ],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="output1")],
        )

        meta = model_meta.ModelMetadata(
            name="name", env=model_env.ModelEnv(), model_type="custom", signatures={"predict": predict_sig}
        )
        new_meta = _utils.add_explain_method_signature(
            model_meta=meta,
            explain_method="explain",
            target_method="predict",
        )

        self.assertIn("explain", new_meta.signatures)
        explain_sig = new_meta.signatures["explain"]
        self.assertEqual(explain_sig.inputs, predict_sig.inputs)

        for input_feature in predict_sig.inputs:
            self.assertIn(
                model_signature.FeatureSpec(
                    dtype=model_signature.DataType.DOUBLE, name=f"{input_feature.name}_explanation"
                ),
                explain_sig.outputs,
            )


if __name__ == "__main__":
    absltest.main()
