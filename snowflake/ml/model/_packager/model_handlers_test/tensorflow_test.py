import os
import tempfile
import warnings
from typing import Optional

import tensorflow as tf
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._signatures import (
    tensorflow_handler,
    utils as model_signature_utils,
)


class SimpleModule(tf.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    @tf.function  # type: ignore[misc]
    def __call__(self, tensor: tf.Tensor) -> tf.Tensor:
        return self.a_variable * tensor + self.non_trainable_variable


class TensorflowHandlerTest(parameterized.TestCase):
    def test_tensorflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            simple_module = SimpleModule(name="simple")
            x = tf.constant([[5.0], [10.0]])
            y_pred = simple_module(x)
            s = {"__call__": model_signature.infer_signature([x], [y_pred])}
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=simple_module,
                    signatures={**s, "another_forward": s["__call__"]},
                    metadata={"author": "halu", "version": "1"},
                )

            with self.assertRaises(NotImplementedError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=simple_module,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=simple_module,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            x_df = model_signature_utils.rename_pandas_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(data=[x], ensure_serializable=False),
                s["__call__"].inputs,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert callable(pk.model)
                tf.assert_equal(pk.model(x), y_pred)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                assert callable(pk.model)
                tf.assert_equal(
                    tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                        pk.model(x_df), s["__call__"].outputs
                    )[0],
                    y_pred,
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1")).save(
                name="model1_no_sig_1",
                model=simple_module,
                sample_input_data=[x],
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            tf.assert_equal(pk.model(x), y_pred)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            tf.assert_equal(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(pk.model(x_df), s["__call__"].outputs)[
                    0
                ],
                y_pred,
            )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2")).save(
                name="model1_no_sig_2",
                model=simple_module,
                sample_input_data=x_df,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig_2"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert callable(pk.model)
            tf.assert_equal(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(pk.model(x_df), s["__call__"].outputs)[
                    0
                ],
                y_pred,
            )


if __name__ == "__main__":
    absltest.main()
