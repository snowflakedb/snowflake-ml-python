import asyncio
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.model._packager import model_packager


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class AnotherDemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(input[["c1", "c2"]])


class ComposeModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            (self.context.model_ref("m1").predict(input)["c1"] + self.context.model_ref("m2").predict(input)["output"])
            / 2,
            columns=["output"],
        )


class AsyncComposeModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    async def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        res1 = await self.context.model_ref("m1").predict.async_run(input)
        res_sum = res1["output"] + self.context.model_ref("m2").predict(input)["output"]
        return pd.DataFrame(res_sum / 2)


class DemoModelWithArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(context.path("bias"), encoding="utf-8") as f:
            v = int(f.read())
        self.bias = v

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class DemoModelWithManyArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(os.path.join(context.path("bias"), "bias1"), encoding="utf-8") as f:
            v1 = int(f.read())
        with open(os.path.join(context.path("bias"), "bias2"), encoding="utf-8") as f:
            v2 = int(f.read())
        self.bias = v1 + v2

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class CustomHandlerTest(absltest.TestCase):
    def test_custom_model_with_multiple_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "bias"))
            with open(os.path.join(tmpdir, "bias", "bias1"), "w", encoding="utf-8") as f:
                f.write("25")
            with open(os.path.join(tmpdir, "bias", "bias2"), "w", encoding="utf-8") as f:
                f.write("68")
            lm = DemoModelWithManyArtifacts(
                custom_model.ModelContext(
                    models={}, artifacts={"bias": os.path.join(tmpdir, "bias", "")}
                )  # Test sanitizing user path input.
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = {"predict": model_signature.infer_signature(d, lm.predict(d))}
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=lm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, DemoModelWithManyArtifacts)
                res = pk.model.predict(d)
                np.testing.assert_allclose(res["output"], pd.Series(np.array([94, 97])))

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, DemoModelWithManyArtifacts)
                res = pk.model.predict(d)
                np.testing.assert_allclose(res["output"], pd.Series(np.array([94, 97])))
                self.assertEqual(pk.meta.metadata["author"] if pk.meta.metadata else None, "halu")

                model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig")).save(
                    name="model1_no_sig",
                    model=lm,
                    sample_input=d,
                    metadata={"author": "halu", "version": "1"},
                    options={"relax_version": False},
                )

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1_no_sig"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, DemoModelWithManyArtifacts)
                res = pk.model.predict(d)
                np.testing.assert_allclose(res["output"], pd.Series(np.array([94, 97])))
                self.assertEqual(s, pk.meta.signatures)

    def test_model_composition(self) -> None:
        arr = np.array([[1, 2, 3], [4, 2, 5]])
        d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
        aclf = AnotherDemoModel(custom_model.ModelContext())
        clf = DemoModel(custom_model.ModelContext())
        model_context = custom_model.ModelContext(
            models={
                "m1": aclf,
                "m2": clf,
            }
        )
        acm = ComposeModel(model_context)
        p1 = clf.predict(d)
        p2 = acm.predict(d)
        s = {"predict": model_signature.infer_signature(d, p2)}
        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=acm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, ComposeModel)
            p3 = pk.model.predict(d)

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, ComposeModel)
            p4 = pk.model.predict(d)
            np.testing.assert_allclose(p1, p2)
            np.testing.assert_allclose(p2, p3)
            np.testing.assert_allclose(p2, p4)

    def test_async_model_composition(self) -> None:
        async def _test(self: "CustomHandlerTest") -> None:
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            clf = DemoModel(custom_model.ModelContext())
            model_context = custom_model.ModelContext(
                models={
                    "m1": clf,
                    "m2": clf,
                }
            )
            acm = AsyncComposeModel(model_context)
            p1 = clf.predict(d)
            p2 = await acm.predict(d)
            s = {"predict": model_signature.infer_signature(d, p2)}
            with tempfile.TemporaryDirectory() as tmpdir:
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=acm,
                    signatures=s,
                    metadata={"author": "halu", "version": "1"},
                )
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, AsyncComposeModel)
                p3 = await pk.model.predict(d)  # type: ignore[misc]

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, AsyncComposeModel)
                p4 = await pk.model.predict(d)  # type: ignore[misc]
                np.testing.assert_allclose(p1, p2)
                np.testing.assert_allclose(p2, p3)
                np.testing.assert_allclose(p2, p4)

        asyncio.get_event_loop().run_until_complete(_test(self))

    def test_custom_model_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            arr = np.array([[1, 2, 3], [4, 2, 5]])
            d = pd.DataFrame(arr, columns=["c1", "c2", "c3"])
            s = {"predict": model_signature.infer_signature(d, lm.predict(d))}
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=lm,
                signatures=s,
                metadata={"author": "halu", "version": "1"},
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, DemoModelWithArtifacts)
            res = pk.model.predict(d)
            np.testing.assert_allclose(res["output"], pd.Series(np.array([11, 14])))

            # test re-init when loading the model
            with open(
                os.path.join(tmpdir, "model1", "models", "model1", "artifacts", "bias"), "w", encoding="utf-8"
            ) as f:
                f.write("20")

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load(as_custom_model=True)
            assert pk.model
            assert pk.meta
            assert isinstance(pk.model, DemoModelWithArtifacts)
            res = pk.model.predict(d)

            np.testing.assert_allclose(res["output"], pd.Series(np.array([21, 24])))
            self.assertEqual(pk.meta.metadata["author"] if pk.meta.metadata else None, "halu")


if __name__ == "__main__":
    absltest.main()
