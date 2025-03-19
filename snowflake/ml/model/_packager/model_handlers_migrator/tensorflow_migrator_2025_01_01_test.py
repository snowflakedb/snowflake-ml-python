import tempfile
from typing import cast

from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers_migrator import (
    tensorflow_migrator_2025_01_01,
)
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}


class TensorflowHandlerMigrator20250101Test(absltest.TestCase):
    def test_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2025-01-01",
                )

                migrator_1 = tensorflow_migrator_2025_01_01.TensorflowHandlerMigrator20250101()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                model_blob_meta_options = cast(
                    model_meta_schema.TensorflowModelBlobOptions, meta.models["model1"].options
                )
                self.assertEqual(model_blob_meta_options["multiple_inputs"], True)


if __name__ == "__main__":
    absltest.main()
