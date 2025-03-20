import tempfile
from typing import cast

from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers_migrator import (
    tensorflow_migrator_2023_12_01,
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
    def test_keras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2023-12-01",
                )

                migrator_1 = tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                model_blob_meta_options = cast(
                    model_meta_schema.TensorflowModelBlobOptions, meta.models["model1"].options
                )
                self.assertEqual(model_blob_meta_options["save_format"], "keras_tf")

        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2023-12-01",
                    options={"is_keras_model": True},  # type: ignore[arg-type]
                )

                migrator_1 = tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                model_blob_meta_options = cast(
                    model_meta_schema.TensorflowModelBlobOptions, meta.models["model1"].options
                )
                self.assertEqual(model_blob_meta_options["save_format"], "keras_tf")

        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2023-12-01",
                    options={"save_format": "keras"},  # type: ignore[arg-type]
                )

                migrator_1 = tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                model_blob_meta_options = cast(
                    model_meta_schema.TensorflowModelBlobOptions, meta.models["model1"].options
                )
                self.assertEqual(model_blob_meta_options["save_format"], "keras_tf")

        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2023-12-01",
                    options={"save_format": "keras_tf"},  # type: ignore[arg-type]
                )

                migrator_1 = tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                model_blob_meta_options = cast(
                    model_meta_schema.TensorflowModelBlobOptions, meta.models["model1"].options
                )
                self.assertEqual(model_blob_meta_options["save_format"], "keras_tf")

        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2023-12-01",
                    options={"save_format": "tf"},  # type: ignore[arg-type]
                )

                migrator_1 = tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                model_blob_meta_options = cast(
                    model_meta_schema.TensorflowModelBlobOptions, meta.models["model1"].options
                )
                self.assertEqual(model_blob_meta_options["save_format"], "tf")

        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1",
                    model_type="custom",
                    path="mock_path",
                    handler_version="2023-12-01",
                    options={"save_format": "cloudpickle"},  # type: ignore[arg-type]
                )

                migrator_1 = tensorflow_migrator_2023_12_01.TensorflowHandlerMigrator20231201()
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "Unable to upgrade keras 3.x model saved by old handler. This is not supposed to happen",
                ):
                    migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)


if __name__ == "__main__":
    absltest.main()
