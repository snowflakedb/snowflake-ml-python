import tempfile

from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
)

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}


class HandlerMigrator_1(base_migrator.BaseModelHandlerMigrator):
    source_version = "version_0"
    target_version = "version_1"

    @staticmethod
    def upgrade(name: str, model_meta: model_meta_api.ModelMetadata, model_blobs_dir_path: str) -> None:
        model_meta.models[name].path = "changed_path"


class HandlerMigrator_2(base_migrator.BaseModelHandlerMigrator):
    source_version = "version_1"
    target_version = "version_2"

    @staticmethod
    def upgrade(name: str, model_meta: model_meta_api.ModelMetadata, model_blobs_dir_path: str) -> None:
        raise base_migrator.UnableToUpgradeError(last_supported_version="1.0.9")


class BaseMigratorTest(absltest.TestCase):
    def test_model_meta_dependencies_no_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with model_meta_api.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = model_blob_meta.ModelBlobMeta(
                    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
                )

                migrator_1 = HandlerMigrator_1()
                migrator_1.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)

                self.assertEqual(meta.models["model1"].path, "changed_path")

                migrator_2 = HandlerMigrator_2()
                with self.assertRaisesRegex(
                    RuntimeError,
                    (
                        "Can not upgrade your model model1 from version version_1 to version_2."
                        "The latest version support the original version of Snowpark ML library is 1.0.9."
                    ),
                ):
                    migrator_2.try_upgrade(name="model1", model_meta=meta, model_blobs_dir_path=tmpdir)


if __name__ == "__main__":
    absltest.main()
