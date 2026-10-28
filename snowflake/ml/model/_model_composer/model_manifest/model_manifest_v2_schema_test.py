from absl.testing import absltest

from snowflake.ml.model._model_composer.model_manifest import model_manifest_v2_schema


class ModelManifestV2SchemaTest(absltest.TestCase):
    def test_version_and_default_path_constants(self) -> None:
        self.assertEqual(model_manifest_v2_schema.MODEL_MANIFEST_VERSION_V2, "2.0")
        self.assertEqual(model_manifest_v2_schema.DEFAULT_MODEL_EXTENSION_SPEC_PATH, "model/model.yaml")


if __name__ == "__main__":
    absltest.main()
