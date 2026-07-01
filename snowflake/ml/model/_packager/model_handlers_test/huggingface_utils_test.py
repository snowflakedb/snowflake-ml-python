from unittest import mock

from absl.testing import absltest

from snowflake.ml.model._packager.model_handlers.huggingface import _utils as hf_utils


class HuggingFaceUtilsTest(absltest.TestCase):
    def test_download_token_for_lazy_upload_returns_plain_token(self) -> None:
        model = mock.Mock()
        model.secret_identifier = None
        model.token_or_secret = "hf_test_token"

        self.assertEqual(
            hf_utils.download_token_for_lazy_upload(model),
            "hf_test_token",
        )

    def test_download_token_for_lazy_upload_returns_none_without_token(self) -> None:
        model = mock.Mock()
        model.secret_identifier = None
        model.token_or_secret = None

        self.assertIsNone(hf_utils.download_token_for_lazy_upload(model))

    def test_download_token_for_lazy_upload_raises_for_secret(self) -> None:
        model = mock.Mock()
        model.secret_identifier = "db.schema.secret"
        model.token_or_secret = "db.schema.secret"

        with self.assertRaises(ValueError) as error_context:
            hf_utils.download_token_for_lazy_upload(model)
        self.assertEqual(
            str(error_context.exception),
            "model upload: HuggingFace lazy upload cannot resolve auth from a Snowflake secret during local logging. "
            "Set the HF_TOKEN environment variable or pass a HuggingFace token when constructing the model.",
        )


if __name__ == "__main__":
    absltest.main()
