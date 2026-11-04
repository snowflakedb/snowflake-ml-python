from typing import Any
from unittest import mock

from absl.testing import absltest

from snowflake.ml.model._client.model import model_version_impl
from tests.integ.snowflake.ml.test_utils.embed_local_ml_library import (
    log_model_embedding_local_ml_library,
)


class _StubModelVersion(model_version_impl.ModelVersion):
    def __init__(self) -> None:
        pass


class LogModelEmbeddingLocalMlLibraryTest(absltest.TestCase):
    def test_defaults_embed_local_ml_library(self) -> None:
        original = mock.Mock(return_value="logged")
        wrapped = log_model_embedding_local_ml_library(original)
        caller_options: dict[str, Any] = {"enable_explainability": False}

        result = wrapped(mock.Mock(), object(), model_name="m", options=caller_options)

        self.assertEqual(result, "logged")
        _, kwargs = original.call_args
        self.assertEqual(
            kwargs["options"],
            {"enable_explainability": False, "embed_local_ml_library": True},
        )
        self.assertNotIn("embed_local_ml_library", caller_options)

    def test_defaults_embed_when_options_omitted(self) -> None:
        original = mock.Mock()
        wrapped = log_model_embedding_local_ml_library(original)

        wrapped(mock.Mock(), object(), model_name="m")

        _, kwargs = original.call_args
        self.assertEqual(kwargs["options"], {"embed_local_ml_library": True})

    def test_preserves_explicit_false(self) -> None:
        original = mock.Mock()
        wrapped = log_model_embedding_local_ml_library(original)

        wrapped(mock.Mock(), object(), options={"embed_local_ml_library": False, "relax_version": False})

        _, kwargs = original.call_args
        self.assertEqual(
            kwargs["options"],
            {"embed_local_ml_library": False, "relax_version": False},
        )

    def test_leaves_model_version_copy_unchanged(self) -> None:
        original = mock.Mock()
        wrapped = log_model_embedding_local_ml_library(original)

        wrapped(mock.Mock(), _StubModelVersion(), model_name="m", version_name="v")

        _, kwargs = original.call_args
        self.assertNotIn("options", kwargs)
        self.assertEqual(kwargs["model_name"], "m")
        self.assertEqual(kwargs["version_name"], "v")


if __name__ == "__main__":
    absltest.main()
