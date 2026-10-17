import functools
from typing import Any, cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities, type_utils
from snowflake.ml._internal.exceptions import error_codes
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model.models import huggingface as peft
from snowflake.ml.test_utils import exception_utils


def _enable_lora_adapters(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with platform_capabilities.PlatformCapabilities.mock_features(
            {platform_capabilities.ENABLE_LORA_ADAPTERS: True}
        ):
            return fn(self, *args, **kwargs)

    return wrapped


def _base_model() -> model_version_impl.ModelVersion:
    return cast(
        model_version_impl.ModelVersion,
        mock.create_autospec(model_version_impl.ModelVersion, instance=True),
    )


class PeftAdapterTest(absltest.TestCase):
    def test_flag_off_refuses_init(self) -> None:
        with platform_capabilities.PlatformCapabilities.mock_features():
            with exception_utils.assert_snowml_exceptions(
                self,
                expected_error_code=error_codes.INVALID_ARGUMENT,
                expected_original_error_type=ValueError,
                expected_regex=r"ENABLE_LORA_ADAPTERS",
            ):
                peft.PeftAdapter(base_model=_base_model(), adapter_path="/tmp/adapter")

    @_enable_lora_adapters
    def test_rejects_zero_sources(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"Exactly one of adapter_path, adapter, or adapter_repo",
        ):
            peft.PeftAdapter(base_model=_base_model())

    @_enable_lora_adapters
    def test_rejects_two_sources(self) -> None:
        cases: list[dict[str, Any]] = [
            {"adapter_path": "/tmp/a", "adapter_repo": "org/a"},
            {"adapter_path": "/tmp/a", "adapter": object()},
            {"adapter": object(), "adapter_repo": "org/a"},
            {"adapter_path": "/tmp/a", "adapter": object(), "adapter_repo": "org/a"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=sorted(kwargs.keys())):
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=r"Exactly one of adapter_path, adapter, or adapter_repo",
                ):
                    peft.PeftAdapter(base_model=_base_model(), **kwargs)

    @_enable_lora_adapters
    def test_accepts_each_single_source(self) -> None:
        base = _base_model()
        from_path = peft.PeftAdapter(base_model=base, adapter_path="/tmp/adapter")
        self.assertEqual(from_path.adapter_path, "/tmp/adapter")
        self.assertIsNone(from_path.adapter)
        self.assertIsNone(from_path.adapter_repo)
        self.assertIsNone(from_path.subfolder)

        from_repo = peft.PeftAdapter(base_model=base, adapter_repo="org/adapter", revision="main")
        self.assertEqual(from_repo.adapter_repo, "org/adapter")
        self.assertEqual(from_repo.revision, "main")
        self.assertIsNone(from_repo.adapter_path)
        self.assertIsNone(from_repo.subfolder)

        fake_peft = mock.Mock()
        with mock.patch.object(type_utils.LazyType, "isinstance", return_value=True, autospec=True):
            from_adapter = peft.PeftAdapter(base_model=base, adapter=fake_peft)
        self.assertIs(from_adapter.adapter, fake_peft)
        self.assertIsNone(from_adapter.adapter_path)
        self.assertIsNone(from_adapter.adapter_repo)
        self.assertIsNone(from_adapter.subfolder)

    @_enable_lora_adapters
    def test_revision_requires_adapter_repo(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"revision is only valid when adapter_repo",
        ):
            peft.PeftAdapter(base_model=_base_model(), adapter_path="/tmp/a", revision="abc")

        with mock.patch.object(type_utils.LazyType, "isinstance", return_value=True, autospec=True):
            with exception_utils.assert_snowml_exceptions(
                self,
                expected_error_code=error_codes.INVALID_ARGUMENT,
                expected_original_error_type=ValueError,
                expected_regex=r"revision is only valid when adapter_repo",
            ):
                peft.PeftAdapter(base_model=_base_model(), adapter=mock.Mock(), revision="abc")

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"Exactly one of adapter_path, adapter, or adapter_repo|revision is only valid",
        ):
            peft.PeftAdapter(base_model=_base_model(), revision="abc")

    @_enable_lora_adapters
    def test_subfolder_accepted_with_path_or_repo(self) -> None:
        from_path = peft.PeftAdapter(
            base_model=_base_model(),
            adapter_path="/tmp/adapter",
            subfolder="loras/style_v1",
        )
        self.assertEqual(from_path.subfolder, "loras/style_v1")

        from_repo = peft.PeftAdapter(
            base_model=_base_model(),
            adapter_repo="org/adapter-lib",
            subfolder="query_rewrite_lora",
        )
        self.assertEqual(from_repo.subfolder, "query_rewrite_lora")

    @_enable_lora_adapters
    def test_subfolder_rejected_with_adapter(self) -> None:
        with mock.patch.object(type_utils.LazyType, "isinstance", return_value=True, autospec=True):
            with exception_utils.assert_snowml_exceptions(
                self,
                expected_error_code=error_codes.INVALID_ARGUMENT,
                expected_original_error_type=ValueError,
                expected_regex=r"subfolder is only valid with adapter_path or adapter_repo",
            ):
                peft.PeftAdapter(base_model=_base_model(), adapter=mock.Mock(), subfolder="alpha")

    @_enable_lora_adapters
    def test_adapter_must_be_peft_model(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=TypeError,
            expected_regex=r"peft.PeftModel",
        ):
            peft.PeftAdapter(base_model=_base_model(), adapter={"not": "peft"})

    @_enable_lora_adapters
    def test_base_model_must_be_model_version(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=TypeError,
            expected_regex=r"base_model must be a ModelVersion",
        ):
            peft.PeftAdapter(base_model="not-a-version", adapter_path="/tmp/a")  # type: ignore[arg-type]


if __name__ == "__main__":
    absltest.main()
