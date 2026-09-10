import functools
import json
import os
import sys
import tempfile
from typing import Any, Optional, cast
from unittest import mock

import yaml
from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities
from snowflake.ml._internal.exceptions import error_codes
from snowflake.ml.model import model_signature, openai_signatures
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers import (
    peft_adapter as peft_adapter_handler,
)
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_meta_schema,
)
from snowflake.ml.model.models import huggingface as peft_adapter
from snowflake.ml.test_utils import exception_utils

_MODEL_NAME = "adapter1"
_BASE_FQN = "DB.SCHEMA.BASE"
_BASE_VERSION = "V1"
_CALLER_SIGNATURES_RE = r"signatures= is not supported"


def _enable_lora_adapters(fn: Any) -> Any:
    @mock.patch.object(
        platform_capabilities.PlatformCapabilities,
        "is_lora_adapters_enabled",
        return_value=True,
        autospec=True,
    )
    @functools.wraps(fn)
    def wrapped(self: Any, mock_enabled: mock.MagicMock, *args: Any, **kwargs: Any) -> Any:
        with platform_capabilities.PlatformCapabilities.mock_features():
            return fn(self, mock_enabled, *args, **kwargs)

    return wrapped


def _with_model_signature() -> model_signature.ModelSignature:
    return openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE["__call__"]


def _make_base() -> model_version_impl.ModelVersion:
    base = mock.create_autospec(model_version_impl.ModelVersion, instance=True)
    base.fully_qualified_model_name = _BASE_FQN
    base.version_name = _BASE_VERSION
    return cast(model_version_impl.ModelVersion, base)


def _peft_blob_options(
    meta: model_meta.ModelMetadata,
) -> model_meta_schema.PeftAdapterModelBlobOptions:
    return cast(model_meta_schema.PeftAdapterModelBlobOptions, meta.models[_MODEL_NAME].options)


def _write_adapter_dir(
    path: str,
    *,
    r: Any = 8,
    extra_vocab: Optional[int] = None,
    include_config: bool = True,
    include_weights: bool = True,
    omit_r: bool = False,
    extra_files: Optional[dict[str, Any]] = None,
    config_overrides: Optional[dict[str, Any]] = None,
) -> str:
    os.makedirs(path, exist_ok=True)
    if include_config:
        config: dict[str, Any] = {"peft_type": "LORA", "task_type": "CAUSAL_LM"}
        if not omit_r:
            config["r"] = r
        if extra_vocab is not None:
            config["lora_extra_vocab_size"] = extra_vocab
        if config_overrides:
            config.update(config_overrides)
        with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
    if include_weights:
        with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
            f.write(b"")
    if extra_files:
        for relpath, content in extra_files.items():
            full = os.path.join(path, relpath)
            parent = os.path.dirname(full)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if isinstance(content, bytes):
                with open(full, "wb") as f:
                    f.write(content)
            else:
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
    return path


def _blob_dir(model_dir: str) -> str:
    return os.path.join(model_dir, model_packager.ModelPackager.MODEL_BLOBS_DIR, _MODEL_NAME, "adapter")


def _model_yaml_path(model_dir: str) -> str:
    return os.path.join(model_dir, model_meta.MODEL_METADATA_FILE)


def _save(
    *,
    adapter: peft_adapter.PeftAdapter,
    model_dir: str,
    signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
    sample_input_data: Optional[Any] = None,
) -> model_meta.ModelMetadata:
    return model_packager.ModelPackager(model_dir).save(
        name=_MODEL_NAME,
        model=adapter,
        signatures=signatures,
        sample_input_data=sample_input_data,
        options={},
    )


class PeftAdapterHandlerTest(absltest.TestCase):
    def _assert_staged_artifact_has_no_signatures(self, model_dir: str) -> None:
        with open(_model_yaml_path(model_dir), encoding="utf-8") as f:
            model_yaml = yaml.safe_load(f)
        self.assertEqual(model_yaml.get("signatures"), {})
        loaded = model_meta.ModelMetadata.load(model_dir)
        self.assertEqual(loaded.signatures, {})

    @_enable_lora_adapters
    def test_can_handle_peft_adapter_not_raw_peft_model(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            self.assertTrue(peft_adapter_handler.PeftAdapterModelHandler.can_handle(wrapper))
            self.assertFalse(peft_adapter_handler.PeftAdapterModelHandler.can_handle(mock.Mock()))
            self.assertFalse(peft_adapter_handler.PeftAdapterModelHandler.can_handle({"not": "peft"}))

    @_enable_lora_adapters
    def test_required_blob_options_fqn_and_version(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                options = _peft_blob_options(meta)
                self.assertEqual(options["base_model_name"], _BASE_FQN)
                self.assertEqual(options["base_model_version"], _BASE_VERSION)
                self.assertEqual(options["peft_type"], "lora")
                self.assertEqual(meta.signatures, {})
                self._assert_staged_artifact_has_no_signatures(model_dir)

    @_enable_lora_adapters
    def test_peft_type_from_adapter_config(self, _mock_enabled: mock.MagicMock) -> None:
        cases: list[tuple[str, dict[str, Any], Optional[str]]] = [
            ("lora", {}, "lora"),
            ("uppercase_lora", {"config_overrides": {"peft_type": "LORA"}}, "lora"),
            ("ia3", {"config_overrides": {"peft_type": "IA3"}}, "ia3"),
            ("missing", {"config_overrides": {"peft_type": None}}, None),
            ("non_string", {"config_overrides": {"peft_type": 1}}, None),
        ]
        for name, kwargs, expected in cases:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as adapter_dir:
                    _write_adapter_dir(adapter_dir, **kwargs)
                    wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
                    with tempfile.TemporaryDirectory() as model_dir:
                        meta = _save(adapter=wrapper, model_dir=model_dir)
                        options = _peft_blob_options(meta)
                        if expected is None:
                            self.assertNotIn("peft_type", options)
                        else:
                            self.assertEqual(options["peft_type"], expected)

    @_enable_lora_adapters
    def test_lora_rank_from_positive_int_r(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir, r=16)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                self.assertEqual(_peft_blob_options(meta)["lora_rank"], 16)

    @_enable_lora_adapters
    def test_missing_r_omits_lora_rank(self, _mock_enabled: mock.MagicMock) -> None:
        cases: list[tuple[str, dict[str, Any]]] = [
            ("missing", {"omit_r": True}),
            ("non_int", {"r": "8"}),
            ("zero", {"r": 0}),
        ]
        for name, kwargs in cases:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as adapter_dir:
                    _write_adapter_dir(adapter_dir, **kwargs)
                    wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
                    with tempfile.TemporaryDirectory() as model_dir:
                        meta = _save(adapter=wrapper, model_dir=model_dir)
                        self.assertNotIn("lora_rank", _peft_blob_options(meta))

    @_enable_lora_adapters
    def test_lora_extra_vocab_size_copied_when_present(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir, extra_vocab=256)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                self.assertEqual(_peft_blob_options(meta)["lora_extra_vocab_size"], 256)

    @_enable_lora_adapters
    def test_lora_extra_vocab_size_omitted_when_absent(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                self.assertNotIn("lora_extra_vocab_size", _peft_blob_options(meta))

    def test_blob_options_round_trip_preserves_capacity_fields(self) -> None:
        # GS log_model(existing_mv) copy is G2; this round-trips packager blob options only.
        blob = model_blob_meta.ModelBlobMeta(
            name=_MODEL_NAME,
            model_type="peft_adapter",
            path="adapter",
            handler_version=peft_adapter_handler.PeftAdapterModelHandler.HANDLER_VERSION,
            options={
                "base_model_name": _BASE_FQN,
                "base_model_version": _BASE_VERSION,
                "peft_type": "lora",
                "lora_rank": 16,
                "lora_extra_vocab_size": 256,
            },
        )
        reloaded = model_blob_meta.ModelBlobMeta.from_dict(yaml.safe_load(yaml.safe_dump(blob.to_dict())))
        options = cast(model_meta_schema.PeftAdapterModelBlobOptions, reloaded.options)
        self.assertEqual(options["peft_type"], "lora")
        self.assertEqual(options["lora_rank"], 16)
        self.assertEqual(options["lora_extra_vocab_size"], 256)
        self.assertEqual(options["base_model_name"], _BASE_FQN)
        self.assertEqual(options["base_model_version"], _BASE_VERSION)

    @_enable_lora_adapters
    def test_rejects_disagreeing_caller_signatures(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            disagreeing = {
                "predict": model_signature.ModelSignature(
                    inputs=[model_signature.FeatureSpec(name="x", dtype=model_signature.DataType.FLOAT)],
                    outputs=[model_signature.FeatureSpec(name="y", dtype=model_signature.DataType.FLOAT)],
                )
            }
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=_CALLER_SIGNATURES_RE,
                ):
                    _save(adapter=wrapper, model_dir=model_dir, signatures=disagreeing)

    @_enable_lora_adapters
    def test_rejects_empty_caller_signatures(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=_CALLER_SIGNATURES_RE,
                ):
                    _save(adapter=wrapper, model_dir=model_dir, signatures={})

    @_enable_lora_adapters
    def test_rejects_matching_caller_signatures(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            expected = _with_model_signature()
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=_CALLER_SIGNATURES_RE,
                ):
                    _save(adapter=wrapper, model_dir=model_dir, signatures={"__call__": expected})

    @_enable_lora_adapters
    def test_rejects_sample_input_data(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=r"sample_input_data is not supported",
                ):
                    _save(adapter=wrapper, model_dir=model_dir, sample_input_data=[{"x": 1}])

    def test_flag_off_refuses_save(self) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir)
            with mock.patch.object(
                platform_capabilities.PlatformCapabilities,
                "is_lora_adapters_enabled",
                return_value=True,
                autospec=True,
            ):
                with platform_capabilities.PlatformCapabilities.mock_features():
                    wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with platform_capabilities.PlatformCapabilities.mock_features():
                with tempfile.TemporaryDirectory() as model_dir:
                    with exception_utils.assert_snowml_exceptions(
                        self,
                        expected_error_code=error_codes.INVALID_ARGUMENT,
                        expected_original_error_type=ValueError,
                        expected_regex=r"ENABLE_LORA_ADAPTERS",
                    ):
                        _save(adapter=wrapper, model_dir=model_dir)

    @_enable_lora_adapters
    def test_requires_adapter_config(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir, include_config=False)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=r"adapter_config.json.*subfolder=",
                ):
                    _save(adapter=wrapper, model_dir=model_dir)

    @_enable_lora_adapters
    def test_missing_weights_still_logs(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir, include_weights=False)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                dest = _blob_dir(model_dir)
                copied = set(os.listdir(dest))
                self.assertIn("adapter_config.json", copied)
                self.assertNotIn("adapter_model.safetensors", copied)
                self.assertEqual(_peft_blob_options(meta)["lora_rank"], 8)

    @_enable_lora_adapters
    def test_skips_file_symlinks_in_adapter_dir(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as root:
            adapter_dir = os.path.join(root, "adapter")
            _write_adapter_dir(adapter_dir, include_weights=False)
            secret_path = os.path.join(root, "secret.txt")
            with open(secret_path, "w", encoding="utf-8") as f:
                f.write("exfiltrate-me")
            link_path = os.path.join(adapter_dir, "adapter_model.safetensors")
            try:
                os.symlink(secret_path, link_path)
            except OSError:
                self.skipTest("symlinks not supported on this platform")
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                _save(adapter=wrapper, model_dir=model_dir)
                dest = _blob_dir(model_dir)
                copied = set(os.listdir(dest))
                self.assertIn("adapter_config.json", copied)
                self.assertNotIn("adapter_model.safetensors", copied)

    @_enable_lora_adapters
    def test_rejects_symlink_adapter_config(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as root:
            adapter_dir = os.path.join(root, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            outside_config = os.path.join(root, "outside_config.json")
            with open(outside_config, "w", encoding="utf-8") as f:
                json.dump({"peft_type": "LORA", "r": 8}, f)
            link_path = os.path.join(adapter_dir, "adapter_config.json")
            try:
                os.symlink(outside_config, link_path)
            except OSError:
                self.skipTest("symlinks not supported on this platform")
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=r"adapter_config.json.*subfolder=",
                ):
                    _save(adapter=wrapper, model_dir=model_dir)

    @_enable_lora_adapters
    def test_copies_all_top_level_files_and_skips_dirs(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(
                adapter_dir,
                extra_files={
                    "training_args.bin": b"junk",
                    "trainer_state.json": "{}",
                    "optimizer.pt": b"opt",
                    "scheduler.pt": b"sched",
                    "rng_state.pth": b"rng",
                    "scaler.pt": b"scaler",
                    "pytorch_model.bin": b"base",
                    "config.json": "{}",
                    "tokenizer.json": "{}",
                    "tokenizer_config.json": "{}",
                    "README.md": "adapter",
                    "new_embeddings.safetensors": b"emb",
                    "unknown.dat": b"nope",
                },
            )
            _write_adapter_dir(os.path.join(adapter_dir, "alpha"), r=8)
            wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=adapter_dir)
            with tempfile.TemporaryDirectory() as model_dir:
                _save(adapter=wrapper, model_dir=model_dir)
                dest = _blob_dir(model_dir)
                copied = set(os.listdir(dest))
                self.assertIn("adapter_config.json", copied)
                self.assertIn("adapter_model.safetensors", copied)
                self.assertIn("training_args.bin", copied)
                self.assertIn("trainer_state.json", copied)
                self.assertIn("optimizer.pt", copied)
                self.assertIn("pytorch_model.bin", copied)
                self.assertIn("config.json", copied)
                self.assertIn("tokenizer.json", copied)
                self.assertIn("new_embeddings.safetensors", copied)
                self.assertIn("unknown.dat", copied)
                self.assertNotIn("alpha", copied)

    @_enable_lora_adapters
    def test_stage_and_snow_url_adapter_path_not_supported(self, _mock_enabled: mock.MagicMock) -> None:
        for path in ("@db.schema.stage/adapter", "snow://model/DB.SCHEMA.ADAPTER/versions/V1"):
            with self.subTest(path=path):
                wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_path=path)
                with tempfile.TemporaryDirectory() as model_dir:
                    with exception_utils.assert_snowml_exceptions(
                        self,
                        expected_error_code=error_codes.INVALID_ARGUMENT,
                        expected_original_error_type=ValueError,
                        expected_regex=r"adapter_path starting with '@' or 'snow://'",
                    ):
                        _save(adapter=wrapper, model_dir=model_dir)

    @_enable_lora_adapters
    def test_adapter_repo_downloads_all_top_level_files(self, _mock_enabled: mock.MagicMock) -> None:
        mock_hub = mock.MagicMock()

        def _fake_download(
            *,
            repo_id: str,
            revision: Optional[str],
            token: Optional[str],
            local_dir: str,
            **kwargs: Any,
        ) -> str:
            _write_adapter_dir(
                local_dir,
                r=4,
                extra_files={"training_args.bin": b"junk", "README.md": "hub"},
            )
            return local_dir

        mock_hub.snapshot_download.side_effect = _fake_download
        wrapper = peft_adapter.PeftAdapter(
            base_model=_make_base(),
            adapter_repo="org/adapter",
            revision="abc",
            token="tok",
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": mock_hub}):
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                mock_hub.snapshot_download.assert_called_once()
                called = mock_hub.snapshot_download.call_args
                self.assertEqual(called.kwargs["repo_id"], "org/adapter")
                self.assertEqual(called.kwargs["revision"], "abc")
                self.assertEqual(called.kwargs["token"], "tok")
                self.assertIsNone(called.kwargs["allow_patterns"])
                dest = _blob_dir(model_dir)
                copied = set(os.listdir(dest))
                self.assertIn("adapter_config.json", copied)
                self.assertIn("adapter_model.safetensors", copied)
                self.assertIn("README.md", copied)
                self.assertIn("training_args.bin", copied)
                self.assertEqual(_peft_blob_options(meta)["lora_rank"], 4)

    @_enable_lora_adapters
    def test_subfolder_with_adapter_path_joins_child(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter_dir(adapter_dir, extra_files={"parent_only.txt": "skip"})
            _write_adapter_dir(
                os.path.join(adapter_dir, "query_rewrite_lora"),
                r=16,
                extra_files={"training_args.bin": b"junk"},
            )
            _write_adapter_dir(os.path.join(adapter_dir, "alpha"), r=8)
            wrapper = peft_adapter.PeftAdapter(
                base_model=_make_base(),
                adapter_path=adapter_dir,
                subfolder="query_rewrite_lora",
            )
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                dest = _blob_dir(model_dir)
                copied = set(os.listdir(dest))
                self.assertIn("adapter_config.json", copied)
                self.assertIn("adapter_model.safetensors", copied)
                self.assertIn("training_args.bin", copied)
                self.assertNotIn("parent_only.txt", copied)
                self.assertNotIn("alpha", copied)
                self.assertEqual(_peft_blob_options(meta)["lora_rank"], 16)

    @_enable_lora_adapters
    def test_subfolder_with_adapter_repo_uses_allow_patterns(self, _mock_enabled: mock.MagicMock) -> None:
        mock_hub = mock.MagicMock()
        nested = "loras/style_v1"

        def _fake_download(
            *,
            repo_id: str,
            revision: Optional[str],
            token: Optional[str],
            local_dir: str,
            **kwargs: Any,
        ) -> str:
            _write_adapter_dir(
                os.path.join(local_dir, nested),
                r=4,
                extra_files={"training_args.bin": b"junk"},
            )
            with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write("repo")
            return local_dir

        mock_hub.snapshot_download.side_effect = _fake_download
        wrapper = peft_adapter.PeftAdapter(
            base_model=_make_base(),
            adapter_repo="org/adapter-lib",
            subfolder=nested,
            revision="abc",
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": mock_hub}):
            with tempfile.TemporaryDirectory() as model_dir:
                meta = _save(adapter=wrapper, model_dir=model_dir)
                called = mock_hub.snapshot_download.call_args
                self.assertEqual(called.kwargs["allow_patterns"], f"{nested}/*")
                dest = _blob_dir(model_dir)
                copied = set(os.listdir(dest))
                self.assertIn("adapter_config.json", copied)
                self.assertIn("adapter_model.safetensors", copied)
                self.assertIn("training_args.bin", copied)
                self.assertNotIn("README.md", copied)
                self.assertEqual(_peft_blob_options(meta)["lora_rank"], 4)

    @_enable_lora_adapters
    def test_missing_config_at_subfolder_names_subfolder(self, _mock_enabled: mock.MagicMock) -> None:
        with tempfile.TemporaryDirectory() as adapter_dir:
            os.makedirs(os.path.join(adapter_dir, "empty_child"))
            wrapper = peft_adapter.PeftAdapter(
                base_model=_make_base(),
                adapter_path=adapter_dir,
                subfolder="empty_child",
            )
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=r"adapter_config.json.*subfolder=",
                ):
                    _save(adapter=wrapper, model_dir=model_dir)

    @_enable_lora_adapters
    def test_nested_hub_layout_without_subfolder_fails_missing_config(self, _mock_enabled: mock.MagicMock) -> None:
        mock_hub = mock.MagicMock()

        def _fake_download(
            *,
            repo_id: str,
            revision: Optional[str],
            token: Optional[str],
            local_dir: str,
            **kwargs: Any,
        ) -> str:
            _write_adapter_dir(os.path.join(local_dir, "query_rewrite_lora"))
            return local_dir

        mock_hub.snapshot_download.side_effect = _fake_download
        wrapper = peft_adapter.PeftAdapter(base_model=_make_base(), adapter_repo="org/adapter-lib")
        with mock.patch.dict(sys.modules, {"huggingface_hub": mock_hub}):
            with tempfile.TemporaryDirectory() as model_dir:
                with exception_utils.assert_snowml_exceptions(
                    self,
                    expected_error_code=error_codes.INVALID_ARGUMENT,
                    expected_original_error_type=ValueError,
                    expected_regex=r"adapter_config.json.*subfolder=",
                ):
                    _save(adapter=wrapper, model_dir=model_dir)

    def test_load_and_convert_rejected(self) -> None:
        dummy_meta = mock.create_autospec(model_meta.ModelMetadata, instance=True)
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"cannot be loaded",
        ):
            peft_adapter_handler.PeftAdapterModelHandler.load_model(
                name=_MODEL_NAME,
                model_meta=dummy_meta,
                model_blobs_dir_path="/tmp",
            )
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_error_code=error_codes.INVALID_ARGUMENT,
            expected_original_error_type=ValueError,
            expected_regex=r"cannot be converted",
        ):
            peft_adapter_handler.PeftAdapterModelHandler.convert_as_custom_model(
                raw_model=mock.Mock(),
                model_meta=dummy_meta,
            )


if __name__ == "__main__":
    absltest.main()
