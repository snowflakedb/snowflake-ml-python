import os
import pathlib
from unittest import mock

from absl.testing import absltest

from snowflake.ml.model._model_composer import huggingface_lazy_uploader


class HuggingFaceLazyUploaderTest(absltest.TestCase):
    @mock.patch("huggingface_hub.utils.get_token", return_value=None)
    @mock.patch("snowflake.ml._internal.file_utils.upload_file_to_stage")
    @mock.patch("huggingface_hub.hf_hub_download")
    def test_stream_upload_puts_each_file(
        self,
        mock_hf_hub_download: mock.Mock,
        mock_upload_file_to_stage: mock.Mock,
        mock_get_token: mock.Mock,
    ) -> None:
        """Each HuggingFace file should be downloaded, uploaded, and removed locally."""

        def _fake_download(*, filename: str, local_dir: str, **kwargs: object) -> str:
            local_path = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as file:
                file.write("data")
            return local_path

        mock_hf_hub_download.side_effect = _fake_download

        lazy = huggingface_lazy_uploader.LazyHFUpload(
            download_kwargs={"repo_id": "org/model", "revision": None},
            files=["config.json", "weights/model.safetensors"],
            file_sizes={"config.json": 100, "weights/model.safetensors": 200},
            relative_stage_dir=pathlib.PurePosixPath("model", "models", "my_model", "model"),
        )
        mock_session = mock.Mock()

        huggingface_lazy_uploader.stream_upload(
            mock_session,
            pathlib.PurePosixPath("@stage/path"),
            lazy,
            max_workers=2,
        )

        self.assertEqual(mock_hf_hub_download.call_count, 2)
        self.assertEqual(mock_upload_file_to_stage.call_count, 2)
        uploaded_local_paths = [call.args[1] for call in mock_upload_file_to_stage.call_args_list]
        for local_path in uploaded_local_paths:
            self.assertFalse(os.path.exists(local_path))

        stage_dirs = {call.args[2] for call in mock_upload_file_to_stage.call_args_list}
        self.assertEqual(
            stage_dirs,
            {
                "@stage/path/model/models/my_model/model",
                "@stage/path/model/models/my_model/model/weights",
            },
        )

    @mock.patch("snowflake.ml.model._model_composer.huggingface_lazy_uploader._lazy_upload_temp_root")
    @mock.patch("shutil.disk_usage")
    def test_create_disk_budget_raises_when_largest_file_exceeds_available_space(
        self,
        mock_disk_usage: mock.Mock,
        mock_temp_root: mock.Mock,
    ) -> None:
        mock_temp_root.return_value = pathlib.Path("/tmp")
        mock_disk_usage.return_value = mock.Mock(free=1000, total=1000, used=0)

        with self.assertRaises(ValueError) as error_context:
            huggingface_lazy_uploader.DiskBudget.from_files(
                {"config.json": 100, "weights/model.safetensors": 1800},
                ["config.json", "weights/model.safetensors"],
            )
        self.assertEqual(
            str(error_context.exception),
            "model upload: insufficient disk space to download HuggingFace model files. "
            "The largest file (weights/model.safetensors) requires 1.76 KiB but only 900 B is available.",
        )
        mock_disk_usage.assert_called_once_with(pathlib.Path("/tmp"))

    def test_disk_budget_reserve_releases_on_exit(self) -> None:
        budget = huggingface_lazy_uploader.DiskBudget(100)
        with budget.reserve(80):
            pass
        with budget.reserve(100):
            pass

    def test_disk_budget_reserve_releases_on_exception(self) -> None:
        budget = huggingface_lazy_uploader.DiskBudget(100)
        with self.assertRaises(RuntimeError):
            with budget.reserve(80):
                raise RuntimeError("upload failed")
        with budget.reserve(100):
            pass

    def test_disk_budget_acquire_raises_on_timeout(self) -> None:
        budget = huggingface_lazy_uploader.DiskBudget(100, acquire_timeout_seconds=0.1)
        with budget.reserve(80):
            with self.assertRaises(TimeoutError) as error_context:
                budget.acquire(50)
        self.assertEqual(
            str(error_context.exception),
            "model upload: timed out waiting for disk space to download HuggingFace model files. "
            "Required 50 B but only 20 B was available.",
        )

    @mock.patch("huggingface_hub.utils.get_token", return_value=None)
    @mock.patch("snowflake.ml._internal.file_utils.upload_file_to_stage")
    @mock.patch("huggingface_hub.hf_hub_download")
    @mock.patch("shutil.disk_usage")
    def test_stream_upload_throttles_concurrent_large_downloads(
        self,
        mock_disk_usage: mock.Mock,
        mock_hf_hub_download: mock.Mock,
        mock_upload_file_to_stage: mock.Mock,
        mock_get_token: mock.Mock,
    ) -> None:
        """Workers should wait for disk budget before downloading large files."""
        mock_disk_usage.return_value = mock.Mock(free=300, total=300, used=0)
        active_downloads = 0
        max_active_downloads = 0

        def _fake_download(*, filename: str, local_dir: str, **kwargs: object) -> str:
            nonlocal active_downloads, max_active_downloads
            active_downloads += 1
            max_active_downloads = max(max_active_downloads, active_downloads)
            local_path = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as file:
                file.write("data")
            active_downloads -= 1
            return local_path

        mock_hf_hub_download.side_effect = _fake_download

        lazy = huggingface_lazy_uploader.LazyHFUpload(
            download_kwargs={"repo_id": "org/model", "revision": None},
            files=["shard-0.bin", "shard-1.bin", "shard-2.bin"],
            file_sizes={"shard-0.bin": 150, "shard-1.bin": 150, "shard-2.bin": 150},
            relative_stage_dir=pathlib.PurePosixPath("model"),
        )

        huggingface_lazy_uploader.stream_upload(
            mock.Mock(),
            pathlib.PurePosixPath("@stage/path"),
            lazy,
            max_workers=3,
        )

        self.assertEqual(mock_hf_hub_download.call_count, 3)
        self.assertEqual(max_active_downloads, 1)

    def test_stream_upload_rejects_path_traversal(self) -> None:
        lazy = huggingface_lazy_uploader.LazyHFUpload(
            download_kwargs={"repo_id": "org/model", "revision": None},
            files=["../../functions/__call__.py"],
            file_sizes={"../../functions/__call__.py": 100},
            relative_stage_dir=pathlib.PurePosixPath("model"),
        )
        with self.assertRaises(ValueError) as error_context:
            huggingface_lazy_uploader.stream_upload(
                mock.Mock(),
                pathlib.PurePosixPath("@stage/path"),
                lazy,
            )
        self.assertEqual(
            str(error_context.exception),
            "model upload: invalid HuggingFace repository file path. "
            "Path must not contain parent-directory segments: '../../functions/__call__.py'.",
        )

    @mock.patch("huggingface_hub.utils.get_token", return_value=None)
    @mock.patch("snowflake.ml._internal.file_utils.upload_file_to_stage")
    @mock.patch("huggingface_hub.hf_hub_download")
    def test_stream_upload_uses_download_token_from_lazy_metadata(
        self,
        mock_hf_hub_download: mock.Mock,
        mock_upload_file_to_stage: mock.Mock,
        mock_get_token: mock.Mock,
    ) -> None:
        local_path = "/tmp/config.json"
        mock_hf_hub_download.return_value = local_path

        lazy = huggingface_lazy_uploader.LazyHFUpload(
            download_kwargs={"repo_id": "org/model", "revision": None},
            files=["config.json"],
            file_sizes={"config.json": 100},
            relative_stage_dir=pathlib.PurePosixPath("model"),
            download_token="hf_test_token",
        )

        huggingface_lazy_uploader.stream_upload(
            mock.Mock(),
            pathlib.PurePosixPath("@stage/path"),
            lazy,
        )

        mock_hf_hub_download.assert_called_once_with(
            filename="config.json",
            local_dir=mock.ANY,
            token="hf_test_token",
            repo_id="org/model",
            revision=None,
        )
        mock_upload_file_to_stage.assert_called_once()


if __name__ == "__main__":
    absltest.main()
