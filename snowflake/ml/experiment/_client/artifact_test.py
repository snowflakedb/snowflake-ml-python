import os
import tempfile

from absl.testing import absltest

from snowflake.ml.experiment._client import artifact


class GetPutPathPairsTest(absltest.TestCase):
    def test_nonexistent_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            artifact.get_put_path_pairs("/path/does/not/exist", "dest")

    def test_single_file_with_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "file.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("content")

            pairs = artifact.get_put_path_pairs(file_path, "base/path")
            self.assertEqual(pairs, [(file_path, "base/path")])

    def test_single_file_empty_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "model.bin")
            with open(file_path, "wb") as f:
                f.write(b"binary")

            pairs = artifact.get_put_path_pairs(file_path, "")
            self.assertEqual(pairs, [(file_path, "")])

    def test_directory_with_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested files
            os.makedirs(os.path.join(tmpdir, "sub", "deeper"))
            top_file = os.path.join(tmpdir, "a.txt")
            sub_file = os.path.join(tmpdir, "sub", "b.bin")
            deeper_file = os.path.join(tmpdir, "sub", "deeper", "c.md")
            for p in (top_file, sub_file, deeper_file):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("x")

            pairs = artifact.get_put_path_pairs(tmpdir, "base")

            # Build mapping from relative file path to destination artifact subdir
            rel_to_dest = {os.path.relpath(fp, tmpdir).replace(os.sep, "/"): dest for fp, dest in pairs}
            self.assertEqual(
                rel_to_dest,
                {
                    "a.txt": "base",
                    "sub/b.bin": "base/sub",
                    "sub/deeper/c.md": "base/sub/deeper",
                },
            )

    def test_directory_without_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "x", "y"))
            f1 = os.path.join(tmpdir, "root.txt")
            f2 = os.path.join(tmpdir, "x", "child.dat")
            f3 = os.path.join(tmpdir, "x", "y", "leaf.log")
            for p in (f1, f2, f3):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("data")

            pairs = artifact.get_put_path_pairs(tmpdir, "")
            rel_to_dest = {os.path.relpath(fp, tmpdir).replace(os.sep, "/"): dest for fp, dest in pairs}
            self.assertEqual(
                rel_to_dest,
                {
                    "root.txt": "",
                    "x/child.dat": "x",
                    "x/y/leaf.log": "x/y",
                },
            )


class GetDownloadPathPairsTest(absltest.TestCase):
    def test_download_paths_with_base_dir(self) -> None:
        artifacts = [
            artifact.ArtifactInfo(name="a.txt", size=1, md5="aaa", last_modified="2024-01-01"),
            artifact.ArtifactInfo(name="sub/b.bin", size=2, md5="bbb", last_modified="2024-01-01"),
            artifact.ArtifactInfo(name="sub/deeper/c.md", size=3, md5="ccc", last_modified="2024-01-01"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            planned = artifact.get_download_path_pairs(artifacts, tmpdir)

            rel_to_local = {rel: local for rel, local in planned}

            self.assertEqual(
                rel_to_local,
                {
                    "a.txt": tmpdir,
                    "sub/b.bin": os.path.join(tmpdir, "sub"),
                    "sub/deeper/c.md": os.path.join(tmpdir, "sub", "deeper"),
                },
            )

    def test_download_paths_with_empty_base_dir(self) -> None:
        artifacts = [
            artifact.ArtifactInfo(name="root.txt", size=1, md5="aaa", last_modified="2024-01-01"),
            artifact.ArtifactInfo(name="x/child.dat", size=2, md5="bbb", last_modified="2024-01-01"),
            artifact.ArtifactInfo(name="x/y/leaf.log", size=3, md5="ccc", last_modified="2024-01-01"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            prev_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                planned = artifact.get_download_path_pairs(artifacts, "")
            finally:
                os.chdir(prev_cwd)

            rel_to_local = {rel: local for rel, local in planned}

            self.assertEqual(
                rel_to_local,
                {
                    "root.txt": "",
                    "x/child.dat": "x",
                    "x/y/leaf.log": "x/y",
                },
            )


if __name__ == "__main__":
    absltest.main()
