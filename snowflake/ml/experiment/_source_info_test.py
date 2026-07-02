import subprocess
import sys
import types
from unittest import mock
from unittest.mock import patch

from absl.testing import absltest

from snowflake.ml.experiment import _source_info


class ZMQInteractiveShell:  # noqa: N801 - mirrors IPython's kernel shell class name
    pass


class TerminalInteractiveShell:  # noqa: N801 - mirrors IPython's terminal shell class name
    pass


def _fake_ipython_module(shell: object) -> types.ModuleType:
    module = types.ModuleType("IPython")
    module.get_ipython = lambda: shell  # type: ignore[attr-defined]
    return module


class GitHelperTest(absltest.TestCase):
    @patch("snowflake.ml.experiment._source_info.subprocess.check_output", autospec=True)
    def test_git_returns_stripped_stdout(self, mock_check_output: object) -> None:
        mock_check_output.return_value = "abc123\n"  # type: ignore[attr-defined]
        self.assertEqual(_source_info._git("rev-parse", "HEAD"), "abc123")

    @patch("snowflake.ml.experiment._source_info.subprocess.check_output", autospec=True)
    def test_git_returns_none_on_empty(self, mock_check_output: object) -> None:
        mock_check_output.return_value = "   \n"  # type: ignore[attr-defined]
        self.assertIsNone(_source_info._git("config", "--get", "missing.key"))

    @patch("snowflake.ml.experiment._source_info.subprocess.check_output", autospec=True)
    def test_git_returns_none_on_subprocess_error(self, mock_check_output: object) -> None:
        mock_check_output.side_effect = subprocess.CalledProcessError(  # type: ignore[attr-defined]
            returncode=128, cmd=("git", "status")
        )
        self.assertIsNone(_source_info._git("status"))

    @patch("snowflake.ml.experiment._source_info.subprocess.check_output", autospec=True)
    def test_git_returns_none_on_timeout(self, mock_check_output: object) -> None:
        mock_check_output.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=2.0)  # type: ignore[attr-defined]
        self.assertIsNone(_source_info._git("status"))

    @patch("snowflake.ml.experiment._source_info.subprocess.check_output", autospec=True)
    def test_git_returns_none_when_git_not_installed(self, mock_check_output: object) -> None:
        mock_check_output.side_effect = FileNotFoundError("git not found")  # type: ignore[attr-defined]
        self.assertIsNone(_source_info._git("--version"))


class GitCwdTest(absltest.TestCase):
    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, return_value="/cwd")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True, return_value="true")
    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["/repo/train/main.py"])
    def test_returns_script_dir_when_inside_work_tree(self, mock_git: object, mock_getcwd: object) -> None:
        # The script-dir probe succeeds, so we should not fall back to CWD.
        self.assertEqual(_source_info._git_cwd(), "/repo/train")
        mock_getcwd.assert_not_called()  # type: ignore[attr-defined]

    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, return_value="/cwd")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["/tmp/scratch.py"])
    def test_returns_none_for_real_script_outside_repo(self, mock_git: object, mock_getcwd: object) -> None:
        # A real script whose dir is not in a work tree must NOT fall back to CWD,
        # which could be an unrelated repo. Drop git instead.
        self.assertIsNone(_source_info._git_cwd())
        mock_getcwd.assert_not_called()  # type: ignore[attr-defined]

    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, return_value="/cwd")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["-c"])
    def test_falls_back_to_cwd_for_non_file_argv0(self, mock_git: object, mock_getcwd: object) -> None:
        # `python -c "..."` — no path hint, so CWD is the only signal.
        self.assertEqual(_source_info._git_cwd(), "/cwd")
        mock_git.assert_not_called()  # type: ignore[attr-defined]

    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, side_effect=OSError)
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["-c"])
    def test_returns_none_when_cwd_fallback_fails(self, mock_git: object, mock_getcwd: object) -> None:
        # Non-file argv[0] and os.getcwd() unavailable: nothing to anchor on.
        self.assertIsNone(_source_info._git_cwd())


class ScrubUrlTest(absltest.TestCase):
    def test_scrubs_https_basic_auth(self) -> None:
        self.assertEqual(
            _source_info._scrub_url("https://user:token@github.com/foo/bar.git"),
            "https://github.com/foo/bar.git",
        )

    def test_scrubs_https_user_only(self) -> None:
        self.assertEqual(
            _source_info._scrub_url("https://user@github.com/foo/bar.git"),
            "https://github.com/foo/bar.git",
        )

    def test_scrubs_http(self) -> None:
        self.assertEqual(
            _source_info._scrub_url("http://user:pw@example.com/repo.git"),
            "http://example.com/repo.git",
        )

    def test_scrubs_ssh_scheme(self) -> None:
        self.assertEqual(
            _source_info._scrub_url("ssh://git:secret@host.example/repo.git"),
            "ssh://host.example/repo.git",
        )

    def test_passes_through_scp_form(self) -> None:
        # Classic SCP form (git@github.com:org/repo.git) has no scheme; the username
        # is not a credential and we deliberately do not rewrite it.
        url = "git@github.com:foo/bar.git"
        self.assertEqual(_source_info._scrub_url(url), url)

    def test_passes_through_credentialless_https(self) -> None:
        url = "https://github.com/foo/bar.git"
        self.assertEqual(_source_info._scrub_url(url), url)


class CollectGitTest(absltest.TestCase):
    def _git_side_effect(self, mapping: dict[tuple[str, ...], str]) -> object:
        def fake(*args: str, cwd: object = None) -> object:
            return mapping.get(args)

        return fake

    @patch("snowflake.ml.experiment._source_info._git_cwd", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    def test_returns_none_when_not_in_repo(self, mock_git: object, mock_cwd: object) -> None:
        mock_git.side_effect = self._git_side_effect({})  # type: ignore[attr-defined]
        self.assertIsNone(_source_info._collect_git())

    @patch("snowflake.ml.experiment._source_info._git_cwd", autospec=True, return_value="/tmp/repo")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    def test_returns_full_payload_on_happy_path(self, mock_git: object, mock_cwd: object) -> None:
        mock_git.side_effect = self._git_side_effect(  # type: ignore[attr-defined]
            {
                ("rev-parse", "--is-inside-work-tree"): "true",
                ("rev-parse", "HEAD"): "deadbeef",
                ("config", "--get", "remote.origin.url"): "https://user:pw@github.com/foo/bar.git",
                ("rev-parse", "--abbrev-ref", "HEAD"): "main",
            }
        )
        self.assertEqual(
            _source_info._collect_git(),
            _source_info.GitInfo(
                remote_url="https://github.com/foo/bar.git",
                commit_hash="deadbeef",
                branch="main",
            ),
        )

    @patch("snowflake.ml.experiment._source_info._git_cwd", autospec=True, return_value="/tmp/repo")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    def test_detached_head_drops_branch(self, mock_git: object, mock_cwd: object) -> None:
        mock_git.side_effect = self._git_side_effect(  # type: ignore[attr-defined]
            {
                ("rev-parse", "--is-inside-work-tree"): "true",
                ("rev-parse", "HEAD"): "abc123",
                ("config", "--get", "remote.origin.url"): "https://github.com/foo/bar.git",
                ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
            }
        )
        result = _source_info._collect_git()
        assert result is not None
        self.assertIsNone(result.branch)
        self.assertEqual(result.commit_hash, "abc123")

    @patch("snowflake.ml.experiment._source_info._git_cwd", autospec=True, return_value="/tmp/repo")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    def test_partial_fields_still_returned(self, mock_git: object, mock_cwd: object) -> None:
        # No remote configured, but everything else resolves.
        mock_git.side_effect = self._git_side_effect(  # type: ignore[attr-defined]
            {
                ("rev-parse", "--is-inside-work-tree"): "true",
                ("rev-parse", "HEAD"): "abc",
                ("rev-parse", "--abbrev-ref", "HEAD"): "feature/x",
            }
        )
        self.assertEqual(
            _source_info._collect_git(),
            _source_info.GitInfo(remote_url=None, commit_hash="abc", branch="feature/x"),
        )

    @patch("snowflake.ml.experiment._source_info._git_cwd", autospec=True, return_value="/tmp/repo")
    @patch("snowflake.ml.experiment._source_info._git", autospec=True)
    def test_returns_none_when_inside_repo_but_everything_else_fails(self, mock_git: object, mock_cwd: object) -> None:
        # `--is-inside-work-tree` succeeded but no other field did — treat as no info.
        mock_git.side_effect = self._git_side_effect(  # type: ignore[attr-defined]
            {("rev-parse", "--is-inside-work-tree"): "true"}
        )
        self.assertIsNone(_source_info._collect_git())


class DetectEntryPointTest(absltest.TestCase):
    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["-c"])
    def test_returns_none_for_dash_c(self) -> None:
        self.assertIsNone(_source_info._detect_entry_point())

    @patch("snowflake.ml.experiment._source_info.sys.argv", new=[""])
    def test_returns_none_for_empty_argv0(self) -> None:
        self.assertIsNone(_source_info._detect_entry_point())

    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["/repo/train/main.py"])
    @patch("snowflake.ml.experiment._source_info._git", autospec=True, return_value="/repo")
    def test_returns_repo_relative_path_when_in_repo(self, mock_git: object) -> None:
        self.assertEqual(_source_info._detect_entry_point(), "train/main.py")

    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["/home/me/standalone.py"])
    @patch("snowflake.ml.experiment._source_info._git", autospec=True, return_value=None)
    def test_returns_basename_when_outside_repo(self, mock_git: object) -> None:
        # Outside a repo we deliberately drop the directory to avoid leaking $HOME.
        self.assertEqual(_source_info._detect_entry_point(), "standalone.py")

    @patch("snowflake.ml.experiment._source_info.sys.argv", new=["/site-packages/ipykernel_launcher.py"])
    @patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True)
    def test_returns_none_in_ipython_kernel(self, mock_in_kernel: object) -> None:
        # In a kernel, argv[0] is the launcher; suppress it rather than emit a misleading path.
        self.assertIsNone(_source_info._detect_entry_point())


class InIpythonKernelTest(absltest.TestCase):
    def test_false_when_ipython_not_imported(self) -> None:
        with mock.patch.dict(sys.modules):
            sys.modules.pop("IPython", None)
            self.assertFalse(_source_info._in_ipython_kernel())

    def test_false_when_no_active_shell(self) -> None:
        with mock.patch.dict(sys.modules, {"IPython": _fake_ipython_module(None)}):
            self.assertFalse(_source_info._in_ipython_kernel())

    def test_false_in_terminal_repl(self) -> None:
        with mock.patch.dict(sys.modules, {"IPython": _fake_ipython_module(TerminalInteractiveShell())}):
            self.assertFalse(_source_info._in_ipython_kernel())

    def test_true_in_kernel(self) -> None:
        with mock.patch.dict(sys.modules, {"IPython": _fake_ipython_module(ZMQInteractiveShell())}):
            self.assertTrue(_source_info._in_ipython_kernel())

    def test_false_when_get_ipython_raises(self) -> None:
        module = types.ModuleType("IPython")

        def _boom() -> object:
            raise RuntimeError("no ipython")

        module.get_ipython = _boom  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"IPython": module}):
            self.assertFalse(_source_info._in_ipython_kernel())


class GitInfoTest(absltest.TestCase):
    def test_is_empty_true_for_all_none(self) -> None:
        self.assertTrue(_source_info.GitInfo().is_empty())

    def test_is_empty_false_when_any_field_set(self) -> None:
        self.assertFalse(_source_info.GitInfo(commit_hash="abc").is_empty())
        self.assertFalse(_source_info.GitInfo(remote_url="https://x").is_empty())
        self.assertFalse(_source_info.GitInfo(branch="main").is_empty())


class SourceInfoTest(absltest.TestCase):
    def test_is_empty_true_for_default(self) -> None:
        self.assertTrue(_source_info.SourceInfo().is_empty())

    def test_is_empty_true_when_git_is_empty(self) -> None:
        self.assertTrue(_source_info.SourceInfo(git=_source_info.GitInfo()).is_empty())

    def test_is_empty_false_when_entry_point_set(self) -> None:
        self.assertFalse(_source_info.SourceInfo(entry_point="m.py").is_empty())

    def test_is_empty_false_when_git_populated(self) -> None:
        self.assertFalse(_source_info.SourceInfo(git=_source_info.GitInfo(commit_hash="abc")).is_empty())

    def test_to_json_dict_empty(self) -> None:
        self.assertEqual(_source_info.SourceInfo().to_json_dict(), {})

    def test_to_json_dict_omits_git_when_none(self) -> None:
        self.assertEqual(
            _source_info.SourceInfo(entry_point="m.py").to_json_dict(),
            {"entry_point": "m.py"},
        )

    def test_to_json_dict_omits_entry_point_when_none(self) -> None:
        self.assertEqual(
            _source_info.SourceInfo(git=_source_info.GitInfo(commit_hash="abc")).to_json_dict(),
            {"git": {"commit_hash": "abc"}},
        )

    def test_to_json_dict_omits_none_git_fields(self) -> None:
        # remote_url is None and must not appear in the rendered git dict.
        self.assertEqual(
            _source_info.SourceInfo(
                entry_point="train/main.py",
                git=_source_info.GitInfo(commit_hash="abc", branch="main"),
            ).to_json_dict(),
            {"entry_point": "train/main.py", "git": {"commit_hash": "abc", "branch": "main"}},
        )

    def test_to_json_dict_omits_git_key_when_all_git_fields_none(self) -> None:
        # A GitInfo with every field None renders no "git" key at all.
        self.assertEqual(
            _source_info.SourceInfo(entry_point="m.py", git=_source_info.GitInfo()).to_json_dict(),
            {"entry_point": "m.py"},
        )

    def test_to_json_dict_drops_entry_point_containing_dollar_quote(self) -> None:
        # A "$$" would break the dollar-quoted SQL literal, so the field is dropped.
        self.assertEqual(
            _source_info.SourceInfo(
                entry_point="we$$ird.ipynb",
                git=_source_info.GitInfo(commit_hash="abc"),
            ).to_json_dict(),
            {"git": {"commit_hash": "abc"}},
        )

    def test_to_json_dict_drops_only_git_field_containing_dollar_quote(self) -> None:
        # Per-field: the offending branch is dropped while other git fields survive.
        self.assertEqual(
            _source_info.SourceInfo(
                entry_point="m.py",
                git=_source_info.GitInfo(commit_hash="abc", branch="we$$ird"),
            ).to_json_dict(),
            {"entry_point": "m.py", "git": {"commit_hash": "abc"}},
        )

    @patch("snowflake.ml.experiment._source_info._collect_git", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True, return_value=None)
    def test_collect_returns_empty_when_nothing_collected(self, mock_entry: object, mock_git: object) -> None:
        result = _source_info.SourceInfo.collect()
        self.assertTrue(result.is_empty())

    @patch(
        "snowflake.ml.experiment._source_info._collect_git",
        autospec=True,
        return_value=_source_info.GitInfo(remote_url="https://x", commit_hash="abc", branch="main"),
    )
    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True, return_value="m.py")
    def test_collect_composes_both_buckets(self, mock_entry: object, mock_git: object) -> None:
        self.assertEqual(
            _source_info.SourceInfo.collect(),
            _source_info.SourceInfo(
                entry_point="m.py",
                git=_source_info.GitInfo(remote_url="https://x", commit_hash="abc", branch="main"),
            ),
        )

    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True)
    def test_collect_never_raises(self, mock_entry: object) -> None:
        mock_entry.side_effect = RuntimeError("boom")  # type: ignore[attr-defined]
        self.assertTrue(_source_info.SourceInfo.collect().is_empty())


if __name__ == "__main__":
    absltest.main()
