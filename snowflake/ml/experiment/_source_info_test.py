import json
import os
import shutil
import subprocess
import sys
import types
from typing import Literal, Optional
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


class _FakeHTTPResponse:
    """Minimal context-manager stand-in for urlopen() returning JSON."""

    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        return False

    def read(self, *args: object) -> bytes:
        return self._payload


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

    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, return_value="/cwd")
    @patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True)
    def test_anchors_on_notebook_dir_in_kernel(self, mock_kernel: object, mock_getcwd: object) -> None:
        # In a kernel we use the notebook's own directory and never CWD, which
        # could point at an unrelated repo (e.g. a Homebrew checkout).
        self.assertEqual(_source_info._git_cwd("/home/me/work/analysis.ipynb"), "/home/me/work")
        mock_getcwd.assert_not_called()  # type: ignore[attr-defined]

    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, return_value="/cwd")
    @patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True)
    def test_returns_none_in_kernel_when_notebook_unresolved(self, mock_kernel: object, mock_getcwd: object) -> None:
        self.assertIsNone(_source_info._git_cwd(None))
        mock_getcwd.assert_not_called()  # type: ignore[attr-defined]

    @patch("snowflake.ml.experiment._source_info.os.getcwd", autospec=True, return_value="/cwd")
    @patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True)
    def test_returns_none_in_kernel_for_relative_notebook_path(self, mock_kernel: object, mock_getcwd: object) -> None:
        # A server-relative path (no known root) cannot be anchored safely.
        self.assertIsNone(_source_info._git_cwd("sub/analysis.ipynb"))
        mock_getcwd.assert_not_called()  # type: ignore[attr-defined]


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

    @patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True)
    def test_returns_notebook_basename_in_kernel(self, mock_in_kernel: object) -> None:
        # In a kernel, argv[0] is the launcher; record the resolved notebook basename.
        self.assertEqual(_source_info._detect_entry_point("/home/me/work/analysis.ipynb"), "analysis.ipynb")

    @patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True)
    def test_returns_none_in_kernel_when_notebook_unresolved(self, mock_in_kernel: object) -> None:
        # Drop the field rather than record the launcher path.
        self.assertIsNone(_source_info._detect_entry_point(None))


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


class JupyterKernelIdTest(absltest.TestCase):
    def test_parses_id_from_connection_file(self) -> None:
        fake = types.ModuleType("ipykernel")
        fake.get_connection_file = lambda: "/run/user/kernel-abc-123.json"  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"ipykernel": fake}):
            self.assertEqual(_source_info._jupyter_kernel_id(), "abc-123")

    def test_none_when_ipykernel_unavailable(self) -> None:
        with mock.patch.dict(sys.modules, {"ipykernel": None}):
            self.assertIsNone(_source_info._jupyter_kernel_id())


class JupyterRunningServersTest(absltest.TestCase):
    def test_collects_from_both_modules(self) -> None:
        modern = types.SimpleNamespace(list_running_servers=lambda: [{"url": "a"}])
        classic = types.SimpleNamespace(list_running_servers=lambda: [{"url": "b"}])

        def fake_import(name: str) -> object:
            return {"jupyter_server.serverapp": modern, "notebook.notebookapp": classic}[name]

        with patch("snowflake.ml.experiment._source_info.importlib.import_module", side_effect=fake_import):
            self.assertEqual(_source_info._jupyter_running_servers(), [{"url": "a"}, {"url": "b"}])

    def test_returns_empty_when_unavailable(self) -> None:
        with patch(
            "snowflake.ml.experiment._source_info.importlib.import_module",
            side_effect=ImportError,
        ):
            self.assertEqual(_source_info._jupyter_running_servers(), [])


class JupyterSessionNotebookPathTest(absltest.TestCase):
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value=None)
    def test_none_without_kernel_id(self, mock_kid: object) -> None:
        self.assertIsNone(_source_info._jupyter_session_notebook_path())

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_running_servers",
        autospec=True,
        return_value=[{"url": "http://localhost:8888/", "token": "tok"}],
    )
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value="kid-123")
    def test_matches_session_by_kernel_id(self, mock_kid: object, mock_servers: object) -> None:
        sessions = [
            {"kernel": {"id": "other"}, "path": "Other.ipynb"},
            {"kernel": {"id": "kid-123"}, "path": "sub/My.ipynb"},
        ]
        with patch(
            "snowflake.ml.experiment._source_info.urllib_request.urlopen",
            return_value=_FakeHTTPResponse(sessions),
        ):
            self.assertEqual(_source_info._jupyter_session_notebook_path(), "sub/My.ipynb")

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_running_servers",
        autospec=True,
        return_value=[{"url": "http://localhost:8888/", "token": "tok", "root_dir": "/home/me/project"}],
    )
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value="kid-123")
    def test_joins_server_root_dir(self, mock_kid: object, mock_servers: object) -> None:
        sessions = [{"kernel": {"id": "kid-123"}, "path": "sub/My.ipynb"}]
        with patch(
            "snowflake.ml.experiment._source_info.urllib_request.urlopen",
            return_value=_FakeHTTPResponse(sessions),
        ):
            self.assertEqual(_source_info._jupyter_session_notebook_path(), "/home/me/project/sub/My.ipynb")

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_running_servers",
        autospec=True,
        return_value=[{"url": "http://localhost:8888/", "token": "tok"}],
    )
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value="kid-123")
    def test_none_when_no_session_matches(self, mock_kid: object, mock_servers: object) -> None:
        sessions = [{"kernel": {"id": "other"}, "path": "Other.ipynb"}]
        with patch(
            "snowflake.ml.experiment._source_info.urllib_request.urlopen",
            return_value=_FakeHTTPResponse(sessions),
        ):
            self.assertIsNone(_source_info._jupyter_session_notebook_path())

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_running_servers",
        autospec=True,
        return_value=[{"url": "http://localhost:8888/"}],
    )
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value="kid-123")
    def test_none_when_urlopen_raises(self, mock_kid: object, mock_servers: object) -> None:
        with patch(
            "snowflake.ml.experiment._source_info.urllib_request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            self.assertIsNone(_source_info._jupyter_session_notebook_path())

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_running_servers",
        autospec=True,
        return_value=[{"url": "http://evil.example.com:8888/", "token": "tok"}],
    )
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value="kid-123")
    def test_skips_non_loopback_server(self, mock_kid: object, mock_servers: object) -> None:
        with patch(
            "snowflake.ml.experiment._source_info.urllib_request.urlopen",
        ) as mock_urlopen:
            self.assertIsNone(_source_info._jupyter_session_notebook_path())
            mock_urlopen.assert_not_called()

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_running_servers",
        autospec=True,
        return_value=[{"url": "http://localhost:8888/", "token": "tok"}],
    )
    @patch("snowflake.ml.experiment._source_info._jupyter_kernel_id", autospec=True, return_value="kid-123")
    def test_token_sent_as_authorization_header(self, mock_kid: object, mock_servers: object) -> None:
        sessions = [{"kernel": {"id": "kid-123"}, "path": "My.ipynb"}]
        with patch(
            "snowflake.ml.experiment._source_info.urllib_request.urlopen",
            return_value=_FakeHTTPResponse(sessions),
        ) as mock_urlopen:
            _source_info._jupyter_session_notebook_path()
            request = mock_urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "http://localhost:8888/api/sessions")
            self.assertEqual(request.get_header("Authorization"), "token tok")


class IsLoopbackUrlTest(absltest.TestCase):
    def test_accepts_localhost(self) -> None:
        self.assertTrue(_source_info._is_loopback_url("http://localhost:8888/"))

    def test_accepts_loopback_ipv4(self) -> None:
        self.assertTrue(_source_info._is_loopback_url("http://127.0.0.1:8888/"))

    def test_rejects_non_http_scheme(self) -> None:
        self.assertFalse(_source_info._is_loopback_url("file:///etc/passwd"))

    def test_rejects_non_loopback_ip(self) -> None:
        self.assertFalse(_source_info._is_loopback_url("http://8.8.8.8:8888/"))

    def test_rejects_unresolvable_host(self) -> None:
        with patch(
            "snowflake.ml.experiment._source_info.socket.getaddrinfo",
            side_effect=OSError("name resolution failed"),
        ):
            self.assertFalse(_source_info._is_loopback_url("http://example.invalid:8888/"))

    def test_rejects_host_resolving_to_public_ip(self) -> None:
        with patch(
            "snowflake.ml.experiment._source_info.socket.getaddrinfo",
            return_value=[(2, 1, 6, "", ("93.184.216.34", 8888))],
        ):
            self.assertFalse(_source_info._is_loopback_url("http://sneaky.example:8888/"))


class ResolveNotebookPathTest(absltest.TestCase):
    @patch("snowflake.ml.experiment._source_info._jupyter_session_notebook_path", autospec=True)
    def test_prefers_vscode_namespace(self, mock_sessions: mock.MagicMock) -> None:
        shell = mock.MagicMock()
        shell.user_ns = {"__vsc_ipynb_file__": "/home/me/Analysis.ipynb"}
        with patch("snowflake.ml.experiment._source_info._ipython_shell", autospec=True, return_value=shell):
            self.assertEqual(_source_info._resolve_notebook_path(), "/home/me/Analysis.ipynb")
        # VS Code resolves it cheaply, so we never hit the sessions API.
        mock_sessions.assert_not_called()

    @patch(
        "snowflake.ml.experiment._source_info._jupyter_session_notebook_path",
        autospec=True,
        return_value="/srv/remote/Notebook.ipynb",
    )
    def test_falls_back_to_sessions_api(self, mock_sessions: object) -> None:
        shell = mock.MagicMock()
        shell.user_ns = {}
        with patch("snowflake.ml.experiment._source_info._ipython_shell", autospec=True, return_value=shell):
            self.assertEqual(_source_info._resolve_notebook_path(), "/srv/remote/Notebook.ipynb")

    @patch("snowflake.ml.experiment._source_info._jupyter_session_notebook_path", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info._ipython_shell", autospec=True, return_value=None)
    def test_returns_none_when_nothing_resolves(self, mock_shell: object, mock_sessions: object) -> None:
        self.assertIsNone(_source_info._resolve_notebook_path())


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

    def test_to_json_dict_includes_snowflake_file_fields(self) -> None:
        self.assertEqual(
            _source_info.SourceInfo(
                entry_point="git_capture_test.ipynb",
                snowflake_file_domain_type="workspace",
                snowflake_file_domain_name='USER$.PUBLIC."ML Runtime Testing"',
            ).to_json_dict(),
            {
                "entry_point": "git_capture_test.ipynb",
                "snowflake_file_domain_type": "workspace",
                "snowflake_file_domain_name": 'USER$.PUBLIC."ML Runtime Testing"',
            },
        )

    def test_to_json_dict_drops_snowflake_field_containing_dollar_quote(self) -> None:
        self.assertEqual(
            _source_info.SourceInfo(
                entry_point="nb.ipynb",
                snowflake_file_domain_type="we$$ird",
            ).to_json_dict(),
            {"entry_point": "nb.ipynb"},
        )

    def test_is_empty_false_when_snowflake_field_set(self) -> None:
        self.assertFalse(_source_info.SourceInfo(snowflake_file_domain_type="workspace").is_empty())
        self.assertFalse(_source_info.SourceInfo(snowflake_file_domain_name="USER$.PUBLIC.NB").is_empty())

    @patch("snowflake.ml.experiment._source_info._collect_snowflake_file", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info._collect_git", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True, return_value=None)
    def test_collect_returns_empty_when_nothing_collected(
        self, mock_entry: object, mock_git: object, mock_snowflake: object
    ) -> None:
        result = _source_info.SourceInfo.collect()
        self.assertTrue(result.is_empty())

    def test_collect_captures_running_file_without_mocks(self) -> None:
        # No mocks: collect() runs against this live test process, so the entry point
        # is resolved from the real sys.argv[0] — its basename is this running file.
        result = _source_info.SourceInfo.collect()
        assert result.entry_point is not None
        self.assertEqual(os.path.basename(result.entry_point), os.path.basename(sys.argv[0]))

    @patch("snowflake.ml.experiment._source_info._collect_snowflake_file", autospec=True, return_value=None)
    @patch(
        "snowflake.ml.experiment._source_info._collect_git",
        autospec=True,
        return_value=_source_info.GitInfo(remote_url="https://x", commit_hash="abc", branch="main"),
    )
    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True, return_value="m.py")
    def test_collect_composes_both_buckets(self, mock_entry: object, mock_git: object, mock_snowflake: object) -> None:
        self.assertEqual(
            _source_info.SourceInfo.collect(),
            _source_info.SourceInfo(
                entry_point="m.py",
                git=_source_info.GitInfo(remote_url="https://x", commit_hash="abc", branch="main"),
            ),
        )

    @patch(
        "snowflake.ml.experiment._source_info._collect_snowflake_file",
        autospec=True,
        return_value=_source_info.SourceInfo(
            entry_point="git_capture_test.ipynb",
            snowflake_file_domain_type="workspace",
            snowflake_file_domain_name='USER$.PUBLIC."ML Runtime Testing"',
        ),
    )
    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True)
    def test_collect_prefers_snowflake_file(self, mock_entry: object, mock_snowflake: object) -> None:
        # A Snowflake-managed file short-circuits the git/notebook-path machinery.
        self.assertEqual(
            _source_info.SourceInfo.collect(),
            _source_info.SourceInfo(
                entry_point="git_capture_test.ipynb",
                snowflake_file_domain_type="workspace",
                snowflake_file_domain_name='USER$.PUBLIC."ML Runtime Testing"',
            ),
        )
        mock_entry.assert_not_called()  # type: ignore[attr-defined]

    @patch("snowflake.ml.experiment._source_info._collect_snowflake_file", autospec=True, return_value=None)
    @patch("snowflake.ml.experiment._source_info._detect_entry_point", autospec=True)
    def test_collect_never_raises(self, mock_entry: object, mock_snowflake: object) -> None:
        mock_entry.side_effect = RuntimeError("boom")  # type: ignore[attr-defined]
        self.assertTrue(_source_info.SourceInfo.collect().is_empty())


class CollectSnowflakeFileTest(absltest.TestCase):
    _ENV_KEYS = ("SNOWFLAKE_FILE_DOMAIN_TYPE", "SNOWFLAKE_FILE_DOMAIN_NAME", "SNOWFLAKE_MAIN_FILE_PATH")

    def _clear_env(self) -> None:
        for key in self._ENV_KEYS:
            os.environ.pop(key, None)

    def test_none_when_not_a_snowflake_file(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            self._clear_env()
            self.assertIsNone(_source_info._collect_snowflake_file())

    def test_collects_all_fields(self) -> None:
        env = {
            "SNOWFLAKE_FILE_DOMAIN_TYPE": "workspace",
            "SNOWFLAKE_FILE_DOMAIN_NAME": 'USER$.PUBLIC."ML Runtime Testing"',
            "SNOWFLAKE_MAIN_FILE_PATH": "git_capture_test.ipynb",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                _source_info._collect_snowflake_file(),
                _source_info.SourceInfo(
                    entry_point="git_capture_test.ipynb",
                    snowflake_file_domain_type="workspace",
                    snowflake_file_domain_name='USER$.PUBLIC."ML Runtime Testing"',
                ),
            )

    def test_partial_env_still_collected(self) -> None:
        with mock.patch.dict(os.environ, {"SNOWFLAKE_FILE_DOMAIN_TYPE": "workspace"}, clear=False):
            os.environ.pop("SNOWFLAKE_FILE_DOMAIN_NAME", None)
            os.environ.pop("SNOWFLAKE_MAIN_FILE_PATH", None)
            self.assertEqual(
                _source_info._collect_snowflake_file(),
                _source_info.SourceInfo(snowflake_file_domain_type="workspace"),
            )

    def test_empty_env_values_treated_as_absent(self) -> None:
        env = {
            "SNOWFLAKE_FILE_DOMAIN_TYPE": "",
            "SNOWFLAKE_FILE_DOMAIN_NAME": "",
            "SNOWFLAKE_MAIN_FILE_PATH": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertIsNone(_source_info._collect_snowflake_file())


@absltest.skipUnless(shutil.which("git") is not None, "git binary not available")
class RealGitTest(absltest.TestCase):
    """End-to-end tests that drive collection against a real on-disk git repo.

    Only ``_in_ipython_kernel`` / ``_resolve_notebook_path`` are faked (to model
    "running in a notebook at this path"); git itself is exercised for real.
    """

    # Isolate from any global/system git config so commits and reads are deterministic.
    _GIT_ENV = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }

    def _git(self, cwd: str, *args: str) -> str:
        result = subprocess.run(
            ("git", "-C", cwd, *args),
            check=True,
            capture_output=True,
            text=True,
            env=self._GIT_ENV,
        )
        return result.stdout.strip()

    def _init_repo(self, path: str, *, remote: Optional[str] = None) -> None:
        self._git(path, "init", "-q")
        self._git(path, "config", "user.email", "test@example.com")
        self._git(path, "config", "user.name", "Test User")
        self._git(path, "config", "commit.gpgsign", "false")
        if remote is not None:
            self._git(path, "remote", "add", "origin", remote)

    def _commit(self, path: str, filename: str) -> str:
        with open(os.path.join(path, filename), "w") as handle:
            handle.write("{}\n")
        self._git(path, "add", "-A")
        self._git(path, "commit", "-qm", f"add {filename}")
        return self._git(path, "rev-parse", "HEAD")

    def _collect_in_kernel(self, notebook_path: str) -> _source_info.SourceInfo:
        with patch(
            "snowflake.ml.experiment._source_info._collect_snowflake_file", autospec=True, return_value=None
        ), patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=True), patch(
            "snowflake.ml.experiment._source_info._resolve_notebook_path",
            autospec=True,
            return_value=notebook_path,
        ):
            return _source_info.SourceInfo.collect()

    def test_notebook_in_repo_captures_scrubbed_git(self) -> None:
        repo = self.create_tempdir().full_path
        self._init_repo(repo, remote="https://user:token@github.com/acme/proj.git")
        commit = self._commit(repo, "a.ipynb")
        self._git(repo, "branch", "-M", "main")

        result = self._collect_in_kernel(os.path.join(repo, "a.ipynb"))

        self.assertEqual(result.entry_point, "a.ipynb")
        assert result.git is not None
        self.assertEqual(result.git.remote_url, "https://github.com/acme/proj.git")
        self.assertEqual(result.git.commit_hash, commit)
        self.assertEqual(result.git.branch, "main")

    def test_git_dropped_when_notebook_dir_not_a_repo_even_if_cwd_is(self) -> None:
        # Regression: in a kernel we must anchor on the notebook's directory and
        # never os.getcwd(), which previously picked up an unrelated repo.
        repo = self.create_tempdir().full_path
        self._init_repo(repo, remote="https://github.com/acme/proj.git")
        self._commit(repo, "code.py")

        notebook_dir = self.create_tempdir().full_path  # a sibling dir, not a repo
        notebook = os.path.join(notebook_dir, "scratch.ipynb")
        open(notebook, "w").close()

        original_cwd = os.getcwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(repo)  # os.getcwd() is now inside a real repo

        result = self._collect_in_kernel(notebook)

        self.assertEqual(result.entry_point, "scratch.ipynb")
        self.assertIsNone(result.git)

    def test_detached_head_has_no_branch(self) -> None:
        repo = self.create_tempdir().full_path
        self._init_repo(repo, remote="https://github.com/acme/proj.git")
        first = self._commit(repo, "a.ipynb")
        self._commit(repo, "b.ipynb")
        self._git(repo, "checkout", "-q", first)  # detached HEAD

        result = self._collect_in_kernel(os.path.join(repo, "a.ipynb"))

        assert result.git is not None
        self.assertEqual(result.git.commit_hash, first)
        self.assertIsNone(result.git.branch)

    def test_repo_without_remote_yields_none_remote_url(self) -> None:
        repo = self.create_tempdir().full_path
        self._init_repo(repo)  # no origin remote configured
        commit = self._commit(repo, "a.ipynb")

        result = self._collect_in_kernel(os.path.join(repo, "a.ipynb"))

        assert result.git is not None
        self.assertIsNone(result.git.remote_url)
        self.assertEqual(result.git.commit_hash, commit)

    def test_git_dropped_for_script_outside_repo_even_if_cwd_is(self) -> None:
        # Regression: `python ../script.py` run from inside a repo must NOT stamp
        # the run with that repo when the script lives outside any work tree.
        # os.getcwd() being a repo is not a safe signal here.
        repo = self.create_tempdir().full_path
        self._init_repo(repo, remote="https://github.com/acme/proj.git")
        self._commit(repo, "code.py")

        script_dir = self.create_tempdir().full_path  # a sibling dir, not a repo
        script = os.path.join(script_dir, "scratch.py")
        open(script, "w").close()

        original_cwd = os.getcwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(repo)  # os.getcwd() is now inside a real repo

        with patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=False), patch(
            "snowflake.ml.experiment._source_info.sys.argv", new=[script]
        ):
            self.assertEqual(_source_info._detect_entry_point(), "scratch.py")
            self.assertIsNone(_source_info._collect_git())

    def test_script_entry_point_is_repo_relative(self) -> None:
        repo = self.create_tempdir().full_path
        self._init_repo(repo, remote="https://github.com/acme/proj.git")
        os.makedirs(os.path.join(repo, "train"))
        script = os.path.join(repo, "train", "main.py")
        open(script, "w").close()
        self._commit(repo, "a.ipynb")

        with patch("snowflake.ml.experiment._source_info._in_ipython_kernel", autospec=True, return_value=False), patch(
            "snowflake.ml.experiment._source_info.sys.argv", new=[script]
        ):
            self.assertEqual(_source_info._detect_entry_point(), os.path.join("train", "main.py"))
            self.assertIsNotNone(_source_info._collect_git())


if __name__ == "__main__":
    absltest.main()
