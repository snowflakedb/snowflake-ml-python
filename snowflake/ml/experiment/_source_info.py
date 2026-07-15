"""Best-effort source provenance for ExperimentTracking.start_run().

The public entry point is :meth:`SourceInfo.collect`, which inspects the
surrounding environment and returns a :class:`SourceInfo`. Its
:meth:`SourceInfo.to_json_dict` renders the JSON-serializable shape embedded in
``ALTER EXPERIMENT ... ADD RUN ... WITH (SOURCE_INFO = ...)``.

Collection is strictly best-effort: any failure (missing ``git`` binary, not in a
repo, REPL) drops the field rather than raising.
"""

import dataclasses
import importlib
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
from typing import Any, Optional
from urllib import parse as urllib_parse, request as urllib_request

_GIT_TIMEOUT_SEC = 2.0

# Cap the localhost call to the Jupyter sessions API; notebook-name resolution is
# best-effort and must never noticeably delay run creation.
_JUPYTER_SESSIONS_TIMEOUT_SEC = 0.5

# Strip embedded credentials from git remote URLs before sending them upstream.
# Examples handled:
#   https://user:token@github.com/foo/bar.git -> https://github.com/foo/bar.git
#   ssh://git:secret@host.example/repo        -> ssh://host.example/repo
_URL_HTTPS_CRED_RE = re.compile(r"^(https?://)[^/@]+@")
_URL_SSH_CRED_RE = re.compile(r"^(ssh://)[^/@]+@")

# Sentinel argv[0] values that don't point to a user file.
_NON_FILE_ARGV0 = frozenset({"", "-c", "-"})

# Environment variables Snowflake injects to describe the running file. Their
# presence is how we recognize a Snowflake-managed file (a notebook or a .py
# workspace file), where the active file has no resolvable local filesystem path.
_SNOWFLAKE_FILE_DOMAIN_TYPE_ENV = "SNOWFLAKE_FILE_DOMAIN_TYPE"
_SNOWFLAKE_FILE_DOMAIN_NAME_ENV = "SNOWFLAKE_FILE_DOMAIN_NAME"
_SNOWFLAKE_MAIN_FILE_PATH_ENV = "SNOWFLAKE_MAIN_FILE_PATH"


@dataclasses.dataclass(frozen=True)
class GitInfo:
    """Git provenance for the surrounding repository.

    Any field may be ``None`` when it could not be resolved (e.g. ``remote_url``
    when no ``origin`` remote is configured, or ``branch`` on a detached HEAD).
    """

    remote_url: Optional[str] = None
    commit_hash: Optional[str] = None
    branch: Optional[str] = None

    def is_empty(self) -> bool:
        return self.remote_url is None and self.commit_hash is None and self.branch is None


@dataclasses.dataclass(frozen=True)
class SourceInfo:
    """Best-effort provenance attached to a run when it is created."""

    entry_point: Optional[str] = None
    git: Optional[GitInfo] = None
    # Provenance specific to Snowflake-managed files (notebooks or .py workspace
    # files), taken from the environment. The file's own path is carried by
    # ``entry_point``, not duplicated here.
    snowflake_file_domain_type: Optional[str] = None
    snowflake_file_domain_name: Optional[str] = None

    def is_empty(self) -> bool:
        """Return True when there is nothing useful to send."""
        return (
            self.entry_point is None
            and (self.git is None or self.git.is_empty())
            and self.snowflake_file_domain_type is None
            and self.snowflake_file_domain_name is None
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Render the payload for ``WITH (SOURCE_INFO = $$...$$)``.

        Keys with no value are omitted entirely (rather than serialized as
        ``null``) so the server only sees fields that were actually resolved.
        Values containing ``$$`` are also dropped, since the payload is embedded
        in a ``$$``-dollar-quoted SQL literal that a ``$$`` would break.

        Returns:
            A JSON-serializable dict whose keys (``entry_point``, ``git``,
            ``snowflake_file_domain_type``, ``snowflake_file_domain_name``) are
            included only when resolved. May be empty when nothing was collected.
        """
        out: dict[str, Any] = {}
        if self.entry_point is not None and "$$" not in self.entry_point:
            out["entry_point"] = self.entry_point
        if self.git is not None:
            git_dict = {
                key: value
                for key, value in dataclasses.asdict(self.git).items()
                if value is not None and "$$" not in value
            }
            if git_dict:
                out["git"] = git_dict
        if self.snowflake_file_domain_type is not None and "$$" not in self.snowflake_file_domain_type:
            out["snowflake_file_domain_type"] = self.snowflake_file_domain_type
        if self.snowflake_file_domain_name is not None and "$$" not in self.snowflake_file_domain_name:
            out["snowflake_file_domain_name"] = self.snowflake_file_domain_name
        return out

    @classmethod
    def collect(cls) -> "SourceInfo":
        """Collect source info from the surrounding environment. Never raises.

        Returns:
            A populated :class:`SourceInfo`, or an empty one when nothing could
            be resolved or collection failed.
        """
        try:
            # Snowflake-managed files (notebooks or .py workspace files) describe
            # the running file via environment variables rather than a resolvable
            # local path, so they are handled first and independently of the
            # git/notebook-path machinery.
            snowflake_file = _collect_snowflake_file()
            if snowflake_file is not None:
                return snowflake_file

            # Resolve the notebook once so entry-point and git agree on the same file.
            notebook_path = _resolve_notebook_path() if _in_ipython_kernel() else None
            return cls(
                entry_point=_detect_entry_point(notebook_path),
                git=_collect_git(notebook_path),
            )
        except Exception:
            return cls()


def _collect_snowflake_file() -> Optional[SourceInfo]:
    """Collect source info when running inside a Snowflake-managed file.

    Snowflake exposes the active file's location (a notebook or a .py workspace
    file) through environment variables instead of a resolvable local filesystem
    path. When any variable is present, the main-file path is recorded as the
    entry point and the domain type and name are captured alongside it.

    Returns:
        A :class:`SourceInfo` populated from the Snowflake file environment, or
        ``None`` when not running inside a Snowflake-managed file.
    """
    domain_type = os.environ.get(_SNOWFLAKE_FILE_DOMAIN_TYPE_ENV) or None
    domain_name = os.environ.get(_SNOWFLAKE_FILE_DOMAIN_NAME_ENV) or None
    main_file_path = os.environ.get(_SNOWFLAKE_MAIN_FILE_PATH_ENV) or None
    if domain_type is None and domain_name is None and main_file_path is None:
        return None
    return SourceInfo(
        entry_point=main_file_path,
        snowflake_file_domain_type=domain_type,
        snowflake_file_domain_name=domain_name,
    )


def _git(*args: str, cwd: Optional[str] = None) -> Optional[str]:
    """Run ``git <args>`` and return stripped stdout, or ``None`` on any failure."""
    try:
        out = subprocess.check_output(
            ("git", *args),
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SEC,
            text=True,
        )
    except Exception:
        return None
    out = out.strip()
    return out or None


def _git_cwd(notebook_path: Optional[str] = None) -> Optional[str]:
    """Pick a working directory for git resolution.

    In a notebook kernel, anchors on the resolved notebook's own directory so the
    captured repo matches the user's work. ``sys.argv[0]`` points at the kernel
    launcher in site-packages and ``os.getcwd()`` is wherever the kernel happened
    to start (often an unrelated repo, e.g. a Homebrew checkout), so neither is a
    safe fallback in a kernel — git is simply dropped when the notebook path is
    unknown.

    Outside a kernel, anchors on the directory of the user's script
    (``dirname(sys.argv[0])``) when that directory is inside a git work tree. A
    real script outside any work tree yields ``None`` rather than ``os.getcwd()``,
    so a run isn't stamped with an unrelated repo that merely happens to be the
    working directory. ``os.getcwd()`` is used only when ``sys.argv[0]`` is not a
    usable path hint (REPL, ``python -c``).

    Args:
        notebook_path: The resolved notebook path when running in a kernel, or
            ``None``.

    Returns:
        An absolute directory to use as the git ``cwd``, or ``None`` if no safe
        candidate is resolvable.
    """
    if _in_ipython_kernel():
        if notebook_path and os.path.isabs(notebook_path):
            return os.path.dirname(notebook_path) or None
        return None

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 in _NON_FILE_ARGV0:
        try:
            return os.getcwd()
        except Exception:
            return None
    try:
        script_dir = os.path.dirname(os.path.abspath(argv0)) or None
    except Exception:
        return None
    if script_dir and _git("rev-parse", "--is-inside-work-tree", cwd=script_dir):
        return script_dir
    return None


def _scrub_url(url: str) -> str:
    """Remove embedded ``user[:pass]@`` credentials from a git remote URL."""
    url = _URL_HTTPS_CRED_RE.sub(r"\1", url)
    url = _URL_SSH_CRED_RE.sub(r"\1", url)
    return url


def _collect_git(notebook_path: Optional[str] = None) -> Optional[GitInfo]:
    """Collect commit hash, remote URL, and branch from the surrounding git repo.

    Args:
        notebook_path: The resolved notebook path when running in a kernel, or
            ``None``. Used to anchor git resolution on the notebook's directory.

    Returns:
        A :class:`GitInfo` whose individual fields may be ``None`` when a single
        field is unresolvable. Returns ``None`` when ``git`` is unavailable, the
        directory is not a work tree, or every field came back empty.
    """
    cwd = _git_cwd(notebook_path)
    # Cheap probe — short-circuits to a single subprocess call when the script
    # is not inside a repo (the overwhelmingly common case for SPCS jobs).
    if not _git("rev-parse", "--is-inside-work-tree", cwd=cwd):
        return None

    commit = _git("rev-parse", "HEAD", cwd=cwd)
    remote = _git("config", "--get", "remote.origin.url", cwd=cwd)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    if branch == "HEAD":  # detached HEAD has no branch
        branch = None

    if not (commit or remote or branch):
        return None

    return GitInfo(
        remote_url=_scrub_url(remote) if remote else None,
        commit_hash=commit,
        branch=branch,
    )


def _ipython_shell() -> Optional[Any]:
    """Return the active IPython shell instance, or ``None`` if there isn't one.

    Read-only: we never import IPython ourselves. If the ``IPython`` module is not
    already loaded we are definitely not in an IPython session.

    Returns:
        The active IPython shell object, or ``None``.
    """
    ipython_module = sys.modules.get("IPython")
    if ipython_module is None:
        return None
    try:
        return ipython_module.get_ipython()
    except Exception:
        return None


def _in_ipython_kernel() -> bool:
    """Return True when running inside an IPython/Jupyter kernel (incl. notebooks).

    The active shell being a ``ZMQInteractiveShell`` distinguishes a kernel
    (Jupyter, Snowflake Notebooks) from a plain terminal REPL
    (``TerminalInteractiveShell``), where ``sys.argv[0]`` is still meaningful.

    Returns:
        True if the current process is an IPython kernel, False otherwise.
    """
    shell = _ipython_shell()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _jupyter_kernel_id() -> Optional[str]:
    """Return this kernel's id from the ipykernel connection file, or ``None``.

    The connection file is named ``kernel-<id>.json``.

    Returns:
        The kernel id, or ``None`` if it cannot be determined.
    """
    try:
        import ipykernel

        stem = os.path.splitext(os.path.basename(ipykernel.get_connection_file()))[0]
    except Exception:
        return None
    _, _, kernel_id = stem.partition("-")
    return kernel_id or None


def _jupyter_running_servers() -> list[dict[str, Any]]:
    """Return info dicts for running Jupyter servers (``url``, ``token``, root).

    Uses the canonical ``list_running_servers()`` from ``jupyter_server`` (modern)
    and ``notebook`` (classic), covering both ``jupyter lab`` and ``jupyter
    notebook``.

    Returns:
        Server info dicts; empty when the Jupyter packages are unavailable.
    """
    servers: list[dict[str, Any]] = []
    for module_name in ("jupyter_server.serverapp", "notebook.notebookapp"):
        try:
            servers.extend(importlib.import_module(module_name).list_running_servers())
        except Exception:
            continue
    return servers


def _is_loopback_url(url: str) -> bool:
    """Return True only when ``url`` targets the local machine over http(s).

    Confines the sessions-API request to the local Jupyter server: the scheme
    must be ``http``/``https`` and the host must resolve exclusively to loopback
    addresses. This prevents a stray or hostile ``list_running_servers()`` entry
    from redirecting the request (and its auth token) to an off-box host.

    Args:
        url: The server URL reported by ``list_running_servers()``.

    Returns:
        True if the URL is safe to request against, False otherwise.
    """
    try:
        parsed = urllib_parse.urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        addr_infos = socket.getaddrinfo(host, parsed.port, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    if not addr_infos:
        return False
    for addr_info in addr_infos:
        ip_text = addr_info[4][0]
        try:
            if not ipaddress.ip_address(ip_text).is_loopback:
                return False
        except ValueError:
            return False
    return True


def _jupyter_session_notebook_path() -> Optional[str]:
    """Resolve the active notebook path via the Jupyter server sessions API.

    Matches this kernel's id against each running server's ``/api/sessions``
    listing. The server-relative session path is joined with the server root to
    yield an absolute path when the root is known. The HTTP call is capped by a
    short timeout.

    Returns:
        The notebook path (absolute when the server root is known), or ``None``
        if no session matches or the lookup fails.
    """
    kernel_id = _jupyter_kernel_id()
    if not kernel_id:
        return None
    for server in _jupyter_running_servers():
        url = str(server.get("url", "")).rstrip("/")
        # Confine the request to the local Jupyter server so the auth token can
        # never be sent off-box via an unexpected server entry.
        if not url or not _is_loopback_url(url):
            continue
        # Carry the token in an Authorization header rather than the query string,
        # keeping the credential out of request/proxy logs and process listings.
        token = server.get("token", "")
        request = urllib_request.Request(f"{url}/api/sessions")
        if token:
            request.add_header("Authorization", f"token {token}")
        try:
            with urllib_request.urlopen(request, timeout=_JUPYTER_SESSIONS_TIMEOUT_SEC) as response:
                sessions = json.load(response)
        except Exception:
            continue
        for session in sessions:
            if session.get("kernel", {}).get("id") != kernel_id:
                continue
            path = session.get("path")
            if isinstance(path, str) and path:
                root = server.get("root_dir") or server.get("notebook_dir")
                return os.path.join(root, path) if isinstance(root, str) and root else path
    return None


def _resolve_notebook_path() -> Optional[str]:
    """Best-effort resolution of the active notebook's absolute path.

    Supports VS Code's Jupyter extension (the ``__vsc_ipynb_file__`` namespace
    entry it injects) and vanilla ``jupyter notebook`` / JupyterLab servers
    (matched via the sessions API). The path anchors both the entry point and
    git resolution; unsupported frontends drop the field rather than guess.
    Snowflake-managed files are handled separately via environment variables (see
    :func:`_collect_snowflake_file`) and never reach this resolver.

    Returns:
        The notebook's absolute path, or ``None`` if it cannot be resolved.
    """
    shell = _ipython_shell()
    if shell is not None:
        try:
            path = shell.user_ns.get("__vsc_ipynb_file__")
        except Exception:
            path = None
        if isinstance(path, str) and path:
            return path
    return _jupyter_session_notebook_path()


def _detect_entry_point(notebook_path: Optional[str] = None) -> Optional[str]:
    """Identify the user's entry-point file, git-root-relative when possible.

    Falls back to a basename outside a repo to avoid leaking ``$HOME`` or
    usernames.

    Args:
        notebook_path: The resolved notebook path when running in a kernel, or
            ``None``.

    Returns:
        The repo-relative path (e.g. ``"train/main.py"``) when inside a git
        work tree, otherwise the basename. In a notebook kernel, the resolved
        ``.ipynb`` basename when available. ``None`` for REPL (``-c``, ``-``),
        bare ``python`` invocations, and notebooks whose name cannot be resolved.
    """
    # In a kernel, sys.argv[0] is the launcher (e.g. ipykernel_launcher.py), not user
    # code. Record the resolved notebook basename instead, and drop the field when it
    # cannot be determined rather than recording the launcher path.
    if _in_ipython_kernel():
        return os.path.basename(notebook_path) if notebook_path else None

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 in _NON_FILE_ARGV0:
        return None

    try:
        abs_main = os.path.abspath(argv0)
    except Exception:
        return os.path.basename(argv0) or None

    root = _git("rev-parse", "--show-toplevel", cwd=os.path.dirname(abs_main) or None)
    if root:
        try:
            return os.path.relpath(abs_main, root)
        except ValueError:
            # relpath raises on Windows when paths span drive letters.
            pass

    return os.path.basename(abs_main) or None
