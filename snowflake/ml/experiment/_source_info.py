"""Best-effort source provenance for ExperimentTracking.start_run().

The public entry point is :meth:`SourceInfo.collect`, which inspects the
surrounding environment and returns a :class:`SourceInfo`. Its
:meth:`SourceInfo.to_json_dict` renders the JSON-serializable shape embedded in
``ALTER EXPERIMENT ... ADD RUN ... WITH (SOURCE_INFO = ...)``.

Collection is strictly best-effort: any failure (missing ``git`` binary, not in a
repo, REPL) drops the field rather than raising.
"""

import dataclasses
import os
import re
import subprocess
import sys
from typing import Any, Optional

_GIT_TIMEOUT_SEC = 2.0

# Strip embedded credentials from git remote URLs before sending them upstream.
# Examples handled:
#   https://user:token@github.com/foo/bar.git -> https://github.com/foo/bar.git
#   ssh://git:secret@host.example/repo        -> ssh://host.example/repo
_URL_HTTPS_CRED_RE = re.compile(r"^(https?://)[^/@]+@")
_URL_SSH_CRED_RE = re.compile(r"^(ssh://)[^/@]+@")

# Sentinel argv[0] values that don't point to a user file.
_NON_FILE_ARGV0 = frozenset({"", "-c", "-"})


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

    def is_empty(self) -> bool:
        """Return True when there is nothing useful to send."""
        return self.entry_point is None and (self.git is None or self.git.is_empty())

    def to_json_dict(self) -> dict[str, Any]:
        """Render the payload for ``WITH (SOURCE_INFO = $$...$$)``.

        Keys with no value are omitted entirely (rather than serialized as
        ``null``) so the server only sees fields that were actually resolved.
        Values containing ``$$`` are also dropped, since the payload is embedded
        in a ``$$``-dollar-quoted SQL literal that a ``$$`` would break.

        Returns:
            A JSON-serializable dict with up to two keys, ``entry_point`` and
            ``git``. May be empty when nothing was collected.
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
        return out

    @classmethod
    def collect(cls) -> "SourceInfo":
        """Collect source info from the surrounding environment. Never raises.

        Returns:
            A populated :class:`SourceInfo`, or an empty one when nothing could
            be resolved or collection failed.
        """
        try:
            return cls(entry_point=_detect_entry_point(), git=_collect_git())
        except Exception:
            return cls()


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


def _git_cwd() -> Optional[str]:
    """Pick a working directory for git resolution.

    Prefers the directory of the user's script (``dirname(sys.argv[0])``) when
    that directory is itself inside a git work tree, so the resolved repo
    matches the code under test rather than the interpreter's launch dir.

    A real script whose directory is not in a work tree yields ``None`` rather
    than falling back to ``os.getcwd()``: otherwise ``python /tmp/scratch.py``
    launched from inside an unrelated repo would stamp the run with that repo's
    git info while ``entry_point`` is just ``scratch.py``. ``os.getcwd()`` is used
    only when ``sys.argv[0]`` is not a usable path hint (REPL, ``python -c``).

    Returns:
        An absolute directory to use as the git ``cwd``, or ``None`` if no safe
        candidate is resolvable.
    """
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


def _collect_git() -> Optional[GitInfo]:
    """Collect commit hash, remote URL, and branch from the surrounding git repo.

    Returns:
        A :class:`GitInfo` whose individual fields may be ``None`` when a single
        field is unresolvable. Returns ``None`` when ``git`` is unavailable, the
        directory is not a work tree, or every field came back empty.
    """
    cwd = _git_cwd()
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


def _in_ipython_kernel() -> bool:
    """Return True when running inside an IPython/Jupyter kernel (incl. notebooks).

    Detection is read-only: we never import IPython ourselves. If the ``IPython``
    module is not already loaded we are definitely not in a kernel. When it is
    loaded, the active shell being a ``ZMQInteractiveShell`` distinguishes a
    kernel (Jupyter, Snowflake Notebooks) from a plain terminal REPL
    (``TerminalInteractiveShell``), where ``sys.argv[0]`` is still meaningful.

    Returns:
        True if the current process is an IPython kernel, False otherwise.
    """
    ipython_module = sys.modules.get("IPython")
    if ipython_module is None:
        return False
    try:
        shell = ipython_module.get_ipython()
    except Exception:
        return False
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _detect_entry_point() -> Optional[str]:
    """Identify the user's entry-point file, git-root-relative when possible.

    Falls back to a basename outside a repo to avoid leaking ``$HOME`` or
    usernames.

    Returns:
        The repo-relative path (e.g. ``"train/main.py"``) when inside a git
        work tree, otherwise the basename. ``None`` for REPL (``-c``, ``-``),
        bare ``python`` invocations, and IPython/Jupyter kernels (where
        ``sys.argv[0]`` points at the kernel launcher rather than user code).
    """
    # In a kernel, sys.argv[0] is the launcher (e.g. ipykernel_launcher.py), which
    # would be misleading as an entry point. Notebook lineage is captured separately.
    if _in_ipython_kernel():
        return None

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
