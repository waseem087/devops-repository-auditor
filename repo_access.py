"""Safe acquisition of a public GitHub repository.

Safety measures (see README for residual risks):
  * only https://github.com/OWNER/REPO URLs are accepted (strict regex);
  * `git clone` runs with hooks disabled, symlink checkout disabled, no submodules,
    no credential helper, no interactive prompts and only the https protocol;
  * the clone is shallow, time-limited and size-checked afterwards;
  * the clone lives in a temporary directory that is always deleted.
Nothing in the cloned repository is ever executed.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from config import Limits
from redact import redact_text


class RepoAccessError(Exception):
    """A problem acquiring the repository. The message is safe to show to the user."""


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def canonical_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,38})/(?P<name>[A-Za-z0-9._-]{1,100})$"
)


def parse_github_url(url: str) -> RepoRef:
    """Validate a GitHub repository URL and return its parts."""
    if not isinstance(url, str) or not url.strip():
        raise RepoAccessError("Repository URL is empty.")
    cleaned = url.strip()
    if len(cleaned) > 300:
        raise RepoAccessError("Repository URL is too long.")
    cleaned = cleaned.rstrip("/")
    cleaned = cleaned.removesuffix(".git")
    match = _URL_RE.match(cleaned)
    if not match or match.group("name") in {".", ".."}:
        raise RepoAccessError(
            "Unsupported repository URL. Expected the form https://github.com/OWNER/REPOSITORY "
            "(public GitHub repositories only; no credentials, query strings or sub-paths)."
        )
    return RepoRef(owner=match.group("owner"), name=match.group("name"))


def safe_git_env() -> dict[str, str]:
    """Environment for git subprocesses: no GIT_* overrides, no API keys or tokens, no prompts."""
    secret_markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith("GIT_") and not any(m in k.upper() for m in secret_markers)
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return env


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _explain_clone_failure(stderr: str) -> str:
    low = stderr.lower()
    if "not found" in low or "repository not found" in low or "could not read username" in low:
        return "Repository not found, deleted, or not public."
    if "rate limit" in low or "403" in low or "429" in low:
        return "GitHub refused or rate-limited the request. Wait a few minutes and retry."
    if "could not resolve host" in low or "unable to access" in low or "failed to connect" in low:
        return "Network error while contacting GitHub. Check your internet connection."
    return "git clone failed: " + redact_text(stderr.strip().replace("\n", " "))[:200]


def clone_repo(ref: RepoRef, dest: Path, limits: Limits) -> Path:
    """Shallow-clone `ref` into `dest` (which must not exist yet)."""
    cmd = [
        "git",
        "-c", f"core.hooksPath={os.devnull}",
        "-c", "core.symlinks=false",
        "-c", "core.fsmonitor=false",
        "-c", "credential.helper=",
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        "clone", "--quiet", "--depth", "50", "--no-tags", "--single-branch",
        "--no-recurse-submodules", "--", ref.canonical_url, str(dest),
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=limits.clone_timeout_s,
            env=safe_git_env(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RepoAccessError(f"Cloning timed out after {limits.clone_timeout_s:.0f}s.") from exc
    except FileNotFoundError as exc:
        raise RepoAccessError("git is not installed or not on PATH.") from exc
    if proc.returncode != 0:
        raise RepoAccessError(_explain_clone_failure(proc.stderr or ""))
    if _dir_size(dest) > limits.max_repo_bytes:
        raise RepoAccessError(f"Repository is larger than the {limits.max_repo_bytes} byte limit.")
    return dest


def _rmtree_handler(func, path, _exc) -> None:  # type: ignore[no-untyped-def]
    """Windows marks git objects read-only; make them writable and retry the delete."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def safe_rmtree(path: str | Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_rmtree_handler)
    else:
        shutil.rmtree(path, onerror=_rmtree_handler)


@contextmanager
def acquire_repo(ref: RepoRef, limits: Limits) -> Iterator[Path]:
    """Clone into a temp directory, yield the checkout path, always clean up."""
    tmp = tempfile.mkdtemp(prefix="auditor-")
    try:
        yield clone_repo(ref, Path(tmp) / "repo", limits)
    finally:
        safe_rmtree(tmp)
