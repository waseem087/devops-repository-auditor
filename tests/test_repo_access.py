import stat
import subprocess

import pytest

import repo_access
from config import Limits
from repo_access import (
    RepoAccessError,
    RepoRef,
    clone_repo,
    parse_github_url,
    safe_rmtree,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World/",
        "https://github.com/octocat/Hello-World.git",
        "  https://github.com/octocat/Hello.World  ",
    ],
)
def test_valid_urls(url):
    ref = parse_github_url(url)
    assert ref.owner == "octocat"
    assert ref.canonical_url.startswith("https://github.com/octocat/Hello")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "http://github.com/a/b",
        "https://gitlab.com/a/b",
        "https://github.com/a",
        "https://github.com/a/b/tree/main",
        "https://user:pw@github.com/a/b",
        "https://github.com/a/b?x=1",
        "https://github.com.evil.com/a/b",
        "git@github.com:a/b.git",
        "file:///etc/passwd",
        "https://github.com/a/..",
        "https://github.com/-bad/repo",
        "https://github.com/a/b c",
    ],
)
def test_invalid_urls(url):
    with pytest.raises(RepoAccessError):
        parse_github_url(url)


def test_non_string_rejected():
    with pytest.raises(RepoAccessError):
        parse_github_url(None)  # type: ignore[arg-type]


def _fake_run(returncode=0, stderr="", exc=None):
    def run(*a, **k):
        if exc:
            raise exc
        return subprocess.CompletedProcess(a, returncode, "", stderr)

    return run


def test_clone_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_access.subprocess, "run", _fake_run(exc=subprocess.TimeoutExpired("git", 1)))
    with pytest.raises(RepoAccessError, match="timed out"):
        clone_repo(RepoRef("a", "b"), tmp_path / "r", Limits())


def test_clone_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_access.subprocess, "run", _fake_run(128, "fatal: repository 'x' not found"))
    with pytest.raises(RepoAccessError, match="not found"):
        clone_repo(RepoRef("a", "b"), tmp_path / "r", Limits())


def test_clone_git_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_access.subprocess, "run", _fake_run(exc=FileNotFoundError()))
    with pytest.raises(RepoAccessError, match="git is not installed"):
        clone_repo(RepoRef("a", "b"), tmp_path / "r", Limits())


def test_clone_uses_hardened_flags(monkeypatch, tmp_path):
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"], seen["env"] = cmd, kwargs["env"]
        (tmp_path / "r").mkdir()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok-test")
    monkeypatch.setattr(repo_access.subprocess, "run", run)
    clone_repo(RepoRef("a", "b"), tmp_path / "r", Limits())
    joined = " ".join(seen["cmd"])
    for flag in ("core.symlinks=false", "--depth", "--no-recurse-submodules", "credential.helper="):
        assert flag in joined
    assert not {"ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SOME_SERVICE_TOKEN"} & set(seen["env"])
    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_size_limit(monkeypatch, tmp_path):
    dest = tmp_path / "r"

    def run(cmd, **kwargs):
        dest.mkdir()
        (dest / "big.txt").write_text("x" * 5000)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(repo_access.subprocess, "run", run)
    with pytest.raises(RepoAccessError, match="larger"):
        clone_repo(RepoRef("a", "b"), dest, Limits(max_repo_bytes=1000))


def test_safe_rmtree_handles_readonly(tmp_path):
    d = tmp_path / "x" / "y"
    d.mkdir(parents=True)
    f = d / "ro.txt"
    f.write_text("a")
    f.chmod(stat.S_IREAD)
    safe_rmtree(tmp_path / "x")
    assert not (tmp_path / "x").exists()
