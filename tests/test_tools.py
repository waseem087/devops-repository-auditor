import os
import shutil

import pytest

from config import Limits
from tools import TOOL_SPECS, RepoTools

from .conftest import can_symlink, fake_github_token, git, write

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


# ---------------------------------------------------------------- list_files
def test_list_files_finds_notable_files(tools):
    r = tools.dispatch("list_files", {})
    assert r["ok"] and r["file_count"] == 7
    assert r["notable"]["workflows"] == [".github/workflows/ci.yml"]
    assert "requirements.txt" in r["notable"]["dependency_manifests"]
    assert "tests/test_app.py" in r["notable"]["tests"]
    assert "README.md" in r["notable"]["docs"]
    assert r["directories"]["src"] == 1


def test_list_files_reports_but_skips_committed_node_modules(repo):
    write(repo, "node_modules/lib/index.js", "x")
    r = RepoTools(repo).dispatch("list_files", {})
    assert "node_modules" in r["skipped_directories"]
    assert not any(p.startswith("node_modules") for p in r["files"])
    inside = RepoTools(repo).dispatch("list_files", {"path": "node_modules"})
    assert inside["files"] == ["node_modules/lib/index.js"]


def test_list_files_empty_repo(tmp_path):
    (tmp_path / "empty").mkdir()
    r = RepoTools(tmp_path / "empty").dispatch("list_files", {})
    assert r["ok"] and r["file_count"] == 0 and "no files" in r["note"]


def test_list_files_count_limit(repo):
    r = RepoTools(repo, Limits(max_files=3, max_listed_files=2)).dispatch("list_files", {})
    assert r["file_count"] == 3 and r["count_truncated"] and len(r["files"]) == 2


def test_list_files_depth_limit(tmp_path):
    write(tmp_path, "a/b/c/deep.txt", "x")
    write(tmp_path, "top.txt", "x")
    r = RepoTools(tmp_path).dispatch("list_files", {"max_depth": 1})
    assert r["depth_limited"] and "a/b/c/deep.txt" not in r["files"]


def test_list_files_subdirectory_and_errors(tools):
    assert tools.dispatch("list_files", {"path": "src"})["files"] == ["src/app.py"]
    assert not tools.dispatch("list_files", {"path": "nope"})["ok"]
    assert not tools.dispatch("list_files", {"path": "README.md"})["ok"]
    assert not tools.dispatch("list_files", {"path": "../"})["ok"]


def test_list_files_unusual_names(tmp_path):
    write(tmp_path, "weird name (1)/caf\u00e9 \u00fc.txt", "x")
    r = RepoTools(tmp_path).dispatch("list_files", {})
    assert r["ok"] and r["file_count"] == 1


def test_list_files_reports_large_files(tmp_path):
    (tmp_path / "big.dat").write_bytes(b"0" * 1_100_000)
    r = RepoTools(tmp_path).dispatch("list_files", {})
    assert r["large_files"][0]["path"] == "big.dat"


# ----------------------------------------------------------------- read_file
def test_read_file_numbers_lines(tools):
    r = tools.dispatch("read_file", {"path": "src/app.py"})
    assert r["ok"] and r["total_lines"] == 2
    assert r["content"].splitlines()[0] == "1: def main():"


def test_read_file_redacts_secret(tools):
    r = tools.dispatch("read_file", {"path": "config/settings.py"})
    assert fake_github_token() not in r["content"]
    assert r["redaction_applied"] is True
    assert "API_TOKEN" in r["content"]


@pytest.mark.parametrize(
    "bad",
    ["../outside.txt", "src/../../outside.txt", "/etc/passwd", "C:\\Windows\\win.ini", ".git/config", "a\x00b", "", "  "],
)
def test_read_file_rejects_unsafe_paths(tools, bad):
    r = tools.dispatch("read_file", {"path": bad})
    assert r["ok"] is False and "error" in r


def test_read_file_rejects_symlink_escape(repo, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks unavailable")
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    os.symlink(outside, repo / "link.txt")
    os.symlink(tmp_path, repo / "dirlink")
    t = RepoTools(repo)
    assert not t.dispatch("read_file", {"path": "link.txt"})["ok"]
    assert not t.dispatch("read_file", {"path": "dirlink/secret.txt"})["ok"]
    listing = t.dispatch("list_files", {})
    assert "link.txt" in listing["symlinks"] and "dirlink" in listing["symlinks"]


def test_read_file_allows_internal_symlink(repo, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks unavailable")
    os.symlink(repo / "README.md", repo / "alias.md")
    assert RepoTools(repo).dispatch("read_file", {"path": "alias.md"})["ok"]


def test_read_file_missing_directory_binary_large(repo):
    (repo / "blob.bin").write_bytes(b"abc\x00def")
    (repo / "huge.txt").write_text("x" * 2000)
    t = RepoTools(repo, Limits(max_file_bytes=1000))
    assert "not found" in t.dispatch("read_file", {"path": "missing.txt"})["error"]
    assert "directory" in t.dispatch("read_file", {"path": "src"})["error"]
    assert "binary" in t.dispatch("read_file", {"path": "blob.bin"})["error"]
    assert "too large" in t.dispatch("read_file", {"path": "huge.txt"})["error"]


def test_read_file_pagination_and_empty_file(repo):
    write(repo, "long.txt", "\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    write(repo, "empty.txt", "")
    t = RepoTools(repo)
    r = t.dispatch("read_file", {"path": "long.txt", "max_lines": 4})
    assert r["end_line"] == 4 and r["truncated"] and r["next_start_line"] == 5
    r2 = t.dispatch("read_file", {"path": "long.txt", "start_line": 9})
    assert r2["content"].splitlines()[0] == "9: line9" and not r2["truncated"]
    assert not t.dispatch("read_file", {"path": "long.txt", "start_line": 99})["ok"]
    e = t.dispatch("read_file", {"path": "empty.txt"})
    assert e["ok"] and e["total_lines"] == 0


def test_read_file_output_cap(repo):
    write(repo, "wide.txt", "\n".join("y" * 200 for _ in range(100)))
    r = RepoTools(repo, Limits(max_output_chars=1000)).dispatch("read_file", {"path": "wide.txt", "max_lines": 100})
    assert r["truncated"] and len(r["content"]) < 1500


def test_read_file_private_key_not_leaked(repo):
    write(repo, "key.pem", "-----BEGIN RSA PRIVATE KEY-----\nSUPERSECRETBODY\n-----END RSA PRIVATE KEY-----\n")
    r = RepoTools(repo).dispatch("read_file", {"path": "key.pem"})
    assert "SUPERSECRETBODY" not in r["content"] and r["total_lines"] == 3


# --------------------------------------------------------------- search_code
def test_search_finds_matches_and_marks_redaction(tools):
    r = tools.dispatch("search_code", {"pattern": "api_token"})
    assert r["ok"] and r["match_count"] == 1
    m = r["matches"][0]
    assert m["path"] == "config/settings.py" and m["line"] == 1 and m["redacted"]
    assert fake_github_token() not in str(r)
    assert "leads" in r["note"]


def test_search_glob_and_case_sensitivity(tools):
    assert tools.dispatch("search_code", {"pattern": "runs-on", "path_glob": ".github/workflows/*.yml"})["match_count"] == 1
    assert tools.dispatch("search_code", {"pattern": "runs-on", "path_glob": "src/*"})["match_count"] == 0
    assert tools.dispatch("search_code", {"pattern": "DEF MAIN", "case_insensitive": False})["match_count"] == 0
    assert tools.dispatch("search_code", {"pattern": "DEF MAIN"})["match_count"] == 1


def test_search_machine_specific_path(tools):
    r = tools.dispatch("search_code", {"pattern": r"[A-Z]:\\\\Users\\\\"})
    assert r["match_count"] == 1


def test_search_invalid_and_dangerous_patterns(tools):
    assert "invalid regular expression" in tools.dispatch("search_code", {"pattern": "("})["error"]
    assert "nested quantifiers" in tools.dispatch("search_code", {"pattern": "(a+)+$"})["error"]
    assert not tools.dispatch("search_code", {"pattern": "a" * 300})["ok"]


def test_search_result_limit(repo):
    write(repo, "many.txt", "\n".join("needle" for _ in range(30)))
    r = RepoTools(repo, Limits(max_search_results=5)).dispatch("search_code", {"pattern": "needle"})
    assert r["match_count"] == 5 and r["truncated"]


def test_search_skips_binary_and_large_files(repo):
    (repo / "b.dat").write_bytes(b"needle\x00")
    (repo / "pic.png").write_bytes(b"needle")
    write(repo, "large.txt", "needle" + "x" * 5000)
    r = RepoTools(repo, Limits(max_file_bytes=1000)).dispatch("search_code", {"pattern": "needle"})
    assert r["match_count"] == 0 and r["files_skipped_too_large"] == 1


def test_search_does_not_follow_symlinks(repo, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks unavailable")
    outside = tmp_path / "out"
    outside.mkdir()
    (outside / "x.txt").write_text("needle")
    os.symlink(outside, repo / "linkdir")
    assert RepoTools(repo).dispatch("search_code", {"pattern": "needle"})["match_count"] == 0


def test_search_deadline(repo):
    for i in range(5):
        write(repo, f"f{i}.txt", "hello")
    r = RepoTools(repo, Limits(search_deadline_s=1e-9)).dispatch("search_code", {"pattern": "zzz"})
    assert r["ok"] and r["timed_out"]


# ------------------------------------------------------------------ git_log
@needs_git
def test_git_log_returns_commits(git_repo):
    r = RepoTools(git_repo).dispatch("git_log", {"max_count": 5})
    assert r["ok"] and [c["message"] for c in r["commits"]] == ["Add extra module", "Initial commit"]
    assert set(r["commits"][0]) == {"hash", "author", "date", "message"}
    assert r["commits_available"] == 2 and r["shallow_clone"] is False


@needs_git
def test_git_log_path_filter_and_limit(git_repo):
    t = RepoTools(git_repo)
    assert len(t.dispatch("git_log", {"path": "src/extra.py"})["commits"]) == 1
    assert len(t.dispatch("git_log", {"max_count": 1})["commits"]) == 1
    assert not t.dispatch("git_log", {"path": "../x"})["ok"]


@needs_git
def test_git_log_no_commits_and_not_a_repo(repo, tmp_path):
    git(repo, "init", "-q")
    r = RepoTools(repo).dispatch("git_log", {})
    assert r["ok"] and r["commits"] == [] and "no commits" in r["note"]
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "not a git repository" in RepoTools(plain).dispatch("git_log", {})["error"]


@needs_git
def test_git_log_strips_remote_credentials_and_redacts_messages(repo):
    git(repo, "init", "-q")
    git(repo, "remote", "add", "origin", "https://bob:hunter2pass@github.com/x/y.git")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"oops token={fake_github_token()}")
    r = RepoTools(repo).dispatch("git_log", {})
    assert r["remote"] == "https://github.com/x/y.git"
    assert "hunter2pass" not in str(r) and fake_github_token() not in str(r)


# ----------------------------------------------------- dispatch & observation
def test_dispatch_rejects_bad_calls(tools):
    assert "unknown tool" in tools.dispatch("rm_rf", {})["error"]
    assert "unknown tool" in tools.dispatch(None, {})["error"]  # type: ignore[arg-type]
    assert "unknown argument" in tools.dispatch("read_file", {"path": "a", "cmd": "ls"})["error"]
    assert "missing required" in tools.dispatch("read_file", {})["error"]
    assert "must be a string" in tools.dispatch("read_file", {"path": 5})["error"]
    assert "must be an integer" in tools.dispatch("git_log", {"max_count": True})["error"]
    assert "between" in tools.dispatch("git_log", {"max_count": 9999})["error"]
    assert "JSON object" in tools.dispatch("list_files", ["x"])["error"]
    assert tools.dispatch("list_files", None)["ok"]
    assert tools.dispatch("read_file", {"path": "README.md", "start_line": None})["ok"]


def test_dispatch_never_raises_on_internal_error(tools, monkeypatch):
    def boom(**_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setitem(tools._handlers, "list_files", boom)
    r = tools.dispatch("list_files", {})
    assert r == {"ok": False, "error": "internal error in tool (RuntimeError)"}


def test_observation_tracking(tools):
    assert not tools.has_observed("src/app.py")
    tools.dispatch("read_file", {"path": "src/app.py", "max_lines": 1})
    assert tools.has_observed("src/app.py") and tools.has_observed("src/app.py", 1)
    assert not tools.has_observed("src/app.py", 2)
    tools.dispatch("search_code", {"pattern": "API_TOKEN"})
    assert tools.has_observed("config/settings.py", 1)
    assert not tools.has_observed("config/settings.py", 2)


def test_tool_specs_match_handlers(tools):
    assert {s["name"] for s in TOOL_SPECS} == set(tools._handlers)
    for s in TOOL_SPECS:
        assert s["input_schema"]["additionalProperties"] is False