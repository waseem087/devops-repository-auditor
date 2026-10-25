"""Shared pytest helpers. Tests are fully offline and use only fabricated data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Limits
from tools import RepoTools


def fake_github_token() -> str:
    """Built at runtime so no token-shaped string sits in the repository."""
    return "ghp_" + "aB3dE5gH7jK9" * 3


def fake_aws_key() -> str:
    return "AKIA" + "QWERTYUIOPASDFGH"


def write(root: Path, rel: str, content: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small, fabricated repository."""
    root = tmp_path / "repo"
    root.mkdir()
    write(root, "README.md", "# Demo\nA demo project.\n")
    write(root, ".gitignore", ".env\n__pycache__/\n")
    write(root, "requirements.txt", "requests>=2.0\n")
    write(root, "src/app.py", "def main():\n    return 1\n")
    write(root, "tests/test_app.py", "def test_main():\n    assert True\n")
    write(root, ".github/workflows/ci.yml", "name: CI\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n")
    write(root, "config/settings.py", f'API_TOKEN = "{fake_github_token()}"\nDB_PATH = "C:\\\\Users\\\\bob\\\\data"\n')
    return root


@pytest.fixture
def tools(repo: Path) -> RepoTools:
    return RepoTools(repo, Limits())


@pytest.fixture
def git_repo(repo: Path) -> Path:
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Initial commit")
    write(repo, "src/extra.py", "x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Add extra module")
    return repo


def can_symlink(tmp: Path) -> bool:
    try:
        os.symlink(tmp, tmp / "probe_link")
        (tmp / "probe_link").unlink()
        return True
    except (OSError, NotImplementedError):
        return False
