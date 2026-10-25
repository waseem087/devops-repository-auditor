"""Read-only repository inspection tools for the audit agent.

Every tool returns a JSON-serialisable dict: {"ok": True, ...} or {"ok": False, "error": "..."}.
Tools never modify the repository and never execute repository code. All paths are
checked against the repository root, and all text shown to the LLM is redacted.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from config import Limits
from redact import redact_text, redact_url
from repo_access import safe_git_env


class ToolError(Exception):
    """Raised for bad input; the message is returned to the LLM as a tool error."""


# Directories that are not descended into (but are reported, since anything present in a
# fresh clone is tracked by git, e.g. a committed node_modules is a hygiene finding).
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build", "target",
    ".gradle", "vendor",
}  # fmt: skip

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".pdf", ".zip", ".gz", ".tar",
    ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".class", ".so", ".dll", ".exe",
    ".dylib", ".pyc", ".pyo", ".whl", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4",
    ".mov", ".avi", ".sqlite", ".db", ".bin", ".pkl", ".npy", ".parquet", ".onnx", ".pt",
}  # fmt: skip

DEPENDENCY_FILES = {
    "pyproject.toml", "setup.py", "setup.cfg", "pipfile", "pipfile.lock", "poetry.lock",
    "uv.lock", "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.mod", "go.sum", "cargo.toml", "cargo.lock", "pom.xml", "build.gradle", "gemfile",
    "gemfile.lock", "composer.json", "composer.lock", "environment.yml", "conda.yaml",
}  # fmt: skip
CONFIG_FILES = {
    "dockerfile", "docker-compose.yml", "docker-compose.yaml", "makefile", "tox.ini",
    ".pre-commit-config.yaml", ".flake8", "ruff.toml", ".ruff.toml", ".editorconfig",
    "pytest.ini", "mypy.ini", ".gitignore", ".gitattributes", "codecov.yml", ".dockerignore",
    "dependabot.yml", "codeowners", ".eslintrc", ".eslintrc.json", ".prettierrc",
}  # fmt: skip
DOC_PREFIXES = ("readme", "contributing", "license", "security", "code_of_conduct", "changelog")
TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs"}

_NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")
_MAX_NOTABLE = 40


def _categorize(rel: str) -> list[str]:
    path = PurePosixPath(rel)
    name = path.name.lower()
    parts_lower = [p.lower() for p in path.parts]
    cats: list[str] = []
    if rel.startswith(".github/workflows/") and name.endswith((".yml", ".yaml")):
        cats.append("workflows")
    if name in DEPENDENCY_FILES or re.fullmatch(r"requirements[\w.\-]*\.txt", name):
        cats.append("dependency_manifests")
    if (
        name in CONFIG_FILES
        or name.startswith(("dockerfile", ".env"))
        or rel.startswith(".github/")
        and not rel.startswith(".github/workflows/")
    ):
        cats.append("config")
    if name.startswith(DOC_PREFIXES) or "pull_request_template" in name:
        cats.append("docs")
    if (
        any(p in TEST_DIR_NAMES for p in parts_lower[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    ):
        cats.append("tests")
    return cats


def _looks_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _split_lines(text: str) -> list[str]:
    """Split on \\n only (so line numbers match editors); drop one trailing empty line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln.removesuffix("\r") for ln in lines]


class RepoTools:
    """Inspection tools bound to one checked-out repository."""

    def __init__(self, root: str | Path, limits: Limits | None = None) -> None:
        self.root = Path(root).resolve()
        self.limits = limits or Limits()
        # What the agent has actually looked at (used by the evidence gate).
        self.read_spans: dict[str, list[tuple[int, int]]] = {}
        self.search_hits: dict[str, set[int]] = {}
        self.listed_paths: set[str] = set()
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_code": self.search_code,
            "git_log": self.git_log,
        }

    # ------------------------------------------------------------------ helpers
    def _rel(self, p: Path) -> str:
        rel = p.relative_to(self.root).as_posix()
        return "." if rel == "" else rel

    def _resolve(self, rel: str) -> Path:
        """Turn a user-supplied relative path into a safe absolute path inside the root."""
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("path must be a non-empty string")
        if "\x00" in rel:
            raise ToolError("invalid path")
        if len(rel) > 500:
            raise ToolError("path too long")
        norm = rel.replace("\\", "/") if os.sep == "\\" else rel
        if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
            raise ToolError("absolute paths are not allowed; use a path relative to the repository root")
        parts = [p for p in PurePosixPath(norm).parts if p not in ("", ".")]
        if ".." in parts:
            raise ToolError("'..' is not allowed in paths")
        if ".git" in parts:
            raise ToolError("access to .git internals is not allowed")
        candidate = self.root.joinpath(*parts)
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ToolError("path could not be resolved") from exc
        if not resolved.is_relative_to(self.root):
            raise ToolError("path escapes the repository root (symlink?)")
        if ".git" in resolved.relative_to(self.root).parts:
            raise ToolError("access to .git internals is not allowed")
        return resolved

    def _walk(self, start: Path, max_depth: int, info: dict[str, Any]) -> Iterator[tuple[str, int]]:
        """Yield (relative_path, size) for regular files below `start`, deterministic order."""
        info.setdefault("skipped_directories", [])
        info.setdefault("symlinks", [])
        info.setdefault("depth_limited", False)
        for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
            here = Path(dirpath)
            depth = len(here.relative_to(start).parts)
            dirnames.sort()
            filenames.sort()
            for d in list(dirnames):
                full = here / d
                if full.is_symlink():
                    info["symlinks"].append(self._rel(full))
                    dirnames.remove(d)
                elif d in SKIP_DIRS:
                    if d != ".git":
                        info["skipped_directories"].append(self._rel(full))
                    dirnames.remove(d)
            if depth >= max_depth and dirnames:
                info["depth_limited"] = True
                dirnames.clear()
            for name in filenames:
                full = here / name
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                if stat.S_ISLNK(st.st_mode):
                    info["symlinks"].append(self._rel(full))
                elif stat.S_ISREG(st.st_mode):
                    yield self._rel(full), st.st_size

    # -------------------------------------------------------------------- tools
    def list_files(self, path: str = ".", max_depth: int | None = None) -> dict[str, Any]:
        """Summarise repository structure: counts, notable files, and a capped file list."""
        depth = self.limits.max_depth if max_depth is None else max_depth
        start = self.root if path in (".", "") else self._resolve(path)
        if not start.exists():
            raise ToolError(f"path not found: {path}")
        if not start.is_dir():
            raise ToolError(f"not a directory: {path}")
        info: dict[str, Any] = {}
        files: list[str] = []
        notable: dict[str, list[str]] = {}
        top_dirs: dict[str, int] = {}
        large: list[dict[str, Any]] = []
        count = 0
        truncated = False
        for rel, size in self._walk(start, depth, info):
            if count >= self.limits.max_files:
                truncated = True
                break
            count += 1
            self.listed_paths.add(rel)
            if len(files) < self.limits.max_listed_files:
                files.append(rel)
            for cat in _categorize(rel):
                bucket = notable.setdefault(cat, [])
                if len(bucket) < _MAX_NOTABLE:
                    bucket.append(rel)
            parts = PurePosixPath(rel).parts
            key = parts[0] if len(parts) > 1 else "(root files)"
            top_dirs[key] = top_dirs.get(key, 0) + 1
            if size > 1_000_000 and len(large) < 20:
                large.append({"path": rel, "size": size})
        result: dict[str, Any] = {
            "ok": True,
            "path": self._rel(start),
            "file_count": count,
            "count_truncated": truncated,
            "files": files,
            "files_truncated": count > len(files),
            "directories": dict(sorted(top_dirs.items())),
            "notable": notable,
            "large_files": large,
            "skipped_directories": info["skipped_directories"][:50],
            "symlinks": info["symlinks"][:50],
            "depth_limited": info["depth_limited"],
            "note": "Everything in a fresh clone is tracked by git. Skipped directories were not "
            "descended into; pass them as `path` to look inside.",
        }
        if count == 0:
            result["note"] = "The repository contains no files (empty repository or empty path)."
        return result

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 400) -> dict[str, Any]:
        """Read a text file with line numbers (secrets redacted)."""
        p = self._resolve(path)
        rel = self._rel(p)
        if not p.exists():
            raise ToolError(f"file not found: {rel}")
        if p.is_dir():
            raise ToolError(f"{rel} is a directory; use list_files")
        if not p.is_file():
            raise ToolError(f"{rel} is not a regular file")
        try:
            size = p.stat().st_size
            if size > self.limits.max_file_bytes:
                raise ToolError(
                    f"file too large ({size} bytes; limit {self.limits.max_file_bytes}). "
                    "Use search_code to find specific lines instead."
                )
            data = p.read_bytes()
        except OSError as exc:
            raise ToolError(f"could not read file: {type(exc).__name__}") from exc
        if _looks_binary(data[:8192]):
            raise ToolError(f"{rel} appears to be a binary file")
        text = data.decode("utf-8", errors="replace")
        original_lines = _split_lines(text)
        redacted_lines = _split_lines(redact_text(text))
        total = len(original_lines)
        if total > 0 and start_line > total:
            raise ToolError(f"start_line {start_line} is beyond the end of the file ({total} lines)")
        end = start_line - 1
        chars = 0
        out: list[str] = []
        for idx in range(start_line - 1, min(total, start_line - 1 + max_lines)):
            line = redacted_lines[idx]
            if len(line) > 500:
                line = line[:500] + " ...[line truncated]"
            chars += len(line) + 8
            if chars > self.limits.max_output_chars and out:
                break
            out.append(f"{idx + 1}: {line}")
            end = idx + 1
        if out:
            self.read_spans.setdefault(rel, []).append((start_line, end))
        elif total == 0:
            self.read_spans.setdefault(rel, []).append((0, 0))
        truncated = end < total
        return {
            "ok": True,
            "path": rel,
            "total_lines": total,
            "start_line": start_line if out else 0,
            "end_line": end if out else 0,
            "truncated": truncated,
            "next_start_line": end + 1 if truncated else None,
            "redaction_applied": original_lines != redacted_lines,
            "content": "\n".join(out),
        }

    def search_code(
        self,
        pattern: str,
        path_glob: str | None = None,
        case_insensitive: bool = True,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Regex search over text files. Matches are leads, not confirmed violations."""
        if len(pattern) > 200:
            raise ToolError("pattern too long (max 200 characters)")
        if _NESTED_QUANTIFIER.search(pattern):
            raise ToolError("pattern rejected: nested quantifiers can be extremely slow; simplify it")
        try:
            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        limit = min(max_results or self.limits.max_search_results, self.limits.max_search_results)
        deadline = time.monotonic() + self.limits.search_deadline_s
        matches: list[dict[str, Any]] = []
        scanned = skipped_large = 0
        truncated = timed_out = False
        info: dict[str, Any] = {}
        for rel, size in self._walk(self.root, 30, info):
            if path_glob and not fnmatch.fnmatch(rel, path_glob):
                continue
            if PurePosixPath(rel).suffix.lower() in BINARY_EXTENSIONS:
                continue
            if size > self.limits.max_file_bytes:
                skipped_large += 1
                continue
            if scanned >= self.limits.max_search_files or time.monotonic() > deadline:
                timed_out = True
                break
            try:
                data = (self.root / rel).read_bytes()
            except OSError:
                continue
            if _looks_binary(data[:8192]):
                continue
            scanned += 1
            text = data.decode("utf-8", errors="replace")
            orig = _split_lines(text)
            shown = _split_lines(redact_text(text))
            for i, line in enumerate(orig):
                if regex.search(line[:1000]):
                    matches.append(
                        {
                            "path": rel,
                            "line": i + 1,
                            "text": shown[i].strip()[:200],
                            "redacted": shown[i] != line,
                        }
                    )
                    self.search_hits.setdefault(rel, set()).add(i + 1)
                    if len(matches) >= limit:
                        truncated = True
                        break
                if i % 500 == 499 and time.monotonic() > deadline:
                    timed_out = True
                    break
            if truncated or timed_out:
                break
        return {
            "ok": True,
            "pattern": redact_text(pattern),
            "match_count": len(matches),
            "matches": matches,
            "truncated": truncated,
            "timed_out": timed_out,
            "files_scanned": scanned,
            "files_skipped_too_large": skipped_large,
            "note": "Matches are investigation leads, not confirmed violations. "
            "Use read_file to confirm context before reporting.",
        }

    def _git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [
            "git", "--no-pager", "--no-optional-locks",
            "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
            "-C", str(self.root), *args,
        ]  # fmt: skip
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.limits.git_timeout_s,
            env=safe_git_env(),
            stdin=subprocess.DEVNULL,
        )

    def git_log(self, max_count: int = 20, path: str | None = None) -> dict[str, Any]:
        """Recent commit metadata (hash, author name, date, subject). Subjects are redacted."""
        count = min(max_count, self.limits.max_log_entries)
        rel_path = self._rel(self._resolve(path)) if path else None
        try:
            if self._git(["rev-parse", "--git-dir"]).returncode != 0:
                raise ToolError("not a git repository; no history available")
            result: dict[str, Any] = {"ok": True, "commits": []}
            remote = self._git(["config", "--get", "remote.origin.url"])
            if remote.returncode == 0 and remote.stdout.strip():
                result["remote"] = redact_url(remote.stdout.strip())
            result["shallow_clone"] = self._git(["rev-parse", "--is-shallow-repository"]).stdout.strip() == "true"
            if self._git(["rev-parse", "--verify", "-q", "HEAD"]).returncode != 0:
                result["note"] = "Repository has no commits."
                return result
            fmt = "--format=%H%x1f%an%x1f%ad%x1f%s%x1e"
            args = ["log", f"--max-count={count}", "--date=iso-strict", fmt]
            if rel_path:
                args += ["--", rel_path]
            proc = self._git(args)
            if proc.returncode != 0:
                raise ToolError("git log failed")
            for record in proc.stdout.split("\x1e"):
                record = record.strip("\n")
                if not record:
                    continue
                fields = record.split("\x1f")
                if len(fields) < 4:
                    continue
                result["commits"].append(
                    {
                        "hash": fields[0][:12],
                        "author": fields[1][:80],
                        "date": fields[2],
                        "message": redact_text(fields[3])[:200],
                    }
                )
            total = self._git(["rev-list", "--count", "HEAD"])
            if total.returncode == 0:
                result["commits_available"] = int(total.stdout.strip() or 0)
            result["note"] = (
                "Commit messages are context only and are not proof of a violation."
                + (" History is limited because this is a shallow clone." if result["shallow_clone"] else "")
            )
            return result
        except subprocess.TimeoutExpired as exc:
            raise ToolError("git timed out") from exc
        except FileNotFoundError as exc:
            raise ToolError("git is not installed") from exc

    def has_observed(self, path: str, line: int | None = None) -> bool:
        """True if the agent has read `path` (covering `line`) or seen it in a search hit."""
        spans = self.read_spans.get(path, [])
        if line is None:
            return bool(spans) or path in self.search_hits
        if any(start <= line <= end for start, end in spans):
            return True
        return line in self.search_hits.get(path, set())

    # ----------------------------------------------------------------- dispatch
    def dispatch(self, name: str, arguments: Any) -> dict[str, Any]:
        """Validate and run one tool call. Never raises; errors come back as {"ok": False}."""
        handler = self._handlers.get(name) if isinstance(name, str) else None
        if handler is None:
            shown = str(name)[:50]
            return {"ok": False, "error": f"unknown tool {shown!r}; available: {sorted(self._handlers)}"}
        schema = next(s for s in TOOL_SPECS if s["name"] == name)["input_schema"]
        try:
            args = validate_arguments(schema, arguments)
            return handler(**args)
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            return {"ok": False, "error": f"internal error in tool ({type(exc).__name__})"}


def validate_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Check LLM-supplied arguments against a (small subset of) JSON schema."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be a JSON object")
    props: dict[str, Any] = schema["properties"]
    args = {k: v for k, v in arguments.items() if v is not None}
    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(f"unknown argument(s): {unknown}; allowed: {sorted(props)}")
    for req in schema.get("required", []):
        if req not in args:
            raise ToolError(f"missing required argument: {req}")
    for key, value in args.items():
        spec = props[key]
        kind = spec["type"]
        if kind == "string":
            if not isinstance(value, str):
                raise ToolError(f"argument {key} must be a string")
            if len(value) > spec.get("maxLength", 500):
                raise ToolError(f"argument {key} is too long")
        elif kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolError(f"argument {key} must be an integer")
            if value < spec.get("minimum", -(10**9)) or value > spec.get("maximum", 10**9):
                raise ToolError(
                    f"argument {key} must be between {spec.get('minimum')} and {spec.get('maximum')}"
                )
        elif kind == "boolean" and not isinstance(value, bool):
            raise ToolError(f"argument {key} must be a boolean")
    return args


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": (
            "List the repository's file structure: file counts, top-level directories, notable files "
            "(workflows, dependency manifests, config, docs, tests) and a capped file list. "
            "Pass `path` to look inside a subdirectory (including skipped ones such as dist/)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to repo root. Default '.'"},
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file from the repository with line numbers. Secrets are redacted. "
            "Use start_line/max_lines to page through long files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to repo root"},
                "start_line": {"type": "integer", "minimum": 1},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_code",
        "description": (
            "Regex search across the repository's text files (case-insensitive by default). Returns "
            "path, line number and a redacted line. Matches are leads, not proof: confirm with read_file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "maxLength": 200, "description": "Python regular expression"},
                "path_glob": {"type": "string", "maxLength": 200, "description": "e.g. '.github/workflows/*.yml'"},
                "case_insensitive": {"type": "boolean"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "git_log",
        "description": (
            "Recent commit metadata (short hash, author name, date, subject), optionally for one path. "
            "Context only: a commit message alone never proves a violation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_count": {"type": "integer", "minimum": 1, "maximum": 50},
                "path": {"type": "string", "description": "Limit history to this path"},
            },
            "additionalProperties": False,
        },
    },
]
