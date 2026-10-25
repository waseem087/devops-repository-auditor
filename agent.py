"""Command-line entry point.

    python agent.py --repo https://github.com/OWNER/REPOSITORY --output report.json

Exit codes: 0 = audit completed and report written; 1 = the audit could not be completed
(clone, API or agent failure; no report is written); 2 = bad usage or configuration;
3 = unexpected internal error; 130 = interrupted. The program never prompts for input.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 10):  # noqa: UP036 - friendly message for older interpreters
    sys.exit("error: Python 3.10 or newer is required.")

import argparse
import json
import os
import tempfile
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from auditor import AgentLimits, AuditAgent, AuditError
from config import Limits
from llm_client import LLMConfigError, LLMError, load_env_file
from providers import ProviderConfig, load_provider
from redact import redact_text
from repo_access import RepoAccessError, RepoRef, acquire_repo, parse_github_url
from schemas import Report, Violation
from tools import RepoTools

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3
EXIT_INTERRUPTED = 130


class UsageError(Exception):
    """Bad command-line input or configuration (exit code 2)."""


def render_report(repository: str, violations: list[Violation]) -> str:
    """Serialize the report deterministically and prove it round-trips through the strict schema."""
    report = Report(repository=repository, violations=violations)
    data = report.model_dump(mode="json")
    text = json.dumps(data, indent=2, ensure_ascii=True) + "\n"
    if json.loads(text) != data:
        raise ValueError("report did not survive a JSON round trip")
    Report.model_validate_json(text)
    return text


def write_json_atomic(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then rename, so no partial file is ever visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _prepare_output(path: Path) -> None:
    """Fail early (before any clone or API call) if the report could not be written."""
    if path.is_dir():
        raise UsageError(f"--output {path} is a directory; give a file path such as report.json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UsageError(f"cannot create the output directory {path.parent}: {type(exc).__name__}") from exc
    if not os.access(path.parent, os.W_OK):
        raise UsageError(f"the output directory {path.parent} is not writable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="Audit a public GitHub repository for DevOps defects using an LLM-driven agent.",
    )
    parser.add_argument("--repo", required=True, help="public repository URL, e.g. https://github.com/OWNER/REPOSITORY")
    parser.add_argument("--output", default="report.json", help="path of the JSON report to write (default: report.json)")
    parser.add_argument("--verbose", action="store_true", help="print the investigation trace to stderr")
    parser.add_argument("--debug", action="store_true", help="print a (redacted) traceback on unexpected errors")
    return parser


def _err(message: str) -> None:
    print(f"error: {redact_text(message)}", file=sys.stderr)


def _run(
    args: argparse.Namespace,
    repo_provider: Callable[[RepoRef, Limits], AbstractContextManager[Path]],
    llm_factory: Callable[[ProviderConfig], Any] | None,
    env_file: str | None,
) -> int:
    if env_file:
        load_env_file(env_file)
        load_env_file(Path(__file__).resolve().parent / ".env")
    try:
        ref = parse_github_url(args.repo)
    except RepoAccessError as exc:
        raise UsageError(str(exc)) from exc
    try:
        provider = load_provider()
        limits = Limits.from_env()
        agent_limits = AgentLimits.from_env()
    except (LLMConfigError, ValueError) as exc:
        raise UsageError(str(exc)) from exc
    output = Path(args.output)
    _prepare_output(output)

    client = llm_factory(provider) if llm_factory else provider.build_client()
    with repo_provider(ref, limits) as root:
        agent = AuditAgent(client, RepoTools(root, limits), ref.canonical_url, agent_limits)
        result = agent.run()

    write_json_atomic(output, render_report(ref.canonical_url, result.violations))

    if args.verbose:
        for entry in result.trace:
            status = "ok" if entry.ok else "ERROR"
            print(f"  {entry.step:>3}. {entry.tool} {entry.args} -> {status}: {entry.summary}", file=sys.stderr)
    if result.limit_reached:
        print(f"warning: resource limit reached ({result.stop_reason}); coverage may be partial.", file=sys.stderr)
    print(f"Audit complete: {len(result.violations)} violation(s) written to {output}")
    print(
        f"  model: {provider.provider}/{provider.model} | stop reason: {result.stop_reason} | tool calls: {result.tool_calls} | model turns: {result.iterations}"
        f" | tokens in/out: {result.input_tokens}/{result.output_tokens}",
        file=sys.stderr,
    )
    return EXIT_OK


def main(
    argv: list[str] | None = None,
    *,
    repo_provider: Callable[[RepoRef, Limits], AbstractContextManager[Path]] = acquire_repo,
    llm_factory: Callable[[ProviderConfig], Any] | None = None,
    env_file: str | None = ".env",
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        return _run(args, repo_provider, llm_factory, env_file)
    except KeyboardInterrupt:
        _err("interrupted; no report was written")
        return EXIT_INTERRUPTED
    except UsageError as exc:
        _err(str(exc))
        return EXIT_USAGE
    except (AuditError, RepoAccessError, LLMError) as exc:
        _err(f"{exc} No report was written.")
        return EXIT_FAILURE
    except OSError as exc:
        _err(f"could not write the report ({type(exc).__name__}). No report was written.")
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - last line of defence; never show raw internals
        _err(f"unexpected internal failure ({type(exc).__name__}). No report was written. Use --debug for details.")
        if args.debug:
            print(redact_text(traceback.format_exc()), file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())